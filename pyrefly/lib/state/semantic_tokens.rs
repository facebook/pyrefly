/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use lsp_types::SemanticToken;
use lsp_types::SemanticTokenModifier;
use lsp_types::SemanticTokenType;
use lsp_types::SemanticTokensLegend;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::types::Type;
use pyrefly_util::visit::Visit as _;
use ruff_python_ast::Arguments;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::binding::binding::Key;

pub struct SemanticTokensLegends {
    token_types_index: HashMap<SemanticTokenType, u32>,
    token_modifiers_index: HashMap<SemanticTokenModifier, u32>,
}

impl SemanticTokensLegends {
    pub fn lsp_semantic_token_legends() -> SemanticTokensLegend {
        SemanticTokensLegend {
            token_types: vec![
                SemanticTokenType::NAMESPACE,
                SemanticTokenType::TYPE,
                SemanticTokenType::CLASS,
                SemanticTokenType::ENUM,
                SemanticTokenType::INTERFACE,
                SemanticTokenType::STRUCT,
                SemanticTokenType::TYPE_PARAMETER,
                SemanticTokenType::PARAMETER,
                SemanticTokenType::VARIABLE,
                SemanticTokenType::PROPERTY,
                SemanticTokenType::ENUM_MEMBER,
                SemanticTokenType::EVENT,
                SemanticTokenType::FUNCTION,
                SemanticTokenType::METHOD,
                SemanticTokenType::MACRO,
                SemanticTokenType::KEYWORD,
                SemanticTokenType::MODIFIER,
                SemanticTokenType::COMMENT,
                SemanticTokenType::STRING,
                SemanticTokenType::NUMBER,
                SemanticTokenType::REGEXP,
                SemanticTokenType::OPERATOR,
                SemanticTokenType::DECORATOR,
            ],
            token_modifiers: vec![
                SemanticTokenModifier::DECLARATION,
                SemanticTokenModifier::DEFINITION,
                SemanticTokenModifier::READONLY,
                SemanticTokenModifier::STATIC,
                SemanticTokenModifier::DEPRECATED,
                SemanticTokenModifier::ABSTRACT,
                SemanticTokenModifier::ASYNC,
                SemanticTokenModifier::MODIFICATION,
                SemanticTokenModifier::DOCUMENTATION,
                SemanticTokenModifier::DEFAULT_LIBRARY,
            ],
        }
    }

    pub fn new() -> Self {
        let lsp_legend = Self::lsp_semantic_token_legends();
        let mut token_types_index = HashMap::new();
        let mut token_modifiers_index = HashMap::new();
        for (i, token_type) in lsp_legend.token_types.iter().enumerate() {
            token_types_index.insert(token_type.clone(), i as u32);
        }
        for (i, token_modifier) in lsp_legend.token_modifiers.iter().enumerate() {
            token_modifiers_index.insert(token_modifier.clone(), i as u32);
        }
        Self {
            token_types_index,
            token_modifiers_index,
        }
    }

    pub fn convert_tokens_into_lsp_semantic_tokens(
        &self,
        tokens: &[SemanticTokenWithFullRange],
        module_info: Module,
        limit_cell_idx: Option<usize>,
    ) -> Vec<SemanticToken> {
        let mut previous_line = 0;
        let mut previous_col = 0;
        let mut lsp_semantic_tokens = Vec::new();
        for token in tokens {
            let cell_idx = module_info.to_cell_for_lsp(token.range.start());
            // Skip tokens different cells if we're filtering for a particular cell
            if cell_idx != limit_cell_idx {
                continue;
            }
            let source_range = module_info.display_range(token.range);
            let length = token.range.len().to_u32();
            if length == 0 {
                continue;
            }
            let current_line = source_range.start.line_within_cell().to_zero_indexed();
            let current_col = source_range.start.column().get() - 1;
            let (delta_line, delta_start) = {
                let delta_line = current_line - previous_line;
                let delta_start = if previous_line == current_line {
                    current_col - previous_col
                } else {
                    current_col
                };
                previous_line = current_line;
                previous_col = current_col;
                (delta_line, delta_start)
            };
            let token_type = *self.token_types_index.get(&token.token_type).unwrap();
            let mut token_modifiers_bitset = 0;
            for modifier in &token.token_modifiers {
                let index = *self.token_modifiers_index.get(modifier).unwrap();
                token_modifiers_bitset |= 1 << index;
            }
            lsp_semantic_tokens.push(SemanticToken {
                delta_line,
                delta_start,
                length,
                token_type,
                token_modifiers_bitset,
            });
        }
        lsp_semantic_tokens.dedup_by(|current, previous| {
            current.delta_line == 0 && current.delta_start == 0 && current.length == previous.length
        });
        lsp_semantic_tokens
    }

    #[cfg(test)]
    pub fn get_modifiers(&self, token_modifiers_bitset: u32) -> Vec<SemanticTokenModifier> {
        let mut modifiers = Vec::new();
        for (modifier, index) in &self.token_modifiers_index {
            let singleton_set = (1 << *index) as u32;
            if (token_modifiers_bitset & singleton_set) == singleton_set {
                modifiers.push(modifier.clone());
            }
        }
        // needed for a deterministic print ordering in tests
        modifiers.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        modifiers
    }
}

pub struct SemanticTokenWithFullRange {
    pub range: TextRange,
    pub token_type: SemanticTokenType,
    pub token_modifiers: Vec<SemanticTokenModifier>,
}

pub struct SemanticTokenBuilder {
    tokens: Vec<SemanticTokenWithFullRange>,
    limit_range: Option<TextRange>,
    disabled_ranges: Vec<TextRange>,
}

impl SemanticTokenBuilder {
    pub fn new(limit_range: Option<TextRange>, mut disabled_ranges: Vec<TextRange>) -> Self {
        disabled_ranges.sort_by(|a, b| {
            a.start()
                .cmp(&b.start())
                .then_with(|| a.end().cmp(&b.end()))
        });
        Self {
            tokens: Vec::new(),
            limit_range,
            disabled_ranges,
        }
    }

    fn push_if_in_range(
        &mut self,
        range: TextRange,
        token_type: SemanticTokenType,
        token_modifiers: Vec<SemanticTokenModifier>,
    ) {
        if self.limit_range.is_none_or(|x| x.contains_range(range)) {
            self.tokens.push(SemanticTokenWithFullRange {
                range,
                token_type,
                token_modifiers,
            })
        }
    }

    fn is_disabled(&self, range: TextRange) -> bool {
        self.disabled_ranges
            .iter()
            .any(|disabled| disabled.contains_range(range))
    }

    pub fn process_key(
        &mut self,
        key: &Key,
        definition_module: ModuleName,
        symbol_kind: SymbolKind,
    ) {
        let reference_range = key.range();
        let (token_type, mut token_modifiers) =
            symbol_kind.to_lsp_semantic_token_type_with_modifiers();
        let is_default_library = {
            let module_name_str = definition_module.as_str();
            module_name_str == "builtins"
                || module_name_str == "typing"
                || module_name_str == "typing_extensions"
        };
        if is_default_library {
            token_modifiers.push(SemanticTokenModifier::DEFAULT_LIBRARY);
        }
        self.push_if_in_range(reference_range, token_type, token_modifiers);
    }

    fn process_arguments(&mut self, args: &Arguments) {
        for keyword in &args.keywords {
            if let Some(arg) = &keyword.arg {
                self.push_if_in_range(arg.range, SemanticTokenType::PARAMETER, Vec::new());
            }
        }
    }

    fn process_expr(
        &mut self,
        x: &Expr,
        get_type_of_attribute: &dyn Fn(TextRange) -> Option<Type>,
    ) {
        match x {
            Expr::Call(call) => {
                self.process_arguments(&call.arguments);
                x.recurse(&mut |x| self.process_expr(x, get_type_of_attribute));
            }
            Expr::Attribute(attr) => {
                // todo(samzhou19815): if the class's base is Enum, it should be ENUM_MEMBER
                let kind = match get_type_of_attribute(attr.range()) {
                    Some(Type::BoundMethod(_)) => SemanticTokenType::METHOD,
                    Some(Type::Function(_) | Type::Callable(_)) => SemanticTokenType::FUNCTION,
                    Some(Type::ClassDef(_)) => SemanticTokenType::CLASS,
                    _ => SemanticTokenType::PROPERTY,
                };
                self.push_if_in_range(attr.attr.range(), kind, Vec::new());
                attr.value
                    .visit(&mut |x| self.process_expr(x, get_type_of_attribute));
            }
            _ => {
                x.recurse(&mut |x| self.process_expr(x, get_type_of_attribute));
            }
        }
    }

    fn process_stmt(&mut self, x: &Stmt) {
        match x {
            Stmt::ClassDef(class_def) => {
                if self.is_disabled(class_def.range) {
                    self.push_if_in_range(
                        class_def.name.range,
                        SemanticTokenType::CLASS,
                        Vec::new(),
                    );
                }
                x.recurse(&mut |x| self.process_stmt(x));
            }
            Stmt::FunctionDef(function_def) => {
                if self.is_disabled(function_def.range) {
                    self.push_if_in_range(
                        function_def.name.range,
                        SemanticTokenType::FUNCTION,
                        Vec::new(),
                    );
                }
                x.recurse(&mut |x| self.process_stmt(x));
            }
            Stmt::Assign(assign) => {
                if self.is_disabled(assign.range()) {
                    for target in &assign.targets {
                        if let Expr::Name(name) = target {
                            self.push_if_in_range(
                                name.range,
                                SemanticTokenType::VARIABLE,
                                Vec::new(),
                            );
                        }
                    }
                }
                x.recurse(&mut |x| self.process_stmt(x));
            }
            Stmt::Try(stmt_try) => {
                for ExceptHandler::ExceptHandler(handler) in stmt_try.handlers.iter() {
                    if let Some(name) = &handler.name {
                        self.push_if_in_range(name.range(), SemanticTokenType::VARIABLE, vec![]);
                    }
                }
            }
            Stmt::With(with) => {
                for with_item in with.items.iter() {
                    if let Some(box name) = &with_item.optional_vars {
                        self.push_if_in_range(name.range(), SemanticTokenType::VARIABLE, vec![]);
                    }
                }
            }
            Stmt::Import(StmtImport { names, .. }) => {
                for alias in names {
                    self.push_if_in_range(alias.name.range, SemanticTokenType::NAMESPACE, vec![]);
                    if let Some(asname) = &alias.asname {
                        self.push_if_in_range(asname.range, SemanticTokenType::NAMESPACE, vec![]);
                    }
                }
            }
            Stmt::ImportFrom(StmtImportFrom {
                module: Some(module),
                ..
            }) => {
                self.push_if_in_range(module.range, SemanticTokenType::NAMESPACE, vec![]);
            }
            Stmt::AnnAssign(ann_assign) => {
                if let Expr::Name(name) = &*ann_assign.target {
                    self.push_if_in_range(name.range, SemanticTokenType::VARIABLE, vec![]);
                }
                x.recurse(&mut |x| self.process_stmt(x));
            }
            _ => x.recurse(&mut |x| self.process_stmt(x)),
        }
    }

    pub fn process_ast(
        &mut self,
        ast: &ModModule,
        get_type_of_attribute: &dyn Fn(TextRange) -> Option<Type>,
    ) {
        for s in &ast.body {
            self.process_stmt(s);
        }
        ast.visit(&mut |e| self.process_expr(e, get_type_of_attribute));
    }

    pub fn all_tokens_sorted(self) -> Vec<SemanticTokenWithFullRange> {
        let mut tokens = self.tokens;
        tokens.sort_by(|a, b| a.range.start().cmp(&b.range.start()));
        tokens
    }
}

fn collect_disabled_ranges_from_block(
    stmts: &[Stmt],
    sys_info: &SysInfo,
    reachable: bool,
    ranges: &mut Vec<TextRange>,
) {
    for stmt in stmts {
        collect_disabled_ranges_from_stmt(stmt, sys_info, reachable, ranges);
    }
}

fn collect_disabled_ranges_from_stmt(
    stmt: &Stmt,
    sys_info: &SysInfo,
    reachable: bool,
    ranges: &mut Vec<TextRange>,
) {
    if !reachable {
        ranges.push(stmt.range());
        return;
    }

    match stmt {
        Stmt::If(if_stmt) => {
            let mut prior_true_branch = false;
            for (test, body) in Ast::if_branches(if_stmt) {
                let eval = test.and_then(|expr| sys_info.evaluate_bool(expr));
                let branch_reachable = if prior_true_branch {
                    false
                } else {
                    !matches!(eval, Some(false))
                };
                collect_disabled_ranges_from_block(body, sys_info, branch_reachable, ranges);
                if !prior_true_branch && matches!(eval, Some(true)) {
                    prior_true_branch = true;
                }
            }
        }
        Stmt::FunctionDef(func) => {
            collect_disabled_ranges_from_block(&func.body, sys_info, reachable, ranges);
        }
        Stmt::ClassDef(class_def) => {
            collect_disabled_ranges_from_block(&class_def.body, sys_info, reachable, ranges);
        }
        Stmt::With(with_stmt) => {
            collect_disabled_ranges_from_block(&with_stmt.body, sys_info, reachable, ranges);
        }
        Stmt::For(for_stmt) => {
            collect_disabled_ranges_from_block(&for_stmt.body, sys_info, reachable, ranges);
            collect_disabled_ranges_from_block(&for_stmt.orelse, sys_info, reachable, ranges);
        }
        Stmt::While(while_stmt) => {
            let condition = sys_info.evaluate_bool(&while_stmt.test);
            let body_reachable = reachable && condition != Some(false);
            collect_disabled_ranges_from_block(&while_stmt.body, sys_info, body_reachable, ranges);
            collect_disabled_ranges_from_block(&while_stmt.orelse, sys_info, reachable, ranges);
        }
        Stmt::Try(try_stmt) => {
            collect_disabled_ranges_from_block(&try_stmt.body, sys_info, reachable, ranges);
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                collect_disabled_ranges_from_block(&handler.body, sys_info, reachable, ranges);
            }
            collect_disabled_ranges_from_block(&try_stmt.orelse, sys_info, reachable, ranges);
            collect_disabled_ranges_from_block(&try_stmt.finalbody, sys_info, reachable, ranges);
        }
        Stmt::Match(match_stmt) => {
            for case in &match_stmt.cases {
                collect_disabled_ranges_from_block(&case.body, sys_info, reachable, ranges);
            }
        }
        _ => {}
    }
}

pub(crate) fn disabled_ranges_for_module(ast: &ModModule, sys_info: &SysInfo) -> Vec<TextRange> {
    let mut ranges = Vec::new();
    collect_disabled_ranges_from_block(&ast.body, sys_info, true, &mut ranges);
    ranges
}
