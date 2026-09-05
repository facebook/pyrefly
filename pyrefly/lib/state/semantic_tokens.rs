/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::LazyLock;

use lsp_types::SemanticToken;
use lsp_types::SemanticTokenModifier;
use lsp_types::SemanticTokenType;
use lsp_types::SemanticTokensLegend;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Type;
use pyrefly_util::visit::Visit as _;
use regex::Regex;
use ruff_python_ast::Arguments;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprContext;
use ruff_python_ast::InterpolatedElement;
use ruff_python_ast::InterpolatedStringElement;
use ruff_python_ast::ModModule;
use ruff_python_ast::Operator;
use ruff_python_ast::Pattern;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::StringFlags as _;
use ruff_python_ast::name::Name;
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::token::Tokens;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Key;
use crate::state::lsp::attribute_symbol_kind_from_type;

const SELF_PARAMETER_MODIFIER: SemanticTokenModifier = SemanticTokenModifier::new("selfParameter");
const BYTE_STRING_MODIFIER: SemanticTokenModifier = SemanticTokenModifier::new("byteString");
const ESCAPE_SEQUENCE_MODIFIER: SemanticTokenModifier =
    SemanticTokenModifier::new("escapeSequence");
const FORMAT_PLACEHOLDER_MODIFIER: SemanticTokenModifier =
    SemanticTokenModifier::new("formatPlaceholder");
const FORMAT_SPECIFIER_MODIFIER: SemanticTokenModifier =
    SemanticTokenModifier::new("formatSpecifier");
const FORMAT_STRING_MODIFIER: SemanticTokenModifier = SemanticTokenModifier::new("formatString");
const RAW_STRING_MODIFIER: SemanticTokenModifier = SemanticTokenModifier::new("rawString");
const STRING_PREFIX_MODIFIER: SemanticTokenModifier = SemanticTokenModifier::new("stringPrefix");
const TEMPLATE_STRING_MODIFIER: SemanticTokenModifier =
    SemanticTokenModifier::new("templateString");

// Mirrors the escape forms consumed by Ruff's string parser. Raw strings bypass these,
// and byte strings use the subset without Unicode escapes.
static STRING_ESCAPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"\\(?:\r\n?|\n|[\\'"abfnrtv]|[0-7]{1,3}|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|N\{[^}\r\n]+\})"#,
    )
    .expect("string escape regex must be valid")
});
static BYTE_STRING_ESCAPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\\(?:\r\n?|\n|[\\'"abfnrtv]|[0-7]{1,3}|x[0-9A-Fa-f]{2})"#)
        .expect("byte string escape regex must be valid")
});
static PRINTF_PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"%(?:%|(?:\([^\r\n)]*\))?[#0 +\-]*(?:\*|[0-9]+)?(?:\.(?:\*|[0-9]*))?[hlL]?[diouxXeEfFgGcrsab%])",
    )
    .expect("printf placeholder regex must be valid")
});

/// Adds the DEFAULT_LIBRARY modifier if the module is a standard library module
/// (builtins, typing, typing_extensions).
fn maybe_add_default_library_modifier(
    module: ModuleName,
    modifiers: &mut Vec<SemanticTokenModifier>,
) {
    if ["builtins", "typing", "typing_extensions"].contains(&module.as_str()) {
        modifiers.push(SemanticTokenModifier::DEFAULT_LIBRARY);
    }
}

fn maybe_add_self_parameter_modifier(name: &str, modifiers: &mut Vec<SemanticTokenModifier>) {
    if name == "self" || name == "cls" {
        modifiers.push(SELF_PARAMETER_MODIFIER.clone());
    }
}

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
                SELF_PARAMETER_MODIFIER.clone(),
                BYTE_STRING_MODIFIER.clone(),
                ESCAPE_SEQUENCE_MODIFIER.clone(),
                FORMAT_PLACEHOLDER_MODIFIER.clone(),
                FORMAT_SPECIFIER_MODIFIER.clone(),
                FORMAT_STRING_MODIFIER.clone(),
                RAW_STRING_MODIFIER.clone(),
                STRING_PREFIX_MODIFIER.clone(),
                TEMPLATE_STRING_MODIFIER.clone(),
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
        limit_range: Option<TextRange>,
        limit_cell_idx: Option<usize>,
    ) -> Vec<SemanticToken> {
        let mut previous_line = 0;
        let mut previous_col = 0;
        let mut lsp_semantic_tokens = Vec::new();
        let source = module_info.contents().as_str();
        for token in tokens {
            let mut push_segment = |segment_range: TextRange| {
                if segment_range.is_empty() {
                    return;
                }
                if !range_overlaps(limit_range, segment_range) {
                    return;
                }
                let cell_idx = module_info.to_cell_for_lsp(segment_range.start());
                // Skip tokens in different cells if we're filtering for a particular cell
                if cell_idx != limit_cell_idx {
                    return;
                }
                let start_pos = module_info.to_lsp_position(segment_range.start());
                let end_pos = module_info.to_lsp_position(segment_range.end());
                debug_assert_eq!(
                    start_pos.line, end_pos.line,
                    "Semantic token segment should be on a single line"
                );
                if start_pos.line != end_pos.line {
                    return;
                }
                let length = end_pos.character.saturating_sub(start_pos.character);
                if length == 0 {
                    return;
                }
                let current_line = start_pos.line;
                let current_col = start_pos.character;
                let delta_line = current_line - previous_line;
                let delta_start = if previous_line == current_line {
                    current_col - previous_col
                } else {
                    current_col
                };
                previous_line = current_line;
                previous_col = current_col;
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
            };
            let mut segment_start = token.range.start();
            let start = token.range.start().to_usize();
            let end = token.range.end().to_usize();
            let token_source = &source[start..end];
            for line in token_source.split_inclusive('\n') {
                let line_without_lf = line.strip_suffix('\n').unwrap_or(line);
                let line_without_ending = line_without_lf
                    .strip_suffix('\r')
                    .unwrap_or(line_without_lf);
                let segment_end = segment_start
                    + TextSize::try_from(line_without_ending.len())
                        .expect("semantic token segment length must fit in TextSize");
                let segment_range = TextRange::new(segment_start, segment_end);
                segment_start += TextSize::try_from(line.len())
                    .expect("semantic token line length must fit in TextSize");
                push_segment(segment_range);
            }
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

fn syntax_token_type(kind: TokenKind) -> Option<SemanticTokenType> {
    if kind.is_keyword() {
        Some(SemanticTokenType::KEYWORD)
    } else if kind.is_operator() {
        Some(SemanticTokenType::OPERATOR)
    } else {
        match kind {
            TokenKind::Comment => Some(SemanticTokenType::COMMENT),
            TokenKind::Int | TokenKind::Float | TokenKind::Complex => {
                Some(SemanticTokenType::NUMBER)
            }
            _ => None,
        }
    }
}

fn range_overlaps(limit_range: Option<TextRange>, range: TextRange) -> bool {
    limit_range.is_none_or(|limit| {
        limit
            .intersect(range)
            .is_some_and(|intersection| !intersection.is_empty())
    })
}

struct StringContentRange {
    range: TextRange,
    modifier: SemanticTokenModifier,
}

fn text_size(value: usize) -> TextSize {
    TextSize::try_from(value).expect("string token offset must fit in TextSize")
}

fn regex_content_ranges(
    regex: &Regex,
    range: TextRange,
    source: &str,
    modifier: &SemanticTokenModifier,
) -> Vec<StringContentRange> {
    let text = &source[range.start().to_usize()..range.end().to_usize()];
    regex
        .find_iter(text)
        .map(|found| StringContentRange {
            range: TextRange::at(
                range.start() + text_size(found.start()),
                text_size(found.len()),
            ),
            modifier: modifier.clone(),
        })
        .collect()
}

fn collect_interpolated_string_ranges(
    elements: &[InterpolatedStringElement],
    modifiers: &[SemanticTokenModifier],
    content_ranges: &mut Vec<StringContentRange>,
    punctuation_ranges: &mut Vec<SemanticTokenWithFullRange>,
) {
    for element in elements {
        let InterpolatedStringElement::Interpolation(InterpolatedElement {
            range,
            conversion,
            format_spec,
            ..
        }) = element
        else {
            continue;
        };
        let format_spec_start = format_spec.as_ref().map(|spec| spec.start());
        if conversion.to_byte().is_some() {
            let end = format_spec_start
                .map(|start| start - text_size(1))
                .unwrap_or(range.end() - text_size(1));
            punctuation_ranges.push(SemanticTokenWithFullRange {
                range: TextRange::new(end - text_size(2), end),
                token_type: SemanticTokenType::STRING,
                token_modifiers: modifiers
                    .iter()
                    .cloned()
                    .chain([FORMAT_SPECIFIER_MODIFIER.clone()])
                    .collect(),
            });
        }
        if let Some(format_spec) = format_spec {
            punctuation_ranges.push(SemanticTokenWithFullRange {
                range: TextRange::at(format_spec.start() - text_size(1), text_size(1)),
                token_type: SemanticTokenType::STRING,
                token_modifiers: modifiers
                    .iter()
                    .cloned()
                    .chain([FORMAT_SPECIFIER_MODIFIER.clone()])
                    .collect(),
            });
            content_ranges.extend(format_spec.elements.literals().map(|literal| {
                StringContentRange {
                    range: literal.range,
                    modifier: FORMAT_SPECIFIER_MODIFIER.clone(),
                }
            }));
            collect_interpolated_string_ranges(
                &format_spec.elements,
                modifiers,
                content_ranges,
                punctuation_ranges,
            );
        }
    }
}

/// Classify an attribute's resolved type into a semantic token kind. For a union,
/// every member must agree on the same kind; any disagreement (or a member that is
/// a plain attribute) falls back to `PROPERTY`.
fn attribute_semantic_token_type(ty: Type) -> SemanticTokenType {
    match ty {
        Type::Union(union) => {
            let mut members = union.members.into_iter();
            let Some(first) = members.next() else {
                return SemanticTokenType::PROPERTY;
            };
            let kind = attribute_semantic_token_type(first);
            if kind == SemanticTokenType::PROPERTY {
                return SemanticTokenType::PROPERTY;
            }
            if members.all(|member| attribute_semantic_token_type(member) == kind) {
                kind
            } else {
                SemanticTokenType::PROPERTY
            }
        }
        Type::Literal(lit) if matches!(lit.value, Lit::Enum(_)) => SemanticTokenType::ENUM_MEMBER,
        _ => {
            attribute_symbol_kind_from_type(&ty)
                .to_lsp_semantic_token_type_with_modifiers()
                .0
        }
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
        if !range.is_empty() && range_overlaps(self.limit_range, range) {
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

    fn push_string_range(
        &mut self,
        range: TextRange,
        modifiers: Vec<SemanticTokenModifier>,
        content_ranges: &[StringContentRange],
    ) {
        let mut boundaries = vec![range.start(), range.end()];
        for content in content_ranges {
            if let Some(intersection) = range.intersect(content.range) {
                boundaries.push(intersection.start());
                boundaries.push(intersection.end());
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        for boundary in boundaries.windows(2) {
            let segment = TextRange::new(boundary[0], boundary[1]);
            let mut segment_modifiers = modifiers.clone();
            for content in content_ranges {
                if content.range.contains_range(segment)
                    && !segment_modifiers.contains(&content.modifier)
                {
                    segment_modifiers.push(content.modifier.clone());
                }
            }
            self.push_if_in_range(segment, SemanticTokenType::STRING, segment_modifiers);
        }
    }

    /// Add syntax-level semantic tokens, using the AST for interpolated and printf formatting.
    pub fn process_syntax_tokens(&mut self, tokens: &Tokens, ast: &ModModule, source: &str) {
        let mut content_ranges = Vec::new();
        let mut punctuation_ranges = Vec::new();
        let mut printf_operands = Vec::new();
        ast.visit(&mut |expr| match expr {
            Expr::FString(expr) => {
                for fstring in expr.value.f_strings() {
                    let mut modifiers = vec![FORMAT_STRING_MODIFIER.clone()];
                    if fstring.flags.prefix().is_raw() {
                        modifiers.push(RAW_STRING_MODIFIER.clone());
                    }
                    collect_interpolated_string_ranges(
                        &fstring.elements,
                        &modifiers,
                        &mut content_ranges,
                        &mut punctuation_ranges,
                    );
                }
            }
            Expr::TString(expr) => {
                for tstring in expr.value.iter() {
                    let mut modifiers = vec![TEMPLATE_STRING_MODIFIER.clone()];
                    if tstring.flags.prefix().is_raw() {
                        modifiers.push(RAW_STRING_MODIFIER.clone());
                    }
                    collect_interpolated_string_ranges(
                        &tstring.elements,
                        &modifiers,
                        &mut content_ranges,
                        &mut punctuation_ranges,
                    );
                }
            }
            Expr::BinOp(binop)
                if binop.op == Operator::Mod
                    && matches!(
                        binop.left.as_ref(),
                        Expr::StringLiteral(_)
                            | Expr::BytesLiteral(_)
                            | Expr::FString(_)
                            | Expr::TString(_)
                    ) =>
            {
                printf_operands.push(binop.left.range());
            }
            _ => {}
        });

        for token in tokens.iter() {
            let kind = token.kind();
            match kind {
                TokenKind::String
                | TokenKind::FStringStart
                | TokenKind::FStringMiddle
                | TokenKind::FStringEnd
                | TokenKind::TStringStart
                | TokenKind::TStringMiddle
                | TokenKind::TStringEnd => {
                    let flags = token.unwrap_string_flags();
                    let mut modifiers = Vec::new();
                    if flags.is_byte_string() {
                        modifiers.push(BYTE_STRING_MODIFIER.clone());
                    }
                    if flags.is_raw_string() {
                        modifiers.push(RAW_STRING_MODIFIER.clone());
                    }
                    match kind {
                        TokenKind::FStringStart
                        | TokenKind::FStringMiddle
                        | TokenKind::FStringEnd => {
                            modifiers.push(FORMAT_STRING_MODIFIER.clone());
                        }
                        TokenKind::TStringStart
                        | TokenKind::TStringMiddle
                        | TokenKind::TStringEnd => {
                            modifiers.push(TEMPLATE_STRING_MODIFIER.clone());
                        }
                        _ => {}
                    }
                    let prefix_len = if matches!(
                        kind,
                        TokenKind::String | TokenKind::FStringStart | TokenKind::TStringStart
                    ) {
                        flags.prefix().text_len()
                    } else {
                        TextSize::default()
                    };
                    if prefix_len > TextSize::default() {
                        self.push_if_in_range(
                            TextRange::at(token.start(), prefix_len),
                            SemanticTokenType::STRING,
                            vec![STRING_PREFIX_MODIFIER.clone()],
                        );
                    }
                    let content_range = TextRange::new(token.start() + prefix_len, token.end());
                    let escape_regex: &Regex = if flags.is_byte_string() {
                        &BYTE_STRING_ESCAPE
                    } else {
                        &STRING_ESCAPE
                    };
                    let mut token_content_ranges = if flags.is_raw_string() {
                        Vec::new()
                    } else {
                        regex_content_ranges(
                            escape_regex,
                            content_range,
                            source,
                            &ESCAPE_SEQUENCE_MODIFIER,
                        )
                    };
                    if printf_operands
                        .iter()
                        .any(|operand| operand.contains_range(token.range()))
                    {
                        token_content_ranges.extend(regex_content_ranges(
                            &PRINTF_PLACEHOLDER,
                            content_range,
                            source,
                            &FORMAT_PLACEHOLDER_MODIFIER,
                        ));
                    }
                    token_content_ranges.extend(
                        content_ranges
                            .iter()
                            .filter(|content| content.range.intersect(content_range).is_some())
                            .map(|content| StringContentRange {
                                range: content.range,
                                modifier: content.modifier.clone(),
                            }),
                    );
                    self.push_string_range(content_range, modifiers, &token_content_ranges);
                }
                _ => {
                    if punctuation_ranges
                        .iter()
                        .any(|special| special.range.contains_range(token.range()))
                    {
                        continue;
                    }
                    if let Some(token_type) = syntax_token_type(kind) {
                        self.push_if_in_range(token.range(), token_type, Vec::new());
                    }
                }
            }
        }
        self.tokens.extend(punctuation_ranges);
    }

    fn process_arguments(&mut self, args: &Arguments) {
        for keyword in &args.keywords {
            if let Some(arg) = &keyword.arg {
                self.push_if_in_range(arg.range, SemanticTokenType::PARAMETER, Vec::new());
            }
        }
    }

    fn process_pattern(&mut self, pattern: &Pattern) {
        Ast::pattern_lvalue(pattern, &mut |name| {
            if !Ast::is_synthesized_empty_identifier(name) {
                self.push_if_in_range(name.range(), SemanticTokenType::VARIABLE, Vec::new());
            }
        });
    }

    fn process_attribute_expr(
        &mut self,
        attr: &ExprAttribute,
        get_type_of_attribute: &dyn Fn(TextRange) -> Option<Type>,
        get_symbol_kind: &dyn Fn(&Key) -> Option<(ModuleName, SymbolKind)>,
    ) {
        let kind = get_type_of_attribute(attr.range())
            .map(attribute_semantic_token_type)
            .unwrap_or(SemanticTokenType::PROPERTY);
        self.push_if_in_range(attr.attr.range(), kind, Vec::new());
        attr.value
            .visit(&mut |x| self.process_expr(x, get_type_of_attribute, get_symbol_kind));
    }

    fn process_expr(
        &mut self,
        x: &Expr,
        get_type_of_attribute: &dyn Fn(TextRange) -> Option<Type>,
        get_symbol_kind: &dyn Fn(&Key) -> Option<(ModuleName, SymbolKind)>,
    ) {
        match x {
            Expr::Name(name) => {
                // Use ExprContext to pick the right key type:
                // Store context -> Definition (name definition sites)
                // Load/Del context -> BoundName (name usages/references)
                let key = match name.ctx {
                    ExprContext::Store => Key::Definition(ShortIdentifier::expr_name(name)),
                    _ => Key::BoundName(ShortIdentifier::expr_name(name)),
                };
                if let Some((def_module, symbol_kind)) = get_symbol_kind(&key) {
                    let (token_type, mut token_modifiers) =
                        symbol_kind.to_lsp_semantic_token_type_with_modifiers();
                    if symbol_kind == SymbolKind::Parameter {
                        maybe_add_self_parameter_modifier(name.id.as_str(), &mut token_modifiers);
                    }
                    maybe_add_default_library_modifier(def_module, &mut token_modifiers);
                    self.push_if_in_range(name.range, token_type, token_modifiers);
                } else if name.ctx == ExprContext::Store {
                    // For Store context (variable definitions), fallback to VARIABLE
                    // even if we can't resolve the symbol kind
                    self.push_if_in_range(name.range, SemanticTokenType::VARIABLE, Vec::new());
                }
            }
            Expr::Call(call) => {
                self.process_arguments(&call.arguments);
                x.recurse(&mut |x| self.process_expr(x, get_type_of_attribute, get_symbol_kind));
            }
            Expr::Attribute(attr) => {
                self.process_attribute_expr(attr, get_type_of_attribute, get_symbol_kind);
            }
            // Comprehensions need special handling because the Visit trait doesn't visit targets
            Expr::ListComp(list_comp) => {
                for comp in &list_comp.generators {
                    comp.target.visit(&mut |e| {
                        self.process_expr(e, get_type_of_attribute, get_symbol_kind)
                    });
                }
                x.recurse(&mut |e| self.process_expr(e, get_type_of_attribute, get_symbol_kind));
            }
            Expr::SetComp(set_comp) => {
                for comp in &set_comp.generators {
                    comp.target.visit(&mut |e| {
                        self.process_expr(e, get_type_of_attribute, get_symbol_kind)
                    });
                }
                x.recurse(&mut |e| self.process_expr(e, get_type_of_attribute, get_symbol_kind));
            }
            Expr::DictComp(dict_comp) => {
                for comp in &dict_comp.generators {
                    comp.target.visit(&mut |e| {
                        self.process_expr(e, get_type_of_attribute, get_symbol_kind)
                    });
                }
                x.recurse(&mut |e| self.process_expr(e, get_type_of_attribute, get_symbol_kind));
            }
            Expr::Generator(generator) => {
                for comp in &generator.generators {
                    comp.target.visit(&mut |e| {
                        self.process_expr(e, get_type_of_attribute, get_symbol_kind)
                    });
                }
                x.recurse(&mut |e| self.process_expr(e, get_type_of_attribute, get_symbol_kind));
            }
            _ => {
                x.recurse(&mut |x| self.process_expr(x, get_type_of_attribute, get_symbol_kind));
            }
        }
    }

    fn process_stmt(
        &mut self,
        x: &Stmt,
        in_class: bool,
        get_symbol_kind: &dyn Fn(&Key) -> Option<(ModuleName, SymbolKind)>,
    ) {
        match x {
            Stmt::ClassDef(class_def) => {
                self.push_if_in_range(class_def.name.range, SemanticTokenType::CLASS, Vec::new());
                if let Some(type_params) = &class_def.type_params {
                    for tp in &type_params.type_params {
                        self.push_if_in_range(
                            tp.name().range(),
                            SemanticTokenType::TYPE_PARAMETER,
                            Vec::new(),
                        );
                    }
                }
                x.recurse(&mut |x| self.process_stmt(x, true, get_symbol_kind));
            }
            Stmt::FunctionDef(function_def) => {
                let token_type = if in_class {
                    SemanticTokenType::METHOD
                } else {
                    SemanticTokenType::FUNCTION
                };
                self.push_if_in_range(function_def.name.range, token_type, Vec::new());
                if let Some(type_params) = &function_def.type_params {
                    for tp in &type_params.type_params {
                        self.push_if_in_range(
                            tp.name().range(),
                            SemanticTokenType::TYPE_PARAMETER,
                            Vec::new(),
                        );
                    }
                }
                // Highlight all parameters as PARAMETER
                for param in function_def.parameters.iter_non_variadic_params() {
                    let mut modifiers = Vec::new();
                    maybe_add_self_parameter_modifier(
                        param.parameter.name.as_str(),
                        &mut modifiers,
                    );
                    self.push_if_in_range(
                        param.parameter.name.range(),
                        SemanticTokenType::PARAMETER,
                        modifiers,
                    );
                }
                if let Some(vararg) = &function_def.parameters.vararg {
                    let mut modifiers = Vec::new();
                    maybe_add_self_parameter_modifier(vararg.name.as_str(), &mut modifiers);
                    self.push_if_in_range(
                        vararg.name.range(),
                        SemanticTokenType::PARAMETER,
                        modifiers,
                    );
                }
                if let Some(kwarg) = &function_def.parameters.kwarg {
                    let mut modifiers = Vec::new();
                    maybe_add_self_parameter_modifier(kwarg.name.as_str(), &mut modifiers);
                    self.push_if_in_range(
                        kwarg.name.range(),
                        SemanticTokenType::PARAMETER,
                        modifiers,
                    );
                }
                x.recurse(&mut |x| self.process_stmt(x, false, get_symbol_kind));
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
                x.recurse(&mut |x| self.process_stmt(x, in_class, get_symbol_kind));
            }
            Stmt::Try(stmt_try) => {
                for ExceptHandler::ExceptHandler(handler) in stmt_try.handlers.iter() {
                    if let Some(name) = &handler.name {
                        self.push_if_in_range(name.range(), SemanticTokenType::VARIABLE, vec![]);
                    }
                }
                x.recurse(&mut |x| self.process_stmt(x, in_class, get_symbol_kind));
            }
            Stmt::With(with) => {
                for with_item in with.items.iter() {
                    if let Some(name) = &with_item.optional_vars {
                        self.push_if_in_range(name.range(), SemanticTokenType::VARIABLE, vec![]);
                    }
                }
                x.recurse(&mut |x| self.process_stmt(x, in_class, get_symbol_kind));
            }
            Stmt::Match(stmt_match) => {
                for case in &stmt_match.cases {
                    self.process_pattern(&case.pattern);
                }
                x.recurse(&mut |x| self.process_stmt(x, in_class, get_symbol_kind));
            }
            Stmt::Import(StmtImport { names, .. }) => {
                for alias in names {
                    // For `import X`, look up the import to get defaultLibrary modifier.
                    // For dotted imports like `import x.y`, the key uses just the first component,
                    // but we can't easily extract that from the AST here, so skip the lookup.
                    let mut modifiers = vec![];
                    if !alias.name.id.contains('.') {
                        let import_key =
                            Key::Import(Box::new((Name::new(&alias.name.id), alias.name.range)));
                        if let Some((def_module, _)) = get_symbol_kind(&import_key) {
                            maybe_add_default_library_modifier(def_module, &mut modifiers);
                        }
                    }
                    self.push_if_in_range(
                        alias.name.range,
                        SemanticTokenType::NAMESPACE,
                        modifiers.clone(),
                    );
                    // If there's an alias, also highlight that as NAMESPACE
                    if let Some(asname) = &alias.asname {
                        self.push_if_in_range(
                            asname.range,
                            SemanticTokenType::NAMESPACE,
                            modifiers,
                        );
                    }
                }
            }
            Stmt::ImportFrom(StmtImportFrom { module, names, .. }) => {
                if let Some(module) = module {
                    self.push_if_in_range(module.range, SemanticTokenType::NAMESPACE, vec![]);
                }
                for alias in names {
                    // Look up the symbol kind using the bound name's key
                    let bound_name = alias.asname.as_ref().unwrap_or(&alias.name);
                    let def_key = Key::Definition(ShortIdentifier::new(bound_name));
                    if let Some((def_module, symbol_kind)) = get_symbol_kind(&def_key) {
                        let (token_type, mut token_modifiers) =
                            symbol_kind.to_lsp_semantic_token_type_with_modifiers();
                        maybe_add_default_library_modifier(def_module, &mut token_modifiers);
                        // If there's an alias, highlight the original name with the resolved type
                        if alias.asname.is_some() {
                            self.push_if_in_range(
                                alias.name.range,
                                token_type.clone(),
                                token_modifiers.clone(),
                            );
                        }
                        self.push_if_in_range(bound_name.range, token_type, token_modifiers);
                    } else {
                        // Fallback to NAMESPACE if we can't resolve
                        if alias.asname.is_some() {
                            self.push_if_in_range(
                                alias.name.range,
                                SemanticTokenType::NAMESPACE,
                                vec![],
                            );
                        }
                        self.push_if_in_range(
                            bound_name.range,
                            SemanticTokenType::NAMESPACE,
                            vec![],
                        );
                    }
                }
            }
            Stmt::AnnAssign(ann_assign) => {
                if let Expr::Name(name) = &*ann_assign.target {
                    self.push_if_in_range(name.range, SemanticTokenType::VARIABLE, vec![]);
                }
                x.recurse(&mut |x| self.process_stmt(x, in_class, get_symbol_kind));
            }
            _ => x.recurse(&mut |x| self.process_stmt(x, in_class, get_symbol_kind)),
        }
    }

    pub fn process_ast(
        &mut self,
        ast: &ModModule,
        get_type_of_attribute: &dyn Fn(TextRange) -> Option<Type>,
        get_symbol_kind: &dyn Fn(&Key) -> Option<(ModuleName, SymbolKind)>,
    ) {
        for s in &ast.body {
            self.process_stmt(s, false, get_symbol_kind);
        }
        ast.visit(&mut |e| self.process_expr(e, get_type_of_attribute, get_symbol_kind));
    }

    pub fn all_tokens_sorted(self) -> Vec<SemanticTokenWithFullRange> {
        let mut tokens = self.tokens;
        tokens.sort_by_key(|a| a.range.start());
        tokens
    }
}

fn collect_disabled_ranges_from_block(
    stmts: &[Stmt],
    sys_info: SysInfo,
    reachable: bool,
    ranges: &mut Vec<TextRange>,
) {
    for stmt in stmts {
        collect_disabled_ranges_from_stmt(stmt, sys_info, reachable, ranges);
    }
}

fn collect_disabled_ranges_from_stmt(
    stmt: &Stmt,
    sys_info: SysInfo,
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

pub(crate) fn disabled_ranges_for_module(ast: &ModModule, sys_info: SysInfo) -> Vec<TextRange> {
    let mut ranges = Vec::new();
    collect_disabled_ranges_from_block(&ast.body, sys_info, true, &mut ranges);
    ranges
}
