/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeMap;

use dupe::Dupe;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use itertools::Itertools;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::CompletionItemLabelDetails;
use lsp_types::CompletionItemTag;
use lsp_types::TextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::keywords::get_keywords;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;
use pyrefly_util::gas::Gas;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Alias;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Deserialize;
use starlark_map::ordered_set::OrderedSet;
use starlark_map::small_map::SmallMap;

use crate::ModuleInfo;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::binding::binding::Key;
use crate::config::error_kind::ErrorKind;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::lsp::module_helpers::collect_symbol_def_paths;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::import_regular_import_edit;
use crate::state::ide::insert_import_edit;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::lsp_attributes::AttributeContext;
use crate::state::require::Require;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;
use crate::types::callable::Param;
use crate::types::module::ModuleType;
use crate::types::type_var::Restriction;
use crate::types::types::Type;

mod quick_fixes;

pub(crate) use self::quick_fixes::extract_function::LocalRefactorCodeAction;

fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum AllOffPartial {
    All,
    #[default]
    Off,
    Partial,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlayHintConfig {
    #[serde(default)]
    pub call_argument_names: AllOffPartial,
    #[serde(default = "default_true")]
    pub function_return_types: bool,
    #[serde(default)]
    pub pytest_parameters: bool,
    #[serde(default = "default_true")]
    pub variable_types: bool,
}

impl Default for InlayHintConfig {
    fn default() -> Self {
        Self {
            call_argument_names: AllOffPartial::Off,
            function_return_types: true,
            pytest_parameters: false,
            variable_types: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ImportFormat {
    #[default]
    Absolute,
    Relative,
}

#[derive(Clone, Copy, Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DisplayTypeErrors {
    #[default]
    Default,
    ForceOff,
    ForceOn,
    /// Only show errors for missing imports and missing sources
    ErrorMissingImports,
}

const RESOLVE_EXPORT_INITIAL_GAS: Gas = Gas::new(100);
pub const MIN_CHARACTERS_TYPED_AUTOIMPORT: usize = 3;

/// Determines what to do when finding definitions. Do we continue searching, or stop somewhere intermediate?
#[derive(Clone, Copy, Debug)]
pub enum ImportBehavior {
    /// Stop at all imports (both renamed and non-renamed)
    StopAtEverything,
    /// Stop at renamed imports (e.g., `from foo import bar as baz`), but jump through non-renamed imports
    StopAtRenamedImports,
    /// Jump through all imports
    JumpThroughEverything,
}

#[derive(Clone, Copy, Debug)]
pub struct FindPreference {
    pub import_behavior: ImportBehavior,
    /// controls whether to prioritize finding pyi or py files. if false, we will search all search paths until a .py file is found before
    /// falling back to a .pyi.
    pub prefer_pyi: bool,
}

impl Default for FindPreference {
    fn default() -> Self {
        Self {
            import_behavior: ImportBehavior::JumpThroughEverything,
            prefer_pyi: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum DefinitionMetadata {
    Attribute,
    Module,
    Variable(Option<SymbolKind>),
    VariableOrAttribute(Option<SymbolKind>),
}

impl DefinitionMetadata {
    pub fn symbol_kind(&self) -> Option<SymbolKind> {
        match self {
            DefinitionMetadata::Attribute => Some(SymbolKind::Attribute),
            DefinitionMetadata::Module => Some(SymbolKind::Module),
            DefinitionMetadata::Variable(symbol_kind) => symbol_kind.as_ref().copied(),
            DefinitionMetadata::VariableOrAttribute(symbol_kind) => symbol_kind.as_ref().copied(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum CalleeKind {
    // Function name
    Function(Identifier),
    // Range of the base expr + method name
    Method(TextRange, Identifier),
    Unknown,
}

fn callee_kind_from_call(call: &ExprCall) -> CalleeKind {
    match call.func.as_ref() {
        Expr::Name(name) => CalleeKind::Function(Ast::expr_name_identifier(name.clone())),
        Expr::Attribute(attr) => CalleeKind::Method(attr.value.range(), attr.attr.clone()),
        _ => CalleeKind::Unknown,
    }
}

/// Generic helper to visit keyword arguments with a custom handler.
/// The handler receives the keyword index and reference, and returns true to stop iteration.
/// This function will also take in a generic function which is used a filter
pub(crate) fn visit_keyword_arguments_until_match<F>(call: &ExprCall, mut filter: F) -> bool
where
    F: FnMut(usize, &Keyword) -> bool,
{
    for (j, kw) in call.arguments.keywords.iter().enumerate() {
        if filter(j, kw) {
            return true;
        }
    }
    false
}

#[derive(Debug)]
pub(crate) enum PatternMatchParameterKind {
    // Name defined using `as`
    // ex: `x` in `case ... as x: ...`, or `x` in `case x: ...`
    AsName,
    // Name defined using keyword argument pattern
    // ex: `x` in `case Foo(x=1): ...`
    KeywordArgName,
    // Name defined using `*` pattern
    // ex: `x` in `case [*x]: ...`
    StarName,
    // Name defined using `**` pattern
    // ex: `x` in case { ..., **x }: ...
    RestName,
}

#[derive(Debug)]
pub(crate) enum IdentifierContext {
    /// An identifier appeared in an expression. ex: `x` in `x + 1`
    Expr(ExprContext),
    /// An identifier appeared as the name of an attribute. ex: `y` in `x.y`
    Attribute {
        /// The range of just the base expression.
        base_range: TextRange,
        /// The range of the entire expression.
        range: TextRange,
    },
    /// An identifier appeared as the name of a keyword argument.
    /// ex: `x` in `f(x=1)`. We also store some info about the callee `f` so
    /// downstream logic can utilize the info.
    KeywordArgument(CalleeKind),
    /// An identifier appeared as the name of an imported module.
    /// ex: `x` in `import x`, or `from x import name`.
    ImportedModule {
        /// Name of the imported module.
        name: ModuleName,
        /// Keeps track of how many leading dots there are for the imported module.
        /// ex: `x.y` in `import x.y` has 0 dots, and `x` in `from ..x.y import z` has 2 dot.
        #[allow(dead_code)]
        dots: u32,
    },
    /// An identifier appeared as the name of a from...import statement.
    /// ex: `x` in `from y import x`.
    ImportedName {
        /// Name of the imported module.
        module_name: ModuleName,
        /// Keeps track of how many leading dots there are for the imported module.
        /// ex: `x.y` in `import x.y` has 0 dots, and `x` in `from ..x.y import z` has 2 dot.
        #[allow(dead_code)]
        dots: u32,
        /// Name of the imported entity in the current module. If there's no as-rename, this will be
        /// the same as the identifier. If there is as-rename, this will be the name after the `as`.
        /// ex: For `from ... import x`, the name is `x`. For `from ... import x as y`, the name is `y`.
        name_after_import: Identifier,
    },
    /// An identifier appeared as the name of a function.
    /// ex: `x` in `def x(...): ...`
    FunctionDef { docstring_range: Option<TextRange> },
    /// An identifier appeared as the name of a method.
    /// ex: `x` in `def x(self, ...): ...` inside a class
    MethodDef { docstring_range: Option<TextRange> },
    /// An identifier appeared as the name of a class.
    /// ex: `x` in `class x(...): ...`
    ClassDef { docstring_range: Option<TextRange> },
    /// An identifier appeared as the name of a parameter.
    /// ex: `x` in `def f(x): ...`
    Parameter,
    /// An identifier appeared as the name of a type parameter.
    /// ex: `T` in `def f[T](...): ...` or `U` in `class C[*U]: ...`
    TypeParameter,
    /// An identifier appeared as the name of an exception declared in
    /// an `except` branch.
    /// ex: `e` in `try ... except Exception as e: ...`
    ExceptionHandler,
    /// An identifier appeared as the name introduced via a `case` branch in a `match` statement.
    /// See [`PatternMatchParameterKind`] for examples.
    #[expect(dead_code)]
    PatternMatch(PatternMatchParameterKind),
}

#[derive(Debug)]
pub(crate) struct IdentifierWithContext {
    pub(crate) identifier: Identifier,
    pub(crate) context: IdentifierContext,
}

#[derive(PartialEq, Eq)]
pub enum AnnotationKind {
    #[allow(dead_code)]
    Parameter,
    Return,
    Variable,
}

impl IdentifierWithContext {
    fn from_stmt_import(id: &Identifier, alias: &Alias) -> Self {
        let identifier = id.clone();
        let module_name = ModuleName::from_str(alias.name.as_str());
        Self {
            identifier,
            context: IdentifierContext::ImportedModule {
                name: module_name,
                dots: 0,
            },
        }
    }

    fn module_name_and_dots(import_from: &StmtImportFrom) -> (ModuleName, u32) {
        (
            if let Some(module) = &import_from.module {
                ModuleName::from_str(module.as_str())
            } else {
                ModuleName::from_str("")
            },
            import_from.level,
        )
    }

    fn from_stmt_import_from_module(id: &Identifier, import_from: &StmtImportFrom) -> Self {
        let identifier = id.clone();
        let (name, dots) = Self::module_name_and_dots(import_from);
        Self {
            identifier,
            context: IdentifierContext::ImportedModule { name, dots },
        }
    }

    fn from_stmt_import_from_name(
        id: &Identifier,
        alias: &Alias,
        import_from: &StmtImportFrom,
    ) -> Self {
        let identifier = id.clone();
        let (module_name, dots) = Self::module_name_and_dots(import_from);
        let name_after_import = if let Some(asname) = &alias.asname {
            asname.clone()
        } else {
            identifier.clone()
        };
        Self {
            identifier,
            context: IdentifierContext::ImportedName {
                module_name,
                dots,
                name_after_import,
            },
        }
    }

    fn from_stmt_function_def(id: &Identifier, docstring_range: Option<TextRange>) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::FunctionDef { docstring_range },
        }
    }

    fn from_stmt_method_def(id: &Identifier, docstring_range: Option<TextRange>) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::MethodDef { docstring_range },
        }
    }

    fn from_stmt_class_def(id: &Identifier, docstring_range: Option<TextRange>) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::ClassDef { docstring_range },
        }
    }

    fn from_parameter(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::Parameter,
        }
    }

    fn from_type_param(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::TypeParameter,
        }
    }

    fn from_exception_handler(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::ExceptionHandler,
        }
    }

    fn from_pattern_match_as(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::PatternMatch(PatternMatchParameterKind::AsName),
        }
    }

    fn from_pattern_match_keyword(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::PatternMatch(PatternMatchParameterKind::KeywordArgName),
        }
    }

    fn from_pattern_match_star(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::PatternMatch(PatternMatchParameterKind::StarName),
        }
    }

    fn from_pattern_match_rest(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::PatternMatch(PatternMatchParameterKind::RestName),
        }
    }

    fn from_keyword_argument(id: &Identifier, call: &ExprCall) -> Self {
        let identifier = id.clone();
        let callee_kind = callee_kind_from_call(call);
        Self {
            identifier,
            context: IdentifierContext::KeywordArgument(callee_kind),
        }
    }

    fn from_expr_attr(id: &Identifier, attr: &ExprAttribute) -> Self {
        let identifier = id.clone();
        Self {
            identifier,
            context: IdentifierContext::Attribute {
                base_range: attr.value.range(),
                range: attr.range(),
            },
        }
    }

    fn from_expr_name(expr_name: &ExprName) -> Self {
        let identifier = Ast::expr_name_identifier(expr_name.clone());
        Self {
            identifier,
            context: IdentifierContext::Expr(expr_name.ctx),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FindDefinitionItemWithDocstring {
    pub metadata: DefinitionMetadata,
    pub definition_range: TextRange,
    pub module: Module,
    pub docstring_range: Option<TextRange>,
    pub display_name: Option<String>,
}

#[derive(Debug)]
pub struct FindDefinitionItem {
    pub metadata: DefinitionMetadata,
    pub definition_range: TextRange,
    pub module: Module,
}

impl<'a> Transaction<'a> {
    pub fn get_type(&self, handle: &Handle, key: &Key) -> Option<Type> {
        let idx = self.get_bindings(handle)?.key_to_idx(key);
        let answers = self.get_answers(handle)?;
        answers.get_type_at(idx)
    }

    pub fn get_type_trace(&self, handle: &Handle, range: TextRange) -> Option<Type> {
        let ans = self.get_answers(handle)?;
        ans.get_type_trace(range)
    }

    fn get_chosen_overload_trace(&self, handle: &Handle, range: TextRange) -> Option<Type> {
        let ans = self.get_answers(handle)?;
        ans.get_chosen_overload_trace(range)
    }

    fn type_from_expression_at(&self, handle: &Handle, position: TextSize) -> Option<Type> {
        let module = self.get_ast(handle)?;
        let covering_nodes = Ast::locate_node(&module, position);
        for node in covering_nodes {
            if node.as_expr_ref().is_none() {
                continue;
            }
            let range = node.range();
            if let Some(callable) = self.get_chosen_overload_trace(handle, range) {
                return Some(callable);
            }
            if let Some(ty) = self.get_type_trace(handle, range) {
                return Some(ty);
            }
        }
        None
    }

    pub(crate) fn identifier_at(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<IdentifierWithContext> {
        let mod_module = self.get_ast(handle)?;
        let covering_nodes = Ast::locate_node(&mod_module, position);
        Self::identifier_from_covering_nodes(&covering_nodes)
    }

    fn identifier_from_covering_nodes(
        covering_nodes: &[AnyNodeRef],
    ) -> Option<IdentifierWithContext> {
        match (
            covering_nodes.first(),
            covering_nodes.get(1),
            covering_nodes.get(2),
            covering_nodes.get(3),
        ) {
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::Alias(alias)),
                Some(AnyNodeRef::StmtImport(_)),
                _,
            ) => {
                // `import id` or `import ... as id`
                Some(IdentifierWithContext::from_stmt_import(id, alias))
            }
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::StmtImportFrom(import_from)),
                _,
                _,
            ) => {
                // `from id import ...`
                Some(IdentifierWithContext::from_stmt_import_from_module(
                    id,
                    import_from,
                ))
            }
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::Alias(alias)),
                Some(AnyNodeRef::StmtImportFrom(import_from)),
                _,
            ) => {
                // `from ... import id`
                Some(IdentifierWithContext::from_stmt_import_from_name(
                    id,
                    alias,
                    import_from,
                ))
            }
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::StmtFunctionDef(stmt)),
                Some(AnyNodeRef::StmtClassDef(_)),
                _,
            ) => {
                // def id(...): ...
                Some(IdentifierWithContext::from_stmt_method_def(
                    id,
                    Docstring::range_from_stmts(&stmt.body),
                ))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::StmtFunctionDef(stmt)), _, _) => {
                // def id(...): ...
                Some(IdentifierWithContext::from_stmt_function_def(
                    id,
                    Docstring::range_from_stmts(&stmt.body),
                ))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::StmtClassDef(stmt)), _, _) => {
                // class id(...): ...
                Some(IdentifierWithContext::from_stmt_class_def(
                    id,
                    Docstring::range_from_stmts(&stmt.body),
                ))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::Parameter(_)), _, _) => {
                // def ...(id): ...
                Some(IdentifierWithContext::from_parameter(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::TypeParamTypeVar(_)), _, _) => {
                // def ...[id](...): ...
                Some(IdentifierWithContext::from_type_param(id))
            }
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::TypeParamTypeVarTuple(_)),
                _,
                _,
            ) => {
                // def ...[*id](...): ...
                Some(IdentifierWithContext::from_type_param(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::TypeParamParamSpec(_)), _, _) => {
                // def ...[**id](...): ...
                Some(IdentifierWithContext::from_type_param(id))
            }
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::ExceptHandlerExceptHandler(_)),
                _,
                _,
            ) => {
                // try ... except ... as id: ...
                Some(IdentifierWithContext::from_exception_handler(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::PatternMatchAs(_)), _, _) => {
                // match ... case ... as id: ...
                Some(IdentifierWithContext::from_pattern_match_as(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::PatternKeyword(_)), _, _) => {
                // match ... case ...(id=...): ...
                Some(IdentifierWithContext::from_pattern_match_keyword(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::PatternMatchStar(_)), _, _) => {
                // match ... case [..., *id]: ...
                Some(IdentifierWithContext::from_pattern_match_star(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::PatternMatchMapping(_)), _, _) => {
                // match ... case {..., **id}: ...
                Some(IdentifierWithContext::from_pattern_match_rest(id))
            }
            (
                Some(AnyNodeRef::Identifier(id)),
                Some(AnyNodeRef::Keyword(_)),
                Some(AnyNodeRef::Arguments(_)),
                Some(AnyNodeRef::ExprCall(call)),
            ) => {
                // XXX(..., id=..., ...)
                Some(IdentifierWithContext::from_keyword_argument(id, call))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::ExprAttribute(attr)), _, _) => {
                // `XXX.id`
                Some(IdentifierWithContext::from_expr_attr(id, attr))
            }
            (Some(AnyNodeRef::ExprName(name)), _, _, _) => {
                Some(IdentifierWithContext::from_expr_name(name))
            }
            _ => None,
        }
    }

    fn callee_at(&self, handle: &Handle, position: TextSize) -> Option<ExprCall> {
        let mod_module = self.get_ast(handle)?;
        fn f(x: &Expr, find: TextSize, res: &mut Option<ExprCall>) {
            if let Expr::Call(call) = x
                && call.func.range().contains_inclusive(find)
            {
                f(call.func.as_ref(), find, res);
                if res.is_some() {
                    return;
                }
                *res = Some(call.clone());
            } else {
                x.recurse(&mut |x| f(x, find, res));
            }
        }
        let mut res = None;
        mod_module.visit(&mut |x| f(x, position, &mut res));
        res
    }

    fn refine_param_location_for_callee(
        &self,
        ast: &ModModule,
        callee_range: TextRange,
        param_name: &Identifier,
    ) -> Option<TextRange> {
        let covering_nodes = Ast::locate_node(ast, callee_range.start());
        match (covering_nodes.first(), covering_nodes.get(1)) {
            (Some(AnyNodeRef::Identifier(_)), Some(AnyNodeRef::StmtFunctionDef(function_def))) => {
                // Only check regular and kwonly params since posonly params cannot be passed by name
                // on the caller side.
                for regular_param in function_def.parameters.args.iter() {
                    if regular_param.name().id() == param_name.id() {
                        return Some(regular_param.name().range());
                    }
                }
                for kwonly_param in function_def.parameters.kwonlyargs.iter() {
                    if kwonly_param.name().id() == param_name.id() {
                        return Some(kwonly_param.name().range());
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn definition_at(&self, handle: &Handle, position: TextSize) -> Option<Key> {
        self.get_bindings(handle)?
            .definition_at_position(position)
            .cloned()
    }

    pub fn get_type_at(&self, handle: &Handle, position: TextSize) -> Option<Type> {
        // TODO(grievejia): Remove the usage of `definition_at()`: it doesn't reliably detect all
        // definitions.
        if let Some(key) = self.definition_at(handle, position) {
            return self.get_type(handle, &key);
        }

        match self.identifier_at(handle, position) {
            Some(IdentifierWithContext {
                identifier: id,
                context: IdentifierContext::Expr(expr_context),
            }) => {
                let key = match expr_context {
                    ExprContext::Store => Key::Definition(ShortIdentifier::new(&id)),
                    ExprContext::Load | ExprContext::Del | ExprContext::Invalid => {
                        Key::BoundName(ShortIdentifier::new(&id))
                    }
                };

                let bindings = self.get_bindings(handle)?;
                if !bindings.is_valid_key(&key) {
                    return None;
                }
                let mut ty = self.get_type(handle, &key)?;
                let call_args_range = self.callee_at(handle, position).and_then(
                    |ExprCall {
                         func, arguments, ..
                     }| (func.range() == id.range).then_some(arguments.range),
                );
                if let Some(arguments_range) = call_args_range {
                    if let Some(ret) = self.get_chosen_overload_trace(handle, arguments_range) {
                        return Some(ret);
                    }
                    ty = self.coerce_type_to_callable(handle, ty);
                }
                Some(ty)
            }
            Some(IdentifierWithContext {
                identifier: _,
                context:
                    IdentifierContext::ImportedModule {
                        name: module_name, ..
                    },
            }) => {
                // TODO: Handle relative import (via ModuleName::new_maybe_relative)
                Some(Type::Module(ModuleType::new(
                    module_name.first_component(),
                    OrderedSet::from_iter([module_name]),
                )))
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::ImportedName { .. },
            }) => {
                // TODO(grievejia): handle definitions of imported names
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context:
                    IdentifierContext::FunctionDef { docstring_range: _ }
                    | IdentifierContext::MethodDef { docstring_range: _ },
            }) => {
                // TODO(grievejia): Handle definitions of functions
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::ClassDef { docstring_range: _ },
            }) => {
                // TODO(grievejia): Handle definitions of classes
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Parameter,
            }) => {
                // TODO(grievejia): Handle definitions of params
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::TypeParameter,
            }) => {
                // TODO(grievejia): Handle definitions of type params
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::ExceptionHandler,
            }) => {
                // TODO(grievejia): Handle definitions of exception names
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::PatternMatch(_),
            }) => {
                // TODO(grievejia): Handle definitions of pattern-introduced names
                None
            }
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::KeywordArgument(callee_kind),
            }) => self
                .find_definition_for_keyword_argument(
                    handle,
                    &identifier,
                    &callee_kind,
                    FindPreference::default(),
                )
                .first()
                .and_then(|item| {
                    self.definition_at(handle, item.definition_range.start())
                        .and_then(|key| self.get_type(handle, &key))
                }),
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Attribute { range, .. },
            }) => {
                if let Some(ExprCall {
                    node_index: _,
                    range: _,
                    func,
                    arguments,
                }) = &self.callee_at(handle, position)
                    && func.range() == range
                    && let Some(ret) = self.get_chosen_overload_trace(handle, arguments.range)
                {
                    Some(ret)
                } else {
                    self.get_type_trace(handle, range)
                }
            }
            None => self.type_from_expression_at(handle, position),
        }
    }

    /// If `ty` represents a callable instance (e.g., a class with `__call__`), return the
    /// bound `__call__` signature. Otherwise, return the type unchanged.
    ///
    /// This enables IDE features like hover and signature help to show parameter lists
    /// when calling instances that implement `__call__`, matching Python's runtime behavior.
    ///
    /// Note that we should only use this when we already know the value is being used as a
    /// callee, since this drops the original type information in favor of a callable type.
    pub(crate) fn coerce_type_to_callable(&self, handle: &Handle, ty: Type) -> Type {
        if ty.is_toplevel_callable() {
            return ty;
        }
        let original = ty.clone();
        self.ad_hoc_solve(handle, |solver| Self::callable_from_type(&solver, ty))
            .and_then(|callable| callable)
            .unwrap_or(original)
    }

    /// Extract a callable type from `ty` by invoking the solver to find its `__call__` method.
    /// Recursively walks through type wrappers (Union, TypeAlias, Type, Quantified).
    /// Returns `None` if the type is not callable.
    fn callable_from_type(solver: &AnswersSolver<TransactionHandle<'_>>, ty: Type) -> Option<Type> {
        if ty.is_toplevel_callable() {
            return Some(ty);
        }
        match ty {
            Type::ClassType(class_type) => solver.type_order().instance_as_dunder_call(&class_type),
            Type::SelfType(class_type) => solver.type_order().instance_as_dunder_call(&class_type),
            Type::Union(box Union { members, .. }) => Self::callable_from_types(solver, members),
            Type::TypeAlias(alias) => Self::callable_from_type(solver, alias.as_type()),
            Type::Type(box inner) => Self::callable_from_type(solver, inner),
            Type::Quantified(box quantified) => match quantified.restriction {
                Restriction::Bound(bound) => Self::callable_from_type(solver, bound),
                Restriction::Constraints(options) => Self::callable_from_types(solver, options),
                Restriction::Unrestricted => None,
            },
            _ => None,
        }
    }

    /// Convert a collection of types into a single callable union, returning `None` if the list
    /// was empty or any member failed to coerce into a callable.
    fn callable_from_types(
        solver: &AnswersSolver<TransactionHandle<'_>>,
        types: Vec<Type>,
    ) -> Option<Type> {
        if types.is_empty() {
            return None;
        }
        let mut converted = Vec::with_capacity(types.len());
        for ty in types {
            let callable = Self::callable_from_type(solver, ty)?;
            converted.push(callable);
        }
        if converted.len() == 1 {
            converted.into_iter().next()
        } else {
            Some(solver.unions(converted))
        }
    }

    fn resolve_named_import(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        name: Name,
        preference: FindPreference,
    ) -> Option<(Handle, Export)> {
        let mut m = module_name;
        let mut gas = RESOLVE_EXPORT_INITIAL_GAS;
        let mut name = name;
        while !gas.stop() {
            let handle = match preference.prefer_pyi {
                true => self.import_handle(handle, m, None).finding()?,
                false => self
                    .import_handle_prefer_executable(handle, m, None)
                    .finding()?,
            };
            match self.get_exports(&handle).get(&name) {
                Some(ExportLocation::ThisModule(export)) => {
                    return Some((handle.clone(), export.clone()));
                }
                Some(ExportLocation::OtherModule(module, aliased_name)) => {
                    if let Some(aliased_name) = aliased_name {
                        name = aliased_name.clone();
                    }
                    m = *module;
                }
                None => return None,
            }
        }
        None
    }

    /// The behavior of import resolution depends on `preference.import_behavior`:
    /// - `JumpThroughNothing`: Stop at all imports (both renamed and non-renamed)
    /// - `JumpThroughRenamedImports`: Stop at renamed imports like `from foo import bar as baz`, but jump through non-renamed imports
    /// - `JumpThroughEverything`: Jump through all imports
    fn resolve_intermediate_definition(
        &self,
        handle: &Handle,
        intermediate_definition: IntermediateDefinition,
        preference: FindPreference,
    ) -> Option<(Handle, Export)> {
        match intermediate_definition {
            IntermediateDefinition::Local(export) => Some((handle.dupe(), export)),
            IntermediateDefinition::NamedImport(
                import_key,
                module_name,
                name,
                original_name_range,
            ) => {
                let (def_handle, export) =
                    self.resolve_named_import(handle, module_name, name, preference)?;
                // Determine whether to stop at the import or follow through
                let should_stop_at_import = match preference.import_behavior {
                    ImportBehavior::StopAtEverything => {
                        // Stop at ALL imports
                        true
                    }
                    ImportBehavior::StopAtRenamedImports => {
                        // Stop only at renamed imports
                        original_name_range.is_some()
                    }
                    ImportBehavior::JumpThroughEverything => {
                        // Follow through all imports
                        false
                    }
                };

                if should_stop_at_import {
                    Some((
                        handle.dupe(),
                        Export {
                            location: import_key,
                            ..export
                        },
                    ))
                } else {
                    Some((def_handle, export))
                }
            }
            IntermediateDefinition::Module(name) => {
                let handle = match preference.prefer_pyi {
                    true => self.import_handle(handle, name, None).finding()?,
                    false => self
                        .import_handle_prefer_executable(handle, name, None)
                        .finding()?,
                };
                let docstring_range = self.get_module_docstring_range(&handle);
                Some((
                    handle,
                    Export {
                        location: TextRange::default(),
                        symbol_kind: Some(SymbolKind::Module),
                        docstring_range,
                        deprecation: None,
                        special_export: None,
                    },
                ))
            }
        }
    }

    fn resolve_attribute_definition(
        &self,
        handle: &Handle,
        attr_name: &Name,
        definition: AttrDefinition,
        docstring_range: Option<TextRange>,
        preference: FindPreference,
    ) -> Option<(TextRangeWithModule, Option<TextRange>)> {
        match definition {
            AttrDefinition::FullyResolved(text_range_with_module_info) => {
                // If prefer_pyi is false and the current module is a .pyi file,
                // try to find the corresponding .py file
                if !preference.prefer_pyi
                    && text_range_with_module_info.module.path().is_interface()
                    && let Some((exec_module, exec_range, exec_docstring)) = self
                        .search_corresponding_py_module_for_attribute(
                            handle,
                            attr_name,
                            &text_range_with_module_info,
                        )
                {
                    return Some((
                        TextRangeWithModule::new(exec_module, exec_range),
                        exec_docstring,
                    ));
                }
                Some((text_range_with_module_info, docstring_range))
            }
            AttrDefinition::PartiallyResolvedImportedModuleAttribute { module_name } => {
                let (handle, export) =
                    self.resolve_named_import(handle, module_name, attr_name.clone(), preference)?;
                let module_info = self.get_module_info(&handle)?;
                Some((
                    TextRangeWithModule::new(module_info, export.location),
                    export.docstring_range,
                ))
            }
        }
    }

    /// Find the .py definition for a corresponding .pyi definition by importing
    /// and parsing the AST, looking for classes/functions.
    fn search_corresponding_py_module_for_attribute(
        &self,
        request_handle: &Handle,
        attr_name: &Name,
        pyi_definition: &TextRangeWithModule,
    ) -> Option<(Module, TextRange, Option<TextRange>)> {
        let context = AttributeContext::from_module(&pyi_definition.module, pyi_definition.range)?;
        let executable_handle = self
            .import_handle_prefer_executable(request_handle, pyi_definition.module.name(), None)
            .finding()?;
        if executable_handle.path().style() != ModuleStyle::Executable {
            return None;
        }
        let _ = self.get_exports(&executable_handle);
        let executable_module = self.get_module_info(&executable_handle)?;
        let ast = self.get_ast(&executable_handle).unwrap_or_else(|| {
            Ast::parse(
                executable_module.contents(),
                executable_module.source_type(),
            )
            .0
            .into()
        });
        let (def_range, docstring_range) =
            crate::state::lsp_attributes::definition_from_executable_ast(
                ast.as_ref(),
                &context,
                attr_name,
            )?;
        Some((executable_module, def_range, docstring_range))
    }

    pub fn key_to_export(
        &self,
        handle: &Handle,
        key: &Key,
        preference: FindPreference,
    ) -> Option<(Handle, Export)> {
        let bindings = self.get_bindings(handle)?;
        let intermediate_definition = key_to_intermediate_definition(&bindings, key)?;
        let (definition_handle, mut export) =
            self.resolve_intermediate_definition(handle, intermediate_definition, preference)?;
        if let Export {
            symbol_kind: Some(symbol_kind),
            ..
        } = &export
            && *symbol_kind == SymbolKind::Variable
            && let Some(type_) = self.get_type(handle, key)
        {
            let symbol_kind = match type_ {
                Type::Callable(_) | Type::Function(_) => SymbolKind::Function,
                Type::BoundMethod(_) => SymbolKind::Method,
                Type::ClassDef(_) | Type::Type(_) => SymbolKind::Class,
                Type::Module(_) => SymbolKind::Module,
                Type::TypeAlias(_) => SymbolKind::TypeAlias,
                _ => *symbol_kind,
            };
            export.symbol_kind = Some(symbol_kind);
        }
        Some((definition_handle, export))
    }

    // This is for cases where we are 100% certain that `identifier` points to a "real" name
    // definition at a known context (e.g. `identifier is the name of a function or class`).
    // If we are not certain (e.g. `identifier` is imported from another module so it's "real"
    // definition could be somewhere else), use `find_definition_for_name_def()` instead.
    fn find_definition_for_simple_def(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        symbol_kind: SymbolKind,
    ) -> Option<FindDefinitionItem> {
        Some(FindDefinitionItem {
            metadata: DefinitionMetadata::Variable(Some(symbol_kind)),
            module: self.get_module_info(handle)?,
            definition_range: identifier.range,
        })
    }

    fn find_export_for_key(
        &self,
        handle: &Handle,
        key: &Key,
        preference: FindPreference,
    ) -> Option<(Handle, Export)> {
        if !self.get_bindings(handle)?.is_valid_key(key) {
            return None;
        }
        self.key_to_export(handle, key, preference)
    }

    fn find_definition_for_name_def(
        &self,
        handle: &Handle,
        name: &Identifier,
        preference: FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        let def_key = Key::Definition(ShortIdentifier::new(name));
        let (
            handle,
            Export {
                location,
                symbol_kind,
                docstring_range,
                ..
            },
        ) = self.find_export_for_key(handle, &def_key, preference)?;
        let module_info = self.get_module_info(&handle)?;
        Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::VariableOrAttribute(symbol_kind),
            definition_range: location,
            module: module_info,
            docstring_range,
            display_name: Some(name.id.to_string()),
        })
    }

    pub fn find_definition_for_name_use(
        &self,
        handle: &Handle,
        name: &Identifier,
        preference: FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        let use_key = Key::BoundName(ShortIdentifier::new(name));
        let (
            handle,
            Export {
                location,
                symbol_kind,
                docstring_range,
                ..
            },
        ) = self.find_export_for_key(handle, &use_key, preference)?;
        Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::Variable(symbol_kind),
            definition_range: location,
            module: self.get_module_info(&handle)?,
            docstring_range,
            display_name: Some(name.id.to_string()),
        })
    }

    fn find_definition_for_base_type(
        &self,
        handle: &Handle,
        preference: FindPreference,
        completions: Vec<AttrInfo>,
        name: &Name,
    ) -> Option<FindDefinitionItemWithDocstring> {
        completions.into_iter().find_map(|x| {
            if &x.name == name {
                let (definition, docstring_range) = self.resolve_attribute_definition(
                    handle,
                    &x.name,
                    x.definition?,
                    x.docstring_range,
                    preference,
                )?;
                Some(FindDefinitionItemWithDocstring {
                    metadata: DefinitionMetadata::Attribute,
                    definition_range: definition.range,
                    module: definition.module,
                    docstring_range,
                    display_name: Some(name.to_string()),
                })
            } else {
                None
            }
        })
    }

    fn find_attribute_definition_for_base_type(
        &self,
        handle: &Handle,
        preference: FindPreference,
        base_type: Type,
        name: &Name,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        self.ad_hoc_solve(handle, |solver| {
            let completions = |ty| solver.completions(ty, Some(name), false);

            match base_type {
                Type::Union(box Union { members: tys, .. }) | Type::Intersect(box (tys, _)) => tys
                    .into_iter()
                    .filter_map(|ty_| {
                        self.find_definition_for_base_type(
                            handle,
                            preference,
                            completions(ty_),
                            name,
                        )
                    })
                    .collect(),
                ty => self
                    .find_definition_for_base_type(handle, preference, completions(ty), name)
                    .map_or(vec![], |item| vec![item]),
            }
        })
        .unwrap_or_default()
    }

    fn find_definition_for_operator(
        &self,
        handle: &Handle,
        covering_nodes: &[AnyNodeRef],
        preference: FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        let Some((base_type, dunder_method_name)) =
            covering_nodes.iter().find_map(|node| match node {
                AnyNodeRef::ExprCompare(compare) => {
                    for op in &compare.ops {
                        if let Some(dunder_name) = dunder::rich_comparison_dunder(*op)
                            && let Some(answers) = self.get_answers(handle)
                            && let Some(left_type) = answers.get_type_trace(compare.left.range())
                        {
                            return Some((left_type, dunder_name));
                        }
                    }
                    None
                }
                AnyNodeRef::ExprBinOp(binop) => {
                    let dunder_name = Name::new_static(binop.op.dunder());
                    if let Some(answers) = self.get_answers(handle)
                        && let Some(left_type) = answers.get_type_trace(binop.left.range())
                    {
                        return Some((left_type, dunder_name));
                    }
                    None
                }
                AnyNodeRef::ExprUnaryOp(unaryop) => {
                    let dunder_name = match unaryop.op {
                        UnaryOp::Invert => Some(dunder::INVERT),
                        UnaryOp::Not => None,
                        UnaryOp::UAdd => Some(dunder::POS),
                        UnaryOp::USub => Some(dunder::NEG),
                    };
                    if let Some(dunder_name) = dunder_name
                        && let Some(answers) = self.get_answers(handle)
                        && let Some(operand_type) = answers.get_type_trace(unaryop.operand.range())
                    {
                        return Some((operand_type, dunder_name));
                    }
                    None
                }
                AnyNodeRef::ExprSubscript(subscript) => {
                    let dunder_name = match subscript.ctx {
                        ExprContext::Load => Some(dunder::GETITEM),
                        ExprContext::Store => Some(dunder::SETITEM),
                        ExprContext::Del => Some(dunder::DELITEM),
                        ExprContext::Invalid => None,
                    };
                    if let Some(dunder_name) = dunder_name
                        && let Some(answers) = self.get_answers(handle)
                        && let Some(base_type) = answers.get_type_trace(subscript.value.range())
                    {
                        return Some((base_type, dunder_name));
                    }
                    None
                }
                _ => None,
            })
        else {
            return vec![];
        };

        // Find the attribute definition for the dunder method on the base type
        self.find_attribute_definition_for_base_type(
            handle,
            preference,
            base_type,
            &dunder_method_name,
        )
    }

    pub fn find_definition_for_attribute(
        &self,
        handle: &Handle,
        base_range: TextRange,
        name: &Name,
        preference: FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        if let Some(answers) = self.get_answers(handle)
            && let Some(base_type) = answers.get_type_trace(base_range)
        {
            self.find_attribute_definition_for_base_type(handle, preference, base_type, name)
        } else {
            vec![]
        }
    }

    fn find_definition_for_imported_module(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        preference: FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        // TODO: Handle relative import (via ModuleName::new_maybe_relative)
        let handle = match preference.prefer_pyi {
            true => self.import_handle(handle, module_name, None).finding()?,
            false => self
                .import_handle_prefer_executable(handle, module_name, None)
                .finding()?,
        };
        // if the module is not yet loaded, force loading by asking for exports
        // necessary for imports that are not in tdeps (e.g. .py when there is also a .pyi)
        // todo(kylei): better solution
        let _ = self.get_exports(&handle);

        let module_info = self.get_module_info(&handle)?;
        Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::Module,
            definition_range: TextRange::default(),
            module: module_info,
            docstring_range: self.get_module_docstring_range(&handle),
            display_name: Some(module_name.to_string()),
        })
    }

    fn find_definition_for_keyword_argument(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        callee_kind: &CalleeKind,
        preference: FindPreference,
    ) -> Vec<FindDefinitionItem> {
        // NOTE(grievejia): There might be a better way to compute this that doesn't require 2 containing node
        // traversal, once we gain access to the callee function def from callee_kind directly.
        let callee_locations = self.get_callee_location(handle, callee_kind, preference);
        if callee_locations.is_empty() {
            return vec![];
        }

        // Group all locations by their containing module, so later we could avoid reparsing
        // the same module multiple times.
        let location_count = callee_locations.len();
        let mut modules_to_ranges: SmallMap<Module, Vec<TextRange>> =
            SmallMap::with_capacity(location_count);
        for TextRangeWithModule { module, range } in callee_locations.into_iter() {
            modules_to_ranges.entry(module).or_default().push(range)
        }

        let mut results: Vec<FindDefinitionItem> = Vec::with_capacity(location_count);
        for (module_info, ranges) in modules_to_ranges.into_iter() {
            let ast = {
                let handle = Handle::new(
                    module_info.name(),
                    module_info.path().dupe(),
                    handle.sys_info().dupe(),
                );
                self.get_ast(&handle).unwrap_or_else(|| {
                    // We may not have the AST available for the handle if it's not opened -- in that case,
                    // Re-parse the module to get the AST.
                    Ast::parse(module_info.contents(), module_info.source_type())
                        .0
                        .into()
                })
            };

            for range in ranges.into_iter() {
                let refined_param_range =
                    self.refine_param_location_for_callee(ast.as_ref(), range, identifier);
                // TODO(grievejia): Should we filter out unrefinable ranges here?
                results.push(FindDefinitionItem {
                    metadata: DefinitionMetadata::Variable(Some(SymbolKind::Variable)),
                    definition_range: refined_param_range.unwrap_or(range),
                    module: module_info.dupe(),
                })
            }
        }
        results
    }

    fn get_callee_location(
        &self,
        handle: &Handle,
        callee_kind: &CalleeKind,
        preference: FindPreference,
    ) -> Vec<TextRangeWithModule> {
        let defs = match callee_kind {
            CalleeKind::Function(name) => self
                .find_definition_for_name_use(handle, name, preference)
                .map_or(vec![], |item| vec![item]),
            CalleeKind::Method(base_range, name) => {
                self.find_definition_for_attribute(handle, *base_range, name.id(), preference)
            }
            CalleeKind::Unknown => vec![],
        };
        defs.into_iter()
            .map(|item| TextRangeWithModule::new(item.module, item.definition_range))
            .collect()
    }

    /// Find the definition, metadata and optionally the docstring for the given position.
    pub fn find_definition(
        &self,
        handle: &Handle,
        position: TextSize,
        preference: FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        let Some(mod_module) = self.get_ast(handle) else {
            return vec![];
        };
        let covering_nodes = Ast::locate_node(&mod_module, position);

        match Self::identifier_from_covering_nodes(&covering_nodes) {
            Some(IdentifierWithContext {
                identifier: id,
                context: IdentifierContext::Expr(expr_context),
            }) => {
                match expr_context {
                    ExprContext::Store => {
                        // This is a variable definition
                        // Can't use `find_definition_for_simple_def()` here because not all assignments
                        // are guaranteed defs: they might be a modification to a name defined somewhere
                        // else.
                        self.find_definition_for_name_def(handle, &id, preference)
                            .map_or(vec![], |item| vec![item])
                    }
                    ExprContext::Load | ExprContext::Del | ExprContext::Invalid => {
                        // This is a usage of the variable
                        self.find_definition_for_name_use(handle, &id, preference)
                            .map_or(vec![], |item| vec![item])
                    }
                }
            }
            Some(IdentifierWithContext {
                identifier: _,
                context:
                    IdentifierContext::ImportedModule {
                        name: module_name, ..
                    },
            }) => self
                .find_definition_for_imported_module(handle, module_name, preference)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier: _,
                context:
                    IdentifierContext::ImportedName {
                        name_after_import, ..
                    },
            }) => self
                .find_definition_for_name_def(handle, &name_after_import, preference)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::MethodDef { docstring_range },
            }) => self.get_module_info(handle).map_or(vec![], |module| {
                vec![FindDefinitionItemWithDocstring {
                    metadata: DefinitionMetadata::Attribute,
                    module,
                    definition_range: identifier.range,
                    docstring_range,
                    display_name: Some(identifier.id.to_string()),
                }]
            }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::FunctionDef { docstring_range },
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Function)
                .map_or(vec![], |item| {
                    vec![FindDefinitionItemWithDocstring {
                        metadata: item.metadata,
                        definition_range: item.definition_range,
                        module: item.module,
                        docstring_range,
                        display_name: Some(identifier.id.to_string()),
                    }]
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ClassDef { docstring_range },
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Class)
                .map_or(vec![], |item| {
                    vec![FindDefinitionItemWithDocstring {
                        metadata: item.metadata,
                        definition_range: item.definition_range,
                        module: item.module,
                        docstring_range,
                        display_name: Some(identifier.id.to_string()),
                    }]
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::Parameter,
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Parameter)
                .map_or(vec![], |item| {
                    vec![FindDefinitionItemWithDocstring {
                        metadata: item.metadata,
                        definition_range: item.definition_range,
                        module: item.module,
                        docstring_range: None,
                        display_name: Some(identifier.id.to_string()),
                    }]
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::TypeParameter,
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::TypeParameter)
                .map_or(vec![], |item| {
                    vec![FindDefinitionItemWithDocstring {
                        metadata: item.metadata,
                        definition_range: item.definition_range,
                        module: item.module,
                        docstring_range: None,
                        display_name: Some(identifier.id.to_string()),
                    }]
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ExceptionHandler | IdentifierContext::PatternMatch(_),
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Variable)
                .map_or(vec![], |item| {
                    vec![FindDefinitionItemWithDocstring {
                        metadata: item.metadata,
                        definition_range: item.definition_range,
                        module: item.module,
                        docstring_range: None,
                        display_name: Some(identifier.id.to_string()),
                    }]
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::KeywordArgument(callee_kind),
            }) => self
                .find_definition_for_keyword_argument(handle, &identifier, &callee_kind, preference)
                .map(|item| FindDefinitionItemWithDocstring {
                    metadata: item.metadata.clone(),
                    definition_range: item.definition_range,
                    module: item.module.clone(),
                    docstring_range: None,
                    display_name: Some(identifier.id.to_string()),
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::Attribute { base_range, .. },
            }) => {
                self.find_definition_for_attribute(handle, base_range, identifier.id(), preference)
            }
            None => {
                // Check if this is a None literal, if so, resolve to NoneType class
                if covering_nodes
                    .iter()
                    .any(|node| matches!(node, AnyNodeRef::ExprNoneLiteral(_)))
                    && let Some(res) = self.find_definition_for_none(handle)
                {
                    return res;
                }
                // Fall back to operator handling
                self.find_definition_for_operator(handle, &covering_nodes, preference)
            }
        }
    }

    /// Get the definition we should point at for `None`.
    fn find_definition_for_none(
        &self,
        handle: &Handle,
    ) -> Option<Vec<FindDefinitionItemWithDocstring>> {
        let stdlib = self.get_stdlib(handle);
        let none_type = stdlib.none_type().clone().to_type();
        let symbol_def_paths = collect_symbol_def_paths(&none_type);
        if symbol_def_paths.is_empty() {
            None
        } else {
            Some(symbol_def_paths.map(|(qname, _)| {
                let module_info = qname.module().clone();
                FindDefinitionItemWithDocstring {
                    metadata: DefinitionMetadata::VariableOrAttribute(Some(SymbolKind::Class)),
                    module: module_info,
                    definition_range: qname.range(),
                    docstring_range: None,
                    display_name: None,
                }
            }))
        }
    }

    pub fn goto_definition(&self, handle: &Handle, position: TextSize) -> Vec<TextRangeWithModule> {
        let mut definitions = self.find_definition(
            handle,
            position,
            FindPreference {
                prefer_pyi: false,
                ..Default::default()
            },
        );
        // Add pyi definitions if we haven't found any py definition
        if definitions.is_empty() {
            definitions.append(&mut self.find_definition(
                handle,
                position,
                FindPreference::default(),
            ));
        }

        definitions.into_map(|item| TextRangeWithModule::new(item.module, item.definition_range))
    }

    pub fn goto_declaration(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Vec<TextRangeWithModule> {
        // Go-to declaration stops at intermediate definitions (imports, type stubs)
        // rather than jumping through to the final implementation
        let definitions = self.find_definition(
            handle,
            position,
            FindPreference {
                import_behavior: ImportBehavior::StopAtEverything,
                prefer_pyi: true,
            },
        );

        definitions.into_map(|item| TextRangeWithModule::new(item.module, item.definition_range))
    }

    pub fn goto_type_definition(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Vec<TextRangeWithModule> {
        let type_ = self.get_type_at(handle, position);

        if let Some(t) = type_ {
            let symbol_def_paths = collect_symbol_def_paths(&t);

            if !symbol_def_paths.is_empty() {
                return symbol_def_paths.map(|(qname, _)| {
                    TextRangeWithModule::new(qname.module().clone(), qname.range())
                });
            }
        }

        self.find_definition(handle, position, FindPreference::default())
            .into_map(|item| TextRangeWithModule::new(item.module, item.definition_range))
    }

    /// This function should not be used for user-facing go-to-definition. However, it is exposed to
    /// tests so that we can test the behavior that's useful for find-refs.
    #[cfg(test)]
    pub(crate) fn goto_definition_do_not_jump_through_renamed_import(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<TextRangeWithModule> {
        self.find_definition(
            handle,
            position,
            FindPreference {
                import_behavior: ImportBehavior::StopAtRenamedImports,
                ..Default::default()
            },
        )
        .into_iter()
        .next()
        .map(|item| TextRangeWithModule::new(item.module, item.definition_range))
    }

    fn search_modules_fuzzy(&self, pattern: &str) -> Vec<ModuleName> {
        let matcher = SkimMatcherV2::default().smart_case();
        let mut results = Vec::new();

        for module_name in self.modules() {
            let module_name_str = module_name.as_str();

            // Skip builtins module
            if module_name_str == "builtins" {
                continue;
            }

            let components = module_name.components();
            let last_component = components.last().map(|name| name.as_str()).unwrap_or("");
            if let Some(score) = matcher.fuzzy_match(last_component, pattern) {
                results.push((score, module_name));
            }
        }

        results.sort_by_key(|(score, _)| Reverse(*score));
        results.into_map(|(_, module_name)| module_name)
    }

    /// Produce code actions that makes edits local to the file.
    pub fn local_quickfix_code_actions_sorted(
        &self,
        handle: &Handle,
        range: TextRange,
        import_format: ImportFormat,
    ) -> Option<Vec<(String, Module, TextRange, String)>> {
        let module_info = self.get_module_info(handle)?;
        let ast = self.get_ast(handle)?;
        let errors = self.get_errors(vec![handle]).collect_errors().shown;
        let mut code_actions = Vec::new();
        for error in errors {
            match error.error_kind() {
                ErrorKind::UnknownName => {
                    let error_range = error.range();
                    if error_range.contains_range(range) {
                        let unknown_name = module_info.code_at(error_range);
                        for (handle_to_import_from, export) in
                            self.search_exports_exact(unknown_name)
                        {
                            let (position, insert_text, _) = insert_import_edit(
                                &ast,
                                self.config_finder(),
                                handle.dupe(),
                                handle_to_import_from.dupe(),
                                unknown_name,
                                import_format,
                            );
                            let range = TextRange::at(position, TextSize::new(0));
                            let is_deprecated = export.deprecation.is_some();
                            let title = format!(
                                "Insert import: `{}`{}",
                                insert_text.trim(),
                                if is_deprecated { " (deprecated)" } else { "" }
                            );

                            let is_private_import = handle_to_import_from
                                .module()
                                .components()
                                .last()
                                .is_some_and(|component| component.as_str().starts_with('_'));

                            code_actions.push((
                                title,
                                module_info.dupe(),
                                range,
                                insert_text,
                                is_deprecated,
                                is_private_import,
                            ));
                        }

                        for module_name in self.search_modules_fuzzy(unknown_name) {
                            if module_name == handle.module() {
                                continue;
                            }
                            if let Some(module_handle) =
                                self.import_handle(handle, module_name, None).finding()
                            {
                                let (position, insert_text) =
                                    import_regular_import_edit(&ast, module_handle);
                                let range = TextRange::at(position, TextSize::new(0));
                                let title = format!("Insert import: `{}`", insert_text.trim());
                                let is_private_import = module_name
                                    .components()
                                    .last()
                                    .is_some_and(|component| component.as_str().starts_with('_'));
                                code_actions.push((
                                    title,
                                    module_info.dupe(),
                                    range,
                                    insert_text,
                                    false,
                                    is_private_import,
                                ));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Sort code actions: non-private first, then non-deprecated, then alphabetically
        code_actions.sort_by(
            |(title1, _, _, _, is_deprecated1, is_private1),
             (title2, _, _, _, is_deprecated2, is_private2)| {
                match (is_private1, is_private2) {
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    _ => match (is_deprecated1, is_deprecated2) {
                        (true, false) => Ordering::Greater,
                        (false, true) => Ordering::Less,
                        _ => title1.cmp(title2),
                    },
                }
            },
        );

        // Keep only the first suggestion for each unique import text (after sorting,
        // this will be the public/non-deprecated version)
        code_actions.dedup_by(|a, b| a.3 == b.3);

        // Drop the deprecated flag and return
        Some(
            code_actions
                .into_iter()
                .map(|(title, module, range, insert_text, _, _)| {
                    (title, module, range, insert_text)
                })
                .collect(),
        )
    }

    pub fn extract_function_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::extract_function::extract_function_code_actions(self, handle, selection)
    }

    pub fn extract_variable_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::extract_variable::extract_variable_code_actions(self, handle, selection)
    }

    /// Determines whether a module is a third-party package.
    ///
    /// Checks if the module's path is located within any of the configured
    /// site-packages directories (e.g., `site-packages/`, `dist-packages/`).
    fn is_third_party_module(&self, module: &Module, handle: &Handle) -> bool {
        let config = self.get_config(handle);
        let module_path = module.path();

        if let Some(config) = config {
            for site_package_path in config.site_package_path() {
                if module_path.as_path().starts_with(site_package_path) {
                    return true;
                }
            }
        }

        false
    }

    fn is_source_file(&self, module: &Module, handle: &Handle) -> bool {
        let config = self.get_config(handle);
        let module_path = module.path();

        if let Some(config) = config {
            // Editable packages are installed in site-packages (via .pth files) but their
            // source code resides in the search_path location. A module is from an editable
            // package if its path starts with an explicitly configured search_path entry.
            // We only check search_path_from_file (user-configured paths) and not import_root
            // (auto-inferred paths), because import_root defaults to the project root and
            // would incorrectly match all modules.
            for search_path in &config.search_path_from_file {
                if module_path.as_path().starts_with(search_path) {
                    return true;
                }
            }
        }

        false
    }

    pub fn prepare_rename(&self, handle: &Handle, position: TextSize) -> Option<TextRange> {
        let identifier_context = self.identifier_at(handle, position);

        let definitions = self.find_definition(handle, position, FindPreference::default());

        for FindDefinitionItemWithDocstring { module, .. } in definitions {
            // Block rename only if it's third-party AND not an editable install/source file.

            if self.is_third_party_module(&module, handle) && !self.is_source_file(&module, handle)
            {
                return None;
            }
        }

        Some(identifier_context?.identifier.range)
    }

    pub fn find_local_references(&self, handle: &Handle, position: TextSize) -> Vec<TextRange> {
        self.find_definition(
            handle,
            position,
            FindPreference {
                import_behavior: ImportBehavior::StopAtRenamedImports,
                ..Default::default()
            },
        )
        .into_iter()
        .filter_map(
            |FindDefinitionItemWithDocstring {
                 metadata,
                 definition_range,
                 module,
                 docstring_range: _,
                 ..
             }| {
                self.local_references_from_definition(handle, metadata, definition_range, &module)
            },
        )
        .concat()
    }

    fn local_references_from_external_definition(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        module: &Module,
    ) -> Option<Vec<TextRange>> {
        let index = self.get_solutions(handle)?.get_index()?;
        let index = index.lock();
        let mut references = Vec::new();
        for ((imported_module_name, imported_name), ranges) in index
            .externally_defined_variable_references
            .iter()
            .chain(&index.renamed_imports)
        {
            if let Some((imported_handle, export)) = self.resolve_named_import(
                handle,
                *imported_module_name,
                imported_name.clone(),
                FindPreference::default(),
            ) && imported_handle.path().as_path() == module.path().as_path()
                && export.location == definition_range
            {
                references.extend(ranges.iter().copied());
            }
        }
        for (attribute_module_path, def_and_ref_ranges) in
            &index.externally_defined_attribute_references
        {
            if attribute_module_path == module.path() {
                for (def_range, ref_range) in def_and_ref_ranges {
                    if def_range == &definition_range {
                        references.push(*ref_range);
                    }
                }
            }
        }
        Some(references)
    }

    fn local_references_from_local_definition(
        &self,
        handle: &Handle,
        definition_metadata: DefinitionMetadata,
        definition_name: &Name,
        definition_range: TextRange,
    ) -> Option<Vec<TextRange>> {
        let mut references = match definition_metadata {
            DefinitionMetadata::Attribute => self.local_attribute_references_from_local_definition(
                handle,
                definition_range,
                definition_name,
            ),
            DefinitionMetadata::Module => Vec::new(),
            DefinitionMetadata::Variable(symbol_kind) => self
                .local_variable_references_from_local_definition(
                    handle,
                    definition_range,
                    definition_name,
                    symbol_kind,
                )
                .unwrap_or_default(),
            DefinitionMetadata::VariableOrAttribute(symbol_kind) => [
                self.local_attribute_references_from_local_definition(
                    handle,
                    definition_range,
                    definition_name,
                ),
                self.local_variable_references_from_local_definition(
                    handle,
                    definition_range,
                    definition_name,
                    symbol_kind,
                )
                .unwrap_or_default(),
            ]
            .concat(),
        };
        references.push(definition_range);
        Some(references)
    }

    fn local_references_from_definition(
        &self,
        handle: &Handle,
        definition_metadata: DefinitionMetadata,
        definition_range: TextRange,
        module: &Module,
    ) -> Option<Vec<TextRange>> {
        let mut references = if handle.path() != module.path() {
            self.local_references_from_external_definition(handle, definition_range, module)?
        } else {
            let definition_name = Name::new(module.code_at(definition_range));
            self.local_references_from_local_definition(
                handle,
                definition_metadata,
                &definition_name,
                definition_range,
            )?
        };
        references.sort_by_key(|range| range.start());
        references.dedup();
        Some(references)
    }

    fn local_attribute_references_from_local_definition(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        expected_name: &Name,
    ) -> Vec<TextRange> {
        // We first find all the attributes of the form `<expr>.<expected_name>`.
        // These are candidates for the references of `definition`.
        let relevant_attributes = if let Some(mod_module) = self.get_ast(handle) {
            fn f(x: &Expr, expected_name: &Name, res: &mut Vec<ExprAttribute>) {
                if let Expr::Attribute(x) = x
                    && &x.attr.id == expected_name
                {
                    res.push(x.clone());
                }
                x.recurse(&mut |x| f(x, expected_name, res));
            }
            let mut res = Vec::new();
            mod_module.visit(&mut |x| f(x, expected_name, &mut res));
            res
        } else {
            Vec::new()
        };
        // For each attribute we found above, we will test whether it actually will jump to the
        // given `definition`.
        self.ad_hoc_solve(handle, |solver| {
            let mut references = Vec::new();
            for attribute in relevant_attributes {
                if let Some(answers) = self.get_answers(handle)
                    && let Some(base_type) = answers.get_type_trace(attribute.value.range())
                {
                    for AttrInfo {
                        name,
                        ty: _,
                        is_deprecated: _,
                        definition: attribute_definition,
                        docstring_range,
                        is_reexport: _,
                    } in solver.completions(base_type, Some(expected_name), false)
                    {
                        if let Some((TextRangeWithModule { module, range }, _)) =
                            attribute_definition.and_then(|definition| {
                                self.resolve_attribute_definition(
                                    handle,
                                    &name,
                                    definition,
                                    docstring_range,
                                    FindPreference::default(),
                                )
                            })
                            && module.path() == module.path()
                            && range == definition_range
                        {
                            references.push(attribute.attr.range());
                        }
                    }
                }
            }
            references
        })
        .unwrap_or_default()
    }

    /// Collects all keyword arguments with a specific name within a module.
    ///
    /// This function traverses the AST of the given module and identifies all function calls
    /// that use a keyword argument matching the expected name. For each match, it captures
    /// both the keyword argument identifier and information about the function being called.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle to the module to search within
    /// * `expected_name` - The name of the keyword argument to search for
    ///
    /// # Returns
    ///
    /// A vector of tuples, where each tuple contains:
    /// - `Identifier`: The keyword argument identifier that matched the expected name
    /// - `CalleeKind`: Information about the function being called with this keyword argument
    ///
    /// Returns an empty vector if the AST cannot be retrieved.
    ///
    /// # Example
    ///
    /// For a module containing calls like `foo(bar=1)` and `baz(bar=2)`, searching for
    /// the name `bar` would return both keyword argument identifiers along with their
    /// respective callee information (`foo` and `baz`).
    pub(self) fn collect_local_keyword_arguments_by_name(
        &self,
        handle: &Handle,
        expected_name: &Name,
    ) -> Vec<(Identifier, CalleeKind)> {
        let Some(mod_module) = self.get_ast(handle) else {
            return Vec::new();
        };

        fn collect_kwargs(
            x: &Expr,
            expected_name: &Name,
            results: &mut Vec<(Identifier, CalleeKind)>,
        ) {
            if let Expr::Call(call) = x {
                visit_keyword_arguments_until_match(call, |_j, kw| {
                    if let Some(arg_identifier) = &kw.arg
                        && arg_identifier.id() == expected_name
                    {
                        let callee_kind = callee_kind_from_call(call);
                        results.push((arg_identifier.clone(), callee_kind));
                    }
                    false
                });
            }
            x.recurse(&mut |x| collect_kwargs(x, expected_name, results));
        }

        let mut results = Vec::new();
        mod_module.visit(&mut |x| collect_kwargs(x, expected_name, &mut results));
        results
    }

    /// Finds all local keyword argument references that correspond to a specific parameter definition.
    ///
    /// Given a parameter's definition range and name, this function identifies all keyword arguments
    /// in function calls within the same module that refer to this parameter. This is useful for
    /// LSP features like "Find All References" for function parameters.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle to the module containing the parameter definition
    /// * `definition_range` - The text range where the parameter is defined
    /// * `expected_name` - The name of the parameter to search for
    ///
    /// # Returns
    ///
    /// Returns `Some(Vec<TextRange>)` containing the text ranges of all keyword argument usages
    /// that reference this parameter definition, or `None` if the AST cannot be retrieved.
    pub(crate) fn local_keyword_argument_references_from_parameter_definition(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        expected_name: &Name,
    ) -> Option<Vec<TextRange>> {
        let ast = self.get_ast(handle)?;
        let keyword_args = self.collect_local_keyword_arguments_by_name(handle, expected_name);
        let mut references = Vec::new();

        let definition_module = self.get_module_info(handle)?;

        for (kw_identifier, callee_kind) in keyword_args {
            let callee_locations =
                self.get_callee_location(handle, &callee_kind, FindPreference::default());

            for TextRangeWithModule {
                module,
                range: callee_def_range,
            } in callee_locations
            {
                if module.path() == definition_module.path() {
                    // Refine to get the actual parameter location
                    if let Some(param_range) = self.refine_param_location_for_callee(
                        ast.as_ref(),
                        callee_def_range,
                        &kw_identifier,
                    ) {
                        // If the parameter location matches our definition, this is a valid reference
                        if param_range == definition_range {
                            references.push(kw_identifier.range);
                        }
                    }
                }
            }
        }

        Some(references)
    }

    fn local_variable_references_from_local_definition(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        expected_name: &Name,
        symbol_kind: Option<SymbolKind>,
    ) -> Option<Vec<TextRange>> {
        let mut references = Vec::new();
        if let Some(mod_module) = self.get_ast(handle) {
            let is_valid_use = |x: &ExprName| {
                if x.id() == expected_name
                    && let Some((def_handle, Export { location, .. })) = self.find_export_for_key(
                        handle,
                        &Key::BoundName(ShortIdentifier::expr_name(x)),
                        FindPreference {
                            import_behavior: ImportBehavior::StopAtRenamedImports,
                            prefer_pyi: false,
                        },
                    )
                    && def_handle.path() == handle.path()
                    && location == definition_range
                {
                    true
                } else {
                    false
                }
            };
            fn f(x: &Expr, is_valid_use: &impl Fn(&ExprName) -> bool, res: &mut Vec<TextRange>) {
                if let Expr::Name(x) = x
                    && is_valid_use(x)
                {
                    res.push(x.range());
                }
                x.recurse(&mut |x| f(x, is_valid_use, res));
            }
            mod_module.visit(&mut |x| f(x, &is_valid_use, &mut references));
        }

        if let Some(kind) = symbol_kind
            && (kind == SymbolKind::Parameter || kind == SymbolKind::Variable)
        {
            let kwarg_references = self
                .local_keyword_argument_references_from_parameter_definition(
                    handle,
                    definition_range,
                    expected_name,
                );

            if let Some(refs) = kwarg_references {
                references.extend(refs);
            }
        }
        Some(references)
    }

    fn add_kwargs_completions(
        &self,
        handle: &Handle,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Some((callables, overload_idx, _, _)) =
            self.get_callables_from_call(handle, position)
            && let Some(callable) = callables.get(overload_idx).cloned()
            && let Some(params) = Self::normalize_singleton_function_type_into_params(callable)
        {
            for param in params {
                match param {
                    Param::Pos(name, ty, _)
                    | Param::PosOnly(Some(name), ty, _)
                    | Param::KwOnly(name, ty, _)
                    | Param::VarArg(Some(name), ty) => {
                        if name.as_str() != "self" {
                            completions.push(CompletionItem {
                                label: format!("{}=", name.as_str()),
                                detail: Some(ty.to_string()),
                                kind: Some(CompletionItemKind::VARIABLE),
                                ..Default::default()
                            });
                        }
                    }
                    Param::VarArg(None, _) | Param::Kwargs(_, _) | Param::PosOnly(None, _, _) => {}
                }
            }
        }
    }

    fn add_magic_method_completions(
        &self,
        identifier: &Identifier,
        completions: &mut Vec<CompletionItem>,
    ) {
        let typed = identifier.as_str();
        if !typed.is_empty() && !typed.starts_with("__") {
            return;
        }
        for name in dunder::MAGIC_METHOD_NAMES {
            if name.starts_with(typed) {
                completions.push(CompletionItem {
                    label: (*name).to_owned(),
                    kind: Some(CompletionItemKind::METHOD),
                    ..Default::default()
                });
            }
        }
    }

    fn add_builtins_autoimport_completions(
        &self,
        handle: &Handle,
        identifier: Option<&Identifier>,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Some(builtin_handle) = self
            .import_handle(handle, ModuleName::builtins(), None)
            .finding()
        {
            let builtin_exports = self.get_exports(&builtin_handle);
            for (name, location) in builtin_exports.iter() {
                if let Some(identifier) = identifier
                    && SkimMatcherV2::default()
                        .smart_case()
                        .fuzzy_match(name.as_str(), identifier.as_str())
                        .is_none()
                {
                    continue;
                }
                let kind = match location {
                    ExportLocation::OtherModule(..) => continue,
                    ExportLocation::ThisModule(export) => export
                        .symbol_kind
                        .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                            Some(k.to_lsp_completion_item_kind())
                        }),
                };
                completions.push(CompletionItem {
                    label: name.as_str().to_owned(),
                    detail: None,
                    kind,
                    data: Some(serde_json::json!("builtin")),
                    ..Default::default()
                });
            }
        }
    }

    fn add_autoimport_completions(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        completions: &mut Vec<CompletionItem>,
        import_format: ImportFormat,
        supports_completion_item_details: bool,
    ) {
        // Auto-import can be slow. Let's only return results if there are no local
        // results for now. TODO: re-enable it once we no longer have perf issues.
        // We should not try to generate autoimport when the user has typed very few
        // characters. It's unhelpful to narrow down suggestions.
        if identifier.as_str().len() >= MIN_CHARACTERS_TYPED_AUTOIMPORT
            && let Some(ast) = self.get_ast(handle)
            && let Some(module_info) = self.get_module_info(handle)
        {
            for (handle_to_import_from, name, export) in
                self.search_exports_fuzzy(identifier.as_str())
            {
                // Using handle itself doesn't always work because handles can be made separately and have different hashes
                if handle_to_import_from.module() == handle.module()
                    || handle_to_import_from.module() == ModuleName::builtins()
                {
                    continue;
                }
                let module_description = handle_to_import_from.module().as_str().to_owned();
                let (insert_text, additional_text_edits, imported_module) = {
                    let (position, insert_text, module_name) = insert_import_edit(
                        &ast,
                        self.config_finder(),
                        handle.dupe(),
                        handle_to_import_from,
                        &name,
                        import_format,
                    );
                    let import_text_edit = TextEdit {
                        range: module_info.to_lsp_range(TextRange::at(position, TextSize::new(0))),
                        new_text: insert_text.clone(),
                    };
                    (insert_text, Some(vec![import_text_edit]), module_name)
                };
                let auto_import_label_detail = format!(" (import {imported_module})");

                completions.push(CompletionItem {
                    label: name,
                    detail: Some(insert_text),
                    kind: export
                        .symbol_kind
                        .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                            Some(k.to_lsp_completion_item_kind())
                        }),
                    additional_text_edits,
                    label_details: supports_completion_item_details.then_some(
                        CompletionItemLabelDetails {
                            detail: Some(auto_import_label_detail),
                            description: Some(module_description),
                        },
                    ),
                    tags: if export.deprecation.is_some() {
                        Some(vec![CompletionItemTag::DEPRECATED])
                    } else {
                        None
                    },
                    ..Default::default()
                });
            }

            for module_name in self.search_modules_fuzzy(identifier.as_str()) {
                if module_name == handle.module() {
                    continue;
                }
                let module_name_str = module_name.as_str().to_owned();
                if let Some(module_handle) = self.import_handle(handle, module_name, None).finding()
                {
                    let (insert_text, additional_text_edits) = {
                        let (position, insert_text) =
                            import_regular_import_edit(&ast, module_handle);
                        let import_text_edit = TextEdit {
                            range: module_info
                                .to_lsp_range(TextRange::at(position, TextSize::new(0))),
                            new_text: insert_text.clone(),
                        };
                        (insert_text, Some(vec![import_text_edit]))
                    };
                    let auto_import_label_detail = format!(" (import {module_name_str})");

                    completions.push(CompletionItem {
                        label: module_name_str.clone(),
                        detail: Some(insert_text),
                        kind: Some(CompletionItemKind::MODULE),
                        additional_text_edits,
                        label_details: supports_completion_item_details.then_some(
                            CompletionItemLabelDetails {
                                detail: Some(auto_import_label_detail),
                                description: Some(module_name_str),
                            },
                        ),
                        ..Default::default()
                    });
                }
            }
        }
    }

    fn get_documentation_from_export(
        &self,
        export_info: Option<(Handle, Export)>,
    ) -> Option<lsp_types::Documentation> {
        let (definition_handle, export) = export_info?;
        let docstring_range = export.docstring_range?;
        let def_module = self.get_module_info(&definition_handle)?;
        let docstring = Docstring(docstring_range, def_module.clone()).resolve();
        let documentation = lsp_types::Documentation::MarkupContent(lsp_types::MarkupContent {
            kind: lsp_types::MarkupKind::Markdown,
            value: docstring,
        });
        Some(documentation)
    }

    /// Adds completions for local variables and returns true if we have added any
    /// If an identifier is present, filter matches
    fn add_local_variable_completions(
        &self,
        handle: &Handle,
        identifier: Option<&Identifier>,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) -> bool {
        let mut has_added_any = false;
        if let Some(bindings) = self.get_bindings(handle)
            && let Some(module_info) = self.get_module_info(handle)
        {
            for idx in bindings.available_definitions(position) {
                let key = bindings.idx_to_key(idx);
                let label = match key {
                    Key::Definition(id) => module_info.code_at(id.range()),
                    Key::Anywhere(id, _) => id,
                    _ => continue,
                };
                if let Some(identifier) = identifier
                    && SkimMatcherV2::default()
                        .fuzzy_match(label, identifier.as_str())
                        .is_none()
                {
                    continue;
                }
                let binding = bindings.get(idx);
                let ty = self.get_type(handle, key);
                let export_info = self.key_to_export(handle, key, FindPreference::default());

                let kind = if let Some((_, ref export)) = export_info {
                    export
                        .symbol_kind
                        .map_or(CompletionItemKind::VARIABLE, |k| {
                            k.to_lsp_completion_item_kind()
                        })
                } else {
                    binding
                        .symbol_kind()
                        .map_or(CompletionItemKind::VARIABLE, |k| {
                            k.to_lsp_completion_item_kind()
                        })
                };

                let is_deprecated = ty.as_ref().is_some_and(|t| {
                    if let Type::ClassDef(cls) = t {
                        self.ad_hoc_solve(handle, |solver| {
                            solver.get_metadata_for_class(cls).deprecation().is_some()
                        })
                        .unwrap_or(false)
                    } else {
                        t.function_deprecation().is_some()
                    }
                });
                let detail = ty.map(|t| t.to_string());
                let documentation = self.get_documentation_from_export(export_info);

                has_added_any = true;
                completions.push(CompletionItem {
                    label: label.to_owned(),
                    detail,
                    kind: Some(kind),
                    documentation,
                    tags: if is_deprecated {
                        Some(vec![CompletionItemTag::DEPRECATED])
                    } else {
                        None
                    },
                    ..Default::default()
                })
            }
        }
        has_added_any
    }

    fn add_keyword_completions(&self, handle: &Handle, completions: &mut Vec<CompletionItem>) {
        get_keywords(handle.sys_info().version())
            .iter()
            .for_each(|name| {
                completions.push(CompletionItem {
                    label: (*name).to_owned(),
                    kind: Some(CompletionItemKind::KEYWORD),
                    ..Default::default()
                })
            });
    }

    fn get_docstring_for_attribute(
        &self,
        handle: &Handle,
        attr_info: &AttrInfo,
    ) -> Option<lsp_types::Documentation> {
        let definition = attr_info.definition.as_ref()?.clone();
        let attribute_definition = self.resolve_attribute_definition(
            handle,
            &attr_info.name,
            definition,
            attr_info.docstring_range,
            FindPreference::default(),
        );

        let (definition, Some(docstring_range)) = attribute_definition? else {
            return None;
        };
        let docstring = Docstring(docstring_range, definition.module);

        Some(lsp_types::Documentation::MarkupContent(
            lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: docstring.resolve().trim().to_owned(),
            },
        ))
    }

    fn add_literal_completions(
        &self,
        handle: &Handle,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
        in_string_literal: bool,
    ) {
        if let Some((callables, chosen_overload_index, active_argument, _)) =
            self.get_callables_from_call(handle, position)
            && let Some(callable) = callables.get(chosen_overload_index)
            && let Some(params) =
                Self::normalize_singleton_function_type_into_params(callable.clone())
            && let Some(arg_index) = Self::active_parameter_index(&params, &active_argument)
            && let Some(param) = params.get(arg_index)
        {
            Self::add_literal_completions_from_type(
                param.as_type(),
                completions,
                in_string_literal,
            );
        }
    }

    fn add_literal_completions_from_type(
        param_type: &Type,
        completions: &mut Vec<CompletionItem>,
        in_string_literal: bool,
    ) {
        match param_type {
            Type::Literal(lit) => {
                // TODO: Pass the flag correctly for whether literal string is single quoted or double quoted
                let label = lit.to_string_escaped(true);
                let insert_text = if in_string_literal {
                    if let Lit::Str(s) = lit {
                        s.to_string()
                    } else {
                        label.clone()
                    }
                } else {
                    label.clone()
                };
                completions.push(CompletionItem {
                    label,
                    kind: Some(CompletionItemKind::VALUE),
                    detail: Some(format!("{param_type}")),
                    insert_text: Some(insert_text),
                    ..Default::default()
                });
            }
            Type::Union(box Union { members, .. }) => {
                for member in members {
                    Self::add_literal_completions_from_type(member, completions, in_string_literal);
                }
            }
            _ => {}
        }
    }

    fn subscript_string_literal_at(
        module: &ModModule,
        position: TextSize,
    ) -> Option<(ExprSubscript, ExprStringLiteral)> {
        let nodes = Ast::locate_node(module, position);
        let mut best: Option<(u8, TextSize, ExprSubscript, ExprStringLiteral)> = None;
        for node in nodes {
            if let AnyNodeRef::ExprSubscript(sub) = node
                && let Expr::StringLiteral(lit) = sub.slice.as_ref()
            {
                let (priority, dist) = Self::string_literal_priority(position, lit.range());
                let should_update = match &best {
                    Some((best_prio, best_dist, _, _)) => {
                        priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                    }
                    None => true,
                };
                if should_update {
                    best = Some((priority, dist, sub.clone(), lit.clone()));
                    if priority == 0 && dist == TextSize::from(0) {
                        break;
                    }
                }
            }
        }
        best.map(|(_, _, sub, lit)| (sub, lit))
    }

    fn string_literal_priority(position: TextSize, range: TextRange) -> (u8, TextSize) {
        if range.contains(position) {
            (0, TextSize::from(0))
        } else if position < range.start() {
            (1, range.start() - position)
        } else {
            (2, position - range.end())
        }
    }

    fn expression_facets(expr: &Expr) -> Option<(Identifier, Vec<FacetKind>)> {
        let mut facets = Vec::new();
        let mut current = expr;
        loop {
            match current {
                Expr::Subscript(sub) => {
                    match sub.slice.as_ref() {
                        Expr::NumberLiteral(ExprNumberLiteral {
                            value: Number::Int(idx),
                            ..
                        }) if idx.as_usize().is_some() => {
                            facets.push(FacetKind::Index(idx.as_usize().unwrap()))
                        }
                        Expr::StringLiteral(lit) => {
                            facets.push(FacetKind::Key(lit.value.to_string()))
                        }
                        _ => return None,
                    }
                    current = sub.value.as_ref();
                }
                Expr::Attribute(attr) => {
                    facets.push(FacetKind::Attribute(attr.attr.id.clone()));
                    current = attr.value.as_ref();
                }
                Expr::Name(name) => {
                    facets.reverse();
                    return Some((Ast::expr_name_identifier(name.clone()), facets));
                }
                _ => return None,
            }
        }
    }

    fn collect_typed_dict_keys(
        &self,
        handle: &Handle,
        base_type: Type,
    ) -> Option<BTreeMap<String, Type>> {
        self.ad_hoc_solve(handle, |solver| {
            let mut map = BTreeMap::new();
            let mut stack = vec![base_type];
            while let Some(ty) = stack.pop() {
                match ty {
                    Type::TypedDict(td) | Type::PartialTypedDict(td) => {
                        for (name, field) in solver.type_order().typed_dict_fields(&td) {
                            map.entry(name.to_string())
                                .or_insert_with(|| field.ty.clone());
                        }
                    }
                    Type::Union(box Union { members, .. }) => {
                        stack.extend(members.into_iter());
                    }
                    _ => {}
                }
            }
            map
        })
    }

    fn add_dict_key_completions(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) {
        let Some((subscript, string_lit)) = Self::subscript_string_literal_at(module, position)
        else {
            return;
        };
        let literal_range = string_lit.range();
        // Allow the cursor to sit a few characters before the literal (e.g. between nested
        // subscripts) so completion requests fired just before the quotes still succeed.
        let allowance = TextSize::from(4);
        let lower_bound = literal_range
            .start()
            .checked_sub(allowance)
            .unwrap_or_else(|| TextSize::new(0));
        if position < lower_bound || position > literal_range.end() {
            return;
        }
        let base_expr = subscript.value.as_ref();
        let mut suggestions: BTreeMap<String, Option<Type>> = BTreeMap::new();

        if let Some(bindings) = self.get_bindings(handle) {
            let base_info = if let Some((identifier, facets)) = Self::expression_facets(base_expr) {
                Some((identifier, facets))
            } else if let Expr::Name(name) = base_expr {
                Some((Ast::expr_name_identifier(name.clone()), Vec::new()))
            } else {
                None
            };

            if let Some((identifier, facets)) = base_info {
                let short_id = ShortIdentifier::new(&identifier);
                let idx_opt = {
                    let bound_key = Key::BoundName(short_id);
                    if bindings.is_valid_key(&bound_key) {
                        Some(bindings.key_to_idx(&bound_key))
                    } else {
                        let def_key = Key::Definition(short_id);
                        if bindings.is_valid_key(&def_key) {
                            Some(bindings.key_to_idx(&def_key))
                        } else {
                            None
                        }
                    }
                };

                if let Some(idx) = idx_opt {
                    let facets_clone = facets.clone();
                    if let Some(keys) = self.ad_hoc_solve(handle, |solver| {
                        let info = solver.get_idx(idx);
                        info.key_facets_at(&facets_clone)
                    }) {
                        for (key, ty_opt) in keys {
                            suggestions.entry(key).or_insert(ty_opt);
                        }
                    }
                }
            }
        }

        if let Some(base_type) = self.get_type_trace(handle, base_expr.range())
            && let Some(typed_keys) = self.collect_typed_dict_keys(handle, base_type)
        {
            for (key, ty) in typed_keys {
                let entry = suggestions.entry(key).or_insert(None);
                if entry.is_none() {
                    *entry = Some(ty);
                }
            }
        }

        if suggestions.is_empty() {
            return;
        }

        for (label, ty_opt) in suggestions {
            let detail = ty_opt.as_ref().map(|ty| ty.to_string());
            completions.push(CompletionItem {
                label,
                detail,
                kind: Some(CompletionItemKind::FIELD),
                ..Default::default()
            });
        }
    }

    // Kept for backwards compatibility - used by external callers (lsp/server.rs, playground.rs)
    // who don't need the is_incomplete flag
    pub fn completion(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        supports_completion_item_details: bool,
    ) -> Vec<CompletionItem> {
        self.completion_with_incomplete(
            handle,
            position,
            import_format,
            supports_completion_item_details,
        )
        .0
    }

    // Returns the completions, and true if they are incomplete so client will keep asking for more completions
    pub fn completion_with_incomplete(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        supports_completion_item_details: bool,
    ) -> (Vec<CompletionItem>, bool) {
        // Check if position is in a disabled range (comments)
        if let Some(module) = self.get_module_info(handle) {
            let disabled_ranges = Self::comment_ranges_for_module(&module);
            if disabled_ranges.iter().any(|range| range.contains(position)) {
                return (Vec::new(), false);
            }
        }

        let (mut results, is_incomplete) = self.completion_sorted_opt_with_incomplete(
            handle,
            position,
            import_format,
            supports_completion_item_details,
        );
        results.sort_by(|item1, item2| {
            item1
                .sort_text
                .cmp(&item2.sort_text)
                .then_with(|| item1.label.cmp(&item2.label))
                .then_with(|| item1.detail.cmp(&item2.detail))
        });
        results.dedup_by(|item1, item2| item1.label == item2.label && item1.detail == item2.detail);
        (results, is_incomplete)
    }

    fn comment_ranges_for_module(module: &ModuleInfo) -> Vec<TextRange> {
        let mut ranges = Vec::new();
        let source = module.lined_buffer().contents();
        let mut offset = TextSize::from(0);

        for line in source.lines() {
            if let Some(comment_pos) = pyrefly_python::ignore::find_comment_start_in_line(line) {
                let comment_start = offset + TextSize::from(comment_pos as u32);
                let comment_end = offset + TextSize::from(line.len() as u32);
                ranges.push(TextRange::new(comment_start, comment_end));
            }
            offset += TextSize::from((line.len() + 1) as u32);
        }

        ranges
    }

    fn completion_sorted_opt_with_incomplete(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        supports_completion_item_details: bool,
    ) -> (Vec<CompletionItem>, bool) {
        let mut result = Vec::new();
        let mut is_incomplete = false;
        match self.identifier_at(handle, position) {
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ImportedName { module_name, .. },
            }) => {
                if let Some(handle) = self.import_handle(handle, module_name, None).finding() {
                    // Because of parser error recovery, `from x impo...` looks like `from x import impo...`
                    // If the user might be typing the `import` keyword, add that as an autocomplete option.
                    if "import".starts_with(identifier.as_str()) {
                        result.push(CompletionItem {
                            label: "import".to_owned(),
                            kind: Some(CompletionItemKind::KEYWORD),
                            ..Default::default()
                        })
                    }
                    let exports = self.get_exports(&handle);
                    for (name, export) in exports.iter() {
                        let is_deprecated = match export {
                            ExportLocation::ThisModule(export) => export.deprecation.is_some(),
                            ExportLocation::OtherModule(_, _) => false,
                        };
                        result.push(CompletionItem {
                            label: name.to_string(),
                            // todo(kylei): completion kind for exports
                            kind: Some(CompletionItemKind::VARIABLE),
                            tags: if is_deprecated {
                                Some(vec![CompletionItemTag::DEPRECATED])
                            } else {
                                None
                            },
                            ..Default::default()
                        })
                    }
                }
            }
            // TODO: Handle relative import (via ModuleName::new_maybe_relative)
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ImportedModule { .. },
            }) => self
                .import_prefixes(handle, ModuleName::from_name(identifier.id()))
                .iter()
                .for_each(|module_name| {
                    result.push(CompletionItem {
                        label: module_name
                            .components()
                            .last()
                            .unwrap_or(&Name::empty())
                            .to_string(),
                        detail: Some(module_name.to_string()),
                        kind: Some(CompletionItemKind::MODULE),
                        ..Default::default()
                    })
                }),
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Attribute { base_range, .. },
            }) => {
                if let Some(answers) = self.get_answers(handle)
                    && let Some(base_type) = answers.get_type_trace(base_range)
                {
                    self.ad_hoc_solve(handle, |solver| {
                        solver
                            .completions(base_type, None, true)
                            .iter()
                            .for_each(|x| {
                                let kind = match x.ty {
                                    Some(Type::BoundMethod(_)) => Some(CompletionItemKind::METHOD),
                                    Some(Type::Function(_)) => Some(CompletionItemKind::FUNCTION),
                                    _ => Some(CompletionItemKind::FIELD),
                                };
                                let ty = &x.ty;
                                let detail =
                                    ty.clone().map(|t| t.as_lsp_string(LspDisplayMode::Hover));
                                let documentation = self.get_docstring_for_attribute(handle, x);
                                result.push(CompletionItem {
                                    label: x.name.as_str().to_owned(),
                                    detail,
                                    kind,
                                    documentation,
                                    sort_text: if x.is_reexport {
                                        Some("1".to_owned())
                                    } else {
                                        None
                                    },
                                    tags: if x.is_deprecated {
                                        Some(vec![CompletionItemTag::DEPRECATED])
                                    } else {
                                        None
                                    },
                                    ..Default::default()
                                });
                            });
                    });
                }
            }
            Some(IdentifierWithContext {
                identifier,
                context,
            }) => {
                if matches!(context, IdentifierContext::MethodDef { .. }) {
                    self.add_magic_method_completions(&identifier, &mut result);
                }
                self.add_kwargs_completions(handle, position, &mut result);
                self.add_keyword_completions(handle, &mut result);
                let has_local_completions = self.add_local_variable_completions(
                    handle,
                    Some(&identifier),
                    position,
                    &mut result,
                );
                if !has_local_completions {
                    self.add_autoimport_completions(
                        handle,
                        &identifier,
                        &mut result,
                        import_format,
                        supports_completion_item_details,
                    );
                }
                // If autoimport completions were skipped due to character threshold,
                // mark the results as incomplete so clients keep asking for completions.
                // This ensures autoimport completions will be checked once the threshold is reached,
                // even if local completions are currently available.
                if identifier.as_str().len() < MIN_CHARACTERS_TYPED_AUTOIMPORT {
                    is_incomplete = true;
                }
                self.add_builtins_autoimport_completions(handle, Some(&identifier), &mut result);
            }
            None => {
                // todo(kylei): optimization, avoid duplicate ast walks
                if let Some(mod_module) = self.get_ast(handle) {
                    let nodes = Ast::locate_node(&mod_module, position);
                    if nodes.is_empty() {
                        self.add_keyword_completions(handle, &mut result);
                        self.add_local_variable_completions(handle, None, position, &mut result);
                        self.add_builtins_autoimport_completions(handle, None, &mut result);
                    }
                    let in_string_literal = nodes
                        .iter()
                        .any(|node| matches!(node, AnyNodeRef::ExprStringLiteral(_)));
                    self.add_literal_completions(handle, position, &mut result, in_string_literal);
                    self.add_dict_key_completions(
                        handle,
                        mod_module.as_ref(),
                        position,
                        &mut result,
                    );
                    // in foo(x=<>, y=2<>), the first containing node is AnyNodeRef::Arguments(_)
                    // in foo(<>), the first containing node is AnyNodeRef::ExprCall
                    if let Some(first) = nodes.first()
                        && matches!(first, AnyNodeRef::ExprCall(_) | AnyNodeRef::Arguments(_))
                    {
                        self.add_kwargs_completions(handle, position, &mut result);
                    }
                }
            }
        }
        for item in &mut result {
            let sort_text = if item
                .tags
                .as_ref()
                .is_some_and(|tags| tags.contains(&CompletionItemTag::DEPRECATED))
            {
                "9"
            } else if item.additional_text_edits.is_some() {
                "4"
            } else if item.label.starts_with("__") {
                "3"
            } else if item.label.as_str().starts_with("_") {
                "2"
            } else if let Some(sort_text) = &item.sort_text {
                // 1 is reserved for re-exports
                sort_text.as_str()
            } else {
                "0"
            }
            .to_owned();
            item.sort_text = Some(sort_text);
        }
        (result, is_incomplete)
    }

    fn export_from_location(
        &self,
        handle: &Handle,
        export_name: &Name,
        location: &ExportLocation,
    ) -> Option<(Handle, Export)> {
        match location {
            ExportLocation::ThisModule(export) => Some((handle.dupe(), export.clone())),
            ExportLocation::OtherModule(module, original_name) => {
                let target_name = original_name.clone().unwrap_or_else(|| export_name.clone());
                self.resolve_named_import(handle, *module, target_name, FindPreference::default())
            }
        }
    }

    /// Used to avoid making use of reexports of private modules for some LSP
    /// uses like auto-import (where we want to import the public API).
    /// - Returns true if both modules should be shown in auto-import suggestions.
    /// - Handles stdlib patterns where a public module (`io`) re-exports from a
    ///   private implementation module (`_io`).
    fn should_include_reexport(original: &Handle, canonical: &Handle) -> bool {
        let canonical_components = canonical.module().components();
        let canonical_component = canonical_components
            .last()
            .map(|name| name.as_str())
            .unwrap_or("");
        let original_components = original.module().components();
        let original_component = original_components
            .last()
            .map(|name| name.as_str())
            .unwrap_or("");
        canonical_component.starts_with('_')
            && canonical_component.trim_start_matches('_') == original_component
    }

    pub fn search_exports_exact(&self, name: &str) -> Vec<(Handle, Export)> {
        self.search_exports(|handle, exports| {
            let name = Name::new(name);
            match exports.get(&name) {
                Some(location) => {
                    if let Some((canonical_handle, export)) =
                        self.export_from_location(handle, &name, location)
                    {
                        let mut results = vec![(canonical_handle.dupe(), export.clone())];
                        if canonical_handle != *handle
                            && Self::should_include_reexport(handle, &canonical_handle)
                        {
                            results.push((handle.dupe(), export));
                        }
                        results
                    } else {
                        Vec::new()
                    }
                }
                None => Vec::new(),
            }
        })
    }

    pub fn search_exports_fuzzy(&self, pattern: &str) -> Vec<(Handle, String, Export)> {
        let mut res = self.search_exports(|handle, exports| {
            let matcher = SkimMatcherV2::default().smart_case();
            let mut results = Vec::new();
            for (name, location) in exports.iter() {
                let name_str = name.as_str();
                if let Some(score) = matcher.fuzzy_match(name_str, pattern)
                    && let Some((canonical_handle, export)) =
                        self.export_from_location(handle, name, location)
                {
                    results.push((
                        score,
                        canonical_handle.dupe(),
                        name_str.to_owned(),
                        export.clone(),
                    ));
                    if canonical_handle != *handle
                        && Self::should_include_reexport(handle, &canonical_handle)
                    {
                        results.push((score, handle.dupe(), name_str.to_owned(), export));
                    }
                }
            }
            results
        });
        res.sort_by_key(|(score, _, _, _)| Reverse(*score));
        res.into_map(|(_, handle, name, export)| (handle, name, export))
    }
}

impl<'a> CancellableTransaction<'a> {
    /// Finds child class implementations of a method definition.
    /// Returns the ranges of child methods that reimplement the given parent method.
    fn find_child_implementations(
        &self,
        handle: &Handle,
        definition: &TextRangeWithModule,
    ) -> Vec<TextRange> {
        let mut child_implementations = Vec::new();

        if let Some(solutions) = self.as_ref().get_solutions(handle)
            && let Some(index) = solutions.get_index()
        {
            let index_lock = index.lock();
            // Search for child methods that have this definition as a parent
            for (child_range, parent_methods) in &index_lock.parent_methods_map {
                for (parent_module_path, parent_range) in parent_methods {
                    // Check if the parent method matches our definition
                    if parent_module_path == definition.module.path()
                        && *parent_range == definition.range
                    {
                        // This child method is a reimplementation of our definition
                        child_implementations.push(*child_range);
                    }
                }
            }
        }

        child_implementations
    }

    /// Computes the set of transitive reverse dependencies for a definition, handling
    /// in-memory files and their filesystem counterparts.
    /// Returns Err if the request is canceled in the middle of a run.
    fn compute_transitive_rdeps_for_definition(
        &mut self,
        sys_info: &SysInfo,
        definition: &TextRangeWithModule,
    ) -> Result<Vec<Handle>, Cancelled> {
        let mut transitive_rdeps = match definition.module.path().details() {
            ModulePathDetails::Memory(path_buf) => {
                let handle_of_filesystem_counterpart = Handle::new(
                    definition.module.name(),
                    ModulePath::filesystem((**path_buf).clone()),
                    sys_info.dupe(),
                );
                // In-memory files can never be found through import resolution (no rdeps),
                // so we must compute the transitive rdeps of its filesystem counterpart instead.
                let mut rdeps = self
                    .as_ref()
                    .get_transitive_rdeps(handle_of_filesystem_counterpart.dupe());
                // We still add itself to the rdeps set, so that we will still find local references
                // within the file.
                rdeps.insert(Handle::new(
                    definition.module.name(),
                    definition.module.path().dupe(),
                    sys_info.dupe(),
                ));
                rdeps
            }
            _ => {
                let definition_handle = Handle::new(
                    definition.module.name(),
                    definition.module.path().dupe(),
                    sys_info.dupe(),
                );
                let rdeps = self.as_ref().get_transitive_rdeps(definition_handle.dupe());
                // We still need to know everything about the definition file, because the index
                // only contains non-local references.
                self.run(&[definition_handle], Require::Everything)?;
                rdeps
            }
        };
        // Remove the filesystem counterpart from candidate list,
        // otherwise we will have results from both filesystem and in-memory version of the file.
        for fs_counterpart_of_in_memory_handles in transitive_rdeps
            .iter()
            .filter_map(|handle| match handle.path().details() {
                ModulePathDetails::Memory(path_buf) => Some(Handle::new(
                    handle.module(),
                    ModulePath::filesystem((**path_buf).clone()),
                    handle.sys_info().dupe(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
        {
            transitive_rdeps.remove(&fs_counterpart_of_in_memory_handles);
        }
        let candidate_handles = transitive_rdeps
            .into_iter()
            .sorted_by_key(|h| h.path().dupe())
            .collect::<Vec<_>>();

        Ok(candidate_handles)
    }

    /// Patches a definition location to handle in-memory files when searching from another module.
    /// For in-memory files, tries to find the corresponding filesystem location to enable
    /// reference finding across modules.
    fn patch_definition_for_handle(
        &self,
        handle: &Handle,
        definition: &TextRangeWithModule,
    ) -> TextRangeWithModule {
        match definition.module.path().details() {
            // Special-case for definition inside in-memory file
            // Calling methods with in-memory definitions naively
            // will find no references outside of the in-memory file because
            // file systems don't contain in-memory files.
            ModulePathDetails::Memory(path_buf)
                // Why do we exclude the case of finding references within the same in-memory file?
                // If we are finding references within the same in-memory file,
                // then there is no problem for us to use the in-memory definition location.
                if handle.path() != definition.module.path() =>
            {
                // Below, we try to patch the definition location to be at the same offset, but
                // making the path to be filesystem path instead. In this way, in the happy case
                // where the in-memory content is exactly the same as the filesystem content,
                // we can successfully find all the references. However, if the content diverges,
                // then we will miss definitions from other files.
                //
                // In general, other than checking the reverse dependency against the in-memory
                // content, there is not much we can do: the in-memory content can diverge from
                // the filesystem content in arbitrary ways.
                let TextRangeWithModule { module, range } = definition;
                let module = if let Some(info) = self.as_ref().get_module_info(&Handle::new(
                    module.name(),
                    ModulePath::filesystem((**path_buf).clone()),
                    handle.sys_info().dupe(),
                )) {
                    info
                } else {
                    module.dupe()
                };
                TextRangeWithModule {
                    module,
                    range: *range,
                }
            }
            _ => definition.clone(),
        }
    }

    /// Processes each transitive reverse dependency for a given definition location.
    ///
    /// This is a common pattern in workspace-wide
    /// references-related features
    fn process_rdeps_with_definition<T>(
        &mut self,
        sys_info: &SysInfo,
        definition: &TextRangeWithModule,
        mut process_fn: impl FnMut(&mut Self, &Handle, &TextRangeWithModule) -> Option<T>,
    ) -> Result<Vec<T>, Cancelled> {
        let candidate_handles =
            self.compute_transitive_rdeps_for_definition(sys_info, definition)?;

        let mut results = Vec::new();
        for handle in candidate_handles {
            // "Patched" means the definition's module path is adjusted for in-memory files
            // to use the filesystem path instead, enabling cross-module reference finding
            let patched_definition = self.patch_definition_for_handle(&handle, definition);
            if let Some(result) = process_fn(self, &handle, &patched_definition) {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Returns Err if the request is canceled in the middle of a run.
    pub fn find_global_references_from_definition(
        &mut self,
        sys_info: &SysInfo,
        definition_kind: DefinitionMetadata,
        definition: TextRangeWithModule,
    ) -> Result<Vec<(Module, Vec<TextRange>)>, Cancelled> {
        let results = self.process_rdeps_with_definition(
            sys_info,
            &definition,
            |transaction, handle, patched_definition| {
                let mut module_refs: Vec<(Module, Vec<TextRange>)> = Vec::new();

                // Find local references
                let references = transaction
                    .as_ref()
                    .local_references_from_definition(
                        handle,
                        definition_kind.clone(),
                        patched_definition.range,
                        &patched_definition.module,
                    )
                    .unwrap_or_default();
                if !references.is_empty()
                    && let Some(module_info) = transaction.as_ref().get_module_info(handle)
                {
                    module_refs.push((module_info, references));
                }

                // Search for child class reimplementations using the parent_methods_map
                let child_implementations =
                    transaction.find_child_implementations(handle, patched_definition);
                if !child_implementations.is_empty()
                    && let Some(module_info) = transaction.as_ref().get_module_info(handle)
                {
                    // Check if we already have this module in our results
                    if let Some((_, ranges)) = module_refs
                        .iter_mut()
                        .find(|(m, _)| m.path() == module_info.path())
                    {
                        ranges.extend(child_implementations);
                    } else {
                        module_refs.push((module_info, child_implementations));
                    }
                }

                if module_refs.is_empty() {
                    None
                } else {
                    Some(module_refs)
                }
            },
        )?;

        // Flatten nested results and merge by module
        let mut global_references: Vec<(Module, Vec<TextRange>)> = Vec::new();
        for module_refs in results {
            for (module, ranges) in module_refs {
                if let Some((_, existing_ranges)) = global_references
                    .iter_mut()
                    .find(|(m, _)| m.path() == module.path())
                {
                    existing_ranges.extend(ranges);
                } else {
                    global_references.push((module, ranges));
                }
            }
        }

        // Sort and deduplicate references in each module
        for (_, references) in &mut global_references {
            references.sort_by_key(|range| range.start());
            references.dedup();
        }

        Ok(global_references)
    }

    /// Finds all implementations (child class methods) of the definition at the given position.
    /// This searches through transitive reverse dependencies to find all child classes that
    /// implement the method.
    /// Returns Err if the request is canceled in the middle of a run.
    pub fn find_global_implementations_from_definition(
        &mut self,
        sys_info: &SysInfo,
        definition: TextRangeWithModule,
    ) -> Result<Vec<TextRangeWithModule>, Cancelled> {
        let results = self.process_rdeps_with_definition(
            sys_info,
            &definition,
            |transaction, handle, patched_definition| {
                // Search for child class reimplementations using the parent_methods_map
                let child_implementations =
                    transaction.find_child_implementations(handle, patched_definition);
                if !child_implementations.is_empty()
                    && let Some(module_info) = transaction.as_ref().get_module_info(handle)
                {
                    let implementations: Vec<TextRangeWithModule> = child_implementations
                        .into_iter()
                        .map(|range| TextRangeWithModule::new(module_info.dupe(), range))
                        .collect();
                    Some(implementations)
                } else {
                    None
                }
            },
        )?;

        // Flatten nested results
        let mut all_implementations: Vec<TextRangeWithModule> =
            results.into_iter().flatten().collect();

        // Sort and deduplicate implementations
        all_implementations.sort_by_key(|impl_| (impl_.module.path().dupe(), impl_.range.start()));
        all_implementations.dedup_by_key(|impl_| (impl_.module.path().dupe(), impl_.range.start()));

        Ok(all_implementations)
    }

    /// Finds all incoming calls(functions that call this function) of a function across the entire codebase.
    ///
    /// This searches transitive reverse dependencies to find all locations where
    /// the target function is called. For each call site, it identifies the
    /// containing function.
    ///
    /// Returns a vector of tuples containing:
    /// - Module where the call occurs
    /// - Vector of (call_site_range, containing_function_name, containing_function_range)
    ///
    /// Returns Err if the request is canceled during execution.
    pub fn find_global_incoming_calls_from_function_definition(
        &mut self,
        sys_info: &SysInfo,
        definition_kind: DefinitionMetadata,
        target_definition: &TextRangeWithModule,
    ) -> Result<Vec<(Module, Vec<(TextRange, String, TextRange)>)>, Cancelled> {
        // Use process_rdeps_with_definition to find references and filter to call sites in a single pass
        let results = self.process_rdeps_with_definition(
            sys_info,
            target_definition,
            |transaction, handle, patched_definition| {
                let module_info = transaction.as_ref().get_module_info(handle)?;
                let ast = transaction.as_ref().get_ast(handle)?;

                let references = transaction
                    .as_ref()
                    .local_references_from_definition(
                        handle,
                        definition_kind.clone(),
                        patched_definition.range,
                        &patched_definition.module,
                    )
                    .unwrap_or_default();

                if references.is_empty() {
                    return None;
                }

                let ref_set: std::collections::HashSet<TextRange> =
                    references.into_iter().collect();

                let mut callers_in_file = Vec::new();

                ast.visit(&mut |expr| {
                    if let Expr::Call(call) = expr
                        && ref_set
                            .iter()
                            .any(|ref_range| call.func.range().contains(ref_range.start()))
                        && let Some((containing_func_name, containing_func_range)) =
                            Self::find_containing_function_for_call(
                                handle,
                                &ast,
                                call.range().start(),
                            )
                    {
                        callers_in_file.push((
                            call.range(),
                            containing_func_name,
                            containing_func_range,
                        ));
                    }
                });

                if callers_in_file.is_empty() {
                    None
                } else {
                    Some((module_info, callers_in_file))
                }
            },
        )?;

        Ok(results)
    }

    /// Finds outgoing calls(functions called by) of a function, resolving across files.
    ///
    /// Given a function definition, this method finds all Call expressions within
    /// that function's body and resolves each call target to its definition,
    /// which may be in other files.
    ///
    /// Returns a vector of tuples containing:
    /// - Target module (where the callee is defined)
    /// - Vector of (call_site_range, callee_definition_range)
    ///
    /// Results are grouped by target module for consistency with find_global_callers_from_definition.
    ///
    /// Returns Err if the request is canceled during execution.
    pub fn find_global_outgoing_calls_from_function_definition(
        &mut self,
        handle: &Handle,
        position: TextSize,
    ) -> Result<Vec<(Module, Vec<(TextRange, TextRange)>)>, Cancelled> {
        // Resolve cursor position to function definition (handles both definitions and call sites)
        let definitions =
            self.as_ref()
                .find_definition(handle, position, FindPreference::default());
        let Some(target_def) = definitions.into_iter().next() else {
            return Ok(Vec::new());
        };

        // Get handle for module where target function is defined (may differ from original handle)
        let target_handle = Handle::new(
            target_def.module.name(),
            target_def.module.path().dupe(),
            handle.sys_info().dupe(),
        );

        let Some(target_ast) = self.as_ref().get_ast(&target_handle) else {
            return Ok(Vec::new());
        };

        let Some(func_def) = Self::find_function_at_position_in_ast(
            &target_ast,
            target_def.definition_range.start(),
        ) else {
            return Ok(Vec::new());
        };

        let mut callees_by_module: std::collections::HashMap<
            ModulePath,
            (Module, Vec<(TextRange, TextRange)>),
        > = std::collections::HashMap::new();

        for stmt in &func_def.body {
            stmt.visit(&mut |expr| {
                if let Expr::Call(call) = expr {
                    let call_pos = call
                        .func
                        .range()
                        .end()
                        .checked_sub(TextSize::from(1))
                        .unwrap_or(call.func.range().start());

                    let definitions = self.as_ref().find_definition(
                        &target_handle,
                        call_pos,
                        FindPreference::default(),
                    );

                    for def in definitions {
                        let module_path = def.module.path().dupe();
                        callees_by_module
                            .entry(module_path)
                            .or_insert_with(|| (def.module.clone(), Vec::new()))
                            .1
                            .push((call.range(), def.definition_range));
                    }
                }
            });
        }

        let mut result: Vec<(Module, Vec<(TextRange, TextRange)>)> =
            callees_by_module.into_values().collect();
        result.sort_by_key(|(module, _)| module.path().dupe());

        Ok(result)
    }

    /// Helper: Finds a function definition at a specific position in an AST.
    ///
    /// This is used by call hierarchy to identify the function being queried.
    /// Returns None if no function is found at the position.
    ///
    /// Uses `locate_node` for efficient single-pass traversal, consistent with
    /// other LSP position-based queries. Handles nested functions and methods.
    pub(crate) fn find_function_at_position_in_ast(
        ast: &ModModule,
        position: TextSize,
    ) -> Option<&StmtFunctionDef> {
        let covering_nodes = Ast::locate_node(ast, position);

        // Find the first StmtFunctionDef in the covering nodes
        // This returns the innermost function containing the position
        for node in covering_nodes.iter() {
            if let AnyNodeRef::StmtFunctionDef(func_def) = node {
                return Some(func_def);
            }
        }
        None
    }

    /// Helper: Creates call hierarchy information for the function containing a call site.
    ///
    /// Given a call site position, finds the enclosing function and returns
    /// information needed to represent it in the call hierarchy.
    fn find_containing_function_for_call(
        handle: &Handle,
        ast: &ModModule,
        position: TextSize,
    ) -> Option<(String, TextRange)> {
        let covering_nodes = Ast::locate_node(ast, position);

        // Look through the node chain for the containing function
        for (i, node) in covering_nodes.iter().enumerate() {
            if let AnyNodeRef::StmtFunctionDef(func_def) = node {
                // Check if this is a method (next node is a ClassDef)
                if let Some(AnyNodeRef::StmtClassDef(class_def)) = covering_nodes.get(i + 1) {
                    let name = format!(
                        "{}.{}.{}",
                        handle.module(),
                        class_def.name.id,
                        func_def.name.id
                    );
                    return Some((name, func_def.name.range()));
                } else {
                    // Top-level function
                    let name = format!("{}.{}", handle.module(), func_def.name.id);
                    return Some((name, func_def.name.range()));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use ruff_python_ast::name::Name;

    use super::CancellableTransaction;
    use super::Transaction;
    use crate::types::callable::Param;
    use crate::types::callable::Required;
    use crate::types::types::AnyStyle;
    use crate::types::types::Type;

    fn any_type() -> Type {
        Type::Any(AnyStyle::Explicit)
    }

    #[test]
    fn param_name_for_positional_argument_marks_vararg_repeats() {
        let params = vec![
            Param::Pos(Name::new_static("x"), any_type(), Required::Required),
            Param::VarArg(Some(Name::new_static("columns")), any_type()),
            Param::KwOnly(Name::new_static("kw"), any_type(), Required::Required),
        ];

        assert_eq!(match_summary(&params, 0), Some(("x", false)));
        assert_eq!(match_summary(&params, 1), Some(("columns", false)));
        assert_eq!(match_summary(&params, 3), Some(("columns", true)));
    }

    #[test]
    fn param_name_for_positional_argument_handles_missing_names() {
        let params = vec![
            Param::PosOnly(None, any_type(), Required::Required),
            Param::VarArg(None, any_type()),
        ];

        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 0).is_none());
        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 1).is_none());
        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 5).is_none());
    }

    #[test]
    fn duplicate_vararg_hints_are_not_emitted() {
        let params = vec![
            Param::Pos(Name::new_static("s"), any_type(), Required::Required),
            Param::VarArg(Some(Name::new_static("args")), any_type()),
            Param::KwOnly(Name::new_static("a"), any_type(), Required::Required),
        ];

        let labels: Vec<&str> = (0..4)
            .filter_map(|idx| {
                Transaction::<'static>::param_name_for_positional_argument(&params, idx)
            })
            .filter(|match_| !match_.is_vararg_repeat)
            .map(|match_| match_.name.as_str())
            .collect();

        assert_eq!(labels, vec!["s", "args"]);
    }

    fn match_summary(params: &[Param], idx: usize) -> Option<(&str, bool)> {
        Transaction::<'static>::param_name_for_positional_argument(params, idx)
            .map(|match_| (match_.name.as_str(), match_.is_vararg_repeat))
    }

    #[test]
    fn test_find_function_at_position_in_ast() {
        use pyrefly_python::ast::Ast;
        use ruff_python_ast::PySourceType;
        use ruff_text_size::TextSize;

        let source = r#"
def top_level():
    pass

class MyClass:
    def method(self):
        pass
"#;
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);

        // Finds top-level function
        let pos_in_func = TextSize::from(5);
        let result = CancellableTransaction::find_function_at_position_in_ast(&ast, pos_in_func);
        assert_eq!(result.unwrap().name.id.as_str(), "top_level");

        // Finds class method
        let pos_in_method = TextSize::from(50);
        let result = CancellableTransaction::find_function_at_position_in_ast(&ast, pos_in_method);
        assert_eq!(result.unwrap().name.id.as_str(), "method");
    }

    #[test]
    fn test_find_function_at_position_in_ast_returns_none_outside_functions() {
        use pyrefly_python::ast::Ast;
        use ruff_python_ast::PySourceType;
        use ruff_text_size::TextSize;

        let source = "x = 10\n";
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);

        let pos_outside = TextSize::from(0);
        let result = CancellableTransaction::find_function_at_position_in_ast(&ast, pos_outside);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_containing_function_for_call() {
        use std::path::PathBuf;

        use pyrefly_build::handle::Handle;
        use pyrefly_python::ast::Ast;
        use pyrefly_python::module_name::ModuleName;
        use pyrefly_python::module_path::ModulePath;
        use pyrefly_python::sys_info::SysInfo;
        use ruff_python_ast::PySourceType;
        use ruff_text_size::TextSize;

        let source = r#"
def my_function():
    x = call()

class MyClass:
    def method(self):
        y = call()
"#;
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);
        let handle = Handle::new(
            ModuleName::from_str("test"),
            ModulePath::memory(PathBuf::from("test.py")),
            SysInfo::default(),
        );

        // Returns qualified name for top-level function
        let pos_in_func = TextSize::from(30);
        let (name, _) =
            CancellableTransaction::find_containing_function_for_call(&handle, &ast, pos_in_func)
                .unwrap();
        assert_eq!(name, "test.my_function");

        // Returns qualified name for class method
        let pos_in_method = TextSize::from(85);
        let (name, _) =
            CancellableTransaction::find_containing_function_for_call(&handle, &ast, pos_in_method)
                .unwrap();
        assert_eq!(name, "test.MyClass.method");
    }
}
