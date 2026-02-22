/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::fs;
use std::path::PathBuf;
use std::sync::LazyLock;

use dupe::Dupe;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use itertools::Itertools;
use lsp_types::CompletionItem;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::types::Union;
use pyrefly_util::gas::Gas;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Alias;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::ModModule;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Deserialize;
use serde::Serialize;
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
use crate::lsp::wasm::completion::CompletionOptions;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::common_alias_target_module;
use crate::state::ide::import_regular_import_edit;
use crate::state::ide::insert_import_edit;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::lsp_attributes::AttributeContext;
use crate::state::require::Require;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;
use crate::types::module::ModuleType;
use crate::types::type_var::Restriction;
use crate::types::types::Type;

mod dict_completions;
mod quick_fixes;

pub(crate) use self::quick_fixes::types::LocalRefactorCodeAction;

#[derive(Debug)]
pub(crate) enum CalleeKind {
    /// A direct function call: `foo()`
    Function(Identifier),
    /// A method call: `obj.method()` - stores base expression range + method name
    Method(TextRange, Identifier),
    /// Unknown callee (e.g., callable returned from another call)
    Unknown,
}

pub(crate) fn callee_kind_from_call(call: &ExprCall) -> CalleeKind {
    match call.func.as_ref() {
        Expr::Name(name) => CalleeKind::Function(Ast::expr_name_identifier(name.clone())),
        Expr::Attribute(attr) => CalleeKind::Method(attr.value.range(), attr.attr.clone()),
        _ => CalleeKind::Unknown,
    }
}

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

/// PEP 610 direct_url.json structure for detecting editable installs.
#[allow(dead_code)]
#[derive(Deserialize)]
struct DirectUrl {
    url: String,
    #[serde(default)]
    dir_info: DirInfo,
}

#[allow(dead_code)]
#[derive(Deserialize, Default)]
struct DirInfo {
    #[serde(default)]
    editable: bool,
}

/// Cache for editable source paths, keyed by sorted site-packages paths.
/// This avoids re-scanning site-packages on every check.
#[allow(dead_code)]
static EDITABLE_PATHS_CACHE: LazyLock<Mutex<SmallMap<Vec<PathBuf>, Vec<PathBuf>>>> =
    LazyLock::new(|| Mutex::new(SmallMap::new()));

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompletionResolveData {
    Export {
        module: String,
        symbol: String,
        path: Option<String>,
        doc_range: Option<TextRange>,
    },
}

impl CompletionResolveData {
    pub fn export_value(
        module: ModuleName,
        symbol: impl Into<String>,
        path: Option<String>,
        doc_range: Option<TextRange>,
    ) -> serde_json::Value {
        let module_name = module.as_str().to_owned();
        let symbol = symbol.into();
        serde_json::to_value(Self::Export {
            module: module_name.clone(),
            symbol: symbol.clone(),
            path,
            doc_range,
        })
        .unwrap_or_else(|err| {
            panic!("failed to serialize completion resolve data for {module_name}::{symbol}: {err}")
        })
    }
}

pub(crate) fn completion_data_handle_path(handle: &Handle) -> String {
    handle.path().as_path().to_string_lossy().into_owned()
}

fn filesystem_docstring(range: TextRange, path: &str) -> Option<lsp_types::Documentation> {
    let contents = fs::read_to_string(path).ok()?;
    let start = range.start().to_usize();
    let end = range.end().to_usize();
    if start > end || end > contents.len() {
        return None;
    }
    let slice = &contents[start..end];
    let cleaned = Docstring::clean(slice);
    Some(lsp_types::Documentation::MarkupContent(
        lsp_types::MarkupContent {
            kind: lsp_types::MarkupKind::Markdown,
            value: cleaned,
        },
    ))
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

#[derive(Debug, PartialEq, Eq)]
struct QuickfixAction {
    title: String,
    module_info: Module,
    range: TextRange,
    insert_text: String,
    is_deprecated: bool,
    is_private_import: bool,
}

impl QuickfixAction {
    fn to_tuple(self) -> (String, Module, TextRange, String) {
        (self.title, self.module_info, self.range, self.insert_text)
    }
}

impl Ord for QuickfixAction {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort import code actions: non-private first, then non-deprecated, then alphabetically
        match (self.is_private_import, other.is_private_import) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => match (self.is_deprecated, other.is_deprecated) {
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => self.title.cmp(&other.title),
            },
        }
    }
}

impl PartialOrd for QuickfixAction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Transaction<'a> {
    fn allows_explicit_reexport(handle: &Handle) -> bool {
        matches!(
            handle.path().details(),
            ModulePathDetails::FileSystem(_)
                | ModulePathDetails::Namespace(_)
                | ModulePathDetails::Memory(_)
        )
    }

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

    fn import_handle_with_preference(
        &self,
        handle: &Handle,
        module: ModuleName,
        preference: FindPreference,
    ) -> Option<Handle> {
        match preference.prefer_pyi {
            true => self.import_handle(handle, module, None).finding(),
            false => self
                .import_handle_prefer_executable(handle, module, None)
                .finding(),
        }
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
                context:
                    IdentifierContext::ImportedName {
                        name_after_import, ..
                    },
            }) => {
                let key = Key::Definition(ShortIdentifier::new(&name_after_import));
                let bindings = self.get_bindings(handle)?;
                if !bindings.is_valid_key(&key) {
                    return None;
                }
                self.get_type(handle, &key)
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
            Type::TypeAlias(box TypeAliasData::Value(alias)) => {
                Self::callable_from_type(solver, alias.as_type())
            }
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
            let handle = self.import_handle_with_preference(handle, m, preference)?;
            match self.get_exports(&handle).get(&name) {
                Some(ExportLocation::ThisModule(export)) => {
                    return Some((handle.clone(), export.clone()));
                }
                Some(ExportLocation::OtherModule(module, aliased_name)) => {
                    if let Some(aliased_name) = aliased_name {
                        name = aliased_name.clone();
                    }
                    if *module == m && handle.path().is_init() {
                        let submodule = m.append(&name);
                        let sub_handle =
                            self.import_handle_with_preference(&handle, submodule, preference)?;
                        let docstring_range = self.get_module_docstring_range(&sub_handle);
                        return Some((
                            sub_handle,
                            Export {
                                location: TextRange::default(),
                                symbol_kind: Some(SymbolKind::Module),
                                docstring_range,
                                deprecation: None,
                                is_final: false,
                                special_export: None,
                            },
                        ));
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
            IntermediateDefinition::Module(import_range, name) => {
                if matches!(preference.import_behavior, ImportBehavior::StopAtEverything) {
                    return Some((
                        handle.dupe(),
                        Export {
                            location: import_range,
                            symbol_kind: Some(SymbolKind::Module),
                            docstring_range: None,
                            deprecation: None,
                            is_final: false,
                            special_export: None,
                        },
                    ));
                }
                let handle = self.import_handle_with_preference(handle, name, preference)?;
                let docstring_range = self.get_module_docstring_range(&handle);
                Some((
                    handle,
                    Export {
                        location: TextRange::default(),
                        symbol_kind: Some(SymbolKind::Module),
                        docstring_range,
                        deprecation: None,
                        is_final: false,
                        special_export: None,
                    },
                ))
            }
        }
    }

    pub(crate) fn resolve_attribute_definition(
        &self,
        handle: &Handle,
        attr_name: &Name,
        definition: AttrDefinition,
        preference: FindPreference,
    ) -> Option<(TextRangeWithModule, Option<TextRange>)> {
        match definition {
            AttrDefinition::FullyResolved {
                cls,
                range,
                docstring_range,
            } => {
                // If prefer_pyi is false and the current module is a .pyi file,
                // try to find the corresponding .py file
                let text_range_with_module_info =
                    TextRangeWithModule::new(cls.module().dupe(), range);
                if !preference.prefer_pyi
                    && cls.module_path().is_interface()
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
            AttrDefinition::Submodule { module_name } => {
                // For submodule access (e.g., `b` in `a.b` when `import a.b.c`),
                // resolve by finding the submodule's __init__.py
                let def =
                    self.find_definition_for_imported_module(handle, module_name, preference)?;
                Some((
                    TextRangeWithModule::new(def.module, def.definition_range),
                    def.docstring_range,
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

    pub(crate) fn find_definition_for_base_type(
        &self,
        handle: &Handle,
        preference: FindPreference,
        completions: Vec<AttrInfo>,
        name: &Name,
    ) -> Option<FindDefinitionItemWithDocstring> {
        completions.into_iter().find_map(|x| {
            if &x.name == name {
                let (definition, docstring_range) =
                    self.resolve_attribute_definition(handle, &x.name, x.definition, preference)?;
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
                        // Handle membership test operators (in/not in) - uses __contains__ on the right operand
                        if matches!(op, CmpOp::In | CmpOp::NotIn)
                            && let Some(answers) = self.get_answers(handle)
                            && let Some(right_type) =
                                answers.get_type_trace(compare.comparators.first()?.range())
                        {
                            return Some((right_type, dunder::CONTAINS));
                        }
                        // Handle rich comparison operators
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
                // Handle iteration `in` keyword in for loops
                AnyNodeRef::StmtFor(stmt_for) => {
                    if let Some(answers) = self.get_answers(handle)
                        && let Some(iter_type) = answers.get_type_trace(stmt_for.iter.range())
                    {
                        return Some((iter_type, dunder::ITER));
                    }
                    None
                }
                // Handle iteration `in` keyword in comprehensions
                AnyNodeRef::Comprehension(comp) => {
                    if let Some(answers) = self.get_answers(handle)
                        && let Some(iter_type) = answers.get_type_trace(comp.iter.range())
                    {
                        return Some((iter_type, dunder::ITER));
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

    pub(crate) fn find_definition_for_imported_module(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        preference: FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        // TODO: Handle relative import (via ModuleName::new_maybe_relative)
        let handle = self.import_handle_with_preference(handle, module_name, preference)?;
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

    fn find_definition_for_dunder_all_entry(
        &self,
        handle: &Handle,
        position: TextSize,
        preference: FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        let module_info = self.get_module_info(handle)?;
        let exports = self.get_exports_data(handle);
        let (_entry_range, name) = exports.dunder_all_name_at(position)?;

        if let Some((definition_handle, export)) =
            self.resolve_named_import(handle, module_info.name(), name.clone(), preference)
        {
            let definition_module = self.get_module_info(&definition_handle)?;
            return Some(FindDefinitionItemWithDocstring {
                metadata: DefinitionMetadata::VariableOrAttribute(export.symbol_kind),
                definition_range: export.location,
                module: definition_module,
                docstring_range: export.docstring_range,
                display_name: Some(name.to_string()),
            });
        }

        if module_info.path().is_init() {
            let submodule = module_info.name().append(&name);
            if let Some(definition) =
                self.find_definition_for_imported_module(handle, submodule, preference)
            {
                return Some(definition);
            }
        }

        None
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

        if covering_nodes
            .iter()
            .any(|node| matches!(node, AnyNodeRef::ExprStringLiteral(_)))
            && let Some(definition) =
                self.find_definition_for_dunder_all_entry(handle, position, preference)
        {
            return vec![definition];
        }

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
                identifier,
                context:
                    IdentifierContext::ImportedModule {
                        name: module_name, ..
                    },
            }) => {
                // Build the module name for lookup based on identifier position.
                let components = module_name.components();

                let target_module_name =
                    if let Some(idx) = components.iter().position(|c| c == &identifier.id) {
                        // Identifier matches a module component.
                        ModuleName::from_parts(&components[..=idx])
                    } else if identifier.as_str() == module_name.as_str() {
                        // Identifier matches full module name; decide which component based on position offset.
                        let module_str = module_name.as_str();
                        let offset = (position - identifier.range.start())
                            .to_usize()
                            .min(module_str.len());
                        let idx = module_str[..offset].matches('.').count();
                        ModuleName::from_parts(&components[..=idx])
                    } else {
                        // Fallback: use the whole module name.
                        module_name
                    };
                self.find_definition_for_imported_module(handle, target_module_name, preference)
                    .map_or(vec![], |item| vec![item])
            }
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
        let answers = self.get_answers(handle)?;
        let none_type = answers.heap().mk_class_type(stdlib.none_type().clone());
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

    pub(crate) fn search_modules_fuzzy(&self, pattern: &str) -> Vec<ModuleName> {
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
        let mut import_actions = Vec::new();
        let mut generate_actions = Vec::new();
        let mut other_actions = Vec::new();
        for error in errors {
            match error.error_kind() {
                ErrorKind::UnknownName => {
                    let error_range = error.range();
                    if error_range.contains_range(range) {
                        let unknown_name = module_info.code_at(error_range);
                        for (handle_to_import_from, export) in
                            self.search_exports_exact(unknown_name)
                        {
                            self.create_quickfix_action_for_export(
                                handle,
                                import_format,
                                &module_info,
                                &ast,
                                &mut import_actions,
                                unknown_name,
                                handle_to_import_from,
                                export,
                            );
                        }

                        let aliased_module = self.create_quickfix_action_for_common_alias_import(
                            handle,
                            &module_info,
                            &ast,
                            &mut import_actions,
                            unknown_name,
                        );

                        for module_name in self.search_modules_fuzzy(unknown_name) {
                            if module_name == handle.module()
                                || aliased_module.is_some_and(|m| m == module_name)
                            {
                                continue;
                            }
                            self.create_quickfix_action_for_fuzzy_match(
                                handle,
                                &module_info,
                                &ast,
                                &mut import_actions,
                                module_name,
                            );
                        }

                        if let Some(mut actions) = quick_fixes::generate_code::generate_code_actions(
                            self,
                            handle,
                            &module_info,
                            ast.as_ref(),
                            error_range,
                            unknown_name,
                        ) {
                            generate_actions.append(&mut actions);
                        }
                    }
                }
                ErrorKind::RedundantCast => {
                    let error_range = error.range();
                    if let Some(action) = quick_fixes::redundant_cast::redundant_cast_code_action(
                        &module_info,
                        &ast,
                        error_range,
                    ) {
                        let call_range = action.2;
                        if error_range.contains_range(range) || call_range.contains_range(range) {
                            other_actions.push(action);
                        }
                    }
                }
                _ => {}
            }
        }

        import_actions.sort();

        // Keep only the first suggestion for each unique import text (after sorting,
        // this will be the public/non-deprecated version)
        import_actions.dedup_by(|a, b| a.insert_text == b.insert_text);

        // Drop the deprecated flag and return
        let mut actions: Vec<(String, Module, TextRange, String)> =
            import_actions.into_iter().map(|a| a.to_tuple()).collect();
        actions.extend(generate_actions);
        actions.extend(other_actions);
        (!actions.is_empty()).then_some(actions)
    }

    fn create_quickfix_action_for_common_alias_import(
        &self,
        handle: &Handle,
        module_info: &Module,
        ast: &std::sync::Arc<ModModule>,
        import_actions: &mut Vec<QuickfixAction>,
        unknown_name: &str,
    ) -> Option<ModuleName> {
        let module_name_str = common_alias_target_module(unknown_name)?;
        let module_name = ModuleName::from_str(module_name_str);
        if module_name == handle.module() {
            return None;
        }
        let module_handle = self.import_handle(handle, module_name, None).finding()?;
        let (position, insert_text, _) =
            import_regular_import_edit(ast, module_handle, Some(unknown_name));
        let range = TextRange::at(position, TextSize::new(0));
        let title = format!("Use common alias: `{}`", insert_text.trim());
        let is_private_import = module_name
            .components()
            .last()
            .is_some_and(|component| component.as_str().starts_with('_'));
        import_actions.push(QuickfixAction {
            title,
            module_info: module_info.dupe(),
            range,
            insert_text,
            is_deprecated: false,
            is_private_import,
        });
        Some(module_name)
    }

    fn create_quickfix_action_for_fuzzy_match(
        &self,
        handle: &Handle,
        module_info: &Module,
        ast: &std::sync::Arc<ModModule>,
        import_actions: &mut Vec<QuickfixAction>,
        module_name: ModuleName,
    ) {
        if let Some(module_handle) = self.import_handle(handle, module_name, None).finding() {
            let (position, insert_text, _) = import_regular_import_edit(ast, module_handle, None);
            let range = TextRange::at(position, TextSize::new(0));
            let title = format!("Insert import: `{}`", insert_text.trim());
            let is_private_import = module_name
                .components()
                .last()
                .is_some_and(|component| component.as_str().starts_with('_'));
            import_actions.push(QuickfixAction {
                title,
                module_info: module_info.dupe(),
                range,
                insert_text,
                is_deprecated: false,
                is_private_import,
            });
        }
    }

    fn create_quickfix_action_for_export(
        &self,
        handle: &Handle,
        import_format: ImportFormat,
        module_info: &Module,
        ast: &std::sync::Arc<ModModule>,
        import_actions: &mut Vec<QuickfixAction>,
        unknown_name: &str,
        handle_to_import_from: Handle,
        export: Export,
    ) {
        let (position, insert_text, _) = insert_import_edit(
            ast,
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

        import_actions.push(QuickfixAction {
            title,
            module_info: module_info.dupe(),
            range,
            insert_text,
            is_deprecated,
            is_private_import,
        });
    }

    pub fn redundant_cast_fix_all_edits(
        &self,
        handle: &Handle,
    ) -> Option<Vec<(Module, TextRange, String)>> {
        let module_info = self.get_module_info(handle)?;
        let ast = self.get_ast(handle)?;
        let errors = self.get_errors(vec![handle]).collect_errors().shown;
        let mut edits = Vec::new();
        for error in errors {
            if error.error_kind() != ErrorKind::RedundantCast {
                continue;
            }
            if let Some((_, module, range, replacement)) =
                quick_fixes::redundant_cast::redundant_cast_code_action(
                    &module_info,
                    &ast,
                    error.range(),
                )
            {
                edits.push((module, range, replacement));
            }
        }
        if edits.is_empty() {
            None
        } else {
            edits.sort_by_key(|(_, range, _)| range.start());
            Some(edits)
        }
    }

    pub fn extract_function_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::extract_function::extract_function_code_actions(self, handle, selection)
    }

    pub fn extract_field_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::extract_field::extract_field_code_actions(self, handle, selection)
    }

    pub fn extract_variable_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::extract_variable::extract_variable_code_actions(self, handle, selection)
    }

    pub fn invert_boolean_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::invert_boolean::invert_boolean_code_actions(self, handle, selection)
    }

    pub fn extract_superclass_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::extract_superclass::extract_superclass_code_actions(self, handle, selection)
    }

    pub fn pull_members_up_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::move_members::pull_members_up_code_actions(self, handle, selection)
    }

    pub fn push_members_down_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::move_members::push_members_down_code_actions(self, handle, selection)
    }

    pub fn move_module_member_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
        import_format: ImportFormat,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::move_module::move_module_member_code_actions(
            self,
            handle,
            selection,
            import_format,
        )
    }

    pub fn make_local_function_top_level_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
        import_format: ImportFormat,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::move_module::make_local_function_top_level_code_actions(
            self,
            handle,
            selection,
            import_format,
        )
    }

    pub fn inline_variable_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::inline_variable::inline_variable_code_actions(self, handle, selection)
    }

    pub fn inline_method_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::inline_method::inline_method_code_actions(self, handle, selection)
    }

    pub fn inline_parameter_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::inline_parameter::inline_parameter_code_actions(self, handle, selection)
    }

    pub fn introduce_parameter_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::introduce_parameter::introduce_parameter_code_actions(self, handle, selection)
    }
    pub fn convert_star_import_code_actions(
        &self,
        handle: &Handle,
        selection: TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>> {
        quick_fixes::convert_star_import::convert_star_import_code_actions(self, handle, selection)
    }

    /// Determines whether a module is a third-party package.
    ///
    /// Checks if the module's path is located within any of the configured
    /// site-packages directories (e.g., `site-packages/`, `dist-packages/`).
    /// Modules in editable install source paths are NOT considered third-party,
    /// even if they appear in sys.path.
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

            // Check editable packages detected via direct_url.json (PEP 610)
            let site_packages: Vec<PathBuf> = config.site_package_path().cloned().collect();
            let editable_paths = Self::get_editable_source_paths(&site_packages);
            for editable_path in &editable_paths {
                if module_path.as_path().starts_with(editable_path) {
                    return true;
                }
            }
        }

        false
    }

    /// Detect editable packages by scanning site-packages for direct_url.json files (PEP 610).
    #[allow(dead_code)]
    fn detect_editable_packages(site_packages: &[PathBuf]) -> Vec<PathBuf> {
        let mut editable_paths = Vec::new();

        for sp in site_packages {
            let Ok(entries) = std::fs::read_dir(sp) else {
                continue;
            };

            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();

                // Look for .dist-info directories
                if !path.is_dir() {
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "dist-info") {
                    continue;
                }

                let direct_url_path = path.join("direct_url.json");
                let Ok(content) = std::fs::read_to_string(&direct_url_path) else {
                    continue;
                };
                let Ok(direct_url) = serde_json::from_str::<DirectUrl>(&content) else {
                    continue;
                };

                if !direct_url.dir_info.editable {
                    continue;
                }

                // Parse the file:// URL and extract the path
                let Ok(url) = lsp_types::Url::parse(&direct_url.url) else {
                    continue;
                };
                if url.scheme() != "file" {
                    continue;
                }

                let path_str = url.path();

                // On Windows, file URLs look like file:///C:/path
                // url.path() returns "/C:/path", we need to strip the leading "/"
                #[cfg(windows)]
                let path_str = path_str.strip_prefix('/').unwrap_or(path_str);

                // Decode percent-encoded characters (e.g., %20 -> space)
                let Ok(decoded) = percent_encoding::percent_decode_str(path_str).decode_utf8()
                else {
                    continue;
                };

                let source_path = PathBuf::from(decoded.as_ref());
                if source_path.is_dir() {
                    editable_paths.push(source_path);
                }
            }
        }

        editable_paths
    }

    /// Get editable source paths for the given site-packages, using cache.
    #[allow(dead_code)]
    fn get_editable_source_paths(site_packages: &[PathBuf]) -> Vec<PathBuf> {
        let mut key: Vec<PathBuf> = site_packages.to_vec();
        key.sort();

        let mut cache = EDITABLE_PATHS_CACHE.lock();
        if let Some(paths) = cache.get(&key) {
            return paths.clone();
        }

        let paths = Self::detect_editable_packages(site_packages);
        cache.insert(key, paths.clone());
        paths
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

    pub(crate) fn local_references_from_definition(
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
                        definition,
                        is_reexport: _,
                    } in solver.completions(base_type, Some(expected_name), false)
                    {
                        if let Some((TextRangeWithModule { module, range }, _)) = self
                            .resolve_attribute_definition(
                                handle,
                                &name,
                                definition,
                                FindPreference::default(),
                            )
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

    pub fn resolve_completion_item(&self, mut item: CompletionItem) -> CompletionItem {
        if item.documentation.is_some() {
            return item;
        }
        let Some(data_value) = item.data.clone() else {
            return item;
        };
        let data = match serde_json::from_value::<CompletionResolveData>(data_value) {
            Ok(data) => data,
            Err(err) => {
                tracing::debug!("Failed to parse completion resolve data: {err}");
                return item;
            }
        };
        match data {
            CompletionResolveData::Export {
                module,
                symbol,
                path,
                doc_range,
            } => {
                let module_name = ModuleName::from_str(&module);
                tracing::debug!(
                    "Resolve completion data: module={}, symbol={}, path={:?}, doc_range={:?}",
                    module,
                    symbol,
                    path,
                    doc_range
                );
                if let Some(documentation) =
                    self.documentation_for_completion_export(module_name, &symbol)
                {
                    item.documentation = Some(documentation);
                } else if let Some(range) = doc_range
                    && let Some(path_str) = path.as_deref()
                {
                    tracing::debug!(
                        "Resolving completion doc from filesystem: {} {:?}",
                        path_str,
                        range
                    );
                    if let Some(documentation) = filesystem_docstring(range, path_str) {
                        item.documentation = Some(documentation);
                    } else {
                        tracing::debug!(
                            "Failed to load completion doc from filesystem: {} {:?}",
                            path_str,
                            range
                        );
                    }
                }
            }
        }
        item
    }

    fn documentation_for_completion_export(
        &self,
        module_name: ModuleName,
        symbol: &str,
    ) -> Option<lsp_types::Documentation> {
        let handle = self
            .handles()
            .into_iter()
            .find(|handle| handle.module() == module_name)?;
        let name = Name::new(symbol);
        let exports = self.get_exports(&handle);
        let export = exports.get(&name)?;
        let export_info = self.export_from_location(&handle, &name, export);
        self.get_documentation_from_export(export_info)
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
            CompletionOptions {
                supports_completion_item_details,
                ..Default::default()
            },
        )
        .0
    }

    // Returns the completions, and true if they are incomplete so client will keep asking for more completions
    pub fn completion_with_incomplete(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        options: CompletionOptions,
    ) -> (Vec<CompletionItem>, bool) {
        self.completion_with_incomplete_impl(
            handle,
            position,
            import_format,
            options,
            None::<fn(&CompletionItem) -> Option<usize>>,
        )
    }

    pub fn completion_with_incomplete_mru<F>(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        options: CompletionOptions,
        mru_index: F,
    ) -> (Vec<CompletionItem>, bool)
    where
        F: FnMut(&CompletionItem) -> Option<usize>,
    {
        self.completion_with_incomplete_impl(
            handle,
            position,
            import_format,
            options,
            Some(mru_index),
        )
    }

    fn completion_with_incomplete_impl<F>(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
        options: CompletionOptions,
        mru_index: Option<F>,
    ) -> (Vec<CompletionItem>, bool)
    where
        F: FnMut(&CompletionItem) -> Option<usize>,
    {
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
            options,
            mru_index,
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
        let canonical_module = canonical.module();
        let original_module = original.module();
        let canonical_components = canonical_module.components();
        let canonical_component = canonical_components
            .last()
            .map(|name| name.as_str())
            .unwrap_or("");
        let original_components = original_module.components();
        let original_component = original_components
            .last()
            .map(|name| name.as_str())
            .unwrap_or("");

        if canonical_component.starts_with('_')
            && canonical_component.trim_start_matches('_') == original_component
        {
            return true;
        }

        // Include re-export if original is a parent package of canonical.
        if canonical_components.len() > original_components.len()
            && canonical_components
                .iter()
                .zip(original_components.iter())
                .all(|(c, o)| c == o)
        {
            return true;
        }
        // Some stdlib shims encode dotted modules with underscores (e.g. _collections_abc).
        if canonical_module.as_str().starts_with('_') && original_module.as_str().contains('.') {
            let canonical_trim = canonical_module.as_str().trim_start_matches('_');
            if canonical_trim == original_module.as_str().replace('.', "_") {
                return true;
            }
        }
        false
    }

    pub fn search_exports_exact(&self, name: &str) -> Vec<(Handle, Export)> {
        self.search_exports(|handle, exports_data, exports| {
            let name = Name::new(name);
            match exports.get(&name) {
                Some(location) => {
                    if let Some((canonical_handle, export)) =
                        self.export_from_location(handle, &name, location)
                    {
                        let mut results = vec![(canonical_handle.dupe(), export.clone())];
                        if canonical_handle != *handle
                            && (Self::should_include_reexport(handle, &canonical_handle)
                                || (exports_data.is_explicit_reexport(&name)
                                    && Self::allows_explicit_reexport(handle)))
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
        let mut res = self.search_exports(|handle, exports_data, exports| {
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
                        && (Self::should_include_reexport(handle, &canonical_handle)
                            || (exports_data.is_explicit_reexport(name)
                                && Self::allows_explicit_reexport(handle)))
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
    pub(crate) fn process_rdeps_with_definition<T>(
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
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pyrefly_types::heap::TypeHeap;
    use ruff_python_ast::name::Name;

    use super::Transaction;
    use crate::types::callable::Param;
    use crate::types::callable::Required;
    use crate::types::types::Type;

    fn any_type() -> Type {
        TypeHeap::new().mk_any_explicit()
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
    fn test_get_editable_source_paths_finds_editable_package() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let source_dir = temp_dir.path().join("mypackage_source");
        fs::create_dir(&source_dir).unwrap();

        // Use Url::from_file_path to construct a proper file URL that works on all platforms
        let source_url = lsp_types::Url::from_file_path(&source_dir).unwrap();
        let direct_url_content = format!(
            r#"{{"url": "{}", "dir_info": {{"editable": true}}}}"#,
            source_url.as_str()
        );
        fs::write(dist_info.join("direct_url.json"), direct_url_content).unwrap();

        let result =
            Transaction::<'static>::get_editable_source_paths(std::slice::from_ref(&site_packages));

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], source_dir);
    }

    #[test]
    fn test_get_editable_source_paths_ignores_non_editable_package() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("requests-2.28.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let source_dir = temp_dir.path().join("requests_source");
        fs::create_dir(&source_dir).unwrap();

        // Use Url::from_file_path to construct a proper file URL that works on all platforms
        let source_url = lsp_types::Url::from_file_path(&source_dir).unwrap();
        let direct_url_content = format!(
            r#"{{"url": "{}", "dir_info": {{"editable": false}}}}"#,
            source_url.as_str()
        );
        fs::write(dist_info.join("direct_url.json"), direct_url_content).unwrap();

        let result = Transaction::<'static>::get_editable_source_paths(&[site_packages]);

        assert!(result.is_empty());
    }

    #[test]
    fn test_get_editable_source_paths_ignores_missing_direct_url_json() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("somepackage-1.0.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let result = Transaction::<'static>::get_editable_source_paths(&[site_packages]);

        assert!(result.is_empty());
    }

    #[test]
    fn test_get_editable_source_paths_ignores_nonexistent_source_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let nonexistent_path = temp_dir.path().join("does_not_exist");

        // Use Url::from_file_path to construct a proper file URL that works on all platforms
        let nonexistent_url = lsp_types::Url::from_file_path(&nonexistent_path).unwrap();
        let direct_url_content = format!(
            r#"{{"url": "{}", "dir_info": {{"editable": true}}}}"#,
            nonexistent_url.as_str()
        );
        fs::write(dist_info.join("direct_url.json"), direct_url_content).unwrap();

        let result = Transaction::<'static>::get_editable_source_paths(&[site_packages]);

        assert!(result.is_empty());
    }
}
