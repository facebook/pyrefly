/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use itertools::Itertools;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::CompletionItemTag;
use lsp_types::DocumentSymbol;
use lsp_types::ParameterInformation;
use lsp_types::ParameterLabel;
use lsp_types::SemanticToken;
use lsp_types::SignatureHelp;
use lsp_types::SignatureInformation;
use lsp_types::TextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::keywords::get_keywords;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
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
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Deserialize;
use starlark_map::ordered_set::OrderedSet;
use starlark_map::small_map::SmallMap;

use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::config::error_kind::ErrorKind;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::graph::index::Idx;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::insert_import_edit;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::require::Require;
use crate::state::semantic_tokens::SemanticTokenBuilder;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::callable::Params;
use crate::types::module::ModuleType;
use crate::types::types::Type;

fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
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
    #[expect(unused)]
    pub call_argument_names: AllOffPartial,
    #[serde(default = "default_true")]
    pub function_return_types: bool,
    #[serde(default)]
    #[expect(unused)]
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
}

const RESOLVE_EXPORT_INITIAL_GAS: Gas = Gas::new(100);
const MIN_CHARACTERS_TYPED_AUTOIMPORT: usize = 3;

#[derive(Clone, Debug)]
pub struct FindPreference {
    pub jump_through_renamed_import: bool,
    /// controls whether to prioritize finding pyi or py files. if false, we will search all search paths until a .py file is found before
    /// falling back to a .pyi.
    pub prefer_pyi: bool,
}

impl Default for FindPreference {
    fn default() -> Self {
        Self {
            jump_through_renamed_import: true,
            prefer_pyi: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum DefinitionMetadata {
    Attribute(Name),
    Module,
    Variable(Option<SymbolKind>),
    VariableOrAttribute(Name, Option<SymbolKind>),
}

impl DefinitionMetadata {
    pub fn symbol_kind(&self) -> Option<SymbolKind> {
        match self {
            DefinitionMetadata::Attribute(_) => Some(SymbolKind::Attribute),
            DefinitionMetadata::Module => Some(SymbolKind::Module),
            DefinitionMetadata::Variable(symbol_kind) => symbol_kind.as_ref().copied(),
            DefinitionMetadata::VariableOrAttribute(_, symbol_kind) => {
                symbol_kind.as_ref().copied()
            }
        }
    }
}

/// A binding that is verified to be a binding for a name in the source code.
/// This data structure carries the proof for the verification,
/// which includes the definition information, and the binding itself.
struct NamedBinding {
    definition_handle: Handle,
    definition_export: Export,
    key: Key,
}

#[derive(Debug)]
enum CalleeKind {
    // Function name
    Function(Identifier),
    // Range of the base expr + method name
    Method(TextRange, Identifier),
    Unknown,
}

#[derive(Debug)]
enum PatternMatchParameterKind {
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
enum IdentifierContext {
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
struct IdentifierWithContext {
    identifier: Identifier,
    context: IdentifierContext,
}

#[derive(PartialEq, Eq)]
pub enum AnnotationKind {
    #[allow(dead_code)]
    Parameter,
    Return,
    Variable,
}

#[derive(Debug)]
pub struct ParameterAnnotation {
    pub text_size: TextSize,
    pub has_annotation: bool,
    pub ty: Option<Type>,
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
        let callee_kind = match call.func.as_ref() {
            Expr::Name(name) => CalleeKind::Function(Ast::expr_name_identifier(name.clone())),
            Expr::Attribute(attr) => CalleeKind::Method(attr.value.range(), attr.attr.clone()),
            _ => CalleeKind::Unknown,
        };
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
}

#[derive(Debug)]
pub struct FindDefinitionItem {
    pub metadata: DefinitionMetadata,
    pub definition_range: TextRange,
    pub module: Module,
}

impl<'a> Transaction<'a> {
    fn get_type(&self, handle: &Handle, key: &Key) -> Option<Type> {
        let idx = self.get_bindings(handle)?.key_to_idx(key);
        let answers = self.get_answers(handle)?;
        answers.get_type_at(idx)
    }

    fn get_type_trace(&self, handle: &Handle, range: TextRange) -> Option<Type> {
        let ans = self.get_answers(handle)?;
        ans.get_type_trace(range)
    }

    fn get_chosen_overload_trace(&self, handle: &Handle, range: TextRange) -> Option<Type> {
        let ans = self.get_answers(handle)?;
        ans.get_chosen_overload_trace(range)
    }

    fn identifier_at(&self, handle: &Handle, position: TextSize) -> Option<IdentifierWithContext> {
        let mod_module = self.get_ast(handle)?;
        let covering_nodes = Ast::locate_node(&mod_module, position);
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

                if self.get_bindings(handle)?.is_valid_key(&key) {
                    if let Some(ExprCall {
                        node_index: _,
                        range: _,
                        func,
                        arguments,
                    }) = &self.callee_at(handle, position)
                        && func.range() == id.range
                        && let Some(ret) = self.get_chosen_overload_trace(handle, arguments.range)
                    {
                        Some(ret)
                    } else {
                        self.get_type(handle, &key)
                    }
                } else {
                    None
                }
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
                    &FindPreference::default(),
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
            None => None,
        }
    }

    fn visit_finding_signature_range(
        x: &Expr,
        find: TextSize,
        res: &mut Option<(TextRange, TextRange, usize)>,
    ) {
        if let Expr::Call(call) = x
            && call.arguments.range.contains_inclusive(find)
        {
            for (i, arg) in call.arguments.args.as_ref().iter().enumerate() {
                if arg.range().contains_inclusive(find) {
                    Self::visit_finding_signature_range(arg, find, res);
                    if res.is_some() {
                        return;
                    }
                    *res = Some((call.func.range(), call.arguments.range, i));
                }
            }
            if res.is_none() {
                *res = Some((
                    call.func.range(),
                    call.arguments.range,
                    call.arguments.len(),
                ));
            }
        } else {
            x.recurse(&mut |x| Self::visit_finding_signature_range(x, find, res));
        }
    }

    /// Finds the callable(s) (multiple if overloads exist) at position in document, returning them, chosen overload index, and arg index
    fn get_callables_from_call(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<(Vec<Type>, usize, usize)> {
        let mod_module = self.get_ast(handle)?;
        let mut res = None;
        mod_module.visit(&mut |x| Self::visit_finding_signature_range(x, position, &mut res));
        let (callee_range, call_args_range, arg_index) = res?;
        let answers = self.get_answers(handle)?;
        if let Some((overloads, chosen_overload_index)) =
            answers.get_all_overload_trace(call_args_range)
        {
            let callables = overloads.into_map(|callable| Type::Callable(Box::new(callable)));
            Some((
                callables,
                chosen_overload_index.unwrap_or_default(),
                arg_index,
            ))
        } else {
            answers
                .get_type_trace(callee_range)
                .map(|t| (vec![t], 0, arg_index))
        }
    }

    pub fn get_signature_help_at(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<SignatureHelp> {
        self.get_callables_from_call(handle, position).map(
            |(callables, chosen_overload_index, arg_index)| SignatureHelp {
                signatures: callables
                    .into_iter()
                    .map(|t| Self::create_signature_information(t, arg_index))
                    .collect_vec(),
                active_signature: Some(chosen_overload_index as u32),
                active_parameter: Some(arg_index as u32),
            },
        )
    }

    fn create_signature_information(type_: Type, arg_index: usize) -> SignatureInformation {
        let type_ = type_.deterministic_printing();
        let label = type_.as_hover_string();
        let (parameters, active_parameter) =
            if let Some(params) = Self::normalize_singleton_function_type_into_params(type_) {
                let active_parameter = if arg_index < params.len() {
                    Some(arg_index as u32)
                } else {
                    None
                };
                (
                    Some(params.map(|param| ParameterInformation {
                        label: ParameterLabel::Simple(format!("{param}")),
                        documentation: None,
                    })),
                    active_parameter,
                )
            } else {
                (None, None)
            };
        SignatureInformation {
            label,
            documentation: None,
            parameters,
            active_parameter,
        }
    }

    fn normalize_singleton_function_type_into_params(type_: Type) -> Option<Vec<Param>> {
        let callable = type_.to_callable()?;
        // We will drop the self parameter for signature help
        if let Params::List(params_list) = callable.params {
            if let Some(Param::PosOnly(Some(name), _, _) | Param::Pos(name, _, _)) =
                params_list.items().first()
                && (name.as_str() == "self" || name.as_str() == "cls")
            {
                let mut params = params_list.into_items();
                params.remove(0);
                return Some(params);
            }
            return Some(params_list.into_items());
        }
        None
    }

    fn resolve_named_import(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        name: Name,
        preference: &FindPreference,
    ) -> Option<(Handle, Export)> {
        let mut m = module_name;
        let mut gas = RESOLVE_EXPORT_INITIAL_GAS;
        let mut name = name;
        while !gas.stop() {
            let handle = match preference.prefer_pyi {
                true => self.import_handle(handle, m, None).ok()?,
                false => self.import_handle_prefer_executable(handle, m, None).ok()?,
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

    /// When `preference.jump_through_renamed_import` is true, we will jump through renamed import like
    /// `from foo import bar as baz`, when we try to compute definition of baz.
    /// Otherwise, we will stop at `baz``.
    fn resolve_intermediate_definition(
        &self,
        handle: &Handle,
        intermediate_definition: IntermediateDefinition,
        preference: &FindPreference,
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
                if !preference.jump_through_renamed_import && original_name_range.is_some() {
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
                    true => self.import_handle(handle, name, None).ok()?,
                    false => self
                        .import_handle_prefer_executable(handle, name, None)
                        .ok()?,
                };
                let docstring_range = self.get_module_docstring_range(&handle);
                Some((
                    handle,
                    Export {
                        location: TextRange::default(),
                        symbol_kind: Some(SymbolKind::Module),
                        docstring_range,
                        is_deprecated: false,
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
        preference: &FindPreference,
    ) -> Option<(TextRangeWithModule, Option<TextRange>)> {
        match definition {
            AttrDefinition::FullyResolved(text_range_with_module_info) => {
                // TODO(kylei): attribute docstrings
                Some((text_range_with_module_info, None))
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

    fn key_to_export(
        &self,
        handle: &Handle,
        key: &Key,
        preference: &FindPreference,
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

    fn find_definition_for_name_def(
        &self,
        handle: &Handle,
        name: &Identifier,
        preference: &FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        let def_key = Key::Definition(ShortIdentifier::new(name));
        if !self.get_bindings(handle)?.is_valid_key(&def_key) {
            return None;
        }
        let (
            handle,
            Export {
                location,
                symbol_kind,
                docstring_range,
                ..
            },
        ) = self.key_to_export(handle, &def_key, preference)?;
        let module_info = self.get_module_info(&handle)?;
        let name = Name::new(module_info.code_at(location));
        Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::VariableOrAttribute(name, symbol_kind),
            definition_range: location,
            module: module_info,
            docstring_range,
        })
    }

    pub fn find_definition_for_name_use(
        &self,
        handle: &Handle,
        name: &Identifier,
        preference: &FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        let use_key = Key::BoundName(ShortIdentifier::new(name));
        if !self.get_bindings(handle)?.is_valid_key(&use_key) {
            return None;
        }
        let (
            handle,
            Export {
                location,
                symbol_kind,
                docstring_range,
                ..
            },
        ) = self.key_to_export(handle, &use_key, preference)?;
        Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::Variable(symbol_kind),
            definition_range: location,
            module: self.get_module_info(&handle)?,
            docstring_range,
        })
    }

    pub fn find_definition_for_attribute(
        &self,
        handle: &Handle,
        base_range: TextRange,
        name: &Identifier,
        preference: &FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        if let Some(answers) = self.get_answers(handle)
            && let Some(base_type) = answers.get_type_trace(base_range)
        {
            self.ad_hoc_solve(handle, |solver| {
                let find_definition_for_base_type = |ty: Type| {
                    solver
                        .completions(ty, Some(name.id()), false)
                        .into_iter()
                        .find_map(|x| {
                            if &x.name == name.id() {
                                let (definition, docstring_range) = self
                                    .resolve_attribute_definition(
                                        handle,
                                        &x.name,
                                        x.definition?,
                                        preference,
                                    )?;
                                Some(FindDefinitionItemWithDocstring {
                                    metadata: DefinitionMetadata::Attribute(x.name),
                                    definition_range: definition.range,
                                    module: definition.module,
                                    docstring_range,
                                })
                            } else {
                                None
                            }
                        })
                };
                match base_type {
                    Type::Union(tys) | Type::Intersect(tys) => tys
                        .into_iter()
                        .filter_map(&find_definition_for_base_type)
                        .collect(),
                    ty => find_definition_for_base_type(ty).map_or(vec![], |item| vec![item]),
                }
            })
            .unwrap_or_default()
        } else {
            vec![]
        }
    }

    fn find_definition_for_imported_module(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        preference: &FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        // TODO: Handle relative import (via ModuleName::new_maybe_relative)
        let handle = match preference.prefer_pyi {
            true => self.import_handle(handle, module_name, None).ok()?,
            false => self
                .import_handle_prefer_executable(handle, module_name, None)
                .ok()?,
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
        })
    }

    fn find_definition_for_keyword_argument(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        callee_kind: &CalleeKind,
        preference: &FindPreference,
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
                    Ast::parse(module_info.contents()).0.into()
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
        preference: &FindPreference,
    ) -> Vec<TextRangeWithModule> {
        let defs = match callee_kind {
            CalleeKind::Function(name) => self
                .find_definition_for_name_use(handle, name, preference)
                .map_or(vec![], |item| vec![item]),
            CalleeKind::Method(base_range, name) => {
                self.find_definition_for_attribute(handle, *base_range, name, preference)
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
        preference: &FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        match self.identifier_at(handle, position) {
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
                    metadata: DefinitionMetadata::Attribute(identifier.id().clone()),
                    module,
                    definition_range: identifier.range,
                    docstring_range,
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
                }),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::Attribute { base_range, .. },
            }) => self.find_definition_for_attribute(handle, base_range, &identifier, preference),
            None => vec![],
        }
    }

    pub fn goto_definition(&self, handle: &Handle, position: TextSize) -> Vec<TextRangeWithModule> {
        let mut definitions = self.find_definition(
            handle,
            position,
            &FindPreference {
                prefer_pyi: false,
                ..Default::default()
            },
        );
        // Add pyi definitions if we haven't found any py definition
        if definitions.is_empty() {
            definitions.append(&mut self.find_definition(
                handle,
                position,
                &FindPreference::default(),
            ));
        }

        definitions.into_map(|item| TextRangeWithModule::new(item.module, item.definition_range))
    }

    pub fn goto_type_definition(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Vec<TextRangeWithModule> {
        self.find_definition(handle, position, &FindPreference::default())
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
            &FindPreference {
                jump_through_renamed_import: false,
                ..Default::default()
            },
        )
        .into_iter()
        .next()
        .map(|item| TextRangeWithModule::new(item.module, item.definition_range))
    }

    /// Produce code actions that makes edits local to the file.
    pub fn local_quickfix_code_actions(
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
                        for handle_to_import_from in self.search_exports_exact(unknown_name) {
                            let (position, insert_text) = insert_import_edit(
                                &ast,
                                self.config_finder(),
                                handle.dupe(),
                                handle_to_import_from,
                                unknown_name,
                                import_format,
                            );
                            let range = TextRange::at(position, TextSize::new(0));
                            let title = format!("Insert import: `{}`", insert_text.trim());
                            code_actions.push((title, module_info.dupe(), range, insert_text));
                        }
                    }
                }
                _ => {}
            }
        }
        code_actions.sort_by(|(title1, _, _, _), (title2, _, _, _)| title1.cmp(title2));
        Some(code_actions)
    }

    pub fn prepare_rename(&self, handle: &Handle, position: TextSize) -> Option<TextRange> {
        self.identifier_at(handle, position).map(
            |IdentifierWithContext {
                 identifier,
                 context: _,
             }| identifier.range,
        )
    }

    pub fn find_local_references(&self, handle: &Handle, position: TextSize) -> Vec<TextRange> {
        self.find_definition(
            handle,
            position,
            &FindPreference {
                jump_through_renamed_import: false,
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
             }| {
                self.local_references_from_definition(handle, metadata, definition_range, module)
            },
        )
        .concat()
    }

    fn local_references_from_definition(
        &self,
        handle: &Handle,
        definition_metadata: DefinitionMetadata,
        definition_range: TextRange,
        module: Module,
    ) -> Option<Vec<TextRange>> {
        if handle.path() != module.path() {
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
                    &FindPreference::default(),
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
            references.sort_by_key(|range| range.start());
            references.dedup();
            return Some(references);
        }
        let mut references = match definition_metadata {
            DefinitionMetadata::Attribute(expected_name) => self
                .local_attribute_references_from_definition(
                    handle,
                    definition_range,
                    &expected_name,
                ),
            DefinitionMetadata::Module => Vec::new(),
            DefinitionMetadata::Variable(_) => self
                .local_variable_references_from_definition(handle, definition_range, &module)
                .unwrap_or_default(),
            DefinitionMetadata::VariableOrAttribute(expected_name, _) => [
                self.local_attribute_references_from_definition(
                    handle,
                    definition_range,
                    &expected_name,
                ),
                self.local_variable_references_from_definition(handle, definition_range, &module)
                    .unwrap_or_default(),
            ]
            .concat(),
        };
        if module.path() == handle.path() {
            references.push(definition_range);
        }
        references.sort_by_key(|range| range.start());
        references.dedup();
        Some(references)
    }

    fn local_attribute_references_from_definition(
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
                    } in solver.completions(base_type, Some(expected_name), false)
                    {
                        if let Some((TextRangeWithModule { module, range }, _)) =
                            attribute_definition.and_then(|definition| {
                                self.resolve_attribute_definition(
                                    handle,
                                    &name,
                                    definition,
                                    &FindPreference::default(),
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

    fn local_variable_references_from_definition(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        module: &Module,
    ) -> Option<Vec<TextRange>> {
        let bindings = self.get_bindings(handle)?;
        let mut references = Vec::new();
        for NamedBinding {
            definition_handle,
            definition_export,
            key,
        } in self.named_bindings(handle, &bindings)
        {
            if definition_handle.path() == module.path()
                && definition_range == definition_export.location
            {
                references.push(key.range());
            }
        }
        Some(references)
    }

    /// Bindings can contain synthetic bindings, which are not meaningful to end users.
    /// This function helps to filter out such bindings and only leave bindings that eventually
    /// jumps to a name in the source.
    fn named_bindings(&self, handle: &Handle, bindings: &Bindings) -> Vec<NamedBinding> {
        let mut named_bindings = Vec::new();
        for idx in bindings.keys::<Key>() {
            let key = bindings.idx_to_key(idx);
            if matches!(key, Key::Phi(..) | Key::Narrow(..)) {
                // These keys are always synthetic and never serves as a name definition.
                continue;
            }
            if let Some((definition_handle, definition_export)) = self.key_to_export(
                handle,
                key,
                &FindPreference {
                    jump_through_renamed_import: false,
                    ..Default::default()
                },
            ) {
                named_bindings.push(NamedBinding {
                    definition_handle,
                    definition_export,
                    key: key.clone(),
                });
            }
        }
        named_bindings
    }

    fn add_kwargs_completions(
        &self,
        handle: &Handle,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Some((callables, overload_idx, _)) = self.get_callables_from_call(handle, position)
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

    fn add_builtins_autoimport_completions(
        &self,
        handle: &Handle,
        identifier: Option<&Identifier>,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Ok(builtin_handle) = self.import_handle(handle, ModuleName::builtins(), None) {
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
                let (insert_text, additional_text_edits) = {
                    let (position, insert_text) = insert_import_edit(
                        &ast,
                        self.config_finder(),
                        handle.dupe(),
                        handle_to_import_from,
                        &name,
                        import_format,
                    );
                    let import_text_edit = TextEdit {
                        range: module_info
                            .lined_buffer()
                            .to_lsp_range(TextRange::at(position, TextSize::new(0))),
                        new_text: insert_text.clone(),
                    };
                    (Some(insert_text), Some(vec![import_text_edit]))
                };
                completions.push(CompletionItem {
                    label: name,
                    detail: insert_text,
                    kind: export
                        .symbol_kind
                        .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                            Some(k.to_lsp_completion_item_kind())
                        }),
                    additional_text_edits,
                    tags: if export.is_deprecated {
                        Some(vec![CompletionItemTag::DEPRECATED])
                    } else {
                        None
                    },
                    ..Default::default()
                });
            }
        }
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
                let kind = binding
                    .symbol_kind()
                    .map_or(CompletionItemKind::VARIABLE, |k| {
                        k.to_lsp_completion_item_kind()
                    });
                let ty = self.get_type(handle, key);
                let is_deprecated = ty.as_ref().is_some_and(|t| {
                    if let Type::ClassDef(cls) = t {
                        self.ad_hoc_solve(handle, |solver| {
                            solver.get_metadata_for_class(cls).is_deprecated()
                        })
                        .unwrap_or(false)
                    } else {
                        t.is_deprecated_function()
                    }
                });
                let detail = ty.map(|t| t.to_string());
                has_added_any = true;
                completions.push(CompletionItem {
                    label: label.to_owned(),
                    detail,
                    kind: Some(kind),
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

    fn add_literal_completions(
        &self,
        handle: &Handle,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) {
        if let Some((callables, chosen_overload_index, arg_index)) =
            self.get_callables_from_call(handle, position)
            && let Some(callable) = callables.get(chosen_overload_index)
            && let Some(params) =
                Self::normalize_singleton_function_type_into_params(callable.clone())
            && let Some(param) = params.get(arg_index)
        {
            Self::add_literal_completions_from_type(param.as_type(), completions);
        }
    }

    fn add_literal_completions_from_type(param_type: &Type, completions: &mut Vec<CompletionItem>) {
        match param_type {
            Type::Literal(lit) => {
                completions.push(CompletionItem {
                    // TODO: Pass the flag correctly for whether literal string is single quoted or double quoted
                    label: lit.to_string_escaped(true),
                    kind: Some(CompletionItemKind::VALUE),
                    detail: Some(format!("{param_type}")),
                    ..Default::default()
                });
            }
            Type::Union(types) => {
                for union_type in types {
                    Self::add_literal_completions_from_type(union_type, completions);
                }
            }
            _ => {}
        }
    }

    pub fn completion(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
    ) -> Vec<CompletionItem> {
        let mut results = self.completion_unsorted_opt(handle, position, import_format);
        for item in &mut results {
            let sort_text = if item.additional_text_edits.is_some() {
                "3"
            } else if item.label.starts_with("__") {
                "2"
            } else if item.label.as_str().starts_with("_") {
                "1"
            } else {
                "0"
            }
            .to_owned();
            item.sort_text = Some(sort_text);
        }
        results.sort_by(|item1, item2| {
            item1
                .sort_text
                .cmp(&item2.sort_text)
                .then_with(|| item1.label.cmp(&item2.label))
                .then_with(|| item1.detail.cmp(&item2.detail))
        });
        results.dedup_by(|item1, item2| item1.label == item2.label && item1.detail == item2.detail);
        results
    }

    fn completion_unsorted_opt(
        &self,
        handle: &Handle,
        position: TextSize,
        import_format: ImportFormat,
    ) -> Vec<CompletionItem> {
        let mut result = Vec::new();
        match self.identifier_at(handle, position) {
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ImportedName { module_name, .. },
            }) => {
                if let Ok(handle) = self.import_handle(handle, module_name, None) {
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
                            ExportLocation::ThisModule(export) => export.is_deprecated,
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
                                let detail = ty.clone().map(|t| t.as_hover_string());
                                result.push(CompletionItem {
                                    label: x.name.as_str().to_owned(),
                                    detail,
                                    kind,
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
            Some(IdentifierWithContext { identifier, .. }) => {
                self.add_kwargs_completions(handle, position, &mut result);
                self.add_keyword_completions(handle, &mut result);
                if !self.add_local_variable_completions(
                    handle,
                    Some(&identifier),
                    position,
                    &mut result,
                ) {
                    self.add_autoimport_completions(
                        handle,
                        &identifier,
                        &mut result,
                        import_format,
                    );
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
                    self.add_literal_completions(handle, position, &mut result);
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
        result
    }

    fn collect_types_from_callees(&self, range: TextRange, handle: &Handle) -> Vec<Type> {
        fn callee_at(mod_module: Arc<ModModule>, position: TextSize) -> Option<ExprCall> {
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
        match self.get_ast(handle) {
            Some(mod_module) => {
                let callee = callee_at(mod_module, range.start());
                match callee {
                    Some(ExprCall {
                        arguments: args, ..
                    }) => args
                        .args
                        .iter()
                        .filter_map(|arg| self.get_type_trace(handle, arg.range()))
                        .collect(),
                    None => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    fn collect_references(
        &self,
        handle: &Handle,
        idx: Idx<Key>,
        bindings: Bindings,
        transaction: &mut CancellableTransaction,
    ) -> Vec<(Module, Vec<TextRange>)> {
        if let Key::Definition(id) = bindings.idx_to_key(idx)
            && let Some(module_info) = self.get_module_info(handle)
        {
            let definition_kind = DefinitionMetadata::VariableOrAttribute(
                Name::from(module_info.code_at(id.range())),
                None,
            );
            if let Ok(references) = transaction.find_global_references_from_definition(
                handle.sys_info(),
                definition_kind,
                TextRangeWithModule::new(module_info, id.range()),
            ) {
                return references;
            }
        }
        Vec::new()
    }

    fn filter_parameters(
        &self,
        param_with_default: ParameterWithDefault,
        handle: &Handle,
    ) -> Option<ParameterAnnotation> {
        if param_with_default.name() == "self" || param_with_default.name() == "cls" {
            return None;
        }
        let ty = match param_with_default.default() {
            Some(expr) => self.get_type_trace(handle, expr.range()),
            None => None,
        };
        Some(ParameterAnnotation {
            text_size: param_with_default.parameter.range().end(),
            ty,
            has_annotation: param_with_default.annotation().is_some(),
        })
    }

    pub fn infer_parameter_annotations(
        &self,
        handle: &Handle,
        cancellable_transaction: &mut CancellableTransaction,
    ) -> Vec<ParameterAnnotation> {
        if let Some(bindings) = self.get_bindings(handle) {
            let transaction = cancellable_transaction;
            fn transpose<T: Clone>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
                if v.is_empty() {
                    return Vec::new();
                }
                let max_len = v.iter().map(|row| row.len()).max().unwrap();
                let mut result = vec![Vec::new(); max_len];
                for row in v {
                    for (i, elem) in row.into_iter().enumerate() {
                        result[i].push(elem);
                    }
                }
                result
            }
            fn zip_types(
                inferred_types: Vec<Vec<Type>>,
                function_arguments: Vec<ParameterAnnotation>,
            ) -> Vec<ParameterAnnotation> {
                let zipped_inferred_types: Vec<Vec<Type>> = transpose(inferred_types);
                let types: Vec<(ParameterAnnotation, Vec<Type>)> =
                    match zipped_inferred_types.is_empty() {
                        true => function_arguments
                            .into_iter()
                            .map(
                                |arg: ParameterAnnotation| -> (ParameterAnnotation, Vec<Type>) {
                                    (arg, vec![])
                                },
                            )
                            .collect(),
                        false => function_arguments
                            .into_iter()
                            .zip(zipped_inferred_types)
                            .collect(),
                    };

                types
                    .into_iter()
                    .map(|(arg, mut ty)| {
                        let mut arg = arg;
                        if let Some(default_type) = arg.ty {
                            ty.push(default_type)
                        }
                        if ty.len() == 1 {
                            arg.ty = Some(ty[0].clone());
                        } else {
                            let ty = ty.into_iter().filter(|x| !x.is_any()).collect();
                            arg.ty = Some(Type::Union(ty));
                        }
                        arg
                    })
                    .collect()
            }

            bindings
                .keys::<Key>()
                .flat_map(|idx| {
                    let binding = bindings.get(idx);
                    // Check if this binding is a function
                    if let Binding::Function(key_function, _, _) = binding {
                        let binding_func =
                            bindings.get(bindings.get(*key_function).undecorated_idx);
                        let args = binding_func.def.parameters.args.clone();
                        let func_args: Vec<ParameterAnnotation> = args
                            .into_iter()
                            .filter_map(|param_with_default| {
                                self.filter_parameters(param_with_default, handle)
                            })
                            .collect();
                        let references =
                            self.collect_references(handle, idx, bindings.clone(), transaction);
                        let ranges: Vec<&TextRange> =
                            references.iter().flat_map(|(_, range)| range).collect();
                        let inferred_types =
                            ranges.map(|range| self.collect_types_from_callees(**range, handle));
                        zip_types(inferred_types, func_args)
                    } else {
                        vec![]
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    pub fn inferred_types(
        &self,
        handle: &Handle,
        return_types: bool,
        containers: bool,
    ) -> Option<Vec<(TextSize, Type, AnnotationKind)>> {
        let is_interesting_type = |x: &Type| !x.is_error();
        let is_interesting_expr = |x: &Expr| !Ast::is_literal(x);
        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                // Return Annotation
                key @ Key::ReturnType(id) if return_types => {
                    match bindings.get(bindings.key_to_idx(&Key::Definition(*id))) {
                        Binding::Function(x, _pred, _class_meta) => {
                            if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                && let Some(ty) = self.get_type(handle, key)
                                && is_interesting_type(&ty)
                            {
                                let fun = bindings.get(bindings.get(*x).undecorated_idx);
                                res.push((
                                    fun.def.parameters.range.end(),
                                    ty,
                                    AnnotationKind::Return,
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                // Only annotate empty containers for now
                key @ Key::Definition(_) if containers => {
                    if let Some(ty) = self.get_type(handle, key) {
                        let e = match bindings.get(idx) {
                            Binding::NameAssign(_, None, e, _) => match &**e {
                                Expr::List(ExprList { elts, .. }) => {
                                    if elts.is_empty() {
                                        Some(&**e)
                                    } else {
                                        None
                                    }
                                }
                                Expr::Dict(ExprDict { items, .. }) => {
                                    if items.is_empty() {
                                        Some(&**e)
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some(e) = e
                            && is_interesting_expr(e)
                            && is_interesting_type(&ty)
                        {
                            res.push((key.range().end(), ty, AnnotationKind::Variable));
                        }
                    }
                }
                _ => {}
            }
        }
        Some(res)
    }

    pub fn inlay_hints(
        &self,
        handle: &Handle,
        inlay_hint_config: InlayHintConfig,
    ) -> Option<Vec<(TextSize, String)>> {
        let is_interesting = |e: &Expr, ty: &Type, class_name: Option<&Name>| {
            !ty.is_error()
                && match e {
                    Expr::Tuple(tuple) => {
                        !tuple.elts.is_empty() && tuple.elts.iter().all(|x| !Ast::is_literal(x))
                    }
                    Expr::Call(ExprCall { func, .. }) => {
                        if let Expr::Name(name) = &**func
                            && let Some(class_name) = class_name
                        {
                            *name.id() != *class_name
                        } else if let Expr::Attribute(attr) = &**func
                            && let Some(class_name) = class_name
                        {
                            *attr.attr.id() != *class_name
                        } else {
                            true
                        }
                    }
                    _ => !Ast::is_literal(e),
                }
        };
        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                key @ Key::ReturnType(id) => {
                    if inlay_hint_config.function_return_types {
                        match bindings.get(bindings.key_to_idx(&Key::Definition(*id))) {
                            Binding::Function(x, _pred, _class_meta) => {
                                if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                    && let Some(mut ty) = self.get_type(handle, key)
                                    && !ty.is_error()
                                {
                                    let fun = bindings.get(bindings.get(*x).undecorated_idx);
                                    if fun.def.is_async
                                        && let Some(Some((_, _, return_ty))) = self
                                            .ad_hoc_solve(handle, |solver| {
                                                solver.unwrap_coroutine(&ty)
                                            })
                                    {
                                        ty = return_ty;
                                    }
                                    res.push((fun.def.parameters.range.end(), format!(" -> {ty}")));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                key @ Key::Definition(_)
                    if inlay_hint_config.variable_types
                        && let Some(ty) = self.get_type(handle, key) =>
                {
                    let e = match bindings.get(idx) {
                        Binding::NameAssign(_, None, e, _) => Some(&**e),
                        Binding::Expr(None, e) => Some(e),
                        _ => None,
                    };
                    // If the inferred type is a class type w/ no type arguments and the
                    // RHS is a call to a function that's the same name as the inferred class,
                    // we assume it's a constructor and do not display an inlay hint
                    let class_name = if let Type::ClassType(cls) = &ty
                        && cls.targs().is_empty()
                    {
                        Some(cls.name())
                    } else {
                        None
                    };
                    if let Some(e) = e
                        && is_interesting(e, &ty, class_name)
                    {
                        let ty = format!(": {ty}");
                        res.push((key.range().end(), ty));
                    }
                }
                _ => {}
            }
        }
        Some(res)
    }

    pub fn semantic_tokens(
        &self,
        handle: &Handle,
        limit_range: Option<TextRange>,
    ) -> Option<Vec<SemanticToken>> {
        let module_info = self.get_module_info(handle)?;
        let bindings = self.get_bindings(handle)?;
        let ast = self.get_ast(handle)?;
        let legends = SemanticTokensLegends::new();
        let mut builder = SemanticTokenBuilder::new(limit_range);
        for NamedBinding {
            definition_handle,
            definition_export,
            key,
        } in self.named_bindings(handle, &bindings)
        {
            if let Export {
                symbol_kind: Some(symbol_kind),
                ..
            } = definition_export
            {
                builder.process_key(&key, definition_handle.module(), symbol_kind)
            }
        }
        builder.process_ast(&ast, &|range| self.get_type_trace(handle, range));
        Some(
            legends
                .convert_tokens_into_lsp_semantic_tokens(&builder.all_tokens_sorted(), module_info),
        )
    }

    #[allow(deprecated)] // The `deprecated` field
    pub fn symbols(&self, handle: &Handle) -> Option<Vec<DocumentSymbol>> {
        let ast = self.get_ast(handle)?;
        let module_info = self.get_module_info(handle)?;
        fn recurse_stmt_adding_symbols<'a>(
            stmt: &'a Stmt,
            symbols: &'a mut Vec<DocumentSymbol>,
            module_info: &Module,
        ) {
            let mut recursed_symbols = Vec::new();
            stmt.recurse(&mut |stmt| {
                recurse_stmt_adding_symbols(stmt, &mut recursed_symbols, module_info)
            });

            match stmt {
                Stmt::FunctionDef(stmt_function_def) => {
                    let mut children = Vec::new();
                    children.append(&mut recursed_symbols);
                    // todo(kylei): better approach to filtering out "" for all symbols
                    let name = match stmt_function_def.name.as_str() {
                        "" => "unknown".to_owned(),
                        name => name.to_owned(),
                    };
                    symbols.push(DocumentSymbol {
                        name,
                        detail: None,
                        kind: lsp_types::SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range: module_info
                            .lined_buffer()
                            .to_lsp_range(stmt_function_def.range),
                        selection_range: module_info
                            .lined_buffer()
                            .to_lsp_range(stmt_function_def.name.range),

                        children: Some(children),
                    });
                }
                Stmt::ClassDef(stmt_class_def) => {
                    let mut children = Vec::new();
                    children.append(&mut recursed_symbols);
                    let name = match stmt_class_def.name.as_str() {
                        "" => "unknown".to_owned(),
                        name => name.to_owned(),
                    };
                    symbols.push(DocumentSymbol {
                        name,
                        detail: None,
                        kind: lsp_types::SymbolKind::CLASS,
                        tags: None,
                        deprecated: None,
                        range: module_info
                            .lined_buffer()
                            .to_lsp_range(stmt_class_def.range),
                        selection_range: module_info
                            .lined_buffer()
                            .to_lsp_range(stmt_class_def.name.range),
                        children: Some(children),
                    });
                }
                _ => {}
            };
            symbols.append(&mut recursed_symbols);
        }
        let mut result = Vec::new();
        ast.body
            .visit(&mut |stmt| recurse_stmt_adding_symbols(stmt, &mut result, &module_info));
        Some(result)
    }

    pub fn workspace_symbols(
        &self,
        query: &str,
    ) -> Option<Vec<(String, lsp_types::SymbolKind, TextRangeWithModule)>> {
        if query.len() < MIN_CHARACTERS_TYPED_AUTOIMPORT {
            return None;
        }
        let mut result = Vec::new();
        for (handle, name, export) in self.search_exports_fuzzy(query) {
            if let Some(module) = self.get_module_info(&handle) {
                let kind = export
                    .symbol_kind
                    .map_or(lsp_types::SymbolKind::VARIABLE, |k| k.to_lsp_symbol_kind());
                let location = TextRangeWithModule {
                    module,
                    range: export.location,
                };
                result.push((name, kind, location));
            }
        }
        Some(result)
    }
}

impl<'a> CancellableTransaction<'a> {
    /// Returns Err if the request is canceled in the middle of a run.
    pub fn find_global_references_from_definition(
        &mut self,
        sys_info: &SysInfo,
        definition_kind: DefinitionMetadata,
        definition: TextRangeWithModule,
    ) -> Result<Vec<(Module, Vec<TextRange>)>, Cancelled> {
        // General strategy:
        // 1: Compute the set of transitive rdeps.
        // 2. Find references in each one of them using the index computed during earlier checking
        let mut transitive_rdeps = match definition.module.path().details() {
            ModulePathDetails::Memory(path_buf) => {
                let handle_of_filesystem_counterpart = Handle::new(
                    definition.module.name(),
                    ModulePath::filesystem(path_buf.clone()),
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
                    ModulePath::filesystem(path_buf.clone()),
                    handle.sys_info().dupe(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
        {
            transitive_rdeps.remove(&fs_counterpart_of_in_memory_handles);
        }
        let candidate_handles_for_references = transitive_rdeps
            .into_iter()
            .sorted_by_key(|h| h.path().dupe())
            .collect::<Vec<_>>();
        let mut global_references = Vec::new();
        for handle in candidate_handles_for_references {
            let definition = match definition.module.path().details() {
                // Special-case for definition inside in-memory file
                // Calling `local_references_from_definition` naively
                // will find no references outside of the in-memory file because
                // file systems don't contain in-memory files.
                ModulePathDetails::Memory(path_buf)
                    // Why do exclude the case of finding references within the same in-memory file?
                    // If we are finding references within the same in-memory file,
                    // then there is no problem for us to use the in-memory definition location.
                    if handle.path() != definition.module.path() =>
                {
                    // Below, we try to patch the definition location to be at the same offset, but
                    // making the path to be filesystem path instead. In this way, in the happy case
                    // where the in-memory content is exactly the same as the filesystem content,
                    // we can successfully find all the references. However, if the content diverge,
                    // then we will miss definitions from other files.
                    //
                    // In general, other than checking the reverse dependency against the in-memory
                    // content, there is not much we can do: the in-memory content can diverge from
                    // the filesystem content in arbitrary ways.
                    let TextRangeWithModule { module, range } = &definition;
                    let module = if let Some(info) = self.as_ref().get_module_info(&Handle::new(
                        module.name(),
                        ModulePath::filesystem(path_buf.clone()),
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
            };
            let references = self
                .as_ref()
                .local_references_from_definition(
                    &handle,
                    definition_kind.clone(),
                    definition.range,
                    definition.module,
                )
                .unwrap_or_default();
            if !references.is_empty()
                && let Some(module_info) = self.as_ref().get_module_info(&handle)
            {
                global_references.push((module_info, references));
            }
        }
        Ok(global_references)
    }
}
