/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::DocumentSymbol;
use lsp_types::ParameterInformation;
use lsp_types::ParameterLabel;
use lsp_types::SemanticToken;
use lsp_types::SignatureHelp;
use lsp_types::SignatureInformation;
use lsp_types::TextEdit;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
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
use starlark_map::ordered_set::OrderedSet;
use starlark_map::small_map::SmallMap;

use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::error::kind::ErrorKind;
use crate::export::definitions::DocString;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::graph::index::Idx;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::short_identifier::ShortIdentifier;
use crate::state::handle::Handle;
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
use crate::types::module::Module;
use crate::types::types::BoundMethodType;
use crate::types::types::Type;

const INITIAL_GAS: Gas = Gas::new(100);
const MIN_CHARACTERS_TYPED_AUTOIMPORT: usize = 3;

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

enum CalleeKind {
    // Function name
    Function(Identifier),
    // Range of the base expr + method name
    Method(TextRange, Identifier),
    Unknown,
}

enum PatternMatchParameterKind {
    // Name defined using `as`
    // ex: `x` in `case ... as x: ...`, or `x` in `case x: ...`
    AsName,
    // Name defined using keyword argument partten
    // ex: `x` in `case Foo(x=1): ...`
    KeywordArgName,
    // Name defined using `*` pattern
    // ex: `x` in `case [*x]: ...`
    StarName,
    // Name defined using `**` pattern
    // ex: `x` in case { ..., **x }: ...
    RestName,
}

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
    FunctionDef,
    /// An identifier appeared as the name of a class.
    /// ex: `x` in `class x(...): ...`
    ClassDef,
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
    /// See [`PatternMatchParaameterKind`] for examples.
    #[expect(dead_code)]
    PatternMatch(PatternMatchParameterKind),
}

struct IdentifierWithContext {
    identifier: Identifier,
    context: IdentifierContext,
}

#[derive(PartialEq, Eq)]
pub enum AnnotationKind {
    #[allow(dead_code)]
    Parameter,
    Return,
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

    fn from_stmt_function_def(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::FunctionDef,
        }
    }

    fn from_stmt_class_def(id: &Identifier) -> Self {
        Self {
            identifier: id.clone(),
            context: IdentifierContext::ClassDef,
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
pub struct FindDefinitionItem {
    pub metadata: DefinitionMetadata,
    pub location: TextRangeWithModuleInfo,
    pub docstring: Option<DocString>,
}

impl<'a> Transaction<'a> {
    fn get_type(&self, handle: &Handle, key: &Key) -> Option<Type> {
        let idx = self.get_bindings(handle)?.key_to_idx(key);
        let ans = self.get_answers(handle)?;
        Some(ans.for_display(ans.get_idx(idx)?.arc_clone_ty()))
    }

    fn get_type_trace(&self, handle: &Handle, range: TextRange) -> Option<Type> {
        let ans = self.get_answers(handle)?;
        Some(ans.for_display(ans.get_type_trace(range)?.arc_clone()))
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
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::StmtFunctionDef(_)), _, _) => {
                // def id(...): ...
                Some(IdentifierWithContext::from_stmt_function_def(id))
            }
            (Some(AnyNodeRef::Identifier(id)), Some(AnyNodeRef::StmtClassDef(_)), _, _) => {
                // class id(...): ...
                Some(IdentifierWithContext::from_stmt_class_def(id))
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
        let callee = callee_at(self.get_ast(handle)?, position);
        match self.identifier_at(handle, position) {
            Some(IdentifierWithContext {
                identifier: id,
                context: IdentifierContext::Expr(_),
            }) => {
                let key = Key::BoundName(ShortIdentifier::new(&id));
                if self.get_bindings(handle)?.is_valid_key(&key) {
                    if let Some(ExprCall {
                        node_index: _,
                        range: _,
                        func,
                        arguments,
                    }) = &callee
                        && func.range() == id.range
                        && let Some(chosen_overload) = self
                            .get_answers(handle)
                            .and_then(|answers| answers.get_chosen_overload_trace(arguments.range))
                    {
                        Some(Type::Callable(Box::new(chosen_overload)))
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
                Some(Type::Module(Module::new(
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
                context: IdentifierContext::FunctionDef,
            }) => {
                // TODO(grievejia): Handle defintions of functions
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::ClassDef,
            }) => {
                // TODO(grievejia): Handle defintions of classes
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Parameter,
            }) => {
                // TODO(grievejia): Handle defintions of params
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::TypeParameter,
            }) => {
                // TODO(grievejia): Handle defintions of type params
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::ExceptionHandler,
            }) => {
                // TODO(grievejia): Handle defintions of exception names
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::PatternMatch(_),
            }) => {
                // TODO(grievejia): Handle defintions of pattern-introduced names
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::KeywordArgument(_),
            }) => {
                // Keyword argument doesn't have a type by itself
                None
            }
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::Attribute { range, .. },
            }) => {
                if let Some(ExprCall {
                    node_index: _,
                    range: _,
                    func,
                    arguments,
                }) = &callee
                    && func.range() == range
                    && let Some(chosen_overload) = self
                        .get_answers(handle)
                        .and_then(|answers| answers.get_chosen_overload_trace(arguments.range))
                {
                    Some(Type::Callable(Box::new(chosen_overload)))
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

    pub fn get_signature_help_at(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<SignatureHelp> {
        let mod_module = self.get_ast(handle)?;
        let mut res = None;
        mod_module.visit(&mut |x| Self::visit_finding_signature_range(x, position, &mut res));
        let (callee_range, call_args_range, arg_index) = res?;
        let answers = self.get_answers(handle)?;
        if let Some((overloads, chosen_overload_index)) =
            answers.get_all_overload_trace(call_args_range)
        {
            let signatures = overloads.into_map(|callable| {
                Self::create_signature_information(Type::Callable(Box::new(callable)), arg_index)
            });
            Some(SignatureHelp {
                signatures,
                active_signature: chosen_overload_index.map(|i| i as u32),
                active_parameter: Some(arg_index as u32),
            })
        } else {
            answers
                .get_type_trace(callee_range)
                .map(|callee_type| SignatureHelp {
                    signatures: vec![Self::create_signature_information(
                        callee_type.arc_clone(),
                        arg_index,
                    )],
                    active_signature: Some(0),
                    active_parameter: Some(arg_index as u32),
                })
        }
    }

    fn create_signature_information(type_: Type, arg_index: usize) -> SignatureInformation {
        let type_ = type_.deterministic_printing();
        let label = format!("{}", type_);
        let (parameters, active_parameter) =
            if let Some(params) = Self::normalize_singleton_function_type_into_params(type_) {
                let active_parameter = if arg_index < params.len() {
                    Some(arg_index as u32)
                } else {
                    None
                };
                (
                    Some(params.map(|param| ParameterInformation {
                        label: ParameterLabel::Simple(format!("{}", param)),
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
        let callable = match type_ {
            Type::Callable(callable) => Some(*callable),
            Type::Function(function) => Some(function.signature),
            Type::BoundMethod(bound_method) => match bound_method.func {
                BoundMethodType::Function(function) => Some(function.signature),
                BoundMethodType::Forall(forall) => Some(forall.body.signature),
                BoundMethodType::Overload(_) => None,
            },
            _ => None,
        }?;
        // We will drop the self parameter for signature help
        if let Params::List(params_list) = callable.params {
            if let Some(Param::PosOnly(Some(name), _, _) | Param::Pos(name, _, _)) =
                params_list.items().first()
                && name.as_str() == "self"
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
        gas: &mut Gas,
    ) -> Option<(Handle, Export)> {
        let mut m = module_name;
        while !gas.stop() {
            let handle = self.import_handle(handle, m, None).ok()?;
            match self.get_exports(&handle).get(&name) {
                Some(ExportLocation::ThisModule(export)) => {
                    return Some((handle.clone(), export.clone()));
                }
                Some(ExportLocation::OtherModule(module)) => m = *module,
                None => return None,
            }
        }
        None
    }

    /// When `jump_through_renamed_import` is true, we will jump through renamed import like
    /// `from foo import bar as baz`, when we try to compute definition of baz.
    /// Otherwise, we will stop at `baz``.
    fn resolve_intermediate_definition(
        &self,
        handle: &Handle,
        intermediate_definition: IntermediateDefinition,
        jump_through_renamed_import: bool,
        mut gas: Gas,
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
                    self.resolve_named_import(handle, module_name, name, &mut gas)?;
                if !jump_through_renamed_import && original_name_range.is_some() {
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
                let handle = self.import_handle(handle, name, None).ok()?;
                let docstring = self.get_module_docstring(&handle);
                Some((
                    handle,
                    Export {
                        location: TextRange::default(),
                        symbol_kind: Some(SymbolKind::Module),
                        docstring,
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
    ) -> Option<(TextRangeWithModuleInfo, Option<DocString>)> {
        match definition {
            AttrDefinition::FullyResolved(text_range_with_module_info) => {
                // TODO(kylei): attribute docstrings
                Some((text_range_with_module_info, None))
            }
            AttrDefinition::PartiallyResolvedImportedModuleAttribute { module_name } => {
                let mut gas = INITIAL_GAS;
                let (handle, export) =
                    self.resolve_named_import(handle, module_name, attr_name.clone(), &mut gas)?;
                let module_info = self.get_module_info(&handle)?;
                Some((
                    TextRangeWithModuleInfo::new(module_info, export.location),
                    export.docstring,
                ))
            }
        }
    }

    fn key_to_export(
        &self,
        handle: &Handle,
        key: &Key,
        jump_through_renamed_import: bool,
        mut gas: Gas,
    ) -> Option<(Handle, Export)> {
        let bindings = self.get_bindings(handle)?;
        let intermediate_definition = key_to_intermediate_definition(&bindings, key, &mut gas)?;
        self.resolve_intermediate_definition(
            handle,
            intermediate_definition,
            jump_through_renamed_import,
            gas,
        )
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
            location: TextRangeWithModuleInfo::new(self.get_module_info(handle)?, identifier.range),
            docstring: None,
        })
    }

    fn find_definition_for_name_def(
        &self,
        handle: &Handle,
        name: &Identifier,
        jump_through_renamed_import: bool,
    ) -> Option<FindDefinitionItem> {
        let def_key = Key::Definition(ShortIdentifier::new(name));
        if !self.get_bindings(handle)?.is_valid_key(&def_key) {
            return None;
        }
        let (
            handle,
            Export {
                location,
                symbol_kind,
                docstring,
            },
        ) = self.key_to_export(handle, &def_key, jump_through_renamed_import, INITIAL_GAS)?;
        let module_info = self.get_module_info(&handle)?;
        let name = Name::new(module_info.code_at(location));
        Some(FindDefinitionItem {
            metadata: DefinitionMetadata::VariableOrAttribute(name, symbol_kind),
            location: TextRangeWithModuleInfo::new(module_info, location),
            docstring,
        })
    }

    fn find_definition_for_name_use(
        &self,
        handle: &Handle,
        name: &Identifier,
        jump_through_renamed_import: bool,
    ) -> Option<FindDefinitionItem> {
        let use_key = Key::BoundName(ShortIdentifier::new(name));
        if !self.get_bindings(handle)?.is_valid_key(&use_key) {
            return None;
        }
        let (
            handle,
            Export {
                location,
                symbol_kind,
                docstring,
            },
        ) = self.key_to_export(handle, &use_key, jump_through_renamed_import, INITIAL_GAS)?;
        Some(FindDefinitionItem {
            metadata: DefinitionMetadata::Variable(symbol_kind),
            location: TextRangeWithModuleInfo::new(self.get_module_info(&handle)?, location),
            docstring,
        })
    }

    fn find_definition_for_attribute(
        &self,
        handle: &Handle,
        base_range: TextRange,
        name: &Identifier,
    ) -> Vec<FindDefinitionItem> {
        if let Some(answers) = self.get_answers(handle)
            && let Some(base_type) = answers.get_type_trace(base_range)
        {
            self.ad_hoc_solve(handle, |solver| {
                let find_definition_for_base_type = |ty: Type| {
                    solver
                        .completions_no_union_intersection(ty, Some(name.id()), false)
                        .into_iter()
                        .find_map(|x| {
                            if &x.name == name.id() {
                                let (definition, docstring) = self.resolve_attribute_definition(
                                    handle,
                                    &x.name,
                                    x.definition?,
                                )?;
                                Some(FindDefinitionItem {
                                    metadata: DefinitionMetadata::Attribute(x.name),
                                    location: definition,
                                    docstring,
                                })
                            } else {
                                None
                            }
                        })
                };
                match base_type.arc_clone() {
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
    ) -> Option<FindDefinitionItem> {
        // TODO: Handle relative import (via ModuleName::new_maybe_relative)
        let handle = self.import_handle(handle, module_name, None).ok()?;
        Some(FindDefinitionItem {
            metadata: DefinitionMetadata::Module,
            location: TextRangeWithModuleInfo::new(
                self.get_module_info(&handle)?,
                TextRange::default(),
            ),
            docstring: self.get_module_docstring(&handle),
        })
    }

    fn find_definition_for_keyword_argument(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        callee_kind: &CalleeKind,
    ) -> Vec<FindDefinitionItem> {
        // NOTE(grievejia): There might be a better way to compute this that doesn't require 2 containing node
        // traversal, once we gain access to the callee function def from callee_kind directly.
        let callee_locations = self.get_callee_location(handle, callee_kind);
        if callee_locations.is_empty() {
            return vec![];
        }

        // Group all locations by their containing module, so later we could avoid reparsing
        // the same module multiple times.
        let location_count = callee_locations.len();
        let mut modules_to_ranges: SmallMap<ModuleInfo, Vec<TextRange>> =
            SmallMap::with_capacity(location_count);
        for TextRangeWithModuleInfo { module_info, range } in callee_locations.into_iter() {
            modules_to_ranges
                .entry(module_info)
                .or_default()
                .push(range)
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
                    location: TextRangeWithModuleInfo::new(
                        module_info.dupe(),
                        refined_param_range.unwrap_or(range),
                    ),
                    docstring: None,
                })
            }
        }
        results
    }

    fn get_callee_location(
        &self,
        handle: &Handle,
        callee_kind: &CalleeKind,
    ) -> Vec<TextRangeWithModuleInfo> {
        let defs = match callee_kind {
            CalleeKind::Function(name) => self
                .find_definition_for_name_use(handle, name, true)
                .map_or(vec![], |item| vec![item]),
            CalleeKind::Method(base_range, name) => {
                self.find_definition_for_attribute(handle, *base_range, name)
            }
            CalleeKind::Unknown => vec![],
        };
        defs.into_iter().map(|item| item.location).collect()
    }

    /// Find the definition, metadata and optionally the docstring for the given position.
    pub fn find_definition(
        &self,
        handle: &Handle,
        position: TextSize,
        jump_through_renamed_import: bool,
    ) -> Vec<FindDefinitionItem> {
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
                        self.find_definition_for_name_def(handle, &id, jump_through_renamed_import)
                            .map_or(vec![], |item| vec![item])
                    }
                    ExprContext::Load | ExprContext::Del | ExprContext::Invalid => {
                        // This is a usage of the variable
                        self.find_definition_for_name_use(handle, &id, jump_through_renamed_import)
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
                .find_definition_for_imported_module(handle, module_name)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier: _,
                context:
                    IdentifierContext::ImportedName {
                        name_after_import, ..
                    },
            }) => self
                .find_definition_for_name_def(
                    handle,
                    &name_after_import,
                    jump_through_renamed_import,
                )
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::FunctionDef,
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Function)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ClassDef,
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Class)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::Parameter,
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Parameter)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::TypeParameter,
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::TypeParameter)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::ExceptionHandler | IdentifierContext::PatternMatch(_),
            }) => self
                .find_definition_for_simple_def(handle, &identifier, SymbolKind::Variable)
                .map_or(vec![], |item| vec![item]),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::KeywordArgument(callee_kind),
            }) => self.find_definition_for_keyword_argument(handle, &identifier, &callee_kind),
            Some(IdentifierWithContext {
                identifier,
                context: IdentifierContext::Attribute { base_range, .. },
            }) => self.find_definition_for_attribute(handle, base_range, &identifier),
            None => vec![],
        }
    }

    pub fn goto_definition(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Vec<TextRangeWithModuleInfo> {
        self.find_definition(handle, position, true)
            .into_map(|item| item.location)
    }

    /// This function should not be used for user-facing go-to-definition. However, it is exposed to
    /// tests so that we can test the behavior that's useful for find-refs.
    #[cfg(test)]
    pub(crate) fn goto_definition_do_not_jump_through_renamed_import(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<TextRangeWithModuleInfo> {
        self.find_definition(handle, position, false)
            .into_iter()
            .next()
            .map(|item| item.location)
    }

    /// Produce code actions that makes edits local to the file.
    pub fn local_quickfix_code_actions(
        &self,
        handle: &Handle,
        range: TextRange,
    ) -> Option<Vec<(String, ModuleInfo, TextRange, String)>> {
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
                            let (position, insert_text) =
                                insert_import_edit(&ast, handle_to_import_from, unknown_name);
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
        self.find_definition(handle, position, false)
            .into_iter()
            .filter_map(
                |FindDefinitionItem {
                     metadata,
                     location,
                     docstring: _,
                 }| {
                    self.local_references_from_definition(handle, metadata, location)
                },
            )
            .concat()
    }

    fn local_references_from_definition(
        &self,
        handle: &Handle,
        definition_metadata: DefinitionMetadata,
        definition: TextRangeWithModuleInfo,
    ) -> Option<Vec<TextRange>> {
        if handle.path() != definition.module_info.path() {
            let index = self.get_solutions(handle)?.get_index()?;
            let index = index.lock();
            let mut references = Vec::new();
            for ((imported_module_name, imported_name), ranges) in index
                .externally_defined_variable_references
                .iter()
                .chain(&index.renamed_imports)
            {
                let mut gas = INITIAL_GAS;
                if let Some((imported_handle, export)) = self.resolve_named_import(
                    handle,
                    *imported_module_name,
                    imported_name.clone(),
                    &mut gas,
                ) && imported_handle.path().as_path() == definition.module_info.path().as_path()
                    && export.location == definition.range
                {
                    references.extend(ranges.iter().copied());
                }
            }
            for (attribute_module_path, def_and_ref_ranges) in
                &index.externally_defined_attribute_references
            {
                if attribute_module_path == definition.module_info.path() {
                    for (def_range, ref_range) in def_and_ref_ranges {
                        if def_range == &definition.range {
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
            DefinitionMetadata::Attribute(expected_name) => {
                self.local_attribute_references_from_definition(handle, &definition, &expected_name)
            }
            DefinitionMetadata::Module => Vec::new(),
            DefinitionMetadata::Variable(_) => self
                .local_variable_references_from_definition(handle, &definition)
                .unwrap_or_default(),
            DefinitionMetadata::VariableOrAttribute(expected_name, _) => [
                self.local_attribute_references_from_definition(
                    handle,
                    &definition,
                    &expected_name,
                ),
                self.local_variable_references_from_definition(handle, &definition)
                    .unwrap_or_default(),
            ]
            .concat(),
        };
        if definition.module_info.path() == handle.path() {
            references.push(definition.range);
        }
        references.sort_by_key(|range| range.start());
        references.dedup();
        Some(references)
    }

    fn local_attribute_references_from_definition(
        &self,
        handle: &Handle,
        definition: &TextRangeWithModuleInfo,
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
                        definition: attribute_definition,
                    } in solver.completions(base_type.arc_clone(), Some(expected_name), false)
                    {
                        if let Some((
                            TextRangeWithModuleInfo {
                                module_info: module,
                                range,
                            },
                            _,
                        )) = attribute_definition.and_then(|definition| {
                            self.resolve_attribute_definition(handle, &name, definition)
                        }) && module.path() == definition.module_info.path()
                            && range == definition.range
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
        definition: &TextRangeWithModuleInfo,
    ) -> Option<Vec<TextRange>> {
        let bindings = self.get_bindings(handle)?;
        let mut references = Vec::new();
        for NamedBinding {
            definition_handle,
            definition_export,
            key,
        } in self.named_bindings(handle, &bindings)
        {
            if definition_handle.path() == definition.module_info.path()
                && definition.range == definition_export.location
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
            if let Key::Phi(..) = key {
                // Phi keys are always synthetic and never serves as a name definition.
                continue;
            }
            if let Some((definition_handle, definition_export)) =
                self.key_to_export(handle, key, false, INITIAL_GAS)
            {
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
        let mod_module = match self.get_ast(handle) {
            Some(module) => module,
            None => return,
        };

        let mut call_context = None;
        mod_module
            .visit(&mut |x| Self::visit_finding_signature_range(x, position, &mut call_context));
        let (callee_range, _call_args_range, _arg_index) = match call_context {
            Some(context) => context,
            None => return,
        };

        let answers = match self.get_answers(handle) {
            Some(answers) => answers,
            None => return,
        };
        let callee_type = match answers.get_type_trace(callee_range) {
            Some(ty) => ty,
            None => return,
        };

        let params =
            match Self::normalize_singleton_function_type_into_params(callee_type.arc_clone()) {
                Some(params) => params,
                None => return,
            };

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

    pub fn completion(&self, handle: &Handle, position: TextSize) -> Vec<CompletionItem> {
        let mut results = self.completion_unsorted_opt(handle, position);
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
        results.sort_by(|item1, item2| item1.sort_text.cmp(&item2.sort_text));
        results.dedup_by(|item1, item2| item1.label == item2.label && item1.detail == item2.detail);
        results
    }

    fn completion_unsorted_opt(&self, handle: &Handle, position: TextSize) -> Vec<CompletionItem> {
        let mut result = Vec::new();
        self.add_kwargs_completions(handle, position, &mut result);

        match self.identifier_at(handle, position) {
            Some(IdentifierWithContext {
                identifier: _,
                context: IdentifierContext::ImportedName { module_name, .. },
            }) => {
                // TODO: Handle relative import (via ModuleName::new_maybe_relative)
                if let Ok(handle) = self.import_handle(handle, module_name, None) {
                    let exports = self.get_exports(&handle);
                    for name in exports.keys() {
                        result.push(CompletionItem {
                            label: name.to_string(),
                            // todo(kylei): completion kind for exports
                            kind: Some(CompletionItemKind::VARIABLE),
                            ..Default::default()
                        })
                    }
                }
            }
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
                if let Some(answers) = self.get_answers(handle) {
                    if let Some(base_type) = answers.get_type_trace(base_range) {
                        self.ad_hoc_solve(handle, |solver| {
                            solver
                                .completions(base_type.arc_clone(), None, true)
                                .iter()
                                .for_each(|x| {
                                    let kind = match x.ty {
                                        Some(Type::BoundMethod(_)) => {
                                            Some(CompletionItemKind::METHOD)
                                        }
                                        Some(Type::Function(_)) => {
                                            Some(CompletionItemKind::FUNCTION)
                                        }
                                        _ => Some(CompletionItemKind::FIELD),
                                    };
                                    result.push(CompletionItem {
                                        label: x.name.as_str().to_owned(),
                                        detail: x.ty.clone().map(|t| t.to_string()),
                                        kind,
                                        ..Default::default()
                                    });
                                });
                        });
                    }
                }
            }
            Some(IdentifierWithContext { identifier, .. }) => {
                if let Some(bindings) = self.get_bindings(handle) {
                    if let Some(module_info) = self.get_module_info(handle) {
                        bindings
                            .available_definitions(position)
                            .into_iter()
                            .for_each(|idx| {
                                let key = bindings.idx_to_key(idx);
                                if let Key::Definition(id) = key {
                                    let binding = bindings.get(idx);
                                    let detail = self.get_type(handle, key).map(|t| t.to_string());
                                    result.push(CompletionItem {
                                        label: module_info.code_at(id.range()).to_owned(),
                                        detail,
                                        kind: binding
                                            .symbol_kind()
                                            .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                                                Some(k.to_lsp_completion_item_kind())
                                            }),
                                        ..Default::default()
                                    })
                                }
                            });
                        // We should not try to generate autoimport when the user has typed very few
                        // characters. It's unhelpful to narrow down suggestions.
                        if identifier.as_str().len() >= MIN_CHARACTERS_TYPED_AUTOIMPORT
                            && let Some(ast) = self.get_ast(handle)
                        {
                            for (handle_to_import_from, name, export) in
                                self.search_exports_fuzzy(identifier.as_str())
                            {
                                let (insert_text, additional_text_edits) =
                                    match handle_to_import_from.module().as_str() {
                                        "builtins" => (None, None),
                                        _ => {
                                            let (position, insert_text) = insert_import_edit(
                                                &ast,
                                                handle_to_import_from,
                                                &name,
                                            );
                                            let import_text_edit = TextEdit {
                                                range: module_info.lined_buffer().to_lsp_range(
                                                    TextRange::at(position, TextSize::new(0)),
                                                ),
                                                new_text: insert_text.clone(),
                                            };
                                            (Some(insert_text), Some(vec![import_text_edit]))
                                        }
                                    };
                                result.push(CompletionItem {
                                    label: name,
                                    detail: insert_text,
                                    kind: export
                                        .symbol_kind
                                        .map_or(Some(CompletionItemKind::VARIABLE), |k| {
                                            Some(k.to_lsp_completion_item_kind())
                                        }),
                                    additional_text_edits,
                                    ..Default::default()
                                });
                            }
                        }
                    }
                }
            }
            None => {}
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
    ) -> Vec<(ModuleInfo, Vec<TextRange>)> {
        if let Key::Definition(id) = bindings.idx_to_key(idx) {
            if let Some(module_info) = self.get_module_info(handle) {
                let definition_kind = DefinitionMetadata::VariableOrAttribute(
                    Name::from(module_info.code_at(id.range())),
                    None,
                );
                if let Ok(references) = transaction.find_global_references_from_definition(
                    handle.sys_info(),
                    definition_kind,
                    TextRangeWithModuleInfo::new(module_info, id.range()),
                ) {
                    return references;
                }
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
                function_arguments
                    .into_iter()
                    .zip(zipped_inferred_types)
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
                        let binding_func = bindings.get(*key_function);
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

    pub fn inferred_types(&self, handle: &Handle) -> Option<Vec<(TextSize, Type, AnnotationKind)>> {
        let is_interesting_type = |x: &Type| !x.is_error();

        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                // Return Annotation
                key @ Key::ReturnType(id) => {
                    match bindings.get(bindings.key_to_idx(&Key::Definition(id.clone()))) {
                        Binding::Function(x, _pred, _class_meta) => {
                            if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                && let Some(ty) = self.get_type(handle, key)
                                && is_interesting_type(&ty)
                            {
                                let fun = bindings.get(*x);
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
                _ => {}
            }
        }
        Some(res)
    }

    pub fn inlay_hints(&self, handle: &Handle) -> Option<Vec<(TextSize, String)>> {
        let is_interesting_type = |x: &Type| !x.is_error();
        let is_interesting_expr = |x: &Expr| !Ast::is_literal(x);

        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                key @ Key::ReturnType(id) => {
                    match bindings.get(bindings.key_to_idx(&Key::Definition(id.clone()))) {
                        Binding::Function(x, _pred, _class_meta) => {
                            if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                && let Some(ty) = self.get_type(handle, key)
                                && is_interesting_type(&ty)
                            {
                                let fun = bindings.get(*x);
                                res.push((fun.def.parameters.range.end(), format!(" -> {ty}")));
                            }
                        }
                        _ => {}
                    }
                }
                key @ Key::Definition(_) if let Some(ty) = self.get_type(handle, key) => {
                    let e = match bindings.get(idx) {
                        Binding::NameAssign(_, None, e) => Some(&**e),
                        Binding::Expr(None, e) => Some(e),
                        _ => None,
                    };
                    if let Some(e) = e
                        && is_interesting_expr(e)
                        && is_interesting_type(&ty)
                    {
                        let ty = format!(": {}", ty);
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
        builder.process_ast(&ast);
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
            module_info: &ModuleInfo,
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
}

impl<'a> CancellableTransaction<'a> {
    /// Returns Err if the request is canceled in the middle of a run.
    pub fn find_global_references_from_definition(
        &mut self,
        sys_info: &SysInfo,
        definition_kind: DefinitionMetadata,
        definition: TextRangeWithModuleInfo,
    ) -> Result<Vec<(ModuleInfo, Vec<TextRange>)>, Cancelled> {
        // General strategy:
        // 1: Compute the set of transitive rdeps.
        // 2. Find references in each one of them using the index computed during earlier checking
        let mut transitive_rdeps = match definition.module_info.path().details() {
            ModulePathDetails::Memory(path_buf) => {
                let handle_of_filesystem_counterpart = Handle::new(
                    definition.module_info.name(),
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
                    definition.module_info.name(),
                    definition.module_info.path().dupe(),
                    sys_info.dupe(),
                ));
                rdeps
            }
            _ => {
                let definition_handle = Handle::new(
                    definition.module_info.name(),
                    definition.module_info.path().dupe(),
                    sys_info.dupe(),
                );
                let rdeps = self.as_ref().get_transitive_rdeps(definition_handle.dupe());
                // We still need to know everything about the definition file, because the index
                // only contains non-local references.
                self.run(&[(definition_handle, Require::Everything)])?;
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
            let definition = match definition.module_info.path().details() {
                // Special-case for definition inside in-memory file
                // Calling `local_references_from_definition` naively
                // will find no references outside of the in-memory file because
                // file systems don't contain in-memory files.
                ModulePathDetails::Memory(path_buf)
                    // Why do exclude the case of finding references within the same in-memory file?
                    // If we are finding references within the same in-memory file, 
                    // then there is no problem for us to use the in-memory definition location.
                    if handle.path() != definition.module_info.path() =>
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
                    let TextRangeWithModuleInfo { module_info, range } = &definition;
                    let module_info = if let Some(info) = self.as_ref().get_module_info(&Handle::new(
                        module_info.name(),
                        ModulePath::filesystem(path_buf.clone()),
                        handle.sys_info().dupe(),
                    )) {
                        info
                    } else {
                        module_info.dupe()
                    };
                    TextRangeWithModuleInfo {
                        module_info,
                        range: *range,
                    }
                }
                _ => definition.clone(),
            };
            let references = self
                .as_ref()
                .local_references_from_definition(&handle, definition_kind.clone(), definition)
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
