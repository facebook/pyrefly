/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Query interface for pyrefly. Just experimenting for the moment - not intended for external use.

use core::panic;
use std::cell::RefCell;
use std::collections::HashMap;
use std::iter;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use dashmap::DashMap;
use dupe::Dupe;
use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::qname::QName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::PrefixParam;
use pyrefly_types::callable::PropertyRole;
use pyrefly_types::callable_residual::CallableResidualKind;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassFields;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::literal::Lit;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::NeverStyle;
use pyrefly_types::types::SuperObj;
use pyrefly_types::types::Type;
use pyrefly_types::types::Union;
use pyrefly_util::display::Fmt;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::lined_buffer::PythonASTRange;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Arguments;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_source_file::OneIndexed;
use ruff_source_file::PositionEncoding;
use ruff_source_file::SourceLocation;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Serialize;
use starlark_map::Hashed;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::Answers;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::binding::KeyTParams;
use crate::binding::bindings::Bindings;
use crate::config::finder::ConfigFinder;
use crate::error::error::ErrorRenderer;
use crate::module::module_info::ModuleInfo;
use crate::state::load::FileContents;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;
use crate::types::display::LspDisplayMode;
use crate::types::display::TypeDisplayContext;

mod type_table;
pub use ruff_python_ast::Decorator as TypeQueryDecorator;
pub use ruff_python_ast::Expr as TypeQueryExpr;
pub use ruff_python_ast::Stmt as TypeQueryStmt;
pub use ruff_python_ast::StmtClassDef as TypeQueryStmtClassDef;
pub use ruff_python_ast::StmtFunctionDef as TypeQueryStmtFunctionDef;
pub use type_table::IndexedTypeShapeKind;
pub use type_table::LocatedTypeTableRef;
pub use type_table::SerializedTypeTableEntry;
use type_table::TypeTableBuilder;
pub use type_table::TypeTableResponseData;
use type_table::located_type_table_refs;
use type_table::type_to_indexed_shape;

const REPR: Name = Name::new_static("__repr__");

/// Cache for type resolution to avoid expensive re-typechecking.
/// Thread-safe via DashMap for concurrent query access.
struct TypeCache {
    // Maps type_string -> Type
    // Key is just the type string (e.g., "int", "typing.List[str]")
    // since module name/path are constant per session
    cache: DashMap<String, Type>,
}

impl TypeCache {
    fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }

    /// Get a cached type by its string representation
    fn get(&self, type_string: &str) -> Option<Type> {
        self.cache
            .get(type_string)
            .map(|entry| entry.value().clone())
    }

    /// Insert a type into the cache
    fn insert(&self, type_string: String, ty: Type) {
        self.cache.insert(type_string, ty);
    }

    /// Clear all cached types (called on file changes)
    fn clear(&self) {
        self.cache.clear();
    }
}

pub struct Query {
    /// The state that we use.
    state: State,
    /// The SysInfo, the same for all handles.
    sys_info: SysInfo,
    /// The files that have been used with `add_files`, used when files change.
    files: Mutex<SmallSet<(ModuleName, ModulePath)>>,
    /// Cache for type resolution
    type_cache: TypeCache,
}

const CALLEE_KIND_FUNCTION: &str = "function";
const CALLEE_KIND_METHOD: &str = "method";
const CALLEE_KIND_CLASSMETHOD: &str = "classmethod";
const CALLEE_KIND_STATICMETHOD: &str = "staticmethod";

pub type TypeQueryExprVisitor<'a> = dyn FnMut(&'a Expr, Option<&'a Expr>) + 'a;
pub type TypeQueryStmtWalker = dyn for<'a> Fn(&'a [Stmt], &mut TypeQueryExprVisitor<'a>);

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Callee {
    /// What kind of callable is this? Distinguishes various kinds of methods from normal functions.
    pub kind: String,
    /// What's the qualified name of the callable. The name `target` is for Pyre compatibility and originates in Pysa vocabulary.
    pub target: String,
    /// If this is a method, what class is it defined on?
    pub class_name: Option<String>,
}

pub struct Attribute {
    pub name: String,
    pub kind: Option<String>,
    pub annotation: String,
    pub is_final: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TypeQueryTiming {
    pub located_count: usize,
    pub setup: Duration,
    pub stringify: Duration,
    pub cache: Duration,
    pub transform: Duration,
    pub total: Duration,
    pub shape_profiles: TypeQueryProfileMap,
}

#[derive(Debug, Clone, Default)]
pub struct TypeQueryProfile {
    pub count: usize,
    pub total: Duration,
}

pub type TypeQueryProfileMap = HashMap<&'static str, TypeQueryProfile>;

#[derive(Debug, Clone, Serialize)]
pub struct TypeShape {
    /// Backwards-compatible string form of this type.
    pub display: String,
    #[serde(flatten)]
    pub kind: TypeShapeKind,
}

/// Structured client-facing categories for Pyrefly types.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TypeShapeKind {
    /// A named type, optionally with type arguments.
    Named {
        name: String,
        args: Vec<TypeShape>,
        #[serde(skip_serializing_if = "Option::is_none")]
        unspecified_type_arg_count: Option<usize>,
        /// Direct traits carried by the outer Pyrefly `Type` variant. These
        /// are not populated from class metadata for definitions or type
        /// objects.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        traits: Vec<TypeShapeTrait>,
    },
    /// A callable type represented by parameter types and return type.
    Callable {
        params: Vec<TypeShape>,
        return_type: Box<TypeShape>,
        /// Whether the callable is a `@staticmethod`. Structured consumers use
        /// this to recover the `typing.StaticMethod[...]` wrapper that is
        /// otherwise only present in the `display` string, so a value-position
        /// reference to a static method (e.g. a `key=` callback) keeps its
        /// static-method identity.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_staticmethod: bool,
    },
    /// A type parameter, with any bound or constraint types attached.
    TypeVariable {
        name: String,
        bounds: Vec<TypeShape>,
    },
}

/// Traits surfaced for collapsed `named` shapes when the outer Pyrefly `Type`
/// already carries a value-shape property.
///
/// These traits are emitted for value shapes such as `Type::Tuple`,
/// `Type::TypedDict`, and `Type::PartialTypedDict`.
/// They are not synthesized from class metadata, and therefore are not emitted
/// for class/type-position shapes that would require metadata lookup.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeShapeTrait {
    /// A non-partial typed dict value shape.
    TypedDict,
    /// A partial typed dict value shape. This is mutually exclusive with
    /// `TypedDict`; callers checking for any typed dict should check both
    /// traits.
    PartialTypedDict,
    /// A tuple value shape.
    Tuple,
}

/// Thin wrapper around `LinedBuffer::python_ast_range_for_expr` that accepts
/// a `ModuleInfo` for convenience. Callers with direct access to a
/// `LinedBuffer` can call the method directly.
pub fn python_ast_range_for_expr(
    module_info: &ModuleInfo,
    original_range: TextRange,
    expr: &Expr,
    parent_expr: Option<&Expr>,
) -> PythonASTRange {
    module_info
        .lined_buffer()
        .python_ast_range_for_expr(original_range, expr, parent_expr)
}

fn is_static_method(ty: &Type) -> bool {
    match ty {
        Type::Union(u) => u.members.iter().all(is_static_method),
        Type::BoundMethod(m) => m.func.metadata().flags.is_staticmethod,
        Type::Function(f) => f.metadata.flags.is_staticmethod,
        Type::Forall(f) => {
            if let Forallable::Function(func) = &f.body {
                func.metadata.flags.is_staticmethod
            } else {
                false
            }
        }
        _ => false,
    }
}

fn bound_of_type_var(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Quantified(q) | Type::QuantifiedValue(q)
            if q.is_type_var()
                && let Restriction::Bound(bound) = &q.restriction =>
        {
            Some(bound)
        }
        _ => None,
    }
}

fn bound_of_type_var_decl(ty: &Type) -> Option<(&QName, &Type)> {
    if let Type::TypeVar(tv) = ty
        && let Restriction::Bound(bound) = tv.restriction()
    {
        Some((tv.qname(), bound))
    } else {
        None
    }
}

fn query_type_display_context<'a>(types: &[&'a Type]) -> TypeDisplayContext<'a> {
    let mut ctx = TypeDisplayContext::new(types);
    ctx.always_display_module_name();
    ctx.always_display_expanded_unions();
    ctx.set_lsp_display_mode(LspDisplayMode::Query);
    ctx
}

fn type_to_string(ty: &Type) -> String {
    let ctx = query_type_display_context(&[ty]);
    if is_static_method(ty) {
        format!("typing.StaticMethod[{}]", ctx.display(ty))
    } else if let Some(bound) = bound_of_type_var(ty) {
        // pyre1 compatibility: return bound for type variable
        format!(
            "Variable[{} (bound to {})]",
            ctx.display(ty),
            type_to_string(bound)
        )
    } else if let Some((qname, bound)) = bound_of_type_var_decl(ty) {
        // pyre1 compatibility: return bound for type variable
        format!(
            "Variable[{} (bound to {})]",
            qname.id(),
            type_to_string(bound)
        )
    } else {
        ctx.display(ty).to_string()
    }
}

struct TypeShapeContext<'a> {
    transaction: &'a Transaction<'a>,
    source_handle: &'a Handle,
    profile: Option<&'a Mutex<TypeQueryProfileMap>>,
}

impl TypeShapeContext<'_> {
    fn declared_type_param_arity_for_class(&self, class: &Class) -> Option<usize> {
        if let Some(tparams) = class.precomputed_tparams() {
            return nonzero_arity(tparams.len());
        }

        let handle = Handle::new(
            class.module_name(),
            class.module_path().dupe(),
            self.source_handle.sys_info().dupe(),
        );
        let bindings = self.transaction.get_bindings(&handle)?;
        let answers = self.transaction.get_answers(&handle)?;
        let idx = bindings.key_to_idx_hashed_opt(Hashed::new(&KeyTParams(class.index())))?;
        nonzero_arity(answers.get_idx(idx)?.len())
    }
}

fn type_shape_from(context: &TypeShapeContext, ty: &Type, display: String) -> TypeShape {
    if let Some(profile) = context.profile {
        let kind_name = type_kind_name(ty);
        let start = Instant::now();
        let kind = type_shape_kind(context, ty);
        let elapsed = start.elapsed();
        let mut profile = profile.lock();
        let entry = profile.entry(kind_name).or_default();
        entry.count += 1;
        entry.total += elapsed;
        TypeShape { display, kind }
    } else {
        TypeShape {
            display,
            kind: type_shape_kind(context, ty),
        }
    }
}

fn type_to_shape(context: &TypeShapeContext, ty: &Type) -> TypeShape {
    type_shape_from(context, ty, type_to_string(ty))
}

fn type_kind_name(ty: &Type) -> &'static str {
    match ty {
        Type::ClassDef(_) => "class_def",
        Type::ClassType(_) => "class_type",
        Type::TypedDict(_) => "typed_dict",
        Type::PartialTypedDict(_) => "partial_typed_dict",
        Type::Type(_) => "type",
        Type::Callable(_) => "callable",
        Type::Function(_) => "function",
        Type::BoundMethod(_) => "bound_method",
        Type::Overload(_) => "overload",
        Type::Forall(_) => "forall",
        Type::Union(_) => "union",
        Type::Intersect(_) => "intersect",
        Type::Tuple(_) => "tuple",
        Type::Literal(_) => "literal",
        Type::Sentinel(_) => "sentinel",
        Type::LiteralString(_) => "literal_string",
        Type::Quantified(_) => "quantified",
        Type::QuantifiedValue(_) => "quantified_value",
        Type::TypeVar(_) => "type_var",
        Type::ParamSpec(_) => "param_spec",
        Type::TypeVarTuple(_) => "type_var_tuple",
        Type::ElementOfTypeVarTuple(_) => "element_of_type_var_tuple",
        Type::TypeGuard(_) => "type_guard",
        Type::TypeIs(_) => "type_is",
        Type::Annotated(_, _) => "annotated",
        Type::Unpack(_) => "unpack",
        Type::Concatenate(_, _) => "concatenate",
        Type::ParamSpecValue(_) => "param_spec_value",
        Type::Args(_) => "args",
        Type::ArgsValue(_) => "args_value",
        Type::Kwargs(_) => "kwargs",
        Type::KwargsValue(_) => "kwargs_value",
        Type::Module(_) => "module",
        Type::TypeAlias(_) => "type_alias",
        Type::UntypedAlias(_) => "untyped_alias",
        Type::SuperInstance(_) => "super_instance",
        Type::SelfType(_) => "self_type",
        Type::CallableResidual(_) => "callable_residual",
        Type::KwCall(_) => "kw_call",
        Type::Any(_) => "any",
        Type::Never(_) => "never",
        Type::None => "none",
        Type::SpecialForm(_) => "special_form",
        Type::Ellipsis => "ellipsis",
        Type::Materialization => "materialization",
        Type::Var(_) => "var",
        Type::ShapedArray(_) => "shaped_array",
        Type::IntTuple(_) => "int_tuple",
        Type::NNModule(_) => "nn_module",
        Type::DataFrame(_) => "data_frame",
        Type::Int(_) => "int",
        Type::TypeForm(_) => "type_form",
    }
}

fn type_shape_kind(context: &TypeShapeContext, ty: &Type) -> TypeShapeKind {
    match ty {
        Type::ClassDef(cls) => named_type_shape_kind(
            "typing.Type",
            vec![{
                let name = qname_to_string(cls.qname());
                TypeShape {
                    display: name.clone(),
                    kind: named_type_shape_kind_with_unspecified_type_arg_count(
                        name,
                        Vec::new(),
                        context.declared_type_param_arity_for_class(cls),
                    ),
                }
            }],
        ),
        Type::ClassType(class_type) => {
            let args = class_type
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_shape(context, ty))
                .collect::<Vec<_>>();
            named_type_shape_kind(qname_to_string(class_type.qname()), args)
        }
        Type::TypedDict(typed_dict) => typed_dict_shape(context, typed_dict, false),
        Type::PartialTypedDict(typed_dict) => typed_dict_shape(context, typed_dict, true),
        Type::Type(inner) => {
            named_type_shape_kind("typing.Type", vec![type_to_shape(context, inner)])
        }
        Type::Callable(callable) => callable_shape(context, callable, false),
        Type::Function(function) => callable_shape(
            context,
            &function.signature,
            function.metadata.flags.is_staticmethod,
        ),
        Type::BoundMethod(bound_method) => {
            let function_type = bound_method.func.clone().as_type();
            named_type_shape_kind(
                "BoundMethod",
                vec![
                    type_to_shape(context, &bound_method.obj),
                    type_to_shape(context, &function_type),
                ],
            )
        }
        Type::Overload(overload) => named_type_shape_kind(
            "typing.Overload",
            overload
                .signatures
                .iter()
                .map(|signature| type_to_shape(context, &signature.as_type()))
                .collect(),
        ),
        Type::Forall(forall) => {
            // The wrapper binds generic parameters; the variables still surface
            // where they appear inside the structured body shape.
            let body_type = forall.body.clone().as_type();
            type_shape_kind(context, &body_type)
        }
        Type::Union(union) => union_shape(context, union),
        Type::Intersect(intersection) => {
            let (members, _fallback) = &**intersection;
            named_type_shape_kind(
                "Intersection",
                members
                    .iter()
                    .map(|ty| type_to_shape(context, ty))
                    .collect(),
            )
        }
        Type::Tuple(tuple) => named_type_shape_kind_with_traits(
            "typing.Tuple",
            tuple_args(context, tuple),
            None,
            vec![TypeShapeTrait::Tuple],
        ),
        Type::Literal(literal) => named_type_shape_kind(
            "typing.Literal",
            vec![named_leaf(literal_value_shape_name(&literal.value))],
        ),
        Type::Sentinel(sentinel) => {
            named_type_shape_kind("sentinel", vec![named_leaf(format!("{}", sentinel))])
        }
        Type::LiteralString(_) => {
            named_type_shape_kind("typing_extensions.LiteralString", Vec::new())
        }
        Type::Quantified(quantified) | Type::QuantifiedValue(quantified) => {
            quantified_variable_shape(context, quantified)
        }
        Type::TypeVar(type_var) => TypeShapeKind::TypeVariable {
            name: type_var.qname().id().to_string(),
            bounds: restriction_bounds(context, type_var.restriction()),
        },
        Type::ParamSpec(param_spec) => TypeShapeKind::TypeVariable {
            name: param_spec.qname().id().to_string(),
            bounds: Vec::new(),
        },
        Type::TypeVarTuple(type_var_tuple) => TypeShapeKind::TypeVariable {
            name: type_var_tuple.qname().id().to_string(),
            bounds: Vec::new(),
        },
        Type::ElementOfTypeVarTuple(quantified) => TypeShapeKind::TypeVariable {
            name: quantified.name.to_string(),
            bounds: Vec::new(),
        },
        Type::TypeGuard(inner) => {
            named_type_shape_kind("typing.TypeGuard", vec![type_to_shape(context, inner)])
        }
        Type::TypeIs(inner) => {
            named_type_shape_kind("typing.TypeIs", vec![type_to_shape(context, inner)])
        }
        Type::Annotated(inner, metadata) => {
            let mut args = vec![type_to_shape(context, inner)];
            args.extend(metadata.iter().map(|ty| type_to_shape(context, ty)));
            named_type_shape_kind("typing.Annotated", args)
        }
        Type::Unpack(inner) => {
            named_type_shape_kind("typing.Unpack", vec![type_to_shape(context, inner)])
        }
        Type::Concatenate(prefix, param_spec) => {
            let mut args: Vec<TypeShape> = prefix
                .iter()
                .map(|param| prefix_param_to_shape(context, param))
                .collect();
            args.push(type_to_shape(context, param_spec));
            named_type_shape_kind("typing.Concatenate", args)
        }
        Type::ParamSpecValue(params) => {
            named_type_shape_kind("ParamSpecValue", param_list_to_shapes(context, params))
        }
        Type::Args(param_spec) | Type::ArgsValue(param_spec) => {
            named_type_shape_kind("ParamSpecArgs", vec![param_spec_shape(context, param_spec)])
        }
        Type::Kwargs(param_spec) | Type::KwargsValue(param_spec) => named_type_shape_kind(
            "ParamSpecKwargs",
            vec![param_spec_shape(context, param_spec)],
        ),
        Type::Module(module) => {
            named_type_shape_kind("Module", vec![named_leaf(module.to_string())])
        }
        Type::TypeAlias(alias) | Type::UntypedAlias(alias) => alias_shape(context, alias),
        Type::SuperInstance(super_instance) => {
            let (start_class, obj) = &**super_instance;
            let object = match obj {
                SuperObj::Instance(class_type) | SuperObj::Class(class_type) => {
                    Type::ClassType(class_type.clone())
                }
            };
            named_type_shape_kind(
                "super",
                vec![
                    type_to_shape(context, &Type::ClassType(start_class.clone())),
                    type_to_shape(context, &object),
                ],
            )
        }
        Type::SelfType(class_type) => {
            let args = class_type
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_shape(context, ty))
                .collect::<Vec<_>>();
            named_type_shape_kind(qname_to_string(class_type.qname()), args)
        }
        Type::CallableResidual(residual) => match &residual.kind {
            CallableResidualKind::Generic { quantified } => {
                quantified_variable_shape(context, quantified)
            }
            CallableResidualKind::Overload { branches, .. } => named_type_shape_kind(
                "typing.Overload",
                branches
                    .iter()
                    .map(|branch| type_to_shape(context, &branch.ty))
                    .collect(),
            ),
        },
        Type::KwCall(call) => type_shape_kind(context, &call.return_ty),
        Type::Any(_) => named_type_shape_kind("typing.Any", Vec::new()),
        Type::Never(style) => named_type_shape_kind(
            match style {
                NeverStyle::NoReturn => "typing.NoReturn",
                NeverStyle::Never => "typing.Never",
            },
            Vec::new(),
        ),
        Type::None => named_type_shape_kind("None", Vec::new()),
        Type::SpecialForm(special_form) => {
            named_type_shape_kind(special_form.to_string(), Vec::new())
        }
        Type::Ellipsis => named_type_shape_kind("...", Vec::new()),
        Type::Materialization => named_type_shape_kind("Materialization", Vec::new()),
        Type::Var(_) => named_type_shape_kind("typing.Any", Vec::new()),
        Type::ShapedArray(_) => named_type_shape_kind("Tensor", Vec::new()),
        Type::IntTuple(_) => named_type_shape_kind("IntTuple", Vec::new()),
        Type::NNModule(module) => {
            let args = module
                .class
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_shape(context, ty))
                .collect::<Vec<_>>();
            named_type_shape_kind(qname_to_string(module.class.qname()), args)
        }
        Type::DataFrame(schema) => type_shape_kind(context, &schema.underlying_type()),
        Type::Int(_) => named_type_shape_kind("Int", Vec::new()),
        Type::TypeForm(inner) => {
            named_type_shape_kind("typing.TypeForm", vec![type_to_shape(context, inner)])
        }
    }
}

fn named_type_shape_kind(name: impl Into<String>, args: Vec<TypeShape>) -> TypeShapeKind {
    named_type_shape_kind_with_unspecified_type_arg_count(name, args, None)
}

/// Use only for bare generic objects whose declared type parameters are not
/// represented as child `args`.
fn named_type_shape_kind_with_unspecified_type_arg_count(
    name: impl Into<String>,
    args: Vec<TypeShape>,
    unspecified_type_arg_count: Option<usize>,
) -> TypeShapeKind {
    named_type_shape_kind_with_traits(name, args, unspecified_type_arg_count, Vec::new())
}

fn named_type_shape_kind_with_traits(
    name: impl Into<String>,
    args: Vec<TypeShape>,
    unspecified_type_arg_count: Option<usize>,
    traits: Vec<TypeShapeTrait>,
) -> TypeShapeKind {
    TypeShapeKind::Named {
        name: name.into(),
        args,
        unspecified_type_arg_count,
        traits,
    }
}

fn named_type_shape(name: impl Into<String>, args: Vec<TypeShape>) -> TypeShape {
    let name = name.into();
    let display = format_type_application(&name, &args);
    TypeShape {
        display,
        kind: named_type_shape_kind(name, args),
    }
}

fn named_leaf(name: impl Into<String>) -> TypeShape {
    named_type_shape(name, Vec::new())
}

/// Render a literal value for the structured type-shape leaf. Enum members must
/// carry their class's full module-qualified name (matching `qname_to_string`,
/// the `ClassType`/`ClassDef` arms, and the module-qualified display string);
/// the bare `Lit::Display` only emits the short class name. Shared with the
/// type-table path in `query::type_table`.
pub(crate) fn literal_value_shape_name(value: &Lit) -> String {
    match value {
        Lit::Enum(lit_enum) => {
            format!(
                "{}.{}",
                qname_to_string(lit_enum.class.qname()),
                lit_enum.member
            )
        }
        _ => value.to_string(),
    }
}

fn format_type_application(name: &str, args: &[TypeShape]) -> String {
    if args.is_empty() {
        name.to_owned()
    } else {
        format!("{name}[{}]", args.iter().map(|arg| &arg.display).join(", "))
    }
}

fn qname_to_string(qname: &QName) -> String {
    format!("{}", Fmt(|f| qname.fmt_with_module(f)))
}

fn typed_dict_shape(
    context: &TypeShapeContext,
    typed_dict: &TypedDict,
    is_partial: bool,
) -> TypeShapeKind {
    match typed_dict {
        TypedDict::TypedDict(inner) => {
            let args = inner
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_shape(context, ty))
                .collect::<Vec<_>>();
            named_type_shape_kind_with_traits(
                qname_to_string(inner.qname()),
                args,
                None,
                typed_dict_traits(is_partial),
            )
        }
        // An anonymous TypedDict has no class identity; structurally it is a
        // `dict[str, <union of field value types>]` (all keys are string
        // literals). Emit it as that dict -- matching the display string and a
        // real dict -- so the structured shape keeps the field value types
        // instead of collapsing to an opaque `TypedDictionary` marker.
        TypedDict::Anonymous(inner) => {
            let heap = TypeHeap::new();
            let value_type = inner.compute_value_type(&heap);
            named_type_shape_kind(
                "builtins.dict",
                vec![
                    named_leaf("builtins.str"),
                    type_to_shape(context, &value_type),
                ],
            )
        }
    }
}

fn typed_dict_traits(is_partial: bool) -> Vec<TypeShapeTrait> {
    if is_partial {
        vec![TypeShapeTrait::PartialTypedDict]
    } else {
        vec![TypeShapeTrait::TypedDict]
    }
}

fn callable_shape(
    context: &TypeShapeContext,
    callable: &Callable,
    is_staticmethod: bool,
) -> TypeShapeKind {
    TypeShapeKind::Callable {
        params: callable_param_types(context, &callable.params),
        return_type: Box::new(type_to_shape(context, &callable.ret)),
        is_staticmethod,
    }
}

fn callable_param_types(context: &TypeShapeContext, params: &Params) -> Vec<TypeShape> {
    match params {
        Params::List(params) | Params::Partial(params) => param_list_to_shapes(context, params),
        Params::ParamSpec(prefix, param_spec) => prefix
            .iter()
            .map(|param| prefix_param_to_shape(context, param))
            .chain(iter::once(type_to_shape(context, param_spec)))
            .collect(),
        Params::Ellipsis | Params::Materialization => Vec::new(),
    }
}

fn param_list_to_shapes(context: &TypeShapeContext, params: &ParamList) -> Vec<TypeShape> {
    params
        .items()
        .iter()
        .map(|param| param_to_shape(context, param))
        .collect()
}

fn prefix_param_to_shape(context: &TypeShapeContext, param: &PrefixParam) -> TypeShape {
    match param {
        PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => type_to_shape(context, ty),
    }
}

fn param_to_shape(context: &TypeShapeContext, param: &Param) -> TypeShape {
    match param {
        Param::PosOnly(_, ty, _)
        | Param::Pos(_, ty, _)
        | Param::Varargs(_, ty)
        | Param::KwOnly(_, ty, _)
        | Param::Kwargs(_, ty) => type_to_shape(context, ty),
    }
}

fn tuple_args(context: &TypeShapeContext, tuple: &Tuple) -> Vec<TypeShape> {
    match tuple {
        Tuple::Concrete(elements) => elements
            .iter()
            .map(|ty| type_to_shape(context, ty))
            .collect(),
        Tuple::Unbounded(element) => vec![type_to_shape(context, element), named_leaf("...")],
        Tuple::Unpacked(unpacked) => {
            let (prefix, middle, suffix) = &**unpacked;
            prefix
                .iter()
                .map(|ty| type_to_shape(context, ty))
                .chain(iter::once(type_to_shape(context, middle)))
                .chain(suffix.iter().map(|ty| type_to_shape(context, ty)))
                .collect()
        }
    }
}

fn param_spec_shape(context: &TypeShapeContext, param_spec: &Quantified) -> TypeShape {
    debug_assert_eq!(param_spec.kind, QuantifiedKind::ParamSpec);
    let ctx = query_type_display_context(&[]);
    TypeShape {
        display: ctx.display_quantified(param_spec).to_string(),
        kind: quantified_variable_shape(context, param_spec),
    }
}

fn quantified_variable_shape(context: &TypeShapeContext, quantified: &Quantified) -> TypeShapeKind {
    TypeShapeKind::TypeVariable {
        name: quantified.name.to_string(),
        bounds: quantified_restriction_bounds(context, &quantified.restriction),
    }
}

fn quantified_restriction_bounds(
    context: &TypeShapeContext,
    restriction: &Restriction,
) -> Vec<TypeShape> {
    match restriction {
        Restriction::Bound(bound) => vec![type_to_shape(context, bound)],
        Restriction::Constraints(_) | Restriction::Unrestricted => Vec::new(),
    }
}

fn restriction_bounds(context: &TypeShapeContext, restriction: &Restriction) -> Vec<TypeShape> {
    match restriction {
        Restriction::Bound(bound) => vec![type_to_shape(context, bound)],
        Restriction::Constraints(constraints) => constraints
            .iter()
            .map(|ty| type_to_shape(context, ty))
            .collect(),
        Restriction::Unrestricted => Vec::new(),
    }
}

fn alias_shape(context: &TypeShapeContext, alias: &TypeAliasData) -> TypeShapeKind {
    match alias {
        TypeAliasData::Value(alias) => named_type_shape_kind(
            "TypeAlias",
            vec![
                named_leaf(alias.name.to_string()),
                type_to_shape(context, &alias.as_type()),
            ],
        ),
        TypeAliasData::Ref(alias) => named_type_shape_kind(
            format!("{}.{}", alias.module_name, alias.name),
            alias
                .args
                .as_ref()
                .map(|args| {
                    args.as_slice()
                        .iter()
                        .map(|ty| type_to_shape(context, ty))
                        .collect()
                })
                .unwrap_or_default(),
        ),
    }
}

fn union_shape(context: &TypeShapeContext, union: &Union) -> TypeShapeKind {
    if union
        .members
        .iter()
        .any(|member| matches!(member, Type::None))
    {
        let mut members = union
            .members
            .iter()
            .filter(|member| !matches!(member, Type::None))
            .map(|ty| type_to_shape(context, ty))
            .collect::<Vec<_>>();
        let inner = if members.len() == 1 {
            members.pop().unwrap()
        } else {
            named_type_shape("typing.Union", members)
        };
        named_type_shape_kind("typing.Optional", vec![inner])
    } else {
        let members = union
            .members
            .iter()
            .map(|ty| type_to_shape(context, ty))
            .collect();
        named_type_shape_kind("typing.Union", members)
    }
}

fn nonzero_arity(arity: usize) -> Option<usize> {
    (arity > 0).then_some(arity)
}

struct CalleesWithLocation<'a> {
    query: &'a Query,
    transaction: Transaction<'a>,
    handle: Handle,
    module_info: Module,
    ast: Arc<ModModule>,
    answers: Arc<Answers>,
}

impl<'a> CalleesWithLocation<'a> {
    pub fn new(
        query: &'a Query,
        transaction: Transaction<'a>,
        handle: Handle,
    ) -> Option<CalleesWithLocation<'a>> {
        let module_info = transaction.get_module_info(&handle)?;
        let answers = transaction.get_answers(&handle)?;
        let ast: Arc<ModModule> = transaction.get_ast(&handle)?;
        Some(Self {
            query,
            transaction,
            handle,
            module_info,
            ast,
            answers,
        })
    }
    fn _try_unwrap_lru_cache_wrapper(
        &self,
        call: &ExprCall,
        func_ty: &Type,
    ) -> Option<Vec<Callee>> {
        if let Type::ClassType(class) = &func_ty
            && class.name() == "_lru_cache_wrapper"
            && let [def] = self
                .transaction
                .find_definition(
                    &self.handle,
                    call.func
                        .range()
                        .end()
                        .checked_sub(TextSize::from(1))
                        .unwrap(),
                    FindPreference {
                        resolve_call_dunders: false,
                        ..FindPreference::default()
                    },
                )
                .map(Vec1::into_vec)
                .unwrap_or_default()
                .into_iter()
                .collect_vec()
                .as_slice()
        {
            let h = self
                .query
                .make_handle(def.module.name(), def.module.path().clone());
            let module = self.transaction.get_module_info(&h)?;
            let bindings = self.transaction.get_bindings(&h)?;
            let answers = self.transaction.get_answers(&h)?;

            let name = module.code_at(def.definition_range);
            let id = Identifier::new(name, def.definition_range);
            let key = bindings.key_to_idx(&KeyDecoratedFunction(ShortIdentifier::new(&id)));

            // Get the undecorated function using ad_hoc_solve
            let answer = answers.get_idx(bindings.get(key).undecorated_idx)?;
            Some(vec![self.callee_from_function_metadata(
                &answer.metadata,
                None,
                None,
            )])
        } else {
            None
        }
    }
    fn process_expr(&self, x: &Expr, res: &mut Vec<(PythonASTRange, Callee)>) {
        let (callees, callee_range) = match x {
            Expr::Attribute(attr) => {
                let callees =
                    if let Some(func_ty) = self.answers.try_get_getter_for_range(attr.range()) {
                        self.callee_from_type(&func_ty, None, attr.range(), None)
                    } else {
                        vec![]
                    };
                (callees, attr.range())
            }
            Expr::Call(ExprCall { func, .. })
                if let Expr::Name(name) = &**func
                    && name.id() == "prod_assert" =>
            {
                // pyrefly has special treatment for prod_assert but for our purposes we still want to see this call
                let callees = vec![Callee {
                    kind: String::from(CALLEE_KIND_FUNCTION),
                    target: String::from("util.prod_assert"),
                    class_name: None,
                }];
                (callees, name.range())
            }
            Expr::Call(call) => {
                let callees = if let Some(func_ty) = self.answers.get_type_trace(call.func.range())
                {
                    self._try_unwrap_lru_cache_wrapper(call, &func_ty)
                        .unwrap_or_else(|| {
                            self.callee_from_type(
                                &func_ty,
                                Some(&*call.func),
                                call.func.range(),
                                Some(&call.arguments),
                            )
                        })
                } else {
                    vec![]
                };
                (callees, call.func.range())
            }
            _ => (vec![], x.range()),
        };
        for callee in callees {
            res.push((
                python_ast_range_for_expr(&self.module_info, callee_range, x, None),
                callee,
            ));
        }

        x.recurse(&mut |x| self.process_expr(x, res));
    }

    fn add_decorators(&self, decorators: &[Decorator], res: &mut Vec<(PythonASTRange, Callee)>) {
        for dec in decorators {
            if matches!(dec.expression, Expr::Name(_) | Expr::Attribute(_))
                && let Some(call_ty) = self.answers.get_type_trace(dec.expression.range())
            {
                self.callee_from_type(&call_ty, None, dec.expression.range(), None)
                    .into_iter()
                    .for_each(|callee| {
                        res.push((
                            python_ast_range_for_expr(
                                &self.module_info,
                                dec.expression.range(),
                                &dec.expression,
                                None,
                            ),
                            callee,
                        ));
                    });
            }
        }
    }
    fn callee_from_text_range<F: FnMut(Callee)>(
        &self,
        target_range: TextRange,
        call_target: Option<&Expr>,
        mut f: F,
    ) {
        if let Some(func_ty) = self.answers.get_type_trace(target_range) {
            let callees = self.callee_from_type(&func_ty, call_target, target_range, None);
            for callee in callees {
                f(callee);
            }
        }
    }

    pub fn process(&self, location: Option<PythonASTRange>) -> Vec<(PythonASTRange, Callee)> {
        let mut res = Vec::new();

        if let Some(target_location) = location {
            // Helper function to convert line/column to TextSize
            fn line_col_to_text_size(
                module_info: &ModuleInfo,
                line: LineNumber,
                col: u32,
            ) -> TextSize {
                module_info.lined_buffer().line_index().offset(
                    SourceLocation {
                        line: OneIndexed::new(line.get() as usize).unwrap(),
                        character_offset: OneIndexed::from_zero_indexed(col as usize),
                    },
                    module_info.lined_buffer().contents(),
                    PositionEncoding::Utf8,
                )
            }

            // Convert PythonASTRange to TextRange using SourceLocation directly
            let start_pos = line_col_to_text_size(
                &self.module_info,
                target_location.start_line,
                target_location.start_col,
            );
            let end_pos = line_col_to_text_size(
                &self.module_info,
                target_location.end_line,
                target_location.end_col,
            );
            let target_range = TextRange::new(start_pos, end_pos);
            self.callee_from_text_range(target_range, None, |c| {
                res.push((target_location.clone(), c));
            });
        } else {
            for stmt in &self.ast.body {
                match &stmt {
                    Stmt::ClassDef(StmtClassDef {
                        decorator_list: d, ..
                    })
                    | Stmt::FunctionDef(StmtFunctionDef {
                        decorator_list: d, ..
                    }) => {
                        self.add_decorators(d, &mut res);
                    }
                    _ => {}
                }
                stmt.visit(&mut |x| self.process_expr(x, &mut res));
            }
        };
        res
    }

    fn qname_to_string(n: &QName) -> String {
        format!("{}.{}", n.module_name(), n.id())
    }
    fn class_name_from_def_kind(kind: &FunctionKind) -> String {
        if let FunctionKind::Def(f) = kind
            && let Some(cls) = &f.cls
        {
            format!("{}.{}", f.module.name(), cls.name())
        } else if let FunctionKind::CallbackProtocol(c) = kind {
            Self::qname_to_string(c.qname())
        } else {
            panic!("class_name_from_def_kind - unsupported function kind: {kind:?}");
        }
    }
    fn target_from_def_kind(kind: &FunctionKind, module_name_override: Option<&str>) -> String {
        match kind {
            FunctionKind::Def(f) => {
                if let Some(module_name_override) = module_name_override {
                    format!("{module_name_override}.{}", f.name)
                } else {
                    match &f.cls {
                        Some(cls) => {
                            format!("{}.{}.{}", f.module.name(), cls.name(), f.name)
                        }
                        None => {
                            format!("{}.{}", f.module.name(), f.name)
                        }
                    }
                }
            }
            FunctionKind::CallbackProtocol(cls) => {
                format!("{}.__call__", Self::qname_to_string(cls.qname()))
            }

            x => x.format(ModuleName::builtins()),
        }
    }
    fn repr_from_arguments(&self, arguments: &Arguments) -> Option<Callee> {
        // Use the type of the first argument to find the callee.
        if let Some(arg_type) = self.answers.get_type_trace(arguments.args[0].range())
            && let Type::ClassType(class) = &arg_type
        {
            let repr_callees =
                self.callee_from_mro(class.class_object(), "__repr__", |solver, c| {
                    if solver
                        .get_class_fields(c)
                        .is_some_and(|f| f.contains(&REPR))
                    {
                        Some(format!("{}.{}.__repr__", c.module_name(), c.name()))
                    } else {
                        None
                    }
                });
            if !repr_callees.is_empty() {
                return Some(repr_callees[0].clone());
            }
        }
        None
    }
    fn callee_from_function(
        &self,
        f: &Function,
        call_target: Option<&Expr>,
        call_arguments: Option<&Arguments>,
    ) -> Callee {
        self.callee_from_function_metadata(&f.metadata, call_target, call_arguments)
    }
    fn callee_from_function_metadata(
        &self,
        metadata: &FuncMetadata,
        call_target: Option<&Expr>,
        call_arguments: Option<&Arguments>,
    ) -> Callee {
        if metadata.flags.is_staticmethod {
            Callee {
                kind: String::from(CALLEE_KIND_STATICMETHOD),
                target: Self::target_from_def_kind(&metadata.kind, None),
                class_name: Some(Self::class_name_from_def_kind(&metadata.kind)),
            }
        } else if metadata.flags.is_classmethod {
            Callee {
                kind: String::from(CALLEE_KIND_CLASSMETHOD),
                target: Self::target_from_def_kind(&metadata.kind, None),
                // TODO: use type of receiver
                class_name: Some(Self::class_name_from_def_kind(&metadata.kind)),
            }
        } else {
            // Check if this is a builtins function that needs special casing.
            if let FunctionKind::Def(def) = &metadata.kind
                && def.module.name().as_str() == "builtins"
                && def.name == "repr"
                && let Some(args) = call_arguments
                && let Some(callee) = self.repr_from_arguments(args)
            {
                return callee;
            }

            let class_name = self.class_name_from_call_target(call_target);
            let kind = if class_name.is_some() {
                String::from(CALLEE_KIND_METHOD)
            } else {
                String::from(CALLEE_KIND_FUNCTION)
            };

            Callee {
                kind,
                target: Self::target_from_def_kind(&metadata.kind, None),
                class_name,
            }
        }
    }
    fn target_from_bound_method_type(m: &BoundMethodType, method_of_typed_dict: bool) -> String {
        let module_name_override = if method_of_typed_dict {
            Some("TypedDictionary")
        } else {
            None
        };
        match m {
            BoundMethodType::Function(f) => {
                Self::target_from_def_kind(&f.metadata.kind, module_name_override)
            }
            BoundMethodType::Forall(f) => {
                Self::target_from_def_kind(&f.body.metadata.kind, module_name_override)
            }
            BoundMethodType::Overload(f) => {
                Self::target_from_def_kind(&f.metadata.kind, module_name_override)
            }
        }
    }
    fn callee_method_kind_from_function_metadata(m: &FuncMetadata) -> String {
        if m.flags.is_staticmethod {
            String::from(CALLEE_KIND_STATICMETHOD)
        } else if m.flags.is_classmethod {
            String::from(CALLEE_KIND_CLASSMETHOD)
        } else {
            String::from(CALLEE_KIND_METHOD)
        }
    }
    fn callee_method_kind_from_bound_method_type(m: &BoundMethodType) -> String {
        match m {
            BoundMethodType::Function(f) => {
                Self::callee_method_kind_from_function_metadata(&f.metadata)
            }
            BoundMethodType::Forall(f) => {
                Self::callee_method_kind_from_function_metadata(&f.body.metadata)
            }
            BoundMethodType::Overload(f) => {
                Self::callee_method_kind_from_function_metadata(&f.metadata)
            }
        }
    }
    fn class_info_for_qname(qname: &QName, is_typed_dict: bool) -> Vec<(String, bool)> {
        vec![(Self::qname_to_string(qname), is_typed_dict)]
    }
    fn class_info_from_bound_obj(ty: &Type) -> Vec<(String, bool)> {
        match ty {
            Type::SelfType(c) => Self::class_info_for_qname(c.qname(), false),
            // TODO: wrap in 'type'
            Type::Type(t) => Self::class_info_from_bound_obj(t),
            Type::ClassType(c) => Self::class_info_for_qname(c.qname(), false),
            Type::ClassDef(c) => Self::class_info_for_qname(c.qname(), false),
            Type::TypedDict(d) => match d {
                TypedDict::TypedDict(inner) => Self::class_info_for_qname(inner.qname(), true),
                TypedDict::Anonymous(_) => vec![],
            },
            Type::Literal(lit) if matches!(lit.value, Lit::Str(_)) => {
                vec![(String::from("builtins.str"), false)]
            }
            Type::LiteralString(_) => {
                vec![(String::from("builtins.str"), false)]
            }
            Type::Literal(lit) if let Lit::Int(_) = lit.value => {
                vec![(String::from("builtins.int"), false)]
            }
            Type::Literal(lit) if let Lit::Bool(_) = lit.value => {
                vec![(String::from("builtins.bool"), false)]
            }
            Type::Quantified(q) => match &q.restriction {
                // for explicit bound - use name of the type used as bound
                Restriction::Bound(b) => Self::class_info_from_bound_obj(b),
                // no bound - use name of the type variable (not very useful but not worse than status quo)
                Restriction::Unrestricted => vec![(q.name().to_string(), false)],
                Restriction::Constraints(tys) => tys
                    .iter()
                    .flat_map(Self::class_info_from_bound_obj)
                    .collect_vec(),
            },
            Type::Union(u) => u
                .members
                .iter()
                .flat_map(Self::class_info_from_bound_obj)
                .collect_vec(),
            Type::Intersect(intersection) => {
                let (_members, fallback) = &**intersection;
                Self::class_info_from_bound_obj(fallback)
            }
            _ => panic!("unexpected type: {ty:?}"),
        }
    }
    fn callee_from_mro<F: Fn(&AnswersSolver<TransactionHandle>, &Class) -> Option<String>>(
        &self,
        c: &Class,
        fallback_name: &str,
        callee_from_ancestor: F,
    ) -> Vec<Callee> {
        let call_target = self
            .transaction
            .ad_hoc_solve(&self.handle, "query_mro", |solver| {
                let mro = solver.get_mro_for_class(c);
                iter::once(c)
                    .chain(mro.ancestors(solver.stdlib).map(|x| x.class_object()))
                    .find_map(|c| callee_from_ancestor(&solver, c))
            });
        let class_name = Self::qname_to_string(c.qname());
        let target = if let Some(Some(t)) = call_target {
            t
        } else {
            format!("{class_name}.{fallback_name}")
        };
        vec![Callee {
            kind: String::from(CALLEE_KIND_METHOD),
            target,
            class_name: Some(class_name),
        }]
    }
    fn for_callable(&self, callee_range: TextRange) -> Vec<Callee> {
        // a bit unfortunate that we have to rely on LSP functionality to get the target
        let defs = self
            .transaction
            .find_definition(
                &self.handle,
                // take location of last included character in range (which should work for identifiers and attributes)
                callee_range.end().checked_sub(TextSize::from(1)).unwrap(),
                FindPreference {
                    resolve_call_dunders: false,
                    ..FindPreference::default()
                },
            )
            .map(Vec1::into_vec)
            .unwrap_or_default()
            .into_iter()
            // filter out attributes since we don't know how to handle them
            .filter(|d| !matches!(d.metadata, DefinitionMetadata::Attribute))
            .collect_vec();
        if defs.is_empty() {
            vec![]
        } else if defs.len() == 1 {
            // TODO: decide what do to with multiple definitions
            let def0 = &defs[0];
            if def0.module.name() == self.handle.module() {
                match &def0.metadata {
                    DefinitionMetadata::Variable(_) => {
                        let name = &self.module_info.code_at(defs[0].definition_range);
                        vec![Callee {
                            kind: String::from(CALLEE_KIND_FUNCTION),
                            target: format!("$parameter${name}"),
                            class_name: None,
                        }]
                    }
                    x => panic!("callable ty - unexpected metadata kind, {x:?}"),
                }
            } else {
                vec![]
            }
        } else {
            panic!(
                "callable ty at [{}] not supported yet, {defs:?}",
                self.module_info.display_range(callee_range)
            )
        }
    }
    fn class_name_from_call_target(&self, call_target: Option<&Expr>) -> Option<String> {
        if let Some(Expr::Attribute(attr)) = call_target
            && let Some(ty) = self.answers.get_type_trace(attr.value.range())
            && !matches!(ty, Type::Module(_))
        {
            // treat calls where targets are attribute access a.b and a is not a module
            // as method calls
            Some(type_to_string(&ty))
        } else {
            None
        }
    }

    fn find_init_or_new(&self, cls: &Class) -> Vec<Callee> {
        self.callee_from_mro(cls, "__init__", |solver, c| {
            // find first class that has __init__ or __new__
            let class_fields = solver.get_class_fields(c);
            let has_init = class_fields.is_some_and(|f| f.contains(&dunder::INIT))
                || solver
                    .get_from_class(c, &KeyClassSynthesizedFields(c.index()))
                    .is_some_and(|f| f.get(&dunder::INIT).is_some());
            if has_init {
                Some(format!("{}.{}.__init__", c.module_name(), c.name()))
            } else if class_fields.is_some_and(|f| f.contains(&dunder::NEW)) {
                Some(format!("{}.{}.__new__", c.module_name(), c.name()))
            } else {
                None
            }
        })
    }
    fn init_or_new_from_union(&self, tys: &[Type], callee_range: TextRange) -> Vec<Callee> {
        tys.iter()
            .flat_map(|t| self.init_or_new_from_type(t, callee_range))
            .unique()
            // return sorted by target
            .sorted_by(|a, b| a.target.cmp(&b.target))
            .collect_vec()
    }
    fn init_or_new_from_type(&self, ty: &Type, callee_range: TextRange) -> Vec<Callee> {
        match ty {
            Type::SelfType(c) | Type::ClassType(c) => self.find_init_or_new(c.class_object()),
            Type::Quantified(q) => match &q.restriction {
                Restriction::Bound(Type::ClassType(c)) => self.find_init_or_new(c.class_object()),
                Restriction::Constraints(tys) => self.init_or_new_from_union(tys, callee_range),
                x => panic!(
                    "unexpected restriction {}: {x:?}",
                    self.module_info.display_range(callee_range)
                ),
            },
            Type::Union(u) => self.init_or_new_from_union(&u.members, callee_range),
            Type::Intersect(intersection) => {
                let (_members, fallback) = &**intersection;
                self.init_or_new_from_type(fallback, callee_range)
            }
            Type::Any(_) => vec![],
            x => {
                panic!(
                    "unexpected type at [{}]: {x:?}",
                    self.module_info.display_range(callee_range)
                );
            }
        }
    }
    pub fn callee_from_type(
        &self,
        ty: &Type,
        call_target: Option<&Expr>,
        callee_range: TextRange,
        call_arguments: Option<&Arguments>,
    ) -> Vec<Callee> {
        match ty {
            Type::Quantified(q) => match &q.restriction {
                Restriction::Bound(b) => {
                    self.callee_from_type(b, call_target, callee_range, call_arguments)
                }
                x => panic!(
                    "unexpected restriction {}: {x:?}",
                    self.module_info.display_range(callee_range)
                ),
            },
            Type::Never(_) => vec![],
            Type::Union(u) => {
                // get callee for each type
                u.members
                    .iter()
                    .flat_map(|t| {
                        self.callee_from_type(t, call_target, callee_range, call_arguments)
                    })
                    .unique()
                    // return sorted by target
                    .sorted_by(|a, b| a.target.cmp(&b.target))
                    .collect_vec()
            }
            Type::Intersect(intersection) => {
                let (_members, fallback) = &**intersection;
                self.callee_from_type(fallback, call_target, callee_range, call_arguments)
            }
            Type::BoundMethod(m) => Self::class_info_from_bound_obj(&m.obj)
                .into_iter()
                .map(|(class_name, class_is_typed_dict)| Callee {
                    kind: Self::callee_method_kind_from_bound_method_type(&m.func),
                    target: Self::target_from_bound_method_type(&m.func, class_is_typed_dict),
                    class_name: Some(class_name),
                })
                .unique()
                // return sorted by target
                .sorted_by(|a, b| a.target.cmp(&b.target))
                .collect_vec(),

            Type::Function(f) => {
                vec![self.callee_from_function(f, call_target, call_arguments)]
            }
            Type::Overload(f) => {
                let class_name = self.class_name_from_call_target(call_target);
                let kind = if class_name.is_some() {
                    String::from(CALLEE_KIND_METHOD)
                } else {
                    String::from(CALLEE_KIND_FUNCTION)
                };
                // assuming that overload represents function and method overloads
                // are handled by BoundMethod case
                vec![Callee {
                    kind,
                    target: Self::target_from_def_kind(&f.metadata.kind, None),
                    class_name,
                }]
            }
            Type::Callable(..) => self.for_callable(callee_range),
            Type::Type(ty) => self.init_or_new_from_type(ty, callee_range),
            // Annotated[T, ...] is not callable (matching as_call_target_impl).
            Type::Annotated(_, _) => vec![],

            Type::ClassDef(cls) => self.find_init_or_new(cls),
            Type::Forall(v) => match &v.body {
                Forallable::Function(func) => {
                    vec![self.callee_from_function(func, call_target, call_arguments)]
                }
                Forallable::Callable(_) => self.for_callable(callee_range),
                Forallable::TypeAlias(TypeAliasData::Value(t)) => {
                    self.callee_from_type(&t.as_type(), call_target, callee_range, call_arguments)
                }
                Forallable::TypeAlias(TypeAliasData::Ref(_)) => vec![],
            },
            Type::SelfType(c) | Type::ClassType(c) => {
                self.callee_from_mro(c.class_object(), "__call__", |solver, c| {
                    if solver
                        .get_class_fields(c)
                        .is_some_and(|f| f.contains(&dunder::CALL))
                    {
                        Some(format!("{}.{}.__call__", c.module_name(), c.name()))
                    } else {
                        None
                    }
                })
            }
            Type::Any(_) => vec![],
            Type::Literal(_) => vec![],
            Type::TypeAlias(data) => match &**data {
                TypeAliasData::Value(t) => {
                    self.callee_from_type(&t.as_type(), call_target, callee_range, call_arguments)
                }
                TypeAliasData::Ref(_) => vec![],
            },
            _ => panic!(
                "unexpected type at [{}]: {ty:?}",
                self.module_info.display_range(callee_range)
            ),
        }
    }
}

impl Query {
    pub fn new(config_finder: ConfigFinder, thread_count: ThreadCount) -> Self {
        let state = State::new(config_finder, thread_count);
        Self {
            state,
            sys_info: SysInfo::default(),
            files: Mutex::new(SmallSet::new()),
            type_cache: TypeCache::new(),
        }
    }

    fn make_handle(&self, name: ModuleName, path: ModulePath) -> Handle {
        let config = self
            .state
            .config_finder()
            .python_file(ModuleNameWithKind::guaranteed(name.dupe()), &path);
        if config.source_db.is_some() {
            panic!("Pyrefly doesn't support sourcedb-powered queries yet");
        }
        // TODO(connernilsen): make this work with build systems
        Handle::new(name, path, self.sys_info.dupe())
    }

    pub fn change_files(&self, events: &CategorizedEvents) {
        // Clear type cache when files change since types may be invalidated
        // Examples: class definitions change, type aliases change, imports change
        self.type_cache.clear();

        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        let new_transaction_mut = transaction.as_mut();
        new_transaction_mut.invalidate_events(events);
        new_transaction_mut.run(&[], Require::Exports, None);
        self.state.commit_transaction(transaction, None);
        let all_files = self.files.lock().iter().cloned().collect::<Vec<_>>();
        self.add_files(all_files);
    }

    /// Load the given files and return any errors associated with them
    pub fn add_files(&self, files: Vec<(ModuleName, ModulePath)>) -> Vec<String> {
        self.files.lock().extend(files.iter().cloned());
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        let handles = files.into_map(|(name, file)| self.make_handle(name, file));
        transaction
            .as_mut()
            .run(&handles, Require::Everything, None);
        let errors = transaction.as_mut().get_errors(&handles);
        self.state.commit_transaction(transaction, None);
        let project_root = PathBuf::new();
        let collected = errors.collect_errors();
        let mut output_errors = collected.ordinary;
        output_errors.extend(collected.directives);
        output_errors.map(|e| {
            // We deliberately don't have a Display for `Error`, to encourage doing the right thing.
            // But we just hack something up as this code is experimental.
            let mut s = Vec::new();
            {
                let mut renderer = ErrorRenderer::plain(&mut s);
                renderer.write(e, project_root.as_path(), false).unwrap();
            }
            String::from_utf8_lossy(&s).into_owned()
        })
    }

    pub fn get_attributes(
        &self,
        name: ModuleName,
        path: ModulePath,
        class_name: &str,
    ) -> Option<Vec<Attribute>> {
        let transaction = self.state.transaction();
        let handle = self.make_handle(name, path);
        let ast = transaction.get_ast(&handle)?;

        // find last declaration of class with specified name in file
        let cls = ast
            .body
            .iter()
            .filter_map(|e| {
                if let Stmt::ClassDef(cls) = e {
                    if cls.name.id.as_str() == class_name {
                        Some(cls)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .last()?;
        let class_ty = transaction.get_type_at(&handle, cls.name.start());
        fn get_kind_and_field_type(ty: &Type) -> (Option<String>, &Type) {
            match ty {
                Type::Function(f)
                    if f.metadata
                        .flags
                        .property_metadata
                        .as_ref()
                        .is_some_and(|meta| matches!(meta.role, PropertyRole::Getter)) =>
                {
                    (Some(String::from("property")), ty)
                }
                Type::ClassType(c)
                    if c.name() == "classproperty" || c.name() == "cached_classproperty" =>
                {
                    let result_ty = c.targs().as_slice().first().unwrap();
                    (Some(String::from("property")), result_ty)
                }
                _ => (None, ty),
            }
        }
        let bindings = transaction.get_bindings(&handle)?;
        let answers = transaction.get_answers(&handle)?;

        if let Some(Type::ClassDef(cd)) = &class_ty {
            let class_fields = bindings
                .get_class_fields(cd.index())
                .cloned()
                .unwrap_or_else(ClassFields::empty);
            let res = class_fields
                .names()
                .filter_map(|n| {
                    let class_field_index = KeyClassField(cd.index(), n.clone());
                    let class_field_idx =
                        bindings.key_to_idx_hashed_opt(Hashed::new(&class_field_index))?;
                    let class_field = bindings.get(class_field_idx);
                    let is_final = match &class_field.definition {
                        ClassFieldDefinition::DeclaredByAnnotation { annotation, .. } => {
                            Some(*annotation)
                        }
                        ClassFieldDefinition::AssignedInBody { annotation, .. } => *annotation,
                        _ => None,
                    }
                    .and_then(|idx| answers.get_idx(idx))
                    .map(|f| f.annotation.is_final())
                    .unwrap_or(false);

                    // Get field type efficiently (avoids expensive position-based lookup)
                    // Priority: annotation type > expression type trace > ClassField.ty()
                    let field_ty = match &class_field.definition {
                        ClassFieldDefinition::AssignedInBody {
                            value,
                            annotation,
                            alias_of: _,
                        } => {
                            annotation
                                .and_then(|idx| answers.get_idx(idx))
                                .and_then(|a| a.annotation.ty.clone())
                                // Fall back to expression type trace
                                .or_else(|| {
                                    if let ExprOrBinding::Expr(expr) = value.as_ref() {
                                        answers.get_type_trace(expr.range())
                                    } else {
                                        None
                                    }
                                })
                                // Final fallback: ClassField.ty()
                                .or_else(|| answers.get_idx(class_field_idx).map(|cf| cf.ty()))
                        }
                        ClassFieldDefinition::DefinedInMethod {
                            values, annotation, ..
                        } => {
                            annotation
                                .and_then(|idx| answers.get_idx(idx))
                                .and_then(|a| a.annotation.ty.clone())
                                // Fall back to expression type trace
                                .or_else(|| {
                                    if let Some(ExprOrBinding::Expr(expr)) = values.first() {
                                        answers.get_type_trace(expr.range())
                                    } else {
                                        None
                                    }
                                })
                                // Final fallback: ClassField.ty()
                                .or_else(|| answers.get_idx(class_field_idx).map(|cf| cf.ty()))
                        }
                        _ => answers.get_idx(class_field_idx).map(|cf| cf.ty()),
                    };
                    let field_ty = field_ty?;
                    let field_ty = answers.solver().for_export_boundary(field_ty);
                    let (kind, field_ty) = get_kind_and_field_type(&field_ty);

                    Some(Attribute {
                        name: n.to_string(),
                        kind,
                        annotation: type_to_string(field_ty),
                        is_final,
                    })
                })
                .collect_vec();
            Some(res)
        } else {
            None
        }
    }

    // fetches information about callees of a callable in a module
    pub fn get_callees_with_location(
        &self,
        name: ModuleName,
        path: ModulePath,
        location: Option<PythonASTRange>,
    ) -> Option<Vec<(PythonASTRange, Callee)>> {
        let transaction = self.state.transaction();
        let handle = self.make_handle(name, path);
        let find_callees = CalleesWithLocation::new(self, transaction, handle)?;
        Some(find_callees.process(location))
    }

    pub fn get_types_in_file(
        &self,
        name: ModuleName,
        path: ModulePath,
    ) -> Option<Vec<(PythonASTRange, String)>> {
        self.get_types_in_file_transformed(name, path, true, None, |_context, _ty, display| display)
    }

    pub fn get_type_shapes_in_file(
        &self,
        name: ModuleName,
        path: ModulePath,
    ) -> Option<Vec<(PythonASTRange, TypeShape)>> {
        self.get_type_shapes_in_file_filtered(name, path, None)
    }

    pub fn get_type_shapes_in_file_filtered(
        &self,
        name: ModuleName,
        path: ModulePath,
        walker: Option<&TypeQueryStmtWalker>,
    ) -> Option<Vec<(PythonASTRange, TypeShape)>> {
        self.get_types_in_file_transformed(name, path, true, walker, type_shape_from)
    }

    pub fn get_type_shapes_in_file_with_timing(
        &self,
        name: ModuleName,
        path: ModulePath,
    ) -> Option<(Vec<(PythonASTRange, TypeShape)>, TypeQueryTiming)> {
        self.get_type_shapes_in_file_with_timing_filtered(name, path, None)
    }

    pub fn get_type_shapes_in_file_with_timing_filtered(
        &self,
        name: ModuleName,
        path: ModulePath,
        walker: Option<&TypeQueryStmtWalker>,
    ) -> Option<(Vec<(PythonASTRange, TypeShape)>, TypeQueryTiming)> {
        self.get_types_in_file_with_timing(name, path, true, walker, type_shape_from)
    }

    /// `include_display` controls whether each located type carries its display
    /// string. Structured clients that resolve types from the table alone can pass
    /// `false` to skip per-location `type_to_string` (and the write-only type-cache
    /// population) on the server and omit `display` from the wire.
    pub fn get_type_table_in_file(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
    ) -> Option<TypeTableResponseData> {
        self.get_type_table_in_file_filtered(name, path, include_display, None)
    }

    pub fn get_type_table_in_file_filtered(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
        walker: Option<&TypeQueryStmtWalker>,
    ) -> Option<TypeTableResponseData> {
        let type_table = RefCell::new(TypeTableBuilder::new());
        let types = self.get_types_in_file_transformed(
            name,
            path,
            include_display,
            walker,
            |context, ty, display| {
                let type_index = type_to_indexed_shape(context, ty, &mut type_table.borrow_mut());
                (type_index, display)
            },
        )?;
        Some(TypeTableResponseData {
            type_table: type_table.into_inner().into_type_table(),
            types: located_type_table_refs(types, include_display),
        })
    }

    pub fn get_type_table_in_file_with_timing(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
    ) -> Option<(TypeTableResponseData, TypeQueryTiming)> {
        self.get_type_table_in_file_with_timing_filtered(name, path, include_display, None)
    }

    pub fn get_type_table_in_file_with_timing_filtered(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
        walker: Option<&TypeQueryStmtWalker>,
    ) -> Option<(TypeTableResponseData, TypeQueryTiming)> {
        let type_table = RefCell::new(TypeTableBuilder::new());
        let (types, timing) = self.get_types_in_file_with_timing(
            name,
            path,
            include_display,
            walker,
            |context, ty, display| {
                let type_index = type_to_indexed_shape(context, ty, &mut type_table.borrow_mut());
                (type_index, display)
            },
        )?;
        Some((
            TypeTableResponseData {
                type_table: type_table.into_inner().into_type_table(),
                types: located_type_table_refs(types, include_display),
            },
            timing,
        ))
    }

    fn get_types_in_file_transformed<T, F>(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
        walker: Option<&TypeQueryStmtWalker>,
        transform: F,
    ) -> Option<Vec<(PythonASTRange, T)>>
    where
        F: Fn(&TypeShapeContext, &Type, String) -> T,
    {
        self.get_types_in_file_with_optional_timing(
            name,
            path,
            include_display,
            walker,
            transform,
            None,
        )
    }

    fn get_types_in_file_with_timing<T, F>(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
        walker: Option<&TypeQueryStmtWalker>,
        transform: F,
    ) -> Option<(Vec<(PythonASTRange, T)>, TypeQueryTiming)>
    where
        F: Fn(&TypeShapeContext, &Type, String) -> T,
    {
        let mut timing = TypeQueryTiming::default();
        let total_start = Instant::now();
        let types = self.get_types_in_file_with_optional_timing(
            name,
            path,
            include_display,
            walker,
            transform,
            Some(&mut timing),
        )?;
        timing.total = total_start.elapsed();
        Some((types, timing))
    }

    fn get_types_in_file_with_optional_timing<T, F>(
        &self,
        name: ModuleName,
        path: ModulePath,
        include_display: bool,
        walker: Option<&TypeQueryStmtWalker>,
        transform: F,
        mut timing: Option<&mut TypeQueryTiming>,
    ) -> Option<Vec<(PythonASTRange, T)>>
    where
        F: Fn(&TypeShapeContext, &Type, String) -> T,
    {
        let setup_start = timing.as_ref().map(|_| Instant::now());
        let handle = self.make_handle(name, path);

        let transaction = self.state.transaction();
        let ast = transaction.get_ast(&handle)?;
        let module_info = transaction.get_module_info(&handle)?;
        let answers = transaction.get_answers(&handle)?;
        let bindings = transaction.get_bindings(&handle)?;
        let profile = timing
            .as_ref()
            .map(|_| Mutex::new(TypeQueryProfileMap::new()));
        let type_shape_context = TypeShapeContext {
            transaction: &transaction,
            source_handle: &handle,
            profile: profile.as_ref(),
        };
        if let (Some(timing), Some(setup_start)) = (timing.as_deref_mut(), setup_start) {
            timing.setup = setup_start.elapsed();
        }

        fn add_type<T, F>(
            ty: &Type,
            e: &Expr,
            parent: Option<&Expr>,
            range: TextRange,
            module_info: &ModuleInfo,
            res: &mut Vec<(PythonASTRange, T)>,
            type_cache: &TypeCache,
            transform: &F,
            type_shape_context: &TypeShapeContext,
            include_display: bool,
            timing: &mut Option<&mut TypeQueryTiming>,
        ) where
            F: Fn(&TypeShapeContext, &Type, String) -> T,
        {
            // The display string is only needed by callers that ship it per location.
            // When they opt out we skip both `type_to_string` and the type-cache
            // population (the cache is keyed on display and only read by is_subtype).
            let display = if include_display {
                let stringify_start = timing.as_ref().map(|_| Instant::now());
                let display = type_to_string(ty);
                if let (Some(timing), Some(stringify_start)) =
                    (timing.as_deref_mut(), stringify_start)
                {
                    timing.stringify += stringify_start.elapsed();
                }

                // Only clone ty if not already in cache
                let cache_start = timing.as_ref().map(|_| Instant::now());
                type_cache
                    .cache
                    .entry(display.clone())
                    .or_insert_with(|| ty.clone());
                if let (Some(timing), Some(cache_start)) = (timing.as_deref_mut(), cache_start) {
                    timing.cache += cache_start.elapsed();
                }
                display
            } else {
                String::new()
            };

            let transform_start = timing.as_ref().map(|_| Instant::now());
            let transformed = transform(type_shape_context, ty, display);
            if let (Some(timing), Some(transform_start)) = (timing.as_deref_mut(), transform_start)
            {
                timing.transform += transform_start.elapsed();
            }

            res.push((
                python_ast_range_for_expr(module_info, range, e, parent),
                transformed,
            ));
            if let Some(timing) = timing.as_deref_mut() {
                timing.located_count += 1;
            }
        }
        fn try_find_key_for_name(name: &ExprName, bindings: &Bindings) -> Option<Key> {
            let key = Key::BoundName(ShortIdentifier::expr_name(name));
            if bindings.is_valid_key(&key) {
                Some(key)
            } else if let key = Key::Definition(ShortIdentifier::expr_name(name))
                && bindings.is_valid_key(&key)
            {
                Some(key)
            } else {
                None
            }
        }
        fn f<T, F>(
            x: &Expr,
            parent: Option<&Expr>,
            module_info: &ModuleInfo,
            answers: &Answers,
            bindings: &Bindings,
            res: &mut Vec<(PythonASTRange, T)>,
            type_cache: &TypeCache,
            transform: &F,
            type_shape_context: &TypeShapeContext,
            include_display: bool,
            timing: &mut Option<&mut TypeQueryTiming>,
        ) where
            F: Fn(&TypeShapeContext, &Type, String) -> T,
        {
            let range = x.range();
            if let Expr::Name(name) = x
                && let Some(key) = try_find_key_for_name(name, bindings)
                && let Some(ty) = answers.get_type_at(bindings.key_to_idx(&key))
            {
                add_type(
                    &ty,
                    x,
                    parent,
                    range,
                    module_info,
                    res,
                    type_cache,
                    transform,
                    type_shape_context,
                    include_display,
                    timing,
                );
            } else if let Some(ty) = answers.get_type_trace(range) {
                add_type(
                    &ty,
                    x,
                    parent,
                    range,
                    module_info,
                    res,
                    type_cache,
                    transform,
                    type_shape_context,
                    include_display,
                    timing,
                );
            }
            x.recurse(&mut |c| {
                f(
                    c,
                    Some(x),
                    module_info,
                    answers,
                    bindings,
                    res,
                    type_cache,
                    transform,
                    type_shape_context,
                    include_display,
                    timing,
                )
            });
        }

        let mut res = Vec::new();
        let mut visit_expr = |x: &Expr, parent: Option<&Expr>| {
            f(
                x,
                parent,
                &module_info,
                &answers,
                &bindings,
                &mut res,
                &self.type_cache,
                &transform,
                &type_shape_context,
                include_display,
                &mut timing,
            )
        };
        if let Some(walker) = walker {
            walker(&ast.body, &mut visit_expr);
        } else {
            for stmt in &ast.body {
                stmt.visit(&mut |x| visit_expr(x, None));
            }
        }
        if let (Some(timing), Some(profile)) = (timing, profile) {
            timing.shape_profiles = profile.into_inner();
        }
        Some(res)
    }

    /// Given an expression, which contains qualified types, guess which imports to add.
    ///
    /// For example `foo.bar.baz` will return `[foo.bar]`.
    ///
    /// The expression comes in as a module because we are parsing it from a raw string
    /// input; we expect it to actually be a type expression.
    fn find_imports(module: &ModModule, t: &Transaction, h: &Handle) -> Vec<String> {
        fn compute_prefix(attr: &ExprAttribute) -> Option<Vec<&Name>> {
            match &*attr.value {
                Expr::Attribute(base) => {
                    let mut res = compute_prefix(base)?;
                    res.push(&base.attr.id);
                    Some(res)
                }
                Expr::Name(base) => Some(vec![&base.id]),
                _ => None,
            }
        }

        fn collect_attribute_prefixes(
            x: &Expr,
            res: &mut SmallSet<String>,
            t: &Transaction,
            h: &Handle,
        ) {
            if let Expr::Attribute(attr) = x {
                // `attr` is a qname of a type. Get its prefix, which is likely the
                // module where it is defined.
                if let Some(mut names) = compute_prefix(attr) {
                    // The initial prefix may not be an actual module, if the type in question
                    // is a nested class. Search recursively for the longest part of the prefix
                    // that is a module, and assume that is where the type is defined.
                    //
                    // Note: in messy codebases that include name collisions between submodules
                    // and attributes of `__init__.py` modules, this rule can fail (in this
                    // scenario it's also possible for the qname to be ambiguous, as in two
                    // distinct types have the same qname). We do not support such codebases.
                    loop {
                        if !names.is_empty() {
                            let module_name = names.map(|name| name.as_str()).join(".");
                            if t.import_handle(
                                h,
                                ModuleName::from_string(module_name.clone()),
                                None,
                            )
                            .finding()
                            .is_some()
                            {
                                // We found the longest matching prefix, assume this is the import.
                                res.insert(names.map(|name| name.as_str()).join("."));
                                break;
                            } else {
                                // No module at this prefix, keep looking.
                                names.pop();
                            }
                        } else {
                            // If we get here, either the name is undefined or it is defined in `builtins`;
                            // either way we can skip it.
                            break;
                        }
                    }
                }
            } else {
                x.recurse(&mut |x| collect_attribute_prefixes(x, res, t, h));
            }
        }
        let mut res = SmallSet::new();
        module.visit(&mut |x| collect_attribute_prefixes(x, &mut res, t, h));
        res.into_iter().collect()
    }

    fn check_snippet(
        &self,
        t: &mut Transaction,
        handle: &Handle,
        path: PathBuf,
        snippet: &str,
    ) -> Result<(), String> {
        let imported = Query::find_imports(&Ast::parse(snippet, PySourceType::Python).0, t, handle);
        let imports = imported.map(|x| format!("import {x}\n")).join("");

        // First, make sure that the types are well-formed and importable, return `Err` if not
        let code = format!("{imports}\n{snippet}\n");
        t.set_memory(vec![(
            path.clone(),
            Some(Arc::new(FileContents::from_source(code))),
        )]);
        t.run(&[handle.dupe()], Require::Everything, None);
        let errors = t.get_errors([handle]).collect_errors();
        if !errors.ordinary.is_empty() {
            let mut res = Vec::new();
            let project_root = PathBuf::new();
            let mut renderer = ErrorRenderer::plain(&mut res);
            for e in errors.ordinary {
                renderer.write(&e, project_root.as_path(), true).unwrap();
            }
            return Err(format!(
                "{}\n\nSource code:\n{snippet}",
                str::from_utf8(&res).unwrap_or("UTF8 error")
            ));
        }
        Ok(())
    }

    fn find_types(
        ast: &ModModule,
        bindings: Bindings,
        answers: &Answers,
        return_first: bool,
    ) -> (Type, Option<Type>) {
        let mut first: Option<Type> = None;
        for p in &ast.body {
            if let Stmt::AnnAssign(assign) = p
                && let Expr::Name(n) = &*assign.target
            {
                let key = bindings.key_to_idx(&Key::Definition(ShortIdentifier::expr_name(n)));
                let ty = answers.get_type_at(key).unwrap();
                if return_first {
                    return (ty, None);
                } else if let Some(v) = first {
                    return (v, Some(ty));
                } else {
                    first = Some(ty);
                }
            }
        }
        unreachable!("No type aliases in ast")
    }

    /// Return `Err` if you can't resolve them to types, otherwise return `lt <: gt`.
    pub fn is_subtype(
        &self,
        name: ModuleName,
        path: PathBuf,
        lt: &str,
        gt: &str,
    ) -> Result<bool, String> {
        let is_typed_dict_request = gt == "TypedDictionary" || gt == "NonTotalTypedDictionary";

        // Check cache for both types
        let cached_lt = self.type_cache.get(lt);
        let cached_gt = if !is_typed_dict_request {
            self.type_cache.get(gt)
        } else {
            None
        };

        // Determine what needs to be computed
        let need_lt = cached_lt.is_none();
        let need_gt = !is_typed_dict_request && cached_gt.is_none();

        // Fast path: everything is cached
        if !need_lt && (!need_gt || is_typed_dict_request) {
            let sub_ty = cached_lt.unwrap();
            if is_typed_dict_request {
                return Ok(matches!(
                    sub_ty,
                    Type::TypedDict(_) | Type::PartialTypedDict(_)
                ));
            } else {
                let super_ty = cached_gt.unwrap();
                let t = self.state.transaction();
                let h = self.make_handle(name, ModulePath::filesystem(path));
                let result = t
                    .ad_hoc_solve(&h, "query_is_subset_eq", |solver| {
                        solver.is_subset_eq(&sub_ty, &super_ty)
                    })
                    .unwrap_or(false);
                return Ok(result);
            }
        }

        // Slow path: compute missing types
        let mut t = self.state.transaction();
        let h = self.make_handle(name, ModulePath::memory(path.clone()));

        // Create minimal snippet for only the types we need
        let snippet = match (need_lt, need_gt) {
            (true, true) => format!("X : ({lt})\nY : ({gt})"),
            (true, false) => format!("X : ({lt})"),
            (false, true) => format!("Y : ({gt})"),
            (false, false) => unreachable!("handled by fast path"),
        };

        self.check_snippet(&mut t, &h, path, &snippet)?;

        let ast = t.get_ast(&h).ok_or("No ast")?;
        let answers = t.get_answers(&h).ok_or("No answers")?;
        let bindings = t.get_bindings(&h).ok_or("No bindings")?;

        // Extract and cache the computed types
        let (sub_ty, super_ty_opt) = match (need_lt, need_gt) {
            (true, true) => {
                // Computed both: X is lt, Y is gt
                let (lt_type, gt_type_opt) = Query::find_types(&ast, bindings, &answers, false);
                let gt_type = gt_type_opt.unwrap();
                self.type_cache.insert(lt.to_owned(), lt_type.clone());
                self.type_cache.insert(gt.to_owned(), gt_type.clone());
                (lt_type, Some(gt_type))
            }
            (true, false) => {
                // Computed only lt: X is lt, use cached gt
                let (lt_type, _) = Query::find_types(&ast, bindings, &answers, true);
                self.type_cache.insert(lt.to_owned(), lt_type.clone());
                (lt_type, cached_gt)
            }
            (false, true) => {
                // Computed only gt: Y is gt, use cached lt
                let (gt_type, _) = Query::find_types(&ast, bindings, &answers, true);
                self.type_cache.insert(gt.to_owned(), gt_type.clone());
                (cached_lt.unwrap(), Some(gt_type))
            }
            (false, false) => unreachable!("handled by fast path"),
        };

        // Compute final result
        let result = if is_typed_dict_request {
            matches!(sub_ty, Type::TypedDict(_) | Type::PartialTypedDict(_))
        } else {
            t.ad_hoc_solve(&h, "query_is_subset_eq", |solver| {
                solver.is_subset_eq(&sub_ty, &super_ty_opt.unwrap())
            })
            .unwrap_or(false)
        };

        Ok(result)
    }

    pub fn resolve_target_from_qualified_name(
        &self,
        name: ModuleName,
        path: PathBuf,
        qualified_name: &str,
    ) -> Option<Vec<Callee>> {
        let mut t = self.state.transaction();
        let h = self.make_handle(name, ModulePath::memory(path.clone()));
        let snippet = format!("x = {qualified_name}");
        // Check and type-check the snippet
        self.check_snippet(&mut t, &h, path, &snippet).unwrap();
        let ast = t.get_ast(&h).unwrap();
        fn find_expr(ast: &ModModule) -> Option<&Expr> {
            for stmt in &ast.body {
                if let Stmt::Assign(assign) = stmt {
                    return Some(&assign.value);
                }
            }
            None
        }
        let find_callees = CalleesWithLocation::new(self, t, h)?;
        let mut res = Vec::new();
        if let Some(expr) = find_expr(&ast) {
            find_callees.callee_from_text_range(expr.range(), Some(expr), |c| {
                res.push(c);
            });
        }
        Some(res)
    }
}
