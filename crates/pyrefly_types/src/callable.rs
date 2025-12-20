/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::cmp::Ord;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display::Fmt;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;

use crate::class::Class;
use crate::class::ClassType;
use crate::display::TypeDisplayContext;
use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::keywords::DataclassTransformMetadata;
use crate::meta_shape_dsl::ShapeDslFunction;
use crate::meta_shape_dsl::ShapeTransform;
use crate::type_output::DisplayOutput;
use crate::type_output::TypeOutput;
use crate::types::AnyStyle;
use crate::types::Type;

/// A wrapper for auxiliary data whose identity should be completely ignored
/// in equality, hashing, ordering, and type-equality comparisons.
/// `IdentityIgnored<T>` always compares as equal, hashes as a no-op, and
/// orders as `Equal` — making it transparent to all identity checks.
///
/// This is useful for attaching auxiliary data (e.g. closure caches) to
/// types that derive `PartialEq`, `Hash`, `Ord`, etc. without affecting
/// their logical identity.
#[derive(Debug, Clone)]
pub struct IdentityIgnored<T>(pub T);

impl<T> PartialEq for IdentityIgnored<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> Eq for IdentityIgnored<T> {}

impl<T> Hash for IdentityIgnored<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<T> PartialOrd for IdentityIgnored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for IdentityIgnored<T> {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl<T> Visit<Type> for IdentityIgnored<T> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl<T> VisitMut<Type> for IdentityIgnored<T> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl<T> TypeEq for IdentityIgnored<T> {
    fn type_eq(&self, _other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        true
    }
}

impl<T> Deref for IdentityIgnored<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Callable {
    pub params: Params,
    pub ret: Type,
}

impl Callable {
    /// Returns true if this callable has the `*args: Any, **kwargs: Any -> Any`
    /// signature (plus an optional unannotated self/cls at index 0).
    /// Used as a heuristic in decorator type resolution for union-typed
    /// decorators.
    pub fn is_args_kwargs_wrapper(&self) -> bool {
        if !matches!(&self.ret, Type::Any(AnyStyle::Implicit)) {
            return false;
        }
        match &self.params {
            Params::List(params) => {
                let items = params.items();
                items.iter().any(|p| matches!(p, Param::Varargs(..)))
                    && items.iter().any(|p| matches!(p, Param::Kwargs(..)))
                    && items.iter().enumerate().all(|(i, p)| match p {
                        Param::Varargs(..) | Param::Kwargs(..) => true,
                        Param::Pos(_, ty, _) | Param::PosOnly(Some(_), ty, _) if i == 0 => {
                            matches!(ty, Type::Any(AnyStyle::Implicit))
                        }
                        _ => false,
                    })
            }
            _ => false,
        }
    }

    pub fn contains_callable_residual(&self) -> bool {
        let check = |t: &Type| matches!(t, Type::CallableResidual(_));
        if self.ret.any(check) {
            return true;
        }
        match &self.params {
            Params::List(params) => params.items().iter().any(|p| p.as_type().any(check)),
            Params::ParamSpec(prefix, p) => {
                prefix.iter().any(|pp| {
                    let ty = match pp {
                        PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => ty,
                    };
                    ty.any(check)
                }) || p.any(check)
            }
            Params::Ellipsis | Params::Materialization => false,
        }
    }

    /// Returns true if this callable carries no real type information: all
    /// parameters and the return type are `Any(Implicit)` (i.e. Unknown).
    pub fn is_fully_unknown(&self) -> bool {
        if !matches!(&self.ret, Type::Any(AnyStyle::Implicit)) {
            return false;
        }
        match &self.params {
            Params::List(params) => params
                .items()
                .iter()
                .all(|p| matches!(p.as_type(), Type::Any(AnyStyle::Implicit))),
            Params::Ellipsis => true,
            _ => false,
        }
    }
}

impl Display for Callable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = Type::Callable(Box::new(self.clone()));
        let ctx = TypeDisplayContext::new(&[&ty]);
        let mut output = DisplayOutput::new(&ctx, f);
        self.fmt_with_type(&mut output, &|t, o| ctx.fmt_helper_generic(t, false, o))
    }
}

#[derive(Debug, Clone)]
pub struct ArgCount {
    pub min: usize,
    pub max: Option<usize>,
}

impl ArgCount {
    fn none_allowed() -> Self {
        Self {
            min: 0,
            max: Some(0),
        }
    }

    fn any_allowed() -> Self {
        Self { min: 0, max: None }
    }

    fn add_arg(&mut self, req: &Required) {
        if *req == Required::Required {
            self.min += 1;
        }
        if let Some(n) = self.max {
            self.max = Some(n + 1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArgCounts {
    pub positional: ArgCount,
    pub keyword: ArgCount,
    pub overall: ArgCount,
}

/// Controls which parameters are displayed by `ParamList::fmt_with_type`
#[derive(Debug, Clone)]
pub enum ParamOverlay {
    /// Display all parameters
    All,
    /// Display only this set of named parameters, plus all anonymous parameters
    Subset(SmallSet<Name>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ParamList(Vec<Param>);

impl ParamList {
    pub fn new(xs: Vec<Param>) -> Self {
        Self(xs)
    }

    /// Create a new ParamList from a list of types,
    /// converting each one into a positional-only param
    pub fn new_types(xs: Vec<PrefixParam>) -> Self {
        Self(xs.into_map(|p| p.into_posonly_param()))
    }

    /// Prepend some parameters for `Concatenate`
    pub fn prepend_types(&self, pre: &[PrefixParam]) -> Cow<'_, ParamList> {
        if pre.is_empty() {
            Cow::Borrowed(self)
        } else {
            // Use `to_subset_param` to preserve Pos v.s. PosOnly
            Cow::Owned(ParamList(
                pre.iter()
                    .map(|p| p.to_param_preserve_name())
                    .chain(self.0.iter().cloned())
                    .collect(),
            ))
        }
    }

    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
        overlay: &ParamOverlay,
    ) -> fmt::Result {
        let mut named_posonly = false;
        let mut kwonly = false;
        let mut skipped_prev = false;
        for (i, param) in self.0.iter().enumerate() {
            // `/` or `*` markers that should be printed before the param
            let mut marker_prefixes = Vec::new();
            if matches!(param, Param::PosOnly(Some(_), _, _)) {
                named_posonly = true;
            } else if named_posonly {
                named_posonly = false;
                marker_prefixes.push("/");
            }
            if !kwonly && matches!(param, Param::KwOnly(..)) {
                kwonly = true;
                marker_prefixes.push("*");
            }
            // Should we elide the param?
            let skip = matches!(overlay, ParamOverlay::Subset(names) if param.name().is_some_and(|name| !names.contains(name)));
            if i > 0 && (!skipped_prev || !marker_prefixes.is_empty() || !skip) {
                output.write_str(", ")?;
            }
            for marker_prefix in marker_prefixes.iter() {
                output.write_str(marker_prefix)?;
                output.write_str(", ")?;
            }
            if skip {
                if !skipped_prev || !marker_prefixes.is_empty() {
                    output.write_str("...")?;
                }
            } else {
                param.fmt_with_type(output, write_type)?;
            }
            skipped_prev = skip;
        }
        if named_posonly {
            output.write_str(", /")?;
        }
        Ok(())
    }

    /// Format parameters each parameter on a new line
    pub fn fmt_with_type_with_newlines<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        let mut named_posonly = false;
        let mut kwonly = false;

        for (i, param) in self.0.iter().enumerate() {
            if i > 0 {
                output.write_str(",\n    ")?;
            }

            if matches!(param, Param::PosOnly(Some(_), _, _)) {
                named_posonly = true;
            } else if named_posonly {
                named_posonly = false;
                output.write_str("/,\n    ")?;
            }

            if !kwonly && matches!(param, Param::KwOnly(..)) {
                kwonly = true;
                output.write_str("*,\n    ")?;
            }

            param.fmt_with_type(output, write_type)?;
        }

        if named_posonly {
            output.write_str(",\n    /")?;
        }

        Ok(())
    }

    pub fn items(&self) -> &[Param] {
        &self.0
    }

    pub fn into_items(self) -> Vec<Param> {
        self.0
    }

    pub fn items_mut(&mut self) -> &mut [Param] {
        &mut self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Type signature that permits everything, namely `*args, **kwargs`.
    pub fn everything() -> ParamList {
        ParamList(vec![
            Param::Varargs(None, Type::any_implicit()),
            Param::Kwargs(None, Type::any_implicit()),
        ])
    }
}

/// Represents a prefix parameter in `Concatenate`.
/// Prefix params can be either positional-only or positional (named).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum PrefixParam {
    PosOnly(Option<Name>, Type, Required),
    Pos(Name, Type, Required),
}

impl PrefixParam {
    /// Create a positional-only prefix param (no name).
    pub fn new(ty: Type, required: Required) -> Self {
        Self::PosOnly(None, ty, required)
    }

    pub fn ty(&self) -> &Type {
        match self {
            Self::PosOnly(_, ty, _) | Self::Pos(_, ty, _) => ty,
        }
    }

    pub fn ty_mut(&mut self) -> &mut Type {
        match self {
            Self::PosOnly(_, ty, _) | Self::Pos(_, ty, _) => ty,
        }
    }

    /// Convert to a positional-only `Param`. Per the typing spec, params in
    /// `Concatenate` are positional-only at the call site. This is also appropriate
    /// for ParamSpec forwarding where prefix params must be passed positionally.
    pub fn into_posonly_param(self) -> Param {
        match self {
            Self::PosOnly(name, ty, required) => Param::PosOnly(name, ty, required),
            Self::Pos(name, ty, required) => Param::PosOnly(Some(name), ty, required),
        }
    }

    /// Convert to a `Param` preserving the Pos vs PosOnly distinction.
    /// Used for subset/subtype checking where name matching matters,
    /// and for direct calls where prefix params should remain keyword-passable.
    pub fn to_param_preserve_name(&self) -> Param {
        match self {
            Self::PosOnly(name, ty, required) => {
                Param::PosOnly(name.clone(), ty.clone(), required.clone())
            }
            Self::Pos(name, ty, required) => Param::Pos(name.clone(), ty.clone(), required.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Params {
    List(ParamList),
    Ellipsis,
    /// All possible materializations of `...`. A subset check with Callable[Materialization, R]
    /// succeeds only if it would succeed with Materialization replaced with any parameter list.
    /// See the comment on Type::Materialization - the intuition here is similar.
    Materialization,
    /// Any arguments to Concatenate, followed by a ParamSpec.
    /// E.g. `Concatenate[int, str, P]` would be `ParamSpec([int, str], P)`,
    /// while `P` alone would be `ParamSpec([], P)`.
    /// `P` may resolve to `Type::ParamSpecValue`, `Type::Concatenate`, or `Type::Ellipsis`
    ParamSpec(Box<[PrefixParam]>, Type),
}

impl Params {
    fn arg_counts(&self) -> ArgCounts {
        match self {
            Self::List(params) => {
                let mut counts = ArgCounts {
                    positional: ArgCount::none_allowed(),
                    keyword: ArgCount::none_allowed(),
                    overall: ArgCount::none_allowed(),
                };
                for param in params.items() {
                    match param {
                        Param::PosOnly(_, _, req) => {
                            counts.positional.add_arg(req);
                            counts.overall.add_arg(req);
                        }
                        Param::Pos(_, _, req) => {
                            counts.positional.add_arg(&Required::Optional(None));
                            counts.keyword.add_arg(&Required::Optional(None));
                            counts.overall.add_arg(req);
                        }
                        Param::KwOnly(_, _, req) => {
                            counts.keyword.add_arg(req);
                            counts.overall.add_arg(req);
                        }
                        Param::Varargs(..) => {
                            counts.positional.max = None;
                            counts.overall.max = None;
                        }
                        Param::Kwargs(..) => {
                            counts.keyword.max = None;
                            counts.overall.max = None;
                        }
                    }
                }
                counts
            }
            Self::Ellipsis | Self::Materialization => ArgCounts {
                positional: ArgCount::any_allowed(),
                keyword: ArgCount::any_allowed(),
                overall: ArgCount::any_allowed(),
            },
            Self::ParamSpec(prefix, _) => ArgCounts {
                positional: ArgCount {
                    min: prefix.len(),
                    max: None,
                },
                keyword: ArgCount::any_allowed(),
                overall: ArgCount {
                    min: prefix.len(),
                    max: None,
                },
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Param {
    PosOnly(Option<Name>, Type, Required),
    Pos(Name, Type, Required),
    Varargs(Option<Name>, Type),
    KwOnly(Name, Type, Required),
    Kwargs(Option<Name>, Type),
}

/// The default value of an optional parameter, containing its type and an optional
/// display string for values whose types don't preserve the literal value (e.g. floats).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DefaultValue {
    pub ty: Type,
    /// Display string for defaults that can't be recovered from the type alone,
    /// e.g. `"3.14"` for float literals whose type is just `float`.
    pub display: Option<String>,
}

/// Visit/VisitMut/TypeEq delegate to `ty` only; `display` is display-only metadata.
impl<To> Visit<To> for DefaultValue
where
    Type: Visit<To>,
{
    const RECURSE_CONTAINS: bool = true;
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a To)) {
        self.ty.visit(f);
    }
}

impl<To> VisitMut<To> for DefaultValue
where
    Type: VisitMut<To>,
{
    const RECURSE_CONTAINS: bool = true;
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut To)) {
        self.ty.visit_mut(f);
    }
}

impl TypeEq for DefaultValue {
    fn type_eq(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        self.ty.type_eq(&other.ty, ctx)
    }
}

impl DefaultValue {
    pub fn new(ty: Type) -> Self {
        Self { ty, display: None }
    }

    pub fn with_display(ty: Type, display: String) -> Self {
        Self {
            ty,
            display: Some(display),
        }
    }
}

/// Requiredness for a function parameter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Required {
    Required,
    /// The parameter is optional, with the default value info if available.
    Optional(Option<DefaultValue>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Function {
    pub signature: Callable,
    pub metadata: FuncMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncMetadata {
    pub kind: FunctionKind,
    pub flags: FuncFlags,
}

impl FuncMetadata {
    pub fn def(module: &Module, cls: Option<&Class>, name: Name) -> Self {
        Self {
            kind: FunctionKind::Def(Arc::new(FuncId {
                module: module.dupe(),
                cls: cls.map(Dupe::dupe),
                name,
                def_index: None,
                outer_funcs: None,
            })),
            flags: FuncFlags::default(),
        }
    }

    pub fn method(cls: &Class, name: Name) -> Self {
        Self::def(cls.module(), Some(cls), name)
    }
}

/// Metadata extracted from a `@deprecated` decorator.
#[derive(
    Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq, PartialOrd, Ord, Hash
)]
pub struct Deprecation {
    pub message: Option<String>,
}

impl Deprecation {
    pub fn new(message: Option<String>) -> Self {
        Self { message }
    }

    /// Format deprecation metadata for error reporting.
    pub fn as_error_detail(&self) -> Option<String> {
        match self.message.as_ref().map(|s| s.trim()) {
            Some(msg) if !msg.is_empty() => Some(msg.to_owned()),
            _ => None,
        }
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Visit, VisitMut, TypeEq
)]
pub enum PropertyRole {
    Getter,
    Setter,
    SetterDecorator,
    DeleterDecorator,
}

/// Shape of a function body that consists of a single placeholder statement.
/// The two variants share the surface form of "trivial body" but have very
/// different semantics: `RaiseNotImplementedError` is an "abstract-ish"
/// placeholder that never returns at runtime, while `ReturnNotImplemented`
/// returns the singleton `NotImplemented` value (a real runtime value used by
/// the dunder protocol). The type checker keeps them separate so it can relax
/// override-consistency only for the abstract-style form, without conflating
/// it with the dunder-protocol form.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Visit, VisitMut, TypeEq
)]
pub enum PlaceholderBodyKind {
    /// Body is exactly `raise NotImplementedError(...)`. This is the canonical
    /// "abstract-ish" placeholder; concrete subclasses override it.
    RaiseNotImplementedError,
    /// Body is exactly `return NotImplemented`. This is the dunder-protocol
    /// signal to defer to the other operand and is not an override placeholder.
    ReturnNotImplemented,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Visit, VisitMut, TypeEq
)]
pub struct PropertyMetadata {
    pub role: PropertyRole,
    pub getter: Type,
    pub setter: Option<Type>,
    pub has_deleter: bool,
}

impl PropertyMetadata {
    /// Build a PropertyMetadata that stores sanitized (metadata-free) copies of getter/setter.
    pub fn from_components(
        role: PropertyRole,
        getter: &Type,
        setter: Option<&Type>,
        has_deleter: bool,
    ) -> Self {
        Self {
            role,
            getter: getter.without_property_metadata(),
            setter: setter.map(|s| s.without_property_metadata()),
            has_deleter,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncFlags {
    pub is_overload: bool,
    pub is_staticmethod: bool,
    pub is_classmethod: bool,
    /// A function decorated with `@deprecated`
    pub deprecation: Option<Deprecation>,
    /// Metadata for `@property`, `@foo.setter`, and `@foo.deleter`.
    pub property_metadata: Option<PropertyMetadata>,
    /// A function decorated with `functools.cached_property` or equivalent.
    pub is_cached_property: bool,
    pub has_enum_member_decoration: bool,
    pub is_override: bool,
    pub has_final_decoration: bool,
    /// A function decorated with `@abc.abstractmethod`
    pub is_abstract_method: bool,
    /// Function body is treated as a stub (e.g. body is `...` or absent in a stub file)
    pub lacks_implementation: bool,
    /// Is the function definition in a `.pyi` file
    pub defined_in_stub_file: bool,
    /// Set when the function was declared with `async def` (NOT when a regular
    /// `def` happens to return a `Coroutine[...]`-typed value). Used to
    /// distinguish async-def placeholders from sync functions explicitly
    /// annotated to return a coroutine, which look identical at the type level
    /// once the async-wrapping into `Coroutine[Any, Any, T]` has happened.
    pub is_async: bool,
    /// Set when the function body is a single placeholder statement (see
    /// `PlaceholderBodyKind`), ignoring a leading docstring. `None` for
    /// ordinary function bodies, and also for trivial bodies (`pass`, `...`,
    /// or empty) — those are tracked separately as stubs, not placeholders.
    pub placeholder_body_kind: Option<PlaceholderBodyKind>,
    /// Set when the function's return type has no user-supplied annotation and
    /// was inferred from the body (corresponds to
    /// `ReturnTypeKind::ShouldInferType`). Used to distinguish a return type
    /// the user wrote (e.g. an explicit `-> Never`) from one Pyrefly inferred,
    /// which lets override-consistency logic relax inferred placeholder returns
    /// without overriding what the user explicitly declared.
    pub is_return_inferred: bool,
    /// A function decorated with `typing.dataclass_transform(...)`, turning it into a
    /// `dataclasses.dataclass`-like decorator. Stores the keyword values passed to the
    /// `dataclass_transform` call. See
    /// https://typing.python.org/en/latest/spec/dataclasses.html#specification.
    pub dataclass_transform_metadata: Option<DataclassTransformMetadata>,
    /// A function decorated with `@uses_shape_dsl`, whose return type should be
    /// refined by evaluating the referenced shape-DSL function at call sites.
    pub shape_transform: Option<Arc<ShapeTransform>>,
}

impl FuncFlags {
    /// Whether the function lacks a runtime implementation and is not defined in a stub file.
    /// This indicates a method that cannot actually be called at runtime (e.g. an abstract
    /// method or protocol method with a `...` or `pass` body in a `.py` file).
    pub fn lacks_runtime_implementation(&self) -> bool {
        self.lacks_implementation && !self.defined_in_stub_file
    }
}

/// The index of a function definition (`def ..():` statement) within the module,
/// used as a reference to data associated with the function.
#[derive(Debug, Clone, Dupe, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[derive(Display, Visit, VisitMut, TypeEq)]
pub struct FuncDefIndex(pub u32);

#[derive(Debug, Clone)]
pub struct FuncId {
    pub module: Module,
    pub cls: Option<Class>,
    pub name: Name,
    pub def_index: Option<FuncDefIndex>,
    /// Dot-separated path of enclosing function names (e.g. `"f1"` for a function nested inside `f1`).
    /// `None` for top-level and class-method functions.
    pub outer_funcs: Option<Name>,
}

impl PartialEq for FuncId {
    fn eq(&self, other: &Self) -> bool {
        self.key_eq().eq(&other.key_eq())
    }
}

impl Eq for FuncId {}
impl TypeEq for FuncId {}

impl Ord for FuncId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key_ord().cmp(&other.key_ord())
    }
}

impl PartialOrd for FuncId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for FuncId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key_eq().hash(state)
    }
}

impl VisitMut<Type> for FuncId {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for FuncId {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// FuncId contains no Type fields, so visiting through Arc is a no-op.
impl VisitMut<Type> for Arc<FuncId> {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for Arc<FuncId> {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl FuncId {
    /// Identity tuple for equality and hashing. `outer_funcs` is intentionally
    /// excluded because it is display-only metadata (the dotted path of enclosing
    /// function names) and does not affect the logical identity of a function.
    fn key_eq(
        &self,
    ) -> (
        ModuleName,
        ModulePath,
        Option<Class>,
        &Name,
        Option<FuncDefIndex>,
    ) {
        (
            self.module.name(),
            self.module.path().to_key_eq(),
            self.cls.clone(),
            &self.name,
            self.def_index,
        )
    }

    fn key_ord(
        &self,
    ) -> (
        ModuleName,
        ModulePath,
        Option<Class>,
        &Name,
        Option<FuncDefIndex>,
    ) {
        self.key_eq()
    }

    fn format_impl(
        func_module: ModuleName,
        func_cls: Option<Class>,
        func_name: &Name,
        current_module: ModuleName,
    ) -> String {
        let module_prefix =
            if func_module == current_module || func_module == ModuleName::builtins() {
                "".to_owned()
            } else {
                format!("{}.", func_module)
            };
        let class_prefix = match &func_cls {
            Some(cls) => {
                format!("{}.", cls.name())
            }
            None => "".to_owned(),
        };
        format!("{module_prefix}{class_prefix}{}", func_name)
    }

    pub fn format(&self, current_module: ModuleName) -> String {
        Self::format_impl(
            self.module.name(),
            self.cls.clone(),
            &self.name,
            current_module,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum FunctionKind {
    IsInstance,
    IsSubclass,
    Dataclass,
    DataclassField,
    DataclassReplace,
    DataclassAsdict,
    /// `typing.dataclass_transform`. Note that this is `dataclass_transform` itself, *not* the
    /// decorator created by a `dataclass_transform(...)` call. See
    /// https://typing.python.org/en/latest/spec/dataclasses.html#specification.
    DataclassTransform,
    ClassMethod,
    Overload,
    Override,
    Cast,
    AssertType,
    RevealType,
    Final,
    RuntimeCheckable,
    Def(Arc<FuncId>),
    AbstractMethod,
    /// Instance of a protocol with a `__call__` method. The function has the `__call__` signature.
    CallbackProtocol(Box<ClassType>),
    TotalOrdering,
    DisjointBase,
    /// `numba.jit()`
    NumbaJit,
    /// `numba.njit()`
    NumbaNjit,
    /// A function whose return type is computed by a shape DSL definition.
    /// The `FuncId` provides identity (module, class, name) for display and
    /// lookup; the `ShapeDslFunction` carries the parsed DSL IR.
    ShapeDsl(
        Arc<FuncId>,
        Arc<ShapeDslFunction>,
        IdentityIgnored<Arc<Vec<Arc<ShapeDslFunction>>>>,
    ),
    /// The `shape_extensions.uses_shape_dsl` decorator function itself.
    UsesShapeDsl,
    /// `sqlalchemy.orm.mapped_column()`
    SqlAlchemyMappedColumn,
}

impl Callable {
    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        match &self.params {
            Params::List(params) => {
                output.write_str("(")?;
                params.fmt_with_type(output, write_type, &ParamOverlay::All)?;
                output.write_str(") -> ")?;
                write_type(&self.ret, output)
            }
            Params::Ellipsis => {
                output.write_str("(...) -> ")?;
                write_type(&self.ret, output)
            }
            Params::Materialization => {
                output.write_str("(Materialization) -> ")?;
                write_type(&self.ret, output)
            }
            Params::ParamSpec(args, pspec) => {
                output.write_str("(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        output.write_str(", ")?;
                    }
                    write_type(arg.ty(), output)?;
                }
                match pspec {
                    Type::ParamSpecValue(params) => {
                        if !args.is_empty() && !params.is_empty() {
                            output.write_str(", ")?;
                        }
                        params.fmt_with_type(output, write_type, &ParamOverlay::All)?;
                    }
                    Type::Ellipsis => {
                        if !args.is_empty() {
                            output.write_str(", ")?;
                        }
                        output.write_str("...")?;
                    }
                    _ => {
                        if !args.is_empty() {
                            output.write_str(", ")?;
                        }
                        output.write_str("ParamSpec(")?;
                        write_type(pspec, output)?;
                        output.write_str(")")?;
                    }
                }
                output.write_str(") -> ")?;
                write_type(&self.ret, output)
            }
        }
    }

    /// Format the function type for use in a hover tooltip. This is similar to `fmt_with_type`, but
    /// it puts args on new lines if there is more than one argument
    pub fn fmt_with_type_with_newlines<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        match &self.params {
            Params::List(params) if params.len() > 1 => {
                // For multiple parameters, put each on a new line with indentation
                output.write_str("(\n    ")?;
                params.fmt_with_type_with_newlines(output, write_type)?;
                output.write_str("\n) -> ")?;
                write_type(&self.ret, output)
            }
            Params::List(..)
            | Params::ParamSpec(..)
            | Params::Ellipsis
            | Params::Materialization => self.fmt_with_type(output, write_type),
        }
    }

    pub fn list(params: ParamList, ret: Type) -> Self {
        Self {
            params: Params::List(params),
            ret,
        }
    }

    pub fn ellipsis(ret: Type) -> Self {
        Self {
            params: Params::Ellipsis,
            ret,
        }
    }

    pub fn param_spec(p: Type, ret: Type) -> Self {
        Self {
            params: Params::ParamSpec(Box::default(), p),
            ret,
        }
    }

    pub fn concatenate(args: Box<[PrefixParam]>, param_spec: Type, ret: Type) -> Self {
        Self {
            params: Params::ParamSpec(args, param_spec),
            ret,
        }
    }

    /// Return a new Callable with the first parameter removed (the `self` param for bound methods).
    pub fn strip_first_param(&self) -> Option<Self> {
        self.split_first_param(&mut Owner::new())
            .map(|(_, rest)| rest)
    }

    pub fn split_first_param<'a>(&'a self, owner: &'a mut Owner<Type>) -> Option<(&'a Type, Self)> {
        match self {
            Self {
                params: Params::List(params),
                ret,
            } => {
                let (first, rest) = params.0.split_first()?;
                if let Param::Varargs(_, first) = first {
                    Some((first, self.clone()))
                } else {
                    Some((
                        first.as_type(),
                        Self::list(ParamList(rest.to_vec()), ret.clone()),
                    ))
                }
            }
            Self {
                params: Params::ParamSpec(ts, p),
                ret,
            } => {
                let (first, rest) = ts.split_first()?;
                Some((
                    first.ty(),
                    Self::concatenate(rest.iter().cloned().collect(), p.clone(), ret.clone()),
                ))
            }
            Self {
                params: Params::Ellipsis,
                ret: _,
            } => Some((owner.push(Type::any_implicit()), self.clone())),
            _ => None,
        }
    }

    pub fn get_first_param(&self) -> Option<Type> {
        self.split_first_param(&mut Owner::new())
            .map(|(first, _)| first.clone())
    }

    pub fn is_typeguard(&self) -> bool {
        matches!(
            self,
            Self {
                params: _,
                ret: Type::TypeGuard(_)
            }
        )
    }

    pub fn is_typeis(&self) -> bool {
        matches!(
            self,
            Self {
                params: _,
                ret: Type::TypeIs(_),
            }
        )
    }

    pub fn subst_self_type_mut(&mut self, replacement: &Type) {
        self.visit_mut(&mut |t: &mut Type| t.subst_self_type_mut(replacement));
    }

    pub fn arg_counts(&self) -> ArgCounts {
        self.params.arg_counts()
    }
}

impl Param {
    fn fmt_default(&self, default: &Option<DefaultValue>) -> String {
        match default {
            Some(DefaultValue {
                display: Some(text),
                ..
            }) => text.clone(),
            Some(DefaultValue {
                ty: Type::Literal(lit),
                ..
            }) => format!("{}", lit.value),
            Some(DefaultValue { ty: Type::None, .. }) => "None".to_owned(),
            _ => "...".to_owned(),
        }
    }

    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        match self {
            Param::PosOnly(None, ty, Required::Required) => write_type(ty, output),
            Param::PosOnly(None, ty, Required::Optional(default)) => {
                output.write_str("_: ")?;
                write_type(ty, output)?;
                output.write_str(" = ")?;
                output.write_str(&self.fmt_default(default))
            }
            Param::PosOnly(Some(name), ty, Required::Required)
            | Param::Pos(name, ty, Required::Required)
            | Param::KwOnly(name, ty, Required::Required) => {
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)
            }
            Param::PosOnly(Some(name), ty, Required::Optional(default))
            | Param::Pos(name, ty, Required::Optional(default))
            | Param::KwOnly(name, ty, Required::Optional(default)) => {
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)?;
                output.write_str(" = ")?;
                output.write_str(&self.fmt_default(default))
            }
            Param::Varargs(Some(name), ty) => {
                output.write_str("*")?;
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)
            }
            Param::Varargs(None, ty) => {
                output.write_str("*")?;
                write_type(ty, output)
            }
            Param::Kwargs(Some(name), ty) => {
                output.write_str("**")?;
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)
            }
            Param::Kwargs(None, ty) => {
                output.write_str("**")?;
                write_type(ty, output)
            }
        }
    }

    pub fn name(&self) -> Option<&Name> {
        match self {
            Param::PosOnly(name, ..) | Param::Varargs(name, ..) | Param::Kwargs(name, ..) => {
                name.as_ref()
            }
            Param::Pos(name, ..) | Param::KwOnly(name, ..) => Some(name),
        }
    }

    pub fn as_type(&self) -> &Type {
        match self {
            Param::PosOnly(_, ty, _)
            | Param::Pos(_, ty, _)
            | Param::Varargs(_, ty)
            | Param::KwOnly(_, ty, _)
            | Param::Kwargs(_, ty) => ty,
        }
    }

    pub fn as_type_mut(&mut self) -> &mut Type {
        match self {
            Param::PosOnly(_, ty, _)
            | Param::Pos(_, ty, _)
            | Param::Varargs(_, ty)
            | Param::KwOnly(_, ty, _)
            | Param::Kwargs(_, ty) => ty,
        }
    }

    pub fn is_required(&self) -> bool {
        match self {
            Param::PosOnly(_, _, Required::Required)
            | Param::Pos(_, _, Required::Required)
            | Param::KwOnly(_, _, Required::Required) => true,
            _ => false,
        }
    }

    /// Format a parameter for display using the proper type display infrastructure.
    /// This ensures consistent formatting with default values, position-only markers, etc.
    ///
    /// This is similar to the `Display` impl, but allows passing in a `TypeDisplayContext`
    /// for context-aware formatting (e.g., disambiguating types with the same name).
    pub fn format_for_signature(&self, type_ctx: &TypeDisplayContext) -> String {
        format!(
            "{}",
            Fmt(|f| {
                let mut output = DisplayOutput::new(type_ctx, f);
                self.fmt_with_type(&mut output, &|ty, o| {
                    type_ctx.fmt_helper_generic(ty, false, o)
                })
            })
        )
    }
}

impl Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ctx = TypeDisplayContext::new(&[self.as_type()]);
        let mut output = DisplayOutput::new(&ctx, f);
        self.fmt_with_type(&mut output, &|t, o| ctx.fmt_helper_generic(t, false, o))
    }
}

impl FunctionKind {
    pub fn from_name(
        module: Module,
        cls: Option<Class>,
        func: &Name,
        def_index: Option<FuncDefIndex>,
        outer_funcs: Option<Name>,
    ) -> Self {
        match (module.name().as_str(), cls.as_ref(), func.as_str()) {
            ("builtins", None, "isinstance") => Self::IsInstance,
            ("builtins", None, "issubclass") => Self::IsSubclass,
            ("builtins", None, "classmethod") => Self::ClassMethod,
            ("dataclasses", None, "dataclass") => Self::Dataclass,
            ("dataclasses", None, "field") => Self::DataclassField,
            ("dataclasses", None, "replace") => Self::DataclassReplace,
            ("dataclasses", None, "asdict") => Self::DataclassAsdict,
            ("typing" | "typing_extensions", None, "overload") => Self::Overload,
            ("typing" | "typing_extensions", None, "override") => Self::Override,
            ("typing" | "typing_extensions", None, "cast") => Self::Cast,
            ("typing" | "typing_extensions", None, "assert_type") => Self::AssertType,
            ("typing" | "typing_extensions", None, "reveal_type") => Self::RevealType,
            ("typing" | "typing_extensions", None, "final") => Self::Final,
            ("typing" | "typing_extensions", None, "runtime_checkable") => Self::RuntimeCheckable,
            ("typing" | "typing_extensions", None, "dataclass_transform") => {
                Self::DataclassTransform
            }
            ("abc", None, "abstractmethod") => Self::AbstractMethod,
            ("functools", None, "total_ordering") => Self::TotalOrdering,
            ("typing" | "typing_extensions", None, "disjoint_base") => Self::DisjointBase,
            ("numba.core.decorators", None, "jit") => Self::NumbaJit,
            ("numba.core.decorators", None, "njit") => Self::NumbaNjit,
            ("shape_extensions", None, "uses_shape_dsl") => Self::UsesShapeDsl,
            ("sqlalchemy.orm", None, "mapped_column")
            | ("sqlalchemy.orm._orm_constructors", None, "mapped_column") => {
                Self::SqlAlchemyMappedColumn
            }
            _ => Self::Def(Arc::new(FuncId {
                module,
                cls,
                name: func.clone(),
                def_index,
                outer_funcs,
            })),
        }
    }

    pub fn module_name(&self) -> ModuleName {
        match self {
            Self::IsInstance => ModuleName::builtins(),
            Self::IsSubclass => ModuleName::builtins(),
            Self::ClassMethod => ModuleName::builtins(),
            Self::Dataclass => ModuleName::dataclasses(),
            Self::DataclassField => ModuleName::dataclasses(),
            Self::DataclassReplace => ModuleName::dataclasses(),
            Self::DataclassAsdict => ModuleName::dataclasses(),
            Self::DataclassTransform => ModuleName::typing(),
            Self::Final => ModuleName::typing(),
            Self::Overload => ModuleName::typing(),
            Self::Override => ModuleName::typing(),
            Self::Cast => ModuleName::typing(),
            Self::AssertType => ModuleName::typing(),
            Self::RevealType => ModuleName::typing(),
            Self::RuntimeCheckable => ModuleName::typing(),
            Self::CallbackProtocol(cls) => cls.qname().module_name(),
            Self::AbstractMethod => ModuleName::abc(),
            Self::TotalOrdering => ModuleName::functools(),
            Self::DisjointBase => ModuleName::typing(),
            Self::NumbaJit => ModuleName::from_str("numba"),
            Self::NumbaNjit => ModuleName::from_str("numba"),
            Self::SqlAlchemyMappedColumn => ModuleName::from_str("sqlalchemy.orm"),
            Self::Def(func_id) => func_id.module.name().dupe(),
            Self::ShapeDsl(id, _, _) => id.module.name().dupe(),
            Self::UsesShapeDsl => ModuleName::from_str("shape_extensions"),
        }
    }

    pub fn function_name(&self) -> Cow<'_, Name> {
        match self {
            Self::IsInstance => Cow::Owned(Name::new_static("isinstance")),
            Self::IsSubclass => Cow::Owned(Name::new_static("issubclass")),
            Self::ClassMethod => Cow::Owned(Name::new_static("classmethod")),
            Self::Dataclass => Cow::Owned(Name::new_static("dataclass")),
            Self::DataclassField => Cow::Owned(Name::new_static("field")),
            Self::DataclassReplace => Cow::Owned(Name::new_static("replace")),
            Self::DataclassAsdict => Cow::Owned(Name::new_static("asdict")),
            Self::DataclassTransform => Cow::Owned(Name::new_static("dataclass_transform")),
            Self::Final => Cow::Owned(Name::new_static("final")),
            Self::Overload => Cow::Owned(Name::new_static("overload")),
            Self::Override => Cow::Owned(Name::new_static("override")),
            Self::Cast => Cow::Owned(Name::new_static("cast")),
            Self::AssertType => Cow::Owned(Name::new_static("assert_type")),
            Self::RevealType => Cow::Owned(Name::new_static("reveal_type")),
            Self::RuntimeCheckable => Cow::Owned(Name::new_static("runtime_checkable")),
            Self::CallbackProtocol(_) => Cow::Owned(dunder::CALL),
            Self::AbstractMethod => Cow::Owned(Name::new_static("abstractmethod")),
            Self::TotalOrdering => Cow::Owned(Name::new_static("total_ordering")),
            Self::DisjointBase => Cow::Owned(Name::new_static("disjoint_base")),
            Self::NumbaJit => Cow::Owned(Name::new_static("jit")),
            Self::NumbaNjit => Cow::Owned(Name::new_static("njit")),
            Self::SqlAlchemyMappedColumn => Cow::Owned(Name::new_static("mapped_column")),
            Self::Def(func_id) => Cow::Borrowed(&func_id.name),
            Self::ShapeDsl(id, _, _) => Cow::Borrowed(&id.name),
            Self::UsesShapeDsl => Cow::Owned(Name::new_static("uses_shape_dsl")),
        }
    }

    pub fn class(&self) -> Option<Class> {
        match self {
            Self::IsInstance => None,
            Self::IsSubclass => None,
            Self::ClassMethod => None,
            Self::Dataclass => None,
            Self::DataclassField => None,
            Self::DataclassReplace => None,
            Self::DataclassAsdict => None,
            Self::DataclassTransform => None,
            Self::Final => None,
            Self::Overload => None,
            Self::Override => None,
            Self::Cast => None,
            Self::AssertType => None,
            Self::RevealType => None,
            Self::RuntimeCheckable => None,
            Self::NumbaJit => None,
            Self::NumbaNjit => None,
            Self::SqlAlchemyMappedColumn => None,
            Self::CallbackProtocol(cls) => Some(cls.class_object().dupe()),
            Self::AbstractMethod => None,
            Self::TotalOrdering => None,
            Self::DisjointBase => None,
            Self::Def(func_id) => func_id.cls.clone(),
            Self::ShapeDsl(id, _, _) => id.cls.clone(),
            Self::UsesShapeDsl => None,
        }
    }

    pub fn outer_funcs(&self) -> Option<&Name> {
        match self {
            Self::Def(func_id) | Self::ShapeDsl(func_id, _, _) => func_id.outer_funcs.as_ref(),
            _ => None,
        }
    }

    pub fn format(&self, current_module: ModuleName) -> String {
        FuncId::format_impl(
            self.module_name(),
            self.class(),
            self.function_name().as_ref(),
            current_module,
        )
    }

    /// Does this decorator require special-casing to be signature-preserving?
    pub fn is_signature_preserving_decorator(&self) -> bool {
        match self {
            Self::NumbaJit | Self::NumbaNjit => true,
            _ => false,
        }
    }
}

pub fn unexpected_keyword(error: &dyn Fn(String), func: &str, keyword: &Keyword) {
    let desc = if let Some(id) = &keyword.arg {
        format!(" `{id}`")
    } else {
        "".to_owned()
    };
    error(format!("`{func}` got an unexpected keyword argument{desc}"));
}

#[cfg(test)]
mod tests {
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_util::visit::Visit;
    use pyrefly_util::visit::VisitMut;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;

    use crate::callable::Callable;
    use crate::callable::DefaultValue;
    use crate::callable::Param;
    use crate::callable::ParamList;
    use crate::callable::PrefixParam;
    use crate::callable::Required;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::Type;

    #[test]
    fn test_arg_counts_positional() {
        // (x: Any, /, y: Any = ...) -> None
        let callable = Callable::list(
            ParamList::new(vec![
                Param::PosOnly(
                    Some(Name::new("x")),
                    Type::any_implicit(),
                    Required::Required,
                ),
                Param::Pos(
                    Name::new("y"),
                    Type::any_implicit(),
                    Required::Optional(None),
                ),
            ]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 1);
        assert_eq!(counts.positional.max, Some(2));
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, Some(1));
    }

    #[test]
    fn test_arg_counts_keyword() {
        // (*, x: Any, y: Any = ...) -> None
        let callable = Callable::list(
            ParamList::new(vec![
                Param::KwOnly(Name::new("x"), Type::any_implicit(), Required::Required),
                Param::KwOnly(
                    Name::new("y"),
                    Type::any_implicit(),
                    Required::Optional(None),
                ),
            ]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, Some(0));
        assert_eq!(counts.keyword.min, 1);
        assert_eq!(counts.keyword.max, Some(2));
    }

    #[test]
    fn test_arg_counts_varargs() {
        // (*args) -> None
        let callable = Callable::list(
            ParamList::new(vec![Param::Varargs(None, Type::any_implicit())]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, Some(0));
    }

    #[test]
    fn test_arg_counts_kwargs() {
        // (**kwargs) -> None
        let callable = Callable::list(
            ParamList::new(vec![Param::Kwargs(None, Type::any_implicit())]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, Some(0));
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }

    #[test]
    fn test_arg_counts_paramlist() {
        // (w, /, x, *args, y, z=...) -> None
        let callable = Callable::list(
            ParamList::new(vec![
                Param::PosOnly(
                    Some(Name::new("w")),
                    Type::any_implicit(),
                    Required::Required,
                ),
                Param::Pos(Name::new("x"), Type::any_implicit(), Required::Required),
                Param::Varargs(None, Type::any_implicit()),
                Param::KwOnly(Name::new("y"), Type::any_implicit(), Required::Required),
                Param::KwOnly(
                    Name::new("z"),
                    Type::any_implicit(),
                    Required::Optional(None),
                ),
            ]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 1);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 1);
        assert_eq!(counts.keyword.max, Some(3));
    }

    #[test]
    fn test_arg_counts_ellipsis() {
        let callable = Callable::ellipsis(Type::None);
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }

    #[test]
    fn test_arg_counts_paramspec() {
        let callable = Callable::concatenate(
            vec![
                PrefixParam::new(Type::None, Required::Required),
                PrefixParam::new(Type::None, Required::Required),
            ]
            .into_boxed_slice(),
            Type::any_implicit(),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 2);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }

    #[test]
    fn test_default_value_visit_delegates_to_ty() {
        let q = Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("T"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
        let quantified_ty = Type::Quantified(Box::new(q));
        let default = DefaultValue::with_display(quantified_ty.clone(), "default".to_owned());

        // Visit should yield the inner type from ty, not the display metadata.
        let mut visited = Vec::new();
        default.visit(&mut |ty: &Type| visited.push(ty.clone()));
        assert_eq!(visited, vec![quantified_ty]);
    }

    #[test]
    fn test_default_value_visit_mut_delegates_to_ty() {
        let q = Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), 1),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("T"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
        let mut default =
            DefaultValue::with_display(Type::Quantified(Box::new(q)), "default".to_owned());

        // VisitMut should be able to mutate the inner type.
        default.visit_mut(&mut |ty: &mut Type| {
            *ty = Type::None;
        });
        assert_eq!(default.ty, Type::None);
        // Display metadata should be unaffected.
        assert_eq!(default.display, Some("default".to_owned()));
    }
}
