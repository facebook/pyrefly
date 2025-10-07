/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::qname::QName;
use pyrefly_util::assert_words;
use pyrefly_util::display::commas_iter;
use pyrefly_util::uniques::Unique;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::callable::Callable;
use crate::callable::FuncId;
use crate::callable::FuncMetadata;
use crate::callable::Function;
use crate::callable::FunctionKind;
use crate::callable::Param;
use crate::callable::ParamList;
use crate::callable::Params;
use crate::class::Class;
use crate::class::ClassKind;
use crate::class::ClassType;
use crate::keywords::DataclassTransformKeywords;
use crate::keywords::KwCall;
use crate::literal::Lit;
use crate::module::ModuleType;
use crate::param_spec::ParamSpec;
use crate::quantified::Quantified;
use crate::quantified::QuantifiedKind;
use crate::simplify::unions;
use crate::special_form::SpecialForm;
use crate::stdlib::Stdlib;
use crate::tuple::Tuple;
use crate::type_var::PreInferenceVariance;
use crate::type_var::Restriction;
use crate::type_var::TypeVar;
use crate::type_var_tuple::TypeVarTuple;
use crate::typed_dict::TypedDict;

/// An introduced synthetic variable to range over as yet unknown types.
#[derive(Debug, Copy, Clone, Dupe, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Var(Unique);

impl Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl Var {
    pub const ZERO: Var = Var(Unique::ZERO);

    pub fn new(uniques: &UniqueFactory) -> Self {
        Self(uniques.fresh())
    }

    pub fn to_type(self) -> Type {
        Type::Var(self)
    }
}

#[derive(PartialEq, Eq)]
pub enum TParamsSource {
    Class,
    TypeAlias,
    Function,
}

impl Display for TParamsSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Class => write!(f, "class"),
            Self::TypeAlias => write!(f, "type alias"),
            Self::Function => write!(f, "function"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TParam {
    pub quantified: Quantified,
    pub variance: PreInferenceVariance,
}

impl Display for TParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl TParam {
    pub fn name(&self) -> &Name {
        self.quantified.name()
    }

    pub fn default(&self) -> Option<&Type> {
        self.quantified.default()
    }

    pub fn restriction(&self) -> &Restriction {
        self.quantified.restriction()
    }
}

/// Wraps a vector of type parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TParams(Vec<TParam>);

/// Implement `VisitMut` for `Arc<TParams>` as a no-op.
///
/// This is not technically correct, because TParams can contain types inside
/// the bounds on `Quantified`, but we only use `VisitMut` to eliminate `Var`s,
/// and we do not need to eliminate vars on tparams.
///
/// Without making this simplifying assumption we would not be able to use `Arc`
/// to share the `TParams`.
impl VisitMut<Type> for Arc<TParams> {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for Arc<TParams> {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.as_ref().recurse(f);
    }
}

impl Display for TParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", commas_iter(|| self.0.iter()))
    }
}

impl TParams {
    pub fn new(tparams: Vec<TParam>) -> TParams {
        Self(tparams)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &TParam> {
        self.0.iter()
    }

    pub fn quantifieds(&self) -> impl ExactSizeIterator<Item = &Quantified> + '_ {
        self.0.iter().map(|x| &x.quantified)
    }

    pub fn contain_type_var_tuple(&self) -> bool {
        self.0
            .iter()
            .any(|tparam| tparam.quantified.kind() == QuantifiedKind::TypeVarTuple)
    }

    pub fn as_vec(&self) -> &[TParam] {
        &self.0
    }

    pub fn extend(&mut self, other: &TParams) {
        self.0.extend(other.iter().cloned());
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TArgs(Box<(Arc<TParams>, Box<[Type]>)>);

impl TArgs {
    pub fn new(tparams: Arc<TParams>, targs: Vec<Type>) -> Self {
        if tparams.len() != targs.len() {
            panic!("TParams and TArgs must have the same length");
        }
        Self(Box::new((tparams, targs.into_boxed_slice())))
    }

    pub fn tparams(&self) -> &TParams {
        &self.0.0
    }

    pub fn iter_paired(&self) -> impl ExactSizeIterator<Item = (&TParam, &Type)> {
        self.0.0.iter().zip(self.0.1.iter())
    }

    pub fn iter_paired_mut(&mut self) -> impl ExactSizeIterator<Item = (&TParam, &mut Type)> {
        self.0.0.iter().zip(self.0.1.iter_mut())
    }

    pub fn len(&self) -> usize {
        self.0.1.len()
    }

    pub fn as_slice(&self) -> &[Type] {
        &self.0.1
    }

    pub fn as_mut(&mut self) -> &mut [Type] {
        &mut self.0.1
    }

    pub fn is_empty(&self) -> bool {
        self.0.1.is_empty()
    }

    /// Apply a substitution to type arguments.
    ///
    /// This is useful mainly to re-express ancestors (which, in the MRO, are in terms of class
    /// type parameters)
    ///
    /// This is mainly useful to take ancestors coming from the MRO (which are always in terms
    /// of the current class's type parameters) and re-express them in terms of the current
    /// class specialized with type arguments.
    pub fn substitute_with(&self, substitution: &Substitution) -> Self {
        let tys = self
            .0
            .1
            .iter()
            .map(|ty| substitution.substitute_into(ty.clone()))
            .collect();
        Self::new(self.0.0.dupe(), tys)
    }

    pub fn substitution_map(&self) -> SmallMap<&Quantified, &Type> {
        let tparams = self.tparams();
        let tys = self.as_slice();
        tparams.quantifieds().zip(tys.iter()).collect()
    }

    pub fn substitution<'a>(&'a self) -> Substitution<'a> {
        Substitution(self.substitution_map())
    }

    pub fn substitute_into_mut(&self, ty: &mut Type) {
        self.substitution().substitute_into_mut(ty)
    }

    pub fn substitute_into(&self, mut ty: Type) -> Type {
        self.substitute_into_mut(&mut ty);
        ty
    }
}

pub struct Substitution<'a>(SmallMap<&'a Quantified, &'a Type>);

impl<'a> Substitution<'a> {
    pub fn substitute_into_mut(&self, ty: &mut Type) {
        ty.subst_mut(&self.0)
    }

    pub fn substitute_into(&self, ty: Type) -> Type {
        ty.subst(&self.0)
    }
}

/// The types of Never. Prefer later ones where we have multiple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum NeverStyle {
    NoReturn,
    Never,
}

/// The types of Any. Prefer later ones where we have multiple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum AnyStyle {
    /// The user wrote `Any` literally.
    Explicit,
    /// The user didn't write a type, so we inferred `Any`.
    Implicit,
    /// There was an error, so we made up `Any`.
    /// If this `Any` is used in an error position, don't report another error.
    Error,
}

impl AnyStyle {
    pub fn propagate(self) -> Type {
        match self {
            Self::Implicit | Self::Error => Type::Any(self),
            Self::Explicit => Type::Any(Self::Implicit),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum TypeAliasStyle {
    /// A type alias declared with the `type` keyword
    Scoped,
    /// A type alias declared with a `: TypeAlias` annotation
    LegacyExplicit,
    /// An unannotated assignment that may be either an implicit type alias or an untyped value
    LegacyImplicit,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TypeAlias {
    pub name: Box<Name>,
    ty: Box<Type>,
    pub style: TypeAliasStyle,
}

impl TypeAlias {
    pub fn new(name: Name, ty: Type, style: TypeAliasStyle) -> Self {
        Self {
            name: Box::new(name),
            ty: Box::new(ty),
            style,
        }
    }

    /// Gets the type contained within the type alias for use in a value
    /// position - for example, for a function call or attribute access.
    pub fn as_value(&self, stdlib: &Stdlib) -> Type {
        if self.style == TypeAliasStyle::Scoped {
            stdlib.type_alias_type().clone().to_type()
        } else {
            *self.ty.clone()
        }
    }

    /// Gets the type contained within the type alias for use in a type
    /// position - for example, in a variable type annotation. Note that
    /// the caller is still responsible for untyping the type. That is,
    /// `type X = int` is represented as `TypeAlias(X, type[int])`, and
    /// `as_type` returns `type[int]`; the caller must turn it into `int`.
    pub fn as_type(&self) -> Type {
        *self.ty.clone()
    }

    pub fn fmt_with_type<'a, D: Display + 'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        wrap: &'a impl Fn(&'a Type) -> D,
        tparams: Option<&TParams>,
    ) -> fmt::Result {
        match (&self.style, tparams) {
            (TypeAliasStyle::LegacyImplicit, _) => {
                write!(f, "{}", wrap(&self.ty))
            }
            (_, None) => {
                write!(f, "TypeAlias[{}, {}]", self.name, wrap(&self.ty))
            }
            (_, Some(tparams)) => {
                write!(
                    f,
                    "TypeAlias[{}[{}], {}]",
                    self.name,
                    commas_iter(|| tparams.iter()),
                    wrap(&self.ty)
                )
            }
        }
    }
}

assert_words!(Type, 4);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum CalleeKind {
    Callable,
    Function(FunctionKind),
    Class(ClassKind),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct BoundMethod {
    /// Type of the self/cls argument,
    pub obj: Type,
    /// Type of the function.
    pub func: BoundMethodType,
}

impl BoundMethod {
    pub fn with_bound_object(&self, obj: Type) -> Self {
        Self {
            obj,
            func: self.func.clone(),
        }
    }

    pub fn as_type(self) -> Type {
        Type::BoundMethod(Box::new(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum BoundMethodType {
    Function(Function),
    Forall(Forall<Function>),
    Overload(Overload),
}

impl BoundMethodType {
    pub fn as_type(self) -> Type {
        match self {
            Self::Function(func) => Type::Function(Box::new(func)),
            Self::Forall(forall) => Forallable::Function(forall.body).forall(forall.tparams),
            Self::Overload(overload) => Type::Overload(overload),
        }
    }

    pub fn subst_self_type_mut(&mut self, replacement: &Type) {
        match self {
            Self::Function(func) => func.signature.subst_self_type_mut(replacement),
            Self::Forall(forall) => forall.body.signature.subst_self_type_mut(replacement),
            Self::Overload(overload) => {
                for sig in overload.signatures.iter_mut() {
                    sig.subst_self_type_mut(replacement)
                }
            }
        }
    }

    pub fn metadata(&self) -> &FuncMetadata {
        match self {
            Self::Function(func) => &func.metadata,
            Self::Forall(forall) => &forall.body.metadata,
            Self::Overload(overload) => &overload.metadata,
        }
    }

    fn is_typeguard(&self) -> bool {
        match self {
            Self::Function(func) => func.signature.is_typeguard(),
            Self::Forall(forall) => forall.body.signature.is_typeguard(),
            Self::Overload(overload) => overload.is_typeguard(),
        }
    }

    fn is_typeis(&self) -> bool {
        match self {
            Self::Function(func) => func.signature.is_typeis(),
            Self::Forall(forall) => forall.body.signature.is_typeis(),
            Self::Overload(overload) => overload.is_typeis(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Overload {
    pub signatures: Vec1<OverloadType>,
    pub metadata: Box<FuncMetadata>,
}

impl Overload {
    fn is_typeguard(&self) -> bool {
        self.signatures.iter().any(|t| t.is_typeguard())
    }

    fn is_typeis(&self) -> bool {
        self.signatures.iter().any(|t| t.is_typeis())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum OverloadType {
    Function(Function),
    Forall(Forall<Function>),
}

impl OverloadType {
    pub fn as_type(&self) -> Type {
        match self {
            Self::Function(f) => Type::Function(Box::new(f.clone())),
            Self::Forall(forall) => {
                Forallable::Function(forall.body.clone()).forall(forall.tparams.clone())
            }
        }
    }

    fn subst_self_type_mut(&mut self, replacement: &Type) {
        match self {
            Self::Function(f) => f.signature.subst_self_type_mut(replacement),
            Self::Forall(forall) => forall.body.signature.subst_self_type_mut(replacement),
        }
    }

    fn is_typeguard(&self) -> bool {
        match self {
            Self::Function(f) => f.signature.is_typeguard(),
            Self::Forall(forall) => forall.body.signature.is_typeguard(),
        }
    }

    fn is_typeis(&self) -> bool {
        match self {
            Self::Function(f) => f.signature.is_typeis(),
            Self::Forall(forall) => forall.body.signature.is_typeis(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Forall<T> {
    pub tparams: Arc<TParams>,
    pub body: T,
}

impl Forall<Forallable> {
    pub fn apply_targs(self, targs: TArgs) -> Type {
        targs.substitute_into(self.body.as_type())
    }
}

/// These are things that can have Forall around them, so often you see `Forall<Forallable>`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Forallable {
    TypeAlias(TypeAlias),
    Function(Function),
    Callable(Callable),
}

impl Forallable {
    pub fn forall(self, tparams: Arc<TParams>) -> Type {
        if tparams.is_empty() {
            self.as_type()
        } else {
            Type::Forall(Box::new(Forall {
                tparams,
                body: self,
            }))
        }
    }

    pub fn name(&self) -> Name {
        match self {
            Self::Function(func) => func.metadata.kind.as_func_id().func,
            Self::Callable(_) => Name::new_static("<callable>"),
            Self::TypeAlias(ta) => (*ta.name).clone(),
        }
    }

    pub fn as_type(self) -> Type {
        match self {
            Self::Function(func) => Type::Function(Box::new(func)),
            Self::Callable(callable) => Type::Callable(Box::new(callable)),
            Self::TypeAlias(ta) => Type::TypeAlias(ta),
        }
    }

    fn is_typeguard(&self) -> bool {
        match self {
            Self::Function(func) => func.signature.is_typeguard(),
            Self::Callable(callable) => callable.is_typeguard(),
            Self::TypeAlias(_) => false,
        }
    }

    fn is_typeis(&self) -> bool {
        match self {
            Self::Function(func) => func.signature.is_typeis(),
            Self::Callable(callable) => callable.is_typeis(),
            Self::TypeAlias(_) => false,
        }
    }
}

/// The second argument (implicit or explicit) to a super() call.
/// Either an instance of a class (inside an instance method) or a
/// class object (inside a classmethod or staticmethod)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum SuperObj {
    Instance(ClassType),
    Class(ClassType),
}

// Note: The fact that Literal and LiteralString are at the front is important for
// optimisations in `unions_with_literals`.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum Type {
    Literal(Lit),
    LiteralString,
    /// typing.Callable
    Callable(Box<Callable>),
    /// A function declared using the `def` keyword.
    /// Note that the FunctionKind metadata doesn't participate in subtyping, and thus two types with distinct metadata are still subtypes.
    Function(Box<Function>),
    /// A method of a class.
    BoundMethod(Box<BoundMethod>),
    /// An overloaded function.
    Overload(Overload),
    Union(Vec<Type>),
    #[allow(dead_code)] // Not currently used, but may be in the future
    Intersect(Vec<Type>),
    /// A class definition has type `Type::ClassDef(cls)`. This type
    /// has special value semantics, and can also be implicitly promoted
    /// to `Type::Type(box Type::ClassType(cls, default_targs))` by looking
    /// up the class `tparams` and setting defaults using gradual types: for
    /// example `list` in an annotation position means `list[Any]`.
    ClassDef(Class),
    /// A value that indicates a concrete, instantiated type with known type
    /// arguments that are validated against the class type parameters. If the
    /// class is not generic, the arguments are empty.
    ///
    /// Instances of classes have this type, and a term of the form `C[arg1, arg2]`
    /// would have the form `Type::Type(box Type::ClassType(C, [arg1, arg2]))`.
    ClassType(ClassType),
    /// Instances of TypedDicts have this type, and a term of the form `TD[arg1, arg2]`
    /// would have the form `Type::Type(box Type::TypedDict(TD, [arg1, arg2]))`. Note
    /// that TypedDict class definitions are still represented as `ClassDef(TD)`, just
    /// like regular classes.
    TypedDict(TypedDict),
    /// Represents a "partial" version of a TypedDict that can be merged into the TypedDict
    /// (e.g., via its `update` method).
    /// For a TypedDict type `C`, `Partial[C]` represents an object with any subset of read-write
    /// keys from `C`, where each present key has the same value type as in `C`.
    PartialTypedDict(TypedDict),
    Tuple(Tuple),
    Module(ModuleType),
    Forall(Box<Forall<Forallable>>),
    Var(Var),
    /// The type of a value which is annotated with a type var.
    Quantified(Box<Quantified>),
    /// The type of type var _value_ itself, after it has been bound to a function or a class.
    /// This is equivalent to Type::TypeVar/ParamSpec/TypeVarTuple as a value, but when used
    /// in a type annotation, it becomes Type::Quantified.
    QuantifiedValue(Box<Quantified>),
    TypeGuard(Box<Type>),
    TypeIs(Box<Type>),
    Unpack(Box<Type>),
    TypeVar(TypeVar),
    ParamSpec(ParamSpec),
    TypeVarTuple(TypeVarTuple),
    SpecialForm(SpecialForm),
    Concatenate(Box<[Type]>, Box<Type>),
    ParamSpecValue(ParamList),
    /// The type of a value which is annotated with `P.args`.
    Args(Box<Quantified>),
    /// The type of a value which is annotated with `P.kwargs`.
    Kwargs(Box<Quantified>),
    /// The type of the _value_ `P.args`.
    /// This is equivalent to `typing.ParamSpecArgs`, but when used in a type annotation it
    /// becomes Type::Args.
    ArgsValue(Box<Quantified>),
    /// The type of the _value_ `P.kwargs`.
    /// This is equivalent to `typing.ParamSpecKwargs`, but when used in a type annotation it
    /// becomes Type::Kwargs.
    KwargsValue(Box<Quantified>),
    /// Used to represent a type that has a value representation, e.g. a class
    Type(Box<Type>),
    Ellipsis,
    Any(AnyStyle),
    Never(NeverStyle),
    TypeAlias(TypeAlias),
    /// Represents the result of a super() call. The first ClassType is the point in the MRO that attribute lookup
    /// on the super instance should start at (*not* the class passed to the super() call), and the second
    /// ClassType is the second argument (implicit or explicit) to the super() call. For example, in:
    ///   class A: ...
    ///   class B(A): ...
    ///   class C(B):
    ///     def f(self):
    ///       super(B, self)
    /// attribute lookup should be done on the class above `B` in the MRO of the type of `self` -
    /// that is, attribute lookup should be done on class `A`. And the type of `self` is class `C`.
    /// So the super instance is represented as `SuperInstance[ClassType(A), ClassType(C)]`.
    SuperInstance(Box<(ClassType, SuperObj)>),
    /// typing.Self with the class definition it appears in. We store the latter as a ClassType
    /// because of how often we need the type of an instance of the class.
    SelfType(ClassType),
    /// Wraps the result of a function call whose keyword arguments have typing effects, like
    /// `typing.dataclass_transform(...)`.
    KwCall(Box<KwCall>),
    None,
}

impl Visit for Type {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Self)) {
        match self {
            Type::Literal(x) => x.visit(f),
            Type::LiteralString => {}
            Type::Callable(x) => x.visit(f),
            Type::Function(x) => x.visit(f),
            Type::BoundMethod(x) => x.visit(f),
            Type::Overload(x) => x.visit(f),
            Type::Union(x) => x.visit(f),
            Type::Intersect(x) => x.visit(f),
            Type::ClassDef(x) => x.visit(f),
            Type::ClassType(x) => x.visit(f),
            Type::TypedDict(x) => x.visit(f),
            Type::PartialTypedDict(x) => x.visit(f),
            Type::Tuple(x) => x.visit(f),
            Type::Module(x) => x.visit(f),
            Type::Forall(x) => x.visit(f),
            Type::Var(x) => x.visit(f),
            Type::Quantified(x) => x.visit(f),
            Type::QuantifiedValue(x) => x.visit(f),
            Type::TypeGuard(x) => x.visit(f),
            Type::TypeIs(x) => x.visit(f),
            Type::Unpack(x) => x.visit(f),
            Type::TypeVar(x) => x.visit(f),
            Type::ParamSpec(x) => x.visit(f),
            Type::TypeVarTuple(x) => x.visit(f),
            Type::SpecialForm(x) => x.visit(f),
            Type::Concatenate(x, _) => x.visit(f),
            Type::ParamSpecValue(x) => x.visit(f),
            Type::Args(x) => x.visit(f),
            Type::Kwargs(x) => x.visit(f),
            Type::ArgsValue(x) => x.visit(f),
            Type::KwargsValue(x) => x.visit(f),
            Type::Type(x) => x.visit(f),
            Type::Ellipsis => {}
            Type::Any(x) => x.visit(f),
            Type::Never(x) => x.visit(f),
            Type::TypeAlias(x) => x.visit(f),
            Type::SuperInstance(x) => x.visit(f),
            Type::SelfType(x) => x.visit(f),
            Type::KwCall(x) => x.visit(f),
            Type::None => {}
        }
    }
}

impl VisitMut for Type {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Self)) {
        match self {
            Type::Literal(x) => x.visit_mut(f),
            Type::LiteralString => {}
            Type::Callable(x) => x.visit_mut(f),
            Type::Function(x) => x.visit_mut(f),
            Type::BoundMethod(x) => x.visit_mut(f),
            Type::Overload(x) => x.visit_mut(f),
            Type::Union(x) => x.visit_mut(f),
            Type::Intersect(x) => x.visit_mut(f),
            Type::ClassDef(x) => x.visit_mut(f),
            Type::ClassType(x) => x.visit_mut(f),
            Type::TypedDict(x) => x.visit_mut(f),
            Type::PartialTypedDict(x) => x.visit_mut(f),
            Type::Tuple(x) => x.visit_mut(f),
            Type::Module(x) => x.visit_mut(f),
            Type::Forall(x) => x.visit_mut(f),
            Type::Var(x) => x.visit_mut(f),
            Type::Quantified(x) => x.visit_mut(f),
            Type::QuantifiedValue(x) => x.visit_mut(f),
            Type::TypeGuard(x) => x.visit_mut(f),
            Type::TypeIs(x) => x.visit_mut(f),
            Type::Unpack(x) => x.visit_mut(f),
            Type::TypeVar(x) => x.visit_mut(f),
            Type::ParamSpec(x) => x.visit_mut(f),
            Type::TypeVarTuple(x) => x.visit_mut(f),
            Type::SpecialForm(x) => x.visit_mut(f),
            Type::Concatenate(x, _) => x.visit_mut(f),
            Type::ParamSpecValue(x) => x.visit_mut(f),
            Type::Args(x) => x.visit_mut(f),
            Type::Kwargs(x) => x.visit_mut(f),
            Type::ArgsValue(x) => x.visit_mut(f),
            Type::KwargsValue(x) => x.visit_mut(f),
            Type::Type(x) => x.visit_mut(f),
            Type::Ellipsis => {}
            Type::Any(x) => x.visit_mut(f),
            Type::Never(x) => x.visit_mut(f),
            Type::TypeAlias(x) => x.visit_mut(f),
            Type::SuperInstance(x) => x.visit_mut(f),
            Type::SelfType(x) => x.visit_mut(f),
            Type::KwCall(x) => x.visit_mut(f),
            Type::None => {}
        }
    }
}

impl Type {
    pub fn arc_clone(self: Arc<Self>) -> Self {
        Arc::unwrap_or_clone(self)
    }

    pub fn never() -> Self {
        Type::Never(NeverStyle::Never)
    }

    pub fn is_function_type(&self) -> bool {
        matches!(
            self,
            Type::Function { .. }
                | Type::Overload { .. }
                | Type::BoundMethod { .. }
                | Type::Callable { .. }
        )
    }

    pub fn as_module(&self) -> Option<&ModuleType> {
        match self {
            Type::Module(m) => Some(m),
            _ => None,
        }
    }

    pub fn callable(params: Vec<Param>, ret: Type) -> Self {
        Type::Callable(Box::new(Callable::list(ParamList::new(params), ret)))
    }

    pub fn callable_ellipsis(ret: Type) -> Self {
        Type::Callable(Box::new(Callable::ellipsis(ret)))
    }

    pub fn callable_param_spec(p: Type, ret: Type) -> Self {
        Type::Callable(Box::new(Callable::param_spec(p, ret)))
    }

    pub fn is_union(&self) -> bool {
        matches!(self, Type::Union(_))
    }

    pub fn is_never(&self) -> bool {
        matches!(self, Type::Never(_))
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, Type::Literal(_))
    }

    pub fn is_literal_string(&self) -> bool {
        match self {
            Type::LiteralString => true,
            Type::Literal(l) if l.is_string() => true,
            _ => false,
        }
    }

    pub fn is_unpack(&self) -> bool {
        matches!(self, Type::Unpack(_))
    }

    pub fn callable_concatenate(args: Box<[Type]>, param_spec: Type, ret: Type) -> Self {
        Type::Callable(Box::new(Callable::concatenate(args, param_spec, ret)))
    }

    pub fn type_form(inner: Type) -> Self {
        Type::Type(Box::new(inner))
    }

    pub fn tuple(elts: Vec<Type>) -> Self {
        Type::Tuple(Tuple::concrete(elts))
    }

    pub fn any_tuple() -> Self {
        Type::Tuple(Tuple::Unbounded(Box::new(Type::Any(AnyStyle::Implicit))))
    }

    pub fn is_any(&self) -> bool {
        matches!(self, Type::Any(_))
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Type::Any(AnyStyle::Error))
    }

    pub fn is_kind_type_var_tuple(&self) -> bool {
        match self {
            Type::TypeVarTuple(_) => true,
            Type::Quantified(q) if q.is_type_var_tuple() => true,
            _ => false,
        }
    }

    pub fn is_type_variable(&self) -> bool {
        match self {
            Type::Var(_)
            | Type::Quantified(_)
            | Type::TypeVarTuple(_)
            | Type::TypeVar(_)
            | Type::ParamSpec(_) => true,
            _ => false,
        }
    }

    pub fn is_kind_param_spec(&self) -> bool {
        match self {
            Type::Ellipsis
            | Type::ParamSpec(_)
            | Type::ParamSpecValue(_)
            | Type::Concatenate(_, _) => true,
            Type::Quantified(q) if q.is_param_spec() => true,
            _ => false,
        }
    }

    pub fn is_typeguard(&self) -> bool {
        match self {
            Type::Callable(box callable)
            | Type::Function(box Function {
                signature: callable,
                metadata: _,
            }) => callable.is_typeguard(),
            Type::Forall(forall) => forall.body.is_typeguard(),
            Type::BoundMethod(method) => method.func.is_typeguard(),
            Type::Overload(overload) => overload.is_typeguard(),
            _ => false,
        }
    }

    pub fn is_typeis(&self) -> bool {
        match self {
            Type::Callable(box callable)
            | Type::Function(box Function {
                signature: callable,
                metadata: _,
            }) => callable.is_typeis(),
            Type::Forall(forall) => forall.body.is_typeis(),
            Type::BoundMethod(method) => method.func.is_typeis(),
            Type::Overload(overload) => overload.is_typeis(),
            _ => false,
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Type::None)
    }

    pub fn callee_kind(&self) -> Option<CalleeKind> {
        match self {
            Type::Callable(_) => Some(CalleeKind::Callable),
            Type::Function(func) => Some(CalleeKind::Function(func.metadata.kind.clone())),
            Type::ClassDef(c) => Some(CalleeKind::Class(c.kind())),
            Type::Forall(forall) => forall.body.clone().as_type().callee_kind(),
            Type::Overload(overload) => Some(CalleeKind::Function(overload.metadata.kind.clone())),
            Type::KwCall(call) => call.return_ty.callee_kind(),
            _ => None,
        }
    }

    pub fn subst_mut_fn(&mut self, mp: &mut dyn FnMut(&Quantified) -> Option<Type>) {
        // We are looking up Quantified in a map, and Quantified may contain a Quantified within it.
        // Therefore, to make sure we still get matches, work top-down (not using `transform`).
        fn f(ty: &mut Type, mp: &mut dyn FnMut(&Quantified) -> Option<Type>) {
            if let Type::Quantified(x) = ty {
                if let Some(w) = mp(x) {
                    *ty = w;
                }
            } else {
                ty.recurse_mut(&mut |x| f(x, mp));
            }
        }
        f(self, mp);
    }

    pub fn subst_mut(&mut self, mp: &SmallMap<&Quantified, &Type>) {
        if !mp.is_empty() {
            self.subst_mut_fn(&mut |x| mp.get(x).map(|t| (*t).clone()));
        }
    }

    pub fn subst(mut self, mp: &SmallMap<&Quantified, &Type>) -> Self {
        self.subst_mut(mp);
        self
    }

    pub fn subst_self_special_form_mut(&mut self, self_type: &Type) {
        self.transform_mut(&mut |x| {
            if x == &Type::SpecialForm(SpecialForm::SelfType) {
                *x = self_type.clone()
            }
        });
    }

    pub fn subst_self_type_mut(&mut self, replacement: &Type) {
        self.transform_mut(&mut |t| {
            if matches!(t, Type::SelfType(_)) {
                *t = replacement.clone();
            }
        })
    }

    pub fn for_each_quantified<'a>(&'a self, f: &mut impl FnMut(&'a Quantified)) {
        self.universe(&mut |x| {
            if let Type::Quantified(x) = x {
                f(x);
            }
        })
    }

    pub fn collect_quantifieds<'a>(&'a self, acc: &mut SmallSet<&'a Quantified>) {
        self.for_each_quantified(&mut |q| {
            acc.insert(q);
        });
    }

    pub fn any(&self, mut predicate: impl FnMut(&Type) -> bool) -> bool {
        fn f(ty: &Type, predicate: &mut dyn FnMut(&Type) -> bool, seen: &mut bool) {
            if *seen || predicate(ty) {
                *seen = true;
            } else {
                ty.recurse(&mut |ty| f(ty, predicate, seen));
            }
        }
        let mut seen = false;
        f(self, &mut predicate, &mut seen);
        seen
    }

    /// Calls a `check` function on this type's function metadata if it is a function. Note that we
    /// do *not* recurse into the type to find nested function types.
    fn check_toplevel_func_metadata<T: Default>(&self, check: &dyn Fn(&FuncMetadata) -> T) -> T {
        match self {
            Type::Function(box func)
            | Type::Forall(box Forall {
                tparams: _,
                body: Forallable::Function(func),
            })
            | Type::BoundMethod(box BoundMethod {
                func:
                    BoundMethodType::Function(func)
                    | BoundMethodType::Forall(Forall {
                        tparams: _,
                        body: func,
                    }),
                ..
            }) => check(&func.metadata),
            Type::Overload(overload)
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Overload(overload),
                ..
            }) => check(&overload.metadata),
            _ => T::default(),
        }
    }

    pub fn is_override(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.is_override)
    }

    pub fn has_enum_member_decoration(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.has_enum_member_decoration)
    }

    pub fn is_property_getter(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.is_property_getter)
    }

    pub fn is_property_setter_decorator(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.is_property_setter_decorator)
    }

    pub fn is_property_setter_with_getter(&self) -> Option<Type> {
        self.check_toplevel_func_metadata(&|meta| meta.flags.is_property_setter_with_getter.clone())
    }

    pub fn is_overload(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.is_overload)
    }

    pub fn is_deprecated_function(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.is_deprecated)
    }

    pub fn has_final_decoration(&self) -> bool {
        self.check_toplevel_func_metadata(&|meta| meta.flags.has_final_decoration)
    }

    pub fn dataclass_transform_metadata(&self) -> Option<DataclassTransformKeywords> {
        self.check_toplevel_func_metadata(&|meta| meta.flags.dataclass_transform_metadata.clone())
    }

    /// Transforms this type's function metadata, if it is a function. Note that we do *not*
    /// recurse into the type to find nested function types.
    pub fn transform_toplevel_func_metadata(&mut self, mut f: impl FnMut(&mut FuncMetadata)) {
        match self {
            Type::Function(box func)
            | Type::Forall(box Forall {
                tparams: _,
                body: Forallable::Function(func),
            })
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Function(func),
                ..
            }) => f(&mut func.metadata),
            Type::Overload(overload) => f(&mut overload.metadata),
            _ => {}
        }
    }

    /// Apply `f` to this type if it is a callable. Note that we do *not* recurse into the type to
    /// find nested callable types.
    pub fn visit_toplevel_callable<'a>(&'a self, mut f: impl FnMut(&'a Callable)) {
        match self {
            Type::Callable(callable) => f(callable),
            Type::Forall(box Forall {
                body: Forallable::Callable(callable),
                ..
            }) => f(callable),
            Type::Function(box func)
            | Type::Forall(box Forall {
                body: Forallable::Function(func),
                ..
            })
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Function(func),
                ..
            })
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Forall(Forall { body: func, .. }),
                ..
            }) => f(&func.signature),
            Type::Overload(overload)
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Overload(overload),
                ..
            }) => {
                for x in overload.signatures.iter() {
                    match x {
                        OverloadType::Function(function) => f(&function.signature),
                        OverloadType::Forall(forall) => f(&forall.body.signature),
                    }
                }
            }
            _ => {}
        }
    }

    fn is_toplevel_callable(&self) -> bool {
        let mut is_callable = false;
        self.visit_toplevel_callable(&mut |_| is_callable = true);
        is_callable
    }

    /// Transform this type if it is a callable. Note that we do *not* recurse into the type to
    /// find nested callable types.
    fn transform_toplevel_callable(&mut self, mut f: impl FnMut(&mut Callable)) {
        match self {
            Type::Callable(callable) => f(callable),
            Type::Forall(box Forall {
                body: Forallable::Callable(callable),
                ..
            }) => f(callable),
            Type::Function(box func)
            | Type::Forall(box Forall {
                body: Forallable::Function(func),
                ..
            })
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Function(func),
                ..
            })
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Forall(Forall { body: func, .. }),
                ..
            }) => f(&mut func.signature),
            Type::Overload(overload)
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Overload(overload),
                ..
            }) => {
                for x in overload.signatures.iter_mut() {
                    match x {
                        OverloadType::Function(function) => f(&mut function.signature),
                        OverloadType::Forall(forall) => f(&mut forall.body.signature),
                    }
                }
            }
            _ => {}
        }
    }

    // This doesn't handle generics currently
    pub fn callable_return_type(&self) -> Option<Type> {
        let mut rets = Vec::new();
        let mut get_ret = |callable: &Callable| {
            rets.push(callable.ret.clone());
        };
        self.visit_toplevel_callable(&mut get_ret);
        if rets.is_empty() {
            None
        } else {
            Some(unions(rets))
        }
    }

    // This doesn't handle generics currently
    pub fn set_callable_return_type(&mut self, ret: Type) {
        let mut set_ret = |callable: &mut Callable| {
            callable.ret = ret.clone();
        };
        self.transform_toplevel_callable(&mut set_ret);
    }

    pub fn callable_first_param(&self) -> Option<Type> {
        let mut params = Vec::new();
        let mut get_param = |callable: &Callable| {
            if let Some(p) = callable.get_first_param() {
                params.push(p);
            }
        };
        self.visit_toplevel_callable(&mut get_param);
        if params.is_empty() {
            None
        } else {
            Some(unions(params))
        }
    }

    pub fn callable_signatures(&self) -> Vec<&Callable> {
        let mut sigs = Vec::new();
        self.visit_toplevel_callable(&mut |sig| sigs.push(sig));
        sigs
    }

    pub fn promote_literals(mut self, stdlib: &Stdlib) -> Type {
        fn g(ty: &mut Type, f: &mut dyn FnMut(&mut Type)) {
            // This isn't quite right: we should decide whether to promote a literal based on
            // whether it is inferred or annotated. But we don't have an easy way to track that
            // right now, and promoting literals in callable signatures is always wrong, so let's
            // special-case callables for now.
            if !ty.is_toplevel_callable() {
                ty.recurse_mut(&mut |ty| g(ty, f));
                f(ty);
            }
        }
        g(&mut self, &mut |ty| match &ty {
            Type::Literal(lit) => *ty = lit.general_class_type(stdlib).clone().to_type(),
            Type::LiteralString => *ty = stdlib.str().clone().to_type(),
            _ => {}
        });
        self
    }

    // Attempt at a function that will convert @ to Any for now.
    pub fn clean_var(self) -> Type {
        self.transform(&mut |ty| match &ty {
            Type::Var(_) => *ty = Type::Any(AnyStyle::Implicit),
            _ => {}
        })
    }

    pub fn any_implicit() -> Self {
        Type::Any(AnyStyle::Implicit)
    }

    pub fn any_explicit() -> Self {
        Type::Any(AnyStyle::Explicit)
    }

    pub fn any_error() -> Self {
        Type::Any(AnyStyle::Error)
    }

    pub fn explicit_any(self) -> Self {
        self.transform(&mut |ty| {
            if let Type::Any(style) = ty {
                *style = AnyStyle::Explicit;
            }
        })
    }

    pub fn noreturn_to_never(self) -> Self {
        self.transform(&mut |ty| {
            if let Type::Never(style) = ty {
                *style = NeverStyle::Never;
            }
        })
    }

    /// type[a | b] -> type[a] | type[b]
    pub fn distribute_type_over_union(self) -> Self {
        self.transform(&mut |ty| {
            if let Type::Type(box Type::Union(members)) = ty {
                *ty = unions(members.drain(..).map(Type::type_form).collect());
            }
        })
    }

    pub fn anon_callables(self) -> Self {
        self.transform(&mut |mut ty| {
            if let Type::Function(func) = ty {
                *ty = Type::Callable(Box::new(func.signature.clone()));
            }
            // Anonymize posonly parameters in callables and paramspec values.
            fn transform_params(params: &mut ParamList) {
                for param in params.items_mut() {
                    if let Param::PosOnly(Some(_), ty, req) = param {
                        *param = Param::PosOnly(None, ty.clone(), req.clone());
                    }
                }
            }
            ty.transform_toplevel_callable(
                &mut |callable: &mut Callable| match &mut callable.params {
                    Params::List(params) => {
                        transform_params(params);
                    }
                    _ => {}
                },
            );
            if let Type::ParamSpecValue(params) = &mut ty {
                transform_params(params);
            }
        })
    }

    pub fn promote_typevar_values(self, stdlib: &Stdlib) -> Self {
        self.transform(&mut |ty| match &ty {
            Type::TypeVar(_) => *ty = stdlib.type_var().clone().to_type(),
            Type::ParamSpec(_) => *ty = stdlib.param_spec().clone().to_type(),
            Type::TypeVarTuple(_) => *ty = stdlib.type_var_tuple().clone().to_type(),
            Type::QuantifiedValue(q) => *ty = q.class_type(stdlib).clone().to_type(),
            _ => {}
        })
    }

    pub fn sort_unions(self) -> Self {
        self.transform(&mut |ty| {
            if let Type::Union(ts) = ty {
                ts.sort();
            }
        })
    }

    /// Used prior to display to ensure unique variables don't leak out non-deterministically.
    pub fn deterministic_printing(self) -> Self {
        self.transform(&mut |ty| {
            match ty {
                Type::Var(v) => {
                    // TODO: Should mostly be forcing these before printing
                    *v = Var::ZERO;
                }
                _ => {}
            }
        })
    }

    /// Visit every type, with the guarantee you will have seen included types before the parent.
    pub fn universe<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        fn g<'a>(ty: &'a Type, f: &mut dyn FnMut(&'a Type)) {
            ty.recurse(&mut |ty| g(ty, f));
            f(ty);
        }
        g(self, f);
    }

    /// Visit every type, with the guarantee you will have seen included types before the parent.
    pub fn transform_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        fn g(ty: &mut Type, f: &mut dyn FnMut(&mut Type)) {
            ty.recurse_mut(&mut |ty| g(ty, f));
            f(ty);
        }
        g(self, f);
    }

    pub fn transform(mut self, f: &mut dyn FnMut(&mut Type)) -> Self {
        self.transform_mut(f);
        self
    }

    pub fn as_quantified(&self) -> Option<Quantified> {
        match self {
            Type::Quantified(q) => Some((**q).clone()),
            _ => None,
        }
    }

    pub fn into_unions(self) -> Vec<Type> {
        match self {
            Type::Union(types) => types,
            _ => vec![self],
        }
    }

    /// Create an optional type (union with None).
    pub fn optional(x: Self) -> Self {
        // We would like the resulting type not nested, and well sorted.
        if let Type::Union(mut xs) = x {
            match xs.binary_search(&Type::None) {
                Ok(_) => Type::Union(xs),
                Err(i) => {
                    xs.insert(i, Type::None);
                    Type::Union(xs)
                }
            }
        } else {
            match x.cmp(&Type::None) {
                Ordering::Equal => Type::None,
                Ordering::Less => Type::Union(vec![x, Type::None]),
                Ordering::Greater => Type::Union(vec![Type::None, x]),
            }
        }
    }

    /// Does this type have a QName associated with it
    pub fn qname(&self) -> Option<&QName> {
        match self {
            Type::ClassDef(cls) => Some(cls.qname()),
            Type::ClassType(c) => Some(c.qname()),
            Type::TypedDict(c) => Some(c.qname()),
            Type::PartialTypedDict(c) => Some(c.qname()),
            Type::TypeVar(t) => Some(t.qname()),
            Type::TypeVarTuple(t) => Some(t.qname()),
            Type::ParamSpec(t) => Some(t.qname()),
            Type::SelfType(cls) => Some(cls.qname()),
            Type::Literal(Lit::Enum(e)) => Some(e.class.qname()),
            _ => None,
        }
    }

    // The result of calling bool() on a value of this type if we can get a definitive answer, None otherwise.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Type::Literal(Lit::Bool(x)) => Some(*x),
            Type::Literal(Lit::Int(x)) => Some(x.as_bool()),
            Type::Literal(Lit::Bytes(x)) => Some(!x.is_empty()),
            Type::Literal(Lit::Str(x)) => Some(!x.is_empty()),
            Type::None => Some(false),
            Type::Tuple(Tuple::Concrete(elements)) => Some(!elements.is_empty()),
            Type::Union(options) => {
                let mut answer = None;
                for option in options {
                    let option_bool = option.as_bool();
                    option_bool?;
                    if answer.is_none() {
                        answer = option_bool;
                    } else if answer != option_bool {
                        return None;
                    }
                }
                answer
            }
            _ => None,
        }
    }

    pub fn to_callable(self) -> Option<Callable> {
        match self {
            Type::Callable(callable) => Some(*callable),
            Type::Function(function) => Some(function.signature),
            Type::BoundMethod(bound_method) => match bound_method.func {
                BoundMethodType::Function(function) => Some(function.signature),
                BoundMethodType::Forall(forall) => Some(forall.body.signature),
                BoundMethodType::Overload(_) => None,
            },
            _ => None,
        }
    }

    /// Return the FuncId if this type corresponds to a function or method.
    pub fn to_funcid(&self) -> Option<FuncId> {
        match &self {
            Type::Function(f) => Some(f.metadata.kind.as_func_id()),
            Type::BoundMethod(m) => Some(m.func.metadata().kind.as_func_id()),
            Type::Overload(o) => Some(o.metadata.kind.as_func_id()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::literal::Lit;
    use crate::types::Type;

    #[test]
    fn test_as_bool() {
        let true_lit = Type::Literal(Lit::Bool(true));
        let false_lit = Type::Literal(Lit::Bool(false));
        let none = Type::None;
        let s = Type::LiteralString;

        assert_eq!(true_lit.as_bool(), Some(true));
        assert_eq!(false_lit.as_bool(), Some(false));
        assert_eq!(none.as_bool(), Some(false));
        assert_eq!(s.as_bool(), None);
    }

    #[test]
    fn test_as_bool_union() {
        let s = Type::LiteralString;
        let false_lit = Type::Literal(Lit::Bool(false));
        let none = Type::None;

        let str_opt = Type::Union(vec![s, none.clone()]);
        let false_opt = Type::Union(vec![false_lit, none]);

        assert_eq!(str_opt.as_bool(), None);
        assert_eq!(false_opt.as_bool(), Some(false));
    }
}
