/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
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

use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::class::Class;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::keywords::DataclassTransformKeywords;
use crate::types::keywords::KwCall;
use crate::types::literal::Lit;
use crate::types::module::Module;
use crate::types::param_spec::ParamSpec;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::simplify::unions;
use crate::types::special_form::SpecialForm;
use crate::types::stdlib::Stdlib;
use crate::types::tuple::Tuple;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Restriction;
use crate::types::type_var::TypeVar;
use crate::types::type_var_tuple::TypeVarTuple;
use crate::types::typed_dict::TypedDict;

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
    pub fn apply_substitution(&self, substitution: &Substitution) -> Self {
        let tys = self
            .0
            .1
            .iter()
            .map(|ty| substitution.substitute(ty.clone()))
            .collect();
        Self::new(self.0.0.dupe(), tys)
    }

    pub fn substitution<'a>(&'a self) -> Substitution<'a> {
        let tparams = self.tparams();
        let tys = self.as_slice();
        Substitution(tparams.quantifieds().zip(tys.iter()).collect())
    }

    pub fn substitute(&self, ty: Type) -> Type {
        self.substitution().substitute(ty)
    }
}

pub struct Substitution<'a>(SmallMap<&'a Quantified, &'a Type>);

impl<'a> Substitution<'a> {
    pub fn substitute(&self, ty: Type) -> Type {
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
    pub obj: Type,
    pub func: BoundMethodType,
}

impl BoundMethod {
    pub fn to_callable(&self) -> Option<Type> {
        self.as_bound_function().to_unbound_callable()
    }

    pub fn as_bound_function(&self) -> Type {
        self.func.as_type()
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
    pub fn as_type(&self) -> Type {
        match self {
            Self::Function(func) => Type::Function(Box::new(func.clone())),
            Self::Forall(forall) => {
                Forallable::Function(forall.body.clone()).forall(forall.tparams.clone())
            }
            Self::Overload(overload) => Type::Overload(overload.clone()),
        }
    }

    pub fn subst_self_type_mut(
        &mut self,
        replacement: &Type,
        is_subset: &dyn Fn(&Type, &Type) -> bool,
    ) {
        match self {
            Self::Function(func) => func.signature.subst_self_type_mut(replacement, is_subset),
            Self::Forall(forall) => forall
                .body
                .signature
                .subst_self_type_mut(replacement, is_subset),
            Self::Overload(overload) => {
                for sig in overload.signatures.iter_mut() {
                    sig.subst_self_type_mut(replacement, is_subset)
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
    Callable(Callable),
    Forall(Forall<Function>),
}

impl OverloadType {
    pub fn as_type(&self) -> Type {
        match self {
            Self::Callable(c) => Type::Callable(Box::new(c.clone())),
            Self::Forall(forall) => {
                Forallable::Function(forall.body.clone()).forall(forall.tparams.clone())
            }
        }
    }

    fn subst_self_type_mut(
        &mut self,
        replacement: &Type,
        is_subset: &dyn Fn(&Type, &Type) -> bool,
    ) {
        match self {
            Self::Callable(c) => c.subst_self_type_mut(replacement, is_subset),
            Self::Forall(forall) => forall
                .body
                .signature
                .subst_self_type_mut(replacement, is_subset),
        }
    }

    fn is_typeguard(&self) -> bool {
        match self {
            Self::Callable(c) => c.is_typeguard(),
            Self::Forall(forall) => forall.body.signature.is_typeguard(),
        }
    }

    fn is_typeis(&self) -> bool {
        match self {
            Self::Callable(c) => c.is_typeis(),
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
        targs.substitute(self.body.as_type())
    }
}

/// These are things that can have Forall around them, so often you see `Forall<Forallable>`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Forallable {
    TypeAlias(TypeAlias),
    Function(Function),
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
            Self::TypeAlias(ta) => (*ta.name).clone(),
        }
    }

    pub fn as_type(self) -> Type {
        match self {
            Self::Function(func) => Type::Function(Box::new(func)),
            Self::TypeAlias(ta) => Type::TypeAlias(ta),
        }
    }

    fn is_typeguard(&self) -> bool {
        match self {
            Self::Function(func) => func.signature.is_typeguard(),
            Self::TypeAlias(_) => false,
        }
    }

    fn is_typeis(&self) -> bool {
        match self {
            Self::Function(func) => func.signature.is_typeis(),
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
    Class(Class),
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
    /// A method of a class. The first `Box<Type>` is the self/cls argument,
    /// and the second is the function.
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
    /// Represents a "partial" version of a TypedDict.
    /// For a TypedDict type `C`, `Partial[C]` represents an object with any subset of keys from `C`,
    /// where each present key has the same value type as in `C`.
    PartialTypedDict(TypedDict),
    Tuple(Tuple),
    Module(Module),
    Forall(Box<Forall<Forallable>>),
    Var(Var),
    Quantified(Quantified),
    TypeGuard(Box<Type>),
    TypeIs(Box<Type>),
    Unpack(Box<Type>),
    TypeVar(TypeVar),
    ParamSpec(ParamSpec),
    TypeVarTuple(TypeVarTuple),
    SpecialForm(SpecialForm),
    Concatenate(Box<[Type]>, Box<Type>),
    ParamSpecValue(ParamList),
    /// Used to represent `P.args`. The spec describes it as an annotation,
    /// but it's easier to think of it as a type that can't occur in nested positions.
    Args(Quantified),
    Kwargs(Quantified),
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

    pub fn as_module(&self) -> Option<&Module> {
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

    /// Convert a bound method into a callable by stripping the first argument.
    /// TODO: Does not handle generics.
    pub fn to_unbound_callable(&self) -> Option<Type> {
        match self {
            Type::Callable(callable) => callable
                .drop_first_param()
                .map(|callable| Type::Callable(Box::new(callable))),
            Type::Function(func) => func.signature.drop_first_param().map(|callable| {
                Type::Function(Box::new(Function {
                    signature: callable,
                    metadata: func.metadata.clone(),
                }))
            }),
            Type::Overload(overload) => overload
                .signatures
                .try_mapped_ref(|x| match x {
                    OverloadType::Callable(c) => c.drop_first_param().ok_or(()),
                    _ => Err(()),
                })
                .ok()
                .map(|signatures| {
                    Type::Overload(Overload {
                        signatures: signatures.mapped(OverloadType::Callable),
                        metadata: overload.metadata.clone(),
                    })
                }),
            _ => None,
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

    pub fn subst(mut self, mp: &SmallMap<&Quantified, &Type>) -> Self {
        // We are looking up Quantified in a map, and Quantified may contain a Quantified within it.
        // Therefore, to make sure we still get matches, work top-down (not using `transform`).
        fn f(ty: &mut Type, mp: &SmallMap<&Quantified, &Type>) {
            if let Type::Quantified(x) = ty {
                if let Some(w) = mp.get(x) {
                    *ty = (*w).clone();
                }
            } else {
                ty.recurse_mut(&mut |x| f(x, mp));
            }
        }
        if !mp.is_empty() {
            f(&mut self, mp);
        }
        self
    }

    pub fn subst_self_special_form_mut(&mut self, self_type: &Type) {
        self.transform_mut(&mut |x| {
            if x == &Type::SpecialForm(SpecialForm::SelfType) {
                *x = self_type.clone()
            }
        });
    }

    pub fn subst_self_type_mut(
        &mut self,
        replacement: &Type,
        is_subset: &dyn Fn(&Type, &Type) -> bool,
    ) {
        self.transform_mut(&mut |t| {
            if matches!(t, Type::SelfType(_)) && is_subset(replacement, t) {
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

    fn check_func_metadata<T: Default>(&self, check: &dyn Fn(&FuncMetadata) -> T) -> T {
        match self {
            Type::Function(box func)
            | Type::Forall(box Forall {
                tparams: _,
                body: Forallable::Function(func),
            })
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Function(func),
                ..
            }) => check(&func.metadata),
            Type::Overload(overload) => check(&overload.metadata),
            _ => T::default(),
        }
    }

    pub fn is_override(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.is_override)
    }

    pub fn has_enum_member_decoration(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.has_enum_member_decoration)
    }

    pub fn is_property_getter(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.is_property_getter)
    }

    pub fn is_property_setter_decorator(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.is_property_setter_decorator)
    }

    pub fn is_property_setter_with_getter(&self) -> Option<Type> {
        self.check_func_metadata(&|meta| meta.flags.is_property_setter_with_getter.clone())
    }

    pub fn is_overload(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.is_overload)
    }

    pub fn is_deprecated(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.is_deprecated)
    }

    pub fn has_final_decoration(&self) -> bool {
        self.check_func_metadata(&|meta| meta.flags.has_final_decoration)
    }

    pub fn dataclass_transform_metadata(&self) -> Option<DataclassTransformKeywords> {
        self.check_func_metadata(&|meta| meta.flags.dataclass_transform_metadata.clone())
    }

    pub fn transform_func_metadata(&mut self, mut f: impl FnMut(&mut FuncMetadata)) {
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

    fn transform_callable(&mut self, mut f: impl FnMut(&mut Callable)) {
        match self {
            Type::Callable(callable) => f(callable),
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
                        OverloadType::Callable(callable) => f(callable),
                        OverloadType::Forall(forall) => f(&mut forall.body.signature),
                    }
                }
            }
            _ => {}
        }
    }

    // This doesn't handle generics currently
    pub fn callable_return_type(&mut self) -> Option<Type> {
        let mut rets = Vec::new();
        let mut get_ret = |callable: &mut Callable| {
            rets.push(callable.ret.clone());
        };
        self.transform_callable(&mut get_ret);
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
        self.transform_callable(&mut set_ret);
    }

    pub fn promote_literals(self, stdlib: &Stdlib) -> Type {
        self.transform(&mut |ty| match &ty {
            Type::Literal(lit) => *ty = lit.general_class_type(stdlib).clone().to_type(),
            _ => {}
        })
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

    pub fn anon_callables(self) -> Self {
        self.transform(&mut |mut ty| {
            if let Type::Function(func) = ty {
                *ty = Type::Callable(Box::new(func.signature.clone()));
            }
            // Anonymize posonly parameters in callables and paramspec values.
            fn transform_params(params: &mut ParamList) {
                for param in params.items_mut() {
                    if let Param::PosOnly(Some(_), ty, req) = param {
                        *param = Param::PosOnly(None, ty.clone(), *req);
                    }
                }
            }
            ty.transform_callable(&mut |callable: &mut Callable| match &mut callable.params {
                Params::List(params) => {
                    transform_params(params);
                }
                _ => {}
            });
            if let Type::ParamSpecValue(params) = &mut ty {
                transform_params(params);
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
            Type::Quantified(q) => Some(q.clone()),
            _ => None,
        }
    }

    pub fn into_unions(self) -> Vec<Type> {
        match self {
            Type::Union(types) => types,
            _ => vec![self],
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
}

#[cfg(test)]
mod tests {
    use crate::types::literal::Lit;
    use crate::types::types::Type;

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
