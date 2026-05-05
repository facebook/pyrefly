/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::module_name::ModuleName;
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
use crate::callable::Deprecation;
use crate::callable::FuncMetadata;
use crate::callable::Function;
use crate::callable::FunctionKind;
use crate::callable::Param;
use crate::callable::ParamList;
use crate::callable::Params;
use crate::callable::PrefixParam;
use crate::callable::PropertyMetadata;
use crate::callable::PropertyRole;
use crate::class::Class;
use crate::class::ClassKind;
use crate::class::ClassType;
use crate::dimension;
use crate::dimension::SizeExpr;
use crate::equality::TypeEqCtx;
use crate::heap::TypeHeap;
use crate::keywords::DataclassTransformMetadata;
use crate::keywords::KwCall;
use crate::literal::Lit;
use crate::literal::LitStyle;
use crate::literal::Literal;
use crate::module::ModuleType;
use crate::param_spec::ParamSpec;
use crate::quantified::Quantified;
use crate::simplify::unions;
use crate::special_form::SpecialForm;
use crate::stdlib::Stdlib;
use crate::tensor::TensorType;
use crate::tuple::Tuple;
use crate::type_alias::TypeAliasData;
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

    pub fn to_type(self, heap: &TypeHeap) -> Type {
        heap.mk_var(self)
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

/// Wraps a vector of type parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TParams(Vec<Quantified>);

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
        write!(
            f,
            "[{}]",
            commas_iter(|| self.0.iter().map(|q| q.display_with_bounds()))
        )
    }
}

impl TParams {
    pub fn new(tparams: Vec<Quantified>) -> TParams {
        Self(tparams)
    }

    pub fn empty() -> TParams {
        Self(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Quantified> {
        self.0.iter()
    }

    pub fn as_vec(&self) -> &[Quantified] {
        &self.0
    }

    pub fn extend(&mut self, other: &TParams) {
        self.0.extend(other.iter().cloned());
    }

    /// Truncate recursive TArgs nesting in Quantified restrictions.
    ///
    /// During iterative fixpoint solving, TParams can grow unboundedly when
    /// classes reference each other in type parameter bounds (either self-referentially
    /// like `class C[T: C]`, or mutually like Session ↔ DataFrame ↔ Catalog).
    /// Each iteration embeds the previous iteration's TParams one level deeper
    /// through the chain: TParams → Restriction → ClassType → TArgs → TParams → ...
    ///
    /// This method enforces a structural invariant: within a Quantified's restriction,
    /// any ClassType whose TArgs embed TParams with their own non-trivial restrictions
    /// (Bound or Constraints) has its TArgs stripped to empty. This prevents recursive
    /// nesting while preserving TArgs for simple cases like `T: List[int]` where the
    /// inner TParams have only unrestricted type variables.
    pub fn truncate_recursive_targs(self) -> Self {
        let quantifieds = self
            .0
            .into_iter()
            .map(|q| {
                let new_restriction = match q.restriction() {
                    Restriction::Bound(ty) => {
                        Restriction::Bound(Self::strip_recursive_class_targs(ty.clone()))
                    }
                    Restriction::Constraints(tys) => Restriction::Constraints(
                        tys.iter()
                            .map(|ty| Self::strip_recursive_class_targs(ty.clone()))
                            .collect(),
                    ),
                    Restriction::Unrestricted => Restriction::Unrestricted,
                };
                q.with_restriction(new_restriction)
            })
            .collect();
        Self(quantifieds)
    }

    /// Walk a type tree and strip TArgs from any ClassType whose TArgs' TParams
    /// have non-trivial restrictions (Bound or Constraints). Such TParams can
    /// participate in recursive nesting across fixpoint iterations.
    fn strip_recursive_class_targs(ty: Type) -> Type {
        ty.transform(&mut |t| {
            if let Type::ClassType(ct) = t
                && !ct.targs().is_empty()
                && Self::tparams_have_restrictions(ct.targs().tparams())
            {
                *t = Type::ClassType(ClassType::new(ct.class_object().dupe(), TArgs::default()));
            }
        })
    }

    /// Check if any Quantified in the TParams has a non-trivial restriction.
    fn tparams_have_restrictions(tparams: &TParams) -> bool {
        tparams.iter().any(|q| q.restriction().is_restricted())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(TypeEq)]
pub struct TArgs(Arc<(Arc<TParams>, Box<[Type]>)>);

impl Visit<Type> for TArgs {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.0.0.visit(f);
        self.0.1.visit(f);
    }
}

impl VisitMut<Type> for TArgs {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        // Arc<TParams> has a no-op VisitMut, so we only need to visit the types.
        let inner = Arc::make_mut(&mut self.0);
        inner.1.visit_mut(f);
    }
}

impl TArgs {
    pub fn new(tparams: Arc<TParams>, targs: Vec<Type>) -> Self {
        if tparams.len() != targs.len() {
            panic!("TParams and TArgs must have the same length");
        }
        Self(Arc::new((tparams, targs.into_boxed_slice())))
    }

    pub fn tparams(&self) -> &TParams {
        &self.0.0
    }

    pub fn iter_paired(&self) -> impl ExactSizeIterator<Item = (&Quantified, &Type)> {
        self.0.0.iter().zip(self.0.1.iter())
    }

    pub fn iter_paired_mut(&mut self) -> impl ExactSizeIterator<Item = (&Quantified, &mut Type)> {
        let inner = Arc::make_mut(&mut self.0);
        inner.0.iter().zip(inner.1.iter_mut())
    }

    pub fn len(&self) -> usize {
        self.0.1.len()
    }

    pub fn as_slice(&self) -> &[Type] {
        &self.0.1
    }

    pub fn as_mut(&mut self) -> &mut [Type] {
        &mut Arc::make_mut(&mut self.0).1
    }

    pub fn is_empty(&self) -> bool {
        self.0.1.is_empty()
    }

    /// Returns the number of type arguments to display, stripping trailing args
    /// that match their parameter defaults (WYSIWYG display per issue #2461).
    pub fn display_count(&self) -> usize {
        let mut last_non_default = 0;
        for (i, (param, arg)) in self.iter_paired().enumerate() {
            if param.default().is_none() || arg != &param.as_gradual_type() {
                last_non_default = i + 1;
            }
        }
        last_non_default
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
        tparams.iter().zip(tys.iter()).collect()
    }

    pub fn substitution<'a>(&'a self) -> Substitution<'a> {
        Substitution(self.substitution_map())
    }

    pub fn substitute_into_mut(&self, ty: &mut Type) {
        match ty {
            Type::TypeAlias(box TypeAliasData::Ref(r))
            | Type::UntypedAlias(box TypeAliasData::Ref(r)) => {
                // We don't have the value of the type alias available to do the substitution, so store
                // the targs so that we can apply them when the value is looked up.
                r.args = Some(self.clone())
            }
            _ => self.substitution().substitute_into_mut(ty),
        }
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
pub enum CallableResidualKind {
    /// A generic residual. The `quantified` is the quantified type variable we
    /// want to use when reconstructing a `Callable` that contains a generic
    /// residual; we'll wrap it in a Forall that scopes all the residuals.
    ///
    /// If it appears anywhere else, the fallback is `quantified.as_gradual_type()`
    Generic { quantified: Quantified },
    /// Per-var overload residual with identity for cross-var correlation.
    ///
    /// Finishing normalizes branch types so an overload residual does not
    /// contain nested overload residual markers in `branches[*].ty`.
    Overload {
        identity: OverloadResidualIdentity,
        branches: Vec<OverloadBranchProjection>,
    },
}

/// Correlation key for matching overload residuals across vars during finalization.
/// The hash is derived from the got-side type at the comparison that produced the
/// residual, making it a stable function of the value rather than of solve order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct OverloadResidualIdentity {
    pub witness_hash: u64,
}

/// Per-branch result for a single var in an overload residual.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct OverloadBranchProjection {
    pub branch_index: usize,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct CallableResidual {
    pub kind: CallableResidualKind,
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
    TypeAlias(TypeAliasData),
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

    pub fn name(&self) -> Cow<'_, Name> {
        match self {
            Self::Function(func) => func.metadata.kind.function_name(),
            Self::Callable(_) => Cow::Owned(Name::new_static("<callable>")),
            Self::TypeAlias(ta) => Cow::Borrowed(ta.name()),
        }
    }

    pub fn as_type(self) -> Type {
        match self {
            Self::Function(func) => Type::Function(Box::new(func)),
            Self::Callable(callable) => Type::Callable(Box::new(callable)),
            Self::TypeAlias(ta) => Type::TypeAlias(Box::new(ta)),
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

#[derive(Debug, Clone, Eq, TypeEq, PartialOrd, Ord)]
pub struct Union {
    pub members: Vec<Type>,
    pub display_name: Option<(ModuleName, Name)>,
}

impl PartialEq for Union {
    fn eq(&self, other: &Self) -> bool {
        self.members == other.members
    }
}

impl Hash for Union {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.members.hash(state)
    }
}

impl Visit<Type> for Union {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        for member in &self.members {
            member.visit(f);
        }
    }
}

impl VisitMut<Type> for Union {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        for member in &mut self.members {
            member.visit_mut(f);
        }
    }
}

/// An nn.Module instance with captured constructor arguments.
///
/// Analogous to how `TensorType` wraps `ClassType` + shape info, `NNModuleType`
/// wraps `ClassType` + a field map of captured init args. This allows DSL forward
/// functions to access constructor parameters (e.g., `kernel_size`, `stride`)
/// directly from the type, without requiring every shape-relevant parameter to
/// be a generic type param on the class.
///
/// Created by init DSL functions during `construct_class`. When `forward` is
/// called on an NNModule instance, the fields are injected as `Val::Module`
/// into the DSL's bound_args.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NNModuleType {
    /// The underlying nn.Module subclass (e.g., MaxPool2d).
    pub class: ClassType,
    /// Captured init args (e.g., kernel_size → Size(3), stride → None).
    /// Ordered by constructor parameter order.
    pub fields: SmallMap<Name, Type>,
}

impl Hash for NNModuleType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.class.hash(state);
        self.fields.len().hash(state);
        for (k, v) in self.fields.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl PartialOrd for NNModuleType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NNModuleType {
    fn cmp(&self, other: &Self) -> Ordering {
        self.class.cmp(&other.class).then_with(|| {
            let len_cmp = self.fields.len().cmp(&other.fields.len());
            if len_cmp != Ordering::Equal {
                return len_cmp;
            }
            for ((k1, v1), (k2, v2)) in self.fields.iter().zip(other.fields.iter()) {
                let c = k1.cmp(k2).then_with(|| v1.cmp(v2));
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        })
    }
}

impl pyrefly_util::visit::Visit<Type> for NNModuleType {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.class.recurse(f);
        for (_, ty) in self.fields.iter() {
            f(ty);
        }
    }
}

impl pyrefly_util::visit::VisitMut<Type> for NNModuleType {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        self.class.recurse_mut(f);
        for (_, ty) in self.fields.iter_mut() {
            f(ty);
        }
    }
}

impl crate::equality::TypeEq for NNModuleType {
    fn type_eq(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        crate::equality::TypeEq::type_eq(&self.class, &other.class, ctx)
            && self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(other.fields.iter())
                .all(|((k1, v1), (k2, v2))| {
                    k1 == k2 && crate::equality::TypeEq::type_eq(v1, v2, ctx)
                })
    }
}

impl NNModuleType {
    /// Create a new NNModuleType with the given class and captured fields.
    pub fn new(class: ClassType, fields: SmallMap<Name, Type>) -> Self {
        Self { class, fields }
    }
}

// Note: The fact that Literal and LiteralString are at the front is important for
// optimisations in `unions_with_literals`.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum Type {
    Literal(Box<Literal>),
    LiteralString(LitStyle),
    /// typing.Callable
    Callable(Box<Callable>),
    /// The result of solving a parameter in a higher-order function call against some part of a
    /// generic or overloaded argument type. This type captures information about the structure
    /// of the argument, so that we can resonstruct the same generic/overload structure if it
    /// appears in a callable type later. Otherwise, we should *flatten* to a fallback type.
    CallableResidual(Box<CallableResidual>),
    /// A function declared using the `def` keyword.
    /// Note that the FunctionKind metadata doesn't participate in subtyping, and thus two types with distinct metadata are still subtypes.
    Function(Box<Function>),
    /// A method of a class.
    BoundMethod(Box<BoundMethod>),
    /// An overloaded function.
    Overload(Overload),
    /// Unions will hold an optional name to use when displaying the type
    Union(Box<Union>),
    /// Our intersection support is partial, so we store a fallback type that we use for operations
    /// that are not yet supported on intersections.
    Intersect(Box<(Vec<Type>, Type)>),
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
    /// Tensor type with shape information
    /// Example: Tensor[2, 3] represents a 2x3 tensor
    Tensor(Box<TensorType>),
    /// nn.Module instance with captured constructor arguments.
    /// Wraps a ClassType + field map of init args, enabling DSL forward
    /// functions to access shape-relevant constructor parameters directly.
    NNModule(Box<NNModuleType>),
    /// Dimension value type - represents values that satisfy Dim bound
    /// Examples:
    ///   - Type::Size(SizeExpr::Literal(6)) for concrete dimension 6
    ///   - Type::Size(SizeExpr::Var(v)) for dimension variables
    ///
    /// This is the type-level representation of dimension values, used when
    /// type variables with Dim bound unify with concrete dimension values.
    Size(SizeExpr),
    /// Symbolic integer type - wraps dimension expressions for use in type annotations
    /// Examples:
    ///   - Type::Dim(SizeExpr(Literal(3))) for Dim[3]
    ///   - Type::Dim(Quantified) for Dim[N]
    ///   - Type::Dim(SizeExpr(Add(...))) for Dim[N+1]
    ///
    /// This is the type annotation form of symbolic integers, distinct from
    /// concrete integer literals which use Type::Literal(Lit::Int(...)).
    Dim(Box<Type>),
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
    /// When we unpack a Type::Quantified TypeVarTuple, this is what we get
    ElementOfTypeVarTuple(Box<Quantified>),
    TypeGuard(Box<Type>),
    TypeIs(Box<Type>),
    /// Used for special form `Annotated[T, ...]`.
    /// This is transparent when resolving annotations, but is not callable and
    /// cannot be assigned to `type[T]`.
    /// The second field carries the metadata items (the `...` in `Annotated[T, ...]`).
    Annotated(Box<Type>, Box<[Type]>),
    Unpack(Box<Type>),
    TypeVar(TypeVar),
    ParamSpec(ParamSpec),
    TypeVarTuple(TypeVarTuple),
    SpecialForm(SpecialForm),
    Concatenate(Box<[PrefixParam]>, Box<Type>),
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
    /// TypeForm[T] — a type form object (PEP 747).
    TypeForm(Box<Type>),
    Ellipsis,
    Any(AnyStyle),
    Never(NeverStyle),
    TypeAlias(Box<TypeAliasData>),
    /// The result of untyping a type alias. For example, if we have `type X = int`, the type alias
    /// stores `type[int]` as its value, which untypes to `int`. Since recursive references cannot
    /// be immediately looked up for untyping (see `TypeAliasData::TypeAliasRef`), `UntypedAlias`
    /// stores a reference that is untyped once we actually look up the value.
    UntypedAlias(Box<TypeAliasData>),
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
    /// All possible materializations of Any. A subset check with Type::Materialization succeeds
    /// only if it would succeed with any type. This behaves like top (`object`) in one direction
    /// and bottom (`Never`) in the other:
    /// * `Materialization` <: `T` succeeds iff `object` <: `T` would succeed
    /// * `T` <: `Materialization` succeeds iff `T` <: `Never` would succeed
    ///
    /// See https://typing.python.org/en/latest/spec/glossary.html#term-materialize.
    Materialization,
    None,
}

impl Visit for Type {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Self)) {
        match self {
            Type::Literal(x) => x.visit(f),
            Type::LiteralString(_) => {}
            Type::Callable(x) => x.visit(f),
            Type::CallableResidual(x) => x.visit(f),
            Type::Function(x) => x.visit(f),
            Type::BoundMethod(x) => x.visit(f),
            Type::Overload(x) => x.visit(f),
            Type::Union(x) => x.visit(f),
            Type::Intersect(x) => x.visit(f),
            Type::ClassDef(x) => x.visit(f),
            Type::ClassType(x) => x.visit(f),
            Type::TypedDict(x) => x.visit(f),
            Type::PartialTypedDict(x) => x.visit(f),
            Type::Tensor(x) => x.visit(f),
            Type::NNModule(x) => x.visit(f),
            Type::Size(x) => x.visit(f),
            Type::Dim(x) => x.visit(f),
            Type::Tuple(x) => x.visit(f),
            Type::Module(x) => x.visit(f),
            Type::Forall(x) => x.visit(f),
            Type::Var(x) => x.visit(f),
            Type::Quantified(x) => x.visit(f),
            Type::QuantifiedValue(x) => x.visit(f),
            Type::ElementOfTypeVarTuple(x) => x.visit(f),
            Type::TypeGuard(x) => x.visit(f),
            Type::TypeIs(x) => x.visit(f),
            Type::Annotated(x, _metadata) => x.visit(f),
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
            Type::TypeForm(x) => x.visit(f),
            Type::Ellipsis => {}
            Type::Any(x) => x.visit(f),
            Type::Never(x) => x.visit(f),
            Type::TypeAlias(x) => x.visit(f),
            Type::UntypedAlias(x) => x.visit(f),
            Type::SuperInstance(x) => x.visit(f),
            Type::SelfType(x) => x.visit(f),
            Type::KwCall(x) => x.visit(f),
            Type::Materialization | Type::None => {}
        }
    }
}

impl VisitMut for Type {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Self)) {
        match self {
            Type::Literal(x) => x.visit_mut(f),
            Type::LiteralString(_) => {}
            Type::Callable(x) => x.visit_mut(f),
            Type::CallableResidual(x) => x.visit_mut(f),
            Type::Function(x) => x.visit_mut(f),
            Type::BoundMethod(x) => x.visit_mut(f),
            Type::Overload(x) => x.visit_mut(f),
            Type::Union(x) => x.visit_mut(f),
            Type::Intersect(x) => x.visit_mut(f),
            Type::ClassDef(x) => x.visit_mut(f),
            Type::ClassType(x) => x.visit_mut(f),
            Type::TypedDict(x) => x.visit_mut(f),
            Type::PartialTypedDict(x) => x.visit_mut(f),
            Type::Tensor(x) => x.visit_mut(f),
            Type::NNModule(x) => x.visit_mut(f),
            Type::Size(x) => x.visit_mut(f),
            Type::Dim(x) => x.visit_mut(f),
            Type::Tuple(x) => x.visit_mut(f),
            Type::Module(x) => x.visit_mut(f),
            Type::Forall(x) => x.visit_mut(f),
            Type::Var(x) => x.visit_mut(f),
            Type::Quantified(x) => x.visit_mut(f),
            Type::QuantifiedValue(x) => x.visit_mut(f),
            Type::ElementOfTypeVarTuple(x) => x.visit_mut(f),
            Type::TypeGuard(x) => x.visit_mut(f),
            Type::TypeIs(x) => x.visit_mut(f),
            Type::Annotated(x, _metadata) => x.visit_mut(f),
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
            Type::TypeForm(x) => x.visit_mut(f),
            Type::Ellipsis => {}
            Type::Any(x) => x.visit_mut(f),
            Type::Never(x) => x.visit_mut(f),
            Type::TypeAlias(x) => x.visit_mut(f),
            Type::UntypedAlias(x) => x.visit_mut(f),
            Type::SuperInstance(x) => x.visit_mut(f),
            Type::SelfType(x) => x.visit_mut(f),
            Type::KwCall(x) => x.visit_mut(f),
            Type::Materialization | Type::None => {}
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

    /// Returns the number of top-level alternatives in a union, or 1 for non-union types.
    pub fn union_width(&self) -> usize {
        match self {
            Type::Union(u) => u.members.len(),
            _ => 1,
        }
    }

    /// Truncate container nesting depth and inner Union width, replacing
    /// over-budget nodes with `any`.
    ///
    /// Two independent limits are enforced recursively throughout the type tree:
    ///
    /// **Depth** (`max_depth`): A top-down counter tracks how many more container
    /// levels are permitted. `ClassType` and `Tuple` each consume one depth level;
    /// other composite types (Union, Callable, …) are transparent. A container
    /// encountered when `remaining == 0` is replaced with `any`; otherwise it is
    /// kept and its children are processed with `remaining - 1`. Transparent
    /// composites pass the same budget to their children unchanged.
    ///
    /// **Inner union width** (`max_inner_union_width`): Any `Union` encountered
    /// *while recursing inside a container's children or a transparent composite*
    /// is replaced with `any` if its member count exceeds `max_inner_union_width`.
    /// The top-level type is exempted: the initial call does not apply the union
    /// check to `self` itself, only to its descendants. This preserves wide
    /// top-level unions (e.g. a function returning `A | B | … | T`) while
    /// truncating runaway unions that accumulate inside type parameters.
    ///
    /// Both limits produce stable fixed points: two consecutive truncations of a
    /// type that exceeds either limit yield the same result, so the fixpoint
    /// converges after at most two truncation steps.
    pub fn truncate_class_nesting(
        self,
        max_depth: usize,
        max_inner_union_width: usize,
        any: &Type,
    ) -> Type {
        /// Truncate `ty` with both limits active (inner call — union width IS checked).
        fn truncate_inner(
            ty: Type,
            remaining: usize,
            max_inner_union_width: usize,
            any: &Type,
        ) -> Type {
            if let Type::Union(ref u) = ty
                && u.members.len() > max_inner_union_width
            {
                return any.clone();
            }
            truncate(ty, remaining, max_inner_union_width, any)
        }

        /// Core truncation — does NOT apply the union-width check to `ty` itself
        /// (that is done by the caller `truncate_inner` when needed), but does
        /// recurse via `truncate_inner` so all descendants are checked.
        fn truncate(ty: Type, remaining: usize, max_inner_union_width: usize, any: &Type) -> Type {
            match ty {
                Type::ClassType(ct) if remaining == 0 => {
                    let _ = ct;
                    any.clone()
                }
                Type::ClassType(mut ct) => {
                    for targ in ct.targs_mut().as_mut().iter_mut() {
                        let orig = std::mem::replace(targ, any.clone());
                        *targ = truncate_inner(orig, remaining - 1, max_inner_union_width, any);
                    }
                    Type::ClassType(ct)
                }
                Type::Tuple(_) if remaining == 0 => any.clone(),
                Type::Tuple(mut t) => {
                    t.visit_mut(&mut |child| {
                        let orig = std::mem::replace(child, any.clone());
                        *child = truncate_inner(orig, remaining - 1, max_inner_union_width, any);
                    });
                    Type::Tuple(t)
                }
                mut other => {
                    other.recurse_mut(&mut |child| {
                        let orig = std::mem::replace(child, any.clone());
                        *child = truncate_inner(orig, remaining, max_inner_union_width, any);
                    });
                    other
                }
            }
        }

        truncate(self, max_depth, max_inner_union_width, any)
    }

    pub fn is_never(&self) -> bool {
        matches!(self, Type::Never(_))
    }

    pub fn is_implicit_literal(&self) -> bool {
        matches!(
            self,
            Type::Literal(box Literal { style: LitStyle::Implicit, ..}) |
            Type::LiteralString(LitStyle::Implicit)
        )
    }

    pub fn is_literal_string(&self) -> bool {
        self.lit_string_style().is_some()
    }

    /// If this type is a literal string (either `LiteralString` or a `Literal` string value),
    /// return its `LitStyle`.
    pub fn lit_string_style(&self) -> Option<&LitStyle> {
        match self {
            Type::LiteralString(style) => Some(style),
            Type::Literal(l) if l.value.is_string() => Some(&l.style),
            _ => None,
        }
    }

    pub fn is_unpack(&self) -> bool {
        matches!(self, Type::Unpack(_))
    }

    pub fn callable_concatenate(args: Box<[PrefixParam]>, param_spec: Type, ret: Type) -> Self {
        Type::Callable(Box::new(Callable::concatenate(args, param_spec, ret)))
    }

    pub fn type_of(inner: Type) -> Self {
        Type::Type(Box::new(inner))
    }

    pub fn concrete_tuple(elts: Vec<Type>) -> Self {
        Type::Tuple(Tuple::Concrete(elts))
    }

    pub fn unbounded_tuple(elt: Type) -> Self {
        if let Type::ElementOfTypeVarTuple(x) = elt {
            Self::unpacked_tuple(Vec::new(), Type::Quantified(x), Vec::new())
        } else {
            Type::Tuple(Tuple::Unbounded(Box::new(elt)))
        }
    }

    pub fn unpacked_tuple(prefix: Vec<Type>, middle: Type, suffix: Vec<Type>) -> Self {
        Type::Tuple(Tuple::unpacked(prefix, middle, suffix))
    }

    pub fn any_tuple() -> Self {
        Self::unbounded_tuple(Type::Any(AnyStyle::Implicit))
    }

    pub fn is_any(&self) -> bool {
        matches!(self, Type::Any(_))
    }

    pub fn is_typed_dict(&self) -> bool {
        matches!(self, Type::TypedDict(_) | Type::PartialTypedDict(_))
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

    /// Is this type an unreplaced reference to a legacy type variable? Note that references to
    /// in-scope legacy type variables in functions and classes are replaced with Quantified, so
    /// this type only appears in cases like a TypeVar definition or an out-of-scope type variable.
    pub fn is_raw_legacy_type_variable(&self) -> bool {
        matches!(
            TypeVariable::new(self),
            Some(
                TypeVariable::LegacyTypeVar(_)
                    | TypeVariable::LegacyTypeVarTuple(_)
                    | TypeVariable::LegacyParamSpec(_)
            )
        )
    }

    fn visit_type_variables<'a>(&'a self, f: &mut dyn FnMut(TypeVariable<'a>)) {
        fn visit<'a>(ty: &'a Type, f: &mut dyn FnMut(TypeVariable<'a>)) {
            if let Some(tv) = TypeVariable::new(ty) {
                f(tv);
                return;
            }
            let mut recurse_targs = |targs: &'a TArgs| {
                for targ in targs.as_slice().iter() {
                    visit(targ, f);
                }
            };
            // IMPORTANT: keep this match in sync with `transform_types_in_type_variable_positions`
            match ty {
                // In `A[X]`, we only check `X` for a couple reasons:
                // * If we were to blindly visit the entire ClassType, we would find Quantifieds in
                //   the definition of the class, which is almost never what we want: we want to
                //   know if `X` contains any references to Quantifieds, not whether `A` is generic.
                //   See https://github.com/facebook/pyrefly/issues/1962.
                // * Not checking the rest of the ClassType is a critical performance optimization
                //   when visiting Vars. See https://github.com/facebook/pyrefly/issues/2016.
                Type::ClassType(cls) => recurse_targs(cls.targs()),
                Type::TypedDict(TypedDict::TypedDict(td)) => recurse_targs(td.targs()),
                // `Self` is a keyword, not a user-written type variable reference, so we don't
                // recurse into it when looking for type variable references.
                Type::SelfType(_) => {}
                // Enum literals contain `ClassType`s that we shouldn't visit.
                Type::Literal(_) => {}
                _ => ty.recurse(&mut |ty| visit(ty, f)),
            }
        }
        visit(self, f)
    }

    pub fn for_each_quantified<'a>(&'a self, f: &mut impl FnMut(&'a Quantified)) {
        self.visit_type_variables(&mut |x| {
            if let TypeVariable::Quantified(x) = x {
                f(x);
            }
        })
    }

    pub fn collect_quantifieds<'a>(&'a self, acc: &mut SmallSet<&'a Quantified>) {
        self.for_each_quantified(&mut |q| {
            acc.insert(q);
        });
    }

    /// Checks if the type contains any reference to a type variable. This may be a reference that
    /// has been resolved to a function- or class-scoped type parameter (i.e., a Quantified) or an
    /// unresolved reference to a legacy type variable.
    pub fn contains_type_variable(&self) -> bool {
        let mut seen = false;
        let mut f = |t| {
            seen |= matches!(
                t,
                TypeVariable::Quantified(_)
                    | TypeVariable::LegacyTypeVar(_)
                    | TypeVariable::LegacyTypeVarTuple(_)
                    | TypeVariable::LegacyParamSpec(_)
            )
        };
        self.visit_type_variables(&mut f);
        seen
    }

    /// Collect unreplaced references to legacy type variables. Note that references to in-scope
    /// legacy type variables in functions and classes are replaced with Quantified, so unreplaced
    /// references only appear in cases like a TypeVar definition or an out-of-scope type variable.
    pub fn collect_raw_legacy_type_variables(&self, acc: &mut Vec<Name>) {
        let mut f = |t| {
            let name = match t {
                TypeVariable::LegacyTypeVar(t) => t.qname().id(),
                TypeVariable::LegacyTypeVarTuple(t) => t.qname().id(),
                TypeVariable::LegacyParamSpec(p) => p.qname().id(),
                _ => return,
            };
            acc.push(name.clone());
        };
        self.visit_type_variables(&mut f)
    }

    fn transform_types_in_type_variable_positions(&mut self, f: &mut dyn FnMut(&mut Type)) {
        fn visit(ty: &mut Type, f: &mut dyn FnMut(&mut Type)) {
            f(ty);
            let mut recurse_targs = |targs: &mut TArgs| {
                for targ in targs.as_mut().iter_mut() {
                    visit(targ, f);
                }
            };
            // IMPORTANT: keep this match in sync with `visit_type_variables`
            match ty {
                Type::ClassType(cls) => recurse_targs(cls.targs_mut()),
                Type::TypedDict(TypedDict::TypedDict(td)) => recurse_targs(td.targs_mut()),
                // `Self` is a keyword, not a user-written type variable reference.
                Type::SelfType(_) => {}
                // Enum literals contain `ClassType`s that we shouldn't visit.
                Type::Literal(_) => {}
                _ => ty.recurse_mut(&mut |ty| visit(ty, f)),
            }
        }
        visit(self, f)
    }

    /// Transform unreplaced references to legacy type variables. Note that references to in-scope
    /// legacy type variables in functions and classes are replaced with Quantified, so unreplaced
    /// references only appear in cases like a TypeVar definition or an out-of-scope type variable.
    pub fn transform_raw_legacy_type_variables(&mut self, f: &mut dyn FnMut(&mut Type)) {
        self.transform_types_in_type_variable_positions(&mut |ty| {
            if ty.is_raw_legacy_type_variable() {
                f(ty);
            }
        })
    }

    /// Check if the type contains a placeholder var. See `collect_maybe_placeholder_vars`.
    pub fn may_contain_placeholder_var(&self) -> bool {
        let mut seen = false;
        self.visit_type_variables(&mut |t| seen |= matches!(t, TypeVariable::Var(_)));
        seen
    }

    /// Collect "placeholder" vars - vars that are placeholders for a not-yet-solved type, like
    /// Variable::Quantified. Contrast this with Variable::Recursive, which serves as a marker that
    /// we've encountered recursion. Note that this function is used in performance hotspots, so we
    /// avoid reading from the variables map to check a var's actual type. Instead, we collect all
    /// vars that we find in positions that placeholders can appear in.
    pub fn collect_maybe_placeholder_vars(&self) -> Vec<Var> {
        let mut vs = Vec::new();
        self.visit_type_variables(&mut |t| {
            if let TypeVariable::Var(v) = t {
                vs.push(v);
            }
        });
        vs
    }

    /// Collects all `Var`s that appear in the type.
    /// IMPORTANT: This function can be expensive. Consider using `collect_maybe_placeholder_vars` instead.
    pub fn collect_all_vars(&self) -> Vec<Var> {
        fn f(t: &Type, vars: &mut Vec<Var>) {
            match t {
                Type::Var(v) => vars.push(*v),
                _ => t.recurse(&mut |t| f(t, vars)),
            }
        }
        let mut vars = vec![];
        f(self, &mut vars);
        vars
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
            Type::Callable(_) | Type::CallableResidual(_) => Some(CalleeKind::Callable),
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
        fn f(
            ty: &mut Type,
            mp: &mut dyn FnMut(&Quantified) -> Option<Type>,
            shadowed: &mut Vec<Quantified>,
        ) {
            if let Type::Quantified(x) = ty {
                if !shadowed.contains(x)
                    && let Some(w) = mp(x)
                {
                    *ty = w;
                }
            } else if let Type::Forall(forall) = ty {
                let old_len = shadowed.len();
                shadowed.extend(forall.tparams.iter().cloned());
                ty.recurse_mut(&mut |x| f(x, mp, shadowed));
                shadowed.truncate(old_len);
            } else {
                ty.recurse_mut(&mut |x| f(x, mp, shadowed));
            }
        }
        f(self, mp, &mut Vec::new());
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

    /// Calls a `visit` function on this type's function metadata if it is a function. Note that we
    /// do *not* recurse into the type to find nested function types.
    pub fn visit_toplevel_func_metadata<'a, T: Default>(
        &'a self,
        visit: &dyn Fn(&'a FuncMetadata) -> T,
    ) -> T {
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
            }) => visit(&func.metadata),
            Type::Overload(overload)
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Overload(overload),
                ..
            }) => visit(&overload.metadata),
            _ => T::default(),
        }
    }

    pub fn has_toplevel_func_metadata(&self) -> bool {
        self.visit_toplevel_func_metadata(&|_| true)
    }

    pub fn is_abstract_method(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.is_abstract_method)
    }

    pub fn is_override(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.is_override)
    }

    pub fn has_enum_member_decoration(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.has_enum_member_decoration)
    }

    pub fn property_metadata(&self) -> Option<&PropertyMetadata> {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.property_metadata.as_ref())
    }

    pub fn is_property_getter(&self) -> bool {
        self.property_metadata()
            .is_some_and(|meta| matches!(meta.role, PropertyRole::Getter))
    }

    pub fn is_cached_property(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.is_cached_property)
    }

    pub fn is_property_setter_decorator(&self) -> bool {
        self.property_metadata()
            .is_some_and(|meta| matches!(meta.role, PropertyRole::SetterDecorator))
    }

    pub fn is_property_setter_with_getter(&self) -> Option<Type> {
        self.property_metadata().and_then(|meta| match meta.role {
            PropertyRole::Setter => Some(meta.getter.clone()),
            _ => None,
        })
    }

    pub fn property_deleter_metadata(&self) -> Option<&PropertyMetadata> {
        self.property_metadata().and_then(|meta| match meta.role {
            PropertyRole::DeleterDecorator => Some(meta),
            _ => None,
        })
    }

    pub fn without_property_metadata(&self) -> Type {
        let mut clone = self.clone();
        clone.transform_toplevel_func_metadata(|meta| {
            meta.flags.property_metadata = None;
        });
        clone
    }

    pub fn is_overload(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.is_overload)
    }

    pub fn function_deprecation(&self) -> Option<&Deprecation> {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.deprecation.as_ref())
    }

    pub fn has_final_decoration(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.has_final_decoration)
    }

    pub fn dataclass_transform_metadata(&self) -> Option<&DataclassTransformMetadata> {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.dataclass_transform_metadata.as_ref())
    }

    /// If a Protocol method lacks an implementation and does not come from a `.pyi` file, then it cannot be called
    pub fn is_non_callable_protocol_method(&self) -> bool {
        self.visit_toplevel_func_metadata(&|meta| meta.flags.lacks_runtime_implementation())
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
                func:
                    BoundMethodType::Function(func)
                    | BoundMethodType::Forall(Forall {
                        tparams: _,
                        body: func,
                    }),
                ..
            }) => f(&mut func.metadata),
            Type::Overload(overload)
            | Type::BoundMethod(box BoundMethod {
                func: BoundMethodType::Overload(overload),
                ..
            }) => f(&mut overload.metadata),
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

    /// Transform this type if it is a callable. Note that we do *not* recurse into the type to
    /// find nested callable types.
    pub fn transform_toplevel_callable<'a>(&'a mut self, mut f: impl FnMut(&'a mut Callable)) {
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

    pub fn is_toplevel_callable(&self) -> bool {
        let mut is_callable = false;
        self.visit_toplevel_callable(&mut |_| is_callable = true);
        is_callable
    }

    // This doesn't handle generics currently
    pub fn callable_return_type(&self, heap: &TypeHeap) -> Option<Type> {
        let mut rets = Vec::new();
        let mut get_ret = |callable: &Callable| {
            rets.push(callable.ret.clone());
        };
        self.visit_toplevel_callable(&mut get_ret);
        if rets.is_empty() {
            None
        } else {
            Some(unions(rets, heap))
        }
    }

    pub fn callable_first_param(&self, heap: &TypeHeap) -> Option<Type> {
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
            Some(unions(params, heap))
        }
    }

    pub fn callable_signatures(&self) -> Vec<&Callable> {
        let mut sigs = Vec::new();
        self.visit_toplevel_callable(&mut |sig| sigs.push(sig));
        sigs
    }

    fn widen_one_implicit_literal(ty: &mut Type, stdlib: &Stdlib) {
        match &*ty {
            Type::Literal(lit) if lit.style == LitStyle::Implicit => {
                *ty = lit.value.general_class_type(stdlib).clone().to_type()
            }
            Type::LiteralString(LitStyle::Implicit) => *ty = stdlib.str().clone().to_type(),
            _ => {}
        }
    }

    /// Like `promote_implicit_literals` but only recurses into unions.
    pub fn promote_shallow_implicit_literals(mut self, stdlib: &Stdlib) -> Type {
        match &mut self {
            Type::Union(union) => {
                for member in &mut union.members {
                    Self::widen_one_implicit_literal(member, stdlib);
                }
            }
            _ => Self::widen_one_implicit_literal(&mut self, stdlib),
        }
        self
    }

    pub fn promote_implicit_literals(mut self, stdlib: &Stdlib) -> Type {
        fn g(ty: &mut Type, f: &mut dyn FnMut(&mut Type)) {
            // Don't recurse into NNModule fields — they carry captured constructor
            // args (e.g., padding=Literal[1]) that DSL forward functions need as
            // concrete literals to compute output shapes.
            if matches!(ty, Type::NNModule(_)) {
                return;
            }
            ty.recurse_mut(&mut |ty| g(ty, f));
            f(ty);
        }
        g(&mut self, &mut |ty| {
            Self::widen_one_implicit_literal(ty, stdlib)
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

    /// Canonicalize a dimension expression to a unique normal form.
    ///
    /// This transforms dimension expressions into a canonical form where:
    /// - Like terms are combined (e.g., 4*N + 2*N = 6*N)
    /// - Divisions are flattened (e.g., (N // M) // K = N // (M*K))
    /// - Factors are GCD-reduced (e.g., (4*N) // (6*M) = (2*N) // (3*M))
    /// - Expressions are ordered consistently
    /// - Type::Any propagates through the entire expression
    ///
    /// This enables structural equality checking after canonicalization.
    pub fn canonicalize(self) -> Self {
        dimension::canonicalize(self)
    }

    pub fn explicit_any(self) -> Self {
        self.transform(&mut |ty| {
            if let Type::Any(style) = ty {
                *style = AnyStyle::Explicit;
            }
        })
    }

    pub fn explicit_literals(self) -> Self {
        self.transform(&mut |ty| {
            if let Type::Literal(lit) = ty {
                lit.style = LitStyle::Explicit;
            } else if let Type::LiteralString(style) = ty {
                *style = LitStyle::Explicit;
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

    /// Extract the literal value from a `SizeExpr::Literal`, if this is one.
    pub fn as_shape_literal(&self) -> Option<i64> {
        match self {
            Type::Size(SizeExpr::Literal(n)) => Some(*n),
            _ => None,
        }
    }

    pub fn into_unions(self) -> Vec<Type> {
        match self {
            Type::Union(box Union { members: types, .. }) => types,
            _ => vec![self],
        }
    }

    /// Create an optional type (union with None).
    pub fn optional(x: Self) -> Self {
        // We would like the resulting type not nested, and well sorted.
        if let Type::Union(box Union {
            members: mut xs, ..
        }) = x
        {
            match xs.binary_search(&Type::None) {
                Ok(_) => Type::union(xs),
                Err(i) => {
                    xs.insert(i, Type::None);
                    Type::union(xs)
                }
            }
        } else {
            match x.cmp(&Type::None) {
                Ordering::Equal => Type::None,
                Ordering::Less => Type::union(vec![x, Type::None]),
                Ordering::Greater => Type::union(vec![Type::None, x]),
            }
        }
    }

    /// Does this type have a QName associated with it
    pub fn qname(&self) -> Option<&QName> {
        match self {
            Type::ClassDef(cls) => Some(cls.qname()),
            Type::ClassType(c) => Some(c.qname()),
            Type::TypedDict(TypedDict::TypedDict(c)) => Some(c.qname()),
            Type::PartialTypedDict(TypedDict::TypedDict(c)) => Some(c.qname()),
            Type::TypeVar(t) => Some(t.qname()),
            Type::TypeVarTuple(t) => Some(t.qname()),
            Type::ParamSpec(t) => Some(t.qname()),
            Type::SelfType(cls) => Some(cls.qname()),
            Type::Literal(lit) if let Lit::Enum(e) = &lit.value => Some(e.class.qname()),
            _ => None,
        }
    }

    // The result of calling bool() on a value of this type if we can get a definitive answer, None otherwise.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Type::Literal(lit) if let Lit::Bool(x) = &lit.value => Some(*x),
            Type::Literal(lit) if let Lit::Int(x) = &lit.value => Some(x.as_bool()),
            Type::Literal(lit) if let Lit::Bytes(x) = &lit.value => Some(!x.is_empty()),
            Type::Literal(lit) if let Lit::Str(x) = &lit.value => Some(!x.is_empty()),
            Type::Type(_) => Some(true),
            Type::None => Some(false),
            Type::Tuple(Tuple::Concrete(elements)) => Some(!elements.is_empty()),
            Type::Union(box Union { members, .. }) => {
                let mut answer = None;
                for option in members {
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

    /// Return the FunctionKind if this type corresponds to a function or method.
    pub fn to_func_kind(&self) -> Option<&FunctionKind> {
        self.visit_toplevel_func_metadata(&|meta| Some(&meta.kind))
    }

    pub fn materialize(&self) -> Self {
        let mut ty = self.clone();
        ty.transform_types_in_type_variable_positions(&mut |ty| {
            if ty.is_any() {
                *ty = Type::Materialization;
            }
            ty.transform_toplevel_callable(&mut |callable: &mut Callable| {
                if matches!(callable.params, Params::Ellipsis) {
                    callable.params = Params::Materialization;
                }
            });
        });
        ty
    }

    /// Creates a union from the provided types without simplifying
    pub fn union(members: Vec<Type>) -> Self {
        Type::Union(Box::new(Union {
            members,
            display_name: None,
        }))
    }

    /// Returns `true` if this type is an explicit type variable — i.e., a `Quantified` or
    /// legacy `TypeVar` that a user wrote in an annotation.
    pub fn is_explicit_type_variable(&self) -> bool {
        match self {
            Type::Quantified(q) => q.is_type_var(),
            Type::TypeVar(_) => true,
            _ => false,
        }
    }
}

/// Various type-variable-like things
enum TypeVariable<'a> {
    /// A function or class type parameter created from a reference to an in-scope legacy or scoped type variable
    Quantified(&'a Quantified),
    /// A legacy typing.TypeVar appearing in a position where it is not resolved to an in-scope type variable
    LegacyTypeVar(&'a TypeVar),
    /// A legacy typing.TypeVarTuple appearing in a position where it is not resolved to an in-scope type variable
    LegacyTypeVarTuple(&'a TypeVarTuple),
    /// A legacy typing.ParamSpec appearing in a position where it is not resolved to an in-scope type variable
    LegacyParamSpec(&'a ParamSpec),
    /// A placeholder type that may have been instantiated from a Quantified
    Var(Var),
}

impl<'a> TypeVariable<'a> {
    fn new(ty: &'a Type) -> Option<Self> {
        match ty {
            Type::Quantified(q) => Some(Self::Quantified(q)),
            Type::TypeVar(t) => Some(Self::LegacyTypeVar(t)),
            Type::TypeVarTuple(t) => Some(Self::LegacyTypeVarTuple(t)),
            Type::ParamSpec(p) => Some(Self::LegacyParamSpec(p)),
            Type::Var(v) => Some(Self::Var(*v)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::literal::Lit;
    use crate::literal::LitStyle;
    use crate::types::Type;

    #[test]
    fn test_as_bool() {
        let true_lit = Lit::Bool(true).to_implicit_type();
        let false_lit = Lit::Bool(false).to_implicit_type();
        let none = Type::None;
        let s = Type::LiteralString(LitStyle::Implicit);

        assert_eq!(true_lit.as_bool(), Some(true));
        assert_eq!(false_lit.as_bool(), Some(false));
        assert_eq!(none.as_bool(), Some(false));
        assert_eq!(s.as_bool(), None);
    }

    #[test]
    fn test_as_bool_union() {
        let s = Type::LiteralString(LitStyle::Implicit);
        let false_lit = Lit::Bool(false).to_implicit_type();
        let none = Type::None;

        let str_opt = Type::union(vec![s, none.clone()]);
        let false_opt = Type::union(vec![false_lit, none]);

        assert_eq!(str_opt.as_bool(), None);
        assert_eq!(false_opt.as_bool(), Some(false));
    }

    #[test]
    fn test_truncate_class_nesting_preserves_top_level_union_width() {
        let wide_union = Type::union(vec![
            Type::None,
            Type::LiteralString(LitStyle::Implicit),
            Lit::Bool(false).to_implicit_type(),
            Lit::Bool(true).to_implicit_type(),
        ]);

        assert_eq!(
            wide_union
                .clone()
                .truncate_class_nesting(3, 3, &Type::any_implicit()),
            wide_union,
        );
    }
}
