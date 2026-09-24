/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_types::dimension::Int;
use pyrefly_types::dimension::gradual_size;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::shaped_array::IntTupleView;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::class_field::ClassFieldVariance;
use crate::alt::types::class_bases::ClassBases;
use crate::types::callable::Callable;
use crate::types::callable::Params;
use crate::types::class::Class;
use crate::types::function::FuncMetadata;
use crate::types::quantified::Quantified;
use crate::types::tuple::Tuple;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Variance;
use crate::types::types::Forallable;
use crate::types::types::OverloadType;
use crate::types::types::TParams;
use crate::types::types::Type;

// This is our variance inference algorithm, which determines variance based on visiting the structure of the type.
// There are a couple of TODO that I [zeina] would like to revisit as I figure them out. There are several types that I'm not visiting (and did not visit similar ones in pyre1),
// And I'm not yet clear what variance inference should do on those:

// Those types are:
// - Concatenate
// - Intersect (Our variance inference algorithm is not defined on this. Unclear to me yet what to do on this type.)
// - Forall (I suspect that we should not visit this, since the forall type is related to a function, and variance makes no sense in the absence of a class definition)
// - Unpack (potentially just visit the inner type recursively?)
// - SpecialForm
// - ParamSpecValue
// - Args and Kwargs
// - SuperInstance
// - TypeGuard
// - TypeIs

// We need to visit the types that we know are required to be visited for variance inference, and appear in the context of a class with type variables.
// For example, SelfType is intentionally skipped and should not be visited because it should not be included in the variance calculation.

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, Default, Visit, VisitMut)]
pub struct VarianceMap(SmallMap<Name, Variance>);

static EMPTY_VARIANCE_MAP: VarianceMap = VarianceMap(SmallMap::new());

impl Display for VarianceMap {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{{")?;
        for (key, value) in self.0.iter() {
            write!(f, "{key}: {value}, ")?;
        }
        write!(f, "}}")
    }
}

impl VarianceMap {
    pub fn empty() -> &'static Self {
        &EMPTY_VARIANCE_MAP
    }

    pub fn get(&self, parameter: &Name) -> Variance {
        self.0
            .get(parameter)
            .copied()
            .unwrap_or(Variance::Invariant)
    }
}

#[derive(Debug, Clone)]
pub struct VarianceViolation {
    pub range: TextRange,
    pub var_name: Name,
    pub position_variance: Variance,
    pub declared_variance: PreInferenceVariance,
}

impl VarianceViolation {
    pub fn format_message(&self) -> String {
        format!(
            "Type variable `{}` is {} but is used in {} position",
            self.var_name, self.declared_variance, self.position_variance
        )
    }
}

/// A concrete direction discovered by structural inference.
///
/// Unlike [`Variance`], this type deliberately has no bivariant case: the inference
/// algorithm represents "no evidence yet" as [`InferenceState::Unresolved`] rather than
/// conflating it with a variance direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectionalVariance {
    Covariant,
    Contravariant,
    Invariant,
}

impl DirectionalVariance {
    fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Covariant, Self::Covariant) => Self::Covariant,
            (Self::Contravariant, Self::Contravariant) => Self::Contravariant,
            _ => Self::Invariant,
        }
    }

    fn compose(self, other: Self) -> Self {
        match (self, other) {
            (Self::Invariant, _) | (_, Self::Invariant) => Self::Invariant,
            (Self::Covariant, Self::Covariant) | (Self::Contravariant, Self::Contravariant) => {
                Self::Covariant
            }
            _ => Self::Contravariant,
        }
    }
}

impl From<DirectionalVariance> for Variance {
    fn from(variance: DirectionalVariance) -> Self {
        match variance {
            DirectionalVariance::Covariant => Self::Covariant,
            DirectionalVariance::Contravariant => Self::Contravariant,
            DirectionalVariance::Invariant => Self::Invariant,
        }
    }
}

/// Structural variance inference is a fixpoint that starts with no evidence for
/// each inferred parameter. An unresolved recursive generic edge can still
/// produce directionally useful evidence because the legacy composition treats
/// bivariant/no evidence as identity, but that evidence is provisional: a later
/// iteration may resolve the edge to the opposite direction. Grounded evidence
/// therefore takes precedence over provisional evidence, while evidence within
/// either category can strengthen from co- or contravariant to invariant.
///
/// An alternative is to model true bivariance with absorbing composition. That
/// could simplify the fixpoint and leave recursive-only or phantom parameters
/// genuinely bivariant. TODO(stroxler): adopting that model requires changing
/// downstream generic comparison, which currently interprets the legacy
/// `Variance::Bivariant` result as invariant/consistency rather than true
/// bivariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferenceState {
    /// No structural occurrence has contributed a direction yet.
    Unresolved,
    /// Evidence whose path crossed an inferred generic parameter that was not yet grounded.
    /// A later fixpoint iteration may therefore change its direction.
    Provisional(DirectionalVariance),
    /// Evidence whose entire path crossed only specified or grounded generic parameters.
    /// It can strengthen to invariant, but never needs to be retracted.
    Grounded(DirectionalVariance),
}

impl InferenceState {
    fn merge(self, incoming: Self) -> Self {
        match (self, incoming) {
            (state, Self::Unresolved) | (Self::Unresolved, state) => state,
            (Self::Provisional(left), Self::Provisional(right)) => {
                Self::Provisional(left.union(right))
            }
            (Self::Provisional(_), Self::Grounded(grounded)) => Self::Grounded(grounded),
            (Self::Grounded(grounded), Self::Provisional(_)) => Self::Grounded(grounded),
            (Self::Grounded(left), Self::Grounded(right)) => Self::Grounded(left.union(right)),
        }
    }

    fn direction(self) -> Option<DirectionalVariance> {
        match self {
            Self::Unresolved => None,
            Self::Provisional(variance) | Self::Grounded(variance) => Some(variance),
        }
    }
}

/// The variance associated with a generic parameter while walking another type.
///
/// Explicitly specified variance does not participate in the fixpoint. Inferred variance
/// retains whether its evidence is unresolved, provisional, or grounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterVariance {
    Inferred(InferenceState),
    Specified(DirectionalVariance),
}

impl ParameterVariance {
    fn from_quantified(param: &Quantified) -> Self {
        match param.variance() {
            PreInferenceVariance::Covariant => Self::Specified(DirectionalVariance::Covariant),
            PreInferenceVariance::Contravariant => {
                Self::Specified(DirectionalVariance::Contravariant)
            }
            PreInferenceVariance::Invariant => Self::Specified(DirectionalVariance::Invariant),
            PreInferenceVariance::Undefined => Self::Inferred(InferenceState::Unresolved),
        }
    }

    fn effective(self) -> Variance {
        match self {
            Self::Specified(variance) => variance.into(),
            Self::Inferred(state) => state.direction().map_or(Variance::Bivariant, Into::into),
        }
    }

    fn needs_inference(self) -> bool {
        matches!(self, Self::Inferred(_))
    }

    fn merge(&mut self, incoming: InferenceState) {
        if let Self::Inferred(state) = self {
            *state = state.merge(incoming);
        }
    }
}

/// Direction and provenance accumulated along a path from a class member or base to a type
/// parameter occurrence.
///
/// An absent direction lets base-class traversal defer choosing one until it crosses a generic
/// parameter. Provisional provenance is sticky: composing with a grounded nested parameter
/// cannot make an earlier unresolved edge grounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VariancePath {
    direction: Option<DirectionalVariance>,
    grounded: bool,
}

impl VariancePath {
    fn identity() -> Self {
        Self {
            direction: None,
            grounded: true,
        }
    }

    fn grounded(variance: DirectionalVariance) -> Self {
        Self {
            direction: Some(variance),
            grounded: true,
        }
    }

    fn compose(self, parameter: ParameterVariance) -> Self {
        match parameter {
            ParameterVariance::Specified(variance)
            | ParameterVariance::Inferred(InferenceState::Grounded(variance)) => {
                self.compose_direction(variance)
            }
            ParameterVariance::Inferred(InferenceState::Provisional(variance)) => {
                self.compose_direction(variance).provisional()
            }
            ParameterVariance::Inferred(InferenceState::Unresolved) => self.provisional(),
        }
    }

    fn compose_direction(self, variance: DirectionalVariance) -> Self {
        Self {
            direction: Some(match self.direction {
                None => variance,
                Some(outer) => outer.compose(variance),
            }),
            ..self
        }
    }

    fn with_direction(self, variance: DirectionalVariance) -> Self {
        Self {
            direction: Some(variance),
            ..self
        }
    }

    fn invert(self) -> Self {
        self.compose_direction(DirectionalVariance::Contravariant)
    }

    fn provisional(self) -> Self {
        Self {
            grounded: false,
            ..self
        }
    }

    fn into_state(self) -> InferenceState {
        match (self.direction, self.grounded) {
            (None, _) => InferenceState::Unresolved,
            (Some(variance), true) => InferenceState::Grounded(variance),
            (Some(variance), false) => InferenceState::Provisional(variance),
        }
    }
}

type InferenceMap = SmallMap<Name, ParameterVariance>;

// A map from class name to tparam environment
// Why is this not Class or ClassObject
type VarianceEnv = SmallMap<Class, InferenceMap>;

fn handle_tuple_type(
    tuple: &Tuple,
    path: VariancePath,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
) {
    match tuple {
        Tuple::Concrete(concrete_types) => {
            for ty in concrete_types {
                on_type(path, ty, on_edge, on_var);
            }
        }
        Tuple::Unbounded(unbounded_ty) => {
            on_type(path, unbounded_ty, on_edge, on_var);
        }
        Tuple::Unpacked(boxed_parts) => {
            let (before, middle, after) = boxed_parts.parts();
            for ty in before {
                on_type(path, ty, on_edge, on_var);
            }
            on_type(path, middle, on_edge, on_var);
            for ty in after {
                on_type(path, ty, on_edge, on_var);
            }
        }
    }
}

fn on_int(
    dim: &Int,
    path: VariancePath,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
) {
    match dim {
        Int::Literal(_) | Int::Int => {}
        Int::Symbolic(ty) => {
            on_type(
                path.with_direction(DirectionalVariance::Invariant),
                ty,
                on_edge,
                on_var,
            );
        }
        Int::Add(left, right)
        | Int::Sub(left, right)
        | Int::Mul(left, right)
        | Int::FloorDiv(left, right)
        | Int::Pow(left, right) => {
            on_int(left, path, on_edge, on_var);
            on_int(right, path, on_edge, on_var);
        }
    }
}

fn on_type(
    path: VariancePath,
    typ: &Type,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
) {
    let mut is_callable = false;
    for (callable, _) in typ.toplevel_callable_signatures() {
        on_callable(path, callable, false, on_edge, on_var);
        is_callable = true;
    }
    if is_callable {
        return;
    }

    match typ {
        Type::Type(t) => {
            on_type(path, t, on_edge, on_var);
        }
        Type::ClassType(class) => {
            let targs = class.targs().as_slice();

            // If targs is empty, nothing to do. Check this before calling on_edge
            // to avoid expensive environment lookups for non-generic classes.
            if targs.is_empty() {
                return;
            }

            let params = on_edge(class.class_object());

            // Zip params (from on_edge) with targs
            // Note: if params.len() != targs.len(), zip will stop at the shorter one
            for (parameter, ty) in params.values().zip(targs) {
                on_type(path.compose(*parameter), ty, on_edge, on_var);
            }
        }
        Type::Quantified(q) => {
            on_var(q.name(), path.into_state(), q.variance());
        }
        Type::Union(f) => {
            for ty in &f.members {
                on_type(path, ty, on_edge, on_var);
            }
        }
        Type::ShapedArray(tensor) => {
            // Tensor dimensions are invariant - Tensor[2, 3] is not a subtype of Tensor[3, 2]
            let mut visit_dim = |ty: &Type| {
                on_type(
                    path.with_direction(DirectionalVariance::Invariant),
                    ty,
                    on_edge,
                    on_var,
                );
            };
            match tensor.shape().view() {
                IntTupleView::Concrete(dims) => {
                    for dim in dims {
                        visit_dim(&Type::Int(dim.clone()));
                    }
                }
                IntTupleView::Gradual => {
                    let middle = gradual_size();
                    visit_dim(&middle);
                }
                IntTupleView::Unpacked {
                    prefix,
                    middle,
                    suffix,
                } => {
                    for dim in prefix {
                        visit_dim(&Type::Int(dim.clone()));
                    }
                    visit_dim(middle);
                    for dim in suffix {
                        visit_dim(&Type::Int(dim.clone()));
                    }
                }
            }
        }
        Type::NNModule(module) => {
            // NNModule fields are invariant
            for (_, ty) in module.fields.iter() {
                on_type(
                    path.with_direction(DirectionalVariance::Invariant),
                    ty,
                    on_edge,
                    on_var,
                );
            }
        }
        Type::DataFrame(schema) => {
            on_type(path, &schema.underlying_type(), on_edge, on_var);
        }
        Type::Series(schema) => {
            on_type(path, &schema.underlying_type(), on_edge, on_var);
        }
        Type::Tuple(t) => {
            handle_tuple_type(t, path, on_edge, on_var);
        }
        Type::Int(dim) => {
            // Symbolic integer expressions contain types, all invariant.
            on_int(dim, path, on_edge, on_var);
        }
        _ => {}
    }
}

fn on_callable(
    path: VariancePath,
    callable: &Callable,
    skip_receiver: bool,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
) {
    // Walk return type covariantly.
    on_type(path, &callable.ret, on_edge, on_var);

    // Walk parameters contravariantly. Receiver-bound methods skip their first parameter
    // because lookup either binds it from dynamic dispatch or requantifies it for class access.
    match &callable.params {
        Params::List(param_list) | Params::Partial(param_list) => {
            for param in param_list.items().iter().skip(usize::from(skip_receiver)) {
                on_type(path.invert(), param.as_type(), on_edge, on_var);
            }
        }
        Params::Ellipsis | Params::Materialization => {
            // Unknown params
        }
        Params::ParamSpec(prefix, param_spec) => {
            for p in prefix.iter().skip(usize::from(skip_receiver)) {
                on_type(path.invert(), p.ty(), on_edge, on_var);
            }
            on_type(path.invert(), param_spec, on_edge, on_var);
        }
    }
}

fn on_method(
    path: VariancePath,
    typ: &Type,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
) {
    on_method_impl(path, typ, true, on_edge, on_var);
}

fn on_method_impl(
    path: VariancePath,
    typ: &Type,
    metadata_free_callable_is_method: bool,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
) {
    let skip_receiver = |metadata: &FuncMetadata| !metadata.flags.is_staticmethod;
    match typ {
        Type::Callable(callable) if metadata_free_callable_is_method => {
            on_callable(path, callable, true, on_edge, on_var)
        }
        Type::Callable(_) => on_type(path, typ, on_edge, on_var),
        Type::Function(func) => on_callable(
            path,
            &func.signature,
            skip_receiver(&func.metadata),
            on_edge,
            on_var,
        ),
        Type::Forall(forall) => match &forall.body {
            Forallable::Callable(callable) if metadata_free_callable_is_method => {
                on_callable(path, callable, true, on_edge, on_var)
            }
            Forallable::Callable(_) => on_type(path, typ, on_edge, on_var),
            Forallable::Function(func) => on_callable(
                path,
                &func.signature,
                skip_receiver(&func.metadata),
                on_edge,
                on_var,
            ),
            Forallable::TypeAlias(_) => on_type(path, typ, on_edge, on_var),
        },
        Type::Overload(overload) => {
            for signature in overload.signatures.iter() {
                match signature {
                    OverloadType::Function(func) => on_callable(
                        path,
                        &func.signature,
                        skip_receiver(&func.metadata),
                        on_edge,
                        on_var,
                    ),
                    OverloadType::Forall(forall) => on_callable(
                        path,
                        &forall.body.signature,
                        skip_receiver(&forall.body.metadata),
                        on_edge,
                        on_var,
                    ),
                }
            }
        }
        Type::Union(union) => {
            for ty in &union.members {
                on_method_impl(path, ty, false, on_edge, on_var);
            }
        }
        _ => on_type(path, typ, on_edge, on_var),
    }
}

fn on_class<'s>(
    class: &Class,
    heap: &TypeHeap,
    on_edge: &mut impl FnMut(&Class) -> InferenceMap,
    on_var: &mut impl FnMut(&Name, InferenceState, PreInferenceVariance),
    get_class_bases: &impl Fn(&Class) -> &'s ClassBases,
    get_fields: &impl Fn(&Class) -> SmallMap<Name, &'s ClassField>,
) {
    fn is_private_field(name: &Name) -> bool {
        let starts_with_underscore = name.starts_with('_');
        let ends_with_double_underscore = name.ends_with("__");

        starts_with_underscore && !ends_with_double_underscore
    }

    for base_type in get_class_bases(class).iter() {
        // A base contributes only the variance of its own parameters, so start
        // with the composition identity rather than adding a positional direction.
        on_type(
            VariancePath::identity(),
            &heap.mk_class_type(base_type.clone()),
            on_edge,
            on_var,
        );
    }

    let fields = get_fields(class);

    // todo zeina: check if we need to check for things like __init_subclass__
    // in pyre 1, we didn't need to.
    for (name, field) in fields.iter() {
        if name == &dunder::INIT || name == &dunder::NEW {
            continue;
        }

        match field.variance_inference() {
            ClassFieldVariance::Method(ty) => {
                on_method(
                    VariancePath::grounded(DirectionalVariance::Covariant),
                    ty,
                    on_edge,
                    on_var,
                );
            }
            ClassFieldVariance::Property(ty) => {
                on_method(
                    VariancePath::grounded(DirectionalVariance::Covariant),
                    ty,
                    on_edge,
                    on_var,
                );
                // For properties with both a getter and setter, the stored type is the setter
                // function, but the getter is stored separately. Walk it so its covariant
                // contribution is counted.
                if let Some(getter) = ty.is_property_setter_with_getter() {
                    on_method(
                        VariancePath::grounded(DirectionalVariance::Covariant),
                        &getter,
                        on_edge,
                        on_var,
                    );
                }
            }
            ClassFieldVariance::Field { ty, read_only } => {
                let variance = if is_private_field(name) || read_only || field.is_final() {
                    DirectionalVariance::Covariant
                } else {
                    DirectionalVariance::Invariant
                };
                on_type(VariancePath::grounded(variance), ty, on_edge, on_var);
            }
        }
    }
}

/// Check a type variable for variance violations.
fn check_typevar(
    name: &Name,
    position_variance: Variance,
    declared_variance: PreInferenceVariance,
    range: TextRange,
    violations: &mut Vec<VarianceViolation>,
) {
    let is_valid = match declared_variance {
        PreInferenceVariance::Covariant => position_variance == Variance::Covariant,
        PreInferenceVariance::Contravariant => position_variance == Variance::Contravariant,
        // Invariant type variables can be used in any position (covariant, contravariant, or both)
        PreInferenceVariance::Invariant => true,
        // PEP695: variance will be inferred, no check needed
        PreInferenceVariance::Undefined => true,
    };
    if !is_valid {
        violations.push(VarianceViolation {
            range,
            var_name: name.clone(),
            position_variance,
            declared_variance,
        });
    }
}

/// Check a single callable signature for variance violations at `range`.
/// The return type is a covariant position; parameters are contravariant.
fn check_callable_variance(
    callable: &Callable,
    range: TextRange,
    violations: &mut Vec<VarianceViolation>,
) {
    if let Type::Quantified(q) = &callable.ret {
        check_typevar(
            q.name(),
            Variance::Covariant,
            q.variance(),
            range,
            violations,
        );
    }
    if let Params::List(param_list) | Params::Partial(param_list) = &callable.params {
        for param in param_list.items().iter() {
            if let Type::Quantified(q) = param.as_type() {
                check_typevar(
                    q.name(),
                    Variance::Contravariant,
                    q.variance(),
                    range,
                    violations,
                );
            }
        }
    }
}

fn initial_inference_map(tparams: Option<&TParams>) -> InferenceMap {
    tparams
        .iter()
        .flat_map(|tparams| tparams.iter())
        .map(|param| {
            (
                param.name().clone(),
                ParameterVariance::from_quantified(param),
            )
        })
        .collect::<InferenceMap>()
}

fn initialize_environment_impl<Ans: LookupAnswer>(
    class: &Class,
    solver: &AnswersSolver<'_, '_, Ans>,
    environment: &mut VarianceEnv,
) -> InferenceMap {
    if let Some(params) = environment.get(class) {
        return params.clone();
    }

    let params = initial_inference_map(solver.get_class_tparams(class).map(|t| &**t));

    environment.insert(class.dupe(), params.clone());
    let mut on_var = |_name: &Name, _state: InferenceState, _: PreInferenceVariance| {};

    // get the variance results of a given class c
    let mut on_edge = |c: &Class| initialize_environment_impl(c, solver, environment);

    on_class(
        class,
        solver.heap,
        &mut on_edge,
        &mut on_var,
        &|c| solver.get_base_types_for_class(c),
        &|c| solver.get_class_field_map(c),
    );

    params
}

fn initialize_environment<Ans: LookupAnswer>(
    class: &Class,
    solver: &AnswersSolver<'_, '_, Ans>,
    environment: &mut VarianceEnv,
) {
    let mut on_var = |_name: &Name, _state: InferenceState, _: PreInferenceVariance| {};
    let mut on_edge = |c: &Class| initialize_environment_impl(c, solver, environment);
    on_class(
        class,
        solver.heap,
        &mut on_edge,
        &mut on_var,
        &|c| solver.get_base_types_for_class(c),
        &|c| solver.get_class_field_map(c),
    );
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    fn compute_variance_env(&self, class: &Class) -> VarianceEnv {
        let initial_inference_map_for_class =
            initial_inference_map(self.get_class_tparams(class).map(|t| &**t));
        let need_inference = initial_inference_map_for_class
            .values()
            .any(|parameter| parameter.needs_inference());
        if !need_inference {
            let mut environment = VarianceEnv::new();
            environment.insert(class.dupe(), initial_inference_map_for_class);
            environment
        } else {
            self.infer_variance_env(class, initial_inference_map_for_class)
        }
    }

    /// Initialize the variance environment for `class` and its related classes,
    /// then run the fixpoint algorithm to infer variances from structural usage.
    fn infer_variance_env(&self, class: &Class, inference_map: InferenceMap) -> VarianceEnv {
        let mut environment = VarianceEnv::new();
        environment.insert(class.dupe(), inference_map);
        initialize_environment(class, self, &mut environment);
        self.fixpoint(environment)
    }

    /// Run the fixpoint to convergence. Evidence within each state is unioned
    /// monotonically, while grounded evidence replaces provisional evidence
    /// instead of unioning with a direction that may have changed across the
    /// unresolved recursive edge.
    fn fixpoint(&self, mut env: VarianceEnv) -> VarianceEnv {
        let mut changed = true;

        while changed {
            changed = false;
            let mut new_environment: VarianceEnv = SmallMap::new();

            for (my_class, params) in env.iter() {
                let mut new_params = params.clone();

                let mut on_var = |name: &Name, state: InferenceState, _: PreInferenceVariance| {
                    if let Some(parameter) = new_params.get_mut(name) {
                        parameter.merge(state);
                    }
                };
                let mut on_edge = |c: &Class| env.get(c).cloned().unwrap_or_default();
                on_class(
                    my_class,
                    self.heap,
                    &mut on_edge,
                    &mut on_var,
                    &|c| self.get_base_types_for_class(c),
                    &|c| self.get_class_field_map(c),
                );
                if &new_params != params {
                    changed = true;
                }
                new_environment.insert(my_class.dupe(), new_params);
            }
            env = new_environment;
        }
        env
    }

    /// Infer variance from structural usage, ignoring declared variance.
    /// All type params are treated as having undefined variance, and the
    /// fixpoint algorithm discovers what the structure implies.
    pub fn infer_variance_ignoring_declared(&self, class: &Class) -> VarianceMap {
        let tparams = self.get_class_tparams(class);
        let inference_map = tparams
            .iter()
            .flat_map(|tparams| tparams.iter())
            .map(|param| {
                (
                    param.name().clone(),
                    ParameterVariance::Inferred(InferenceState::Unresolved),
                )
            })
            .collect::<InferenceMap>();
        let environment = self.infer_variance_env(class, inference_map);
        let class_variances = environment
            .get(class)
            .expect("class must be present in environment")
            .iter()
            .map(|(name, parameter)| (name.clone(), parameter.effective()))
            .collect::<SmallMap<_, _>>();
        VarianceMap(class_variances)
    }

    /// Compute variance for a class.
    pub fn compute_variance(&self, class: &Class) -> VarianceMap {
        let env = self.compute_variance_env(class);
        let class_variances = env
            .get(class)
            .expect("class name must be present in environment")
            .iter()
            .map(|(name, parameter)| (name.clone(), parameter.effective()))
            .collect::<SmallMap<_, _>>();
        VarianceMap(class_variances)
    }

    /// Check a class for variance violations.
    ///
    /// Checking behavior:
    /// - Base classes: DEEP checking (recurse into all nested generics)
    /// - Methods: SHALLOW checking (only direct TypeVar usage, not nested Callables)
    /// - Fields: NO checking (mutable fields constrain variance during inference only)
    pub fn check_variance_violations(
        &self,
        class: &Class,
        class_bases: &ClassBases,
        field_map: &SmallMap<Name, &ClassField>,
    ) -> Vec<VarianceViolation> {
        let mut violations = Vec::new();

        // Check base classes deeply using on_type for traversal
        for (base_type, range) in class_bases.iter_with_ranges() {
            let mut on_var =
                |name: &Name, state: InferenceState, declared: PreInferenceVariance| {
                    // An unresolved path establishes no position to validate. Treating it as
                    // bivariant here would conflate missing evidence with a variance direction.
                    if let Some(variance) = state.direction() {
                        check_typevar(name, variance.into(), declared, range, &mut violations);
                    }
                };
            let mut on_edge =
                |c: &Class| initial_inference_map(self.get_class_tparams(c).map(|t| &**t));
            on_type(
                VariancePath::grounded(DirectionalVariance::Covariant),
                &base_type.clone().to_type(),
                &mut on_edge,
                &mut on_var,
            );
        }

        // Check methods shallowly
        let class_fields = self.get_class_fields(class);
        for (name, field) in field_map.iter() {
            if name == &dunder::INIT || name == &dunder::NEW {
                continue;
            }
            let (ty, _, _) = field.for_variance_inference();
            if ty.is_toplevel_callable() {
                let range = class_fields
                    .and_then(|f| f.field_decl_range(name))
                    .unwrap_or_else(|| class.range());
                self.check_method_shallow(ty, range, &mut violations);
            }
        }

        violations
    }

    /// The `def`-name range of `metadata`'s function.
    /// Current-module only: variance checks a class's own fields, so `def_index` is always
    /// local — we don't resolve cross-module `FuncDefId`s.
    fn func_def_range(&self, metadata: &FuncMetadata) -> Option<TextRange> {
        let func_id = metadata.kind.as_func_def_id()?;
        self.bindings().function_def_range(func_id.def_index)
    }

    /// Check a method's signatures for variance violations (shallow: direct
    /// TypeVars in params/return only, not nested Callables).
    ///
    /// Each overload arm is reported at its own `def` range, taken from that arm's
    /// own metadata — not the merged group metadata, which points at the
    /// implementation or first overload — so the error lands on the offending
    /// overload. Other callable shapes report at `field_range`.
    fn check_method_shallow(
        &self,
        ty: &Type,
        field_range: TextRange,
        violations: &mut Vec<VarianceViolation>,
    ) {
        if let Type::Overload(overload) = ty {
            for signature in overload.signatures.iter() {
                let (callable, metadata) = match signature {
                    OverloadType::Function(func) => (&func.signature, &func.metadata),
                    OverloadType::Forall(forall) => (&forall.body.signature, &forall.body.metadata),
                };
                let range = self.func_def_range(metadata).unwrap_or(field_range);
                check_callable_variance(callable, range, violations);
            }
            return;
        }
        for (callable, _) in ty.toplevel_callable_signatures() {
            check_callable_variance(callable, field_range, violations);
        }
    }
}
