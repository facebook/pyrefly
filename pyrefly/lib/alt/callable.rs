/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::mem;

use itertools::Itertools;
use pyrefly_python::dunder;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::meta_shape_dsl::MetaShapeFunction;
use pyrefly_types::meta_shape_dsl::ShapeTransform;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::TParams;
use pyrefly_util::display::count;
use pyrefly_util::display::pluralize;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::ordered_map::OrderedMap;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::TypeCheckOptions;
use crate::alt::expr::ExprOptions;
use crate::alt::expr::TypeOrExpr;
use crate::alt::solve::Iterable;
use crate::alt::unwrap::HintRef;
use crate::alt::unwrap::MAX_CALL_HINT_WIDTH;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::display::function_suffix;
use crate::solver::solver::ArgumentSide;
use crate::solver::solver::CallContext;
use crate::solver::solver::QuantifiedHandle;
use crate::solver::solver::TypeVarSpecializationError;
use crate::types::callable::Callable;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::quantified::Quantified;
use crate::types::types::Type;
use crate::types::types::Var;

/// Structure to turn TypeOrExprs into Types.
/// This is used to avoid re-inferring types for arguments multiple types.
///
/// Implemented by keeping an `Owner` to hand out references to `Type`.
pub struct CallWithTypes(Owner<Type>);

impl CallWithTypes {
    pub fn new() -> Self {
        Self(Owner::new())
    }

    pub fn type_or_expr<'a, 'b: 'a, Ans: LookupAnswer>(
        &'a self,
        x: TypeOrExpr<'b>,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> TypeOrExpr<'a> {
        match x {
            TypeOrExpr::Expr(e @ (Expr::Dict(_) | Expr::List(_) | Expr::Set(_))) => {
                // Hack: don't flatten mutable builtin containers into types before calling a
                // function, as we know these containers often need to be contextually typed using
                // the function's parameter types.
                TypeOrExpr::Expr(e)
            }
            TypeOrExpr::Expr(e) => {
                let t = solver.expr_infer(e, errors);
                TypeOrExpr::Type(self.0.push(t), e.range())
            }
            TypeOrExpr::Type(t, r) => TypeOrExpr::Type(t, r),
        }
    }

    pub fn call_arg<'a, 'b: 'a, Ans: LookupAnswer>(
        &'a self,
        x: &CallArg<'b>,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> CallArg<'a> {
        match x {
            CallArg::Arg(x) => CallArg::Arg(self.type_or_expr(*x, solver, errors)),
            CallArg::Star(x, r) => CallArg::Star(self.type_or_expr(*x, solver, errors), *r),
        }
    }

    pub fn call_keyword<'a, 'b: 'a, Ans: LookupAnswer>(
        &'a self,
        x: &CallKeyword<'b>,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> CallKeyword<'a> {
        CallKeyword {
            range: x.range,
            arg: x.arg,
            value: self.type_or_expr(x.value, solver, errors),
        }
    }

    pub fn vec_call_arg<'a, 'b: 'a, Ans: LookupAnswer>(
        &'a self,
        xs: &[CallArg<'b>],
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> Vec<CallArg<'a>> {
        xs.map(|x| self.call_arg(x, solver, errors))
    }

    pub fn vec_call_keyword<'a, 'b: 'a, Ans: LookupAnswer>(
        &'a self,
        xs: &[CallKeyword<'b>],
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> Vec<CallKeyword<'a>> {
        xs.map(|x| self.call_keyword(x, solver, errors))
    }
}

#[derive(Clone, Debug)]
pub struct CallKeyword<'a> {
    pub range: TextRange,
    pub arg: Option<&'a Identifier>,
    pub value: TypeOrExpr<'a>,
}

impl Ranged for CallKeyword<'_> {
    fn range(&self) -> TextRange {
        self.range
    }
}

impl<'a> CallKeyword<'a> {
    pub fn new(x: &'a Keyword) -> Self {
        Self {
            range: x.range,
            arg: x.arg.as_ref(),
            value: TypeOrExpr::Expr(&x.value),
        }
    }

    pub fn materialize<Ans: LookupAnswer>(
        &self,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
    ) -> (Self, bool) {
        let transformation = |ty: &Type| {
            if self.arg.is_none() && ty.is_any() {
                // See test::overload::test_kwargs_materialization - we need to turn this
                // into Mapping[str, Any] to correctly materialize the `**kwargs` type.
                solver
                    .heap
                    .mk_class_type(solver.stdlib.mapping(
                        solver.heap.mk_class_type(solver.stdlib.str().clone()),
                        ty.clone(),
                    ))
                    .materialize()
            } else {
                ty.materialize()
            }
        };
        let (materialized, changed) = self.value.transform(solver, errors, owner, transformation);
        (
            Self {
                range: self.range,
                arg: self.arg,
                value: materialized,
            },
            changed,
        )
    }
}

#[derive(Clone, Debug)]
pub enum CallArg<'a> {
    Arg(TypeOrExpr<'a>),
    Star(TypeOrExpr<'a>, TextRange),
}

impl Ranged for CallArg<'_> {
    fn range(&self) -> TextRange {
        match self {
            Self::Arg(x) => x.range(),
            Self::Star(_, r) => *r,
        }
    }
}

impl<'a> CallArg<'a> {
    pub fn arg(x: TypeOrExpr<'a>) -> Self {
        Self::Arg(x)
    }

    pub fn expr(x: &'a Expr) -> Self {
        Self::Arg(TypeOrExpr::Expr(x))
    }

    pub fn ty(ty: &'a Type, range: TextRange) -> Self {
        Self::Arg(TypeOrExpr::Type(ty, range))
    }

    pub fn expr_maybe_starred(x: &'a Expr) -> Self {
        match x {
            Expr::Starred(inner) => Self::Star(TypeOrExpr::Expr(&inner.value), x.range()),
            _ => Self::expr(x),
        }
    }

    pub fn materialize<Ans: LookupAnswer>(
        &self,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
    ) -> (Self, bool) {
        match self {
            Self::Arg(value) => {
                let (materialized, changed) =
                    value.transform(solver, errors, owner, |ty| ty.materialize());
                (Self::Arg(materialized), changed)
            }
            Self::Star(value, range) => {
                let (materialized, changed) = value.transform(solver, errors, owner, |ty| {
                    if ty.is_any() {
                        // See test::overload::test_varargs_materialization - we need to turn this
                        // into Iterable[Any] to correctly materialize the `*args` type.
                        solver
                            .heap
                            .mk_class_type(solver.stdlib.iterable(ty.clone()))
                            .materialize()
                    } else {
                        ty.materialize()
                    }
                });
                (Self::Star(materialized, *range), changed)
            }
        }
    }

    // Splat arguments might be fixed-length tuples, which are handled precisely, or have unknown
    // length. This function evaluates splat args to determine how many params should be consumed,
    // but does not evaluate other expressions, which might be contextually typed.
    fn pre_eval<Ans: LookupAnswer>(
        &self,
        solver: &AnswersSolver<Ans>,
        arg_errors: &ErrorCollector,
    ) -> CallArgPreEval<'_> {
        match self {
            Self::Arg(TypeOrExpr::Type(ty, _)) => CallArgPreEval::Type(ty, false),
            Self::Arg(TypeOrExpr::Expr(e)) => CallArgPreEval::Expr(e, false),
            Self::Star(e, _range) => {
                // Special-case list/set/tuple literals with statically known element count.
                // Only do this if there are no starred elements inside the literal.
                if let TypeOrExpr::Expr(expr) = e {
                    let literal_elts: Option<&[Expr]> = match expr {
                        Expr::List(list_expr) => Some(&list_expr.elts),
                        Expr::Set(set_expr) => Some(&set_expr.elts),
                        Expr::Tuple(tuple_expr) => Some(&tuple_expr.elts),
                        _ => None,
                    };
                    if let Some(elts) = literal_elts {
                        let has_starred = elts.iter().any(|elt| matches!(elt, Expr::Starred(_)));
                        if !has_starred {
                            let tys: Vec<Type> = elts
                                .iter()
                                .map(|elt| solver.expr_infer(elt, arg_errors))
                                .collect();
                            return CallArgPreEval::Fixed(tys, 0);
                        }
                    }
                }
                let ty = e.infer(solver, arg_errors);
                let iterables = solver.iterate(&ty, *_range, arg_errors, None);
                // If we have a union of iterables, use a fixed length only if every iterable is
                // fixed and has the same length. Otherwise, use star.
                let mut fixed_lens = Vec::new();
                for x in iterables.iter() {
                    match x {
                        Iterable::FixedLen(xs) => fixed_lens.push(xs.len()),
                        Iterable::OfType(_)
                        | Iterable::Unpacked { .. }
                        | Iterable::OfTypeVarTuple(_) => {}
                    }
                }
                if !fixed_lens.is_empty()
                    && fixed_lens.len() == iterables.len()
                    && fixed_lens.iter().all(|len| *len == fixed_lens[0])
                {
                    let mut fixed_tys = vec![Vec::new(); fixed_lens[0]];
                    for x in iterables {
                        if let Iterable::FixedLen(xs) = x {
                            for (i, ty) in xs.into_iter().enumerate() {
                                fixed_tys[i].push(ty);
                            }
                        }
                    }
                    let tys = fixed_tys.into_map(|tys| solver.unions(tys));
                    CallArgPreEval::Fixed(tys, 0)
                } else {
                    let ty = solver.get_produced_type(iterables);
                    CallArgPreEval::Star(ty, false)
                }
            }
        }
    }
}

// Pre-evaluated args are iterable. Type/Expr/Star variants iterate once (tracked via bool field),
// Fixed variant iterates over the vec (tracked via usize field).
#[derive(Clone, Debug)]
enum CallArgPreEval<'a> {
    Type(&'a Type, bool),
    Expr(&'a Expr, bool),
    Star(Type, bool),
    Fixed(Vec<Type>, usize),
}

impl CallArgPreEval<'_> {
    fn step(&self) -> bool {
        match self {
            Self::Type(_, done) | Self::Expr(_, done) | Self::Star(_, done) => !*done,
            Self::Fixed(tys, i) => *i < tys.len(),
        }
    }

    fn is_star(&self) -> bool {
        matches!(self, Self::Star(..))
    }

    fn inferred_type<Ans: LookupAnswer>(
        &self,
        solver: &AnswersSolver<Ans>,
        arg_errors: &ErrorCollector,
    ) -> Type {
        match self {
            Self::Type(ty, _) => (*ty).clone(),
            Self::Expr(expr, _) => solver.expr_infer(expr, arg_errors),
            Self::Star(ty, _) => ty.clone(),
            Self::Fixed(tys, idx) => tys[*idx].clone(),
        }
    }

    /// Check the argument against a parameter hint and return the inferred argument type.
    fn post_check<Ans: LookupAnswer>(
        &mut self,
        solver: &AnswersSolver<Ans>,
        callable_name: Option<&FunctionKind>,
        hint: &Type,
        param_name: Option<&Name>,
        vararg: bool,
        range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        call_context: &CallContext,
    ) -> Option<Type> {
        let tcc = &|| {
            TypeCheckContext::of_kind(if vararg {
                TypeCheckKind::CallVarArgs(false, param_name.cloned(), callable_name.cloned())
            } else {
                TypeCheckKind::CallArgument(param_name.cloned(), callable_name.cloned())
            })
            .with_context(context.map(|ctx| ctx()))
        };
        match self {
            Self::Type(ty, done) => {
                *done = true;
                solver.check_type_with_options(
                    ty,
                    hint,
                    range,
                    TypeCheckOptions::new(call_errors, tcc).with_call_context(call_context),
                );
                Some((*ty).clone())
            }
            Self::Expr(x, done) => {
                *done = true;
                // PEP 747: when the parameter type is TypeForm, evaluate
                // string literal arguments as forward-reference type forms.
                if matches!(hint, Type::TypeForm(_))
                    && let Some(ty) = solver.try_string_literal_as_typeform(
                        x,
                        hint,
                        range,
                        call_errors,
                        tcc,
                        call_context,
                    )
                {
                    return Some(ty);
                }
                if matches!(
                    solver.canonicalize_shape_dsl_type(hint.clone()),
                    Type::ShapedArray(_)
                ) {
                    let ty = solver.reproject_tuple_carrier_shape(
                        solver.canonicalize_shape_dsl_type(solver.expr_infer(x, arg_errors)),
                    );
                    solver.check_type_with_options(
                        &ty,
                        hint,
                        range,
                        TypeCheckOptions::new(call_errors, tcc).with_call_context(call_context),
                    );
                    return Some(ty);
                }
                Some(
                    solver
                        .expr_with_options(
                            x,
                            ExprOptions::check(
                                hint,
                                arg_errors,
                                call_errors,
                                tcc,
                                Some(call_context),
                            ),
                        )
                        .into_ty(),
                )
            }
            Self::Star(ty, done) => {
                *done = vararg;
                solver.check_type_with_options(
                    ty,
                    hint,
                    range,
                    TypeCheckOptions::new(call_errors, tcc).with_call_context(call_context),
                );
                Some(ty.clone())
            }
            Self::Fixed(tys, i) => {
                let arg_ty = tys[*i].clone();
                solver.check_type_with_options(
                    &arg_ty,
                    hint,
                    range,
                    TypeCheckOptions::new(call_errors, tcc).with_call_context(call_context),
                );
                *i += 1;
                Some(arg_ty)
            }
        }
    }

    // Step the argument or mark it as done similar to `post_infer`, but without checking the type
    // Intended for arguments matched to unpack-annotated *args, which are typechecked separately later
    fn post_skip(&mut self) {
        match self {
            Self::Type(_, done) | Self::Expr(_, done) | Self::Star(_, done) => {
                *done = true;
            }
            Self::Fixed(_, i) => {
                *i += 1;
            }
        }
    }

    // Similar to post_skip but it skips to the end of any fixed length arguments.
    fn mark_done(&mut self) {
        match self {
            Self::Type(_, done) | Self::Expr(_, done) | Self::Star(_, done) => {
                *done = true;
            }
            Self::Fixed(tys, i) => {
                *i = tys.len();
            }
        }
    }

    fn post_infer<Ans: LookupAnswer>(
        &mut self,
        solver: &AnswersSolver<Ans>,
        arg_errors: &ErrorCollector,
    ) {
        match self {
            Self::Expr(x, _) => {
                solver.expr_infer(x, arg_errors);
            }
            _ => {}
        }
    }
}

/// The parameter an argument was matched against.
#[derive(Debug, Clone)]
pub struct MatchedParam {
    pub ty: Type,
    pub name: Option<Name>,
}

impl MatchedParam {
    fn new(ty: Type, name: Option<Name>) -> Self {
        Self { ty, name }
    }
}

#[derive(Debug, Clone)]
pub struct ArgMap {
    pub range_to_param: HashMap<TextRange, MatchedParam>,
    /// Required parameters that were left unmatched
    pub unmatched_params: SmallSet<Option<Name>>,
}

impl ArgMap {
    pub fn new() -> Self {
        Self {
            range_to_param: HashMap::new(),
            unmatched_params: SmallSet::new(),
        }
    }

    fn insert(&mut self, range: TextRange, ty: Type, name: Option<Name>) -> Option<MatchedParam> {
        self.range_to_param
            .insert(range, MatchedParam::new(ty, name))
    }
}

/// Helps track matching of arguments against positional parameters in AnswersSolver::callable_infer_params.
#[derive(PartialEq, Eq)]
enum PosParamKind {
    PositionalOnly,
    Positional,
    Unpacked,
    Variadic,
}

/// Helps track matching of arguments against positional parameters in AnswersSolver::callable_infer_params.
struct PosParam<'a> {
    ty: &'a Type,
    name: Option<&'a Name>,
    kind: PosParamKind,
}

impl<'a> PosParam<'a> {
    fn new(p: &'a Param) -> Option<Self> {
        match p {
            Param::PosOnly(name, ty, _required) => Some(Self {
                ty,
                name: name.as_ref(),
                kind: PosParamKind::PositionalOnly,
            }),
            Param::Pos(name, ty, _required) => Some(Self {
                ty,
                name: Some(name),
                kind: PosParamKind::Positional,
            }),
            Param::Varargs(name, Type::Unpack(ty)) => Some(Self {
                ty: &**ty,
                name: name.as_ref(),
                kind: PosParamKind::Unpacked,
            }),
            Param::Varargs(name, ty) => Some(Self {
                ty,
                name: name.as_ref(),
                kind: PosParamKind::Variadic,
            }),
            Param::KwOnly(..) | Param::Kwargs(..) => None,
        }
    }
}

/// The origin of a name that has been encountered in a function call
#[derive(Clone, Debug)]
enum NameOrigin<'a> {
    /// Named parameter
    Param,
    /// An unpacked kwargs parameter, e.g., `**kwargs: Unpack[TD]` where `TD` is a TypedDict`.
    /// In this example, the encountered name would be the name of a `TD` field, and the origin
    /// would be the param name "kwargs".
    UnpackedKwargs(Option<&'a Name>),
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn is_param_spec_args(&self, x: &CallArg, q: &Quantified, errors: &ErrorCollector) -> bool {
        match x {
            CallArg::Star(x, _) => {
                let mut ty = x.infer(self, errors);
                self.expand_mut(&mut ty);
                // This can either be `P.args` or `tuple[Any, ...]`
                matches!(&ty, Type::Args(q2) if &**q2 == q)
                    || self.is_subset_eq(&ty, &self.heap.mk_unbounded_tuple(self.heap.mk_never()))
            }
            _ => false,
        }
    }

    fn is_param_spec_kwargs(
        &self,
        x: &CallKeyword,
        q: &Quantified,
        errors: &ErrorCollector,
    ) -> bool {
        let mut ty = x.value.infer(self, errors);
        self.expand_mut(&mut ty);
        // This can either be `P.kwargs` or `dict[str, Any]`
        matches!(&ty, Type::Kwargs(q2) if &**q2 == q)
            || self.is_subset_eq(
                &ty,
                &self.heap.mk_class_type(self.stdlib.dict(
                    self.heap.mk_class_type(self.stdlib.str().clone()),
                    self.heap.mk_never(),
                )),
            )
    }

    /// Validate that a quantified ParamSpec forwarding pattern has the expected
    /// `*P.args` / `**P.kwargs` as the last positional and keyword arguments.
    /// Called when `var_to_rparams` returns `Err(q)` (the Var resolved to a
    /// still-quantified ParamSpec `q`).
    ///
    /// `current_arg` is the arg that triggered ParamSpec expansion (first call
    /// site only). When present, we check that it is `*P.args` — this catches
    /// extra args *before* `*P.args`. We also always check that `args.last()`
    /// is `*P.args` — this catches extra args *after* it and the case where
    /// `*P.args` is missing entirely. On success, return the remaining
    /// arguments after stripping the trailing `*P.args` / `**P.kwargs` pair.
    fn paramspec_forwarding<'b>(
        &self,
        q: &Quantified,
        current_arg: Option<&CallArg<'b>>,
        args: &'b [CallArg<'b>],
        keywords: &'b [CallKeyword<'b>],
        arguments_range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Option<(&'b [CallArg<'b>], &'b [CallKeyword<'b>])> {
        let current_ok = current_arg.is_none_or(|x| self.is_param_spec_args(x, q, arg_errors));
        let last_ok = args
            .last()
            .is_some_and(|x| self.is_param_spec_args(x, q, arg_errors));
        let args_ok = current_ok && last_ok;
        let kwargs_ok = keywords
            .last()
            .is_some_and(|x| self.is_param_spec_kwargs(x, q, arg_errors));
        if !args_ok || !kwargs_ok {
            self.error_with_context(
                call_errors,
                arguments_range,
                ErrorKind::InvalidParamSpec,
                format!(
                    "Expected *-unpacked {}.args and **-unpacked {}.kwargs",
                    q.name(),
                    q.name()
                ),
                context,
            );
            None
        } else {
            Some((&args[..args.len() - 1], &keywords[..keywords.len() - 1]))
        }
    }

    // See comment on `callable_infer` about `arg_errors` and `call_errors`.
    /// Match arguments against parameters, type-check each argument, and return
    /// a map from each argument's source range to the parameter type it was
    /// matched against.
    fn callable_infer_params(
        &self,
        callable_name: Option<&FunctionKind>,
        params: &ParamList,
        // A ParamSpec Var (if any) that comes at the end of the parameter list.
        // See test::paramspec::test_paramspec_twice for an example of this.
        mut paramspec: Option<Var>,
        self_arg: Option<CallArg>,
        mut self_qs: Option<QuantifiedHandle>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        call_context: &CallContext,
        // If Some, records parameter-name → argument-type bindings (for meta-shape inference).
        bound_args: &mut Option<HashMap<String, Type>>,
    ) -> ArgMap {
        let record = |bound: &mut Option<HashMap<String, Type>>, name: &Name, ty: Type| {
            if let Some(map) = bound.as_mut() {
                map.insert(
                    name.to_string(),
                    self.reproject_tuple_carrier_shape(self.canonicalize_shape_dsl_type(ty)),
                );
            }
        };
        let mut argmap = ArgMap::new();
        // We want to work mostly with references, but some things are taken from elsewhere,
        // so have some owners to capture them.
        let param_list_owner = Owner::new();
        let name_owner = Owner::new();
        let type_owner = Owner::new();

        let error = |errors, range, kind, msg: String| {
            self.error_with_context(
                errors,
                range,
                kind,
                format!(
                    "{}{}",
                    msg,
                    function_suffix(callable_name, self.module().name())
                ),
                context,
            )
        };

        let keyword_arg_names: SmallSet<&Name> = keywords
            .iter()
            .filter_map(|kw| kw.arg.map(|id| &id.id))
            .collect();

        // Creates a reversed copy of the parameters that we iterate through from back to front,
        // so that we can easily peek at and pop from the end.
        let mut rparams = params.items().iter().rev().collect::<Vec<_>>();
        let mut num_positional_params = 0;
        let mut extra_positional_args = Vec::new();
        // Map from seen parameter name to (Type, NameOrigin, definitely_seen).
        // NotRequired fields of unpacked typed dicts are not definitely seen.
        let mut seen_names = SmallMap::new();
        let mut extra_arg_pos = None;
        let mut unpacked_vararg = None;
        let mut unpacked_vararg_matched_args = Vec::new();
        let mut variadic_name = None;
        let mut variadic_collected = Vec::new();

        // Resolve a deferred ParamSpec Var into additional parameters.
        // Returns `Err(q)` when the Var resolved to a quantified ParamSpec `q`
        // (forwarding case), meaning the caller should validate that the
        // remaining args are `*P.args` / `**P.kwargs` and stop matching.
        let var_to_rparams = |var| -> Result<Vec<&Param>, Box<Quantified>> {
            let ps = match self.solver().force_var(var) {
                Type::ParamSpecValue(ps) => ps,
                Type::Any(_) | Type::Ellipsis => ParamList::everything(),
                Type::Concatenate(prefix, _) => {
                    // TODO: handle second component of Type::Concatenate
                    let ps = ParamList::everything();
                    ps.prepend_types(&prefix).into_owned()
                }
                // The ParamSpec Var resolved to another quantified ParamSpec (e.g.,
                // one generic helper forwarding `*args: P.args, **kwargs: P.kwargs`
                // to another). There are no concrete parameters to contribute;
                // the caller must validate the forwarding pattern.
                Type::Quantified(q) if q.is_param_spec() => return Err(q),
                t => {
                    error(
                        call_errors,
                        arguments_range,
                        ErrorKind::BadArgumentType,
                        format!("Expected `{}` to be a ParamSpec value", self.for_display(t)),
                    );
                    ParamList::everything()
                }
            };
            Ok(param_list_owner.push(ps).items().iter().rev().collect())
        };
        for arg in self_arg.iter().chain(args.iter()) {
            let mut arg_pre = arg.pre_eval(self, arg_errors);
            while arg_pre.step() {
                let param = if let Some(p) = rparams.last() {
                    PosParam::new(p)
                } else if let Some(var) = paramspec {
                    // We've run out of parameters but haven't finished matching arguments. If we
                    // have a ParamSpec Var, it may contribute more parameters; force it and tack
                    // the result onto the parameter list.
                    match var_to_rparams(var) {
                        Ok(new_rparams) => rparams = new_rparams,
                        Err(q) => {
                            // Quantified ParamSpec forwarding: validate that the
                            // current arg is `*P.args`, it is the last positional
                            // arg, and the last keyword is `**P.kwargs`.
                            let _ = self.paramspec_forwarding(
                                &q,
                                Some(arg),
                                args,
                                keywords,
                                arguments_range,
                                arg_errors,
                                call_errors,
                                context,
                            );
                            return argmap;
                        }
                    }
                    paramspec = None;
                    continue;
                } else {
                    None
                };
                match param {
                    Some(PosParam {
                        ty,
                        name,
                        kind: kind @ (PosParamKind::PositionalOnly | PosParamKind::Positional),
                    }) => {
                        // For unknown-length star args, stop consuming positional parameters
                        // when we reach a one that has a corresponding keyword argument.
                        // This is unsound, but prevents false positive "multiple values" errors.
                        if arg_pre.is_star()
                            && kind == PosParamKind::Positional
                            && name.is_some_and(|n| keyword_arg_names.contains(n))
                        {
                            arg_pre.mark_done();
                            break;
                        }
                        num_positional_params += 1;
                        rparams.pop();
                        if let Some(name) = name
                            && kind == PosParamKind::Positional
                        {
                            // Remember names of positional parameters to detect duplicates.
                            // We ignore positional-only parameters because they can't be passed in by name.
                            seen_names.insert(name, (ty, NameOrigin::Param, true));
                        }
                        argmap.insert(arg.range(), ty.clone(), name.cloned());
                        let unhinted_arg_ty = bound_args
                            .as_ref()
                            .map(|_| arg_pre.inferred_type(self, arg_errors));
                        let arg_ty = arg_pre.post_check(
                            self,
                            callable_name,
                            ty,
                            name,
                            false,
                            arg.range(),
                            arg_errors,
                            call_errors,
                            context,
                            call_context,
                        );
                        if let Some(name) = name
                            && let Some(ty) = unhinted_arg_ty.or(arg_ty)
                        {
                            record(bound_args, name, ty);
                        }
                    }
                    Some(PosParam {
                        ty,
                        name,
                        kind: PosParamKind::Unpacked,
                    }) => {
                        // Store args that get matched to an unpacked *args param
                        // Matched args are typechecked separately later
                        argmap.insert(arg.range(), ty.clone(), name.cloned());
                        unpacked_vararg = Some((name, ty));
                        unpacked_vararg_matched_args.push(arg_pre.clone());
                        arg_pre.post_skip();
                    }
                    Some(PosParam {
                        ty,
                        name,
                        kind: PosParamKind::Variadic,
                    }) => {
                        argmap.insert(arg.range(), ty.clone(), name.cloned());
                        let unhinted_arg_ty = bound_args
                            .as_ref()
                            .map(|_| arg_pre.inferred_type(self, arg_errors));
                        let arg_ty = arg_pre.post_check(
                            self,
                            callable_name,
                            ty,
                            name,
                            true,
                            arg.range(),
                            arg_errors,
                            call_errors,
                            context,
                            call_context,
                        );
                        if bound_args.is_some() {
                            if let Some(name) = name {
                                variadic_name = Some(name);
                            }
                            if let Some(ty) = unhinted_arg_ty.or(arg_ty) {
                                variadic_collected.push(ty);
                            }
                        }
                    }
                    None => {
                        arg_pre.post_infer(self, arg_errors);
                        if !arg_pre.is_star() {
                            extra_positional_args.push(arg.range());
                        }
                        if extra_arg_pos.is_none() && !arg_pre.is_star() {
                            extra_arg_pos = Some(arg.range());
                        }
                        break;
                    }
                }
            }
            // `self_qs` contains type parameters referenced in the `self` type. Pyrefly follows
            // mypy and pyright's lead in solving type parameters in `self` as soon as `self` is
            // matched. That is:
            //     class A:
            //         def f[T](self: T, other: T): ...
            //     A().f(0)  # T = A, passing 0 is an error
            // Contrast this to how type parameters usually behave:
            //     def f[T](x: T, other: T): ...
            //     f(A(), 0)  # T = A | int
            if let Some(self_qs) = mem::take(&mut self_qs) {
                let specialization_errors =
                    self.finish_quantified(self_qs, self.solver().infer_with_first_use);
                if let Err(errors) = specialization_errors {
                    self.add_specialization_errors(errors, arg.range(), call_errors, context);
                }
            }
        }
        // Record collected variadic args as a tuple for meta-shape binding.
        if let Some(name) = variadic_name {
            record(
                bound_args,
                name,
                Type::Tuple(Tuple::Concrete(variadic_collected)),
            );
        }
        if let Some((unpacked_name, unpacked_param_ty)) = unpacked_vararg {
            let mut prefix = Vec::new();
            let mut middle = Vec::new();
            let mut suffix = Vec::new();
            for arg in unpacked_vararg_matched_args {
                match arg {
                    CallArgPreEval::Type(ty, _) => {
                        if middle.is_empty() {
                            prefix.push(ty.clone())
                        } else {
                            suffix.push(ty.clone())
                        }
                    }
                    CallArgPreEval::Expr(e, _) => {
                        if middle.is_empty() {
                            prefix.push(self.expr_infer(e, arg_errors))
                        } else {
                            suffix.push(self.expr_infer(e, arg_errors))
                        }
                    }
                    CallArgPreEval::Fixed(tys, idx) => {
                        if middle.is_empty() {
                            prefix.push(tys[idx].clone());
                        } else {
                            suffix.push(tys[idx].clone());
                        }
                    }
                    CallArgPreEval::Star(ty, _) => {
                        if !middle.is_empty() {
                            middle.extend(suffix);
                            suffix = Vec::new();
                        }
                        middle.push(ty);
                    }
                }
            }
            let unpacked_args_ty = match middle.len() {
                0 => self.heap.mk_concrete_tuple(prefix),
                1 => self.heap.mk_unpacked_tuple(
                    prefix,
                    self.heap.mk_unbounded_tuple(middle.pop().unwrap()),
                    suffix,
                ),
                _ => {
                    let unpacked_variadic_args_count = middle
                        .iter()
                        .filter(|x| matches!(x, Type::ElementOfTypeVarTuple(_)))
                        .count();
                    if unpacked_variadic_args_count > 1 {
                        error(
                            arg_errors,
                            arguments_range,
                            ErrorKind::BadArgumentType,
                            "Expected at most one unpacked variadic argument".to_owned(),
                        );
                    }
                    self.heap.mk_unpacked_tuple(
                        prefix,
                        self.heap.mk_unbounded_tuple(self.unions(middle)),
                        suffix,
                    )
                }
            };
            // The args side (unpacked_args_ty) is always a tuple built from call
            // arguments, e.g., tuple[*Cs] or tuple[int, str]. The param side
            // (unpacked_param_ty) is the raw Ts from stripping * off the type
            // annotation *Ts. Wrap it in a tuple so both sides have the same
            // structure: tuple[*Cs] ⊆ tuple[*Ts] or tuple[int, str] ⊆ tuple[*Ts].
            let unpacked_param_tuple =
                self.heap
                    .mk_unpacked_tuple(Vec::new(), unpacked_param_ty.clone(), Vec::new());
            self.check_type(
                &unpacked_args_ty,
                &unpacked_param_tuple,
                arguments_range,
                call_errors,
                &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::CallVarArgs(
                        true,
                        unpacked_name.cloned(),
                        callable_name.cloned(),
                    ))
                    .with_context(context.map(|ctx| ctx()))
                },
            );
        }
        // Missing positional-only arguments, split by whether the corresponding parameters
        // in the callable have names. E.g., functions declared with `def` have named posonly
        // parameters and `typing.Callable`s have unnamed ones.
        let mut missing_unnamed_posonly = 0;
        let mut missing_named_posonly = SmallSet::new();
        let mut kwparams = OrderedMap::new();
        let mut kwargs = None;
        // Parameters with default values that are not matched to call args.
        let mut default_check = Vec::new();
        loop {
            let p = match rparams.pop() {
                Some(p) => p,
                None if let Some(var) = paramspec => {
                    // We've reached the end of our regular parameter list. Now check if we have more parameters from a ParamSpec.
                    match var_to_rparams(var) {
                        Ok(new_rparams) => rparams = new_rparams,
                        Err(q) => {
                            // Quantified ParamSpec forwarding: no current
                            // positional arg triggered expansion; check that
                            // `*P.args` is the last positional arg and
                            // `**P.kwargs` is the last keyword.
                            let _ = self.paramspec_forwarding(
                                &q,
                                None,
                                args,
                                keywords,
                                arguments_range,
                                arg_errors,
                                call_errors,
                                context,
                            );
                            return argmap;
                        }
                    }
                    paramspec = None;
                    continue;
                }
                None => {
                    break;
                }
            };
            match p {
                Param::PosOnly(name, ty, required) => match required {
                    Required::Required => {
                        if let Some(name) = name {
                            missing_named_posonly.insert(name);
                        } else {
                            missing_unnamed_posonly += 1;
                        }
                    }
                    Required::Optional(Some(default)) => {
                        default_check.push((name.as_ref(), ty, default))
                    }
                    Required::Optional(None) => {}
                },
                Param::Varargs(_, Type::Unpack(unpacked)) => {
                    // If we have a TypeVarTuple *args with no matched arguments, resolve it to empty tuple
                    self.is_subset_eq(unpacked, &self.heap.mk_concrete_tuple(Vec::new()));
                }
                Param::Varargs(..) => {}
                Param::Pos(name, ty, required) | Param::KwOnly(name, ty, required) => {
                    kwparams.insert(name, (ty, NameOrigin::Param, required));
                }
                Param::Kwargs(name, Type::Unpack(f)) if let Type::TypedDict(typed_dict) = &**f => {
                    self.typed_dict_fields(typed_dict).into_iter().for_each(
                        |(field_name, field)| {
                            kwparams.insert(
                                name_owner.push(field_name),
                                (
                                    type_owner.push(field.ty),
                                    NameOrigin::UnpackedKwargs(name.as_ref()),
                                    if field.required {
                                        &Required::Required
                                    } else {
                                        &Required::Optional(None)
                                    },
                                ),
                            );
                        },
                    );
                    if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(typed_dict) {
                        kwargs = Some((name.as_ref(), Some(type_owner.push(extra.ty))))
                    } else {
                        kwargs = Some((name.as_ref(), None))
                    }
                }
                Param::Kwargs(name, ty) => {
                    kwargs = Some((name.as_ref(), Some(ty)));
                }
            }
        }
        let mut unexpected_keyword_error = |name: &Name, range| {
            if missing_named_posonly.shift_remove(name) {
                error(
                    call_errors,
                    range,
                    ErrorKind::UnexpectedKeyword,
                    format!("Expected argument `{name}` to be positional"),
                );
            } else {
                error(
                    call_errors,
                    range,
                    ErrorKind::UnexpectedKeyword,
                    format!("Unexpected keyword argument `{name}`"),
                );
            }
        };
        let mut splat_kwargs = Vec::new();
        for kw in keywords {
            match kw.arg {
                None => {
                    let ty = kw.value.infer(self, arg_errors);
                    if let Type::TypedDict(typed_dict) = ty {
                        for (name, field) in self.typed_dict_fields(&typed_dict).into_iter() {
                            let name = name_owner.push(name);
                            let mut hint = kwargs.as_ref().and_then(|(_, ty)| *ty);
                            if let Some((ty, _, definitely_seen)) = seen_names.get(name) {
                                // For Required fields, the conflict is guaranteed, so report
                                // BadKeywordArgument. For NotRequired fields, the conflict is
                                // only potential (field may be absent at runtime), so report
                                // PotentialBadKeywordArgument instead. This allows users to
                                // opt-in to the stricter check while avoiding false positives
                                // in basic mode.
                                let error_kind = if field.required && *definitely_seen {
                                    ErrorKind::BadKeywordArgument
                                } else {
                                    ErrorKind::PotentialBadKeywordArgument
                                };
                                error(
                                    call_errors,
                                    kw.range,
                                    error_kind,
                                    format!("Multiple values for argument `{name}`"),
                                );
                                hint = Some(*ty);
                            } else if let Some((ty, origin, _)) = kwparams.get(name) {
                                seen_names.insert(name, (*ty, origin.clone(), field.required));
                                hint = Some(*ty)
                            } else if kwargs.is_none() {
                                unexpected_keyword_error(name, kw.range);
                            }
                            if let Some(want) = &hint {
                                self.check_type_with_options(
                                    &field.ty,
                                    want,
                                    kw.range,
                                    TypeCheckOptions::new(call_errors, &|| {
                                        TypeCheckContext::of_kind(TypeCheckKind::CallArgument(
                                            Some(name.clone()),
                                            callable_name.cloned(),
                                        ))
                                        .with_context(context.map(|ctx| ctx()))
                                    })
                                    .with_call_context(call_context),
                                );
                            }
                        }
                    } else {
                        match self.unwrap_mapping(&ty) {
                            Some((key, value)) => {
                                if self.is_subset_eq(
                                    &key,
                                    &self.heap.mk_class_type(self.stdlib.str().clone()),
                                ) {
                                    if let Some((name, Some(want))) = kwargs.as_ref() {
                                        self.check_type_with_options(
                                            &value,
                                            want,
                                            kw.range,
                                            TypeCheckOptions::new(call_errors, &|| {
                                                TypeCheckContext::of_kind(
                                                    TypeCheckKind::CallKwArgs(
                                                        None,
                                                        name.cloned(),
                                                        callable_name.cloned(),
                                                    ),
                                                )
                                                .with_context(context.map(|ctx| ctx()))
                                            })
                                            .with_call_context(call_context),
                                        );
                                    };
                                    splat_kwargs.push((value, kw.range));
                                } else {
                                    error(
                                        call_errors,
                                        kw.value.range(),
                                        ErrorKind::BadUnpacking,
                                        format!(
                                            "Expected argument after ** to have `str` keys, got: {}",
                                            self.for_display(key)
                                        ),
                                    );
                                }
                            }
                            None => {
                                error(
                                    call_errors,
                                    kw.value.range(),
                                    ErrorKind::BadUnpacking,
                                    format!(
                                        "Expected argument after ** to be a mapping, got: {}",
                                        self.for_display(ty)
                                    ),
                                );
                            }
                        }
                    }
                }
                Some(id) => {
                    let mut hint = kwargs.as_ref().and_then(|(name, ty)| {
                        ty.map(|ty| (NameOrigin::UnpackedKwargs(*name), ty))
                    });
                    let mut has_matching_param = false;
                    if let Some((ty, origin, definitely_seen)) = seen_names.get(&id.id) {
                        // Use PotentialBadKeywordArgument when the prior entry came from a
                        // NotRequired TypedDict field — the conflict is only potential.
                        let error_kind = if !*definitely_seen {
                            ErrorKind::PotentialBadKeywordArgument
                        } else {
                            ErrorKind::BadKeywordArgument
                        };
                        error(
                            call_errors,
                            kw.range,
                            error_kind,
                            format!("Multiple values for argument `{}`", id.id),
                        );
                        hint = Some((origin.clone(), *ty));
                        has_matching_param = true;
                    } else if let Some((ty, origin, req)) = kwparams.get(&id.id) {
                        seen_names
                            .insert(&id.id, (*ty, origin.clone(), **req == Required::Required));
                        hint = Some((origin.clone(), *ty));
                        has_matching_param = true;
                    } else if kwargs.is_none_or(|(_, ty)| ty.is_none()) {
                        unexpected_keyword_error(&id.id, id.range);
                    }
                    if let Some((origin, expected)) = &hint {
                        let name = match origin {
                            NameOrigin::Param => Some(id.id.clone()),
                            NameOrigin::UnpackedKwargs(kwargs_name) => kwargs_name.cloned(),
                        };
                        argmap.insert(kw.range, (*expected).clone(), name);
                    }
                    let unhinted_arg_ty = bound_args
                        .as_ref()
                        .map(|_| kw.value.infer(self, arg_errors));
                    let tcc: &dyn Fn() -> TypeCheckContext = &|| {
                        TypeCheckContext::of_kind(if has_matching_param {
                            TypeCheckKind::CallArgument(Some(id.id.clone()), callable_name.cloned())
                        } else {
                            TypeCheckKind::CallKwArgs(
                                Some(id.id.clone()),
                                kwargs.as_ref().and_then(|(name, _)| name.cloned()),
                                callable_name.cloned(),
                            )
                        })
                        .with_context(context.map(|ctx| ctx()))
                    };
                    let arg_ty = match kw.value {
                        TypeOrExpr::Expr(x) => self
                            .expr_with_options(
                                x,
                                match hint {
                                    Some((_, ty)) => ExprOptions::check(
                                        ty,
                                        arg_errors,
                                        call_errors,
                                        tcc,
                                        Some(call_context),
                                    ),
                                    None => ExprOptions::infer(arg_errors, None),
                                },
                            )
                            .into_ty(),
                        TypeOrExpr::Type(x, range) => {
                            if let Some((_, hint)) = &hint
                                && !hint.is_any()
                            {
                                self.check_type_with_options(
                                    x,
                                    hint,
                                    range,
                                    TypeCheckOptions::new(call_errors, tcc)
                                        .with_call_context(call_context),
                                );
                            }
                            (*x).clone()
                        }
                    };
                    record(bound_args, &id.id, unhinted_arg_ty.unwrap_or(arg_ty));
                }
            }
        }
        if missing_unnamed_posonly > 0 || !missing_named_posonly.is_empty() {
            let range = keywords.first().map_or(arguments_range, |kw| kw.range);
            let msg = if missing_unnamed_posonly == 0 {
                format!(
                    "Missing {} {}",
                    pluralize(missing_named_posonly.len(), "positional argument"),
                    missing_named_posonly
                        .iter()
                        .map(|name| format!("`{name}`"))
                        .join(", "),
                )
            } else {
                format!(
                    "Expected {}",
                    count(
                        missing_unnamed_posonly + missing_named_posonly.len(),
                        "more positional argument"
                    ),
                )
            };
            error(call_errors, range, ErrorKind::BadArgumentCount, msg);
        }
        let missing_self_param = self_arg.is_some() && num_positional_params == 0;
        // We'll attempt to match extra positional arguments to kw-only parameters for better error messages.
        let mut extra_posargs_iter = extra_positional_args.iter();
        if missing_self_param {
            // The first extra arg is `self`, so it shouldn't be matched to a kw-only parameter.
            extra_posargs_iter.next();
        }
        let mut extra_posargs_matched = 0;
        for (name, (want, origin, required)) in kwparams.iter() {
            if !seen_names.contains_key(name) {
                match required {
                    Required::Required => {
                        if splat_kwargs.is_empty() {
                            if let Some(arg_range) = extra_posargs_iter.next() {
                                error(
                                    call_errors,
                                    *arg_range,
                                    ErrorKind::UnexpectedPositionalArgument,
                                    format!("Expected argument `{name}` to be passed by name"),
                                );
                                extra_posargs_matched += 1;
                            } else {
                                argmap.unmatched_params.insert(match origin {
                                    NameOrigin::Param => Some((*name).clone()),
                                    NameOrigin::UnpackedKwargs(name) => name.cloned(),
                                });
                                error(
                                    call_errors,
                                    arguments_range,
                                    ErrorKind::MissingArgument,
                                    format!("Missing argument `{name}`"),
                                );
                            }
                        }
                    }
                    Required::Optional(Some(default)) => {
                        default_check.push((Some(name), want, default))
                    }
                    Required::Optional(None) => {}
                }
                for (ty, range) in &splat_kwargs {
                    self.check_type_with_options(
                        ty,
                        want,
                        *range,
                        TypeCheckOptions::new(call_errors, &|| {
                            TypeCheckContext::of_kind(TypeCheckKind::CallUnpackKwArg(
                                (*name).clone(),
                                callable_name.cloned(),
                            ))
                            .with_context(context.map(|ctx| ctx()))
                        })
                        .with_call_context(call_context),
                    );
                }
            }
        }
        for (name, ty, default) in default_check {
            // `ty` may contain type variables, so we record quantified bounds from the default and
            // check for inconsistent solutions. We mark any literals in the default as implicit so
            // that type variables get solved to promoted types (`int` rather than `Literal[N]`).
            let default_ty = default.ty.clone().with_literal_style(LitStyle::Implicit);
            self.check_type_with_options(
                &default_ty,
                ty,
                arguments_range,
                TypeCheckOptions::new(call_errors, &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::CallArgument(
                        name.cloned(),
                        callable_name.cloned(),
                    ))
                })
                .with_call_context(call_context),
            );
        }
        let num_extra_positional_args = extra_positional_args.len();
        if let Some(arg_range) = extra_arg_pos
            // This error is redundant if we've already reported an error for every individual arg.
            && extra_posargs_matched < num_extra_positional_args
        {
            let (expected, actual) = if missing_self_param {
                (
                    "0 positional arguments".to_owned(),
                    format!("{num_extra_positional_args} (including implicit `self`)"),
                )
            } else {
                let num_positional_params = num_positional_params - (self_arg.is_some() as usize);
                (
                    count(num_positional_params, "positional argument"),
                    (num_positional_params + num_extra_positional_args).to_string(),
                )
            };
            error(
                call_errors,
                arg_range,
                ErrorKind::BadArgumentCount,
                format!("Expected {expected}, got {actual}"),
            );
        }
        argmap
    }

    /// Helper used by `callable_infer` and Expr::Lambda inference to distribute over hints.
    pub fn callable_infer_with_hint<R>(
        &self,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
        mut inner: impl FnMut(Option<&Type>, &ErrorCollector) -> R,
        result_type: impl Fn(&R) -> &Type,
    ) -> R {
        let owner = Owner::new();
        let hint = match hint {
            // Optimization: no-hint and single-hint cases can return immediately.
            None => return inner(None, errors),
            Some(hint) if hint.types().len() == 1 => return inner(hint.types().first(), errors),
            // Optimization: discard overly wide hints.
            Some(hint) if hint.types().len() > MAX_CALL_HINT_WIDTH => return inner(None, errors),
            Some(hint) => hint,
        };
        let mut hints = hint.types().map(Some);
        // Push a marker so we know when no individual hint has matched. We'll try a combined union hint.
        // Constructing the union is expensive, so we use the marker to avoid unnecessary construction.
        hints.push(None);
        let mut ret_with_error = None;
        for mut cur_hint in hints {
            if cur_hint.is_none() {
                let combined_hint = Type::union(hint.types().to_vec());
                cur_hint = Some(owner.push(combined_hint));
            }
            let cur_errors = self.error_collector();
            let ret = inner(cur_hint, &cur_errors);
            if cur_errors.is_empty()
                && cur_hint.is_none_or(|hint| {
                    let snapshot = self
                        .solver()
                        .snapshot_vars(&hint.collect_maybe_placeholder_vars());
                    let res = self.is_subset_eq(result_type(&ret), hint);
                    self.solver().restore_vars(snapshot);
                    res
                })
            {
                return ret;
            } else if ret_with_error.is_none() {
                ret_with_error = Some((ret, cur_errors));
            }
        }
        let (ret, cur_errors) = ret_with_error.unwrap();
        errors.extend(cur_errors);
        ret
    }

    // Call a function with the given arguments. The arguments are contextually typed, if possible.
    // We pass two error collectors into this function and return specialization errors separately:
    // * arg_errors is used to infer the types of arguments, before passing them to the function.
    // * call_errors is used for (1) call signature matching, e.g. arity issues and (2) checking the
    //   types of arguments against the types of parameters.
    // * We often use call_errors to check whether a call succeeded, which specialization errors
    //   should not affect, so we return them separately. The caller must add them to the appropriate
    //   error collector.
    // Callers can pass the same error collector for both, and most callers do. We use two collectors
    // for overload matching.
    //
    // Returns: (return_type, specialization_errors, argmap) where argmap maps each
    // argument's source range to the parameter it was matched against.
    pub fn callable_infer(
        &self,
        callable: Callable,
        callable_name: Option<&FunctionKind>,
        shape_transform: Option<&ShapeTransform>,
        tparams: Option<&TParams>,
        self_obj: Option<Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
        mut ctor_targs: Option<&mut TArgs>,
    ) -> (Type, Vec<TypeVarSpecializationError>, ArgMap) {
        self.callable_infer_with_hint(
            hint,
            call_errors,
            |cur_hint, cur_call_errors| {
                self.callable_infer_inner(
                    callable.clone(),
                    callable_name,
                    shape_transform,
                    tparams,
                    self_obj.clone(),
                    args,
                    keywords,
                    arguments_range,
                    arg_errors,
                    cur_call_errors,
                    context,
                    cur_hint,
                    &mut ctor_targs,
                )
            },
            |ret| &ret.0,
        )
    }

    fn callable_infer_inner(
        &self,
        callable: Callable,
        callable_name: Option<&FunctionKind>,
        shape_transform: Option<&ShapeTransform>,
        tparams: Option<&TParams>,
        mut self_obj: Option<Type>,
        mut args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<&Type>,
        ctor_targs: &mut Option<&mut TArgs>,
    ) -> (Type, Vec<TypeVarSpecializationError>, ArgMap) {
        let call_context = CallContext::outside()
            .with_argument_side(ArgumentSide::Got)
            .require_boundary_consumption();

        let shape_transform_func = shape_transform.map(|t| t.to_meta_shape_function());
        let meta_shape_func: Option<&dyn MetaShapeFunction> = shape_transform_func.as_deref();
        let mut bound_args: Option<HashMap<String, Type>> = meta_shape_func.map(|_| HashMap::new());

        let (callable_qs, mut callable) = if let Some(tparams) = tparams {
            // If we have a hint, we want to try to instantiate against it first, so we can contextually type
            // arguments. If we don't match the hint, we need to throw away any instantiations we might have made.
            // By invariant, hint will be None if we are calling a constructor.
            if let Some(hint) = hint {
                let (qs, callable_) = self.instantiate_fresh_callable(tparams, callable.clone());
                if self.is_subset_eq(&callable_.ret, hint)
                    && !self.solver().has_instantiation_errors(&qs)
                {
                    (qs, callable_)
                } else {
                    // Even though these quantifieds aren't used, let's make sure to not leave
                    // unfinished quantifieds around.
                    let _ = self.finish_quantified(qs, false);
                    self.instantiate_fresh_callable(tparams, callable)
                }
            } else {
                self.instantiate_fresh_callable(tparams, callable)
            }
        } else {
            (QuantifiedHandle::empty(), callable)
        };
        call_context.register_fresh_quantified_vars(callable_qs.vars());
        let (self_qs, remaining_callable_qs) = if self_obj.is_some()
            && let Some(first_param) = callable.get_first_param()
            // TODO(https://github.com/facebook/pyrefly/issues/105): handle nested vars
            && matches!(first_param, Type::Var(_))
        {
            // Quantifieds in `self` need to be finished as soon as `self_arg` is matched, unlike
            // other quantifieds that are finished at the end of the call, so we split them out to
            // be handled separately.
            let (self_qs, remaining_qs) = callable_qs.partition_by(&first_param);
            (Some(self_qs), remaining_qs)
        } else {
            (None, callable_qs)
        };
        let ctor_qs = if let Some(targs) = ctor_targs.as_mut() {
            let qs = self.solver().freshen_class_targs(targs, self.uniques);
            call_context.register_fresh_quantified_vars(qs.vars());
            let mp = targs.substitution_map();
            callable.params.visit_mut(&mut |t| t.subst_mut(&mp));
            if let Some(obj) = self_obj.as_mut() {
                obj.subst_mut(&mp);
            } else if let Some(id) = callable_name
                && id.function_name().as_ref() == &dunder::NEW
                && let Some((first, rest)) = args.split_first()
                && let CallArg::Arg(TypeOrExpr::Type(obj, _)) = first
            {
                // hack: we inserted a class type into the args list, but we need to substitute it
                self_obj = Some((*obj).clone().subst(&mp));
                args = rest;
            }
            qs
        } else {
            QuantifiedHandle::empty()
        };
        let self_arg = self_obj.as_ref().map(|ty| CallArg::ty(ty, arguments_range));
        let argmap = match callable.params {
            Params::List(params) | Params::Partial(params) => self.callable_infer_params(
                callable_name,
                &params,
                None,
                self_arg,
                self_qs,
                args,
                keywords,
                arguments_range,
                arg_errors,
                call_errors,
                context,
                &call_context,
                &mut bound_args,
            ),
            Params::Ellipsis | Params::Materialization => {
                // Deal with Callable[..., R]
                for arg in self_arg.iter().chain(args.iter()) {
                    arg.pre_eval(self, arg_errors).post_infer(self, arg_errors)
                }
                ArgMap::new()
            }
            Params::ParamSpec(concatenate, p) => {
                let p = self.solver().expand(p);
                match p {
                    Type::ParamSpecValue(params) => self.callable_infer_params(
                        callable_name,
                        &params.prepend_types(&concatenate),
                        None,
                        self_arg,
                        self_qs,
                        args,
                        keywords,
                        arguments_range,
                        arg_errors,
                        call_errors,
                        context,
                        &call_context,
                        &mut bound_args,
                    ),
                    // This can happen with a signature like `(f: Callable[P, None], *args: P.args, **kwargs: P.kwargs)`.
                    // Before we match an argument to `f`, we don't know what `P` is, so we don't have an answer for the Var yet.
                    // Use to_subset_param to preserve Pos vs PosOnly: prefix params from a
                    // function definition should remain keyword-passable in direct calls.
                    Type::Var(var) => self.callable_infer_params(
                        callable_name,
                        &ParamList::new(
                            concatenate
                                .iter()
                                .map(|p| p.to_param_preserve_name())
                                .collect(),
                        ),
                        Some(var),
                        self_arg,
                        self_qs,
                        args,
                        keywords,
                        arguments_range,
                        arg_errors,
                        call_errors,
                        context,
                        &call_context,
                        &mut bound_args,
                    ),
                    Type::Quantified(q) => {
                        if let Some((args, keywords)) = self.paramspec_forwarding(
                            &q,
                            None,
                            args,
                            keywords,
                            arguments_range,
                            arg_errors,
                            call_errors,
                            context,
                        ) {
                            self.callable_infer_params(
                                callable_name,
                                &ParamList::new_types(concatenate.into_vec()),
                                None,
                                self_arg,
                                self_qs,
                                args,
                                keywords,
                                arguments_range,
                                arg_errors,
                                call_errors,
                                context,
                                &call_context,
                                &mut bound_args,
                            )
                        } else {
                            ArgMap::new()
                        }
                    }
                    Type::Any(_) | Type::Ellipsis => ArgMap::new(),
                    _ => {
                        // This could well be our error, but not really sure
                        self.error_with_context(
                            call_errors,
                            arguments_range,
                            ErrorKind::InvalidParamSpec,
                            format!("Unexpected ParamSpec type: `{}`", self.for_display(p)),
                            context,
                        );
                        ArgMap::new()
                    }
                }
            }
        };
        if let Some(targs) = ctor_targs {
            let residual_vars = call_context.captured_vars();
            self.solver().generalize_class_targs(targs, &residual_vars);
        }
        let mut errors = self
            .solver()
            .finish_quantified(
                remaining_callable_qs,
                self.solver().infer_with_first_use,
                self.type_order(),
                Some(&call_context),
            )
            .map_or_else(|e| e.to_vec(), |_| Vec::new());
        if let Err(e) = self.finish_quantified(ctor_qs, self.solver().infer_with_first_use) {
            errors.extend(e);
        }

        // Apply meta-shape inference if bound args were collected
        let ret = if let Some(meta_shape_func) = meta_shape_func
            && let Some(mut bound) = bound_args
        {
            // For bound method calls, ensure `self` is in bound_args so that
            // inject_module_attrs can resolve module fields (e.g., start_dim, end_dim).
            // The self param may not be recorded by callable_infer_params if it's
            // positional-only without a name.
            if let Some(ref obj) = self_obj {
                let obj = self
                    .reproject_tuple_carrier_shape(self.canonicalize_shape_dsl_type(obj.clone()));
                bound.entry("self".to_owned()).or_insert(obj);
            }
            // Auto-inject module field values for DSL params not in bound_args.
            // When a DSL function expects params like `start_dim` that aren't method
            // parameters but are fields on `self`, resolve them from the module instance.
            self.inject_module_attrs(&mut bound, meta_shape_func, arguments_range);

            self.apply_meta_shape(
                callable.ret.clone(),
                meta_shape_func,
                &bound,
                arguments_range,
                arg_errors,
            )
        } else {
            callable.ret.clone()
        };

        (
            self.reproject_tuple_carrier_shape(self.solver().for_return_boundary(ret)),
            errors,
            argmap,
        )
    }

    /// After a call's return type is resolved, re-project the shape of registered
    /// shaped arrays from their (now-substituted) base-class shape argument.
    ///
    /// Generic returns like `Array[S, float]` are stored shapeless at annotation
    /// time because the shape argument `S` carries no per-dimension information.
    /// Once `S` is bound to a concrete shape by call inference, the base-class
    /// argument projects to a real shape, so we re-read it here.
    ///
    /// Scoped to TypeVar-mode shape parameters; TypeVarTuple
    /// shapes are parsed into the shape field directly and need no reprojection.
    fn reproject_tuple_carrier_shape(&self, ty: Type) -> Type {
        ty.transform(&mut |ty| {
            let shaped_array = match ty {
                Type::ClassType(cls) => self.try_reproject_class_type_to_shaped_array(cls),
                Type::ShapedArray(shaped_array) => {
                    self.try_reproject_class_type_to_shaped_array(&shaped_array.base_class)
                }
                _ => return,
            };
            if let Some(shaped_array) = shaped_array {
                *ty = shaped_array.to_type();
            }
        })
    }

    /// Auto-inject module field values into `bound_args` for DSL parameters
    /// that aren't method parameters but match fields on `self`.
    ///
    /// This enables DSL functions for module methods (e.g., `nn.Flatten.forward`)
    /// to access constructor-captured values (e.g., `start_dim`, `end_dim`) without
    /// extending the DSL grammar. The DSL function declares them as regular parameters
    /// with defaults, and this method resolves them from `self`'s class fields.
    ///
    /// For fields typed as `Int[T]`, unwraps to `T` so the DSL's `extract_dsl_val`
    /// can handle them as plain literal ints.
    fn inject_module_attrs(
        &self,
        bound_args: &mut HashMap<String, Type>,
        meta_shape_func: &dyn MetaShapeFunction,
        _range: TextRange,
    ) {
        // For NNModule instances, inject captured fields directly into bound_args.
        // The NNModule's fields already contain plain Type values from the constructor,
        // so no Int[T] unwrapping is needed.
        if let Some(Type::NNModule(module)) = bound_args.get("self") {
            let module = module.clone();
            for param_name in meta_shape_func.param_names() {
                if param_name == "self" || bound_args.contains_key(param_name) {
                    continue;
                }
                let name = Name::new(param_name);
                if let Some(ty) = module.fields.get(&name) {
                    bound_args.insert(param_name.to_owned(), ty.clone());
                }
            }
            return;
        }

        let cls = match bound_args.get("self") {
            Some(Type::ClassType(cls)) => cls.clone(),
            _ => return,
        };

        for param_name in meta_shape_func.param_names() {
            if param_name == "self" || bound_args.contains_key(param_name) {
                continue;
            }

            // Look up the field directly on the class, avoiding error reporting.
            let attr_name = Name::new(param_name);
            let field = match self.get_field_from_current_class_only(cls.class_object(), &attr_name)
            {
                Some(f) => f,
                None => continue,
            };

            // Substitute type parameters (e.g., _Dim[S] with S=1 → Int[1]).
            let field_ty = cls.targs().substitution().substitute_into(field.ty());

            // For optional dimension fields, if the dimension is unbound (Any), resolve to None;
            // the DSL models missing values as None.
            // TODO(stroxler): It is unresolved whether returning the first `Size` member (rather
            // than `None` for an unbound/Any dimension) still preserves the old "unbound dimension
            // resolves to None" semantics, now that an unbound `Dim[Any]` lowers to a gradual `Size`
            // and a union may hold both a concrete `Size` and `Any`. This feature is corpus-guided,
            // so there is no data on the intended behavior until a real case surfaces.
            let unwrapped = match field_ty {
                Type::Union(ref u) => {
                    let size = u.members.iter().find_map(|m| match m {
                        Type::Int(_) => Some(m.clone()),
                        _ => None,
                    });
                    match size {
                        Some(size) => size,
                        None if u.members.iter().any(Type::is_any) => Type::None,
                        None => field_ty,
                    }
                }
                other => other,
            };
            bound_args.insert(param_name.to_owned(), unwrapped);
        }
    }

    /// Apply a meta-shape function using pre-bound arguments.
    fn apply_meta_shape(
        &self,
        ret_type: Type,
        meta_shape_func: &dyn MetaShapeFunction,
        bound_args: &HashMap<String, Type>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let ret_type = self.canonicalize_shape_dsl_type(ret_type);
        match meta_shape_func.evaluate(bound_args, &ret_type) {
            // The meta-shape evaluator (in `pyrefly_types`, which has no metadata
            // knowledge) rebuilds the result `ShapedArray` with the computed shape
            // but leaves the base-class tuple carrier stale. Re-sync each result
            // carrier to its projected shape so tuple-carrier `.shape` stays
            // coherent.
            Some(Ok(ty)) => ty.transform(&mut |ty| {
                if let Type::ShapedArray(shaped_array) = ty {
                    *ty = self
                        .shaped_array_with_shape(shaped_array, shaped_array.shape())
                        .to_type();
                }
            }),
            Some(Err(shape_error)) => {
                errors
                    .error_builder(
                        range,
                        ErrorKind::InvalidArgument,
                        format!("{}", shape_error),
                    )
                    .emit();
                ret_type
            }
            None => ret_type,
        }
    }

    fn canonicalize_shape_dsl_type(&self, ty: Type) -> Type {
        ty.transform(&mut |ty| {
            if let Type::ClassType(cls) = ty
                && self.shaped_array_shape_for_class_type(cls).is_some()
            {
                *ty = self
                    .shaped_array_classtype_to_shaped_array_type(cls)
                    .to_type();
            }
        })
    }
}
