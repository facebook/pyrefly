/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type-checks `functools.partial(...)` bound arguments and synthesizes the remaining signature.

use itertools::Itertools;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::types::Var;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallTargetLookup;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::unwrap::HintRef;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::callable::Callable;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::PrefixParam;
use crate::types::callable::Required;
use crate::types::function::Function;
use crate::types::types::Overload;
use crate::types::types::OverloadType;
use crate::types::types::Type;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Handle a `functools.partial(func, ...)` call, synthesizing the remaining signature instead of
    /// deferring to the typeshed stub.
    pub fn call_functools_partial(
        &self,
        partial_ty: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let Type::ClassDef(partial_cls) = partial_ty else {
            unreachable!("call_functools_partial dispatched on a non-class callee");
        };
        let Some(CallArg::Arg(target)) = args.first() else {
            // No inspectable callable target (e.g. a `*args` splat); defer to the stub.
            return self
                .freeform_call_infer(
                    partial_ty.clone(),
                    args,
                    kws,
                    callee_range,
                    arg_range,
                    hint,
                    errors,
                )
                .ty;
        };
        let target_ty = target.infer(self, errors);
        // A class object / `type[C]` is callable via its constructor; normalize to that signature
        // so the same argument checking and partial construction apply, with the instance as the return.
        let target_ty = match target_ty {
            // A bare protocol or abstract class can't be instantiated, so flag it at construction
            // where the problem originates (a `type[C]` value below can still be a concrete
            // subclass). Mirror the direct-instantiation path in `call.rs`.
            Type::ClassDef(cls) => {
                let metadata = self.get_metadata_for_class(&cls);
                if metadata.is_protocol() {
                    self.error(
                        errors,
                        callee_range,
                        ErrorKind::BadInstantiation,
                        format!(
                            "Cannot instantiate `{}` because it is a protocol",
                            cls.name()
                        ),
                    );
                } else {
                    let abstract_members = self.get_abstract_members_for_class(&cls);
                    let unimplemented = abstract_members.unimplemented_abstract_methods();
                    if !unimplemented.is_empty() {
                        self.error(
                            errors,
                            callee_range,
                            ErrorKind::BadInstantiation,
                            format!(
                                "Cannot instantiate `{}` because the following members are abstract: {}",
                                cls.name(),
                                unimplemented
                                    .iter()
                                    .map(|x| format!("`{x}`"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        );
                    } else if metadata.is_explicitly_abstract() {
                        self.error(
                            errors,
                            callee_range,
                            ErrorKind::DirectAbstractBaseInstantiation,
                            format!(
                                "Cannot instantiate `{}` because it directly extends `ABC` or uses `ABCMeta`",
                                cls.name()
                            ),
                        );
                    }
                }
                match self.promote_silently(&cls) {
                    Type::ClassType(instance) => self.constructor_to_callable(&instance),
                    _ => Type::ClassDef(cls),
                }
            }
            Type::Type(inner) => match *inner {
                Type::ClassType(instance) => self.constructor_to_callable(&instance),
                other => Type::Type(Box::new(other)),
            },
            // A callback-protocol instance normalizes to its `__call__` bound method (a
            // `Type::BoundMethod`), whose receiver the signature match below strips. A plain
            // instance with no `__call__` defers to the stub.
            Type::ClassType(instance) => match self.instance_as_dunder_call(&instance) {
                Some(dunder_call) => dunder_call,
                None => Type::ClassType(instance),
            },
            other => other,
        };
        // A union target with a non-callable member can never be a valid partial target for that
        // member, so flag it the way a direct call would; the stub still reports the whole union as
        // not assignable to `func`.
        if let Type::Union(union) = &target_ty {
            for member in &union.members {
                if matches!(
                    self.as_call_target(member.clone()),
                    CallTargetLookup::Error(..)
                ) {
                    self.error(
                        errors,
                        target.range(),
                        ErrorKind::NotCallable,
                        format!(
                            "Expected a callable, got `{}`",
                            self.for_display(member.clone())
                        ),
                    );
                }
            }
        }
        // Fall back to the stub, reusing the already-inferred target so it isn't inferred twice.
        let fallback = |me: &Self| {
            let mut args_with_ty = args.to_vec();
            args_with_ty[0] = CallArg::ty(&target_ty, target.range());
            let result = me.freeform_call_infer(
                partial_ty.clone(),
                &args_with_ty,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            );
            if matches!(&target_ty, Type::Overload(overload) if overload.signatures.iter().any(|x| matches!(x, OverloadType::Forall(_))))
            {
                me.specialize(
                    partial_cls,
                    vec![me.heap.mk_any_implicit()],
                    callee_range,
                    errors,
                )
            } else {
                result.ty
            }
        };
        // `partial(f)` with nothing bound is a pure forwarder; for an overloaded target hand the
        // overload back unchanged so ordinary overload resolution still applies at the call site (a
        // single remaining parameter list can't preserve overload branches).
        if args.len() == 1 && kws.is_empty() && matches!(target_ty, Type::Overload(_)) {
            return target_ty;
        }
        // The remaining parameters are keyed by the names the bound keywords consume. A `**` splat of an
        // `Unpack[TypedDict]` binds exactly the TypedDict's declared fields, so expand it to those
        // names; a splat of any other type can't be reduced structurally, so defer to the stub.
        let Some(bound_kw_names) = self.partial_bound_kw_names(kws) else {
            return fallback(self);
        };
        // Overloaded target with bound arguments: drop branches the bound arguments can't satisfy and
        // recombine the surviving partial callables into an overload, so per-call resolution still works.
        if let Type::Overload(overload) = &target_ty {
            let mut partials: Vec<Callable> = Vec::new();
            for ot in overload.signatures.iter() {
                let (branch_sig, quantified) = match ot {
                    OverloadType::Function(func) => (func.signature.clone(), None),
                    OverloadType::Forall(forall) => {
                        let (qs, sig) = self.instantiate_fresh_callable(
                            &forall.tparams,
                            forall.body.signature.clone(),
                        );
                        (sig, Some(qs))
                    }
                };
                // Trial-check the bound arguments against this branch; keep it only if they fit.
                // Optional params keep the still-unbound parameters from erroring as missing.
                let mut probe = branch_sig.clone();
                make_params_optional(&mut probe);
                let trial = self.error_collector();
                self.freeform_call_infer(
                    self.heap.mk_callable_from(probe),
                    &args[1..],
                    kws,
                    target.range(),
                    arg_range,
                    None,
                    &trial,
                );
                let generic = quantified.is_some();
                // Keep solutions learned from the bound arguments. Variables left unsolved in a
                // generic branch become gradual rather than escaping without a binder.
                let specialization_ok = quantified
                    .map(|qs| self.finish_quantified(qs, false).is_ok())
                    .unwrap_or(true);
                if !trial.is_empty() || !specialization_ok {
                    continue;
                }
                let branch_sig = if generic {
                    let mut ty = self.heap.mk_callable_from(branch_sig);
                    self.solver().expand_mut(&mut ty);
                    let Type::Callable(callable) = ty else {
                        unreachable!("built by mk_callable_from");
                    };
                    *callable
                } else {
                    branch_sig
                };
                // Defer the whole overload rather than silently drop a matched branch we can't
                // represent, which would break a call that only matched that branch.
                match partial_callable(&branch_sig, &args[1..], &bound_kw_names) {
                    Some(partial) => partials.push(partial),
                    None => return fallback(self),
                }
            }
            return match partials.len() {
                0 => fallback(self),
                1 => self.heap.mk_callable_from(partials.pop().unwrap()),
                _ => {
                    let branches = partials
                        .into_iter()
                        .map(|c| {
                            OverloadType::Function(Function {
                                signature: c,
                                metadata: (*overload.metadata).clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    Type::Overload(Overload {
                        signatures: Vec1::try_from_vec(branches).unwrap(),
                        metadata: overload.metadata.clone(),
                    })
                }
            };
        }
        // We handle a directly-typed function/callable and a generic (`Forall`-wrapped) function.
        // A generic target keeps its `tparams`: type variables the bound args don't pin stay symbolic
        // in the remaining signature and are finalized at the result boundary, so a partial over a
        // generic function (including decorator use) preserves its genericity instead of leaking them
        // through the stub. Class objects, bound methods, and unions defer.
        let Ok((sig, tparams)) = target_ty.toplevel_callable_signatures().exactly_one() else {
            return fallback(self);
        };
        let mut sig = if matches!(target_ty, Type::BoundMethod(_)) {
            // Strip the already-bound `self`/`cls` so the partial has only the remaining parameters;
            // bound-argument checking against `target_ty` still binds the receiver as usual.
            match sig.strip_first_param() {
                Some(stripped_sig) => stripped_sig,
                None => return fallback(self),
            }
        } else {
            sig.clone()
        };
        // Only plain type variables are re-scoped correctly; a `ParamSpec` or `TypeVarTuple` target
        // needs structural partial handling we don't do, so defer it to the stub.
        if let Some(tparams) = &tparams
            && !tparams.iter().all(|q| q.kind() == QuantifiedKind::TypeVar)
        {
            return fallback(self);
        }
        if !matches!(sig.params, Params::List(_)) {
            return fallback(self);
        }
        self.expand_unpack_kwargs(&mut sig);
        // Nominal `partial[ret]` fallback for when no precise signature can be built. For a generic target
        // erase the target's own type vars so they don't leak out of scope; otherwise defer to the stub.
        let nominal_partial = |me: &Self, ret: Type| -> Type {
            match &tparams {
                None => fallback(me),
                Some(tparams) => {
                    let Type::ClassDef(cls) = partial_ty else {
                        unreachable!("call_functools_partial dispatched on a non-class callee");
                    };
                    let mut ret = ret;
                    ret.subst_mut_fn(&mut |q| {
                        tparams
                            .iter()
                            .any(|tp| tp == q)
                            .then(|| q.as_gradual_type())
                    });
                    me.specialize(cls, vec![ret], callee_range, errors)
                }
            }
        };
        // Type-check the bound arguments; making every parameter optional means binding only a prefix
        // doesn't report the remaining parameters as missing. For a generic target we instantiate the
        // type parameters as fresh vars and check against those, so a bound argument can *solve* a
        // typevar (e.g. pin it to an enclosing-scope typevar); the partial is then built from the
        // solved signature, and only the typevars the bound args left unsolved are restored as
        // quantifieds marked for finalization at the result boundary.
        let (sig, generic_target) = match &tparams {
            None => {
                let mut callee = target_ty.clone();
                callee.transform_toplevel_callable_signatures(|c: &mut Callable, _| {
                    self.expand_unpack_kwargs(c);
                    make_params_optional(c);
                });
                self.freeform_call_infer(
                    callee,
                    &args[1..],
                    kws,
                    target.range(),
                    arg_range,
                    None,
                    errors,
                );
                (sig, None)
            }
            Some(tparams) => {
                let (qs, inst) = self.instantiate_fresh_callable(tparams, sig);
                let var_to_q: SmallMap<Var, Quantified> = qs
                    .vars()
                    .iter()
                    .copied()
                    .zip(tparams.iter().cloned())
                    .collect();
                let mut callee = self.heap.mk_callable_from(inst.clone());
                callee.transform_toplevel_callable_signatures(|c: &mut Callable, _| {
                    self.expand_unpack_kwargs(c);
                    make_params_optional(c);
                });
                let outcome = self.freeform_call_infer(
                    callee,
                    &args[1..],
                    kws,
                    target.range(),
                    arg_range,
                    None,
                    errors,
                );
                // A typevar in a *required* remaining param stays symbolic so a later call arg can re-solve
                // it (as a direct call would); one only in optional params keeps its solved value (GH #3546).
                let mut regeneric_vars: SmallSet<Var> = SmallSet::new();
                if let Some(remaining) = partial_params(&inst, &args[1..], &bound_kw_names) {
                    for param in remaining.items() {
                        let (ty, required) = match param {
                            Param::PosOnly(_, t, r)
                            | Param::Pos(_, t, r)
                            | Param::KwOnly(_, t, r) => (t, r),
                            // `*args`/`**kwargs` carry no `Required` flag, but a later call can always
                            // pass more arguments through them, so their typevars must stay symbolic.
                            Param::Varargs(_, t) | Param::Kwargs(_, t) => (t, &Required::Required),
                        };
                        if !matches!(required, Required::Required) {
                            continue;
                        }
                        for v in ty.collect_all_vars() {
                            if var_to_q.contains_key(&v) {
                                regeneric_vars.insert(v);
                            }
                        }
                    }
                }
                // Finalize the vars `instantiate_fresh_callable` registered. Their substitution is
                // applied by `partial_solved_signature` below, and any specialization error was reported by
                // the bound-argument check via `freeform_call_infer`, so dropping the result is safe.
                let _ = self.finish_quantified(qs, false);
                let solved = self.partial_solved_signature(&inst, &var_to_q, &regeneric_vars);
                (
                    solved,
                    Some((outcome.overload_table, inst, var_to_q, regeneric_vars)),
                )
            }
        };
        let build = |sig: &Callable| {
            let remaining = partial_params(sig, &args[1..], &bound_kw_names)?;
            // A `TypeGuard`/`TypeIs` narrows only in a direct call; the partial just returns `bool`.
            let ret = match &sig.ret {
                Type::TypeGuard(_) | Type::TypeIs(_) => self.stdlib.bool().clone().to_type(),
                other => other.clone(),
            };
            let callable = Callable::partial(remaining, ret);
            // A free type parameter left in the partial is one the partial declares, which
            // finishing the type at this boundary does.
            Some(
                self.heap
                    .mk_callable_from(callable)
                    .finalize_exposed_free_quantifieds(),
            )
        };
        // The arguments can't be reduced to a precise signature (e.g. too many bound positionals); hand
        // back the nominal `partial[ret]` rather than re-running the stub over a `Forall`.
        let Some(result) = build(&sig) else {
            return nominal_partial(self, sig.ret);
        };
        // Rebuild the partial under each solution to preserve correlations between solved variables.
        if let Some((overload_table, inst, var_to_q, regeneric_vars)) = &generic_target {
            let per_row = self.solver().per_row(overload_table, || {
                build(&self.partial_solved_signature(inst, var_to_q, regeneric_vars))
                    .unwrap_or_else(|| result.clone())
            });
            return self.combine_overload_results(per_row, overload_table);
        }
        result
    }

    /// The target's signature with the bound arguments' solutions applied.
    ///
    /// A type parameter still standing in a required remaining parameter is one a later call can
    /// solve, so it is marked for finalization before expanding rather than being frozen to what
    /// the bound arguments happened to imply. `expand_with_bounds` then substitutes the ones a
    /// bound argument pinned and leaves the rest, which are marked likewise. Finishing instead of
    /// expanding would erase them all to `Any`.
    fn partial_solved_signature(
        &self,
        inst: &Callable,
        var_to_q: &SmallMap<Var, Quantified>,
        regeneric_vars: &SmallSet<Var>,
    ) -> Callable {
        let mut solved = self.heap.mk_callable_from(inst.clone());
        solved = solved.transform(&mut |t: &mut Type| {
            if let Type::Var(v) = t
                && regeneric_vars.contains(v)
                && let Some(q) = var_to_q.get(v)
            {
                *t = self.heap.mk_quantified(q.clone().with_needs_finalization());
            }
        });
        self.solver().expand_with_bounds(&mut solved);
        let solved = solved.transform(&mut |t: &mut Type| {
            if let Type::Var(v) = t
                && let Some(q) = var_to_q.get(v)
            {
                *t = self.heap.mk_quantified(q.clone().with_needs_finalization());
            }
        });
        match solved {
            Type::Callable(c) => *c,
            _ => unreachable!("built by mk_callable_from"),
        }
    }

    /// The parameter names the bound keyword arguments consume, used to build the partial. A named
    /// keyword contributes its own name; a `**` splat of an `Unpack[TypedDict]` contributes every
    /// field the TypedDict declares. Returns `None` for a splat of any other type, which can't be
    /// reduced to a fixed set of names.
    fn partial_bound_kw_names(&self, keywords: &[CallKeyword]) -> Option<Vec<Name>> {
        let mut names = Vec::new();
        for kw in keywords {
            match kw.arg {
                Some(id) => names.push(id.id.clone()),
                None => {
                    // The bound arguments are re-inferred and validated below; here we only want the
                    // splat's field names, so any inference error is reported there, not here.
                    let ty = kw.value.infer(self, &self.error_collector());
                    let Type::TypedDict(td) = ty else {
                        return None;
                    };
                    names.extend(
                        self.typed_dict_fields(&td)
                            .into_iter()
                            .map(|(name, _)| name),
                    );
                }
            }
        }
        Some(names)
    }

    /// Expand each `**kwargs: Unpack[TypedDict]` into one keyword-only param per field so the ordinary
    /// partial-signature logic handles them. An open TypedDict's extra items become a trailing `**kwargs`.
    fn expand_unpack_kwargs(&self, callable: &mut Callable) {
        let Params::List(params) = &mut callable.params else {
            return;
        };
        let mut expanded: Vec<Param> = Vec::with_capacity(params.items().len());
        for param in params.items() {
            match param {
                Param::Kwargs(_, ty) if let Some(td) = ty.unpacked_typed_dict() => {
                    for (name, ty, required) in self.typed_dict_kw_param_info(td) {
                        expanded.push(Param::KwOnly(name, ty, required));
                    }
                    if let ExtraItems::Extra(extra) = self.typed_dict_extra_items(td) {
                        expanded.push(Param::Kwargs(None, extra.ty));
                    }
                }
                _ => expanded.push(param.clone()),
            }
        }
        *params = ParamList::new(expanded);
    }
}

/// Make every parameter of a callable optional, so a `functools.partial` construction can bind a
/// prefix of the arguments without the remaining parameters being reported as missing.
fn make_params_optional(callable: &mut Callable) {
    match &mut callable.params {
        Params::List(params) => {
            for param in params.items_mut() {
                match param {
                    Param::PosOnly(_, _, r) | Param::Pos(_, _, r) | Param::KwOnly(_, _, r) => {
                        *r = Required::Optional(None)
                    }
                    Param::Varargs(..) | Param::Kwargs(..) => {}
                }
            }
        }
        // The trailing `ParamSpec` already absorbs extra arguments; only the prefix needs relaxing.
        Params::ParamSpec(prefix, _) => {
            for param in prefix.iter_mut() {
                match param {
                    PrefixParam::PosOnly(_, _, r) | PrefixParam::Pos(_, _, r) => {
                        *r = Required::Optional(None)
                    }
                }
            }
        }
        Params::Partial(_) | Params::Ellipsis | Params::Materialization => {}
    }
}

/// Partial `Callable` for one matched overload branch, or `None` if it can't be represented
/// structurally (the caller then defers instead of dropping the branch).
fn partial_callable(
    branch: &Callable,
    bound_args: &[CallArg],
    keyword_names: &[Name],
) -> Option<Callable> {
    match &branch.params {
        Params::List(_) => partial_params(branch, bound_args, keyword_names)
            .map(|params| Callable::partial(params, branch.ret.clone())),
        // `(...)` still accepts anything after binding a prefix.
        Params::Ellipsis => Some(Callable::ellipsis(branch.ret.clone())),
        // `Concatenate[..., P]` binds its prefix first; the partial keeps the unbound prefix and `P`.
        Params::ParamSpec(prefix, tail) => {
            partial_paramspec_prefix(prefix, bound_args, keyword_names).map(|prefix| Callable {
                params: Params::ParamSpec(prefix, tail.clone()),
                ret: branch.ret.clone(),
            })
        }
        Params::Partial(_) | Params::Materialization => None,
    }
}

/// Peel the bound `bound_args`/`keywords` off a `Concatenate` prefix. `None` if an argument would
/// fall through into the trailing `ParamSpec`, which can't be peeled structurally.
fn partial_paramspec_prefix(
    prefix: &[PrefixParam],
    bound_args: &[CallArg],
    keyword_names: &[Name],
) -> Option<Box<[PrefixParam]>> {
    let mut remaining = prefix.to_vec();
    for arg in bound_args {
        let CallArg::Arg(_) = arg else {
            return None;
        };
        if remaining.is_empty() {
            return None;
        }
        remaining.remove(0);
    }
    for name in keyword_names {
        let idx = remaining
            .iter()
            .position(|p| matches!(p, PrefixParam::Pos(n, ..) if n == name))?;
        remaining.remove(idx);
    }
    Some(remaining.into_boxed_slice())
}

/// Parameters remaining after binding arguments to `callable`. Returns `None` when the
/// target/arguments can't be reduced.
fn partial_params(
    callable: &Callable,
    bound_args: &[CallArg],
    keyword_names: &[Name],
) -> Option<ParamList> {
    let Params::List(params) = &callable.params else {
        return None;
    };
    let mut remaining = params.items().to_vec();
    for arg in bound_args {
        let CallArg::Arg(_) = arg else {
            return None;
        };
        let idx = remaining
            .iter()
            .position(|p| matches!(p, Param::PosOnly(..) | Param::Pos(..) | Param::Varargs(..)))?;
        if !matches!(remaining[idx], Param::Varargs(..)) {
            remaining.remove(idx);
        }
    }
    for name in keyword_names {
        let idx = remaining.iter().position(|p| {
            matches!(p, Param::Pos(n, ..) | Param::KwOnly(n, ..) if n == name)
                || matches!(p, Param::Kwargs(..))
        })?;
        if matches!(remaining[idx], Param::Kwargs(..)) {
            continue;
        }
        // A positional can't follow a bound keyword, so demote later positionals to keyword-only.
        if matches!(remaining[idx], Param::Pos(..)) {
            for later in remaining.iter_mut().skip(idx + 1) {
                if let Param::Pos(n, t, r) = later {
                    *later = Param::KwOnly(n.clone(), t.clone(), r.clone());
                }
            }
        }
        let (n, t) = match &remaining[idx] {
            Param::Pos(n, t, _) | Param::KwOnly(n, t, _) => (n.clone(), t.clone()),
            _ => unreachable!("matched a positional or keyword-only parameter"),
        };
        remaining[idx] = Param::KwOnly(n, t, Required::Optional(None));
    }
    Some(ParamList::new(remaining))
}
