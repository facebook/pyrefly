/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::max;
use std::collections::HashMap;

use itertools::Either;
use itertools::Itertools;
use pyrefly_types::callable::ArgCount;
use pyrefly_types::callable::ArgCounts;
use pyrefly_types::callable::Param;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Union;
use pyrefly_util::gas::Gas;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers::OverloadTrace;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::TargetWithTParams;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::callable::CallWithTypes;
use crate::alt::expr::TypeOrExpr;
use crate::alt::unwrap::HintRef;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::solver::solver::TypeVarSpecializationError;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Params;
use crate::types::literal::Lit;
use crate::types::types::Type;
use crate::types::types::Var;

struct CalledOverload<'f> {
    func: &'f TargetWithTParams<Function>,
    res: Type,
    ctor_targs: Option<TArgs>,
    call_errors: ErrorCollector,
    specialization_errors: Vec<TypeVarSpecializationError>,
    /// Maps each argument's source range to the parameter type it was matched against.
    expected_types: HashMap<TextRange, Type>,
}

impl CalledOverload<'_> {
    fn num_errors(&self) -> usize {
        self.call_errors.len() + self.specialization_errors.len()
    }
}

/// Performs argument type expansion for arguments to an overloaded function.
pub struct ArgsExpander<'a, Ans: LookupAnswer> {
    /// The index of the next argument to expand. Left is positional args; right, keyword args.
    idx: Either<usize, usize>,
    /// Current argument lists.
    arg_lists: Vec<(Vec<CallArg<'a>>, Vec<CallKeyword<'a>>)>,
    /// Hard-coded limit to how many times we'll expand.
    gas: Gas,
    solver: &'a AnswersSolver<'a, Ans>,
}

impl<'a, Ans: LookupAnswer> ArgsExpander<'a, Ans> {
    const GAS: usize = 100;

    pub fn new(
        posargs: Vec<CallArg<'a>>,
        keywords: Vec<CallKeyword<'a>>,
        solver: &'a AnswersSolver<'a, Ans>,
    ) -> Self {
        Self {
            idx: if posargs.is_empty() {
                Either::Right(0)
            } else {
                Either::Left(0)
            },
            arg_lists: vec![(posargs, keywords)],
            gas: Gas::new(Self::GAS as isize),
            solver,
        }
    }

    /// Expand the next argument and return the expanded argument lists.
    pub fn expand(
        &mut self,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
    ) -> Option<Vec<(Vec<CallArg<'a>>, Vec<CallKeyword<'a>>)>> {
        let idx = self.idx;
        let (posargs, keywords) = self.arg_lists.first()?;
        // Determine the value to try expanding, and also the idx of the value we will try next if needed.
        let value = match idx {
            Either::Left(i) => match &posargs[i] {
                CallArg::Arg(value) | CallArg::Star(value, ..) => {
                    self.idx = if i < posargs.len() - 1 {
                        Either::Left(i + 1)
                    } else {
                        Either::Right(0)
                    };
                    value
                }
            },
            Either::Right(i) if i < keywords.len() => {
                let CallKeyword { value, .. } = &keywords[i];
                self.idx = Either::Right(i + 1);
                value
            }
            Either::Right(_) => {
                return None;
            }
        };
        let expanded_types = self.expand_type(value.infer(self.solver, errors));
        if expanded_types.is_empty() {
            // Nothing to expand here, try the next argument.
            self.expand(errors, owner)
        } else {
            let expanded_types = expanded_types.into_map(|t| owner.push(t));
            let mut new_arg_lists = Vec::new();
            for (posargs, keywords) in self.arg_lists.iter() {
                for ty in expanded_types.iter() {
                    let mut new_posargs = posargs.clone();
                    let mut new_keywords = keywords.clone();
                    match idx {
                        Either::Left(i) => {
                            let new_value = TypeOrExpr::Type(ty, posargs[i].range());
                            new_posargs[i] = match posargs[i] {
                                CallArg::Arg(_) => CallArg::Arg(new_value),
                                CallArg::Star(_, range) => CallArg::Star(new_value, range),
                            }
                        }
                        Either::Right(i) => {
                            let new_value = TypeOrExpr::Type(ty, keywords[i].range());
                            new_keywords[i] = CallKeyword {
                                range: keywords[i].range(),
                                arg: keywords[i].arg,
                                value: new_value,
                            }
                        }
                    }
                    new_arg_lists.push((new_posargs, new_keywords));
                    if self.gas.stop() {
                        // We've hit our hard-coded limit; stop expanding, and move `idx` past the
                        // end of the keywords so that subsequent `expand` calls know we're done.
                        self.idx = Either::Right(keywords.len());
                        return None;
                    }
                }
            }
            self.arg_lists = new_arg_lists.clone();
            Some(new_arg_lists)
        }
    }

    /// Expands a type according to https://typing.python.org/en/latest/spec/overload.html#argument-type-expansion.
    fn expand_type(&self, ty: Type) -> Vec<Type> {
        match ty {
            Type::Union(box Union { members: ts, .. }) => ts,
            Type::ClassType(cls) if cls.is_builtin("bool") => {
                vec![
                    Lit::Bool(true).to_implicit_type(),
                    Lit::Bool(false).to_implicit_type(),
                ]
            }
            Type::ClassType(cls)
                if self
                    .solver
                    .get_metadata_for_class(cls.class_object())
                    .is_enum() =>
            {
                self.solver
                    .get_enum_members(cls.class_object())
                    .into_iter()
                    .map(Lit::to_implicit_type)
                    .collect()
            }
            Type::Type(box Type::Union(box Union { members: ts, .. })) => {
                ts.into_map(|t| self.solver.heap.mk_type_of(t))
            }
            Type::Tuple(Tuple::Concrete(elements)) => {
                let mut count: usize = 1;
                let mut changed = false;
                let mut element_expansions = Vec::new();
                for e in elements {
                    let element_expansion = self.expand_type(e.clone());
                    if element_expansion.is_empty() {
                        element_expansions.push(vec![e].into_iter());
                    } else {
                        let len = element_expansion.len();
                        count = count.saturating_mul(len);
                        if count > Self::GAS {
                            return Vec::new();
                        }
                        changed = true;
                        element_expansions.push(element_expansion.into_iter());
                    }
                }
                // Enforce a hard-coded limit on the number of expansions for perf reasons.
                if count <= Self::GAS && changed {
                    element_expansions
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|x| self.solver.heap.mk_concrete_tuple(x))
                        .collect()
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Calls an overloaded function, returning the return type and the closest matching overload signature.
    pub fn call_overloads(
        &self,
        overloads: Vec1<TargetWithTParams<Function>>,
        metadata: &FuncMetadata,
        self_obj: Option<Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
        // If we're constructing a class, its type arguments. A successful call will fill these in.
        ctor_targs: Option<&mut TArgs>,
    ) -> (Type, Callable) {
        // There may be Expr values in args and keywords.
        // If we infer them for each overload, we may end up inferring them multiple times.
        // If those overloads contain nested overloads, then we can easily end up with O(2^n) perf.
        // Therefore, flatten all TypeOrExpr's into Type before we start
        let call = CallWithTypes::new();
        let args = call.vec_call_arg(args, self, errors);
        let keywords = call.vec_call_keyword(keywords, self, errors);

        // Evaluate the call following https://typing.python.org/en/latest/spec/overload.html#overload-call-evaluation.

        // Step 1: eliminate overloads that accept an incompatible number of arguments.
        let mut arity_closest_overload = None;
        let arity_compatible_overloads = overloads
            .iter()
            .filter(|overload| {
                let arg_counts = overload.1.signature.arg_counts();
                let mismatch_size =
                    self.arity_mismatch_size(&arg_counts, self_obj.as_ref(), &args, &keywords);
                if arity_closest_overload
                    .as_ref()
                    .is_none_or(|(_, n)| *n > mismatch_size)
                {
                    arity_closest_overload = Some((*overload, mismatch_size));
                }
                mismatch_size == 0
            })
            .collect::<Vec<_>>();
        let (closest_overload, matched) = match Vec1::try_from_vec(arity_compatible_overloads) {
            Err(_) => (
                CalledOverload {
                    func: arity_closest_overload.unwrap().0,
                    res: self.heap.mk_any_error(),
                    ctor_targs: None,
                    call_errors: self.error_collector(),
                    specialization_errors: Vec::new(),
                    expected_types: HashMap::new(),
                },
                false,
            ),
            Ok(arity_compatible_overloads) => {
                // Step 2: evaluate each overload as a regular (non-overloaded) call.
                // Note: steps 4-6 are performed in `find_closest_overload`.
                let (mut closest_overload, mut matched) = self.find_closest_overload(
                    &arity_compatible_overloads,
                    metadata,
                    self_obj.as_ref(),
                    &args,
                    &keywords,
                    arguments_range,
                    errors,
                    hint,
                    &ctor_targs,
                );

                // Step 3: perform argument type expansion.
                let mut args_expander = ArgsExpander::new(args.clone(), keywords.clone(), self);
                let owner = Owner::new();
                'outer: while !matched && let Some(arg_lists) = args_expander.expand(errors, &owner)
                {
                    // Expand by one argument (for example, try splitting up union types), and try the call with each
                    // resulting arguments list.
                    // - If all expanded lists match, we union all return types together and declare a successful match
                    // - If any do not match, we move on to the next splittable argument (if we run out of args to split,
                    //   we'll wind up with a failed match and our best guess at the correct overload)
                    let mut matched_overloads = Vec::new();
                    for (cur_args, cur_keywords) in arg_lists.clone().iter() {
                        let (cur_closest, cur_matched) = self.find_closest_overload(
                            &arity_compatible_overloads,
                            metadata,
                            self_obj.as_ref(),
                            cur_args,
                            cur_keywords,
                            arguments_range,
                            errors,
                            hint,
                            &ctor_targs,
                        );
                        if !cur_matched {
                            continue 'outer;
                        }
                        matched_overloads.push(cur_closest);
                    }
                    if let Some(first_overload) = matched_overloads.first() {
                        let func = first_overload.func;
                        let ctor_targs = first_overload.ctor_targs.clone();
                        let expected_types = first_overload.expected_types.clone();
                        let specialization_errors = first_overload.specialization_errors.clone();
                        closest_overload = CalledOverload {
                            func,
                            ctor_targs,
                            expected_types,
                            res: self.unions(matched_overloads.into_map(|o| o.res)),
                            call_errors: self.error_collector(),
                            specialization_errors,
                        };
                        matched = true;
                        break;
                    }
                }
                (
                    closest_overload,
                    // If there was only one overload with the right arity, it definitely matched.
                    matched || arity_compatible_overloads.len() == 1,
                )
            }
        };

        if matched
            && let Some(targs) = ctor_targs
            && let Some(chosen_targs) = closest_overload.ctor_targs
        {
            *targs = chosen_targs;
        }
        // Record the closest overload to power IDE services.
        let mut overload_trace = |target: &TargetWithTParams<Function>| {
            let tparams = target
                .0
                .as_ref()
                .filter(|tparams| !tparams.is_empty())
                .cloned();
            OverloadTrace::new(target.1.signature.clone(), tparams)
        };
        let all_overload_traces = overloads.iter().map(&mut overload_trace).collect();
        let closest_overload_trace = overload_trace(closest_overload.func);
        self.record_overload_trace(
            arguments_range,
            all_overload_traces,
            closest_overload_trace,
            matched,
        );
        if matched {
            // If the selected overload is deprecated, we log a deprecation error.
            if let Some(deprecation) = &closest_overload.func.1.metadata.flags.deprecation {
                let msg = deprecation.as_error_message(format!(
                    "Call to deprecated overload `{}`",
                    closest_overload
                        .func
                        .1
                        .metadata
                        .kind
                        .format(self.module().name())
                ));
                errors.add(
                    arguments_range,
                    ErrorInfo::new(ErrorKind::Deprecated, context),
                    msg,
                );
            }
            errors.extend(closest_overload.call_errors);
            if let Ok(specialization_errors) =
                Vec1::try_from_vec(closest_overload.specialization_errors)
            {
                self.add_specialization_errors(
                    specialization_errors,
                    arguments_range,
                    errors,
                    None,
                );
            }
            (
                closest_overload.res,
                closest_overload.func.1.signature.clone(),
            )
        } else {
            // Build a string showing the argument types for error messages
            let mut arg_type_strs = Vec::new();
            for arg in &args {
                let (ty, prefix) = match arg {
                    CallArg::Arg(value) => (value.infer(self, errors), ""),
                    CallArg::Star(value, _) => (value.infer(self, errors), "*"),
                };
                let ty_display = self.for_display(ty);
                arg_type_strs.push(format!("{}{}", prefix, ty_display));
            }
            for kw in &keywords {
                let ty = kw.value.infer(self, errors);
                let ty_display = self.for_display(ty);
                if let Some(arg_name) = kw.arg {
                    arg_type_strs.push(format!("{}={}", arg_name.as_str(), ty_display));
                } else {
                    arg_type_strs.push(format!("**{}", ty_display));
                }
            }
            let args_display = format!("({})", arg_type_strs.join(", "));

            let mut msg = vec1![
                format!(
                    "No matching overload found for function `{}` called with arguments: {}",
                    metadata.kind.format(self.module().name()),
                    args_display
                ),
                "Possible overloads:".to_owned(),
            ];
            for overload in &overloads {
                let suffix = if overload.1.signature == closest_overload.func.1.signature {
                    " [closest match]"
                } else {
                    ""
                };
                let signature = match self_obj {
                    Some(_) => overload
                        .1
                        .signature
                        .split_first_param(&mut Owner::new())
                        .map(|(_, signature)| signature)
                        .unwrap_or_else(|| overload.1.signature.clone()),
                    None => overload.1.signature.clone(),
                };
                let signature = self
                    .solver()
                    .for_display(self.heap.mk_callable_from(signature));
                msg.push(format!("{signature}{suffix}"));
            }
            // We intentionally discard closest_overload.call_errors. When no overload matches,
            // there's a high likelihood that the "closest" one by our heuristic isn't the right
            // one, in which case the call errors are just noise.
            errors.add(
                arguments_range,
                ErrorInfo::new(ErrorKind::NoMatchingOverload, context),
                msg,
            );
            (
                self.heap.mk_any_error(),
                closest_overload.func.1.signature.clone(),
            )
        }
    }

    fn arity_mismatch_size(
        &self,
        expected_arg_counts: &ArgCounts,
        self_obj: Option<&Type>,
        posargs: &[CallArg],
        keywords: &[CallKeyword],
    ) -> usize {
        // If the number of non-variadic args is less than the min or more than the max, get the
        // absolute difference between actual and expected. We ignore variadic args because we
        // can't figure out how many args they contribute without inferring their types, which we
        // want to avoid to keep this arity check lightweight.
        let (n_posargs, has_varargs) = {
            let n = posargs
                .iter()
                .filter(|arg| matches!(arg, CallArg::Arg(_)))
                .count();
            ((self_obj.is_some() as usize) + n, posargs.len() > n)
        };
        let n_keywords = keywords.iter().filter(|kw| kw.arg.is_some()).count();
        let has_kwargs = keywords.len() > n_keywords;
        let mismatch_size = |count: &ArgCount, n, variadic| {
            // Check for too few args.
            let min_mismatch = count
                .min
                .saturating_sub(if variadic { count.min } else { n });
            // Check for too many args.
            let max_mismatch = n.saturating_sub(count.max.unwrap_or(n));
            max(min_mismatch, max_mismatch)
        };
        let pos_mismatch = mismatch_size(&expected_arg_counts.positional, n_posargs, has_varargs);
        let kw_mismatch = mismatch_size(&expected_arg_counts.keyword, n_keywords, has_kwargs);
        let overall_mismatch = mismatch_size(
            &expected_arg_counts.overall,
            n_posargs + n_keywords,
            has_varargs || has_kwargs,
        );
        // overall_mismatch will double-count, but this is ok because all we care about is whether
        // the mismatch is 0 (correct arity) and relative mismatch sizes between overloads
        pos_mismatch + kw_mismatch + overall_mismatch
    }

    /// Returns the overload that matches the given arguments, or the one that produces the fewest
    /// errors if none matches, plus a bool to indicate whether we found a match.
    fn find_closest_overload<'c>(
        &self,
        overloads: &Vec1<&'c TargetWithTParams<Function>>,
        metadata: &FuncMetadata,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        errors: &ErrorCollector,
        hint: Option<HintRef>,
        ctor_targs: &Option<&mut TArgs>,
    ) -> (CalledOverload<'c>, bool) {
        // Collect placeholder vars so we can save/restore them around each overload evaluation. This
        // prevents premature pinning of vars on failed overload calls.
        let placeholder_vars = self.collect_placeholder_vars(self_obj, args, keywords);

        let mut matched_overloads = Vec::with_capacity(overloads.len());
        let mut closest_unmatched_overload: Option<CalledOverload<'c>> = None;
        for callable in overloads {
            let snapshot = self.solver().snapshot_vars(&placeholder_vars);
            let called_overload = self.call_overload(
                callable,
                metadata,
                self_obj,
                args,
                keywords,
                arguments_range,
                errors,
                None, // don't use the hint yet, it shouldn't influence overload selection
                ctor_targs,
            );
            self.solver().restore_vars(snapshot);
            let n_errors = called_overload.num_errors();
            if n_errors == 0 {
                matched_overloads.push(called_overload);
            } else {
                match &closest_unmatched_overload {
                    Some(overload) if overload.num_errors() <= n_errors => {}
                    _ => {
                        closest_unmatched_overload = Some(called_overload);
                    }
                }
            }
        }
        if matched_overloads.is_empty() {
            // There's always at least one overload, so if none of them matched, the closest overload must be non-None.
            (closest_unmatched_overload.unwrap(), false)
        } else {
            // If there are multiple overloads, use steps 4-6 here to select one:
            // https://typing.python.org/en/latest/spec/overload.html#overload-call-evaluation.
            let spec_compliant = self.solver().spec_compliant_overloads;
            if matched_overloads.len() > 1 {
                // Step 4: if any arguments supply an unknown number of args and at least one
                // overload has a corresponding variadic parameter, eliminate overloads without
                // this parameter.
                let nargs_unknown = args.iter().any(|arg| match arg {
                    CallArg::Arg(_) => false,
                    CallArg::Star(val, _) => {
                        !matches!(val.infer(self, errors), Type::Tuple(Tuple::Concrete(_)))
                    }
                });
                if nargs_unknown {
                    let has_varargs = |o: &CalledOverload<'_>| {
                        matches!(
                            &o.func.1.signature.params, Params::List(params)
                            if params.items().iter().any(|p| matches!(p, Param::Varargs(..))))
                    };
                    if matched_overloads.iter().any(has_varargs) {
                        matched_overloads.retain(has_varargs);
                    }
                }
                let nkeywords_unknown = keywords.iter().any(|kw| {
                    kw.arg.is_none() && !matches!(kw.value.infer(self, errors), Type::TypedDict(_))
                });
                if nkeywords_unknown {
                    let has_kwargs = |o: &CalledOverload<'_>| {
                        matches!(
                            &o.func.1.signature.params, Params::List(params)
                            if params.items().iter().any(|p| matches!(p, Param::Kwargs(..))))
                    };
                    if matched_overloads.iter().any(has_kwargs) {
                        matched_overloads.retain(has_kwargs);
                    }
                }
            }
            if matched_overloads.len() > 1 {
                // Step 5, part 1: for each overload, check whether it's the case that all possible
                // materializations of each argument are assignable to the corresponding parameter.
                // If so, eliminate all subsequent overloads.
                //
                // Additional filter (non-spec-compliant): only materialize arguments that have
                // multiple possible parameter types. If an argument contains `Any` but has the
                // same parameter type in all candidate overloads, it does not contribute to
                // ambiguity in overload selection. This matches pyright, mypy, and ty.
                let owner = Owner::new();
                let mut changed = false;
                let should_materialize = |arg_range| {
                    if spec_compliant {
                        return true;
                    }
                    let mut param_types = matched_overloads
                        .iter()
                        .filter_map(|o| o.expected_types.get(&arg_range));
                    let Some(first) = param_types.next() else {
                        // If we can't find the expected type, be conservative and assume there may be multiple.
                        return true;
                    };
                    for t in param_types {
                        if !self.is_equivalent(first, t) {
                            return true;
                        }
                    }
                    false
                };
                let materialized_args = args.map(|arg| {
                    let (materialized_arg, arg_changed) = if should_materialize(arg.range()) {
                        arg.materialize(self, errors, &owner)
                    } else {
                        (arg.clone(), false)
                    };
                    changed |= arg_changed;
                    materialized_arg
                });
                let materialized_keywords = keywords.map(|kw| {
                    let (materialized_kw, kw_changed) = if should_materialize(kw.range()) {
                        kw.materialize(self, errors, &owner)
                    } else {
                        (kw.clone(), false)
                    };
                    changed |= kw_changed;
                    materialized_kw
                });
                let split_point = if !changed {
                    // Shortcut: if the arguments haven't changed, we know that the first overload
                    // matches and we can eliminate all the rest.
                    Some(1)
                } else {
                    matched_overloads
                        .iter()
                        .find_position(|o| {
                            let snapshot = self.solver().snapshot_vars(&placeholder_vars);
                            let res = self.call_overload(
                                o.func,
                                metadata,
                                self_obj,
                                &materialized_args,
                                &materialized_keywords,
                                arguments_range,
                                errors,
                                None, // don't use the hint yet, it shouldn't influence overload selection
                                &None,
                            );
                            self.solver().restore_vars(snapshot);
                            res.num_errors() == 0
                        })
                        .map(|(split_point, _)| split_point + 1)
                };
                if let Some(split_point) = split_point {
                    let _ = matched_overloads.split_off(split_point);
                }
            }
            let selected_overload = if spec_compliant {
                self.disambiguate_overloads_spec_compliant(&matched_overloads)
            } else {
                self.disambiguate_overloads(&matched_overloads)
            };
            if let Some(idx) = selected_overload {
                let overload = matched_overloads
                    .into_iter()
                    .nth(idx)
                    .expect("Could not find selected overload");
                // Now that we've selected an overload, use the hint to contextually type the arguments.
                let contextual_overload = self.call_overload(
                    overload.func,
                    metadata,
                    self_obj,
                    args,
                    keywords,
                    arguments_range,
                    &self.error_collector(),
                    hint,
                    ctor_targs,
                );
                (
                    // Intentionally check only `call_errors` and not `specialization_errors`. The
                    // contextual pass re-runs with the hint and may legitimately introduce
                    // specialization errors that the matched-overload step already accounted for,
                    // so we only fall back to the no-hint version on hard call errors. See
                    // `test::generic_restriction::test_nested_call_of_overloaded_function_preserves_bound`.
                    if contextual_overload.call_errors.is_empty() {
                        contextual_overload
                    } else {
                        overload
                    },
                    true,
                )
            } else {
                // Ambiguous call, return Any. Arbitrarily use the first overload as the matched one.
                let first_overload = matched_overloads
                    .into_iter()
                    .next()
                    .expect("Expected at least one overload");
                (
                    CalledOverload {
                        res: self.heap.mk_any_implicit(),
                        ..first_overload
                    },
                    true,
                )
            }
        }
    }

    fn disambiguate_overloads_spec_compliant(
        &self,
        matched_overloads: &[CalledOverload<'_>],
    ) -> Option<usize> {
        // Step 5, part 2: are all remaining return types equivalent to one another?
        // If not, the call is ambiguous.
        let mut matched_overloads = matched_overloads.iter();
        let first_overload = matched_overloads
            .next()
            .expect("Expected at least one overload");
        if matched_overloads.any(|o| !self.is_equivalent(&first_overload.res, &o.res)) {
            return None;
        }
        // Step 6: if there are still multiple matches, pick the first one.
        Some(0)
    }

    fn disambiguate_overloads(&self, matched_overloads: &[CalledOverload<'_>]) -> Option<usize> {
        // When a call to an overloaded function may match multiple overloads, the spec says to
        // return Any when the return types are not all equivalent.
        // However, neither mypy nor pyright fully follows this part of the spec, and many
        // third-party libraries have come to rely on mypy and pyright's behavior. So we do the
        // following for ecosystem compatibility:
        //
        // Step 6 (non-spec-compliant): does there exist a return type such that all
        // materializations of every other return type are assignable to it? If so, use this
        // return type. Else, return Any.
        //
        // We check materializations rather than assignability so that we end up with the most
        // "general" return type. E.g., if the candidates are `A[None]` and `A[Any]`, we want
        // to select `A[Any]`.
        //
        // First, find a candidate return type.
        let mut candidate = 0;
        for (i, o) in matched_overloads.iter().enumerate().skip(1) {
            if !self.is_subset_eq(&o.res.materialize(), &matched_overloads[candidate].res) {
                candidate = i;
            }
        }
        // We've already checked every return type after the candidate.
        // Check every return type before the candidate.
        for o in matched_overloads.iter().take(candidate) {
            if !self.is_subset_eq(&o.res.materialize(), &matched_overloads[candidate].res) {
                return None;
            }
        }
        Some(candidate)
    }

    /// Collect placeholder vars from self_obj and Type-valued arguments.
    fn collect_placeholder_vars(
        &self,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
    ) -> Vec<Var> {
        let mut placeholder_vars: Vec<Var> = Vec::new();
        let mut collect = |ty: &Type| {
            for var in ty.collect_maybe_placeholder_vars() {
                if !placeholder_vars.contains(&var) {
                    placeholder_vars.push(var);
                }
            }
        };
        if let Some(obj) = self_obj {
            collect(obj);
        }
        for arg in args {
            if let CallArg::Arg(TypeOrExpr::Type(ty, _))
            | CallArg::Star(TypeOrExpr::Type(ty, _), _) = arg
            {
                collect(ty);
            }
        }
        for kw in keywords {
            if let TypeOrExpr::Type(ty, _) = &kw.value {
                collect(ty);
            }
        }
        placeholder_vars
    }

    fn call_overload<'c>(
        &self,
        callable: &'c TargetWithTParams<Function>,
        metadata: &FuncMetadata,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        errors: &ErrorCollector,
        hint: Option<HintRef>,
        ctor_targs: &Option<&mut TArgs>,
    ) -> CalledOverload<'c> {
        // Create a copy of the class type arguments (if any) that should be filled in by this call.
        // The `callable_infer` call below will fill in this copy with the type arguments set
        // by the current overload, and we'll later use the copy to fill in the original
        // ctor_targs if this overload is chosen.
        let mut overload_ctor_targs = ctor_targs.as_ref().map(|x| (**x).clone());
        let tparams = callable.0.as_deref();

        let call_errors = self.error_collector();
        let (res, specialization_errors, expected_types) = self.callable_infer(
            callable.1.signature.clone(),
            Some(&metadata.kind),
            tparams,
            self_obj.cloned(),
            args,
            keywords,
            arguments_range,
            errors,
            &call_errors,
            // We intentionally drop the context here, as arg errors don't need it,
            // and if there are any call errors, we'll log a "No matching overloads"
            // error with the necessary context.
            None,
            hint,
            overload_ctor_targs.as_mut(),
        );

        CalledOverload {
            func: callable,
            res,
            ctor_targs: overload_ctor_targs,
            call_errors,
            specialization_errors,
            expected_types,
        }
    }
}
