/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::max;

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
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Params;
use crate::types::literal::Lit;
use crate::types::types::Type;

struct CalledOverload {
    func: TargetWithTParams<Function>,
    res: Type,
    ctor_targs: Option<TArgs>,
    call_errors: ErrorCollector,
}

/// Performs argument type expansion for arguments to an overloaded function.
struct ArgsExpander<'a> {
    /// The index of the next argument to expand. Left is positional args; right, keyword args.
    idx: Either<usize, usize>,
    /// Current argument lists.
    arg_lists: Vec<(Vec<CallArg<'a>>, Vec<CallKeyword<'a>>)>,
    /// Hard-coded limit to how many times we'll expand.
    gas: Gas,
}

impl<'a> ArgsExpander<'a> {
    const GAS: usize = 100;

    fn new(posargs: Vec<CallArg<'a>>, keywords: Vec<CallKeyword<'a>>) -> Self {
        Self {
            idx: if posargs.is_empty() {
                Either::Right(0)
            } else {
                Either::Left(0)
            },
            arg_lists: vec![(posargs, keywords)],
            gas: Gas::new(Self::GAS as isize),
        }
    }

    /// Expand the next argument and return the expanded argument lists.
    fn expand<Ans: LookupAnswer>(
        &mut self,
        solver: &'a AnswersSolver<Ans>,
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
        let expanded_types = Self::expand_type(value.infer(solver, errors), solver);
        if expanded_types.is_empty() {
            // Nothing to expand here, try the next argument.
            self.expand(solver, errors, owner)
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
    fn expand_type<Ans: LookupAnswer>(ty: Type, solver: &AnswersSolver<Ans>) -> Vec<Type> {
        match ty {
            Type::Union(box Union { members: ts, .. }) => ts,
            Type::ClassType(cls) if cls.is_builtin("bool") => vec![
                Type::Literal(Lit::Bool(true)),
                Type::Literal(Lit::Bool(false)),
            ],
            Type::ClassType(cls) if solver.get_metadata_for_class(cls.class_object()).is_enum() => {
                solver
                    .get_enum_members(cls.class_object())
                    .into_iter()
                    .map(Type::Literal)
                    .collect()
            }
            Type::Type(box Type::Union(box Union { members: ts, .. })) => {
                ts.into_map(Type::type_form)
            }
            Type::Tuple(Tuple::Concrete(elements)) => {
                let mut count: usize = 1;
                let mut changed = false;
                let mut element_expansions = Vec::new();
                for e in elements {
                    let element_expansion = Self::expand_type(e.clone(), solver);
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
                        .map(Type::concrete_tuple)
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
        metadata: FuncMetadata,
        self_obj: Option<Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
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
                    func: arity_closest_overload.unwrap().0.clone(),
                    res: Type::any_error(),
                    ctor_targs: None,
                    call_errors: self.error_collector(),
                },
                false,
            ),
            Ok(arity_compatible_overloads) => {
                // Step 2: evaluate each overload as a regular (non-overloaded) call.
                // Note: steps 4-6 are performed in `find_closest_overload`.
                let (mut closest_overload, mut matched) = self.find_closest_overload(
                    &arity_compatible_overloads,
                    &metadata,
                    self_obj.as_ref(),
                    &args,
                    &keywords,
                    range,
                    errors,
                    hint,
                    &ctor_targs,
                );

                // Step 3: perform argument type expansion.
                let mut args_expander = ArgsExpander::new(args.clone(), keywords.clone());
                let owner = Owner::new();
                'outer: while !matched
                    && let Some(arg_lists) = args_expander.expand(self, errors, &owner)
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
                            &metadata,
                            self_obj.as_ref(),
                            cur_args,
                            cur_keywords,
                            range,
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
                        closest_overload = CalledOverload {
                            func: first_overload.func.clone(),
                            ctor_targs: first_overload.ctor_targs.clone(),
                            res: self.unions(matched_overloads.into_map(|o| o.res)),
                            call_errors: self.error_collector(),
                        };
                        matched = true;
                        break;
                    }
                }
                (closest_overload, matched)
            }
        };

        if matched
            && let Some(targs) = ctor_targs
            && let Some(chosen_targs) = closest_overload.ctor_targs
        {
            *targs = chosen_targs;
        }
        // Record the closest overload to power IDE services.
        self.record_overload_trace(
            range,
            overloads.map(|TargetWithTParams(_, Function { signature, .. })| signature),
            &closest_overload.func.1.signature,
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
                errors.add(range, ErrorInfo::new(ErrorKind::Deprecated, context), msg);
            }
            (closest_overload.res, closest_overload.func.1.signature)
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
            for overload in overloads {
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
                        .unwrap_or(overload.1.signature),
                    None => overload.1.signature,
                };
                let signature = self
                    .solver()
                    .for_display(Type::Callable(Box::new(signature)));
                msg.push(format!("{signature}{suffix}"));
            }
            // We intentionally discard closest_overload.call_errors. When no overload matches,
            // there's a high likelihood that the "closest" one by our heuristic isn't the right
            // one, in which case the call errors are just noise.
            errors.add(
                range,
                ErrorInfo::new(ErrorKind::NoMatchingOverload, context),
                msg,
            );
            (Type::any_error(), closest_overload.func.1.signature)
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
        mismatch_size(&expected_arg_counts.positional, n_posargs, has_varargs)
            + mismatch_size(&expected_arg_counts.keyword, n_keywords, has_kwargs)
    }

    /// Returns the overload that matches the given arguments, or the one that produces the fewest
    /// errors if none matches, plus a bool to indicate whether we found a match.
    fn find_closest_overload(
        &self,
        overloads: &Vec1<&TargetWithTParams<Function>>,
        metadata: &FuncMetadata,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        hint: Option<HintRef>,
        ctor_targs: &Option<&mut TArgs>,
    ) -> (CalledOverload, bool) {
        let mut matched_overloads = Vec::with_capacity(overloads.len());
        let mut closest_unmatched_overload: Option<CalledOverload> = None;
        for callable in overloads {
            let called_overload = self.try_call_overload(
                callable, metadata, self_obj, args, keywords, range, errors, hint, ctor_targs,
            );
            if called_overload.call_errors.is_empty() {
                matched_overloads.push(called_overload);
            } else {
                match &closest_unmatched_overload {
                    Some(overload)
                        if overload.call_errors.len() <= called_overload.call_errors.len() => {}
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
                    let has_varargs = |o: &CalledOverload| {
                        matches!(
                            &o.func.1.signature.params, Params::List(params)
                            if params.items().iter().any(|p| matches!(p, Param::VarArg(..))))
                    };
                    if matched_overloads.iter().any(has_varargs) {
                        matched_overloads.retain(has_varargs);
                    }
                }
                let nkeywords_unknown = keywords.iter().any(|kw| {
                    kw.arg.is_none() && !matches!(kw.value.infer(self, errors), Type::TypedDict(_))
                });
                if nkeywords_unknown {
                    let has_kwargs = |o: &CalledOverload| {
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
                let owner = Owner::new();
                let mut changed = false;
                let materialized_args = args.map(|arg| {
                    let (materialized_arg, arg_changed) = arg.materialize(self, errors, &owner);
                    changed |= arg_changed;
                    materialized_arg
                });
                let materialized_keywords = keywords.map(|kw| {
                    let (materialized_kw, kw_changed) = kw.materialize(self, errors, &owner);
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
                            let res = self.try_call_overload(
                                &o.func,
                                metadata,
                                self_obj,
                                &materialized_args,
                                &materialized_keywords,
                                range,
                                errors,
                                hint,
                                &None,
                            );
                            res.call_errors.is_empty()
                        })
                        .map(|(split_point, _)| split_point + 1)
                };
                if let Some(split_point) = split_point {
                    let _ = matched_overloads.split_off(split_point);
                }
            }
            // Step 5, part 2: are all remaining return types equivalent to one another?
            // If not, the call is ambiguous.
            let mut matched_overloads = matched_overloads.into_iter();
            let first_overload = matched_overloads.next().unwrap();
            if matched_overloads.any(|o| !self.is_equal(&first_overload.res, &o.res)) {
                return (
                    CalledOverload {
                        res: Type::any_implicit(),
                        ..first_overload
                    },
                    true,
                );
            }
            // Step 6: if there are still multiple matches, pick the first one.
            (first_overload, true)
        }
    }

    fn try_call_overload(
        &self,
        callable: &TargetWithTParams<Function>,
        metadata: &FuncMetadata,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        hint: Option<HintRef>,
        ctor_targs: &Option<&mut TArgs>,
    ) -> CalledOverload {
        // Create a copy of the class type arguments (if any) that should be filled in by this call.
        // The `callable_infer` call below will fill in this copy with the type arguments set
        // by the current overload, and we'll later use the copy to fill in the original
        // ctor_targs if this overload is chosen.
        let mut overload_ctor_targs = ctor_targs.as_ref().map(|x| (**x).clone());
        let tparams = callable.0.as_deref();

        let mut try_call = |hint| {
            let call_errors = self.error_collector();
            let res = self.callable_infer(
                callable.1.signature.clone(),
                Some(&metadata.kind),
                tparams,
                self_obj.cloned(),
                args,
                keywords,
                range,
                errors,
                &call_errors,
                // We intentionally drop the context here, as arg errors don't need it,
                // and if there are any call errors, we'll log a "No matching overloads"
                // error with the necessary context.
                None,
                hint,
                overload_ctor_targs.as_mut(),
            );
            (call_errors, res)
        };

        // We want to use our hint to contextually type the arguments, but errors resulting
        // from the hint should not influence overload selection. If there are call errors, we
        // try again without a hint in case we can still match this overload.
        let (call_errors, res) = try_call(hint);
        let (call_errors, res) = if tparams.is_some() && hint.is_some() && !call_errors.is_empty() {
            try_call(None)
        } else {
            (call_errors, res)
        };

        CalledOverload {
            func: callable.clone(),
            res,
            ctor_targs: overload_ctor_targs,
            call_errors,
        }
    }
}
