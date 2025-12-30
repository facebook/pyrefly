/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter;

use itertools::EitherOrBoth;
use itertools::Itertools;
use itertools::izip;
use pyrefly_python::dunder;
use pyrefly_types::literal::Lit;
use pyrefly_types::read_only::ReadOnlyReason;
use pyrefly_types::typed_dict::ExtraItem;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::typed_dict::TypedDictField;
use pyrefly_types::types::Union;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::solver::solver::OpenTypedDictSubsetError;
use crate::solver::solver::Subset;
use crate::solver::solver::SubsetError;
use crate::solver::solver::TypedDictSubsetError;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::ClassType;
use crate::types::quantified::QuantifiedKind;
use crate::types::simplify::unions;
use crate::types::tuple::Tuple;
use crate::types::type_var::Restriction;
use crate::types::type_var::Variance;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Type;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TypedDictFieldId {
    Name(Name),
    ExtraItems,
}

impl TypedDictFieldId {
    fn display_name(&self) -> Name {
        match self {
            TypedDictFieldId::Name(name) => name.clone(),
            TypedDictFieldId::ExtraItems => Name::from("<extra_items>"),
        }
    }
}

fn ok_or(b: bool, e: SubsetError) -> Result<(), SubsetError> {
    b.then_some(()).ok_or(e)
}

fn all<T>(
    it: impl Iterator<Item = T>,
    mut check: impl FnMut(T) -> Result<(), SubsetError>,
) -> Result<(), SubsetError> {
    for x in it {
        check(x)?;
    }
    Ok(())
}

fn any<T>(
    it: impl Iterator<Item = T>,
    mut check: impl FnMut(T) -> Result<(), SubsetError>,
) -> Result<(), SubsetError> {
    let mut err = None;
    for x in it {
        match check(x) {
            Ok(()) => return Ok(()),
            Err(e) if err.is_none() => err = Some(e),
            Err(_) => {}
        }
    }
    Err(err.unwrap_or(SubsetError::Other))
}

impl<'a, Ans: LookupAnswer> Subset<'a, Ans> {
    /// Can a function with l_args be called as a function with u_args?
    fn is_subset_param_list(
        &mut self,
        l_args: &[Param],
        u_args: &[Param],
    ) -> Result<(), SubsetError> {
        let mut l_args = l_args.iter();
        let mut u_args = u_args.iter();
        let mut l_arg = l_args.next();
        let mut u_arg = u_args.next();
        // This holds any Param::Pos from `u` that matched *args from `l`.
        // When handling keyword params, we make sure that they can be passed by name.
        let mut u_param_matched_with_l_varargs = Vec::new();
        // Handle positional args
        loop {
            match (l_arg, u_arg) {
                (None, None) => {
                    if u_param_matched_with_l_varargs.is_empty() {
                        return Ok(());
                    } else {
                        // We can't return early since we need to check that the matched params from `u`
                        // can be called by name.
                        break;
                    }
                }
                (
                    Some(Param::PosOnly(_, l, l_req) | Param::Pos(_, l, l_req)),
                    Some(Param::PosOnly(_, u, u_req)),
                ) if (*u_req == Required::Required || matches!(l_req, Required::Optional(_))) => {
                    self.is_subset_eq(u, l)?;
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(Param::Pos(l_name, l, l_req)), Some(Param::Pos(u_name, u, u_req)))
                    if *u_req == Required::Required || matches!(l_req, Required::Optional(_)) =>
                {
                    if l_name != u_name {
                        return Err(SubsetError::PosParamName(l_name.clone(), u_name.clone()));
                    }
                    self.is_subset_eq(u, l)?;
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(Param::VarArg(_, Type::Unpack(l))), None) => {
                    self.is_subset_eq(&Type::concrete_tuple(Vec::new()), l)?;
                    l_arg = l_args.next();
                }
                (None, Some(Param::VarArg(_, Type::Unpack(u)))) => {
                    self.is_subset_eq(&Type::concrete_tuple(Vec::new()), u)?;
                    u_arg = u_args.next();
                }
                (
                    Some(
                        Param::PosOnly(_, _, Required::Optional(_))
                        | Param::Pos(_, _, Required::Optional(_))
                        | Param::VarArg(_, _),
                    ),
                    None,
                ) => {
                    l_arg = l_args.next();
                }
                (Some(Param::KwOnly(_, _, Required::Optional(_)) | Param::Kwargs(_, _)), None) => {
                    if u_param_matched_with_l_varargs.is_empty() {
                        l_arg = l_args.next();
                    } else {
                        // Don't consume kw-only and kwarg params from `l` yet, we need them to
                        // check that the matched params from `u` can be called by name
                        break;
                    }
                }
                (
                    Some(Param::VarArg(_, Type::Unpack(l))),
                    Some(Param::PosOnly(_, _, Required::Required)),
                ) => {
                    let mut u_types = Vec::new();
                    loop {
                        if let Some(Param::PosOnly(_, u, Required::Required)) = u_arg {
                            u_types.push(u.clone());
                            u_arg = u_args.next();
                        } else if let Some(Param::VarArg(_, Type::Unpack(u))) = u_arg {
                            self.is_subset_eq(
                                &Type::unpacked_tuple(u_types, (**u).clone(), Vec::new()),
                                l,
                            )?;
                            l_arg = l_args.next();
                            u_arg = u_args.next();
                            break;
                        } else if let Some(Param::VarArg(_, u)) = u_arg {
                            self.is_subset_eq(
                                &Type::unpacked_tuple(
                                    u_types,
                                    Type::unbounded_tuple(u.clone()),
                                    Vec::new(),
                                ),
                                l,
                            )?;
                            l_arg = l_args.next();
                            u_arg = u_args.next();
                            break;
                        } else {
                            self.is_subset_eq(&Type::concrete_tuple(u_types), l)?;
                            l_arg = l_args.next();
                            break;
                        }
                    }
                }
                (
                    Some(Param::PosOnly(_, _, _) | Param::Pos(_, _, _)),
                    Some(Param::VarArg(_, Type::Unpack(u))),
                ) => {
                    let mut l_types = Vec::new();
                    loop {
                        if let Some(Param::PosOnly(_, l, _) | Param::Pos(_, l, _)) = l_arg {
                            l_types.push(l.clone());
                            l_arg = l_args.next();
                        } else if let Some(Param::VarArg(_, Type::Unpack(l))) = l_arg {
                            self.is_subset_eq(
                                u,
                                &Type::unpacked_tuple(l_types, (**l).clone(), Vec::new()),
                            )?;
                            l_arg = l_args.next();
                            u_arg = u_args.next();
                            break;
                        } else if let Some(Param::VarArg(_, l)) = l_arg {
                            self.is_subset_eq(
                                u,
                                &Type::unpacked_tuple(
                                    l_types,
                                    Type::unbounded_tuple(l.clone()),
                                    Vec::new(),
                                ),
                            )?;
                            l_arg = l_args.next();
                            u_arg = u_args.next();
                            break;
                        } else {
                            self.is_subset_eq(u, &Type::concrete_tuple(l_types))?;
                            u_arg = u_args.next();
                            break;
                        }
                    }
                }
                (Some(Param::VarArg(_, l)), Some(Param::PosOnly(_, u, _))) => {
                    self.is_subset_eq(u, l)?;
                    u_arg = u_args.next();
                }
                (Some(Param::VarArg(_, l)), Some(Param::Pos(name, u, _))) => {
                    // Param::Pos can be passed positionally or by name, so if it matches *args
                    // we need to make sure it matches an optional kw-only argument or *kwargs
                    self.is_subset_eq(u, l)?;
                    u_param_matched_with_l_varargs.push((name, u));
                    u_arg = u_args.next();
                }
                (
                    Some(Param::VarArg(_, Type::Unpack(l))),
                    Some(Param::VarArg(_, Type::Unpack(u))),
                ) => {
                    self.is_subset_eq(u, l)?;
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(Param::VarArg(_, Type::Any(_))), Some(Param::VarArg(_, Type::Unpack(_)))) => {
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(Param::VarArg(_, l)), Some(Param::VarArg(_, Type::Unpack(u)))) => {
                    self.is_subset_eq(u, &Type::unbounded_tuple(l.clone()))?;
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(Param::VarArg(_, Type::Unpack(l))), Some(Param::VarArg(_, u))) => {
                    self.is_subset_eq(&Type::unbounded_tuple(u.clone()), l)?;
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(Param::VarArg(_, l)), Some(Param::VarArg(_, u))) => {
                    self.is_subset_eq(u, l)?;
                    l_arg = l_args.next();
                    u_arg = u_args.next();
                }
                (Some(_), Some(Param::KwOnly(_, _, _) | Param::Kwargs(_, _))) => {
                    break;
                }
                _ => return Err(SubsetError::Other),
            }
        }
        // We can use a HashMap for `l_keywords` since the order does not matter
        let mut l_keywords = HashMap::new();
        let mut l_kwargs = None;
        for arg in Option::into_iter(l_arg).chain(l_args) {
            match arg {
                Param::KwOnly(name, ty, required) | Param::Pos(name, ty, required) => {
                    l_keywords.insert(name.clone(), (ty.clone(), *required == Required::Required));
                }
                Param::Kwargs(_, ty) => l_kwargs = Some(ty.clone()),
                _ => (),
            }
        }
        let mut u_keywords = SmallMap::new();
        let mut u_kwargs = None;
        for arg in Option::into_iter(u_arg).chain(u_args) {
            match arg {
                Param::KwOnly(name, ty, required) => {
                    u_keywords.insert(name.clone(), (ty.clone(), *required == Required::Required));
                }
                Param::Kwargs(_, ty) => u_kwargs = Some(ty.clone()),
                _ => (),
            }
        }
        let object_type = self.type_order.stdlib().object().clone().to_type();
        // Expand typed dict kwargs if necessary, check regular kwargs
        let l_kwargs = match (l_kwargs, u_kwargs) {
            (
                Some(Type::Unpack(box Type::TypedDict(l_typed_dict))),
                Some(Type::Unpack(box Type::TypedDict(u_typed_dict))),
            ) => {
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&l_typed_dict)
                {
                    l_keywords.insert(name, (ty, required == Required::Required));
                }
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&u_typed_dict)
                {
                    u_keywords.insert(name, (ty, required == Required::Required));
                }
                Some(object_type)
            }
            (Some(Type::Unpack(box Type::TypedDict(l_typed_dict))), _) => {
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&l_typed_dict)
                {
                    l_keywords.insert(name, (ty, required == Required::Required));
                }
                Some(object_type)
            }
            (l_kwargs @ Some(_), Some(Type::Unpack(box Type::TypedDict(u_typed_dict)))) => {
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&u_typed_dict)
                {
                    u_keywords.insert(name, (ty, required == Required::Required));
                }
                l_kwargs
            }
            (Some(l), Some(u)) => {
                self.is_subset_eq(&u, &l)?;
                Some(l)
            }
            (None, Some(_)) => {
                return Err(SubsetError::Other);
            }
            (l_kwargs, _) => l_kwargs,
        };
        // These parameters from `u` may be passed by name or position. We matched the positional
        // case with *args from `l` already; now we check that they can be passed by name.
        for (name, u_ty) in u_param_matched_with_l_varargs {
            if let Some((l_ty, l_req)) = l_keywords.remove(name) {
                // Matched kw-only param from `l` must be optional, since the argument will not be
                // present if passed positionally.
                if l_req {
                    return Err(SubsetError::Other);
                }
                self.is_subset_eq(u_ty, &l_ty)?;
            } else if let Some(l_ty) = &l_kwargs {
                self.is_subset_eq(u_ty, l_ty)?;
            } else {
                return Err(SubsetError::Other);
            }
        }
        // Handle keyword-only args
        for (name, (u_ty, u_req)) in u_keywords.iter() {
            if let Some((l_ty, l_req)) = l_keywords.remove(name) {
                if !*u_req && l_req {
                    return Err(SubsetError::Other);
                }
                self.is_subset_eq(u_ty, &l_ty)?;
            } else if let Some(l_ty) = &l_kwargs {
                self.is_subset_eq(u_ty, l_ty)?;
            } else {
                return Err(SubsetError::Other);
            }
        }
        for (_, l_req) in l_keywords.values() {
            if *l_req {
                return Err(SubsetError::Other);
            }
        }
        Ok(())
    }

    fn is_subset_params(
        &mut self,
        l_params: &Params,
        u_params: &Params,
    ) -> Result<(), SubsetError> {
        match (l_params, u_params) {
            (Params::Ellipsis, Params::ParamSpec(_, pspec)) => {
                self.is_subset_eq(&Type::Ellipsis, pspec)
            }
            (Params::ParamSpec(_, pspec), Params::Ellipsis) => {
                self.is_subset_eq(pspec, &Type::Ellipsis)
            }
            (Params::Ellipsis, _) | (_, Params::Ellipsis) => Ok(()),
            (Params::List(l_args), Params::List(u_args)) => {
                self.is_subset_param_list(l_args.items(), u_args.items())
            }
            (Params::List(ls), Params::ParamSpec(args, pspec)) => {
                self.is_paramlist_subset_of_paramspec(ls, args, pspec)
            }
            (Params::ParamSpec(args, pspec), Params::List(ls)) => {
                self.is_paramspec_subset_of_paramlist(args, pspec, ls)
            }
            (Params::ParamSpec(ls, p1), Params::ParamSpec(us, p2)) => {
                self.is_paramspec_subset_of_paramspec(ls, p1, us, p2)
            }
            (Params::Materialization, _) => Err(SubsetError::Other),
            (_, Params::Materialization) => {
                self.is_subset_params(l_params, &Params::List(ParamList::everything()))
            }
        }
    }

    fn is_subset_protocol(&mut self, got: Type, protocol: ClassType) -> Result<(), SubsetError> {
        // First check exact (Type, Type) recursive assumptions
        let recursive_check = (got.clone(), Type::ClassType(protocol.clone()));
        if !self.recursive_assumptions.insert(recursive_check.clone()) {
            // Assume recursive checks are true
            return Ok(());
        }

        // For class-level coinductive reasoning: if the `got` type's type arguments
        // contain Vars, we're likely in a recursive pattern (e.g., checking method return
        // types that reference the same classes). Use (Class, Class) matching to detect
        // cycles that would otherwise be missed due to fresh Var creation.
        let class_check = if let Type::ClassType(got_class) = &got {
            // Check if any targ contains a Var (indicating we're in recursive pattern)
            let mut contains_var = false;
            for targ in got_class.targs().as_slice() {
                targ.universe(&mut |t| contains_var |= matches!(t, Type::Var(_)));
                if contains_var {
                    break;
                }
            }

            if contains_var {
                let key = (
                    got_class.class_object().clone(),
                    protocol.class_object().clone(),
                );
                if !self.class_protocol_assumptions.insert(key.clone()) {
                    // Coinductive: assume this recursive class-level check succeeds
                    self.recursive_assumptions.shift_remove(&recursive_check);
                    return Ok(());
                }
                Some(key)
            } else {
                None
            }
        } else {
            None
        };

        let res = self.is_subset_protocol_inner(got, protocol);

        // Clean up assumptions
        self.recursive_assumptions.shift_remove(&recursive_check);
        if let Some(key) = class_check {
            self.class_protocol_assumptions.shift_remove(&key);
        }

        res
    }

    fn is_subset_protocol_inner(
        &mut self,
        got: Type,
        protocol: ClassType,
    ) -> Result<(), SubsetError> {
        // TODO: Remove this once pandas 2.x is no longer supported.
        // This is fixed in pandas 3.0 stubs. Until then, we hard-code that list/tuple satisfy
        // SequenceNotStr. See https://github.com/pandas-dev/pandas/issues/56995
        if protocol.has_qname("pandas._typing", "SequenceNotStr")
            && let Type::ClassType(got_cls) = &got
            && (got_cls.is_builtin("list") || got_cls.is_builtin("tuple"))
        {
            return Ok(());
        }
        let protocol_members = self
            .type_order
            .get_protocol_member_names(protocol.class_object());
        for name in protocol_members {
            if name == dunder::INIT || name == dunder::NEW {
                // Protocols can't be instantiated
                continue;
            }
            if name == dunder::SLOTS {
                // Skip `__slots__` check
                continue;
            }
            if matches!(
                got,
                Type::Callable(_) | Type::Function(_) | Type::BoundMethod(_)
            ) && name == dunder::CALL
                && let Some(want) = self.type_order.instance_as_dunder_call(&protocol)
            {
                if let Type::BoundMethod(method) = &want
                    && let Some(want_no_self) =
                        self.type_order.bind_boundmethod(method, &mut |got, want| {
                            self.is_subset_eq(got, want).is_ok()
                        })
                {
                    self.is_subset_eq(&got, &want_no_self)?;
                } else {
                    self.is_subset_eq(&got, &want)?;
                }
            } else {
                self.type_order.is_protocol_subset_at_attr(
                    &got,
                    &protocol,
                    &name,
                    &mut |got, want| self.is_subset_eq(got, want),
                )?;
            }
        }
        Ok(())
    }

    fn is_subset_tuple(&mut self, got: &Tuple, want: &Tuple) -> Result<(), SubsetError> {
        match (got, want) {
            (Tuple::Concrete(lelts), Tuple::Concrete(uelts)) => {
                if lelts.len() == uelts.len() {
                    all(lelts.iter().zip(uelts), |(l, u)| self.is_subset_eq(l, u))
                } else {
                    Err(SubsetError::Other)
                }
            }
            (Tuple::Unbounded(box Type::Any(_)), _) | (_, Tuple::Unbounded(box Type::Any(_))) => {
                Ok(())
            }
            (Tuple::Concrete(lelts), Tuple::Unbounded(u)) => {
                all(lelts.iter(), |l| self.is_subset_eq(l, u))
            }
            (Tuple::Unbounded(l), Tuple::Unbounded(u)) => self.is_subset_eq(l, u),
            (Tuple::Concrete(lelts), Tuple::Unpacked(box (u_prefix, u_middle, u_suffix))) => {
                if lelts.len() < u_prefix.len() + u_suffix.len() {
                    Err(SubsetError::Other)
                } else {
                    let mut l_middle = Vec::new();
                    all(lelts.iter().enumerate(), |(idx, l)| {
                        if idx < u_prefix.len() {
                            self.is_subset_eq(l, &u_prefix[idx])
                        } else if idx >= lelts.len() - u_suffix.len() {
                            self.is_subset_eq(l, &u_suffix[idx + u_suffix.len() - lelts.len()])
                        } else {
                            l_middle.push(l.clone());
                            Ok(())
                        }
                    })?;
                    self.is_subset_eq(&Type::Tuple(Tuple::Concrete(l_middle)), u_middle)
                }
            }
            (Tuple::Unbounded(_), Tuple::Unpacked(box (u_prefix, u_middle, u_suffix))) => {
                if u_prefix.is_empty() && u_suffix.is_empty() {
                    self.is_subset_eq(&Type::Tuple(got.clone()), u_middle)
                } else {
                    Err(SubsetError::Other)
                }
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Unbounded(u)) => {
                all(l_prefix.iter(), |l| self.is_subset_eq(l, u))?;
                all(l_suffix.iter(), |l| self.is_subset_eq(l, u))?;
                self.is_subset_eq(l_middle, &Type::Tuple(want.clone()))
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Concrete(uelts)) => {
                if uelts.len() < l_prefix.len() + l_suffix.len() {
                    Err(SubsetError::Other)
                } else {
                    let mut u_middle = Vec::new();
                    all(uelts.iter().enumerate(), |(idx, u)| {
                        if idx < l_prefix.len() {
                            self.is_subset_eq(&l_prefix[idx], u)
                        } else if idx >= uelts.len() - l_suffix.len() {
                            self.is_subset_eq(&l_suffix[idx + l_suffix.len() - uelts.len()], u)
                        } else {
                            u_middle.push(u.clone());
                            Ok(())
                        }
                    })?;
                    self.is_subset_eq(l_middle, &Type::Tuple(Tuple::Concrete(u_middle)))
                }
            }
            (
                Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)),
                Tuple::Unpacked(box (u_prefix, u_middle, u_suffix)),
            ) => {
                // Invariant: 0-2 of these are non-empty
                // l_before and u_before cannot both be non-empty
                // l_after and u_after cannot both be non-empty
                let mut l_before = Vec::new();
                let mut l_after = Vec::new();
                let mut u_before = Vec::new();
                let mut u_after = Vec::new();
                all(
                    l_prefix.iter().zip_longest(u_prefix.iter()),
                    |pair| match pair {
                        EitherOrBoth::Both(l, u) => self.is_subset_eq(l, u),
                        EitherOrBoth::Left(l) => {
                            l_before.push(l.clone());
                            Ok(())
                        }
                        EitherOrBoth::Right(u) => {
                            u_before.push(u.clone());
                            Ok(())
                        }
                    },
                )?;
                all(
                    l_suffix.iter().rev().zip_longest(u_suffix.iter().rev()),
                    |pair| match pair {
                        EitherOrBoth::Both(l, u) => self.is_subset_eq(l, u),
                        EitherOrBoth::Left(l) => {
                            l_after.push(l.clone());
                            Ok(())
                        }
                        EitherOrBoth::Right(u) => {
                            u_after.push(u.clone());
                            Ok(())
                        }
                    },
                )?;
                l_after.reverse();
                u_after.reverse();

                self.is_subset_eq(
                    &Type::unpacked_tuple(l_before, l_middle.clone(), l_after),
                    u_middle,
                )?;
                self.is_subset_eq(
                    l_middle,
                    &Type::unpacked_tuple(u_before, u_middle.clone(), u_after),
                )?;
                Ok(())
            }
            _ => Err(SubsetError::Other),
        }
    }

    fn is_paramlist_subset_of_paramspec(
        &mut self,
        got: &ParamList,
        want_ts: &[(Type, Required)],
        want_pspec: &Type,
    ) -> Result<(), SubsetError> {
        if got.len() < want_ts.len() {
            return Err(SubsetError::Other);
        }
        let args = ParamList::new_types(want_ts.to_owned());
        let (pre, post) = got.items().split_at(args.len());
        self.is_subset_param_list(pre, args.items())?;
        self.is_subset_eq(
            &Type::ParamSpecValue(ParamList::new(post.to_vec())),
            want_pspec,
        )
    }

    fn is_paramspec_subset_of_paramlist(
        &mut self,
        got_ts: &[(Type, Required)],
        got_pspec: &Type,
        want: &ParamList,
    ) -> Result<(), SubsetError> {
        if want.len() < got_ts.len() {
            return Err(SubsetError::Other);
        }
        let args = ParamList::new_types(got_ts.to_owned());
        let (pre, post) = want.items().split_at(args.len());
        self.is_subset_param_list(args.items(), pre)?;
        self.is_subset_eq(
            got_pspec,
            &Type::ParamSpecValue(ParamList::new(post.to_vec())),
        )
    }

    fn is_paramspec_subset_of_paramspec(
        &mut self,
        got_ts: &[(Type, Required)],
        got_pspec: &Type,
        want_ts: &[(Type, Required)],
        want_pspec: &Type,
    ) -> Result<(), SubsetError> {
        // TODO: consider required-ness in prepended params
        match got_ts.len().cmp(&want_ts.len()) {
            Ordering::Greater => {
                let (got_ts_pre, got_ts_post) = got_ts.split_at(want_ts.len());
                for (l, u) in got_ts_pre.iter().zip(want_ts.iter()) {
                    self.is_subset_eq(&u.0, &l.0)?;
                }
                let got_ts_post = got_ts_post.to_vec().into_boxed_slice();
                self.is_subset_eq(
                    want_pspec,
                    &Type::Concatenate(got_ts_post, Box::new(got_pspec.clone())),
                )
            }
            Ordering::Less => {
                let (want_ts_pre, want_ts_post) = want_ts.split_at(got_ts.len());
                for (l, u) in got_ts.iter().zip(want_ts_pre.iter()) {
                    self.is_subset_eq(&u.0, &l.0)?;
                }
                let want_ts_post = want_ts_post.to_vec().into_boxed_slice();
                self.is_subset_eq(
                    &Type::Concatenate(want_ts_post, Box::new(want_pspec.clone())),
                    got_pspec,
                )
            }
            Ordering::Equal => {
                for (l, u) in got_ts.iter().zip(want_ts.iter()) {
                    self.is_subset_eq(&u.0, &l.0)?;
                }
                self.is_subset_eq(want_pspec, got_pspec)
            }
        }
    }

    fn typed_dict_extra_items_field(&self, extra_items: ExtraItems) -> TypedDictField {
        let ExtraItem { ty, read_only } = extra_items.extra_item(self.type_order.stdlib());
        TypedDictField {
            ty,
            required: false,
            read_only_reason: if read_only {
                Some(ReadOnlyReason::ReadOnlyQualifier)
            } else {
                None
            },
        }
    }

    fn get_typed_dict_fields(&self, td: &TypedDict) -> SmallMap<TypedDictFieldId, TypedDictField> {
        self.type_order
            .typed_dict_fields(td)
            .into_iter()
            .map(|(name, field)| (TypedDictFieldId::Name(name), field))
            .collect()
    }

    fn is_subset_typed_dict_field(
        &mut self,
        got: (&Name, &TypedDictField),
        want: (&Name, &TypedDictField),
        field_name: &Name,
    ) -> Result<(), SubsetError> {
        let (got_name, got_v) = got;
        let (want_name, want_v) = want;
        // For each key in `want`, `got` has the corresponding key
        // and the corresponding value type in `got` is consistent with the value type in `want`.
        // For each required key in `want`, the corresponding key is required in `got`.
        // For each non-required, non-readonly key in `want`, the corresponding key is not required in `got`.
        match (got_v.is_read_only(), want_v.is_read_only()) {
            // ReadOnly cannot be assigned to Non-ReadOnly
            (true, false) => {
                return Err(SubsetError::TypedDict(Box::new(
                    TypedDictSubsetError::ReadOnlyMismatch {
                        got: got_name.clone(),
                        want: want_name.clone(),
                        field: field_name.clone(),
                    },
                )));
            }
            // Non-ReadOnly fields are invariant
            (false, false) => self.is_equal(&got_v.ty, &want_v.ty).map_err(|_| {
                SubsetError::TypedDict(Box::new(TypedDictSubsetError::InvariantFieldMismatch {
                    got: got_name.clone(),
                    got_field_ty: got_v.ty.clone(),
                    want: want_name.clone(),
                    want_field_ty: want_v.ty.clone(),
                    field: field_name.clone(),
                }))
            })?,
            // ReadOnly `want` fields are covariant
            (_, true) => self.is_subset_eq(&got_v.ty, &want_v.ty).map_err(|_| {
                SubsetError::TypedDict(Box::new(TypedDictSubsetError::CovariantFieldMismatch {
                    got: got_name.clone(),
                    got_field_ty: got_v.ty.clone(),
                    want: want_name.clone(),
                    want_field_ty: want_v.ty.clone(),
                    field: field_name.clone(),
                }))
            })?,
        }
        if want_v.required {
            if !got_v.required {
                return Err(SubsetError::TypedDict(Box::new(
                    TypedDictSubsetError::RequiredMismatch {
                        got: got_name.clone(),
                        want: want_name.clone(),
                        field: field_name.clone(),
                    },
                )));
            }
        } else if !want_v.is_read_only() && got_v.required {
            // `want` field is `NotRequired` + read-write, `got` field is `Required`
            return Err(SubsetError::TypedDict(Box::new(
                TypedDictSubsetError::NotRequiredReadWriteMismatch {
                    got: got_name.clone(),
                    want: want_name.clone(),
                    field: field_name.clone(),
                },
            )));
        }
        Ok(())
    }

    fn is_subset_typed_dict(
        &mut self,
        got: &TypedDict,
        want: &TypedDict,
    ) -> Result<(), SubsetError> {
        let got_name = got.name();
        let want_name = want.name();
        let (got_fields, want_fields) = {
            let mut got_fields = self.get_typed_dict_fields(got);
            let mut want_fields = self.get_typed_dict_fields(want);
            let got_extra_items = self.type_order.typed_dict_extra_items(got);
            let want_extra_items = self.type_order.typed_dict_extra_items(want);
            if [&got_extra_items, &want_extra_items]
                .iter()
                .any(|extra| !matches!(extra, ExtraItems::Default))
            {
                // If either TypedDict has extra_items restrictions, add extra_items as a
                // non-required pseudo-field.
                got_fields.insert(
                    TypedDictFieldId::ExtraItems,
                    self.typed_dict_extra_items_field(got_extra_items),
                );
                want_fields.insert(
                    TypedDictFieldId::ExtraItems,
                    self.typed_dict_extra_items_field(want_extra_items),
                );
            }
            (got_fields, want_fields)
        };
        all(want_fields.iter(), |(k, want_v)| {
            let field_name = k.display_name();
            got_fields
                .get(k)
                .or_else(|| got_fields.get(&TypedDictFieldId::ExtraItems))
                .map_or(
                    Err(SubsetError::TypedDict(Box::new(
                        TypedDictSubsetError::MissingField {
                            got: got_name.clone(),
                            want: want_name.clone(),
                            field: field_name.clone(),
                        },
                    ))),
                    |got_v| {
                        self.is_subset_typed_dict_field(
                            (got_name, got_v),
                            (want_name, want_v),
                            &field_name,
                        )
                    },
                )
        })?;
        want_fields
            .get(&TypedDictFieldId::ExtraItems)
            .map_or(Ok(()), |want_v| {
                // Make sure all fields in `got` that aren't on `want` match the latter's `extra_items` type.
                all(got_fields.iter(), |(k, got_v)| {
                    if want_fields.contains_key(k) {
                        Ok(())
                    } else {
                        self.is_subset_typed_dict_field(
                            (got_name, got_v),
                            (want_name, want_v),
                            &k.display_name(),
                        )
                    }
                })
            })
    }

    fn is_subset_partial_typed_dict_field(
        &mut self,
        got_ty: &Type,
        want_field: &TypedDictField,
    ) -> Result<(), SubsetError> {
        if want_field.is_read_only() {
            // ReadOnly can only be updated with Never (i.e., no update)
            self.is_subset_eq(got_ty, &Type::never())
        } else {
            self.is_subset_eq(got_ty, &want_field.ty)
        }
    }

    /// Check TypedDict[got] <: PartialTypedDict[want]
    fn is_subset_partial_typed_dict(
        &mut self,
        got: &TypedDict,
        want: &TypedDict,
    ) -> Result<(), SubsetError> {
        let got_fields = self.type_order.typed_dict_fields(got);
        let want_fields = self.type_order.typed_dict_fields(want);
        let got_extra_items = self.type_order.typed_dict_extra_items(got);
        let want_extra_items = self.type_order.typed_dict_extra_items(want);
        all(want_fields.iter(), |(k, want_v)| {
            let got_field_ty = got_fields.get(k).map(|got_v| &got_v.ty);
            let got_ty = match (got_field_ty, &got_extra_items) {
                (Some(got_ty), _) => got_ty,
                (None, ExtraItems::Extra(item)) => &item.ty,
                (None, ExtraItems::Closed) => {
                    // If `got` is closed, it definitely doesn't have this item, so we can skip it.
                    return Ok(());
                }
                (None, ExtraItems::Default) => {
                    // A subclass of `got` could have this item with an incompatible type.
                    return Err(SubsetError::OpenTypedDict(Box::new(
                        OpenTypedDictSubsetError::MissingField {
                            got: got.name().clone(),
                            want: want.name().clone(),
                            field: k.clone(),
                        },
                    )));
                }
            };
            self.is_subset_partial_typed_dict_field(got_ty, want_v)
        })?;
        all(got_fields.iter(), |(k, got_v)| {
            if want_fields.contains_key(k) {
                Ok(())
            } else {
                self.is_subset_partial_typed_dict_field(
                    &got_v.ty,
                    &self.typed_dict_extra_items_field(want_extra_items.clone()),
                )
            }
        })?;
        match (got_extra_items, want_extra_items) {
            (_, ExtraItems::Default) => {
                // When `want` is open, checking extra items is more likely to cause false positives than catch real errors.
                Ok(())
            }
            (ExtraItems::Default, want_extra_items) => Err(SubsetError::OpenTypedDict(Box::new(
                OpenTypedDictSubsetError::UnknownFields {
                    got: got.name().clone(),
                    want: want.name().clone(),
                    extra_items: want_extra_items.extra_item(self.type_order.stdlib()).ty,
                },
            ))),
            (got_extra_items, want_extra_items) => self.is_subset_eq(
                &got_extra_items.extra_item(self.type_order.stdlib()).ty,
                &want_extra_items.extra_item(self.type_order.stdlib()).ty,
            ),
        }
    }

    /// Implementation of subset equality for Type, other than Var.
    pub fn is_subset_eq_impl(&mut self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        if matches!(got, Type::Materialization) {
            return self
                .is_subset_eq_impl(&self.type_order.stdlib().object().clone().to_type(), want);
        } else if matches!(want, Type::Materialization) {
            return self.is_subset_eq_impl(got, &Type::never());
        }
        match (got, want) {
            (Type::Any(_), _) => {
                // Special case in Python, because we want `x: int = Any` to be valid,
                // as Any is more the lack of information, rather than the actual union it is modelled to be.
                Ok(())
            }
            (_, Type::Any(_)) => Ok(()),
            (Type::Never(_), _) => Ok(()),
            (_, Type::ClassType(want)) if want.is_builtin("object") => {
                Ok(()) // everything is an instance of `object`
            }
            (Type::Quantified(q), Type::Ellipsis) | (Type::Ellipsis, Type::Quantified(q))
                if q.kind() == QuantifiedKind::ParamSpec =>
            {
                Ok(())
            }
            // Given `A | B <: C | D` we must always split the LHS first, but a quantified might be hiding a LHS union in its bounds.
            // Given (Quantified(bounds = A | B), A | B), we need to examine the bound _before_ splitting up the RHS union.
            // But given (T@Quantified(bounds = ...), T | Something), we need to split the union.
            // Therefore try these quantified cases, but only pick them if they work.
            (Type::Quantified(q), u)
                if let Restriction::Bound(bound) = q.restriction()
                    && self.is_subset_eq(bound, u).is_ok() =>
            {
                Ok(())
            }
            (Type::Quantified(q), u)
                if let Restriction::Constraints(constraints) = q.restriction()
                    && constraints
                        .iter()
                        .all(|constraint| self.is_subset_eq(constraint, u).is_ok()) =>
            {
                Ok(())
            }
            (Type::Quantified(q), u @ Type::Tuple(_)) if q.is_type_var_tuple() => self
                .is_subset_eq(
                    &Type::unbounded_tuple(self.type_order.stdlib().object().clone().to_type()),
                    u,
                ),
            (Type::Quantified(q), Type::ClassType(cls))
                if q.is_type_var_tuple()
                    && let Some(want) = self.type_order.as_tuple_type(cls) =>
            {
                self.is_subset_eq(
                    &Type::unbounded_tuple(self.type_order.stdlib().object().clone().to_type()),
                    &want,
                )
            }
            (Type::Intersect(l), u) => any(l.0.iter(), |l| self.is_subset_eq(l, u)),
            (Type::Union(box Union { members: ls, .. }), u) => {
                all(ls.iter(), |l| self.is_subset_eq(l, u))
            }
            (t1, Type::Quantified(q)) => match q.restriction() {
                // This only works for constraints and not bounds, because a TypeVar must resolve to exactly one of its constraints.
                Restriction::Constraints(constraints) => all(constraints.iter(), |constraint| {
                    self.is_subset_eq(t1, constraint)
                }),
                _ => Err(SubsetError::Other),
            },
            (l, Type::Intersect(u)) => all(u.0.iter(), |u| self.is_subset_eq(l, u)),
            (l, Type::Union(box Union { members: us, .. })) => {
                // Check var and non-var elements separately, so that if we match a non-var, we
                // don't pin the vars.
                let (vars, nonvars): (Vec<_>, Vec<_>) = us.iter().partition(|u| {
                    let mut contains_var = false;
                    u.universe(&mut |t| contains_var |= matches!(t, Type::Var(_)));
                    contains_var
                });
                any(nonvars.iter(), |u| self.is_subset_eq(l, u))
                    .or_else(|_| any(vars.iter(), |u| self.is_subset_eq(l, u)))
            }
            (l, Type::Overload(overload)) => all(overload.signatures.iter(), |u| {
                self.is_subset_eq(l, &u.as_type())
            }),
            (Type::Quantified(q), u) if !q.restriction().is_restricted() => {
                self.is_subset_eq(&self.type_order.stdlib().object().clone().to_type(), u)
            }
            (Type::Module(_), Type::ClassType(cls)) if cls.has_qname("types", "ModuleType") => {
                Ok(())
            }
            (
                Type::Callable(_)
                | Type::Function(_)
                | Type::Overload(_)
                | Type::Forall(box Forall {
                    body: Forallable::Function(_) | Forallable::Callable(_),
                    ..
                }),
                Type::ClassType(cls),
            ) if cls.has_qname("types", "FunctionType") => Ok(()),
            (Type::BoundMethod(_), Type::ClassType(cls))
                if cls.has_qname("types", "MethodType") =>
            {
                Ok(())
            }
            (Type::Overload(overload), u) => any(overload.signatures.iter(), |l| {
                self.is_subset_eq(&l.as_type(), u)
            }),
            (Type::BoundMethod(method), Type::Callable(_) | Type::Function(_))
                if let Some(l_no_self) =
                    self.type_order.bind_boundmethod(method, &mut |got, want| {
                        self.is_subset_eq(got, want).is_ok()
                    }) =>
            {
                self.is_subset_eq(&l_no_self, want)
            }
            (Type::Callable(_) | Type::Function(_), Type::BoundMethod(method))
                if let Some(u_no_self) =
                    self.type_order.bind_boundmethod(method, &mut |got, want| {
                        self.is_subset_eq(got, want).is_ok()
                    }) =>
            {
                self.is_subset_eq(got, &u_no_self)
            }
            (Type::BoundMethod(l), Type::BoundMethod(u))
                if let Some(l_no_self) = self
                    .type_order
                    .bind_boundmethod(l, &mut |got, want| self.is_subset_eq(got, want).is_ok())
                    && let Some(u_no_self) =
                        self.type_order.bind_boundmethod(u, &mut |got, want| {
                            self.is_subset_eq(got, want).is_ok()
                        }) =>
            {
                self.is_subset_eq(&l_no_self, &u_no_self)
            }
            (
                Type::Callable(box l)
                | Type::Function(box Function {
                    signature: l,
                    metadata: _,
                }),
                Type::Callable(box u)
                | Type::Function(box Function {
                    signature: u,
                    metadata: _,
                }),
            ) => {
                self.is_subset_params(&l.params, &u.params)?;
                self.is_subset_eq(&l.ret, &u.ret)
            }
            (Type::TypedDict(got), Type::TypedDict(want)) => self.is_subset_typed_dict(got, want),
            (Type::TypedDict(got), Type::PartialTypedDict(want)) => {
                self.is_subset_partial_typed_dict(got, want)
            }
            (Type::TypedDict(TypedDict::TypedDict(_)), Type::SelfType(cls))
                if cls == self.type_order.stdlib().typed_dict_fallback() =>
            {
                // Allow substituting a TypedDict for Self when we call methods
                Ok(())
            }
            (Type::TypedDict(td), _) => {
                let stdlib = self.type_order.stdlib();
                if let Some(value_type) = self
                    .type_order
                    .get_typed_dict_value_type_as_builtins_dict(td)
                {
                    self.is_subset_eq(
                        &stdlib
                            .dict(stdlib.str().clone().to_type(), value_type)
                            .to_type(),
                        want,
                    )
                } else if matches!(td, TypedDict::Anonymous(_)) {
                    self.is_subset_eq(
                        &stdlib
                            .dict(
                                stdlib.str().clone().to_type(),
                                self.type_order.get_typed_dict_value_type(td),
                            )
                            .to_type(),
                        want,
                    )
                } else {
                    self.is_subset_eq(
                        &stdlib
                            .mapping(
                                stdlib.str().clone().to_type(),
                                self.type_order.get_typed_dict_value_type(td),
                            )
                            .to_type(),
                        want,
                    )
                }
            }
            (Type::Kwargs(_), _) => {
                // We know kwargs will always be a dict w/ str keys
                self.is_subset_eq(
                    &self
                        .type_order
                        .stdlib()
                        .param_spec_kwargs_as_dict()
                        .to_type(),
                    want,
                )
            }
            (Type::Args(_), _) => {
                // We know args will always be a tuple
                self.is_subset_eq(
                    &self
                        .type_order
                        .stdlib()
                        .param_spec_args_as_tuple()
                        .to_type(),
                    want,
                )
            }
            (Type::ClassType(ty), _) | (_, Type::ClassType(ty))
                if self.type_order.extends_any(ty.class_object()) =>
            {
                Ok(())
            }
            (Type::ClassType(got), Type::ClassType(want))
                if want.is_builtin("float")
                    && (got.is_builtin("int") || got.is_builtin("bool")) =>
            {
                Ok(())
            }
            (Type::ClassType(got), Type::ClassType(want))
                if want.is_builtin("complex")
                    && (got.is_builtin("int")
                        || got.is_builtin("float")
                        || got.is_builtin("bool")) =>
            {
                Ok(())
            }
            (Type::ClassType(got), Type::ClassType(want)) => {
                let got_is_protocol = self.type_order.is_protocol(got.class_object());
                let want_is_protocol = self.type_order.is_protocol(want.class_object());
                if got_is_protocol && !want_is_protocol {
                    // Protocols are never assignable to concrete types
                    return Err(SubsetError::Other);
                }
                match self.type_order.as_superclass(got, want.class_object()) {
                    Some(got) => self.check_targs(&got, want),
                    // Structural checking for assigning to protocols
                    None if want_is_protocol => {
                        self.is_subset_protocol(got.clone().to_type(), want.clone())
                    }
                    _ => Err(SubsetError::Other),
                }
            }
            (Type::LiteralString | Type::Literal(Lit::Str(_)), Type::ClassType(want))
                if want.has_qname("typing", "Container")
                    || want.has_qname("typing", "Collection") =>
            {
                // The signature of `typing.Container.__contains__` is weird.
                // `str` matches it by direct inheritance, but we cannot convert `LiteralString` to `str`
                // otherwise it would be difficult to match protocols like `Interface[LiteralString]`
                //
                // https://github.com/python/typeshed/blob/5c8b7fcbbeb4af2d7e9f33e745a7863e401c2578/stdlib/typing.pyi#L638
                Ok(())
            }
            (_, Type::ClassType(want)) if self.type_order.is_protocol(want.class_object()) => {
                self.is_subset_protocol(got.clone(), want.clone())
            }
            // Protocols/classes that define __call__
            (
                Type::ClassType(got),
                Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_),
            ) if let Some(call_ty) = self.type_order.instance_as_dunder_call(got) => {
                self.is_subset_eq(&call_ty, want)
            }
            // Constructors as callables
            (
                Type::Type(box Type::ClassType(got)),
                Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_),
            ) => self.is_subset_eq(&self.type_order.constructor_to_callable(got), want),
            (Type::ClassDef(got), Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_)) => {
                self.is_subset_eq(
                    &Type::type_form(self.type_order.promote_silently(got)),
                    want,
                )
            }
            (Type::ClassDef(got), Type::ClassDef(want)) => ok_or(
                self.type_order.has_superclass(got, want),
                SubsetError::Other,
            ),
            // Although the object created by a NewType call behaves like a class for type-checking
            // purposes, it isn't one at runtime, so don't allow it to match `type`.
            (Type::ClassDef(got), Type::Type(_)) if self.type_order.is_new_type(got) => {
                Err(SubsetError::Other)
            }
            (Type::ClassDef(got), Type::ClassType(want))
                if self.type_order.is_new_type(got) && want.is_builtin("type") =>
            {
                Err(SubsetError::Other)
            }
            (Type::ClassDef(got), Type::Type(box Type::ClassType(want)))
                if self.type_order.is_protocol(want.class_object())
                    && self.type_order.is_protocol(got) =>
            {
                // We only allow concrete class names to be assigned to `type[T]` if `T` is a protocol
                Err(SubsetError::TypeOfProtocolNeedsConcreteClass(
                    want.name().clone(),
                ))
            }
            (Type::ClassDef(got), Type::Type(want)) => {
                self.is_subset_eq(&self.type_order.promote_silently(got), want)
            }
            (Type::Type(box Type::ClassType(got)), Type::ClassDef(want)) => ok_or(
                self.type_order.has_superclass(got.class_object(), want),
                SubsetError::Other,
            ),
            (Type::ClassDef(got), Type::ClassType(want)) => {
                ok_or(self.type_order.has_metaclass(got, want), SubsetError::Other)
            }
            (Type::Type(box Type::ClassType(got)), Type::ClassType(want)) => ok_or(
                self.type_order.has_metaclass(got.class_object(), want),
                SubsetError::Other,
            ),
            (Type::Type(box Type::Any(_)), Type::ClassDef(_)) => Ok(()),
            (Type::ClassType(cls), want @ Type::Tuple(_))
                if let Some(got) = self.type_order.as_tuple_type(cls) =>
            {
                self.is_subset_eq(&got, want)
            }
            (Type::ClassType(got), Type::SelfType(want)) => ok_or(
                self.type_order
                    .has_superclass(got.class_object(), want.class_object()),
                SubsetError::Other,
            ),
            (Type::Type(box Type::ClassType(got)), Type::SelfType(want)) => ok_or(
                self.type_order.has_metaclass(got.class_object(), want),
                SubsetError::Other,
            ),
            (Type::SelfType(_), Type::SelfType(_)) => Ok(()),
            (Type::SelfType(got), _) => self.is_subset_eq(&Type::ClassType(got.clone()), want),
            (Type::Tuple(l), Type::Tuple(u)) => self.is_subset_tuple(l, u),
            (Type::Tuple(Tuple::Concrete(left_elts)), _) => {
                let tuple_type = self
                    .type_order
                    .stdlib()
                    .tuple(unions(left_elts.clone()))
                    .to_type();
                self.is_subset_eq(&tuple_type, want)
            }
            (Type::Tuple(Tuple::Unbounded(left_elt)), _) => {
                let tuple_type = self
                    .type_order
                    .stdlib()
                    .tuple((**left_elt).clone())
                    .to_type();
                self.is_subset_eq(&tuple_type, want)
            }
            (
                Type::Tuple(Tuple::Unpacked(box (
                    prefix,
                    Type::Tuple(Tuple::Unbounded(middle)),
                    suffix,
                ))),
                _,
            ) => {
                let elts = prefix
                    .iter()
                    .chain(iter::once(&**middle))
                    .chain(suffix)
                    .cloned()
                    .collect::<Vec<_>>();
                let tuple_type = self.type_order.stdlib().tuple(unions(elts)).to_type();
                self.is_subset_eq(&tuple_type, want)
            }
            (Type::Tuple(Tuple::Unpacked(box (prefix, middle, suffix))), _) => {
                let elts = prefix.iter().chain(suffix).cloned().collect::<Vec<_>>();
                let tuple_type = self.type_order.stdlib().tuple(unions(elts)).to_type();
                self.is_subset_eq(&tuple_type, want)?;
                self.is_subset_eq(middle, want)?;
                Ok(())
            }
            (Type::Literal(lit), Type::LiteralString) => ok_or(lit.is_string(), SubsetError::Other),
            (Type::Literal(lit), t @ Type::ClassType(_)) => self.is_subset_eq(
                &lit.general_class_type(self.type_order.stdlib())
                    .clone()
                    .to_type(),
                t,
            ),
            (Type::Literal(l_lit), Type::Literal(u_lit)) => {
                ok_or(l_lit == u_lit, SubsetError::Other)
            }
            (_, Type::SelfType(cls))
                if got.is_literal_string() && cls == self.type_order.stdlib().str() =>
            {
                Ok(())
            }
            (Type::LiteralString, _) => {
                self.is_subset_eq(&self.type_order.stdlib().str().clone().to_type(), want)
            }
            (Type::Type(l), Type::Type(u)) => self.is_subset_eq(l, u),
            (Type::Type(_), _) => self.is_subset_eq(
                &self.type_order.stdlib().builtins_type().clone().to_type(),
                want,
            ),
            (
                Type::ClassType(class),
                Type::Type(_)
                | Type::ClassDef(_)
                | Type::BoundMethod(_)
                | Type::Callable(_)
                | Type::Function(_),
            ) => {
                let type_type = self.type_order.stdlib().builtins_type();
                if class == type_type {
                    // Unparameterized `type` is equivalent to `type[Any]`
                    Ok(())
                } else if let Some(got_as_type) = self
                    .type_order
                    .as_superclass(class, type_type.class_object())
                    && got_as_type.targs().is_empty()
                {
                    // The class extends unparameterized `type`
                    Ok(())
                } else {
                    Err(SubsetError::Other)
                }
            }
            (Type::TypeGuard(l), Type::TypeGuard(u)) => {
                // TypeGuard is covariant
                self.is_subset_eq(l, u)
            }
            (Type::TypeGuard(_) | Type::TypeIs(_), _) => {
                self.is_subset_eq(&self.type_order.stdlib().bool().clone().to_type(), want)
            }
            (Type::Ellipsis, Type::ParamSpecValue(_) | Type::Concatenate(_, _))
            | (Type::ParamSpecValue(_) | Type::Concatenate(_, _), Type::Ellipsis) => Ok(()),
            (Type::ParamSpecValue(ls), Type::ParamSpecValue(us)) => {
                self.is_subset_param_list(ls.items(), us.items())
            }
            (Type::ParamSpecValue(ls), Type::Concatenate(us, u_pspec)) => {
                self.is_paramlist_subset_of_paramspec(ls, us, u_pspec)
            }
            (Type::Concatenate(ls, l_pspec), Type::ParamSpecValue(us)) => {
                self.is_paramspec_subset_of_paramlist(ls, l_pspec, us)
            }
            (Type::Concatenate(ls, l_pspec), Type::Concatenate(us, u_pspec)) => {
                self.is_paramspec_subset_of_paramspec(ls, l_pspec, us, u_pspec)
            }
            (Type::Ellipsis, _)
                if let Some(ellipsis) = self.type_order.stdlib().ellipsis_type() =>
            {
                // Bit of a weird case - pretty sure we should be modelling these slightly differently
                // - probably not as a dedicated Type alternative.
                self.is_subset_eq(&ellipsis.clone().to_type(), want)
            }
            (Type::None, _) => self.is_subset_eq(
                &self.type_order.stdlib().none_type().clone().to_type(),
                want,
            ),
            (_, Type::None) => {
                self.is_subset_eq(got, &self.type_order.stdlib().none_type().clone().to_type())
            }
            (Type::Forall(forall), _) => {
                // Finalizing the quantified vars returns instantiation errors
                let (vs, got) = self.type_order.instantiate_fresh_forall((**forall).clone());
                let result = self.is_subset_eq(&got, want);
                result.and(
                    self.finish_quantified(vs)
                        .map_err(SubsetError::TypeVarSpecialization),
                )
            }
            (_, Type::Forall(forall)) => self.is_subset_eq(got, &forall.body.clone().as_type()),
            (Type::TypeAlias(ta), _) => {
                self.is_subset_eq(&ta.as_value(self.type_order.stdlib()), want)
            }
            _ => Err(SubsetError::Other),
        }
    }

    fn check_targs(
        &mut self,
        got_class: &ClassType,
        want_class: &ClassType,
    ) -> Result<(), SubsetError> {
        let got = got_class.targs();
        let want = want_class.targs();
        let params = want_class.tparams();
        let got = got.as_slice();
        let want = want.as_slice();
        if !(got.len() == want.len() && want.len() == params.len()) {
            // This state should be impossible in static code, but during an
            // incremental update it's possible to get two `Class` values that are
            // not the same because they come from different states of the codebase,
            // and yet compare as equal. We need to treat them as not the same type here
            // to avoid arity mismatches later.
            //
            // TODO(stroxler): Find a way to write a test that crashes if we try to assert here;
            // having a test setup to stress what happens on code change will help us make
            // Pyrefly incremental more robust.
        }

        let variances = self
            .type_order
            .get_variance_from_class(got_class.class_object());

        for (got_arg, want_arg, param) in izip!(got, want, params.iter()) {
            if param.quantified.kind() == QuantifiedKind::TypeVarTuple {
                self.is_equal(got_arg, want_arg)?;
            } else {
                match variances.get(param.name()) {
                    Variance::Covariant => self.is_subset_eq(got_arg, want_arg)?,
                    Variance::Contravariant => self.is_subset_eq(want_arg, got_arg)?,
                    // Technically, the right thing to do for bivariance would be to skip the
                    // subset check. However, this leads to confusing and unintuitive behavior,
                    // so we treat bivariant type parameters as invariant instead.
                    Variance::Invariant | Variance::Bivariant => {
                        self.is_equal(got_arg, want_arg)?
                    }
                }
            }
        }
        Ok(())
    }
}
