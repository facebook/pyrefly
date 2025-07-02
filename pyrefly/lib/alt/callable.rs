/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use pyrefly_util::display::count;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::expr::TypeOrExpr;
use crate::alt::solve::Iterable;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::display::function_suffix;
use crate::error::kind::ErrorKind;
use crate::types::callable::Callable;
use crate::types::callable::FuncId;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::quantified::Quantified;
use crate::types::tuple::Tuple;
use crate::types::types::Type;
use crate::types::types::Var;

/// Structure to turn TypeOrExprs into Types.
/// This is used to avoid re-infering types for arguments multiple types.
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

    pub fn opt_call_arg<'a, 'b: 'a, Ans: LookupAnswer>(
        &'a self,
        x: Option<&CallArg<'b>>,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> Option<CallArg<'a>> {
        x.map(|x| self.call_arg(x, solver, errors))
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

    // Splat arguments might be fixed-length tuples, which are handled precisely, or have unknown
    // length. This function evaluates splat args to determine how many params should be consumed,
    // but does not evaluate other expressions, which might be contextually typed.
    fn pre_eval<Ans: LookupAnswer>(
        &self,
        solver: &AnswersSolver<Ans>,
        arg_errors: &ErrorCollector,
    ) -> CallArgPreEval {
        match self {
            Self::Arg(TypeOrExpr::Type(ty, _)) => CallArgPreEval::Type(ty, false),
            Self::Arg(TypeOrExpr::Expr(e)) => CallArgPreEval::Expr(e, false),
            Self::Star(e, range) => {
                let ty = e.infer(solver, arg_errors);
                let iterables = solver.iterate(&ty, *range, arg_errors);
                // If we have a union of iterables, use a fixed length only if every iterable is
                // fixed and has the same length. Otherwise, use star.
                let mut fixed_lens = Vec::new();
                for x in iterables.iter() {
                    match x {
                        Iterable::FixedLen(xs) => fixed_lens.push(xs.len()),
                        Iterable::OfType(_) => {}
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
                    let mut star_tys = Vec::new();
                    for x in iterables {
                        match x {
                            Iterable::OfType(ty) => star_tys.push(ty.clone()),
                            Iterable::FixedLen(tys) => star_tys.extend(tys),
                        }
                    }
                    let ty = solver.unions(star_tys);
                    CallArgPreEval::Star(ty, false)
                }
            }
        }
    }
}

// Pre-evaluated args are iterable. Type/Expr/Star variants iterate once (tracked via bool field),
// Fixed variant iterates over the the vec (tracked via usize field).
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

    fn post_check<Ans: LookupAnswer>(
        &mut self,
        solver: &AnswersSolver<Ans>,
        callable_name: Option<&FuncId>,
        hint: &Type,
        param_name: Option<&Name>,
        vararg: bool,
        range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        let tcc = &|| TypeCheckContext {
            kind: if vararg {
                TypeCheckKind::CallVarArgs(false, param_name.cloned(), callable_name.cloned())
            } else {
                TypeCheckKind::CallArgument(param_name.cloned(), callable_name.cloned())
            },
            context: context.map(|ctx| ctx()),
        };
        match self {
            Self::Type(ty, done) => {
                *done = true;
                solver.check_type(hint, ty, range, call_errors, tcc);
            }
            Self::Expr(x, done) => {
                *done = true;
                solver.expr_with_separate_check_errors(
                    x,
                    Some((hint, tcc, call_errors)),
                    arg_errors,
                );
            }
            Self::Star(ty, done) => {
                *done = vararg;
                solver.check_type(hint, ty, range, call_errors, tcc);
            }
            Self::Fixed(tys, i) => {
                solver.check_type(hint, &tys[*i], range, call_errors, tcc);
                *i += 1;
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

/// Helps track matching of arguments against positional parameters in AnswersSolver::callable_infer_params.
#[derive(PartialEq, Eq)]
enum PosParamKind {
    PositionalOnly,
    Positional,
    Unpacked,
    Variadic,
}

/// Helps track matching of arguments against positional parameters in AnswersSolver::callable_infer_params.
struct PosParam {
    ty: Type,
    name: Option<Name>,
    kind: PosParamKind,
}

impl PosParam {
    fn new(p: &Param) -> Option<Self> {
        match p {
            Param::PosOnly(name, ty, _required) => Some(Self {
                ty: ty.clone(),
                name: name.clone(),
                kind: PosParamKind::PositionalOnly,
            }),
            Param::Pos(name, ty, _required) => Some(Self {
                ty: ty.clone(),
                name: Some(name.clone()),
                kind: PosParamKind::Positional,
            }),
            Param::VarArg(name, Type::Unpack(ty)) => Some(Self {
                ty: (**ty).clone(),
                name: name.clone(),
                kind: PosParamKind::Unpacked,
            }),
            Param::VarArg(name, ty) => Some(Self {
                ty: ty.clone(),
                name: name.clone(),
                kind: PosParamKind::Variadic,
            }),
            Param::KwOnly(..) | Param::Kwargs(..) => None,
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn is_param_spec_args(&self, x: &CallArg, q: Quantified, errors: &ErrorCollector) -> bool {
        match x {
            CallArg::Star(x, _) => {
                let mut ty = x.infer(self, errors);
                self.expand_type_mut(&mut ty);
                ty == Type::Args(q)
            }
            _ => false,
        }
    }

    fn is_param_spec_kwargs(
        &self,
        x: &CallKeyword,
        q: Quantified,
        errors: &ErrorCollector,
    ) -> bool {
        let mut ty = x.value.infer(self, errors);
        self.expand_type_mut(&mut ty);
        ty == Type::Kwargs(q)
    }

    // See comment on `callable_infer` about `arg_errors` and `call_errors`.
    fn callable_infer_params(
        &self,
        callable_name: Option<FuncId>,
        params: &ParamList,
        // A ParamSpec Var (if any) that comes at the end of the parameter list.
        // See test::paramspec::test_paramspec_twice for an example of this.
        mut paramspec: Option<Var>,
        self_arg: Option<CallArg>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        let error = |errors, range, kind, msg: String| {
            self.error(
                errors,
                range,
                kind,
                context,
                format!(
                    "{}{}",
                    msg,
                    function_suffix(callable_name.as_ref(), self.module_info().name())
                ),
            )
        };
        let iargs = self_arg.iter().chain(args.iter());
        // Creates a reversed copy of the parameters that we iterate through from back to front,
        // so that we can easily peek at and pop from the end.
        let mut rparams = params.items().iter().cloned().rev().collect::<Vec<_>>();
        let mut num_positional_params = 0;
        let mut num_positional_args = 0;
        let mut seen_names = SmallMap::new();
        let mut extra_arg_pos = None;
        let mut unpacked_vararg = None;
        let mut unpacked_vararg_matched_args = Vec::new();
        let var_to_rparams = |var| {
            let ps = match self.solver().force_var(var) {
                Type::ParamSpecValue(ps) => ps,
                Type::Any(_) | Type::Ellipsis => ParamList::everything(),
                Type::Concatenate(prefix, _) => {
                    // TODO: handle second component of Type::Concatenate
                    let ps = ParamList::everything();
                    ps.prepend_types(&prefix).into_owned()
                }
                t => {
                    error(
                        call_errors,
                        range,
                        ErrorKind::BadArgumentType,
                        format!("Expected `{}` to be a ParamSpec value", self.for_display(t)),
                    );
                    ParamList::everything()
                }
            };
            ps.items().iter().cloned().rev().collect()
        };
        for arg in iargs {
            let mut arg_pre = arg.pre_eval(self, arg_errors);
            while arg_pre.step() {
                num_positional_args += 1;
                let param = if let Some(p) = rparams.last() {
                    PosParam::new(p)
                } else if let Some(var) = paramspec {
                    // We've run out of parameters but haven't finished matching arguments. If we
                    // have a ParamSpec Var, it may contribute more parameters; force it and tack
                    // the result onto the parameter list.
                    rparams = var_to_rparams(var);
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
                        num_positional_params += 1;
                        rparams.pop();
                        if let Some(name) = &name
                            && kind == PosParamKind::Positional
                        {
                            // Remember names of positional parameters to detect duplicates.
                            // We ignore positional-only parameters because they can't be passed in by name.
                            seen_names.insert(name.clone(), ty.clone());
                        }
                        arg_pre.post_check(
                            self,
                            callable_name.as_ref(),
                            &ty,
                            name.as_ref(),
                            false,
                            arg.range(),
                            arg_errors,
                            call_errors,
                            context,
                        )
                    }
                    Some(PosParam {
                        ty,
                        name,
                        kind: PosParamKind::Unpacked,
                    }) => {
                        // Store args that get matched to an unpacked *args param
                        // Matched args are typechecked separately later
                        unpacked_vararg = Some((name, ty));
                        unpacked_vararg_matched_args.push(arg_pre.clone());
                        arg_pre.post_skip();
                    }
                    Some(PosParam {
                        ty,
                        name,
                        kind: PosParamKind::Variadic,
                    }) => arg_pre.post_check(
                        self,
                        callable_name.as_ref(),
                        &ty,
                        name.as_ref(),
                        true,
                        arg.range(),
                        arg_errors,
                        call_errors,
                        context,
                    ),
                    None => {
                        arg_pre.post_infer(self, arg_errors);
                        if arg_pre.is_star() {
                            num_positional_args -= 1;
                        }
                        if extra_arg_pos.is_none() && !arg_pre.is_star() {
                            extra_arg_pos = Some(arg.range());
                        }
                        break;
                    }
                }
            }
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
            let unpacked_args_ty = match middle.as_slice() {
                [] => Type::tuple(prefix),
                [middle] => Type::Tuple(Tuple::unpacked(
                    prefix,
                    Type::Tuple(Tuple::unbounded(middle.clone())),
                    suffix,
                )),
                _ => Type::Tuple(Tuple::unpacked(
                    prefix,
                    Type::Tuple(Tuple::Unbounded(Box::new(self.unions(middle)))),
                    suffix,
                )),
            };
            self.check_type(
                &unpacked_param_ty,
                &unpacked_args_ty,
                range,
                arg_errors,
                &|| TypeCheckContext {
                    kind: TypeCheckKind::CallVarArgs(
                        true,
                        unpacked_name.clone(),
                        callable_name.clone(),
                    ),
                    context: context.map(|ctx| ctx()),
                },
            );
        }
        if let Some(arg_range) = extra_arg_pos {
            let (expected, actual) = if self_arg.is_none() {
                (
                    count(num_positional_params as usize, "positional argument"),
                    num_positional_args.to_string(),
                )
            } else if num_positional_params < 1 {
                (
                    "0 positional arguments".to_owned(),
                    format!("{} (including implicit `self`)", num_positional_args),
                )
            } else {
                (
                    count(num_positional_params as usize - 1, "positional argument"),
                    (num_positional_args - 1).to_string(),
                )
            };
            error(
                call_errors,
                arg_range,
                ErrorKind::BadArgumentType,
                format!("Expected {expected}, got {actual}"),
            );
        }
        // Missing positional-only arguments, split by whether the corresponding parameters
        // in the callable have names. E.g., functions declared with `def` have named posonly
        // parameters and `typing.Callable`s have unnamed ones.
        let mut missing_unnamed_posonly = 0;
        let mut missing_named_posonly = SmallSet::new();
        let mut kwparams = SmallMap::new();
        let mut kwargs = None;
        let mut kwargs_is_unpack = false;
        loop {
            let p = match rparams.pop() {
                Some(p) => p,
                None if let Some(var) = paramspec => {
                    // We've reached the end of our regular parameter list. Now check if we have more parameters from a ParamSpec.
                    rparams = var_to_rparams(var);
                    paramspec = None;
                    continue;
                }
                None => {
                    break;
                }
            };
            match p {
                Param::PosOnly(name, _, required) => {
                    if required == Required::Required {
                        if let Some(name) = name {
                            missing_named_posonly.insert(name);
                        } else {
                            missing_unnamed_posonly += 1;
                        }
                    }
                }
                Param::VarArg(..) => {}
                Param::Pos(name, ty, required) | Param::KwOnly(name, ty, required) => {
                    kwparams.insert(name.clone(), (ty, required == Required::Required));
                }
                Param::Kwargs(_, Type::Unpack(box Type::TypedDict(typed_dict))) => {
                    self.typed_dict_fields(&typed_dict)
                        .into_iter_hashed()
                        .for_each(|(name, field)| {
                            kwparams.insert_hashed(name, (field.ty, field.required));
                        });
                    kwargs_is_unpack = true;
                }
                Param::Kwargs(name, ty) => {
                    kwargs = Some((name, ty));
                }
            }
        }
        let mut unexpected_keyword_error = |name: &Name, range| {
            if missing_named_posonly.shift_remove(name) {
                error(
                    call_errors,
                    range,
                    ErrorKind::UnexpectedKeyword,
                    format!("Expected argument `{}` to be positional", name),
                );
            } else {
                error(
                    call_errors,
                    range,
                    ErrorKind::UnexpectedKeyword,
                    format!("Unexpected keyword argument `{}`", name),
                );
            }
        };
        let mut splat_kwargs = Vec::new();
        for kw in keywords {
            match kw.arg {
                None => {
                    let ty = kw.value.infer(self, arg_errors);
                    if let Type::TypedDict(typed_dict) = ty {
                        for (name, field) in self.typed_dict_fields(&typed_dict).iter() {
                            let mut hint = kwargs.as_ref().map(|(_, ty)| ty.clone());
                            if let Some(ty) = seen_names.get(name) {
                                error(
                                    call_errors,
                                    kw.range,
                                    ErrorKind::BadKeywordArgument,
                                    format!("Multiple values for argument `{}`", name),
                                );
                                hint = Some(ty.clone());
                            } else if let Some((ty, required)) = kwparams.get(name) {
                                seen_names.insert(name.clone(), ty.clone());
                                if *required && !field.required {
                                    error(
                                        call_errors,
                                        kw.range,
                                        ErrorKind::MissingArgument,
                                        format!("Expected key `{}` to be required", name),
                                    );
                                }
                                hint = Some(ty.clone())
                            } else if kwargs.is_none() && !kwargs_is_unpack {
                                unexpected_keyword_error(name, kw.range);
                            }
                            if let Some(want) = &hint {
                                self.check_type(want, &field.ty, kw.range, call_errors, &|| {
                                    TypeCheckContext {
                                        kind: TypeCheckKind::CallArgument(
                                            Some(name.clone()),
                                            callable_name.clone(),
                                        ),
                                        context: context.map(|ctx| ctx()),
                                    }
                                });
                            }
                        }
                    } else {
                        match self.unwrap_mapping(&ty) {
                            Some((key, value)) => {
                                if self.is_subset_eq(&key, &self.stdlib.str().clone().to_type()) {
                                    if let Some((name, want)) = kwargs.as_ref() {
                                        self.check_type(
                                            want,
                                            &value,
                                            kw.range,
                                            call_errors,
                                            &|| TypeCheckContext {
                                                kind: TypeCheckKind::CallKwArgs(
                                                    None,
                                                    name.clone(),
                                                    callable_name.clone(),
                                                ),
                                                context: context.map(|ctx| ctx()),
                                            },
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
                    let mut hint = kwargs.as_ref().map(|(_, ty)| ty.clone());
                    let mut has_matching_param = false;
                    if let Some(ty) = seen_names.get(&id.id) {
                        error(
                            call_errors,
                            kw.range,
                            ErrorKind::BadKeywordArgument,
                            format!("Multiple values for argument `{}`", id.id),
                        );
                        hint = Some(ty.clone());
                        has_matching_param = true;
                    } else if let Some((ty, _)) = kwparams.get(&id.id) {
                        seen_names.insert(id.id.clone(), ty.clone());
                        hint = Some(ty.clone());
                        has_matching_param = true;
                    } else if kwargs.is_none() {
                        unexpected_keyword_error(&id.id, id.range);
                    }
                    let tcc: &dyn Fn() -> TypeCheckContext = &|| TypeCheckContext {
                        kind: if has_matching_param {
                            TypeCheckKind::CallArgument(Some(id.id.clone()), callable_name.clone())
                        } else {
                            TypeCheckKind::CallKwArgs(
                                Some(id.id.clone()),
                                kwargs.as_ref().and_then(|(name, _)| name.clone()),
                                callable_name.clone(),
                            )
                        },
                        context: context.map(|ctx| ctx()),
                    };
                    match kw.value {
                        TypeOrExpr::Expr(x) => {
                            self.expr_with_separate_check_errors(
                                x,
                                hint.as_ref().map(|ty| (ty, tcc, call_errors)),
                                arg_errors,
                            );
                        }
                        TypeOrExpr::Type(x, range) => {
                            if let Some(hint) = &hint
                                && !hint.is_any()
                            {
                                self.check_type(hint, x, range, call_errors, tcc);
                            }
                        }
                    }
                }
            }
        }
        if missing_unnamed_posonly > 0 || !missing_named_posonly.is_empty() {
            let range = keywords.first().map_or(range, |kw| kw.range);
            let msg = if missing_unnamed_posonly == 0 {
                format!(
                    "Missing positional argument{} {}",
                    if missing_named_posonly.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
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
        for (name, (want, required)) in kwparams.iter() {
            if !seen_names.contains_key(name) {
                if splat_kwargs.is_empty() && *required {
                    error(
                        call_errors,
                        range,
                        ErrorKind::MissingArgument,
                        format!("Missing argument `{}`", name),
                    );
                }
                for (ty, range) in &splat_kwargs {
                    self.check_type(want, ty, *range, call_errors, &|| TypeCheckContext {
                        kind: TypeCheckKind::CallUnpackKwArg(name.clone(), callable_name.clone()),
                        context: context.map(|ctx| ctx()),
                    });
                }
            }
        }
    }

    // Call a function with the given arguments. The arguments are contextually typed, if possible.
    // We pass two error collectors into this function:
    // * arg_errors is used to infer the types of arguments, before passing them to the function.
    // * call_errors is used for (1) call signature matching, e.g. arity issues and (2) checking the
    //   types of arguments against the types of parameters.
    // Callers can pass the same error collector for both, and most callers do. We use two collectors
    // for overload matching.
    pub fn callable_infer(
        &self,
        callable: Callable,
        callable_name: Option<FuncId>,
        self_arg: Option<CallArg>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        arg_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        match callable.params {
            Params::List(params) => {
                self.callable_infer_params(
                    callable_name,
                    &params,
                    None,
                    self_arg,
                    args,
                    keywords,
                    range,
                    arg_errors,
                    call_errors,
                    context,
                );
            }
            Params::Ellipsis => {
                // Deal with Callable[..., R]
                for arg in self_arg.iter().chain(args.iter()) {
                    arg.pre_eval(self, arg_errors).post_infer(self, arg_errors)
                }
            }
            Params::ParamSpec(concatenate, p) => {
                let p = self.solver().expand(p);
                match p {
                    Type::ParamSpecValue(params) => self.callable_infer_params(
                        callable_name,
                        &params.prepend_types(&concatenate),
                        None,
                        self_arg,
                        args,
                        keywords,
                        range,
                        arg_errors,
                        call_errors,
                        context,
                    ),
                    // This can happen with a signature like `(f: Callable[P, None], *args: P.args, **kwargs: P.kwargs)`.
                    // Before we match an argument to `f`, we don't know what `P` is, so we don't have an answer for the Var yet.
                    Type::Var(var) => self.callable_infer_params(
                        callable_name,
                        &ParamList::new_types(&concatenate),
                        Some(var),
                        self_arg,
                        args,
                        keywords,
                        range,
                        arg_errors,
                        call_errors,
                        context,
                    ),
                    Type::Quantified(q) => {
                        if !args
                            .last()
                            .is_some_and(|x| self.is_param_spec_args(x, q.clone(), arg_errors))
                            || !keywords.last().is_some_and(|x| {
                                self.is_param_spec_kwargs(x, q.clone(), arg_errors)
                            })
                        {
                            self.error(
                                call_errors,
                                range,
                                ErrorKind::InvalidParamSpec,
                                context,
                                format!(
                                    "Expected *-unpacked {}.args and **-unpacked {}.kwargs",
                                    q.name(),
                                    q.name()
                                ),
                            );
                        } else {
                            self.callable_infer_params(
                                callable_name,
                                &ParamList::new_types(&concatenate),
                                None,
                                self_arg,
                                &args[0..args.len() - 1],
                                &keywords[0..keywords.len() - 1],
                                range,
                                arg_errors,
                                call_errors,
                                context,
                            );
                        }
                    }
                    Type::Any(_) | Type::Ellipsis => {}
                    _ => {
                        // This could well be our error, but not really sure
                        self.error(
                            call_errors,
                            range,
                            ErrorKind::InvalidParamSpec,
                            context,
                            format!("Unexpected ParamSpec type: `{}`", self.for_display(p)),
                        );
                    }
                }
            }
        };
        self.solver().expand(callable.ret)
    }
}
