/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::dunder;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::TParams;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::callable::CallWithTypes;
use crate::alt::class::class_field::DescriptorBase;
use crate::alt::unwrap::HintRef;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::class::ClassType;
use crate::types::keywords::KwCall;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::type_var::Restriction;
use crate::types::typed_dict::TypedDict;
use crate::types::types::AnyStyle;
use crate::types::types::BoundMethod;
use crate::types::types::OverloadType;
use crate::types::types::Type;

pub enum CallStyle<'a> {
    Method(&'a Name),
    FreeForm,
}

/// A thing that can be called (see as_call_target and call_infer).
/// Note that a single "call" may invoke multiple functions under the hood,
/// e.g., `__new__` followed by `__init__` for Class.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallTarget {
    /// A typing.Callable.
    Callable(TargetWithTParams<Callable>),
    /// A function.
    Function(TargetWithTParams<Function>),
    /// Method of a class. The `Type` is the self/cls argument.
    BoundMethod(Type, TargetWithTParams<Function>),
    /// A class object.
    Class(ClassType),
    /// A TypedDict.
    TypedDict(TypedDict),
    /// An overloaded function.
    FunctionOverload(Vec1<TargetWithTParams<Function>>, FuncMetadata),
    /// An overloaded method.
    BoundMethodOverload(Type, Vec1<TargetWithTParams<Function>>, FuncMetadata),
    /// A union of call targets.
    Union(Vec<CallTarget>),
    /// Any, as a call target.
    Any(AnyStyle),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetWithTParams<T>(pub Option<Arc<TParams>>, pub T);

impl CallTarget {
    fn function_metadata(&self) -> Option<&FuncMetadata> {
        match self {
            Self::Function(func) | Self::BoundMethod(_, func) => Some(&func.1.metadata),
            Self::FunctionOverload(_, metadata) | Self::BoundMethodOverload(_, _, metadata) => {
                Some(metadata)
            }
            _ => None,
        }
    }
}

struct CalledOverload {
    signature: Callable,
    call_errors: ErrorCollector,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn error_call_target(
        &self,
        errors: &ErrorCollector,
        range: TextRange,
        msg: String,
        error_kind: ErrorKind,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> CallTarget {
        self.error(errors, range, ErrorInfo::new(error_kind, context), msg);
        CallTarget::Any(AnyStyle::Error)
    }

    pub fn as_call_target(&self, ty: Type) -> Option<CallTarget> {
        self.as_call_target_impl(ty, None)
    }

    fn as_call_target_impl(&self, ty: Type, quantified: Option<Quantified>) -> Option<CallTarget> {
        match ty {
            Type::Callable(c) => Some(CallTarget::Callable(TargetWithTParams(None, *c))),
            Type::Function(func) => Some(CallTarget::Function(TargetWithTParams(None, *func))),
            Type::Overload(overload) => {
                let funcs = overload.signatures.mapped(|ty| match ty {
                    OverloadType::Function(function) => TargetWithTParams(None, function),
                    OverloadType::Forall(forall) => {
                        TargetWithTParams(Some(forall.tparams), forall.body)
                    }
                });
                Some(CallTarget::FunctionOverload(funcs, *overload.metadata))
            }
            Type::BoundMethod(box BoundMethod { obj, func }) => {
                match self.as_call_target_impl(func.as_type(), quantified) {
                    Some(CallTarget::Function(func)) => Some(CallTarget::BoundMethod(obj, func)),
                    Some(CallTarget::FunctionOverload(overloads, meta)) => {
                        Some(CallTarget::BoundMethodOverload(obj, overloads, meta))
                    }
                    _ => None,
                }
            }
            Type::ClassDef(cls) => {
                self.as_call_target_impl(Type::type_form(self.instantiate(&cls)), quantified)
            }
            Type::Type(box Type::ClassType(cls)) | Type::Type(box Type::SelfType(cls)) => {
                Some(CallTarget::Class(cls))
            }
            Type::Type(box Type::Tuple(tuple)) => {
                Some(CallTarget::Class(self.erase_tuple_type(tuple)))
            }
            Type::Type(box Type::Quantified(quantified)) => {
                Some(CallTarget::Callable(TargetWithTParams(
                    None,
                    Callable {
                        // TODO: use upper bound to determine input parameters
                        params: Params::Ellipsis,
                        ret: Type::Quantified(quantified),
                    },
                )))
            }
            Type::Type(box Type::Any(style)) => Some(CallTarget::Any(style)),
            Type::Forall(forall) => {
                let mut target = self.as_call_target_impl(forall.body.as_type(), quantified);
                match &mut target {
                    Some(
                        CallTarget::Callable(TargetWithTParams(x, _))
                        | CallTarget::Function(TargetWithTParams(x, _)),
                    ) => {
                        *x = Some(forall.tparams);
                    }
                    _ => {}
                }
                target
            }
            Type::Var(v) if let Some(_guard) = self.recurser.recurse(v) => {
                self.as_call_target_impl(self.solver().force_var(v), quantified)
            }
            Type::Union(xs) => {
                let targets = xs
                    .into_iter()
                    .map(|x| self.as_call_target_impl(x, quantified.clone()))
                    .collect::<Option<Vec<_>>>()?;
                if targets.len() == 1 {
                    Some(targets.into_iter().next().unwrap())
                } else {
                    Some(CallTarget::Union(targets))
                }
            }
            Type::Any(style) => Some(CallTarget::Any(style)),
            Type::TypeAlias(ta) => self.as_call_target_impl(ta.as_value(self.stdlib), quantified),
            Type::ClassType(cls) | Type::SelfType(cls) => {
                if let Some(quantified) = quantified {
                    self.quantified_instance_as_dunder_call(quantified.clone(), &cls)
                        .and_then(|ty| self.as_call_target_impl(ty, Some(quantified)))
                } else {
                    self.instance_as_dunder_call(&cls)
                        .and_then(|ty| self.as_call_target_impl(ty, quantified))
                }
            }
            Type::Type(box Type::TypedDict(typed_dict)) => Some(CallTarget::TypedDict(typed_dict)),
            Type::Quantified(q) if q.is_type_var() => match q.restriction() {
                Restriction::Unrestricted => None,
                Restriction::Bound(bound) => match bound {
                    Type::Union(members) => {
                        let mut targets = Vec::new();
                        for member in members {
                            if let Some(target) = self.as_call_target_impl(
                                member.clone(),
                                Some(
                                    q.clone()
                                        .with_restriction(Restriction::Bound(member.clone())),
                                ),
                            ) {
                                targets.push(target);
                            } else {
                                return None;
                            }
                        }
                        Some(CallTarget::Union(targets))
                    }
                    _ => self.as_call_target_impl(bound.clone(), Some(*q)),
                },
                Restriction::Constraints(constraints) => {
                    let mut targets = Vec::new();
                    for constraint in constraints {
                        if let Some(target) = self.as_call_target_impl(
                            constraint.clone(),
                            Some(q.clone().with_restriction(Restriction::Constraints(vec![
                                constraint.clone(),
                            ]))),
                        ) {
                            targets.push(target);
                        } else {
                            return None;
                        }
                    }
                    Some(CallTarget::Union(targets))
                }
            },
            Type::KwCall(call) => self.as_call_target_impl(call.return_ty, quantified),
            _ => None,
        }
    }

    pub fn as_call_target_or_error(
        &self,
        ty: Type,
        call_style: CallStyle,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> CallTarget {
        match self.as_call_target(ty.clone()) {
            Some(target) => {
                let metadata = target.function_metadata();
                if let Some(m) = metadata
                    && m.flags.is_deprecated
                {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::Deprecated, context),
                        format!(
                            "`{}` is deprecated",
                            m.kind.as_func_id().format(self.module().name())
                        ),
                    );
                }
                target
            }
            None => {
                let expect_message = match call_style {
                    CallStyle::Method(method) => {
                        format!("Expected `{method}` to be a callable")
                    }
                    CallStyle::FreeForm => "Expected a callable".to_owned(),
                };
                self.error_call_target(
                    errors,
                    range,
                    format!("{}, got `{}`", expect_message, self.for_display(ty)),
                    ErrorKind::NotCallable,
                    context,
                )
            }
        }
    }

    fn make_call_target_and_call(
        &self,
        callee_ty: Type,
        method_name: &Name,
        range: TextRange,
        args: &[CallArg],
        keywords: &[CallKeyword],
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let call_target = self.as_call_target_or_error(
            callee_ty,
            CallStyle::Method(method_name),
            range,
            errors,
            context,
        );
        self.call_infer(
            call_target,
            args,
            keywords,
            range,
            errors,
            context,
            None,
            None,
        )
    }

    /// Calls a magic dunder method. If no attribute exists with the given method name, returns None without attempting the call.
    ///
    /// Note that this method is only expected to be used for magic dunder methods and is not expected to
    /// produce correct results for arbitrary kinds of attributes. If you don't know whether an attribute is a magic
    /// dunder attribute, it's highly likely that this method isn't the right thing to do for you. Examples of
    /// magic dunder methods include: `__getattr__`, `__eq__`, `__contains__`, etc. Also see [`Self::type_of_magic_dunder_attr`].
    pub fn call_magic_dunder_method(
        &self,
        ty: &Type,
        method_name: &Name,
        range: TextRange,
        args: &[CallArg],
        keywords: &[CallKeyword],
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Option<Type> {
        let callee_ty = self.type_of_magic_dunder_attr(
            ty,
            method_name,
            range,
            errors,
            context,
            "Expr::call_method",
        )?;
        Some(self.make_call_target_and_call(
            callee_ty,
            method_name,
            range,
            args,
            keywords,
            errors,
            context,
        ))
    }

    /// Calls a method. If no attribute exists with the given method name, logs an error and calls the method with
    /// an assumed type of Callable[..., Any].
    pub fn call_method_or_error(
        &self,
        ty: &Type,
        method_name: &Name,
        range: TextRange,
        args: &[CallArg],
        keywords: &[CallKeyword],
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let callee_ty =
            self.type_of_attr_get(ty, method_name, range, errors, context, "Expr::call_method");
        self.make_call_target_and_call(
            callee_ty,
            method_name,
            range,
            args,
            keywords,
            errors,
            context,
        )
    }

    /// If the metaclass defines a custom `__call__`, call it. If the `__call__` comes from `type`, ignore
    /// it because `type.__call__` behavior is baked into our constructor logic.
    fn call_metaclass(
        &self,
        cls: &ClassType,
        range: TextRange,
        args: &[CallArg],
        keywords: &[CallKeyword],
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
    ) -> Option<Type> {
        let dunder_call = self.get_metaclass_dunder_call(cls)?;
        // Clone targs because we don't want instantiations from metaclass __call__
        let mut ctor_targs = cls.targs().clone();
        let mut ret = self.call_infer(
            self.as_call_target_or_error(
                dunder_call,
                CallStyle::Method(&dunder::CALL),
                range,
                errors,
                context,
            ),
            args,
            keywords,
            range,
            errors,
            context,
            hint,
            Some(&mut ctor_targs),
        );
        self.solver()
            .finish_class_targs(&mut ctor_targs, self.uniques);
        ret.subst_mut(&ctor_targs.substitution_map());
        Some(ret)
    }

    fn construct_class(
        &self,
        mut cls: ClassType,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
    ) -> Type {
        // Based on https://typing.readthedocs.io/en/latest/spec/constructors.html.
        if let Some(hint) = hint {
            self.solver()
                .freshen_class_targs(cls.targs_mut(), self.uniques);

            self.is_subset_eq(&cls.clone().to_type(), hint.ty());
            self.solver().generalize_class_targs(cls.targs_mut());
        }
        let hint = None; // discard hint
        if let Some(ret) = self.call_metaclass(&cls, range, args, keywords, errors, context, hint)
            && !self.is_compatible_constructor_return(&ret, cls.class_object())
        {
            if let Some(metaclass_dunder_call) = self.get_metaclass_dunder_call(&cls) {
                // Not quite an overload, but close enough
                self.record_overload_trace_from_type(range, metaclass_dunder_call);
            }
            // Got something other than an instance of the class under construction.
            return ret;
        }
        let mut dunder_new_ret = None;
        let (overrides_new, dunder_new_has_errors) =
            if let Some(new_method) = self.get_dunder_new(&cls) {
                let cls_ty = Type::type_form(cls.clone().to_type());
                let full_args = iter::once(CallArg::ty(&cls_ty, range))
                    .chain(args.iter().cloned())
                    .collect::<Vec<_>>();
                let dunder_new_errors = self.error_collector();
                let ret = self.call_infer(
                    self.as_call_target_or_error(
                        new_method.clone(),
                        CallStyle::Method(&dunder::NEW),
                        range,
                        errors,
                        context,
                    ),
                    &full_args,
                    keywords,
                    range,
                    &dunder_new_errors,
                    context,
                    hint,
                    Some(cls.targs_mut()),
                );
                let has_errors = !dunder_new_errors.is_empty();
                errors.extend(dunder_new_errors);
                // Not quite an overload, but close enough
                self.record_overload_trace_from_type(range, new_method);
                if self.is_compatible_constructor_return(&ret, cls.class_object()) {
                    dunder_new_ret = Some(ret);
                } else {
                    // Got something other than an instance of the class under construction.
                    self.solver()
                        .finish_class_targs(cls.targs_mut(), self.uniques);
                    return ret.subst(&cls.targs().substitution_map());
                }
                (true, has_errors)
            } else {
                (false, false)
            };

        // If the class overrides `object.__new__` but not `object.__init__`, the `__init__` call
        // always succeeds at runtime, so we skip analyzing it.
        let get_object_init = !overrides_new;
        if let Some(init_method) = self.get_dunder_init(&cls, get_object_init) {
            let dunder_init_errors = self.error_collector();
            self.call_infer(
                self.as_call_target_or_error(
                    init_method.clone(),
                    CallStyle::Method(&dunder::INIT),
                    range,
                    errors,
                    context,
                ),
                args,
                keywords,
                range,
                &dunder_init_errors,
                context,
                hint,
                Some(cls.targs_mut()),
            );
            // Report `__init__` errors only when there are no `__new__` errors, to avoid redundant errors.
            if !dunder_new_has_errors {
                errors.extend(dunder_init_errors);
            }
            // Not quite an overload, but close enough
            self.record_overload_trace_from_type(range, init_method);
        }
        self.solver()
            .finish_class_targs(cls.targs_mut(), self.uniques);
        if let Some(mut ret) = dunder_new_ret {
            ret.subst_mut(&cls.targs().substitution_map());
            ret
        } else {
            cls.to_type()
        }
    }

    fn construct_typed_dict(
        &self,
        mut typed_dict: TypedDict,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
    ) -> Type {
        if let Some(hint) = hint {
            self.solver()
                .freshen_class_targs(typed_dict.targs_mut(), self.uniques);

            self.is_subset_eq(&typed_dict.clone().to_type(), hint.ty());
            self.solver().generalize_class_targs(typed_dict.targs_mut());
        }
        let hint = None; // discard hint
        // We know `__init__` exists because we synthesize it.
        let init_method = self.get_typed_dict_dunder_init(&typed_dict).unwrap();
        self.call_infer(
            self.as_call_target_or_error(
                init_method,
                CallStyle::Method(&dunder::INIT),
                range,
                errors,
                context,
            ),
            args,
            keywords,
            range,
            errors,
            context,
            hint,
            Some(typed_dict.targs_mut()),
        );
        self.solver()
            .finish_class_targs(typed_dict.targs_mut(), self.uniques);
        Type::TypedDict(typed_dict)
    }

    fn first_arg_type(&self, args: &[CallArg], errors: &ErrorCollector) -> Option<Type> {
        if let Some(first_arg) = args.first() {
            match first_arg {
                CallArg::Arg(x) => Some(x.infer(self, errors)),
                CallArg::Star(..) => None,
            }
        } else {
            None
        }
    }

    pub fn call_infer(
        &self,
        call_target: CallTarget,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
        ctor_targs: Option<&mut TArgs>,
    ) -> Type {
        let metadata = call_target.function_metadata();
        // Does this call target correspond to a function whose keyword arguments we should save?
        let kw_metadata = {
            if let Some(m) = metadata
                && (matches!(
                    m.kind,
                    FunctionKind::Dataclass | FunctionKind::DataclassTransform
                ) || m.flags.dataclass_transform_metadata.is_some())
            {
                Some(m.clone())
            } else {
                None
            }
        };
        let res = match call_target {
            CallTarget::Class(cls) => {
                if self
                    .get_metadata_for_class(cls.class_object())
                    .is_protocol()
                {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::BadInstantiation, context),
                        format!(
                            "Cannot instantiate `{}` because it is a protocol",
                            cls.name()
                        ),
                    );
                }
                if cls.has_qname("builtins", "bool") {
                    match self.first_arg_type(args, errors) {
                        None => (),
                        Some(ty) => self.check_dunder_bool_is_callable(&ty, range, errors),
                    }
                };
                self.construct_class(cls, args, keywords, range, errors, context, hint)
            }
            CallTarget::TypedDict(td) => {
                self.construct_typed_dict(td, args, keywords, range, errors, context, hint)
            }
            CallTarget::BoundMethod(
                obj,
                TargetWithTParams(
                    tparams,
                    Function {
                        signature,
                        metadata,
                    },
                ),
            ) => self.callable_infer(
                signature,
                Some(metadata.kind.as_func_id()),
                tparams.as_deref(),
                Some(obj),
                args,
                keywords,
                range,
                errors,
                errors,
                context,
                hint,
                ctor_targs,
            ),
            CallTarget::Callable(TargetWithTParams(tparams, callable)) => self.callable_infer(
                callable,
                None,
                tparams.as_deref(),
                None,
                args,
                keywords,
                range,
                errors,
                errors,
                context,
                hint,
                ctor_targs,
            ),
            CallTarget::Function(TargetWithTParams(
                tparams,
                Function {
                    signature: callable,
                    metadata,
                },
            )) => self.callable_infer(
                callable,
                Some(metadata.kind.as_func_id()),
                tparams.as_deref(),
                None,
                args,
                keywords,
                range,
                errors,
                errors,
                context,
                hint,
                ctor_targs,
            ),
            CallTarget::FunctionOverload(overloads, metadata) => {
                self.call_overloads(
                    overloads, metadata, None, args, keywords, range, errors, context, hint,
                    ctor_targs,
                )
                .0
            }
            CallTarget::BoundMethodOverload(obj, overloads, meta) => {
                self.call_overloads(
                    overloads,
                    meta,
                    Some(obj),
                    args,
                    keywords,
                    range,
                    errors,
                    context,
                    hint,
                    ctor_targs,
                )
                .0
            }
            CallTarget::Union(targets) => {
                let call = CallWithTypes::new();
                let args = call.vec_call_arg(args, self, errors);
                let keywords = call.vec_call_keyword(keywords, self, errors);
                self.unions(targets.into_map(|t| {
                    let ctor_targs = None; // hack
                    self.call_infer(
                        t, &args, &keywords, range, errors, context, hint, ctor_targs,
                    )
                }))
            }
            CallTarget::Any(style) => {
                // Make sure we still catch errors in the arguments.
                for arg in args {
                    match arg {
                        CallArg::Arg(e) | CallArg::Star(e, _) => {
                            e.infer(self, errors);
                        }
                    }
                }
                for kw in keywords {
                    kw.value.infer(self, errors);
                }
                style.propagate()
            }
        };
        if let Some(func_metadata) = kw_metadata {
            let mut kws = TypeMap::new();
            for kw in keywords {
                if let Some(name) = kw.arg {
                    kws.0.insert(name.id.clone(), kw.value.infer(self, errors));
                }
            }
            Type::KwCall(Box::new(KwCall {
                func_metadata,
                keywords: kws,
                return_ty: res,
            }))
        } else {
            res
        }
    }

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
        ctor_targs: Option<&mut TArgs>,
    ) -> (Type, Callable) {
        // There may be Expr values in args and keywords.
        // If we infer them for each overload, we may end up inferring them multiple times.
        // If those overloads contain nested overloads, then we can easily end up with O(2^n) perf.
        // Therefore, flatten all TypeOrExpr's into Type before we start
        let call = CallWithTypes::new();
        let args = call.vec_call_arg(args, self, errors);
        let keywords = call.vec_call_keyword(keywords, self, errors);

        let mut closest_overload: Option<CalledOverload> = None;
        for callable in overloads.iter() {
            let mut ctor_targs_ = ctor_targs.as_ref().map(|x| (**x).clone());
            let tparams = callable.0.as_deref();

            let mut try_call = |hint| {
                let call_errors = self.error_collector();
                let res = self.callable_infer(
                    callable.1.signature.clone(),
                    Some(metadata.kind.as_func_id()),
                    tparams,
                    self_obj.clone(),
                    &args,
                    &keywords,
                    range,
                    errors,
                    &call_errors,
                    // We intentionally drop the context here, as arg errors don't need it,
                    // and if there are any call errors, we'll log a "No matching overloads"
                    // error with the necessary context.
                    None,
                    hint,
                    ctor_targs_.as_mut(),
                );
                (call_errors, res)
            };

            // We want to use our hint to contextually type the arguments, but errors resulting
            // from the hint should not influence overload selection. If there are call errors, we
            // try again without a hint in case we can still match this overload.
            let (call_errors, res) = try_call(hint);
            let (call_errors, res) =
                if tparams.is_some() && hint.is_some() && !call_errors.is_empty() {
                    try_call(None)
                } else {
                    (call_errors, res)
                };

            if call_errors.is_empty() {
                // An overload is chosen, we should record it to power IDE services.
                self.record_overload_trace(
                    range,
                    overloads.map(|TargetWithTParams(_, Function { signature, .. })| signature),
                    &callable.1.signature,
                    true,
                );

                // If the selected overload is deprecated, we log a deprecation error.
                if callable.1.metadata.flags.is_deprecated {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::Deprecated, context),
                        format!(
                            "Call to deprecated overload `{}`",
                            callable
                                .1
                                .metadata
                                .kind
                                .as_func_id()
                                .format(self.module().name())
                        ),
                    );
                }
                if let Some(targs) = ctor_targs {
                    *targs = ctor_targs_.unwrap();
                }
                return (res, callable.1.signature.clone());
            }
            let called_overload = CalledOverload {
                signature: callable.1.signature.clone(),
                call_errors,
            };
            match &closest_overload {
                Some(overload)
                    if overload.call_errors.len() <= called_overload.call_errors.len() => {}
                _ => {
                    closest_overload = Some(called_overload);
                }
            }
        }
        // We're guaranteed to have at least one overload.
        let closest_overload = closest_overload.unwrap();
        self.record_overload_trace(
            range,
            overloads.map(|TargetWithTParams(_, Function { signature, .. })| signature),
            &closest_overload.signature,
            false,
        );

        let mut msg = vec1![
            format!(
                "No matching overload found for function `{}`",
                metadata.kind.as_func_id().format(self.module().name())
            ),
            "Possible overloads:".to_owned(),
        ];
        for overload in overloads {
            let suffix = if overload.1.signature == closest_overload.signature {
                " [closest match]"
            } else {
                ""
            };
            let signature = match self_obj {
                Some(_) => overload
                    .1
                    .signature
                    .split_first_param()
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
        (Type::any_error(), closest_overload.signature)
    }

    /// Helper function hide details of call synthesis from the attribute resolution code.
    pub fn call_property_getter(
        &self,
        getter_method: Type,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let call_target = self.as_call_target_or_error(
            getter_method,
            CallStyle::FreeForm,
            range,
            errors,
            context,
        );
        self.call_infer(call_target, &[], &[], range, errors, context, None, None)
    }

    /// Helper function hide details of call synthesis from the attribute resolution code.
    pub fn call_property_setter(
        &self,
        setter_method: Type,
        got: CallArg,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let call_target = self.as_call_target_or_error(
            setter_method,
            CallStyle::FreeForm,
            range,
            errors,
            context,
        );
        self.call_infer(call_target, &[got], &[], range, errors, context, None, None)
    }

    /// Helper function hide details of call synthesis from the attribute resolution code.
    pub fn call_descriptor_getter(
        &self,
        getter_method: Type,
        base: DescriptorBase,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        // When a descriptor is accessed on an instance, it gets the instance and the class object as
        // the `obj` and `objtype` arguments. When it is accessed on a class, it gets `None` as `obj`
        // and the class object as `objtype`.
        let (objtype, obj) = match base {
            DescriptorBase::Instance(classtype) => (
                Type::ClassDef(classtype.class_object().dupe()),
                Type::ClassType(classtype),
            ),
            DescriptorBase::ClassDef(class) => (Type::ClassDef(class), Type::None),
        };
        let args = [CallArg::ty(&obj, range), CallArg::ty(&objtype, range)];
        let call_target = self.as_call_target_or_error(
            getter_method,
            CallStyle::FreeForm,
            range,
            errors,
            context,
        );
        self.call_infer(call_target, &args, &[], range, errors, context, None, None)
    }

    /// Helper function hide details of call synthesis from the attribute resolution code.
    pub fn call_descriptor_setter(
        &self,
        setter_method: Type,
        class_type: ClassType,
        got: CallArg,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        // When a descriptor is set on an instance, it gets the instance `class_type` and the value `got` as arguments.
        // Descriptor setters cannot be called on a class (an attempt to assign will overwrite the
        // descriptor itself rather than call the setter).
        let instance = Type::ClassType(class_type);
        let args = [CallArg::ty(&instance, range), got];
        let call_target = self.as_call_target_or_error(
            setter_method,
            CallStyle::FreeForm,
            range,
            errors,
            context,
        );
        self.call_infer(call_target, &args, &[], range, errors, context, None, None)
    }

    pub fn call_getattr_or_delattr(
        &self,
        getattr_ty: Type,
        attr_name: Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let call_target =
            self.as_call_target_or_error(getattr_ty, CallStyle::FreeForm, range, errors, context);
        let attr_name_ty = Type::Literal(Lit::Str(attr_name.as_str().into()));
        self.call_infer(
            call_target,
            &[CallArg::ty(&attr_name_ty, range)],
            &[],
            range,
            errors,
            context,
            None,
            None,
        )
    }

    pub fn call_setattr(
        &self,
        setattr_ty: Type,
        arg: CallArg,
        attr_name: Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let call_target =
            self.as_call_target_or_error(setattr_ty, CallStyle::FreeForm, range, errors, context);
        let attr_name_ty = Type::Literal(Lit::Str(attr_name.as_str().into()));
        self.call_infer(
            call_target,
            &[CallArg::ty(&attr_name_ty, range), arg],
            &[],
            range,
            errors,
            context,
            None,
            None,
        )
    }

    pub fn constructor_to_callable(&self, cls: &ClassType) -> Type {
        let class_type = cls.clone().to_type();
        if let Some(metaclass_call_attr_ty) = self.get_metaclass_dunder_call(cls) {
            // If the class has a custom metaclass and the return type of the metaclass's __call__
            // is not a subclass of the current class, use that and ignore __new__ and __init__
            if metaclass_call_attr_ty
                .callable_return_type()
                .is_some_and(|ret| !self.is_compatible_constructor_return(&ret, cls.class_object()))
            {
                return metaclass_call_attr_ty;
            }
        }
        // Default constructor that takes no args and returns Self.
        let default_constructor = || {
            Type::Callable(Box::new(Callable::list(
                ParamList::new(Vec::new()),
                class_type.clone(),
            )))
        };
        // Check the __new__ method and whether it comes from object or has been overridden
        let (new_attr_ty, overrides_new) = if let Some(t) = self
            .get_dunder_new(cls)
            .and_then(|t| self.bind_dunder_new(&t, cls.clone()))
        {
            if t.callable_return_type()
                .is_some_and(|ret| !self.is_compatible_constructor_return(&ret, cls.class_object()))
            {
                // If the return type of __new__ is not a subclass of the current class, use that and ignore __init__
                return t;
            }
            (t, true)
        } else {
            (default_constructor(), false)
        };
        // Check the __init__ method and whether it comes from object or has been overridden
        let (init_attr_ty, overrides_init) = if let Some(mut t) = self.get_dunder_init(cls, false) {
            // Replace the return type with Self (the current class)
            t.set_callable_return_type(class_type.clone());
            (t, true)
        } else {
            (default_constructor(), false)
        };
        if !overrides_new && overrides_init {
            // If `__init__` is overridden and `__new__` is inherited from object, use `__init__`
            init_attr_ty
        } else if overrides_new && !overrides_init {
            // If `__new__` is overridden and `__init__` is inherited from object, use `__new__`
            new_attr_ty
        } else {
            // If both are overridden, take the union
            // Only if neither are overridden, use the `__new__` and `__init__` from object
            self.unions(vec![new_attr_ty, init_attr_ty])
        }
    }
}
