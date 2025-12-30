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
use pyrefly_types::typed_dict::TypedDictInner;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Union;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::callable::CallWithTypes;
use crate::alt::class::class_field::DescriptorBase;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::Key;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructorKind {
    // `MyClass`
    BareClassName,
    // `type[MyClass]` or `type[Self]`
    TypeOfClass,
}

/// A thing that can be called (see as_call_target and call_infer).
/// Note that a single "call" may invoke multiple functions under the hood,
/// e.g., `__new__` followed by `__init__` for Class.
#[derive(Debug, Clone)]
pub enum CallTarget {
    /// A typing.Callable.
    Callable(TargetWithTParams<Callable>),
    /// A function.
    Function(TargetWithTParams<Function>),
    /// Method of a class. The `Type` is the self/cls argument.
    BoundMethod(Type, TargetWithTParams<Function>),
    /// A class object.
    Class(ClassType, ConstructorKind),
    /// A TypedDict.
    TypedDict(TypedDictInner),
    /// An overloaded function.
    FunctionOverload(Vec1<TargetWithTParams<Function>>, FuncMetadata),
    /// An overloaded method.
    BoundMethodOverload(Type, Vec1<TargetWithTParams<Function>>, FuncMetadata),
    /// A union of call targets.
    Union(Vec<CallTarget>),
    /// Any, as a call target.
    Any(AnyStyle),
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum CallTargetLookup {
    /// When a type is callable, this represents what can be called.
    Ok(Box<CallTarget>),
    /// When a type is not callable, still collect what can be called in callable "subcases". This is
    /// for example used for a union type that is not callable, but some of its "subcases" are callable.
    Error(Vec<CallTarget>),
}

impl CallTargetLookup {
    pub fn is_error(&self) -> bool {
        match self {
            CallTargetLookup::Ok(..) => false,
            CallTargetLookup::Error(..) => true,
        }
    }
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

    /// We only raise here for calls where we know the callee is the
    /// abstract definition itself. That includes:
    ///   * direct calls on the defining class object (e.g. Base.build())
    ///   * super() lookups that surface the abstract method
    ///
    /// We skip calls via variables of type[Base] for non-final concrete classes,
    /// because those values might point at a concrete subclass.
    fn should_error_for_abstract_call(&self, call_target: &CallTarget) -> bool {
        match call_target {
            CallTarget::BoundMethod(obj, _) | CallTarget::BoundMethodOverload(obj, ..) => match obj
            {
                Type::ClassDef(_) | Type::SuperInstance(_) => true,
                Type::ClassType(cls) | Type::SelfType(cls) => {
                    let metadata = self.get_metadata_for_class(cls.class_object());
                    metadata.is_final() && !metadata.is_protocol()
                }
                Type::Type(inner) => match &**inner {
                    Type::ClassType(cls) | Type::SelfType(cls) => {
                        let metadata = self.get_metadata_for_class(cls.class_object());
                        metadata.is_final() && !metadata.is_protocol()
                    }
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        }
    }

    pub fn as_call_target(&self, ty: Type) -> CallTargetLookup {
        self.as_call_target_impl(ty, None, /* dunder_call */ false)
    }

    fn as_call_target_impl(
        &self,
        ty: Type,
        quantified: Option<Quantified>,
        dunder_call: bool,
    ) -> CallTargetLookup {
        match ty {
            Type::Callable(c) => {
                CallTargetLookup::Ok(Box::new(CallTarget::Callable(TargetWithTParams(None, *c))))
            }
            Type::Function(func) => CallTargetLookup::Ok(Box::new(CallTarget::Function(
                TargetWithTParams(None, *func),
            ))),
            Type::Overload(overload) => {
                let funcs = overload.signatures.mapped(|ty| match ty {
                    OverloadType::Function(function) => TargetWithTParams(None, function),
                    OverloadType::Forall(forall) => {
                        TargetWithTParams(Some(forall.tparams), forall.body)
                    }
                });
                CallTargetLookup::Ok(Box::new(CallTarget::FunctionOverload(
                    funcs,
                    *overload.metadata,
                )))
            }
            Type::BoundMethod(bm) => {
                let BoundMethod { obj, func } = *bm;
                match self.as_call_target_impl(func.as_type(), quantified, dunder_call) {
                    CallTargetLookup::Ok(box CallTarget::Function(func)) => {
                        CallTargetLookup::Ok(Box::new(CallTarget::BoundMethod(obj, func)))
                    }
                    CallTargetLookup::Ok(box CallTarget::FunctionOverload(overloads, meta)) => {
                        CallTargetLookup::Ok(Box::new(CallTarget::BoundMethodOverload(
                            obj, overloads, meta,
                        )))
                    }
                    _ => CallTargetLookup::Error(vec![]),
                }
            }
            Type::ClassDef(cls) => match self.instantiate(&cls) {
                // `instantiate` can only return `ClassType` or `TypedDict`
                Type::ClassType(cls) => CallTargetLookup::Ok(Box::new(CallTarget::Class(
                    cls,
                    ConstructorKind::BareClassName,
                ))),
                Type::TypedDict(TypedDict::TypedDict(typed_dict)) => {
                    CallTargetLookup::Ok(Box::new(CallTarget::TypedDict(typed_dict)))
                }
                _ => unreachable!(),
            },
            Type::Type(box Type::ClassType(cls)) | Type::Type(box Type::SelfType(cls)) => {
                CallTargetLookup::Ok(Box::new(CallTarget::Class(
                    cls,
                    ConstructorKind::TypeOfClass,
                )))
            }
            Type::Type(box Type::Tuple(tuple)) => CallTargetLookup::Ok(Box::new(
                CallTarget::Class(self.erase_tuple_type(tuple), ConstructorKind::TypeOfClass),
            )),
            Type::Type(box Type::Quantified(quantified)) => {
                CallTargetLookup::Ok(Box::new(CallTarget::Callable(TargetWithTParams(
                    None,
                    Callable {
                        // TODO: use upper bound to determine input parameters
                        params: Params::Ellipsis,
                        ret: Type::Quantified(quantified),
                    },
                ))))
            }
            Type::Type(inner) if let Type::Any(style) = *inner => {
                CallTargetLookup::Ok(Box::new(CallTarget::Any(style)))
            }
            Type::Forall(forall) => {
                let mut target =
                    self.as_call_target_impl(forall.body.as_type(), quantified, dunder_call);
                match &mut target {
                    CallTargetLookup::Ok(
                        box (CallTarget::Callable(TargetWithTParams(x, _))
                        | CallTarget::Function(TargetWithTParams(x, _))),
                    ) => {
                        *x = Some(forall.tparams);
                    }
                    _ => {}
                }
                target
            }
            Type::Var(v) if let Some(_guard) = self.recurse(v) => {
                self.as_call_target_impl(self.solver().force_var(v), quantified, dunder_call)
            }
            Type::Union(box Union { members: xs, .. }) => {
                let xs_length = xs.len();
                let targets = xs
                    .into_iter()
                    .filter_map(|x| {
                        match self.as_call_target_impl(x, quantified.clone(), dunder_call) {
                            CallTargetLookup::Ok(target) => Some(*target),
                            CallTargetLookup::Error(..) => None,
                        }
                    })
                    .collect::<Vec<_>>();
                let targets_length = targets.len();
                if xs_length > targets_length {
                    CallTargetLookup::Error(targets)
                } else if targets_length == 1 {
                    CallTargetLookup::Ok(Box::new(targets.into_iter().next().unwrap()))
                } else {
                    CallTargetLookup::Ok(Box::new(CallTarget::Union(targets)))
                }
            }
            Type::Any(style) => CallTargetLookup::Ok(Box::new(CallTarget::Any(style))),
            Type::TypeAlias(ta) => {
                self.as_call_target_impl(ta.as_value(self.stdlib), quantified, dunder_call)
            }
            Type::ClassType(cls) => {
                if let Some(quantified) = quantified {
                    self.quantified_instance_as_dunder_call(quantified.clone(), &cls)
                        .map_or(CallTargetLookup::Error(vec![]), |ty| {
                            self.as_call_target_impl(ty, Some(quantified), dunder_call)
                        })
                } else if dunder_call {
                    // Avoid infinite recursion
                    CallTargetLookup::Error(vec![])
                } else {
                    self.instance_as_dunder_call(&cls).map_or(
                        CallTargetLookup::Error(vec![]),
                        |ty| {
                            self.as_call_target_impl(ty, quantified, /* dunder_call */ true)
                        },
                    )
                }
            }
            Type::SelfType(cls) => {
                // Ignoring `quantified` is okay here because Self is not a valid typevar bound.
                self.self_as_dunder_call(&cls)
                    .map_or(CallTargetLookup::Error(vec![]), |ty| {
                        self.as_call_target_impl(ty, None, dunder_call)
                    })
            }
            Type::Type(box Type::TypedDict(TypedDict::TypedDict(typed_dict))) => {
                CallTargetLookup::Ok(Box::new(CallTarget::TypedDict(typed_dict)))
            }
            Type::Type(box Type::TypedDict(td @ TypedDict::Anonymous(_))) => {
                let value_ty = self.get_typed_dict_value_type(&td);
                let cls = self
                    .stdlib
                    .dict(self.stdlib.str().clone().to_type(), value_ty);
                CallTargetLookup::Ok(Box::new(CallTarget::Class(
                    cls,
                    ConstructorKind::TypeOfClass,
                )))
            }
            Type::Type(box Type::Intersect(box (_, fallback))) => {
                // TODO(rechen): implement calling `type[A & B]`
                self.as_call_target_impl(Type::type_form(fallback), quantified, dunder_call)
            }
            Type::Quantified(q) if q.is_type_var() => match q.restriction() {
                Restriction::Unrestricted => CallTargetLookup::Error(vec![]),
                Restriction::Bound(bound) => match bound {
                    Type::Union(box Union { members, .. }) => {
                        let mut targets = Vec::new();
                        for member in members {
                            if let CallTargetLookup::Ok(target) = self.as_call_target_impl(
                                member.clone(),
                                Some(
                                    q.clone()
                                        .with_restriction(Restriction::Bound(member.clone())),
                                ),
                                dunder_call,
                            ) {
                                targets.push(*target);
                            } else {
                                return CallTargetLookup::Error(vec![]);
                            }
                        }
                        CallTargetLookup::Ok(Box::new(CallTarget::Union(targets)))
                    }
                    _ => self.as_call_target_impl(bound.clone(), Some(*q), dunder_call),
                },
                Restriction::Constraints(constraints) => {
                    let mut targets = Vec::new();
                    for constraint in constraints {
                        if let CallTargetLookup::Ok(target) = self.as_call_target_impl(
                            constraint.clone(),
                            Some(q.clone().with_restriction(Restriction::Constraints(vec![
                                constraint.clone(),
                            ]))),
                            dunder_call,
                        ) {
                            targets.push(*target);
                        } else {
                            return CallTargetLookup::Error(vec![]);
                        }
                    }
                    CallTargetLookup::Ok(Box::new(CallTarget::Union(targets)))
                }
            },
            Type::KwCall(call) => self.as_call_target_impl(call.return_ty, quantified, dunder_call),
            Type::Literal(Lit::Enum(enum_)) => {
                self.as_call_target_impl(enum_.class.to_type(), quantified, dunder_call)
            }
            _ => CallTargetLookup::Error(vec![]),
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
            CallTargetLookup::Ok(target) => {
                let metadata = target.function_metadata();
                if let Some(m) = metadata
                    && let Some(deprecation) = &m.flags.deprecation
                {
                    // We manually construct an error using the message from the context but a
                    // Deprecated error kind so that the error is shown at the Deprecated severity
                    // (default: WARN) rather than the severity of the context's error kind.
                    let mut msg = deprecation.as_error_message(format!(
                        "`{}` is deprecated",
                        m.kind.format(self.module().name())
                    ));
                    if let Some(ctx) = context {
                        msg.insert(0, ctx().format());
                    }
                    errors.add(range, ErrorInfo::Kind(ErrorKind::Deprecated), msg);
                }
                *target
            }
            CallTargetLookup::Error(..) => {
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
            true,
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
        self.record_resolved_trace(range, callee_ty.clone());
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
        arguments_range: TextRange,
        callee_range: Option<TextRange>,
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
        if let Some(ret) =
            self.call_metaclass(&cls, arguments_range, args, keywords, errors, context, hint)
            && !self.is_compatible_constructor_return(&ret, cls.class_object())
        {
            if let Some(metaclass_dunder_call) = self.get_metaclass_dunder_call(&cls) {
                if let Some(callee_range) = callee_range
                    && let Some(metaclass) = self
                        .get_metadata_for_class(cls.class_object())
                        .custom_metaclass()
                {
                    self.record_external_attribute_definition_index(
                        &metaclass.clone().to_type(),
                        &dunder::CALL,
                        callee_range,
                    );
                }
                self.record_resolved_trace(arguments_range, metaclass_dunder_call);
            }
            // Got something other than an instance of the class under construction.
            return ret;
        }
        let mut dunder_new_ret = None;
        let (overrides_new, dunder_new_has_errors) =
            if let Some(new_method) = self.get_dunder_new(&cls) {
                let cls_ty = Type::type_form(cls.clone().to_type());
                let full_args = iter::once(CallArg::ty(&cls_ty, arguments_range))
                    .chain(args.iter().cloned())
                    .collect::<Vec<_>>();
                let dunder_new_errors = self.error_collector();
                let ret = self.call_infer(
                    self.as_call_target_or_error(
                        new_method.clone(),
                        CallStyle::Method(&dunder::NEW),
                        arguments_range,
                        errors,
                        context,
                    ),
                    &full_args,
                    keywords,
                    arguments_range,
                    &dunder_new_errors,
                    context,
                    hint,
                    Some(cls.targs_mut()),
                );
                let has_errors = !dunder_new_errors.is_empty();
                errors.extend(dunder_new_errors);
                if let Some(callee_range) = callee_range {
                    self.record_external_attribute_definition_index(
                        &cls.clone().to_type(),
                        &dunder::NEW,
                        callee_range,
                    );
                }
                self.record_resolved_trace(arguments_range, new_method);
                if self.is_compatible_constructor_return(&ret, cls.class_object()) {
                    dunder_new_ret = Some(ret);
                } else if !matches!(ret, Type::Any(AnyStyle::Error | AnyStyle::Implicit)) {
                    // Got something other than an instance of the class under construction.
                    // According to the spec, the actual type (as opposed to the class under construction)
                    // should take priority. However, if the actual type comes from a type error or an implicit
                    // Any, using the class under construction is still more useful.
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
                    arguments_range,
                    errors,
                    context,
                ),
                args,
                keywords,
                arguments_range,
                &dunder_init_errors,
                context,
                hint,
                Some(cls.targs_mut()),
            );
            // Report `__init__` errors only when there are no `__new__` errors, to avoid redundant errors.
            if !dunder_new_has_errors {
                errors.extend(dunder_init_errors);
            }
            if let Some(callee_range) = callee_range {
                self.record_external_attribute_definition_index(
                    &cls.clone().to_type(),
                    &dunder::INIT,
                    callee_range,
                );
            }
            self.record_resolved_trace(arguments_range, init_method);
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
        mut typed_dict: TypedDictInner,
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
        let init_method = self.get_typed_dict_dunder_init(&typed_dict);
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
        Type::TypedDict(TypedDict::TypedDict(typed_dict))
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

    fn call_infer_with_range(
        &self,
        call_target: CallTarget,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        callee_range: Option<TextRange>,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
        ctor_targs: Option<&mut TArgs>,
    ) -> Type {
        let metadata = call_target.function_metadata();
        if let Some(meta) = metadata
            && meta.flags.is_abstract_method
            && self.should_error_for_abstract_call(&call_target)
        {
            let method_name = meta.kind.format(self.module().name());
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::AbstractMethodCall, context),
                format!("Cannot call abstract method `{method_name}`"),
            );
        }
        // Does this call target correspond to a function whose keyword arguments we should save?
        let kw_metadata = {
            if let Some(m) = metadata
                && (matches!(
                    m.kind,
                    FunctionKind::Dataclass | FunctionKind::DataclassTransform
                ) || m.kind.is_signature_preserving_decorator()
                    || m.flags.dataclass_transform_metadata.is_some())
            {
                Some(m.clone())
            } else {
                None
            }
        };
        let res = match call_target {
            CallTarget::Class(cls, constructor_kind) => {
                if cls.has_qname("typing", "Any") {
                    return self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::BadInstantiation, context),
                        format!("`{}` can not be instantiated", cls.name()),
                    );
                }
                let metadata = self.get_metadata_for_class(cls.class_object());
                if metadata.is_protocol() && constructor_kind == ConstructorKind::BareClassName {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::BadInstantiation, context),
                        format!(
                            "Cannot instantiate `{}` because it is a protocol",
                            cls.name()
                        ),
                    );
                } else {
                    let abstract_members = self.get_abstract_members_for_class(cls.class_object());
                    let unimplemented_abstract_methods =
                        abstract_members.unimplemented_abstract_methods();
                    if constructor_kind == ConstructorKind::BareClassName
                        && !unimplemented_abstract_methods.is_empty()
                    {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::new(ErrorKind::BadInstantiation, context),
                            format!(
                                "Cannot instantiate `{}` because the following members are abstract: {}",
                                cls.name(),
                                unimplemented_abstract_methods
                                    .iter()
                                    .map(|x| format!("`{x}`"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        );
                    }
                }
                if cls.has_qname("builtins", "bool") {
                    match self.first_arg_type(args, errors) {
                        None => (),
                        Some(ty) => self.check_dunder_bool_is_callable(&ty, range, errors),
                    }
                };
                self.construct_class(
                    cls,
                    args,
                    keywords,
                    range,
                    callee_range,
                    errors,
                    context,
                    hint,
                )
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
                Some(&metadata.kind),
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
                Some(&metadata.kind),
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
                    self.call_infer_with_range(
                        t,
                        &args,
                        &keywords,
                        range,
                        callee_range,
                        errors,
                        context,
                        hint,
                        ctor_targs,
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
        self.call_infer_with_range(
            call_target,
            args,
            keywords,
            range,
            None,
            errors,
            context,
            hint,
            ctor_targs,
        )
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

    pub fn expr_call_infer(
        &self,
        x: &ExprCall,
        mut callee_ty: Type,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        if matches!(&callee_ty, Type::ClassDef(cls) if cls.is_builtin("super")) {
            // Because we have to construct a binding for super in order to fill in implicit arguments,
            // we can't handle things like local aliases to super. If we hit a case where the binding
            // wasn't constructed, fall back to `Any`.
            self.get_hashed_opt(Hashed::new(&Key::SuperInstance(x.range)))
                .map_or_else(Type::any_implicit, |type_info| type_info.arc_clone_ty())
        } else {
            self.expand_vars_mut(&mut callee_ty);

            let args;
            let kws;
            let call = CallWithTypes::new();
            if callee_ty.is_union() {
                // If we have a union we will distribute over it, and end up duplicating each function call.
                args = x
                    .arguments
                    .args
                    .map(|x| call.call_arg(&CallArg::expr_maybe_starred(x), self, errors));
                kws = x
                    .arguments
                    .keywords
                    .map(|x| call.call_keyword(&CallKeyword::new(x), self, errors));
            } else {
                args = x.arguments.args.map(CallArg::expr_maybe_starred);
                kws = x.arguments.keywords.map(CallKeyword::new);
            }

            self.distribute_over_union(&callee_ty, |ty| match ty.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::AssertType)) => self
                    .call_assert_type(
                        &x.arguments.args,
                        &x.arguments.keywords,
                        x.arguments.range,
                        hint,
                        errors,
                    ),
                Some(CalleeKind::Function(FunctionKind::RevealType)) => self
                    .call_reveal_type(
                        &x.arguments.args,
                        &x.arguments.keywords,
                        x.arguments.range,
                        hint,
                        errors,
                    ),
                Some(CalleeKind::Function(FunctionKind::Cast)) => {
                    // For typing.cast, we have to hard-code a check for whether the first argument
                    // is a type, so it's simplest to special-case the entire call.
                    self.call_typing_cast(
                        &x.arguments.args,
                        &x.arguments.keywords,
                        x.arguments.range,
                        errors,
                    )
                }
                Some(CalleeKind::Function(FunctionKind::DataclassReplace)) => {
                    self.call_dataclasses_replace(
                        ty,
                        &args,
                        &kws,
                        x.func.range(),
                        x.arguments.range,
                        hint,
                        errors,
                    )
                }
                // Treat assert_type and reveal_type like pseudo-builtins for convenience. Note that we still
                // log a name-not-found error, but we also assert/reveal the type as requested.
                None if ty.is_error() && is_special_name(&x.func, "assert_type") => self
                    .call_assert_type(
                        &x.arguments.args,
                        &x.arguments.keywords,
                        x.arguments.range,
                        hint,
                        errors,
                    ),
                None if ty.is_error() && is_special_name(&x.func, "reveal_type") => self
                    .call_reveal_type(
                        &x.arguments.args,
                        &x.arguments.keywords,
                        x.arguments.range,
                        hint,
                        errors,
                    ),
                Some(CalleeKind::Function(FunctionKind::IsInstance))
                    if self.has_exactly_two_posargs(&x.arguments) =>
                {
                    self.call_isinstance(&x.arguments.args[0], &x.arguments.args[1], errors)
                }
                Some(CalleeKind::Function(FunctionKind::IsSubclass))
                    if self.has_exactly_two_posargs(&x.arguments) =>
                {
                    self.call_issubclass(&x.arguments.args[0], &x.arguments.args[1], errors)
                }
                _ if matches!(ty, Type::ClassDef(cls) if cls == self.stdlib.builtins_type().class_object())
                    && x.arguments.args.len() == 1 && x.arguments.keywords.is_empty() =>
                {
                    // We may be able to provide a more precise type when the constructor for `builtins.type`
                    // is called with a single argument.
                    let arg_ty = self.expr_infer(&x.arguments.args[0], errors);
                    self.type_of(arg_ty)
                }
                // Decorators can be applied in two ways:
                //   - (common, idiomatic) via `@decorator`:
                //     @staticmethod
                //     def f(): ...
                //   - (uncommon, mostly seen in legacy code) via a function call:
                //     def f(): ...
                //     f = staticmethod(f)
                // Check if this call applies a decorator with known typing effects to a function.
                _ if let Some(ret) = self.maybe_apply_function_decorator(ty, &args, &kws, errors) => ret,
                _ => self.freeform_call_infer(ty.clone(), &args, &kws, x.func.range(), x.arguments.range(), hint, errors),
            })
        }
    }

    pub fn freeform_call_infer(
        &self,
        ty: Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let callable =
            self.as_call_target_or_error(ty, CallStyle::FreeForm, callee_range, errors, None);
        self.call_infer_with_range(
            callable,
            args,
            kws,
            arg_range,
            Some(callee_range),
            errors,
            None,
            hint,
            None,
        )
    }

    fn has_exactly_two_posargs(&self, arguments: &Arguments) -> bool {
        arguments.keywords.is_empty()
            && arguments.args.len() == 2
            && arguments
                .args
                .iter()
                .all(|e| !matches!(e, Expr::Starred(_)))
    }
}

/// Match on an expression by name. Should be used only for special names that we essentially treat like keywords,
/// like reveal_type.
fn is_special_name(x: &Expr, name: &str) -> bool {
    match x {
        // Note that this matches on a bare name regardless of whether it's been imported.
        // It's convenient to be able to call functions like reveal_type in the course of
        // debugging without scrolling to the top of the file to add an import.
        Expr::Name(x) => x.id.as_str() == name,
        _ => false,
    }
}
