/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;

use dupe::Dupe;
use pyrefly_python::dunder;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::DescriptorBase;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::callable::CallWithTypes;
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
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::types::types::Var;

pub enum CallStyle<'a> {
    Method(&'a Name),
    FreeForm,
}

/// A pair of a call target (see Target) and the Quantifieds it uses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallTarget {
    qs: Vec<Var>,
    target: Target,
}

impl CallTarget {
    fn new(target: Target) -> Self {
        Self {
            qs: Vec::new(),
            target,
        }
    }

    fn forall(qs: Vec<Var>, target: Target) -> Self {
        Self { qs, target }
    }
}

/// A thing that can be called (see as_call_target and call_infer).
/// Note that a single "call" may invoke multiple functions under the hood,
/// e.g., `__new__` followed by `__init__` for Class.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Target {
    /// A typing.Callable.
    Callable(Callable),
    /// A function.
    Function(Function),
    /// Method of a class. The `Type` is the self/cls argument.
    BoundMethod(Type, Function),
    /// A class object.
    Class(ClassType),
    /// A TypedDict.
    TypedDict(TypedDict),
    /// An overloaded function.
    FunctionOverload(Vec1<Function>, FuncMetadata),
    /// An overloaded method.
    BoundMethodOverload(Type, Vec1<Function>, FuncMetadata),
    /// A union of call targets.
    Union(Vec<Target>),
    /// Any, as a call target.
    Any(AnyStyle),
}

impl Target {
    fn function_metadata(&self) -> Option<&FuncMetadata> {
        match self {
            Self::Function(func) | Self::BoundMethod(_, func) => Some(&func.metadata),
            Self::FunctionOverload(_, metadata) | Self::BoundMethodOverload(_, _, metadata) => {
                Some(metadata)
            }
            _ => None,
        }
    }
}

struct CalledOverload {
    signature: Callable,
    arg_errors: ErrorCollector,
    call_errors: ErrorCollector,
    return_type: Type,
    is_deprecated: bool,
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
        CallTarget::new(Target::Any(AnyStyle::Error))
    }

    fn fresh_quantified_function(&self, tparams: &TParams, func: Function) -> (Vec<Var>, Function) {
        let (qs, t) =
            self.solver()
                .fresh_quantified(tparams, Type::Function(Box::new(func)), self.uniques);
        match t {
            Type::Function(func) => (qs, *func),
            // We passed a Function to fresh_quantified(), so we know we get a Function back out.
            _ => unreachable!(),
        }
    }

    /// Return a pair of the quantified variables I had to instantiate, and the resulting call target.
    pub fn as_call_target(&self, ty: Type) -> Option<CallTarget> {
        match ty {
            Type::Callable(c) => Some(CallTarget::new(Target::Callable(*c))),
            Type::Function(func) => Some(CallTarget::new(Target::Function(*func))),
            Type::Overload(overload) => {
                let mut qs = Vec::new();
                let funcs = overload.signatures.mapped(|ty| match ty {
                    OverloadType::Callable(function) => function,
                    OverloadType::Forall(forall) => {
                        let (qs2, func) =
                            self.fresh_quantified_function(&forall.tparams, forall.body);
                        qs.extend(qs2);
                        func
                    }
                });
                Some(CallTarget::forall(
                    qs,
                    Target::FunctionOverload(funcs, *overload.metadata),
                ))
            }
            Type::BoundMethod(box BoundMethod { obj, mut func }) => {
                let self_replacement = if func.metadata().flags.is_classmethod
                    && let Type::ClassDef(c) = &obj
                {
                    &self.instantiate(c)
                } else {
                    &obj
                };
                func.subst_self_type_mut(self_replacement, &|a, b| self.is_subset_eq(a, b));
                match self.as_call_target(func.as_type()) {
                    Some(CallTarget {
                        qs,
                        target: Target::Function(func),
                    }) => Some(CallTarget::forall(qs, Target::BoundMethod(obj, func))),
                    Some(CallTarget {
                        qs,
                        target: Target::FunctionOverload(overloads, meta),
                    }) => Some(CallTarget::forall(
                        qs,
                        Target::BoundMethodOverload(obj, overloads, meta),
                    )),
                    _ => None,
                }
            }
            Type::ClassDef(cls) => {
                self.as_call_target(Type::type_form(self.instantiate_fresh(&cls)))
            }
            Type::Type(box Type::ClassType(cls)) | Type::Type(box Type::SelfType(cls)) => {
                Some(CallTarget::new(Target::Class(cls)))
            }
            Type::Type(box Type::Tuple(tuple)) => {
                Some(CallTarget::new(Target::Class(self.erase_tuple_type(tuple))))
            }
            Type::Type(box Type::Quantified(quantified)) => {
                Some(CallTarget::new(Target::Callable(Callable {
                    // TODO: use upper bound to determine input parameters
                    params: Params::Ellipsis,
                    ret: Type::Quantified(quantified),
                })))
            }
            Type::Type(box Type::Any(style)) => Some(CallTarget::new(Target::Any(style))),
            Type::Forall(forall) => {
                let (qs, t) = self.instantiate_forall(*forall);
                self.as_call_target(t)
                    .map(|x| CallTarget::forall(qs.into_iter().chain(x.qs).collect(), x.target))
            }
            Type::Var(v) if let Some(_guard) = self.recurser.recurse(v) => {
                self.as_call_target(self.solver().force_var(v))
            }
            Type::Union(xs) => {
                let res = xs
                    .into_iter()
                    .map(|x| self.as_call_target(x))
                    .collect::<Option<SmallSet<_>>>()?;
                if res.len() == 1 {
                    Some(res.into_iter().next().unwrap())
                } else {
                    let (qs, ts): (Vec<Vec<Var>>, Vec<Target>) =
                        res.into_iter().map(|x| (x.qs, x.target)).unzip();
                    let qs = qs.into_iter().flatten().collect();
                    Some(CallTarget::forall(qs, Target::Union(ts)))
                }
            }
            Type::Any(style) => Some(CallTarget::new(Target::Any(style))),
            Type::TypeAlias(ta) => self.as_call_target(ta.as_value(self.stdlib)),
            Type::ClassType(cls) | Type::SelfType(cls) => self
                .instance_to_method(&cls)
                .and_then(|ty| self.as_call_target(ty)),
            Type::Type(box Type::TypedDict(typed_dict)) => {
                Some(CallTarget::new(Target::TypedDict(typed_dict)))
            }
            // TODO: this is wrong, because we lose the information that this is a type variable
            // TODO: handle type[T]
            Type::Quantified(q) if q.is_type_var() => match q.restriction() {
                Restriction::Bound(bound) => self.as_call_target(bound.clone()),
                // TODO: handle constraints
                Restriction::Constraints(_) | Restriction::Unrestricted => None,
            },
            Type::KwCall(call) => self.as_call_target(call.return_ty),
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
            Some(target) => target,
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
                    format!("{}, got {}", expect_message, self.for_display(ty)),
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
        self.call_infer(call_target, args, keywords, range, errors, context, None)
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
        hint: Option<&Type>,
    ) -> Option<Type> {
        let dunder_call = self.get_metaclass_dunder_call(cls)?;
        Some(self.call_infer(
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
        ))
    }

    fn construct_class(
        &self,
        cls: ClassType,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<&Type>,
    ) -> Type {
        // Based on https://typing.readthedocs.io/en/latest/spec/constructors.html.
        let instance_ty = Type::ClassType(cls.clone());
        let mut overall_ret = None;
        if let Some(ret) = self.call_metaclass(&cls, range, args, keywords, errors, context, hint)
            && !self.is_compatible_constructor_return(&ret, cls.class_object())
        {
            // Got something other than an instance of the class under construction.
            return ret;
        }
        let (overrides_new, dunder_new_has_errors) =
            if let Some(new_method) = self.get_dunder_new(&cls) {
                let cls_ty = Type::type_form(instance_ty.clone());
                let full_args = iter::once(CallArg::ty(&cls_ty, range))
                    .chain(args.iter().cloned())
                    .collect::<Vec<_>>();
                let dunder_new_errors = self.error_collector();
                let ret = self.call_infer(
                    self.as_call_target_or_error(
                        new_method,
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
                );
                let has_errors = !dunder_new_errors.is_empty();
                errors.extend(dunder_new_errors);
                if self.is_compatible_constructor_return(&ret, cls.class_object()) {
                    overall_ret = Some(ret);
                } else {
                    // Got something other than an instance of the class under construction.
                    return ret;
                }
                (true, has_errors)
            } else {
                (false, false)
            };

        // If the class overrides `object.__new__` but not `object.__init__`, the `__init__` call
        // always succeeds at runtime, so we skip analyzing it.
        // If we have Any as a base class, we shouldn't fall back to `object.__init__`.
        let get_object_init = !overrides_new
            && !self
                .get_metadata_for_class(cls.class_object())
                .has_base_any();

        if let Some(init_method) = self.get_dunder_init(&cls, get_object_init) {
            let dunder_init_errors = self.error_collector();
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
                &dunder_init_errors,
                context,
                hint,
            );
            // Report `__init__` errors only when there are no `__new__` errors, to avoid redundant errors.
            if !dunder_new_has_errors {
                errors.extend(dunder_init_errors);
            }
        }
        if let Some(mut ret) = overall_ret {
            ret.subst_self_type_mut(&instance_ty, &|_, _| true);
            ret
        } else {
            instance_ty
        }
    }

    fn construct_typed_dict(
        &self,
        typed_dict: TypedDict,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<&Type>,
    ) -> Type {
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
        );
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
        hint: Option<&Type>,
    ) -> Type {
        let metadata = call_target.target.function_metadata();
        if let Some(m) = metadata
            && m.flags.is_deprecated
        {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::Deprecated, context),
                format!(
                    "Call to deprecated function `{}`",
                    m.kind.as_func_id().format(self.module().name())
                ),
            );
        }
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
        let res = match call_target.target {
            Target::Class(cls) => {
                if let Some(hint) = hint {
                    // If a hint is provided, use it to bind any variables in the return type
                    // We only care about the side effect here, not the result
                    self.is_subset_eq(&Type::ClassType(cls.clone()), hint);
                }
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
            Target::TypedDict(td) => {
                if let Some(hint) = hint {
                    // If a hint is provided, use it to bind any variables in the return type
                    // We only care about the side effect here, not the result
                    self.is_subset_eq(&Type::TypedDict(td.clone()), hint);
                }
                self.construct_typed_dict(td, args, keywords, range, errors, context, hint)
            }
            Target::BoundMethod(
                obj,
                Function {
                    signature,
                    metadata,
                },
            ) => self.callable_infer(
                signature,
                Some(metadata.kind.as_func_id()),
                Some(obj),
                args,
                keywords,
                range,
                errors,
                errors,
                context,
                hint,
            ),
            Target::Callable(callable) => self.callable_infer(
                callable, None, None, args, keywords, range, errors, errors, context, hint,
            ),
            Target::Function(Function {
                signature: mut callable,
                metadata,
            }) => {
                // Most instances of typing.Self are replaced in as_call_target, but __new__ is a
                // staticmethod, so we don't have access to the first argument until we get here.
                let id = metadata.kind.as_func_id();
                let first_arg_type = if id.func == dunder::NEW {
                    self.first_arg_type(args, errors)
                } else {
                    None
                };
                let self_obj = match first_arg_type {
                    Some(Type::Type(box Type::ClassType(c))) => Some(c.to_type()),
                    Some(Type::ClassDef(class)) => {
                        Some(self.as_class_type_unchecked(&class).to_type())
                    }
                    _ => None,
                };
                if let Some(self_obj) = self_obj {
                    callable.subst_self_type_mut(&self_obj, &|a, b| self.is_subset_eq(a, b));
                }
                self.callable_infer(
                    callable,
                    Some(id),
                    None,
                    args,
                    keywords,
                    range,
                    errors,
                    errors,
                    context,
                    hint,
                )
            }
            Target::FunctionOverload(overloads, meta) => {
                self.call_overloads(
                    overloads, meta, None, args, keywords, range, errors, context, hint,
                )
                .0
            }
            Target::BoundMethodOverload(obj, overloads, meta) => {
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
                )
                .0
            }
            Target::Union(targets) => {
                let call = CallWithTypes::new();
                let args = call.vec_call_arg(args, self, errors);
                let keywords = call.vec_call_keyword(keywords, self, errors);
                self.unions(targets.into_map(|t| {
                    self.call_infer(
                        CallTarget::new(t),
                        &args,
                        &keywords,
                        range,
                        errors,
                        context,
                        hint,
                    )
                }))
            }
            Target::Any(style) => {
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
        self.solver().finish_quantified(&call_target.qs);
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
        overloads: Vec1<Function>,
        metadata: FuncMetadata,
        self_obj: Option<Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<&Type>,
    ) -> (Type, Callable) {
        // There may be Expr values in args and keywords.
        // If we infer them for each overload, we may end up infering them multiple times.
        // If those overloads contain nested overloads, then we can easily end up with O(2^n) perf.
        // Therefore, flatten all TypeOrExpr's into Type before we start
        let call = CallWithTypes::new();
        let method_name = metadata.kind.as_func_id().func;
        let args = call.vec_call_arg(args, self, errors);
        let keywords = call.vec_call_keyword(keywords, self, errors);

        let mut closest_overload: Option<CalledOverload> = None;
        for callable in overloads.iter() {
            let arg_errors = self.error_collector();
            let call_errors = self.error_collector();
            let res = self.callable_infer(
                callable.signature.clone(),
                Some(metadata.kind.as_func_id()),
                self_obj.clone(),
                &args,
                &keywords,
                range,
                &arg_errors,
                &call_errors,
                // We intentionally drop the context here, as arg errors don't need it,
                // and if there are any call errors, we'll log a "No matching overloads"
                // error with the necessary context.
                None,
                hint,
            );
            if arg_errors.is_empty() && call_errors.is_empty() {
                // An overload is chosen, we should record it to power IDE services.
                self.record_overload_trace(
                    range,
                    overloads.map(|Function { signature, .. }| signature),
                    &callable.signature,
                    true,
                );
                // It's only safe to return immediately if both arg_errors and call_errors are
                // empty, as parameter types from the overload signature may be used as hints when
                // evaluating arguments, producing arg_errors for some overloads but not others.
                // See test::overload::test_pass_generic_class_to_overload for an example.

                // If the selected overload is deprecated, we log a deprecation error.
                if callable.metadata.flags.is_deprecated {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::new(ErrorKind::Deprecated, context),
                        format!("Call to deprecated overload `{method_name}`"),
                    );
                }
                return (res, callable.signature.clone());
            }
            let called_overload = CalledOverload {
                signature: callable.signature.clone(),
                arg_errors,
                call_errors,
                return_type: res,
                is_deprecated: callable.metadata.flags.is_deprecated,
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
            overloads.map(|Function { signature, .. }| signature),
            &closest_overload.signature,
            false,
        );
        errors.extend(closest_overload.arg_errors);
        if closest_overload.call_errors.is_empty() {
            // No overload evaluated completely successfully, but we still say we found a match if
            // there were only arg_errors, since they may be unrelated. For example, in:
            //
            //   @overload
            //   def f(x: int) -> int: ...
            //   @overload
            //   def f(x: int, y: str) -> str: ...
            //
            //   f(1+"2")
            //
            // the call to f should match the first overload, even though `1 + "2"` generates an
            // arg error for both overloads.

            // If the closest overload is deprecated, we log a deprecation error.
            if closest_overload.is_deprecated {
                self.error(
                    errors,
                    range,
                    ErrorInfo::new(ErrorKind::Deprecated, context),
                    format!("Call to deprecated overload `{method_name}`"),
                );
            }
            (closest_overload.return_type, closest_overload.signature)
        } else {
            let mut msg = vec1![
                format!(
                    "No matching overload found for function `{}`",
                    metadata.kind.as_func_id().format(self.module().name())
                ),
                "Possible overloads:".to_owned(),
            ];
            for overload in overloads {
                let suffix = if overload.signature == closest_overload.signature {
                    " [closest match]"
                } else {
                    ""
                };
                let signature = match self_obj {
                    Some(_) => overload
                        .signature
                        .drop_first_param()
                        .unwrap_or(overload.signature),
                    None => overload.signature,
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
        self.call_infer(call_target, &[], &[], range, errors, context, None)
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
        self.call_infer(call_target, &[got], &[], range, errors, context, None)
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
        self.call_infer(call_target, &args, &[], range, errors, context, None)
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
        self.call_infer(call_target, &args, &[], range, errors, context, None)
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
            .and_then(|t| t.drop_first_param_of_unbound_callable())
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
