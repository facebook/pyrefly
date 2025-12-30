/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Most function calls are resolved by converting the callee to a CallTarget and
 * calling AnswersSolver::call_infer with the call target and the arguments. This
 * file contains the implementations of a few special calls that need to be hard-coded.
 */

use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::types::Union;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::decorated_function::Decorator;
use crate::alt::unwrap::HintRef;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::callable::FunctionKind;
use crate::types::callable::unexpected_keyword;
use crate::types::class::Class;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::types::AnyStyle;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn call_assert_type(
        &self,
        args: &[Expr],
        keywords: &[Keyword],
        range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let ret = if args.len() == 2 {
            let expr_a = &args[0];
            let expr_b = &args[1];
            let a = self.expr_infer_with_hint(expr_a, hint, errors);
            let b = self.expr_untype(expr_b, TypeFormContext::FunctionArgument, errors);
            let self_form = Type::SpecialForm(SpecialForm::SelfType);
            let normalize_type = |ty: Type, expr: &Expr| {
                let mut ty = self
                    .canonicalize_all_class_types(
                        self.solver().deep_force(ty),
                        expr.range(),
                        errors,
                    )
                    .promote_typevar_values(self.stdlib)
                    .explicit_any()
                    .noreturn_to_never()
                    .anon_callables()
                    .anon_typed_dicts(self.stdlib)
                    .distribute_type_over_union();
                // Make assert_type(Self@SomeClass, typing.Self) work.
                ty.subst_self_type_mut(&self_form);
                // Re-sort unions & drop any display names.
                // Make sure to keep this as the final step before comparison.
                ty.sort_unions_and_drop_names()
            };
            let a = normalize_type(a, expr_a);
            let b = normalize_type(b, expr_b);
            if a != b {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::AssertType),
                    format!(
                        "assert_type({}, {}) failed",
                        self.for_display(a.clone()),
                        self.for_display(b)
                    ),
                );
            }
            a
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadArgumentCount),
                format!(
                    "assert_type needs 2 positional arguments, got {:#?}",
                    args.len()
                ),
            );
            Type::any_error()
        };
        for keyword in keywords {
            unexpected_keyword(
                &|msg| {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::UnexpectedKeyword),
                        msg,
                    );
                },
                "assert_type",
                keyword,
            );
        }
        ret
    }

    pub fn call_reveal_type(
        &self,
        args: &[Expr],
        keywords: &[Keyword],
        range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let ret = if args.len() == 1 {
            let mut type_info = self.expr_infer_type_info_with_hint(&args[0], hint, errors);
            let ret = type_info.ty().clone();
            type_info.visit_mut(&mut |ty| {
                *ty = self.for_display(ty.clone());
            });
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::RevealType),
                format!("revealed type: {type_info}"),
            );
            ret
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadArgumentCount),
                format!(
                    "reveal_type needs 1 positional argument, got {}",
                    args.len()
                ),
            );
            Type::any_error()
        };
        for keyword in keywords {
            unexpected_keyword(
                &|msg| {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::UnexpectedKeyword),
                        msg,
                    );
                },
                "reveal_type",
                keyword,
            );
        }
        ret
    }

    /// Simulates a call to `typing.cast`, whose signature is
    /// `(typ: type[T], val: Any) -> T: ...`
    /// (ignoring corner cases like special forms and forward references).
    /// The actual definition has additional overloads to accommodate said corner
    /// cases, with imprecise return types, which is why we need to hard-code this.
    pub fn call_typing_cast(
        &self,
        args: &[Expr],
        keywords: &[Keyword],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let mut typ = None;
        let mut val = None;
        let mut extra = 0;
        match args {
            [] => {}
            [arg1] => {
                typ = Some(arg1);
            }
            [arg1, arg2, tail @ ..] => {
                typ = Some(arg1);
                val = Some(arg2);
                extra += tail.len();
            }
        }
        for keyword in keywords {
            match keyword.arg.as_ref().map(|id| id.as_str()) {
                Some("typ") => {
                    if typ.is_some() {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "`typing.cast` got multiple values for argument `typ`".to_owned(),
                        );
                    }
                    typ = Some(&keyword.value);
                }
                Some("val") => {
                    if val.is_some() {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "`typing.cast` got multiple values for argument `val`".to_owned(),
                        );
                    }
                    val = Some(&keyword.value);
                }
                _ => {
                    extra += 1;
                }
            }
        }
        if extra > 0 {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadArgumentCount),
                format!("`typing.cast` expected 2 arguments, got {}", extra + 2),
            );
        }
        let ret = if let Some(t) = typ {
            match self.untype_opt(self.expr_infer(t, errors), range, errors) {
                Some(t) => t,
                None => self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadArgumentType),
                    "First argument to `typing.cast` must be a type".to_owned(),
                ),
            }
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::MissingArgument),
                "`typing.cast` missing required argument `typ`".to_owned(),
            )
        };
        if val.is_none() {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::MissingArgument),
                "`typing.cast` missing required argument `val`".to_owned(),
            );
        }
        if let Some(val_expr) = val {
            let val_type = self.expr_infer(val_expr, errors);
            if !val_type.is_any() && val_type == ret {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::RedundantCast),
                    format!(
                        "Redundant cast: `{}` is the same type as `{}`",
                        val_type.deterministic_printing(),
                        ret.clone().deterministic_printing()
                    ),
                );
            }
        }
        ret
    }

    pub fn call_isinstance(
        &self,
        obj: &Expr,
        class_or_tuple: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        self.check_arg_is_class_object(obj, class_or_tuple, &FunctionKind::IsInstance, errors);
        self.stdlib.bool().clone().to_type()
    }

    pub fn call_issubclass(
        &self,
        cls: &Expr,
        class_or_tuple: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        self.check_arg_is_class_object(cls, class_or_tuple, &FunctionKind::IsSubclass, errors);
        self.stdlib.bool().clone().to_type()
    }

    fn check_type_is_class_object(
        &self,
        ty: Type,
        object_type: Option<Type>,
        contains_subscript: bool,
        range: TextRange,
        func_kind: &FunctionKind,
        errors: &ErrorCollector,
    ) {
        for ty in self.as_class_info(ty) {
            if let Type::ClassDef(cls) = &ty {
                if cls.has_toplevel_qname("typing", "Any") {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
                        "Expected class object, got `Any`".to_owned(),
                    );
                }
                let metadata = self.get_metadata_for_class(cls);
                let func_display = || format!("{}()", func_kind.format(self.module().name()));
                if metadata.is_new_type() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
                        format!("NewType `{}` not allowed in {}", cls.name(), func_display(),),
                    );
                }
                // Check if this is a TypedDict
                if metadata.is_typed_dict() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
                        format!(
                            "TypedDict `{}` not allowed as second argument to {}",
                            cls.name(),
                            func_display()
                        ),
                    );
                }
                // Check if this is a protocol that needs @runtime_checkable
                if metadata.is_protocol() && !metadata.is_typed_dict() {
                    if !metadata.is_runtime_checkable_protocol() {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            format!("Protocol `{}` is not decorated with @runtime_checkable and cannot be used with {}", cls.name(), func_display()),
                        );
                    } else {
                        // Additional validation for runtime checkable protocols:
                        // issubclass() can only be used with non-data protocols
                        if *func_kind == FunctionKind::IsSubclass
                            && self.is_data_protocol(cls, range)
                        {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                format!("Protocol `{}` has non-method members and cannot be used with issubclass()", cls.name()),
                            );
                        }
                        // Check for unsafe overlap:
                        // https://typing.python.org/en/latest/spec/protocol.html#runtime-checkable-decorator-and-narrowing-types-by-isinstance
                        // We need to check if there is any field with unassignable types, since the `isinstance` check only
                        // checks for the presence of the fields, not their types.
                        //
                        // Type arguments for the protocol are not provided, so we'll use
                        // fresh vars and solve them during the `is_subset_eq` check below.
                        let protocol_metadata = metadata.protocol_metadata().unwrap();
                        if let Some(object_type) = &object_type
                            && let Type::ClassType(protocol_class_type) =
                                self.instantiate_fresh_class(cls)
                        {
                            let mut unsafe_overlap_errors = vec![];
                            for field_name in &protocol_metadata.members {
                                if !self.has_attr(object_type, field_name) {
                                    // It's okay if the field is missing, since
                                    // we only care about unsafe overlaps
                                    continue;
                                }
                                if let Err(subset_err) = self.is_protocol_subset_at_attr(
                                    object_type,
                                    &protocol_class_type,
                                    field_name,
                                    &mut |x, y| self.is_subset_eq_with_reason(x, y),
                                ) {
                                    let error_msg = subset_err
                                        .to_error_msg()
                                        .map(|msg| format!(": {msg}"))
                                        .unwrap_or_default();
                                    unsafe_overlap_errors.push(format!(
                                        "Attribute `{}` has incompatible types{}",
                                        field_name, error_msg,
                                    ));
                                }
                            }
                            if !unsafe_overlap_errors.is_empty() {
                                let mut full_msg = vec1![format!(
                                    "Runtime checkable protocol `{}` has an unsafe overlap with type `{}`",
                                    cls.name(),
                                    self.for_display(object_type.clone())
                                )];
                                full_msg.extend(unsafe_overlap_errors);
                                errors.add(
                                    range,
                                    ErrorInfo::Kind(ErrorKind::UnsafeOverlap),
                                    full_msg,
                                );
                            }
                        }
                    }
                }
            } else if contains_subscript
                && matches!(&ty, Type::Type(box Type::ClassType(cls)) if !cls.targs().is_empty())
            {
                // If the raw expression contains something that structurally looks like `A[T]` and
                // part of the expression resolves to a parameterized class type, then we likely have a
                // literal parameterized type, which is a runtime exception.
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    format!(
                        "Expected class object, got parameterized generic type: `{}`",
                        self.for_display(ty)
                    ),
                );
            } else if self.unwrap_class_object_silently(&ty).is_none() {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                    format!("Expected class object, got `{}`", self.for_display(ty)),
                );
            } else {
                self.check_type(
                    &ty,
                    &self.stdlib.builtins_type().clone().to_type(),
                    range,
                    errors,
                    &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::CallArgument(
                            Some(Name::new_static("class_or_tuple")),
                            Some(func_kind.clone()),
                        ))
                    },
                );
            }
        }
    }

    /// Check if a protocol is a data protocol (has non-method members)
    fn is_data_protocol(&self, cls: &Class, range: TextRange) -> bool {
        // A data protocol has at least one non-method member
        // Use protocol metadata to get the member names
        let metadata = self.get_metadata_for_class(cls);
        if let Some(protocol_metadata) = metadata.protocol_metadata() {
            for field_name in &protocol_metadata.members {
                // Use the class type to access the field
                let class_type = self.as_class_type_unchecked(cls);
                let ty = self.type_of_attr_get(
                    &class_type.to_type(),
                    field_name,
                    range,
                    &self.error_swallower(),
                    None,
                    "is_data_protocol",
                );

                // If it's not a callable type, it's a data member
                if !ty.is_toplevel_callable() {
                    return true;
                }
            }
        }
        false
    }

    // isinstance(object, classinfo) / issubclass(class, classinfo)
    fn check_arg_is_class_object(
        &self,
        object_or_class_expr: &Expr,
        classinfo_expr: &Expr,
        func_kind: &FunctionKind,
        errors: &ErrorCollector,
    ) {
        let classinfo_type = self.expr_infer(classinfo_expr, errors);
        let mut contains_subscript = false;
        classinfo_expr.visit(&mut |e| {
            if matches!(e, Expr::Subscript(_)) {
                contains_subscript = true;
            }
        });

        let object_type = if matches!(func_kind, FunctionKind::IsInstance) {
            Some(self.expr_infer(object_or_class_expr, errors))
        } else if matches!(func_kind, FunctionKind::IsSubclass) {
            let ty = self.expr_infer(object_or_class_expr, errors);
            // Verify that the `cls` argument has type `type`.
            self.check_type(
                &ty,
                &self.stdlib.builtins_type().clone().to_type(),
                object_or_class_expr.range(),
                errors,
                &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::CallArgument(
                        Some(Name::new_static("cls")),
                        Some(FunctionKind::IsSubclass),
                    ))
                },
            );
            // Untype to get the class object type
            self.untype_opt(ty, object_or_class_expr.range(), errors)
        } else {
            unreachable!("unexpected function kind in check_arg_is_class_object")
        };

        self.check_type_is_class_object(
            classinfo_type,
            object_type,
            contains_subscript,
            classinfo_expr.range(),
            func_kind,
            errors,
        );
    }

    /// Returns the list of types passed as the second argument to `isinstance` or `issubclass`.
    pub fn as_class_info(&self, ty: Type) -> Vec<Type> {
        fn f<'a, Ans: LookupAnswer>(me: &AnswersSolver<'a, Ans>, t: Type, res: &mut Vec<Type>) {
            match t {
                Type::Var(v) if let Some(_guard) = me.recurse(v) => {
                    f(me, me.solver().force_var(v), res)
                }
                Type::ClassType(ref c)
                    if let [arg] = c.targs().as_slice()
                        && c.class_object() == me.stdlib.tuple_object() =>
                {
                    f(me, arg.clone(), res)
                }
                Type::ClassType(ref c) if Some(c) == me.stdlib.union_type() => {
                    // Could be anything inside here, so add in Any.
                    res.push(Type::Any(AnyStyle::Implicit));
                }
                Type::Tuple(Tuple::Concrete(ts)) | Type::Union(box Union { members: ts, .. }) => {
                    for t in ts {
                        f(me, t, res)
                    }
                }
                Type::Tuple(Tuple::Unbounded(box t)) => f(me, t, res),
                Type::Tuple(Tuple::Unpacked(box (pre, mid, post))) => {
                    for t in pre {
                        f(me, t, res)
                    }
                    f(me, mid, res);
                    for t in post {
                        f(me, t, res)
                    }
                }
                Type::Type(box Type::Union(box Union { members: ts, .. })) => {
                    for t in ts {
                        f(me, Type::type_form(t), res)
                    }
                }
                Type::TypeAlias(ta) => f(me, ta.as_value(me.stdlib), res),
                _ => res.push(t),
            }
        }
        let mut res = Vec::new();
        f(self, ty, &mut res);
        res
    }

    pub fn maybe_apply_function_decorator(
        &self,
        callee: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let decorator = Decorator {
            ty: callee.clone(),
            deprecation: None,
        };
        let special_decorator = self.get_special_decorator(&decorator)?;
        // Does this call have a single positional argument?
        // If not, it cannot be a decorator application.
        if kws.is_empty()
            && let [CallArg::Arg(arg)] = args
        {
            let mut arg_ty = arg.infer(self, errors);
            // Try to apply the decorator to arg_ty. Does nothing if the decorator does not have known
            // typing effects or if arg_ty is not a function.
            let mut applied = false;
            arg_ty.transform_toplevel_func_metadata(|meta: &mut FuncMetadata| {
                applied |=
                    self.set_flag_from_special_decorator(&mut meta.flags, &special_decorator);
            });
            if applied { Some(arg_ty) } else { None }
        } else {
            None
        }
    }
}
