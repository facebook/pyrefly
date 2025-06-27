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

use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::error::collector::ErrorCollector;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::kind::ErrorKind;
use crate::types::callable::FunctionKind;
use crate::types::callable::unexpected_keyword;
use crate::types::class::Class;
use crate::types::special_form::SpecialForm;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn call_assert_type(
        &self,
        args: &[Expr],
        keywords: &[Keyword],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if args.len() == 2 {
            let expr_a = &args[0];
            let expr_b = &args[1];
            let a = self.expr_infer(expr_a, errors);
            let b = self.expr_untype(expr_b, TypeFormContext::FunctionArgument, errors);
            let mut a = self
                .canonicalize_all_class_types(self.solver().deep_force(a), expr_a.range())
                .explicit_any()
                .noreturn_to_never()
                .anon_callables();
            let mut b = self
                .canonicalize_all_class_types(self.solver().deep_force(b), expr_b.range())
                .explicit_any()
                .noreturn_to_never()
                .anon_callables();
            // Make assert_type(Self@SomeClass, typing.Self) work.
            let self_form = Type::SpecialForm(SpecialForm::SelfType);
            a.subst_self_type_mut(&self_form, &|_, _| true);
            b.subst_self_type_mut(&self_form, &|_, _| true);
            if a != b {
                self.error(
                    errors,
                    range,
                    ErrorKind::AssertType,
                    None,
                    format!(
                        "assert_type({}, {}) failed",
                        self.for_display(a),
                        self.for_display(b)
                    ),
                );
            }
        } else {
            self.error(
                errors,
                range,
                ErrorKind::BadArgumentCount,
                None,
                format!(
                    "assert_type needs 2 positional arguments, got {:#?}",
                    args.len()
                ),
            );
        }
        for keyword in keywords {
            unexpected_keyword(
                &|msg| {
                    self.error(errors, range, ErrorKind::UnexpectedKeyword, None, msg);
                },
                "assert_type",
                keyword,
            );
        }
        Type::None
    }

    pub fn call_reveal_type(
        &self,
        args: &[Expr],
        keywords: &[Keyword],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if args.len() == 1 {
            let mut type_info = self.expr_infer_type_info(&args[0], errors);
            type_info.visit_mut(&mut |ty| {
                *ty = self.for_display(ty.clone());
            });
            self.error(
                errors,
                range,
                ErrorKind::RevealType,
                None,
                format!("revealed type: {}", type_info),
            );
        } else {
            self.error(
                errors,
                range,
                ErrorKind::BadArgumentCount,
                None,
                format!(
                    "reveal_type needs 1 positional argument, got {}",
                    args.len()
                ),
            );
        }
        for keyword in keywords {
            unexpected_keyword(
                &|msg| {
                    self.error(errors, range, ErrorKind::UnexpectedKeyword, None, msg);
                },
                "reveal_type",
                keyword,
            );
        }
        Type::None
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
                            ErrorKind::InvalidArgument,
                            None,
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
                            ErrorKind::InvalidArgument,
                            None,
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
                ErrorKind::BadArgumentCount,
                None,
                format!("`typing.cast` expected 2 arguments, got {}", extra + 2),
            );
        }
        let ret = if let Some(t) = typ {
            match self.untype_opt(self.expr_infer(t, errors), range) {
                Some(t) => t,
                None => self.error(
                    errors,
                    range,
                    ErrorKind::BadArgumentType,
                    None,
                    "First argument to `typing.cast` must be a type".to_owned(),
                ),
            }
        } else {
            self.error(
                errors,
                range,
                ErrorKind::MissingArgument,
                None,
                "`typing.cast` missing required argument `typ`".to_owned(),
            )
        };
        if val.is_none() {
            self.error(
                errors,
                range,
                ErrorKind::MissingArgument,
                None,
                "`typing.cast` missing required argument `val`".to_owned(),
            );
        }
        ret
    }

    pub fn call_isinstance(
        &self,
        obj: &Expr,
        class_or_tuple: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        // We call expr_infer in order to check for errors, but we don't need to do anything with
        // the result, as the `obj` parameter has type `object`.
        self.expr_infer(obj, errors);
        self.check_arg_is_class_object(class_or_tuple, &FunctionKind::IsInstance, errors);
        self.stdlib.bool().clone().to_type()
    }

    pub fn call_issubclass(
        &self,
        cls: &Expr,
        class_or_tuple: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        // Verify that the `cls` argument has type `type`.
        self.check_type(
            &self.stdlib.builtins_type().clone().to_type(),
            &self.expr_infer(cls, errors),
            cls.range(),
            errors,
            &|| {
                TypeCheckContext::of_kind(TypeCheckKind::CallArgument(
                    Some(Name::new_static("cls")),
                    Some(FunctionKind::IsSubclass.as_func_id()),
                ))
            },
        );
        self.check_arg_is_class_object(class_or_tuple, &FunctionKind::IsSubclass, errors);
        self.stdlib.bool().clone().to_type()
    }

    fn check_type_is_class_object(
        &self,
        ty: Type,
        contains_subscript: bool,
        range: TextRange,
        func_kind: &FunctionKind,
        errors: &ErrorCollector,
    ) {
        if let Some(ts) = ty.as_decomposed_tuple_or_union(self.stdlib) {
            for t in ts {
                self.check_type_is_class_object(t, contains_subscript, range, func_kind, errors);
            }
        } else if let Type::ClassDef(cls) = &ty {
            let metadata = self.get_metadata_for_class(cls);
            let func_display = || {
                format!(
                    "{}()",
                    func_kind.as_func_id().format(self.module_info().name())
                )
            };
            if metadata.is_new_type() {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidArgument,
                    None,
                    format!("NewType `{}` not allowed in {}", cls.name(), func_display(),),
                );
            }
            // Check if this is a TypedDict
            if metadata.is_typed_dict() {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidArgument,
                    None,
                    format!(
                        "TypedDict `{}` not allowed as second argument to {}",
                        cls.name(),
                        func_display()
                    ),
                );
            }
            // Check if this is a protocol that needs @runtime_checkable
            if metadata.is_protocol() && !metadata.is_runtime_checkable_protocol() {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidArgument,
                    None,
                    format!("Protocol `{}` is not decorated with @runtime_checkable and cannot be used with {}", cls.name(), func_display()),
                );
            } else if metadata.is_protocol() && metadata.is_runtime_checkable_protocol() {
                // Additional validation for runtime checkable protocols:
                // issubclass() can only be used with non-data protocols
                if *func_kind == FunctionKind::IsSubclass && self.is_data_protocol(cls, range) {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidArgument,
                        None,
                        format!("Protocol `{}` has non-method members and cannot be used with issubclass()", cls.name()),
                    );
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
                ErrorKind::InvalidArgument,
                None,
                format!(
                    "Expected class object, got parameterized generic type: `{}`",
                    self.for_display(ty)
                ),
            );
        } else if self.unwrap_class_object_silently(&ty).is_none() {
            self.error(
                errors,
                range,
                ErrorKind::InvalidArgument,
                None,
                format!("Expected class object, got `{}`", self.for_display(ty)),
            );
        } else {
            self.check_type(
                &self.stdlib.builtins_type().clone().to_type(),
                &ty,
                range,
                errors,
                &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::CallArgument(
                        Some(Name::new_static("class_or_tuple")),
                        Some(func_kind.as_func_id()),
                    ))
                },
            );
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
                if !ty.is_function_type() {
                    return true;
                }
            }
        }
        false
    }

    fn check_arg_is_class_object(
        &self,
        arg_expr: &Expr,
        func_kind: &FunctionKind,
        errors: &ErrorCollector,
    ) {
        let arg_class_type = self.expr_infer(arg_expr, errors);
        let mut contains_subscript = false;
        arg_expr.visit(&mut |e| {
            if matches!(e, Expr::Subscript(_)) {
                contains_subscript = true;
            }
        });

        self.check_type_is_class_object(
            arg_class_type,
            contains_subscript,
            arg_expr.range(),
            func_kind,
            errors,
        );
    }
}
