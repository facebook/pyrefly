/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice;

use pyrefly_types::literal::LitStyle;
use pyrefly_types::types::Union;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::lit_int::LitInt;
use crate::types::literal::Lit;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::types::AnyStyle;
use crate::types::types::Type;

fn is_chained_attribute_access(x: &Expr) -> bool {
    match x {
        Expr::Name(_) => true,
        Expr::Attribute(ExprAttribute { value, .. }) => is_chained_attribute_access(value),
        _ => false,
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn extra_unpack_error(&self, errors: &ErrorCollector, range: TextRange) -> Type {
        self.error(
            errors,
            range,
            ErrorInfo::Kind(ErrorKind::BadUnpacking),
            "Only one unbounded type is allowed to be unpacked".to_owned(),
        )
    }

    // Check if type args can be used to construct a valid tuple type
    // If successful, returns the constructed tuple along with whether any arguments were unpacked
    // Otherwise, records an error and return None
    fn check_args_and_construct_tuple(
        &self,
        arguments: &[Expr],
        errors: &ErrorCollector,
    ) -> Option<(Tuple, bool)> {
        let mut prefix: Vec<Type> = Vec::new();
        let mut suffix: Vec<Type> = Vec::new();
        let mut middle: Option<Type> = None;
        let mut has_unpack = false;
        for value in arguments {
            if matches!(value, Expr::EllipsisLiteral(_)) {
                if let [t] = prefix.as_slice()
                    && middle.is_none()
                    && arguments.len() == 2
                {
                    if has_unpack || t.is_unpack() {
                        self.error(
                            errors,
                            value.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            "`...` cannot be used with an unpacked `TypeVarTuple` or tuple"
                                .to_owned(),
                        );
                        return None;
                    } else {
                        return Some((Tuple::Unbounded(Box::new(t.clone())), false));
                    }
                } else {
                    self.error(
                        errors,
                        value.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
                        "Invalid position for `...`".to_owned(),
                    );
                    return None;
                }
            }
            let ty = self.expr_untype(value, TypeFormContext::TupleOrCallableParam, errors);
            match ty {
                Type::Unpack(box Type::Tuple(Tuple::Concrete(elts))) => {
                    has_unpack = true;
                    if middle.is_none() {
                        prefix.extend(elts)
                    } else {
                        suffix.extend(elts)
                    }
                }
                Type::Unpack(box ty @ Type::Tuple(Tuple::Unbounded(_))) => {
                    has_unpack = true;
                    if middle.is_none() {
                        middle = Some(ty)
                    } else {
                        self.extra_unpack_error(errors, value.range());
                        return None;
                    }
                }
                Type::Unpack(box Type::Tuple(Tuple::Unpacked(box (pre, mid, suff)))) => {
                    has_unpack = true;
                    if middle.is_none() {
                        prefix.extend(pre);
                        middle = Some(mid);
                        suffix.extend(suff)
                    } else {
                        self.extra_unpack_error(errors, value.range());
                        return None;
                    }
                }
                Type::Unpack(ty) if ty.is_kind_type_var_tuple() => {
                    has_unpack = true;
                    if middle.is_none() {
                        middle = Some(*ty)
                    } else {
                        self.extra_unpack_error(errors, value.range());
                        return None;
                    }
                }
                Type::Unpack(ty) => {
                    self.error(
                        errors,
                        value.range(),
                        ErrorInfo::Kind(ErrorKind::BadUnpacking),
                        format!("Expected a tuple or `TypeVarTuple`, got `{ty}`"),
                    );
                    return None;
                }
                ty if ty.is_kind_type_var_tuple() => {
                    self.error(
                        errors,
                        value.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                        "`TypeVarTuple` must be unpacked".to_owned(),
                    );
                    return None;
                }
                _ => {
                    if middle.is_none() {
                        prefix.push(ty)
                    } else {
                        suffix.push(ty)
                    }
                }
            }
        }
        if let Some(middle) = middle {
            Some((Tuple::unpacked(prefix, middle, suffix), has_unpack))
        } else {
            Some((Tuple::Concrete(prefix), has_unpack))
        }
    }

    fn apply_literal(&self, x: &Expr, errors: &ErrorCollector, literals: &mut Vec<Type>) {
        match x {
            Expr::UnaryOp(ExprUnaryOp {
                op: UnaryOp::UAdd,
                operand,
                ..
            }) if let Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(i),
                ..
            }) = &**operand =>
            {
                literals.push(LitInt::from_ast(i).to_explicit_type())
            }
            Expr::UnaryOp(ExprUnaryOp {
                op: UnaryOp::USub,
                operand,
                ..
            }) if let Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(i),
                ..
            }) = &**operand =>
            {
                literals.push(LitInt::from_ast(i).negate().to_explicit_type())
            }
            Expr::NumberLiteral(n) if let Number::Int(i) = &n.value => {
                literals.push(LitInt::from_ast(i).to_explicit_type())
            }
            Expr::StringLiteral(x) => match Lit::from_string_literal(x) {
                Some(lit) => literals.push(lit.to_explicit_type()),
                None => literals.push(self.heap.mk_literal_string(LitStyle::Explicit)),
            },
            Expr::BytesLiteral(x) => literals.push(Lit::from_bytes_literal(x).to_explicit_type()),
            Expr::BooleanLiteral(x) => {
                literals.push(Lit::from_boolean_literal(x).to_explicit_type())
            }
            Expr::NoneLiteral(_) => literals.push(self.heap.mk_none()),
            Expr::Name(_) => {
                fn is_valid_literal(x: &Type) -> bool {
                    match x {
                        Type::None | Type::Literal(_) | Type::Any(AnyStyle::Error) => true,
                        Type::Union(box Union { members: xs, .. }) => {
                            xs.iter().all(is_valid_literal)
                        }
                        _ => false,
                    }
                }
                let t = self.expr_untype(x, TypeFormContext::TypeArgument, errors);
                if is_valid_literal(&t) {
                    literals.push(t)
                } else {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                        format!("Invalid type inside literal, `{t}`"),
                    );
                    literals.push(self.heap.mk_any_error())
                }
            }
            Expr::Attribute(ExprAttribute {
                node_index: _,
                range,
                value,
                attr: member_name,
                ctx: _,
            }) if is_chained_attribute_access(value) => {
                let ty = self.expr_infer(value, errors);
                match ty {
                    Type::ClassDef(c)
                        if let Some(e) = self.get_enum_member(&c, &member_name.id) =>
                    {
                        literals.push(e.to_explicit_type())
                    }
                    ty @ Type::Any(AnyStyle::Error) => literals.push(ty),
                    _ => {
                        self.error(
                            errors,
                            *range,
                            ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                            format!(
                                "`{}.{}` is not a valid enum member",
                                value.display_with(self.module()),
                                member_name.id
                            ),
                        );
                        literals.push(self.heap.mk_any_error())
                    }
                }
            }
            Expr::Subscript(_) => {
                let ty = self.expr_infer(x, errors);
                self.map_over_union(&ty, |ty| match ty {
                    Type::Type(box lit @ Type::Literal(_)) => literals.push(lit.clone()),
                    ty @ Type::Any(AnyStyle::Error) => literals.push(ty.clone()),
                    _ => {
                        self.error(
                            errors,
                            x.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                            "Invalid literal expression".to_owned(),
                        );
                        literals.push(self.heap.mk_any_error())
                    }
                });
            }
            _ => {
                self.error(
                    errors,
                    x.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                    "Invalid literal expression".to_owned(),
                );
                literals.push(self.heap.mk_any_error())
            }
        }
    }

    pub fn apply_special_form(
        &self,
        special_form: SpecialForm,
        arguments: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let (arguments, parens) = match arguments {
            Expr::Tuple(x) => (x.elts.as_slice(), x.parenthesized),
            _ => (slice::from_ref(arguments), false),
        };

        match special_form {
            SpecialForm::Optional if arguments.len() == 1 => {
                self.heap
                    .mk_type_form(self.heap.mk_optional(self.expr_untype(
                        &arguments[0],
                        TypeFormContext::TypeArgument,
                        errors,
                    )))
            }
            SpecialForm::Optional => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "`Optional` requires exactly one argument but {} was found",
                    arguments.len()
                ),
            ),
            SpecialForm::Union => self.heap.mk_type_form(self.unions(
                arguments.map(|arg| self.expr_untype(arg, TypeFormContext::TypeArgument, errors)),
            )),
            SpecialForm::Tuple => match self.check_args_and_construct_tuple(arguments, errors) {
                Some((tuple, _)) => self.heap.mk_type_form(self.heap.mk_tuple(tuple)),
                None => self
                    .heap
                    .mk_type_form(self.heap.mk_unbounded_tuple(self.heap.mk_any_error())),
            },
            SpecialForm::Literal => {
                if parens {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                        "`Literal` arguments cannot be parenthesized".to_owned(),
                    );
                }
                let mut literals = Vec::new();
                arguments
                    .iter()
                    .for_each(|x| self.apply_literal(x, errors, &mut literals));
                self.heap.mk_type_form(self.unions(literals))
            }
            SpecialForm::Concatenate => {
                if arguments.len() < 2 {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::BadSpecialization),
                        format!(
                            "`Concatenate` must take at least two arguments, got {}",
                            arguments.len()
                        ),
                    )
                } else {
                    let args = arguments[0..arguments.len() - 1]
                        .iter()
                        .map(|x| {
                            (
                                self.expr_untype(x, TypeFormContext::TupleOrCallableParam, errors),
                                Required::Required,
                            )
                        })
                        .collect();
                    let pspec = self.expr_untype(
                        arguments.last().unwrap(),
                        TypeFormContext::TypeArgument,
                        errors,
                    );
                    if !pspec.is_kind_param_spec() {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadSpecialization),
                            format!(
                                "Expected a `ParamSpec` for the second argument of `Concatenate`, got {pspec}",
                            ),
                        );
                    }
                    self.heap
                        .mk_type_form(Type::Concatenate(args, Box::new(pspec)))
                }
            }
            SpecialForm::Callable if arguments.len() == 2 => {
                let ret = self.expr_untype(
                    &arguments[1],
                    TypeFormContext::TypeArgumentCallableReturn,
                    errors,
                );
                match &arguments[0] {
                    Expr::List(ExprList { elts, .. }) => {
                        match self.check_args_and_construct_tuple(elts, errors) {
                            Some((tuple, true)) => {
                                self.heap.mk_type_form(self.heap.mk_callable_from_vec(
                                    vec![Param::VarArg(
                                        None,
                                        self.heap.mk_unpack(self.heap.mk_tuple(tuple)),
                                    )],
                                    ret,
                                ))
                            }
                            Some((Tuple::Concrete(elts), false)) => {
                                self.heap.mk_type_form(self.heap.mk_callable_from_vec(
                                    elts.map(|t| {
                                        Param::PosOnly(None, t.clone(), Required::Required)
                                    }),
                                    ret,
                                ))
                            }
                            Some(_) => {
                                self.error(
                                    errors,
                                    range,
                                    ErrorInfo::Kind(ErrorKind::BadSpecialization),
                                    "Unrecognized callable type form".to_owned(),
                                );
                                self.heap.mk_type_form(
                                    self.heap.mk_callable_ellipsis(self.heap.mk_any_error()),
                                )
                            }
                            None => self.heap.mk_type_form(
                                self.heap.mk_callable_ellipsis(self.heap.mk_any_error()),
                            ),
                        }
                    }
                    Expr::EllipsisLiteral(_) => {
                        self.heap.mk_type_form(self.heap.mk_callable_ellipsis(ret))
                    }
                    name @ Expr::Name(_) => {
                        let ty = self.expr_untype(name, TypeFormContext::TypeArgument, errors);
                        if ty.is_kind_param_spec() {
                            self.heap
                                .mk_type_form(self.heap.mk_callable_param_spec(ty, ret))
                        } else {
                            self.error(errors, name.range(),ErrorInfo::Kind(ErrorKind::BadSpecialization), format!("Callable types can only have `ParamSpec` in this position, got `{}`", self.for_display(ty)));
                            self.heap.mk_type_form(
                                self.heap.mk_callable_ellipsis(self.heap.mk_any_error()),
                            )
                        }
                    }
                    x @ Expr::Subscript(_) => {
                        let ty = self.expr_untype(x, TypeFormContext::TypeArgument, errors);
                        match ty {
                            Type::Concatenate(args, pspec) => self
                                .heap
                                .mk_type_form(self.heap.mk_callable_concatenate(args, *pspec, ret)),
                            _ => {
                                self.error(errors, x.range(),ErrorInfo::Kind(ErrorKind::BadSpecialization), format!("Callable types can only have `Concatenate` in this position, got `{}`", self.for_display(ty)));
                                self.heap.mk_type_form(
                                    self.heap.mk_callable_ellipsis(self.heap.mk_any_error()),
                                )
                            }
                        }
                    }
                    x => {
                        self.error(
                            errors,
                            x.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                            "Invalid `Callable` type".to_owned(),
                        );
                        self.heap
                            .mk_type_form(self.heap.mk_callable_ellipsis(self.heap.mk_any_error()))
                    }
                }
            }
            SpecialForm::Callable => {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadSpecialization),
                    format!(
                        "`Callable` requires exactly two arguments but {} was found",
                        arguments.len()
                    ),
                );
                self.heap
                    .mk_type_form(self.heap.mk_callable_ellipsis(self.heap.mk_any_error()))
            }
            SpecialForm::TypeGuard if arguments.len() == 1 => {
                self.heap
                    .mk_type_form(self.heap.mk_type_guard(self.expr_untype(
                        &arguments[0],
                        TypeFormContext::TypeArgument,
                        errors,
                    )))
            }
            SpecialForm::TypeGuard => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "`TypeGuard` requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::TypeIs if arguments.len() == 1 => {
                self.heap
                    .mk_type_form(self.heap.mk_type_is(self.expr_untype(
                        &arguments[0],
                        TypeFormContext::TypeArgument,
                        errors,
                    )))
            }
            SpecialForm::TypeIs => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "`TypeIs` requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Unpack if arguments.len() == 1 => {
                self.heap.mk_type_form(self.heap.mk_unpack(self.expr_untype(
                    &arguments[0],
                    TypeFormContext::TypeArgument,
                    errors,
                )))
            }
            SpecialForm::Unpack => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "`Unpack` requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Type if arguments.len() == 1 => {
                self.heap
                    .mk_type_form(self.heap.mk_type_form(self.expr_untype(
                        &arguments[0],
                        TypeFormContext::TypeArgumentForType,
                        errors,
                    )))
            }
            SpecialForm::Type => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "`Type` requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Annotated if arguments.len() > 1 => self.heap.mk_type_form(
                self.expr_untype(&arguments[0], TypeFormContext::TypeArgument, errors),
            ),
            SpecialForm::Annotated => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                "`Annotated` needs at least one piece of metadata in addition to the type"
                    .to_owned(),
            ),
            // Keep this in sync with `SpecialForm::can_be_subscripted``
            SpecialForm::SelfType
            | SpecialForm::LiteralString
            | SpecialForm::Never
            | SpecialForm::NoReturn
            | SpecialForm::TypeAlias
            | SpecialForm::TypedDict => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                format!("`{special_form}` may not be subscripted"),
            ),
            SpecialForm::ClassVar
            | SpecialForm::Final
            | SpecialForm::Generic
            | SpecialForm::Protocol
            | SpecialForm::ReadOnly
            | SpecialForm::NotRequired
            | SpecialForm::Required => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                format!("`{special_form}` is not allowed in this context"),
            ),
        }
    }
}
