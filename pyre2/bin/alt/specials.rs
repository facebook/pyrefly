/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use super::answers::AnswersSolver;
use crate::types::callable::Arg;
use crate::types::callable::Required;
use crate::types::literal::Lit;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::types::Type;
use crate::util::prelude::SliceExt;

impl<'a> AnswersSolver<'a> {
    pub fn apply_special_form(
        &self,
        special_form: SpecialForm,
        arguments: &[Expr],
        range: TextRange,
    ) -> Type {
        match special_form {
            SpecialForm::Optional if arguments.len() == 1 => {
                Type::Type(Box::new(Type::Union(vec![
                    self.expr_untype(&arguments[0]),
                    Type::None,
                ])))
            }
            SpecialForm::Optional => self.error(
                range,
                format!(
                    "Optional requires exactly one argument but {} was found",
                    arguments.len()
                ),
            ),
            SpecialForm::Union => Type::Type(Box::new(Type::Union(
                arguments.map(|arg| self.expr_untype(arg)),
            ))),
            SpecialForm::Tuple => {
                let types: Vec<Type> = arguments.map(|x| self.expr_untype(x));
                match types.as_slice() {
                    [Type::Ellipsis, Type::Ellipsis] => self.error(
                        arguments[0].range(),
                        "Invalid position for `...`".to_string(),
                    ),
                    [t, Type::Ellipsis] => {
                        Type::Type(Box::new(Type::Tuple(Tuple::unbounded(t.clone()))))
                    }
                    _ => {
                        for (index, value) in arguments.iter().enumerate() {
                            if matches!(types[index], Type::Ellipsis) {
                                return self.error(
                                    value.range(),
                                    "Invalid position for `...`".to_string(),
                                );
                            }
                        }
                        Type::Type(Box::new(Type::Tuple(Tuple::concrete(types))))
                    }
                }
            }
            SpecialForm::Literal => {
                let mut literals = Vec::new();
                for x in arguments.iter() {
                    let lit = Lit::from_expr(
                        x,
                        self.module_info,
                        &|cls| self.get_mro_for_class(cls),
                        &|identifier| self.get_class_for_identifier(&identifier),
                        self.errors,
                    );
                    literals.push(Type::Literal(lit));
                }
                Type::Type(Box::new(self.unions(&literals)))
            }
            SpecialForm::Callable if arguments.len() == 2 => {
                let ret = self.expr_untype(&arguments[1]);
                match &arguments[0] {
                    Expr::List(args) => Type::Type(Box::new(Type::callable(
                        args.elts
                            .map(|x| Arg::PosOnly(self.expr_untype(x), Required::Required)),
                        ret,
                    ))),
                    Expr::EllipsisLiteral(_) => Type::Type(Box::new(Type::callable_ellipsis(ret))),
                    name @ Expr::Name(_) => {
                        let ty = self.expr_untype(name);
                        Type::Type(Box::new(Type::callable_param_spec(ty, ret)))
                    }
                    x => self.error_todo("expr_infer, Callable type", x),
                }
            }
            SpecialForm::Callable => self.error(
                range,
                format!(
                    "Callable requires exactly two arguments but {} was found",
                    arguments.len()
                ),
            ),
            SpecialForm::TypeGuard if arguments.len() == 1 => Type::Type(Box::new(
                Type::TypeGuard(Box::new(self.expr_untype(&arguments[0]))),
            )),
            SpecialForm::TypeGuard => self.error(
                range,
                format!(
                    "TypeGuard requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::TypeIs if arguments.len() == 1 => Type::Type(Box::new(Type::TypeIs(
                Box::new(self.expr_untype(&arguments[0])),
            ))),
            SpecialForm::TypeIs => self.error(
                range,
                format!(
                    "TypeIs requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Unpack if arguments.len() == 1 => Type::Type(Box::new(Type::Unpack(
                Box::new(self.expr_untype(&arguments[0])),
            ))),
            SpecialForm::Unpack => self.error(
                range,
                format!(
                    "Unpack requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Type if arguments.len() == 1 => Type::Type(Box::new(Type::Type(
                Box::new(self.expr_untype(&arguments[0])),
            ))),
            SpecialForm::Type => self.error(
                range,
                format!(
                    "Type requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Annotated if arguments.len() > 1 => self.expr(&arguments[0], None),
            _ => self.error_todo(
                &format!(
                    "Answers::apply_special_form cannot handle `{special_form}[{}]`",
                    arguments
                        .map(|x| self.module_info.display(x).to_string())
                        .join(", ")
                ),
                range,
            ),
        }
    }
}
