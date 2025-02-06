/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::binding::binding::Key;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::literal::Lit;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::types::Type;
use crate::util::prelude::SliceExt;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn apply_special_form(
        &self,
        special_form: SpecialForm,
        arguments: &[Expr],
        range: TextRange,
    ) -> Type {
        match special_form {
            SpecialForm::Optional if arguments.len() == 1 => Type::type_form(Type::Union(vec![
                self.expr_untype(&arguments[0]),
                Type::None,
            ])),
            SpecialForm::Optional => self.error(
                range,
                format!(
                    "Optional requires exactly one argument but {} was found",
                    arguments.len()
                ),
            ),
            SpecialForm::Union => {
                Type::type_form(Type::Union(arguments.map(|arg| self.expr_untype(arg))))
            }
            SpecialForm::Tuple => {
                let types: Vec<Type> = arguments.map(|x| self.expr_untype(x));
                match types.as_slice() {
                    [Type::Ellipsis, Type::Ellipsis] => self.error(
                        arguments[0].range(),
                        "Invalid position for `...`".to_owned(),
                    ),
                    [t, Type::Ellipsis] => {
                        Type::type_form(Type::Tuple(Tuple::unbounded(t.clone())))
                    }
                    _ => {
                        for (index, value) in arguments.iter().enumerate() {
                            if matches!(types[index], Type::Ellipsis) {
                                return self
                                    .error(value.range(), "Invalid position for `...`".to_owned());
                            }
                        }
                        Type::type_form(Type::Tuple(Tuple::concrete(types)))
                    }
                }
            }
            SpecialForm::Literal => {
                let mut literals = Vec::new();
                for x in arguments.iter() {
                    let lit = Lit::from_expr(
                        x,
                        self.module_info(),
                        &|enum_name, member_name| {
                            let key = self
                                .bindings()
                                .key_to_idx(&Key::Usage(ShortIdentifier::new(&enum_name)));
                            let cls = self.get_idx_class_def(key)?;
                            self.get_enum_member(&cls, member_name)
                        },
                        self.errors(),
                    );
                    literals.push(lit);
                }
                Type::type_form(self.unions(literals))
            }
            SpecialForm::Concatenate => {
                if arguments.len() < 2 {
                    self.error(
                        range,
                        format!(
                            "`Concatenate` must take at least two arguments, got {}",
                            arguments.len()
                        ),
                    )
                } else {
                    let args = arguments[0..arguments.len() - 1]
                        .iter()
                        .map(|x| self.expr_untype(x))
                        .collect();
                    let pspec = self.expr_untype(arguments.last().unwrap());
                    Type::type_form(Type::Concatenate(args, Box::new(pspec)))
                }
            }
            SpecialForm::Callable if arguments.len() == 2 => {
                let ret = self.expr_untype(&arguments[1]);
                match &arguments[0] {
                    Expr::List(params) => Type::type_form(Type::callable(
                        params
                            .elts
                            .map(|x| Param::PosOnly(self.expr_untype(x), Required::Required)),
                        ret,
                    )),
                    Expr::EllipsisLiteral(_) => Type::type_form(Type::callable_ellipsis(ret)),
                    name @ Expr::Name(_) => {
                        let ty = self.expr_untype(name);
                        Type::type_form(Type::callable_param_spec(ty, ret))
                    }
                    x @ Expr::Subscript(_) => {
                        let ty = self.expr_untype(x);
                        match ty {
                            Type::Concatenate(args, pspec) => {
                                Type::type_form(Type::callable_concatenate(args , *pspec, ret))
                            }
                            _ => self.error(x.range(), format!("Callable types can only have `Concatenate` in this position, got `{}`", ty.deterministic_printing())),
                        }
                    }
                    x => self.todo("expr_infer, Callable type", x),
                }
            }
            SpecialForm::Callable => self.error(
                range,
                format!(
                    "Callable requires exactly two arguments but {} was found",
                    arguments.len()
                ),
            ),
            SpecialForm::TypeGuard if arguments.len() == 1 => {
                Type::type_form(Type::TypeGuard(Box::new(self.expr_untype(&arguments[0]))))
            }
            SpecialForm::TypeGuard => self.error(
                range,
                format!(
                    "TypeGuard requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::TypeIs if arguments.len() == 1 => {
                Type::type_form(Type::TypeIs(Box::new(self.expr_untype(&arguments[0]))))
            }
            SpecialForm::TypeIs => self.error(
                range,
                format!(
                    "TypeIs requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Unpack if arguments.len() == 1 => {
                Type::type_form(Type::Unpack(Box::new(self.expr_untype(&arguments[0]))))
            }
            SpecialForm::Unpack => self.error(
                range,
                format!(
                    "Unpack requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Type if arguments.len() == 1 => {
                Type::type_form(Type::type_form(self.expr_untype(&arguments[0])))
            }
            SpecialForm::Type => self.error(
                range,
                format!(
                    "Type requires exactly one argument but got {}",
                    arguments.len()
                ),
            ),
            SpecialForm::Annotated if arguments.len() > 1 => self.expr(&arguments[0], None),
            _ => self.todo(
                &format!(
                    "Answers::apply_special_form cannot handle `{special_form}[{}]`",
                    arguments
                        .map(|x| self.module_info().display(x).to_string())
                        .join(", ")
                ),
                range,
            ),
        }
    }
}
