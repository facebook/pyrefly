/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::error::collector::ErrorCollector;
use crate::error::kind::ErrorKind;
use crate::types::special_form::SpecialForm;
use crate::types::types::Type;

/// Private helper type used to share part of the logic needed for the
/// binding-level work of finding legacy type parameters versus the type-level
/// work of computing inheritance information and the MRO.
#[derive(Debug, Clone)]
pub enum BaseClass {
    TypedDict,
    Generic(Vec<Type>),
    Protocol(Vec<Type>),
    Expr(Expr),
    NamedTuple(TextRange),
}

impl BaseClass {
    pub fn can_apply(&self) -> bool {
        matches!(self, BaseClass::Generic(_) | BaseClass::Protocol(_))
    }

    pub fn apply(&mut self, args: Vec<Type>) {
        match self {
            BaseClass::Generic(xs) | BaseClass::Protocol(xs) => {
                xs.extend(args);
            }
            _ => panic!("cannot apply base class"),
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// This helper deals with special cases where we want to intercept an `Expr`
    /// manually and create a special variant of `BaseClass` instead of calling
    /// `expr_untype` and creating a `BaseClass::Type`.
    ///
    /// TODO(stroxler): See if there's a way to express this more clearly in the types.
    fn special_base_class(&self, base_expr: &Expr, errors: &ErrorCollector) -> Option<BaseClass> {
        let name = match base_expr {
            Expr::Name(x) => &x.id,
            Expr::Attribute(x) => &x.attr.id,
            _ => return None,
        };
        if !["Protocol", "Generic", "TypedDict", "NamedTuple"].contains(&name.as_str()) {
            // Calling expr_infer when figuring out the base class leads to cycles, so we really want to try
            // and avoid doing it unless there is a high likelihood of a special form.
            // Downside is that you can't alias `Generic` etc, but I'm not sure you should want to.
            return None;
        }

        match self.expr_infer(base_expr, errors) {
            Type::Type(box Type::SpecialForm(special)) => match special {
                SpecialForm::Protocol => Some(BaseClass::Protocol(Vec::new())),
                SpecialForm::Generic => Some(BaseClass::Generic(Vec::new())),
                SpecialForm::TypedDict => Some(BaseClass::TypedDict),
                _ => None,
            },
            Type::ClassDef(cls) if cls.has_qname("typing", "NamedTuple") => {
                Some(BaseClass::NamedTuple(base_expr.range()))
            }
            _ => None,
        }
    }

    pub fn base_class_of(&self, base_expr: &Expr, errors: &ErrorCollector) -> BaseClass {
        if let Some(special_base_class) = self.special_base_class(base_expr, errors) {
            // This branch handles cases like `Protocol`
            special_base_class
        } else if let Expr::Subscript(subscript) = base_expr
            && let Some(mut special_base_class) = self.special_base_class(&subscript.value, errors)
            && special_base_class.can_apply()
        {
            // This branch handles `Generic[...]` and `Protocol[...]`
            let mut type_var_tuple_count = 0;
            let args = Ast::unpack_slice(&subscript.slice).map(|x| {
                let ty = self.expr_untype(x, TypeFormContext::GenericBase, errors);
                if let Type::Unpack(unpacked) = &ty
                    && unpacked.is_kind_type_var_tuple()
                {
                    if type_var_tuple_count == 1 {
                        self.error(
                            errors,
                            x.range(),
                            ErrorKind::InvalidInheritance,
                            None,
                            "There cannot be more than one TypeVarTuple type parameter".to_owned(),
                        );
                    }
                    type_var_tuple_count += 1;
                }
                ty
            });
            special_base_class.apply(args);
            special_base_class
        } else {
            // This branch handles all other base classes.
            BaseClass::Expr(base_expr.clone())
        }
    }
}
