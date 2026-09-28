/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `Shaped[T, "<shape>"]`, the spelling of a shape annotation that other type
//! checkers can read.
//!
//! `shape_extensions.Shaped` is `typing.Annotated`, so other checkers see `T` and
//! ignore the string. Inside a `@shape_vars` scope the binder parses the string
//! and binds the parsed shape in its place. `T` then decides how the shape
//! applies:
//!
//! - A generic class receives the string's comma-separated arguments after its
//!   explicit ones, so `Shaped[ndarray, "[M, N]"]` is `ndarray[[M, N]]` and
//!   `Shaped[Encoder[Input], "A, B"]` is `Encoder[Input, A, B]`.
//! - A class with no type parameters, such as real torch's `Tensor`, ignores the
//!   shape. A library annotated for the Pyrefly array stubs then still checks
//!   against the real library.
//! - `int`, as in `Shaped[int, "N"]`, is replaced by the dimension `Int[N]`.
//! - A tuple, as in `ndarray[Shaped[tuple[int, int], "[M, N]"], dt]`, is replaced
//!   by the `IntTuple` it stands for, and must agree with it. Explicit `Any` is a
//!   tuple of unknown rank.
//!
//! A malformed `Shaped` is reported and then means its base alone, as it does to
//! other checkers. An appended argument is an ordinary type argument, checked as
//! one. Qualifiers such as `ClassVar` and `NotRequired` go outside `Shaped`, not
//! inside it. Only annotations, `assert_type`, and type parameter bounds read the
//! shape string: in a class base, a `cast`, or a type alias, `Shaped` is plain
//! `Annotated`.

use std::slice;

use pyrefly_python::ast::Ast;
use pyrefly_types::dimension::Int;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::shaped_array::IntTupleView;
use pyrefly_types::shaped_array::tuple_carrier_to_shape;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ExprTuple;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::types::AnyStyle;
use crate::types::types::Type;

/// The rank a carrier states to other checkers, if `base` is a carrier: a tuple
/// type, or an explicit `Any`, which states no rank.
fn carrier_int_tuple(base: &Type) -> Option<IntTuple> {
    match base {
        Type::Any(AnyStyle::Explicit) => Some(IntTuple::shapeless()),
        Type::Tuple(_) => tuple_carrier_to_shape(base),
        _ => None,
    }
}

/// Whether a carrier and a shape agree: every rank the shape allows is one the
/// carrier allows, and every literal dimension the carrier states is one the
/// shape states too.
fn carrier_agrees(carrier: &IntTuple, shape: &IntTuple) -> bool {
    let (prefix, suffix): (&[Int], &[Int]) = match shape.view() {
        IntTupleView::Concrete(dims) => return carrier_allows(carrier, dims),
        IntTupleView::Gradual => (&[], &[]),
        IntTupleView::Unpacked { prefix, suffix, .. } => (prefix, suffix),
    };
    match carrier.view() {
        IntTupleView::Gradual => true,
        IntTupleView::Concrete(_) => false,
        // Filling the shape's variadic middle with up to as many gradual dimensions
        // as the carrier states covers every way the dimensions the carrier states
        // can line up with the ones the shape states.
        IntTupleView::Unpacked {
            prefix: carrier_prefix,
            suffix: carrier_suffix,
            ..
        } => (0..=carrier_prefix.len() + carrier_suffix.len()).all(|middle| {
            carrier_allows(carrier, &[prefix, &vec![Int::Int; middle], suffix].concat())
        }),
    }
}

/// Whether a carrier allows a shape of known rank.
fn carrier_allows(carrier: &IntTuple, dims: &[Int]) -> bool {
    match carrier.view() {
        IntTupleView::Concrete(carrier) => {
            carrier.len() == dims.len() && literals_agree(carrier, dims)
        }
        IntTupleView::Unpacked { prefix, suffix, .. } => {
            dims.len() >= prefix.len() + suffix.len()
                && literals_agree(prefix, dims)
                && literals_agree(suffix, &dims[dims.len() - suffix.len()..])
        }
        IntTupleView::Gradual => true,
    }
}

/// Whether every literal the carrier states, aligned at the start with `dims`, is
/// the same literal there. A carrier literal promises other checkers that
/// dimension, so a shape that allows any other value there disagrees.
fn literals_agree(carrier: &[Int], dims: &[Int]) -> bool {
    carrier.iter().zip(dims).all(|pair| match pair {
        (Int::Literal(carrier), dim) => dim == &Int::Literal(*carrier),
        _ => true,
    })
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Parse `Shaped[T, "<shape>"]`, if the binder parsed its shape string.
    ///
    /// Returning `None` leaves the subscript to ordinary parsing, where `Shaped`
    /// is `Annotated`.
    pub(crate) fn parse_shaped_annotation(
        &self,
        slice: &Expr,
        range: TextRange,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !self
            .bindings()
            .shape_declarations
            .is_shaped_annotation(range)
        {
            return None;
        }
        let [base, shape] = Ast::unpack_slice(slice) else {
            unreachable!("the binder only records `Shaped` with two arguments")
        };
        Some(self.shaped_type(base, shape, range, type_form_context, errors))
    }

    fn shaped_type(
        &self,
        base_expr: &Expr,
        shape_expr: &Expr,
        range: TextRange,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        // A bare generic base reports the type arguments the shape supplies, so its
        // errors are kept only if the shape is not appended to it.
        let base_errors = self.error_collector();
        let base = self.expr_untype(base_expr, TypeFormContext::type_argument(), &base_errors);
        match &base {
            Type::ClassType(cls) if cls.is_builtin("int") => {
                errors.extend(base_errors);
                // `Shaped[int, "N"]` is `Int[N]`.
                return match self.parse_int_type(
                    slice::from_ref(shape_expr),
                    range,
                    type_form_context,
                    errors,
                ) {
                    Type::Any(AnyStyle::Error) => base,
                    int => self.untype(int, range, errors),
                };
            }
            Type::ClassType(cls) if !cls.tparams().is_empty() => {
                let (class_expr, explicit_args) = match base_expr {
                    Expr::Subscript(subscript) => {
                        (&*subscript.value, Ast::unpack_slice(&subscript.slice))
                    }
                    _ => (base_expr, &[][..]),
                };
                let shape_args = match shape_expr {
                    Expr::Tuple(tuple) => &tuple.elts[..],
                    _ => slice::from_ref(shape_expr),
                };
                let args = Expr::Tuple(ExprTuple {
                    node_index: AtomicNodeIndex::default(),
                    range,
                    elts: explicit_args.iter().chain(shape_args).cloned().collect(),
                    ctx: ExprContext::Load,
                    parenthesized: false,
                });
                let class = self.expr_infer(class_expr, errors);
                let specialized = self.subscript_infer_for_type(&class, &args, range, errors);
                return self.untype(specialized, range, errors);
            }
            // A class without type parameters ignores the shape, and an unresolved
            // base was already reported.
            Type::ClassType(_) | Type::Any(AnyStyle::Error | AnyStyle::Implicit) => {
                errors.extend(base_errors);
                return base;
            }
            _ => errors.extend(base_errors),
        }
        let Some(carrier) = carrier_int_tuple(&base) else {
            self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                format!(
                    "`Shaped` needs a class, an integer tuple, or `Any`, got `{}`",
                    self.for_display(base.clone())
                ),
            );
            return base;
        };

        let shape_context = TypeFormContext::TypeArgument(&type_form_context);
        let shape = match shape_expr {
            Expr::List(list) => {
                match self.parse_int_tuple_shape_args(&list.elts, shape_context, errors) {
                    Some(shape) => shape.to_shape_arg_type(),
                    None => return base,
                }
            }
            _ => {
                let shape = self.expr_untype(shape_expr, shape_context, errors);
                // An invalid shape was already reported.
                if matches!(shape, Type::Any(AnyStyle::Error)) {
                    return base;
                }
                if !self.is_int_tuple_dsl_argument(&shape) {
                    self.error(
                        errors,
                        shape_expr.range(),
                        ErrorKind::InvalidAnnotation,
                        format!(
                            "A `Shaped` shape must be a list of dimensions, a variadic \
                             shape, or a shape function call, got `{}`",
                            self.for_display(shape)
                        ),
                    );
                    return base;
                }
                shape
            }
        };
        let message = if carrier_agrees(
            &carrier,
            // A shape that is not an `IntTuple`, such as a shape function call, has
            // no known rank.
            &self
                .shape_arg_to_int_tuple(&shape)
                .unwrap_or_else(IntTuple::shapeless),
        ) {
            return shape;
        } else {
            format!(
                "`Shaped` tuple `{}` does not agree with its shape `{}`",
                self.for_display(base.clone()),
                self.for_display(shape)
            )
        };
        // A malformed annotation means what other checkers read: the base alone.
        self.error(errors, range, ErrorKind::InvalidAnnotation, message);
        base
    }
}
