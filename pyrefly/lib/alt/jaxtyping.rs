/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Jaxtyping annotation support.
//!
//! This module handles parsing and processing of jaxtyping-style tensor
//! annotations like `Float[Tensor, "batch channels"]`. Static jaxtyping stubs
//! expose dtype wrappers (Float, Int, Shaped, etc.) as `Annotated` aliases.
//! Pyrefly uses those wrappers only as markers for jaxtyping shape syntax; it
//! does not model dtype refinements. The parsed shape is applied to the unique
//! gradual `IntTuple`-bounded type argument of an ordinary generic class.
//! Ordinary classes with defaults
//! that depend on that shape parameter are not expanded yet, because replacing
//! the shape without re-instantiating the dependent defaults would leave stale
//! type arguments. Generic type aliases are also left to normal `Annotated`
//! handling because alias parameters do not retain a stable one-to-one identity
//! with the expanded class parameters.
//!
//! ## Shape string syntax
//!
//! The shape string is whitespace-separated and supports:
//! - Named dims (`"batch"`) → quantified `IntVar`s
//! - Integer literals (`"3"`) → `Type::Int(Int::Literal(3))`
//! - Anonymous dim (`"_"`) → `Type::Any(AnyStyle::Implicit)`
//! - Variadic (`"*batch"`) → `TypeVar`s bounded by `IntTuple`
//! - Ellipsis (`"..."`) → anonymous variadic (any number of any-sized dims)
//! - Broadcast (`"#batch"`) → treated as `"batch"` (conservative, safe)
//! - Combined (`"*#batch"`) → variadic `IntTuple`, broadcast prefix stripped
//! - Arithmetic (`"dim+1"`, `"n-1"`) → `Type::Int(Int::Add/Sub(...))`
//! - Parenthesized (`"(1+T)"`) → parens stripped, parsed as arithmetic
//! - Scalar (`""`) → rank-0 tensor
//!
//! ## Implicit TypeVars
//!
//! Unlike native tensor syntax where TypeVars are explicitly declared in the
//! function signature (`def f[N, M](...)`), jaxtyping dimensions are implicitly
//! created from the shape string. This module collects these implicit TypeVars
//! and adds them to the function's `Forall` wrapper so they participate in
//! type inference.
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::keywords::is_valid_identifier;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::class::ClassType;
use pyrefly_types::dimension::Int;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::binding::binding::Binding;
use crate::binding::binding::ImportBinding;
use crate::binding::binding::Key;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::types::AnyStyle;
use crate::types::types::Forallable;
use crate::types::types::Type;

const JAXTYPING_WRAPPERS: &[&str] = &[
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "BFloat16",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Integer",
    "Key",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Bool",
    "Num",
    "Real",
    "Shaped",
    "Complex",
    "Complex64",
    "Complex128",
    "Inexact",
];

struct JaxtypingTarget {
    base_class: ClassType,
    shape_arg_index: usize,
}

/// One operand of a jaxtyping dimension expression.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ShapeAtom {
    Literal(i64),
    Named(Name),
}

/// One dimension of a jaxtyping shape string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ShapeDim {
    /// `3`, `-1`
    Literal(i64),
    /// `_`, matching any one dimension without naming it.
    Anonymous,
    /// `batch`. A `#` broadcast prefix is stripped and does not survive parsing,
    /// so `#batch` and `batch` are the same dimension.
    Named(Name),
    /// `dim+1`, and `(1+T)` once the parentheses are stripped.
    Add(ShapeAtom, ShapeAtom),
    /// `n-1`
    Sub(ShapeAtom, ShapeAtom),
}

/// The variadic segment of a shape: `*name`, or `...` which names nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShapeVariadic(Option<Name>);

/// A shape string split around its variadic, if it has one.
///
/// A shape has at most one variadic because its desugared form,
/// `IntTuple::unpacked`, has exactly one unpacked middle: there is no way to
/// divide a concrete run of dimensions between two variadic segments. Keeping
/// the variadic in its own field rather than inline makes that unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedShape {
    prefix: Vec<ShapeDim>,
    variadic: Option<ShapeVariadic>,
    suffix: Vec<ShapeDim>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ShapeStringError {
    MultipleVariadics,
    UnsupportedExpression(String),
}

/// Parse one operand of a dimension expression.
fn parse_shape_atom(token: &str) -> Option<ShapeAtom> {
    token
        .parse::<i64>()
        .map(ShapeAtom::Literal)
        .ok()
        .or_else(|| is_valid_identifier(token).then(|| ShapeAtom::Named(Name::new(token))))
}

/// Split a dimension token on its last `+` or `-`.
///
/// The operator is sought from the right and never at position 0, so a negative
/// literal like `-3` stays a literal rather than becoming a subtraction.
fn parse_shape_arithmetic(token: &str) -> Result<Option<ShapeDim>, ()> {
    let Some((position, operator)) = token
        .char_indices()
        .rev()
        .find(|&(index, c)| index > 0 && (c == '+' || c == '-'))
    else {
        return Ok(None);
    };
    let left = &token[..position];
    let right = &token[position + 1..];
    let (left, right) = (
        parse_shape_atom(left).ok_or(())?,
        parse_shape_atom(right).ok_or(())?,
    );
    Ok(Some(match operator {
        '+' => ShapeDim::Add(left, right),
        '-' => ShapeDim::Sub(left, right),
        _ => unreachable!("only '+' and '-' are matched above"),
    }))
}

/// Parse a jaxtyping shape string into its dimensions.
///
/// This is the whole of the shape-string grammar, kept free of solver state so
/// it can be tested directly. Resolving the names it produces against the
/// dimensions a function declares is a separate step.
fn parse_shape_string(shape: &str) -> Result<ParsedShape, ShapeStringError> {
    let mut prefix = Vec::new();
    let mut variadic = None;
    let mut suffix = Vec::new();
    for token in shape.split_whitespace() {
        // Parentheses only prevent Python from evaluating the expression.
        let expression = token
            .strip_prefix('(')
            .and_then(|inner| inner.strip_suffix(')'))
            .unwrap_or(token);
        if expression == "..." || expression.starts_with('*') {
            if variadic.is_some() {
                return Err(ShapeStringError::MultipleVariadics);
            }
            let name = expression.strip_prefix('*').and_then(|name| {
                let name = name.strip_prefix('#').unwrap_or(name);
                is_valid_identifier(name).then(|| Name::new(name))
            });
            if expression != "..." && name.is_none() {
                return Err(ShapeStringError::UnsupportedExpression(token.to_owned()));
            }
            variadic = Some(ShapeVariadic(name));
            continue;
        }

        // Broadcast is accepted and ignored: a broadcastable dimension has no
        // representation of its own in the desugared shape, so `#batch` constrains
        // no differently from `batch`.
        let expression = expression.strip_prefix('#').unwrap_or(expression);
        let dim = if expression == "_" {
            ShapeDim::Anonymous
        } else if let Ok(literal) = expression.parse::<i64>() {
            ShapeDim::Literal(literal)
        } else {
            let arithmetic = parse_shape_arithmetic(expression)
                .map_err(|()| ShapeStringError::UnsupportedExpression(token.to_owned()))?;
            match arithmetic {
                Some(arithmetic) => arithmetic,
                None if is_valid_identifier(expression) => ShapeDim::Named(Name::new(expression)),
                None => return Err(ShapeStringError::UnsupportedExpression(token.to_owned())),
            }
        };
        if variadic.is_some() {
            suffix.push(dim);
        } else {
            prefix.push(dim);
        }
    }
    Ok(ParsedShape {
        prefix,
        variadic,
        suffix,
    })
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Check if an expression resolves to one of jaxtyping's public dtype wrappers.
    pub fn is_jaxtyping_wrapper_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Name(name) => self.name_is_jaxtyping_wrapper(name),
            Expr::Attribute(attr) => {
                if !JAXTYPING_WRAPPERS
                    .iter()
                    .any(|wrapper| attr.attr.id.as_str() == *wrapper)
                {
                    return false;
                }
                self.is_jaxtyping_module_expr(&attr.value)
            }
            _ => false,
        }
    }

    fn binding_origin(&self, mut idx: Idx<Key>) -> &Binding {
        for _ in 0..16 {
            match self.bindings().get(idx) {
                Binding::Forward(inner)
                | Binding::PromoteForward(inner)
                | Binding::ForwardToFirstUse(inner) => idx = *inner,
                Binding::PossibleLegacyTParam(legacy_tparam, ..) => {
                    idx = self.bindings().get(*legacy_tparam).idx()
                }
                binding => return binding,
            }
        }
        unreachable!("exceeded binding depth limit while resolving jaxtyping origin")
    }

    fn binding_origin_for_name(&self, name: &ruff_python_ast::ExprName) -> Option<&Binding> {
        let key = Key::BoundName(ShortIdentifier::expr_name(name));
        let idx = self.bindings().key_to_idx_hashed_opt(Hashed::new(&key))?;
        Some(self.binding_origin(idx))
    }

    fn import_is_jaxtyping_wrapper(import: &ImportBinding) -> bool {
        import.module.as_str() == "jaxtyping"
            && JAXTYPING_WRAPPERS
                .iter()
                .any(|wrapper| import.name.as_str() == *wrapper)
    }

    fn name_is_jaxtyping_wrapper(&self, name: &ruff_python_ast::ExprName) -> bool {
        self.binding_origin_for_name(name)
            .is_some_and(|binding| match binding {
                Binding::Import(import) => Self::import_is_jaxtyping_wrapper(import),
                _ => false,
            })
    }

    fn is_jaxtyping_module_expr(&self, expr: &Expr) -> bool {
        if let Expr::Name(name) = expr
            && self
                .binding_origin_for_name(name)
                .is_some_and(|binding| match binding {
                    Binding::Module(module) => module.0.as_str() == "jaxtyping",
                    Binding::Import(import) => {
                        import.module.as_str() == "jaxtyping" && import.name.as_str() == "jaxtyping"
                    }
                    _ => false,
                })
        {
            return true;
        }
        let silent_errors = self.error_swallower();
        matches!(
            self.expr_infer(expr, &silent_errors),
            Type::Module(module)
                if module.parts().len() == 1
                    && module.parts()[0].as_str() == "jaxtyping"
        )
    }

    /// Parse an origin-aware jaxtyping type form such as `Float[Tensor, "batch"]`.
    ///
    /// This hook is intentionally for annotation/type-form parsing only. In value
    /// expressions, jaxtyping aliases should keep their ordinary `Annotated[...]`
    /// runtime behavior. Returning `Some` is the commit point: normal
    /// `Annotated` parsing will not run, and this hook may emit diagnostics for
    /// malformed jaxtyping shape syntax.
    pub fn parse_jaxtyping_type_form(
        &self,
        value: &Expr,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let xs = Ast::unpack_slice(slice);
        if xs.is_empty()
            || !self.solver().config.tensor_shapes
            || !self.solver().config.jaxtyping
            || !self.is_jaxtyping_wrapper_expr(value)
        {
            return None;
        }
        let base_head = match &xs[0] {
            Expr::Subscript(subscript) => subscript.value.as_ref(),
            base => base,
        };
        let (base_is_type_alias, bare_class) = match self
            .expr_infer(base_head, &self.error_swallower())
        {
            Type::TypeAlias(_) => (true, None),
            Type::Forall(forall) if matches!(forall.body, Forallable::TypeAlias(_)) => (true, None),
            Type::ClassDef(cls) if !matches!(xs[0], Expr::Subscript(_)) => (false, Some(cls)),
            _ => (false, None),
        };
        let base_errors = self.error_collector();
        let target = self.jaxtyping_target(&xs[0], base_is_type_alias, &base_errors)?;
        let shape_parameter = target
            .base_class
            .tparams()
            .iter()
            .nth(target.shape_arg_index)
            .cloned();
        let base_range = xs[0].range();
        if let (Some(shape_parameter), Some(bare_class)) = (shape_parameter, bare_class) {
            errors.extend_filtered(base_errors, |error| {
                error.error_kind() != ErrorKind::ImplicitAnyTypeArgument
                    || error.range() != base_range
            });
            self.promote_ignoring_implicit_any_for(
                &bare_class,
                base_range,
                errors,
                &shape_parameter,
            );
        } else {
            errors.extend(base_errors);
        }
        Some(self.parse_jaxtyping_annotation(xs, target, range, errors))
    }

    fn jaxtyping_target(
        &self,
        base_expr: &Expr,
        base_is_type_alias: bool,
        errors: &ErrorCollector,
    ) -> Option<JaxtypingTarget> {
        match self.expr_untype(base_expr, TypeFormContext::type_argument(), errors) {
            Type::ClassType(base_class) if !base_is_type_alias => {
                self.jaxtyping_generic_target(base_class)
            }
            _ => None,
        }
    }

    fn jaxtyping_generic_target(&self, base_class: ClassType) -> Option<JaxtypingTarget> {
        let shape_arg_index = {
            let mut candidates =
                base_class
                    .targs()
                    .iter_paired()
                    .enumerate()
                    .filter(|(_, (param, arg))| {
                        param.kind() == QuantifiedKind::TypeVar
                            && matches!(param.restriction(), Restriction::Bound(Type::IntTuple(_)))
                            && (matches!(arg, Type::Any(AnyStyle::Explicit | AnyStyle::Implicit))
                                || IntTuple::from_shape_arg_or_tuple_carrier(arg)
                                    .is_some_and(|shape| shape.is_shapeless()))
                    });
            let (shape_arg_index, (shape_param, _)) = candidates.next()?;
            if candidates.next().is_some() {
                return None;
            }
            let has_dependent_default = base_class
                .tparams()
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != shape_arg_index)
                .filter_map(|(_, param)| param.default())
                .any(|default| {
                    let mut depends_on_shape = false;
                    default.for_each_quantified(&mut |q| depends_on_shape |= q == shape_param);
                    depends_on_shape
                });
            if has_dependent_default {
                return None;
            }
            shape_arg_index
        };
        Some(JaxtypingTarget {
            base_class,
            shape_arg_index,
        })
    }

    fn jaxtyping_target_type(&self, mut target: JaxtypingTarget, shape: IntTuple) -> Type {
        target.base_class.targs_mut().as_mut()[target.shape_arg_index] = shape.to_shape_arg_type();
        target.base_class.to_type()
    }

    /// Parse a jaxtyping annotation like `Float[Tensor, "batch channels"]`.
    fn parse_jaxtyping_annotation(
        &self,
        xs: &[Expr],
        target: JaxtypingTarget,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if xs.len() != 2 {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                format!(
                    "jaxtyping annotations require exactly 2 arguments \
                     (array type and shape string), got {}",
                    xs.len()
                ),
            );
        }

        // Extract shape string from xs[1]
        let shape_str = match &xs[1] {
            Expr::StringLiteral(ExprStringLiteral { value, .. }) => value.to_str(),
            _ => {
                return self.error(
                    errors,
                    xs[1].range(),
                    ErrorKind::InvalidAnnotation,
                    "Second argument to jaxtyping annotation must be a string literal".to_owned(),
                );
            }
        };

        let parsed = match parse_shape_string(shape_str) {
            Ok(parsed) => parsed,
            Err(ShapeStringError::MultipleVariadics) => {
                return self.error(
                    errors,
                    xs[1].range(),
                    ErrorKind::InvalidAnnotation,
                    "Tensor shape can have at most one variadic dimension".to_owned(),
                );
            }
            Err(ShapeStringError::UnsupportedExpression(token)) => {
                return self.error(
                    errors,
                    xs[1].range(),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Unsupported dimension expression `{token}`; use a name, `name+1`, or `name-1`"
                    ),
                );
            }
        };

        let prefix = self.jaxtyping_dim_types(&parsed.prefix);
        let Some(variadic) = parsed.variadic else {
            // An empty shape string leaves no dimensions, giving a rank-0 array.
            return self.jaxtyping_target_type(target, IntTuple::from_types(prefix));
        };
        let suffix = self.jaxtyping_dim_types(&parsed.suffix);
        let middle = match &variadic.0 {
            // `...` matches any number of dimensions of any size.
            None => IntTuple::shapeless().to_shape_arg_type(),
            Some(name) => {
                let quantified = self
                    .get_or_create_jaxtyping_variadic_shape(name.clone(), QuantifiedKind::TypeVar);
                Type::Quantified(Box::new(quantified))
            }
        };
        self.jaxtyping_target_type(
            target,
            IntTuple::unpacked_from_types(prefix, middle, suffix),
        )
    }

    fn jaxtyping_dim_types(&self, dims: &[ShapeDim]) -> Vec<Type> {
        dims.iter()
            .map(|dim| match dim {
                ShapeDim::Anonymous => Type::any_implicit(),
                ShapeDim::Literal(value) => self.heap.mk_int(Int::literal(*value)),
                ShapeDim::Named(name) => self.jaxtyping_named_dim(name),
                ShapeDim::Add(left, right) => self.heap.mk_int(Int::add(
                    self.jaxtyping_atom_type(left),
                    self.jaxtyping_atom_type(right),
                )),
                ShapeDim::Sub(left, right) => self.heap.mk_int(Int::sub(
                    self.jaxtyping_atom_type(left),
                    self.jaxtyping_atom_type(right),
                )),
            })
            .collect()
    }

    fn jaxtyping_atom_type(&self, atom: &ShapeAtom) -> Type {
        match atom {
            ShapeAtom::Literal(value) => self.heap.mk_int(Int::literal(*value)),
            ShapeAtom::Named(name) => self.jaxtyping_named_dim(name),
        }
    }

    fn jaxtyping_named_dim(&self, name: &Name) -> Type {
        let quantified =
            self.get_or_create_jaxtyping_dimension(name.clone(), QuantifiedKind::IntVar);
        Type::Quantified(Box::new(quantified))
    }

    /// Collect implicit jaxtyping TypeVars from a callable's signature and
    /// extend `tparams` with them.
    ///
    /// Returns the (potentially extended) `TParams` to use for the function's
    /// `Forall` wrapper.
    pub fn collect_jaxtyping_tparams(
        &self,
        callable: &impl Visit<Type>,
        tparams: &Arc<TParams>,
    ) -> Arc<TParams> {
        if !self.solver().config.tensor_shapes || !self.solver().config.jaxtyping {
            return tparams.dupe();
        }

        let mut jaxtyping_extras = Vec::new();
        let mut collect_implicit = |q: &Quantified| {
            if self.is_jaxtyping_quantified(q)
                && !tparams.iter().any(|existing| existing == q)
                && !jaxtyping_extras.contains(q)
            {
                jaxtyping_extras.push(q.clone());
            }
        };
        // Visit all types in the callable (params + return) to find jaxtyping
        // Quantified types.
        callable.visit(&mut |ty: &Type| {
            // `Visit<Type>` exposes the callable's top-level types, while dimensions
            // inside `IntTuple` are stored as `Int::Symbolic(Type)`. Use the semantic
            // type-variable traversal so wrappers such as unions, tuples, and callables
            // cannot hide an implicit jaxtyping variable. Quantifieds owned by a nested
            // `Forall` belong to that generic value and must not be hoisted into this callable.
            ty.for_each_free_quantified(&mut collect_implicit);
        });
        if jaxtyping_extras.is_empty() {
            tparams.dupe()
        } else {
            let mut params: Vec<_> = tparams.as_vec().to_vec();
            params.extend(jaxtyping_extras);
            Arc::new(TParams::new(params))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: &str) -> ShapeDim {
        ShapeDim::Named(Name::new(name))
    }

    fn atom(name: &str) -> ShapeAtom {
        ShapeAtom::Named(Name::new(name))
    }

    fn parse(shape: &str) -> ParsedShape {
        parse_shape_string(shape).expect("shape string should parse")
    }

    /// The dimensions of a shape that has no variadic.
    fn flat(shape: &str) -> Vec<ShapeDim> {
        let parsed = parse(shape);
        assert_eq!(parsed.variadic, None, "expected no variadic in {shape:?}");
        assert!(parsed.suffix.is_empty());
        parsed.prefix
    }

    #[test]
    fn empty_shape_is_rank_zero() {
        assert_eq!(flat(""), Vec::new());
        assert_eq!(flat("   "), Vec::new());
    }

    #[test]
    fn dimensions_split_on_whitespace() {
        assert_eq!(
            flat("batch channels"),
            vec![named("batch"), named("channels")]
        );
        // Surrounding whitespace is not significant.
        assert_eq!(
            flat("  batch\tchannels\n"),
            vec![named("batch"), named("channels")]
        );
    }

    #[test]
    fn literals_and_anonymous_dimensions() {
        assert_eq!(
            flat("3 _ -1"),
            vec![
                ShapeDim::Literal(3),
                ShapeDim::Anonymous,
                ShapeDim::Literal(-1)
            ]
        );
    }

    #[test]
    fn broadcast_prefix_is_stripped() {
        // A broadcastable dimension has no representation of its own in the
        // desugared shape, so `#batch` parses identically to `batch`.
        assert_eq!(flat("#batch"), vec![named("batch")]);
        assert_eq!(flat("(#batch)"), vec![named("batch")]);
        assert_eq!(
            parse("*#batch").variadic,
            Some(ShapeVariadic(Some(Name::new("batch"))))
        );
        assert_eq!(
            parse("(*batch)").variadic,
            Some(ShapeVariadic(Some(Name::new("batch"))))
        );
        assert_eq!(
            parse("(*#batch)").variadic,
            Some(ShapeVariadic(Some(Name::new("batch"))))
        );
    }

    #[test]
    fn arithmetic_splits_on_the_last_operator() {
        assert_eq!(
            flat("dim+1"),
            vec![ShapeDim::Add(atom("dim"), ShapeAtom::Literal(1))]
        );
        assert_eq!(
            flat("n-1"),
            vec![ShapeDim::Sub(atom("n"), ShapeAtom::Literal(1))]
        );
        // Parentheses exist only to stop Python evaluating the expression.
        assert_eq!(
            flat("(1+T)"),
            vec![ShapeDim::Add(ShapeAtom::Literal(1), atom("T"))]
        );
        assert_eq!(flat("(n)"), vec![named("n")]);
        assert_eq!(flat("(3)"), vec![ShapeDim::Literal(3)]);
    }

    #[test]
    fn a_negative_literal_is_not_a_subtraction() {
        assert_eq!(flat("-3"), vec![ShapeDim::Literal(-3)]);
    }

    #[test]
    fn a_variadic_splits_the_shape() {
        assert_eq!(
            parse("a *batch b c"),
            ParsedShape {
                prefix: vec![named("a")],
                variadic: Some(ShapeVariadic(Some(Name::new("batch")))),
                suffix: vec![named("b"), named("c")],
            }
        );
    }

    #[test]
    fn ellipsis_is_a_variadic_that_names_nothing() {
        assert_eq!(
            parse("... c"),
            ParsedShape {
                prefix: Vec::new(),
                variadic: Some(ShapeVariadic(None)),
                suffix: vec![named("c")],
            }
        );
    }

    #[test]
    fn two_variadics_have_no_desugared_form() {
        // `IntTuple::unpacked` has exactly one middle, so there is no way to
        // divide the concrete dimensions between two variadic segments.
        assert_eq!(
            parse_shape_string("*a *b c"),
            Err(ShapeStringError::MultipleVariadics)
        );
        assert_eq!(
            parse_shape_string("... *b"),
            Err(ShapeStringError::MultipleVariadics)
        );
    }

    #[test]
    fn chained_arithmetic_is_rejected() {
        assert_eq!(
            parse_shape_string("a+b+c"),
            Err(ShapeStringError::UnsupportedExpression("a+b+c".to_owned()))
        );
    }
}
