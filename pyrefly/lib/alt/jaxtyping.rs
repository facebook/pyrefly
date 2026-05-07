/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Jaxtyping annotation support.
//!
//! This module handles parsing and processing of jaxtyping-style tensor
//! annotations like `Float[Tensor, "batch channels"]`. The jaxtyping library
//! uses dtype wrapper classes (Float, Int, Shaped, etc.) subscripted with a
//! tensor class and a shape string.
//!
//! ## Shape string syntax
//!
//! The shape string is whitespace-separated and supports:
//! - Named dims (`"batch"`) → Quantified TypeVars
//! - Integer literals (`"3"`) → `Type::Size(SizeExpr::Literal(3))`
//! - Anonymous dim (`"_"`) → `Type::Any(AnyStyle::Implicit)`
//! - Variadic (`"*batch"`) → Quantified TypeVarTuples
//! - Ellipsis (`"..."`) → anonymous variadic (any number of any-sized dims)
//! - Broadcast (`"#batch"`) → treated as `"batch"` (conservative, safe)
//! - Combined (`"*#batch"`) → variadic TypeVarTuple, broadcast prefix stripped
//! - Arithmetic (`"dim+1"`, `"n-1"`) → `Type::Size(SizeExpr::Add/Sub(...))`
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
//!
//! ## Mixed syntax detection
//!
//! Native (`Tensor[N, M]`) and jaxtyping (`Float[Tensor, "N M"]`) syntax
//! cannot be mixed in the same function. This module detects and reports
//! such mixing.

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_types::dimension::SizeExpr;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::tensor::TensorShape;
use pyrefly_types::tensor::TensorSyntax;
use pyrefly_types::tensor::TensorType;
use pyrefly_types::types::TParams;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::class::Class;
use crate::types::types::AnyStyle;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Check if a class is a jaxtyping dtype wrapper (Float, Int, Shaped, etc.)
    pub fn is_jaxtyping_wrapper(&self, cls: &Class) -> bool {
        const WRAPPERS: &[&str] = &[
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
            "UInt",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Bool",
            "Num",
            "Shaped",
            "Complex",
            "Complex64",
            "Complex128",
            "Inexact",
        ];
        WRAPPERS
            .iter()
            .any(|name| cls.has_toplevel_qname("jaxtyping", name))
    }

    /// Parse a jaxtyping annotation like `Float[Tensor, "batch channels"]` into a TensorType.
    pub fn parse_jaxtyping_annotation(
        &self,
        xs: &[Expr],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if xs.len() != 2 {
            return self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                format!(
                    "jaxtyping annotations require exactly 2 arguments \
                     (array type and shape string), got {}",
                    xs.len()
                ),
            );
        }

        // Resolve xs[0] as the base array type (e.g., torch.Tensor).
        // With tensor_shapes enabled and shared fixtures, bare `Tensor` in a type
        // argument position resolves to Type::Tensor(shapeless). We extract the
        // base_class from it and apply the jaxtyping shape string instead.
        let base_type = self.expr_untype(&xs[0], TypeFormContext::TypeArgument, errors);
        let base_class = match &base_type {
            Type::Tensor(tensor_type) if tensor_type.is_shapeless() => {
                tensor_type.base_class.clone()
            }
            _ => {
                return self.error(
                    errors,
                    xs[0].range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!(
                        "First argument to jaxtyping annotation must be a tensor class, got `{}`",
                        self.for_display(base_type)
                    ),
                );
            }
        };

        // Extract shape string from xs[1]
        let shape_str = match &xs[1] {
            Expr::StringLiteral(ExprStringLiteral { value, .. }) => value.to_str(),
            _ => {
                return self.error(
                    errors,
                    xs[1].range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "Second argument to jaxtyping annotation must be a string literal".to_owned(),
                );
            }
        };

        // Parse shape string: split by whitespace.
        // split_whitespace() handles leading/trailing whitespace (jaxtyping convention).
        let tokens: Vec<&str> = shape_str.split_whitespace().collect();
        if tokens.is_empty() {
            // Empty shape string means scalar tensor (rank 0), like Tensor[()]
            let tensor_shape = TensorShape::from_types(vec![]);
            return TensorType::new(base_class, tensor_shape)
                .with_syntax(TensorSyntax::Jaxtyping)
                .to_type();
        }

        // Find variadic token: "*name", "*#name", or "...".
        // At most one variadic specifier is allowed per annotation.
        let var_pos = tokens
            .iter()
            .position(|t| t.starts_with('*') || *t == "...");

        if let Some(var_idx) = var_pos {
            // Check for multiple variadics
            if tokens[var_idx + 1..]
                .iter()
                .any(|t| t.starts_with('*') || *t == "...")
            {
                return self.error(
                    errors,
                    xs[1].range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "Tensor shape can have at most one variadic dimension".to_owned(),
                );
            }

            let prefix = self.parse_jaxtyping_dim_tokens(&tokens[..var_idx]);
            let suffix = self.parse_jaxtyping_dim_tokens(&tokens[var_idx + 1..]);

            let middle = if tokens[var_idx] == "..." {
                // Ellipsis: anonymous variadic matching any number of any-sized dims.
                // Represented as tuple[Any, ...], same as shapeless tensor middle.
                Type::any_tuple()
            } else {
                // "*name" or "*#name": named TypeVarTuple.
                // Strip leading '*', then strip optional broadcast '#' prefix.
                let var_name = &tokens[var_idx][1..];
                let var_name = var_name.strip_prefix('#').unwrap_or(var_name);
                let q = self
                    .get_or_create_jaxtyping_dim(Name::new(var_name), QuantifiedKind::TypeVarTuple);
                Type::Quantified(Box::new(q))
            };

            let tensor_shape = TensorShape::unpacked(prefix, middle, suffix);
            TensorType::new(base_class, tensor_shape)
                .with_syntax(TensorSyntax::Jaxtyping)
                .to_type()
        } else {
            // Concrete shape: all tokens are non-variadic dims
            let dims = self.parse_jaxtyping_dim_tokens(&tokens);
            let tensor_shape = TensorShape::from_types(dims);
            TensorType::new(base_class, tensor_shape)
                .with_syntax(TensorSyntax::Jaxtyping)
                .to_type()
        }
    }

    /// Parse a list of jaxtyping dimension tokens into types.
    ///
    /// Each token is processed through a prefix-stripping state machine matching
    /// jaxtyping's parser behavior:
    /// 1. Strip broadcast `#` prefix (treated as regular dim — conservative, safe)
    /// 2. `_` → `Type::Any(AnyStyle::Implicit)` (anonymous, any size)
    /// 3. Integer → `Type::Size(SizeExpr::Literal(n))`
    /// 4. Parenthesized → strip outer parens, parse inner as arithmetic
    /// 5. Contains `+`/`-` (not at position 0) → arithmetic expression
    /// 6. Named identifier → Quantified TypeVar (cached per module)
    fn parse_jaxtyping_dim_tokens(&self, tokens: &[&str]) -> Vec<Type> {
        tokens
            .iter()
            .map(|token| {
                // Strip broadcast prefix '#' (treated as regular dim for now)
                let token = token.strip_prefix('#').unwrap_or(token);

                // Anonymous dim: "_" matches any single dimension, not bound to a name
                if token == "_" {
                    return Type::Any(AnyStyle::Implicit);
                }

                // Integer literal: "3", "-1", etc.
                if let Ok(n) = token.parse::<i64>() {
                    return self.heap.mk_size(SizeExpr::literal(n));
                }

                // Parenthesized expression: "(dim+1)" → strip parens, parse as arithmetic
                if let Some(inner) = token.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                    && let Some(ty) = self.parse_jaxtyping_arithmetic(inner)
                {
                    return ty;
                }

                // Arithmetic: token contains '+' or '-' not at position 0
                if let Some(ty) = self.parse_jaxtyping_arithmetic(token) {
                    return ty;
                }

                // Named dimension: "batch", "channels", etc.
                let q = self.get_or_create_jaxtyping_dim(Name::new(token), QuantifiedKind::TypeVar);
                Type::Quantified(Box::new(q))
            })
            .collect()
    }

    /// Try to parse a jaxtyping dimension token as an arithmetic expression.
    ///
    /// Looks for the last `+` or `-` not at position 0 (to avoid treating
    /// negative integer literals like "-3" as subtraction). Splits into
    /// left/right atoms and creates `SizeExpr::Add` or `SizeExpr::Sub`.
    ///
    /// Returns `None` if the token contains no arithmetic operator.
    fn parse_jaxtyping_arithmetic(&self, token: &str) -> Option<Type> {
        // Find the last '+' or '-' not at position 0
        let (pos, op) = token
            .char_indices()
            .rev()
            .find(|&(i, c)| i > 0 && (c == '+' || c == '-'))?;

        let left_str = &token[..pos];
        let right_str = &token[pos + 1..];

        // Both operands must be non-empty
        if left_str.is_empty() || right_str.is_empty() {
            return None;
        }

        // Parse each operand as an integer literal or named dim
        let parse_atom = |s: &str| -> Type {
            if let Ok(n) = s.parse::<i64>() {
                self.heap.mk_size(SizeExpr::literal(n))
            } else {
                let q = self.get_or_create_jaxtyping_dim(Name::new(s), QuantifiedKind::TypeVar);
                Type::Quantified(Box::new(q))
            }
        };

        let left = parse_atom(left_str);
        let right = parse_atom(right_str);

        let size_expr = match op {
            '+' => SizeExpr::add(left, right),
            '-' => SizeExpr::sub(left, right),
            _ => unreachable!("only '+' and '-' are matched above"),
        };

        Some(self.heap.mk_size(size_expr))
    }

    /// Collect implicit jaxtyping TypeVars from a callable's signature and    /// extend `tparams` with them. Also detects and reports mixing of native
    /// and jaxtyping tensor annotation syntax in the same function.
    ///
    /// Returns the (potentially extended) `TParams` to use for the function's
    /// `Forall` wrapper.
    pub fn collect_jaxtyping_tparams(
        &self,
        callable: &impl Visit<Type>,
        tparams: &Arc<TParams>,
        name_range: TextRange,
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        if !self.solver().tensor_shapes {
            return tparams.dupe();
        }

        let mut jaxtyping_extras = Vec::new();
        let mut has_native = false;
        let mut has_jaxtyping = false;
        // Visit all types in the callable (params + return) to find jaxtyping
        // Quantified types and detect mixed tensor annotation syntax.
        callable.visit(&mut |ty: &Type| {
            if let Type::Quantified(q) = ty
                && self.is_jaxtyping_dim(q)
                && !tparams.iter().any(|existing| existing == q.as_ref())
                && !jaxtyping_extras.contains(q.as_ref())
            {
                jaxtyping_extras.push(q.as_ref().clone());
            }
            if let Type::Tensor(tensor_type) = ty
                && !tensor_type.is_shapeless()
            {
                match tensor_type.syntax {
                    TensorSyntax::Native => has_native = true,
                    TensorSyntax::Jaxtyping => has_jaxtyping = true,
                }
            }
        });
        if has_native && has_jaxtyping {
            self.error(
                errors,
                name_range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "Cannot mix native tensor syntax (Tensor[N, M]) and jaxtyping syntax \
                 (Float[Tensor, \"N M\"]) in the same function"
                    .to_owned(),
            );
        }
        if jaxtyping_extras.is_empty() {
            tparams.dupe()
        } else {
            let mut params: Vec<_> = tparams.as_vec().to_vec();
            params.extend(jaxtyping_extras);
            Arc::new(TParams::new(params))
        }
    }
}
