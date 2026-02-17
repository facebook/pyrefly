/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Dimension types and operations for tensor shape inference.
//!
//! This module provides:
//! - `SizeExpr`: Symbolic dimension expressions (literals, arithmetic operations)
//! - Simplification: Algebraic simplification of dimension expressions
//! - Canonicalization: Normalization to unique canonical forms for comparison

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;

use crate::equality::TypeEq;
use crate::literal::Lit;
use crate::literal::Literal;
use crate::types::AnyStyle;
use crate::types::Type;

/// A dimension expression in a tensor shape.
///
/// Dimensions can be:
/// - Concrete literals: `Tensor[2, 3]`
/// - Symbolic expressions: `Tensor[N, N+1]`, `Tensor[N*M]`
///
/// Type variables (`Type::Quantified`), solver variables (`Type::Var`), and
/// unknown dimensions (`Type::Any`) are represented directly as `Type` in
/// `TensorShape.dims`, not wrapped in `SizeExpr`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SizeExpr {
    /// Concrete dimension: Tensor[2, 3]
    /// Only positive integers are allowed
    Literal(i64),

    /// Addition: N + M (for concat, etc.)
    Add(Box<Type>, Box<Type>),

    /// Subtraction: N - M
    Sub(Box<Type>, Box<Type>),

    /// Multiplication: N * M (for reshape, etc.)
    Mul(Box<Type>, Box<Type>),

    /// Floor division: N // M
    FloorDiv(Box<Type>, Box<Type>),
}

impl SizeExpr {
    pub fn literal(value: i64) -> Self {
        Self::Literal(value)
    }

    pub fn as_literal(&self) -> Option<i64> {
        match self {
            Self::Literal(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }

    /// Helper constructors for expressions.
    /// Take Type arguments to support type variables in expressions.
    pub fn add(left: Type, right: Type) -> Self {
        Self::Add(Box::new(left), Box::new(right))
    }

    pub fn sub(left: Type, right: Type) -> Self {
        Self::Sub(Box::new(left), Box::new(right))
    }

    pub fn mul(left: Type, right: Type) -> Self {
        Self::Mul(Box::new(left), Box::new(right))
    }

    pub fn floor_div(left: Type, right: Type) -> Self {
        Self::FloorDiv(Box::new(left), Box::new(right))
    }

    /// Convert a Type to a SizeExpr (used for extracting literal dimensions).
    /// Returns None if the type is not a concrete literal or expression.
    /// Type variables, Vars, and Any should remain as Type in TensorShape.dims.
    pub fn from_type(ty: &Type) -> Option<SizeExpr> {
        match ty {
            // SizeExpr type -> unwrap and return the SizeExpr directly
            Type::Size(dim) => Some(dim.clone()),
            // Literal integer -> Literal dimension
            Type::Literal(box Literal {
                value: Lit::Int(i), ..
            }) => i.as_i64().map(SizeExpr::Literal),
            // Symbolic integer -> recursively extract SizeExpr from the Type
            Type::Dim(ty) => SizeExpr::from_type(ty),
            // All other types (Quantified, Var, Any, etc.) should remain as Type
            _ => None,
        }
    }
}

impl Display for SizeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(n) => write!(f, "{}", n),
            Self::Add(left, right) => write!(f, "({} + {})", left, right),
            Self::Sub(left, right) => write!(f, "({} - {})", left, right),
            Self::Mul(left, right) => {
                // Simplify display: (1 * x) -> x, (x * 1) -> x
                match (left.as_ref(), right.as_ref()) {
                    (Type::Size(SizeExpr::Literal(1)), _) => write!(f, "{}", right),
                    (_, Type::Size(SizeExpr::Literal(1))) => write!(f, "{}", left),
                    _ => write!(f, "({} * {})", left, right),
                }
            }
            Self::FloorDiv(left, right) => {
                // Simplify display: (x // 1) -> x
                if matches!(right.as_ref(), Type::Size(SizeExpr::Literal(1))) {
                    write!(f, "{}", left)
                } else {
                    write!(f, "({} // {})", left, right)
                }
            }
        }
    }
}

// ============================================================================
// Simplification
// ============================================================================

/// Simplify a dimension expression to canonical form.
///
/// This applies algebraic simplifications:
/// - Constant folding: 2 + 3 → 5
/// - Identity elimination: N + 0 → N, N * 1 → N
/// - Zero propagation: N * 0 → 0
/// - Any propagation: expressions containing Any become Any
///
/// Any subexpression with only literal leaves is evaluated.
pub fn simplify(ty: Type) -> Type {
    match ty {
        Type::Size(dim) => {
            if contains_any_in_sizeexpr(&dim) {
                return Type::Any(AnyStyle::Explicit);
            }
            simplify_sizeexpr(dim)
        }
        // Quantified, Var, Any, Dim are already canonical
        other => other,
    }
}

/// Inner simplification that skips the Any check.
/// Called after the top-level `simplify` has already verified no Any is present.
fn simplify_inner(ty: Type) -> Type {
    match ty {
        Type::Size(dim) => simplify_sizeexpr(dim),
        other => other,
    }
}

fn simplify_sizeexpr(dim: SizeExpr) -> Type {
    match dim {
        // Addition: N + M
        SizeExpr::Add(left, right) => {
            let left = simplify_inner(*left);
            let right = simplify_inner(*right);

            match (left.as_shape_literal(), right.as_shape_literal()) {
                // Both literal - evaluate
                (Some(a), Some(b)) => Type::Size(SizeExpr::Literal(a + b)),
                // N + 0 = N
                (_, Some(0)) => left,
                // 0 + N = N
                (Some(0), _) => right,
                // Can't simplify further
                _ => Type::Size(SizeExpr::Add(Box::new(left), Box::new(right))),
            }
        }

        // Subtraction: N - M
        SizeExpr::Sub(left, right) => {
            let left = simplify_inner(*left);
            let right = simplify_inner(*right);

            match (left.as_shape_literal(), right.as_shape_literal()) {
                // Both literal (only simplify if result is non-negative)
                (Some(a), Some(b)) if a >= b => Type::Size(SizeExpr::Literal(a - b)),
                // N - 0 = N
                (_, Some(0)) => left,
                // Can't simplify further
                _ => Type::Size(SizeExpr::Sub(Box::new(left), Box::new(right))),
            }
        }

        // Multiplication: N * M
        SizeExpr::Mul(left, right) => {
            let left = simplify_inner(*left);
            let right = simplify_inner(*right);
            let left_lit = left.as_shape_literal();
            let right_lit = right.as_shape_literal();

            match (left_lit, right_lit) {
                // Both literal - evaluate
                (Some(a), Some(b)) => Type::Size(SizeExpr::Literal(a * b)),
                // N * 0 = 0 or 0 * N = 0
                (Some(0), _) | (_, Some(0)) => Type::Size(SizeExpr::Literal(0)),
                // N * 1 = N
                (_, Some(1)) => left,
                // 1 * N = N
                (Some(1), _) => right,
                // Can't simplify further
                _ => Type::Size(SizeExpr::Mul(Box::new(left), Box::new(right))),
            }
        }

        // Floor Division: N // M
        SizeExpr::FloorDiv(left, right) => {
            let left = simplify_inner(*left);
            let right = simplify_inner(*right);
            let left_lit = left.as_shape_literal();
            let right_lit = right.as_shape_literal();

            match (left_lit, right_lit) {
                // Both literal (avoid division by zero)
                (Some(a), Some(b)) if b != 0 => Type::Size(SizeExpr::Literal(a / b)),
                // N / 1 = N
                (_, Some(1)) => left,
                // 0 / N = 0
                (Some(0), _) => Type::Size(SizeExpr::Literal(0)),
                // Can't simplify further
                _ => Type::Size(SizeExpr::FloorDiv(Box::new(left), Box::new(right))),
            }
        }

        // Literal - already canonical
        SizeExpr::Literal(_) => Type::Size(dim),
    }
}

// ============================================================================
// Canonicalization
// ============================================================================

/// Canonicalize a dimension expression to a unique normal form.
///
/// This transforms dimension expressions into a canonical form where:
/// - Like terms are combined (e.g., 4*N + 2*N = 6*N)
/// - Divisions are flattened (e.g., (N // M) // K = N // (M*K))
/// - Factors are GCD-reduced (e.g., (4*N) // (6*M) = (2*N) // (3*M))
/// - Expressions are ordered consistently
/// - Type::Any propagates through the entire expression
///
/// This enables structural equality checking after canonicalization.
pub fn canonicalize(ty: Type) -> Type {
    // Normalize and canonicalize based on type
    match ty {
        Type::Size(dim) => {
            // Check for Any - if present anywhere, entire expression becomes Any
            if contains_any_in_sizeexpr(&dim) {
                return Type::Any(AnyStyle::Implicit);
            }
            canonicalize_sizeexpr(dim)
        }
        // Quantified, Var, Any, Dim, Literal are already canonical
        other => other,
    }
}

/// Inner canonicalization that skips the Any check.
/// Called after the top-level `canonicalize` has already verified no Any is present.
fn canonicalize_inner(ty: Type) -> Type {
    match ty {
        Type::Size(dim) => canonicalize_sizeexpr(dim),
        other => other,
    }
}

/// Check if Type::Any appears anywhere in the expression tree
fn contains_any(ty: &Type) -> bool {
    match ty {
        Type::Any(_) => true,
        Type::Size(dim) => contains_any_in_sizeexpr(dim),
        _ => false,
    }
}

fn contains_any_in_sizeexpr(dim: &SizeExpr) -> bool {
    match dim {
        SizeExpr::Add(left, right)
        | SizeExpr::Sub(left, right)
        | SizeExpr::Mul(left, right)
        | SizeExpr::FloorDiv(left, right) => contains_any(left) || contains_any(right),
        SizeExpr::Literal(_) => false,
    }
}

/// Main canonicalization function for SizeExpr expressions
fn canonicalize_sizeexpr(dim: SizeExpr) -> Type {
    match dim {
        SizeExpr::Literal(_) => Type::Size(dim),
        SizeExpr::Add(left, right) => canonicalize_sum(*left, *right),
        SizeExpr::Sub(left, right) => {
            // Normalize: a - b → a + (-1) * b
            let neg_one = Type::Size(SizeExpr::Literal(-1));
            let neg_right = Type::Size(SizeExpr::Mul(Box::new(neg_one), right));
            canonicalize_sum(*left, neg_right)
        }
        SizeExpr::Mul(left, right) => canonicalize_product(*left, *right),
        SizeExpr::FloorDiv(left, right) => canonicalize_division(*left, *right),
    }
}

/// Canonicalize a sum expression
fn canonicalize_sum(left: Type, right: Type) -> Type {
    // Step 1: Recursively canonicalize operands
    let left_canon = canonicalize_inner(left);
    let right_canon = canonicalize_inner(right);

    // Step 2: Flatten to list of terms
    let mut terms = Vec::new();
    collect_terms(left_canon, &mut terms);
    collect_terms(right_canon, &mut terms);

    // Step 3: Combine like terms by extracting coefficients
    #[allow(clippy::mutable_key_type)]
    let mut term_map: HashMap<Type, i64> = HashMap::new();

    for term in terms {
        let (coeff, non_literal_part) = extract_coefficient(term.clone());
        *term_map.entry(non_literal_part).or_insert(0) += coeff;
    }

    // Step 4: Rebuild terms, filtering out zero coefficients
    let mut new_terms = Vec::new();
    for (part, coeff) in term_map {
        if coeff == 0 {
            continue;
        }

        if matches!(part, Type::Size(SizeExpr::Literal(1))) {
            // Coefficient only (no non-literal part)
            new_terms.push(Type::Size(SizeExpr::Literal(coeff)));
        } else if coeff == 1 {
            // Coefficient is 1, just use the part
            new_terms.push(part);
        } else {
            // General case: coeff * part
            let coeff_ty = Type::Size(SizeExpr::Literal(coeff));
            new_terms.push(Type::Size(SizeExpr::Mul(
                Box::new(coeff_ty),
                Box::new(part),
            )));
        }
    }

    // Step 5: Sort terms by canonical order
    new_terms.sort_by(compare_type);

    // Step 6: Build result
    rebuild_sum(new_terms)
}

/// Generic function to collect operands from a binary SizeExpr expression.
fn collect_operands(
    ty: Type,
    items: &mut Vec<Type>,
    extract: fn(&SizeExpr) -> Option<(&Type, &Type)>,
) {
    match &ty {
        Type::Size(dim) => {
            if let Some((left, right)) = extract(dim) {
                collect_operands(left.clone(), items, extract);
                collect_operands(right.clone(), items, extract);
            } else {
                items.push(ty);
            }
        }
        _ => items.push(ty),
    }
}

fn extract_add_operands(dim: &SizeExpr) -> Option<(&Type, &Type)> {
    match dim {
        SizeExpr::Add(l, r) => Some((l.as_ref(), r.as_ref())),
        _ => None,
    }
}

fn extract_mul_operands(dim: &SizeExpr) -> Option<(&Type, &Type)> {
    match dim {
        SizeExpr::Mul(l, r) => Some((l.as_ref(), r.as_ref())),
        _ => None,
    }
}

fn collect_terms(ty: Type, terms: &mut Vec<Type>) {
    collect_operands(ty, terms, extract_add_operands);
}

/// Rebuild a sum expression from a list of terms.
fn rebuild_sum(terms: Vec<Type>) -> Type {
    if terms.is_empty() {
        Type::Size(SizeExpr::Literal(0))
    } else if terms.len() == 1 {
        terms.into_iter().next().unwrap()
    } else {
        let mut iter = terms.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, term| {
            Type::Size(SizeExpr::Add(Box::new(acc), Box::new(term)))
        })
    }
}

/// Separate literal factors from non-literal factors, computing their product.
fn separate_literal_factors(factors: Vec<Type>) -> (i64, Vec<Type>) {
    let literal_product: i64 = factors
        .iter()
        .filter_map(|f| f.as_shape_literal())
        .product();

    let non_literal: Vec<Type> = factors
        .into_iter()
        .filter(|f| f.as_shape_literal().is_none())
        .collect();

    (literal_product, non_literal)
}

/// Extract coefficient and non-literal part from a term
fn extract_coefficient(term: Type) -> (i64, Type) {
    match term {
        Type::Size(SizeExpr::Literal(n)) => (n, Type::Size(SizeExpr::Literal(1))),
        Type::Size(SizeExpr::Mul(_, _)) => {
            // Collect all factors
            let mut factors = Vec::new();
            collect_factors(term, &mut factors);

            // Separate literal from non-literal factors
            let (coeff, non_literal_factors) = separate_literal_factors(factors);

            let non_literal_part = if non_literal_factors.is_empty() {
                Type::Size(SizeExpr::Literal(1))
            } else {
                rebuild_product(non_literal_factors)
            };

            (coeff, non_literal_part)
        }
        other => (1, other),
    }
}

/// Canonicalize a product expression
fn canonicalize_product(left: Type, right: Type) -> Type {
    // Step 1: Recursively canonicalize operands
    let left_canon = canonicalize_inner(left);
    let right_canon = canonicalize_inner(right);

    // Step 2: Flatten to list of factors
    let mut factors = Vec::new();
    collect_factors(left_canon, &mut factors);
    collect_factors(right_canon, &mut factors);

    // Step 3: Check for zero
    if factors
        .iter()
        .any(|f| matches!(f, Type::Size(SizeExpr::Literal(0))))
    {
        return Type::Size(SizeExpr::Literal(0));
    }

    // Step 4: Separate literals from non-literals
    let (literal_product, mut non_literal_factors) = separate_literal_factors(factors);

    // Step 5: Sort factors by canonical order
    non_literal_factors.sort_by(compare_type);

    // Step 6: Add literal coefficient if not 1
    let mut all_factors = Vec::new();
    if literal_product != 1 {
        all_factors.push(Type::Size(SizeExpr::Literal(literal_product)));
    }
    all_factors.extend(non_literal_factors);

    // Step 7: Build result
    if all_factors.is_empty() {
        Type::Size(SizeExpr::Literal(1))
    } else {
        rebuild_product(all_factors)
    }
}

fn collect_factors(ty: Type, factors: &mut Vec<Type>) {
    collect_operands(ty, factors, extract_mul_operands);
}

fn rebuild_product(factors: Vec<Type>) -> Type {
    if factors.is_empty() {
        Type::Size(SizeExpr::Literal(1))
    } else if factors.len() == 1 {
        factors.into_iter().next().unwrap()
    } else {
        let mut iter = factors.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, f| {
            Type::Size(SizeExpr::Mul(Box::new(acc), Box::new(f)))
        })
    }
}

/// Canonicalize a floor division expression
fn canonicalize_division(num: Type, den: Type) -> Type {
    // Step 1: Canonicalize the numerator
    let canonical_num = canonicalize_inner(num);

    // Step 2: Check if numerator is a division - if so, flatten
    if let Type::Size(SizeExpr::FloorDiv(inner_num, inner_den)) = canonical_num {
        // Apply composition law: (a // b) // c = a // (b * c)
        let new_den = Type::Size(SizeExpr::Mul(inner_den, Box::new(den)));
        return canonicalize_division(*inner_num, new_den);
    }

    // Step 3: Now canonicalize the denominator
    let canonical_den = canonicalize_inner(den);

    // Step 4: Apply simplifications
    match (&canonical_num, &canonical_den) {
        // 0 // a = 0
        (Type::Size(SizeExpr::Literal(0)), _) => Type::Size(SizeExpr::Literal(0)),

        // a // 1 = a
        (_, Type::Size(SizeExpr::Literal(1))) => canonical_num,

        // Both literals: compute
        (Type::Size(SizeExpr::Literal(n)), Type::Size(SizeExpr::Literal(d))) if *d != 0 => {
            Type::Size(SizeExpr::Literal(n / d))
        }

        // Try cancellation
        _ => {
            let (new_num, new_den) = try_cancel_common_factors(canonical_num, canonical_den);

            // If denominator is 1 after cancellation, return numerator
            if matches!(new_den, Type::Size(SizeExpr::Literal(1))) {
                new_num
            } else {
                Type::Size(SizeExpr::FloorDiv(Box::new(new_num), Box::new(new_den)))
            }
        }
    }
}

/// Try to cancel common factors between numerator and denominator
fn try_cancel_common_factors(num: Type, den: Type) -> (Type, Type) {
    // Extract factors from numerator and denominator
    let mut num_factors = Vec::new();
    let mut den_factors = Vec::new();
    collect_factors(num, &mut num_factors);
    collect_factors(den, &mut den_factors);

    // Step 1: Separate literals from non-literals
    let (num_literal, mut num_factors) = separate_literal_factors(num_factors);
    let (den_literal, mut den_factors) = separate_literal_factors(den_factors);

    // Step 2: Apply GCD to literals
    let g = gcd(num_literal.abs(), den_literal.abs());
    let new_num_literal = num_literal / g;
    let new_den_literal = den_literal / g;

    // Step 3: Find and remove structurally equal non-literal factors
    let mut i = 0;
    while i < num_factors.len() {
        if let Some(pos) = den_factors.iter().position(|df| num_factors[i] == *df) {
            // Found a common factor - cancel it
            num_factors.remove(i);
            den_factors.remove(pos);
            // Don't increment i, check the same position again
        } else {
            i += 1;
        }
    }

    // Step 4: Rebuild numerator
    if new_num_literal != 1 {
        num_factors.insert(0, Type::Size(SizeExpr::Literal(new_num_literal)));
    }
    let new_num = rebuild_product(num_factors);

    // Step 5: Rebuild denominator
    if new_den_literal != 1 {
        den_factors.insert(0, Type::Size(SizeExpr::Literal(new_den_literal)));
    }
    let new_den = rebuild_product(den_factors);

    (new_num, new_den)
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Compare types for canonical ordering.
/// Ordering: Literal < Quantified < SizeExpr(FloorDiv) < SizeExpr(Mul) < SizeExpr(Add) < SizeExpr(Sub)
fn compare_type(a: &Type, b: &Type) -> Ordering {
    match (a, b) {
        // Literals: compare numerically
        (Type::Size(SizeExpr::Literal(n1)), Type::Size(SizeExpr::Literal(n2))) => n1.cmp(n2),

        // Type ordering
        (Type::Size(SizeExpr::Literal(_)), _) => Ordering::Less,
        (_, Type::Size(SizeExpr::Literal(_))) => Ordering::Greater,

        // Quantified
        (Type::Quantified(q1), Type::Quantified(q2)) => q1.cmp(q2),
        (Type::Quantified(_), Type::Size(_)) => Ordering::Less,
        (Type::Size(_), Type::Quantified(_)) => Ordering::Greater,

        // SizeExpr variants
        (Type::Size(d1), Type::Size(d2)) => compare_sizeexpr(d1, d2),

        // Fallback: types that shouldn't appear in dimension expressions
        _ => Ordering::Equal,
    }
}

fn compare_sizeexpr(a: &SizeExpr, b: &SizeExpr) -> Ordering {
    use SizeExpr::*;
    match (a, b) {
        (Literal(n1), Literal(n2)) => n1.cmp(n2),

        // Type ordering: Literal < FloorDiv < Mul < Add < Sub
        (Literal(_), _) => Ordering::Less,
        (_, Literal(_)) => Ordering::Greater,

        (FloorDiv(_, _), Mul(_, _) | Add(_, _) | Sub(_, _)) => Ordering::Less,
        (Mul(_, _) | Add(_, _) | Sub(_, _), FloorDiv(_, _)) => Ordering::Greater,

        (Mul(_, _), Add(_, _) | Sub(_, _)) => Ordering::Less,
        (Add(_, _) | Sub(_, _), Mul(_, _)) => Ordering::Greater,

        (Add(_, _), Sub(_, _)) => Ordering::Less,
        (Sub(_, _), Add(_, _)) => Ordering::Greater,

        // Same variant: compare lexicographically
        (FloorDiv(n1, d1), FloorDiv(n2, d2))
        | (Mul(n1, d1), Mul(n2, d2))
        | (Add(n1, d1), Add(n2, d2))
        | (Sub(n1, d1), Sub(n2, d2)) => match compare_type(n1, n2) {
            Ordering::Equal => compare_type(d1, d2),
            other => other,
        },
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl pyrefly_util::visit::Visit<Type> for SizeExpr {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        match self {
            SizeExpr::Literal(_) => {}
            SizeExpr::Add(left, right)
            | SizeExpr::Sub(left, right)
            | SizeExpr::Mul(left, right)
            | SizeExpr::FloorDiv(left, right) => {
                f(left);
                f(right);
            }
        }
    }
}

impl pyrefly_util::visit::VisitMut<Type> for SizeExpr {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        match self {
            SizeExpr::Literal(_) => {}
            SizeExpr::Add(left, right)
            | SizeExpr::Sub(left, right)
            | SizeExpr::Mul(left, right)
            | SizeExpr::FloorDiv(left, right) => {
                f(left);
                f(right);
            }
        }
    }
}

impl TypeEq for SizeExpr {}

// ============================================================================
// Shape Errors
// ============================================================================

/// Errors that can occur during shape/dimension checking
#[derive(Debug, Clone)]
pub enum ShapeError {
    /// Tensor ranks don't match
    RankMismatch { got: usize, want: usize },

    /// Invalid dimension value (e.g., negative or zero)
    InvalidDimension { value: i64, reason: String },

    /// Structural mismatch between dimension types
    StructuralMismatch {
        got: String,
        got_canonical: String,
        want: String,
        want_canonical: String,
    },

    /// Type variable in nested position cannot be inferred
    /// For example: passing Dim[(A * B) // 2] to parameter Dim[X // 2]
    /// X appears in a nested position (inside // 2) and cannot be inferred
    NestedTypeVarNotInferred,

    /// Cannot index a scalar (rank-0) tensor
    ScalarIndex,

    /// Too many indices for tensor rank
    TooManyIndices { got: usize, max: usize },
}

impl Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { got, want } => {
                write!(
                    f,
                    "Tensor rank mismatch: expected {} dimensions, got {} dimensions",
                    want, got
                )
            }
            Self::InvalidDimension { value, reason } => {
                write!(f, "Invalid dimension value {}: {}", value, reason)
            }
            Self::StructuralMismatch {
                got: _,
                got_canonical,
                want: _,
                want_canonical,
            } => {
                write!(
                    f,
                    "Size mismatch: expected {}, got {}",
                    want_canonical, got_canonical
                )
            }
            Self::NestedTypeVarNotInferred => {
                write!(f, "Type variable cannot be inferred from a nested position")
            }
            Self::ScalarIndex => {
                write!(f, "Cannot index scalar tensor (rank 0)")
            }
            Self::TooManyIndices { got, max } => {
                write!(
                    f,
                    "Too many indices for tensor: got {}, expected at most {}",
                    got, max
                )
            }
        }
    }
}

impl ShapeError {
    pub fn rank_mismatch(got: usize, want: usize) -> Self {
        Self::RankMismatch { got, want }
    }

    pub fn invalid_dimension(value: i64, reason: impl Into<String>) -> Self {
        Self::InvalidDimension {
            value,
            reason: reason.into(),
        }
    }

    pub fn structural_mismatch(
        got: impl Into<String>,
        got_canonical: impl Into<String>,
        want: impl Into<String>,
        want_canonical: impl Into<String>,
    ) -> Self {
        Self::StructuralMismatch {
            got: got.into(),
            got_canonical: got_canonical.into(),
            want: want.into(),
            want_canonical: want_canonical.into(),
        }
    }

    pub fn nested_type_var_not_inferred() -> Self {
        Self::NestedTypeVarNotInferred
    }
}

/// Check if a dimension type contains a solver Var anywhere in its structure.
/// This is used to detect when a type variable in a nested position cannot be inferred.
pub fn contains_var_in_type(ty: &Type) -> bool {
    match ty {
        Type::Var(_) => true,
        Type::Size(dim) => contains_var_in_size_expr(dim),
        _ => false,
    }
}

fn contains_var_in_size_expr(dim: &SizeExpr) -> bool {
    match dim {
        SizeExpr::Add(left, right)
        | SizeExpr::Sub(left, right)
        | SizeExpr::Mul(left, right)
        | SizeExpr::FloorDiv(left, right) => {
            contains_var_in_type(left) || contains_var_in_type(right)
        }
        _ => false,
    }
}
