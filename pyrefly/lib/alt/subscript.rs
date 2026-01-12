/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use num_traits::ToPrimitive;
use pyrefly_python::dunder;
use pyrefly_types::lit_int::LitInt;
use pyrefly_types::literal::Lit;
use pyrefly_types::tuple::Tuple;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprSlice;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// When indexing/slicing tuples with literals, try to infer a more precise type
    pub fn infer_tuple_subscript(
        &self,
        tuple: Tuple,
        index: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let fallback = || {
            self.call_method_or_error(
                &Type::Tuple(tuple.clone()),
                &dunder::GETITEM,
                range,
                &[CallArg::expr(index)],
                &[],
                errors,
                context,
            )
        };
        match index {
            Expr::Slice(slice) => self
                .infer_tuple_slice(&tuple, slice)
                .unwrap_or_else(fallback),
            _ => self
                .infer_tuple_index(&tuple, index, errors)
                .unwrap_or_else(fallback),
        }
    }

    fn infer_tuple_slice(&self, tuple: &Tuple, slice: &ExprSlice) -> Option<Type> {
        if slice.step.is_some() {
            return None;
        }
        match tuple {
            Tuple::Concrete(elts) => self.infer_concrete_slice(elts, &slice.lower, &slice.upper),
            Tuple::Unpacked(box (prefix, middle, suffix)) => {
                self.infer_unpacked_slice(prefix, middle, suffix, &slice.lower, &slice.upper)
            }
            _ => None,
        }
    }

    fn infer_tuple_index(
        &self,
        tuple: &Tuple,
        index: &Expr,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let idx_type = self.expr_infer(index, errors);
        match &idx_type {
            Type::Literal(lit) if let Some(idx) = lit.as_index_i64() => match tuple {
                Tuple::Concrete(elts) => {
                    self.infer_concrete_index(elts, idx, index.range(), errors)
                }
                Tuple::Unpacked(box (prefix, _middle, suffix)) => {
                    self.infer_unpacked_index(prefix, suffix, idx)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns Ok(Some(_)) if the expression is a literal integer
    /// Returns Ok(None) if the expression is absent
    /// Returns Err(()) if the expression is present but is not a literal integer
    fn parse_slice_literal(&self, expr: &Option<Box<Expr>>) -> Result<Option<i64>, ()> {
        if let Some(expr) = expr {
            let literal_type = self.expr_infer(expr, &self.error_swallower());
            match &literal_type {
                Type::Literal(lit) => Ok(lit.as_index_i64()),
                _ => Err(()),
            }
        } else {
            Ok(None)
        }
    }

    fn infer_concrete_slice(
        &self,
        elts: &[Type],
        lower_expr: &Option<Box<Expr>>,
        upper_expr: &Option<Box<Expr>>,
    ) -> Option<Type> {
        let lower_literal = self.parse_slice_literal(lower_expr).ok()?.unwrap_or(0);
        let upper_literal = self
            .parse_slice_literal(upper_expr)
            .ok()?
            .unwrap_or(elts.len() as i64);
        if lower_literal >= 0 && upper_literal >= 0 && upper_literal <= elts.len() as i64 {
            if lower_literal >= upper_literal {
                Some(Type::concrete_tuple(Vec::new()))
            } else {
                Some(Type::concrete_tuple(
                    elts[lower_literal as usize..upper_literal as usize].to_vec(),
                ))
            }
        } else {
            None
        }
    }

    fn infer_concrete_index(
        &self,
        elts: &[Type],
        idx: i64,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let elt_idx = if idx >= 0 {
            idx
        } else {
            elts.len() as i64 + idx
        } as usize;
        if let Some(elt) = elts.get(elt_idx) {
            Some(elt.clone())
        } else {
            Some(self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadIndex),
                format!(
                    "Index {idx} out of range for tuple with {} elements",
                    elts.len()
                ),
            ))
        }
    }

    fn infer_unpacked_slice(
        &self,
        prefix: &[Type],
        middle: &Type,
        suffix: &[Type],
        lower_expr: &Option<Box<Expr>>,
        upper_expr: &Option<Box<Expr>>,
    ) -> Option<Type> {
        let lower_literal = self.parse_slice_literal(lower_expr).ok()?.unwrap_or(0);
        let upper_literal = self.parse_slice_literal(upper_expr).ok()?;
        match (lower_literal, upper_literal) {
            // Positive lower, positive upper
            (lower, Some(upper)) if lower >= 0 && upper >= 0 => {
                let lower_usize = lower as usize;
                let upper_usize = upper as usize;
                if lower_usize >= upper_usize {
                    return Some(Type::concrete_tuple(Vec::new()));
                }
                if upper_usize <= prefix.len() {
                    // Slice is entirely within prefix
                    Some(Type::concrete_tuple(
                        prefix[lower_usize..upper_usize].to_vec(),
                    ))
                } else if lower_usize <= prefix.len() {
                    // Slice starts in/immediately after prefix and extends beyond
                    let prefix_tail = prefix[lower_usize..].to_vec();
                    Some(Type::Tuple(Tuple::unpacked(
                        prefix_tail,
                        middle.clone(),
                        suffix.to_vec(),
                    )))
                } else {
                    // Slice starts beyond prefix
                    None
                }
            }
            // Positive lower, negative upper
            (lower, Some(upper)) if lower >= 0 && upper < 0 => {
                let lower_usize = lower as usize;
                let neg_upper = (-upper) as usize;
                if neg_upper <= suffix.len() {
                    let prefix_tail = prefix[lower_usize.min(prefix.len())..].to_vec();
                    let suffix_head = suffix[..suffix.len() - neg_upper].to_vec();
                    if lower_usize <= prefix.len() {
                        Some(Type::Tuple(Tuple::unpacked(
                            prefix_tail,
                            middle.clone(),
                            suffix_head,
                        )))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            // No lower, negative upper
            (0, Some(upper)) if upper < 0 => {
                let neg_upper = (-upper) as usize;
                if neg_upper <= suffix.len() {
                    let suffix_head = suffix[..suffix.len() - neg_upper].to_vec();
                    Some(Type::Tuple(Tuple::unpacked(
                        prefix.to_vec(),
                        middle.clone(),
                        suffix_head,
                    )))
                } else {
                    None
                }
            }
            // Positive lower, no upper
            (lower, None) if lower >= 0 && (lower as usize) <= prefix.len() => {
                let lower_usize = lower as usize;
                let prefix_tail = prefix[lower_usize..].to_vec();
                Some(Type::Tuple(Tuple::unpacked(
                    prefix_tail,
                    middle.clone(),
                    suffix.to_vec(),
                )))
            }
            _ => None,
        }
    }

    // Handle a positive index within the prefix or a negative index within the suffix
    fn infer_unpacked_index(&self, prefix: &[Type], suffix: &[Type], idx: i64) -> Option<Type> {
        if idx >= 0 {
            let elt_idx = idx as usize;
            if elt_idx < prefix.len() {
                Some(prefix[elt_idx].clone())
            } else {
                None
            }
        } else {
            let neg_idx = (-idx) as usize;
            if neg_idx <= suffix.len() {
                Some(suffix[suffix.len() - neg_idx].clone())
            } else {
                None
            }
        }
    }

    pub fn subscript_str_literal(
        &self,
        value: &str,
        base_type: &Type,
        index_expr: &Expr,
        errors: &ErrorCollector,
        range: TextRange,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let fallback = || {
            self.call_method_or_error(
                base_type,
                &dunder::GETITEM,
                range,
                &[CallArg::expr(index_expr)],
                &[],
                errors,
                context,
            )
        };

        if matches!(index_expr, Expr::Tuple(_)) {
            return fallback();
        }

        let literal_index = |expr: &Expr| -> Option<i64> {
            match self.expr_infer(expr, errors) {
                Type::Literal(ref lit) => lit.as_index_i64(),
                _ => None,
            }
        };

        let chars: Vec<char> = value.chars().collect();
        let len_usize = chars.len();
        if len_usize > i64::MAX as usize {
            return fallback();
        }
        let len = len_usize as i64;

        if let Expr::Slice(slice) = index_expr {
            let step = match slice.step.as_deref() {
                Some(expr) => match literal_index(expr) {
                    Some(value) if value != 0 => value,
                    _ => return fallback(),
                },
                None => 1,
            };

            if step == i64::MIN {
                return fallback();
            }

            let mut start = match slice.lower.as_deref() {
                Some(expr) => match literal_index(expr) {
                    Some(value) => value,
                    None => return fallback(),
                },
                None => {
                    if step < 0 {
                        len.saturating_sub(1)
                    } else {
                        0
                    }
                }
            };

            let mut stop = match slice.upper.as_deref() {
                Some(expr) => match literal_index(expr) {
                    Some(value) => value,
                    None => return fallback(),
                },
                None => {
                    if step < 0 {
                        match len.checked_add(1) {
                            Some(v) => -v,
                            None => return fallback(),
                        }
                    } else {
                        len
                    }
                }
            };

            if step > 0 {
                if start < 0 {
                    start += len;
                    if start < 0 {
                        start = 0;
                    }
                } else if start > len {
                    start = len;
                }

                if stop < 0 {
                    stop += len;
                    if stop < 0 {
                        stop = 0;
                    }
                } else if stop > len {
                    stop = len;
                }
            } else {
                if start < 0 {
                    start += len;
                    if start < 0 {
                        start = -1;
                    }
                } else if start >= len {
                    start = len.saturating_sub(1);
                }

                if stop < 0 {
                    stop += len;
                    if stop < 0 {
                        stop = -1;
                    }
                } else if stop >= len {
                    stop = len.saturating_sub(1);
                }
            }

            let slice_length = if step < 0 {
                if stop < start {
                    (start - stop - 1) / (-step) + 1
                } else {
                    0
                }
            } else if start < stop {
                (stop - start - 1) / step + 1
            } else {
                0
            };

            if slice_length <= 0 {
                return Type::Literal(Lit::Str("".into()));
            }

            if slice_length as usize as i64 != slice_length {
                return fallback();
            }

            let mut result = String::new();
            let mut idx = start;
            for _ in 0..slice_length as usize {
                if idx < 0 || idx >= len {
                    return fallback();
                }
                let Some(&ch) = chars.get(idx as usize) else {
                    return fallback();
                };
                result.push(ch);
                idx = match idx.checked_add(step) {
                    Some(next) => next,
                    None => return fallback(),
                };
            }

            Type::Literal(Lit::Str(result.into()))
        } else {
            let idx_ty = self.expr_infer(index_expr, errors);
            if let Type::Literal(lit) = idx_ty
                && let Some(idx) = lit.as_index_i64()
            {
                let normalized = if idx < 0 { len + idx } else { idx };
                if normalized >= 0 && normalized < len {
                    let ch = chars[normalized as usize];
                    let mut buf = String::new();
                    buf.push(ch);
                    return Type::Literal(Lit::Str(buf.into()));
                } else {
                    return self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::BadIndex),
                        format!(
                            "Index `{idx}` out of range for string with {} elements",
                            chars.len()
                        ),
                    );
                }
            }
            fallback()
        }
    }

    pub fn subscript_bytes_literal(
        &self,
        bytes: &[u8],
        index_expr: &Expr,
        errors: &ErrorCollector,
        range: TextRange,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let index_ty = self.expr_infer(index_expr, errors);
        match &index_ty {
            Type::Literal(lit) => {
                if let Some(idx) = lit.as_index_i64() {
                    if idx >= 0
                        && let Some(byte) = idx.to_usize().and_then(|idx| bytes.get(idx))
                    {
                        Type::Literal(Lit::Int(LitInt::new((*byte).into())))
                    } else if idx < 0
                        && let Some(byte) = idx
                            .checked_neg()
                            .and_then(|idx| idx.to_usize())
                            .and_then(|idx| bytes.len().checked_sub(idx))
                            .and_then(|idx| bytes.get(idx))
                    {
                        Type::Literal(Lit::Int(LitInt::new((*byte).into())))
                    } else {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadIndex),
                            format!(
                                "Index `{idx}` out of range for bytes with {} elements",
                                bytes.len()
                            ),
                        )
                    }
                } else {
                    self.call_method_or_error(
                        &self.stdlib.bytes().clone().to_type(),
                        &dunder::GETITEM,
                        range,
                        &[CallArg::expr(index_expr)],
                        &[],
                        errors,
                        context,
                    )
                }
            }
            _ => self.call_method_or_error(
                &self.stdlib.bytes().clone().to_type(),
                &dunder::GETITEM,
                range,
                &[CallArg::expr(index_expr)],
                &[],
                errors,
                context,
            ),
        }
    }
}
