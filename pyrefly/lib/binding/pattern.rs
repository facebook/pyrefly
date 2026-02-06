/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::nesting_context::NestingContext;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::MatchCase;
use ruff_python_ast::Number;
use ruff_python_ast::Pattern;
use ruff_python_ast::PatternKeyword;
use ruff_python_ast::StmtMatch;
use ruff_text_size::Ranged;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::ExhaustivenessKind;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::NarrowUseLocation;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::UnpackedPosition;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::expr::Usage;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::NarrowOp;
use crate::binding::narrow::NarrowOps;
use crate::binding::narrow::NarrowSource;
use crate::binding::narrow::NarrowingSubject;
use crate::binding::narrow::expr_to_subjects;
use crate::binding::scope::FlowStyle;
use crate::config::error_kind::ErrorKind;
use crate::error::context::ErrorInfo;
use crate::export::special::SpecialExport;
use crate::types::facet::UnresolvedFacetKind;

impl<'a> BindingsBuilder<'a> {
    // Traverse a pattern and bind all the names; key is the reference for the value that's being matched on
    fn bind_pattern(
        &mut self,
        match_subject: Option<NarrowingSubject>,
        pattern: Pattern,
        subject_idx: Idx<Key>,
    ) -> NarrowOps {
        // In typical code, match patterns are more like static types than normal values, so
        // we ignore match patterns for first-usage tracking.
        let narrowing_usage = &mut Usage::Narrowing(None);
        match pattern {
            Pattern::MatchValue(mut p) => {
                self.ensure_expr(&mut p.value, narrowing_usage);
                if let Some(subject) = match_subject {
                    NarrowOps::from_single_narrow_op_for_subject(
                        subject,
                        AtomicNarrowOp::Eq((*p.value).clone()),
                        p.range(),
                    )
                } else {
                    NarrowOps::new()
                }
            }
            Pattern::MatchSingleton(p) => {
                let value = Ast::pattern_match_singleton_to_expr(&p);
                if let Some(subject) = match_subject {
                    NarrowOps::from_single_narrow_op_for_subject(
                        subject,
                        AtomicNarrowOp::Is(value),
                        p.range(),
                    )
                } else {
                    NarrowOps::new()
                }
            }
            Pattern::MatchAs(p) => {
                // If there's no name for this pattern, refine the variable being matched
                // If there is a new name, refine that instead
                let original_subject = match_subject.clone();
                let alias_name = p.name.as_ref().map(|name| name.id.clone());
                let mut subject = match_subject;
                if let Some(name) = &p.name {
                    self.bind_definition(name, Binding::Forward(subject_idx), FlowStyle::Other);
                    subject = Some(NarrowingSubject::Name(name.id.clone()));
                };
                if let Some(pattern) = p.pattern {
                    let mut narrow_ops = self.bind_pattern(subject, *pattern, subject_idx);
                    if let (Some(alias_name), Some(original_subject)) =
                        (&alias_name, &original_subject)
                        && alias_name != original_subject.name()
                        && let Some((alias_op, range)) = narrow_ops.0.get(alias_name).cloned()
                    {
                        narrow_ops.and_for_subject(
                            original_subject,
                            alias_op.for_subject(original_subject),
                            range,
                        );
                    }
                    narrow_ops
                } else {
                    NarrowOps::new()
                }
            }
            Pattern::MatchSequence(x) => {
                let mut narrow_ops = NarrowOps::new();
                let num_patterns = x.patterns.len();
                let num_non_star_patterns = x
                    .patterns
                    .iter()
                    .filter(|x| !matches!(x, Pattern::MatchStar(_)))
                    .count();
                let mut subject_idx = subject_idx;
                let synthesized_len = Expr::NumberLiteral(ExprNumberLiteral {
                    node_index: AtomicNodeIndex::default(),
                    range: x.range,
                    value: Number::Int(Int::from(num_non_star_patterns as u64)),
                });
                if let Some(subject) = &match_subject {
                    // Narrow the match subject by:
                    // 1. IsSequence - confirms the subject is a sequence type
                    // 2. Length - confirms the sequence has the right length
                    let len_narrow_op = if num_patterns == num_non_star_patterns {
                        AtomicNarrowOp::LenEq(synthesized_len)
                    } else {
                        AtomicNarrowOp::LenGte(synthesized_len)
                    };

                    // Create a combined narrowing: IsSequence AND LenXxx
                    let combined_narrow_op = NarrowOp::And(vec![
                        NarrowOp::Atomic(None, AtomicNarrowOp::IsSequence),
                        NarrowOp::Atomic(None, len_narrow_op.clone()),
                    ]);

                    subject_idx = self.insert_binding(
                        Key::PatternNarrow(x.range()),
                        Binding::Narrow(
                            subject_idx,
                            Box::new(combined_narrow_op.clone()),
                            NarrowUseLocation::Span(x.range()),
                        ),
                    );

                    // Add the combined narrow op to the returned narrow_ops.
                    // We insert directly instead of using and_all twice to avoid
                    // Placeholder issues when starting from an empty NarrowOps.
                    let name = match subject {
                        NarrowingSubject::Name(name) => name.clone(),
                        NarrowingSubject::Facets(name, _) => name.clone(),
                    };
                    narrow_ops.0.insert(name, (combined_narrow_op, x.range));
                }
                let mut seen_star = false;
                for (i, x) in x.patterns.into_iter().enumerate() {
                    // Process each sub-pattern in the sequence pattern
                    match x {
                        Pattern::MatchStar(p) => {
                            if let Some(name) = &p.name {
                                let position = UnpackedPosition::Slice(i, num_patterns - i - 1);
                                self.bind_definition(
                                    name,
                                    Binding::UnpackedValue(None, subject_idx, p.range, position),
                                    FlowStyle::Other,
                                );
                            }
                            seen_star = true;
                        }
                        _ => {
                            let position = if seen_star {
                                UnpackedPosition::ReverseIndex(num_patterns - i)
                            } else {
                                UnpackedPosition::Index(i)
                            };
                            let key_for_subpattern = self.insert_binding(
                                Key::Anon(x.range()),
                                Binding::UnpackedValue(None, subject_idx, x.range(), position),
                            );
                            let subject_for_subpattern = match_subject.clone().and_then(|s| {
                                if !seen_star {
                                    Some(s.with_facet(UnresolvedFacetKind::Index(i)))
                                } else {
                                    None
                                }
                            });
                            narrow_ops.and_all(self.bind_pattern(
                                subject_for_subpattern,
                                x,
                                key_for_subpattern,
                            ));
                        }
                    }
                }
                let expect = if num_patterns != num_non_star_patterns {
                    SizeExpectation::Ge(num_non_star_patterns)
                } else {
                    SizeExpectation::Eq(num_patterns)
                };
                self.insert_binding(
                    KeyExpect::UnpackedLength(x.range),
                    BindingExpect::UnpackedLength(subject_idx, x.range, expect),
                );
                narrow_ops
            }
            Pattern::MatchMapping(x) => {
                let mut narrow_ops = NarrowOps::new();
                let mut subject_idx = subject_idx;
                if let Some(subject) = &match_subject {
                    let narrow_op = AtomicNarrowOp::IsMapping;
                    subject_idx = self.insert_binding(
                        Key::PatternNarrow(x.range()),
                        Binding::Narrow(
                            subject_idx,
                            Box::new(NarrowOp::Atomic(None, narrow_op.clone())),
                            NarrowUseLocation::Span(x.range()),
                        ),
                    );
                    narrow_ops.and_all(NarrowOps::from_single_narrow_op_for_subject(
                        subject.clone(),
                        narrow_op,
                        x.range,
                    ));
                }
                x.keys
                    .into_iter()
                    .zip(x.patterns)
                    .for_each(|(mut match_key_expr, pattern)| {
                        let mut match_key =
                            self.declare_current_idx(Key::Anon(match_key_expr.range()));
                        let key_name = match &match_key_expr {
                            Expr::StringLiteral(ExprStringLiteral { value: key, .. }) => {
                                Some(key.to_string())
                            }
                            _ => {
                                self.ensure_expr(&mut match_key_expr, match_key.usage());
                                None
                            }
                        };
                        let match_key_idx = self.insert_binding_current(
                            match_key,
                            Binding::PatternMatchMapping(match_key_expr, subject_idx),
                        );
                        let subject_at_key = key_name.and_then(|key| {
                            match_subject
                                .clone()
                                .map(|s| s.with_facet(UnresolvedFacetKind::Key(key)))
                        });
                        narrow_ops.and_all(self.bind_pattern(
                            subject_at_key,
                            pattern,
                            match_key_idx,
                        ))
                    });
                if let Some(rest) = x.rest {
                    self.bind_definition(&rest, Binding::Forward(subject_idx), FlowStyle::Other);
                }
                narrow_ops
            }
            Pattern::MatchClass(mut x) => {
                self.ensure_expr(&mut x.cls, narrowing_usage);
                let narrow_op = AtomicNarrowOp::IsInstance((*x.cls).clone(), NarrowSource::Pattern);
                // Redefining subject_idx to apply the class level narrowing,
                // which is used for additional narrowing for attributes below.
                let subject_idx = self.insert_binding(
                    Key::PatternNarrow(x.range()),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(NarrowOp::Atomic(None, narrow_op.clone())),
                        NarrowUseLocation::Span(x.cls.range()),
                    ),
                );

                // Check if this is a single-positional-slot builtin type
                // These types (bool, bytearray, bytes, dict, float, frozenset, int, list, set, str, tuple)
                // bind the entire narrowed value when used with a single positional pattern
                let is_single_slot_builtin = if let Expr::Name(name) = x.cls.as_ref() {
                    SpecialExport::new(&name.id)
                        .map(|se| se.is_single_positional_slot_builtin())
                        .unwrap_or(false)
                } else {
                    false
                };

                // For single-slot builtins with exactly one positional arg, the pattern matches
                // all instances of the type, so we don't need a placeholder
                let is_exhaustive_single_slot = is_single_slot_builtin
                    && x.arguments.patterns.len() == 1
                    && x.arguments.keywords.is_empty();

                let mut narrow_ops = if let Some(ref subject) = match_subject {
                    let mut narrow_for_subject = NarrowOps::from_single_narrow_op_for_subject(
                        subject.clone(),
                        narrow_op,
                        x.cls.range(),
                    );
                    // We're not sure whether the pattern matches all possible instances of a class, and
                    // the placeholder prevents negative narrowing from removing the class in later branches.
                    // However, if there are no arguments, it's just an isinstance check, so we don't need
                    // the placeholder. Similarly, single-slot builtins with one positional arg are exhaustive.
                    if (!x.arguments.patterns.is_empty() || !x.arguments.keywords.is_empty())
                        && !is_exhaustive_single_slot
                    {
                        let placeholder = NarrowOps::from_single_narrow_op_for_subject(
                            subject.clone(),
                            AtomicNarrowOp::Placeholder,
                            x.cls.range(),
                        );
                        narrow_for_subject.and_all(placeholder);
                    }
                    narrow_for_subject
                } else {
                    NarrowOps::new()
                };

                // Handle positional patterns
                if is_exhaustive_single_slot {
                    // For single-positional-slot builtins with exactly one positional pattern,
                    // bind the pattern directly to the narrowed subject (like MatchAs)
                    let pattern = x.arguments.patterns.into_iter().next().unwrap();
                    let inner_narrow_ops =
                        self.bind_pattern(match_subject.clone(), pattern, subject_idx);
                    // Only combine if the inner pattern produced narrow ops.
                    // If it's empty (e.g., a simple MatchAs like `value`), we don't want
                    // and_all to add Placeholders that would invalidate our outer narrow.
                    if !inner_narrow_ops.0.is_empty() {
                        narrow_ops.and_all(inner_narrow_ops);
                    }
                    return narrow_ops;
                }
                // Normal MatchClass handling
                // TODO: narrow class type vars based on pattern arguments
                x.arguments
                    .patterns
                    .into_iter()
                    .enumerate()
                    .for_each(|(idx, pattern)| {
                        let attr_key = self.insert_binding(
                            Key::Anon(pattern.range()),
                            Binding::PatternMatchClassPositional(
                                x.cls.clone(),
                                idx,
                                subject_idx,
                                pattern.range(),
                            ),
                        );
                        // TODO: narrow attributes in positional patterns
                        narrow_ops.and_all(self.bind_pattern(None, pattern.clone(), attr_key))
                    });
                x.arguments.keywords.into_iter().for_each(
                    |PatternKeyword {
                         node_index: _,
                         range: _,
                         attr,
                         pattern,
                     }| {
                        let subject_for_attr = match_subject
                            .clone()
                            .map(|s| s.with_facet(UnresolvedFacetKind::Attribute(attr.id.clone())));
                        let attr_key = self.insert_binding(
                            Key::Anon(attr.range()),
                            Binding::PatternMatchClassKeyword(x.cls.clone(), attr, subject_idx),
                        );
                        narrow_ops.and_all(self.bind_pattern(subject_for_attr, pattern, attr_key))
                    },
                );
                narrow_ops
            }
            Pattern::MatchOr(x) => {
                let mut narrow_ops: Option<NarrowOps> = None;
                self.start_fork(x.range);
                let n_subpatterns = x.patterns.len();
                for (idx, pattern) in x.patterns.into_iter().enumerate() {
                    self.start_branch();
                    if pattern.is_irrefutable() && idx != n_subpatterns - 1 {
                        self.error(
                            pattern.range(),
                            ErrorInfo::Kind(ErrorKind::BadMatch),
                            "Only the last subpattern in MatchOr may be irrefutable".to_owned(),
                        )
                    }
                    let new_narrow_ops =
                        self.bind_pattern(match_subject.clone(), pattern, subject_idx);
                    if let Some(ref mut ops) = narrow_ops {
                        ops.or_all(new_narrow_ops)
                    } else {
                        narrow_ops = Some(new_narrow_ops);
                    }
                    self.finish_branch();
                }
                self.finish_match_or_fork();
                narrow_ops.unwrap_or_default()
            }
            Pattern::MatchStar(p) => {
                if let Some(name) = &p.name {
                    self.bind_definition(name, Binding::Forward(subject_idx), FlowStyle::Other);
                }
                NarrowOps::new()
            }
        }
    }

    pub fn stmt_match(&mut self, mut x: StmtMatch, parent: &NestingContext) {
        let mut subject = self.declare_current_idx(Key::Anon(x.subject.range()));
        self.ensure_expr(&mut x.subject, subject.usage());
        let subject_idx =
            self.insert_binding_current(subject, Binding::Expr(None, *x.subject.clone()));
        let match_narrowing_subject = expr_to_subjects(&x.subject).first().cloned();
        let mut exhaustive = false;
        self.start_fork(x.range);
        // Type narrowing operations that are carried over from one case to the next. For example, in:
        //   match x:
        //     case None:
        //       pass
        //     case _:
        //       pass
        // x is bound to Narrow(x, Eq(None)) in the first case, and the negation, Narrow(x, NotEq(None)),
        // is carried over to the fallback case.
        let mut negated_prev_ops = NarrowOps::new();
        for case in x.cases {
            let MatchCase {
                pattern,
                guard,
                body,
                range: case_range,
                ..
            } = case;
            self.start_branch();
            let case_is_irrefutable = pattern.is_wildcard() || pattern.is_irrefutable();
            if case_is_irrefutable {
                exhaustive = true;
            }
            self.bind_narrow_ops(
                &negated_prev_ops,
                NarrowUseLocation::Start(case_range),
                &Usage::Narrowing(None),
            );
            // Create a narrowed subject_idx for this case by applying negated_prev_ops.
            // This ensures that patterns like MatchMapping use the narrowed type
            // (e.g., after matching `None`, the type excludes `None`).
            let case_subject_idx = if let Some(ref narrowing_subject) = match_narrowing_subject
                && let Some((narrow_op, op_range)) =
                    negated_prev_ops.0.get(narrowing_subject.name())
            {
                self.insert_binding(
                    Key::PatternNarrow(case_range),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(narrow_op.clone()),
                        NarrowUseLocation::Start(*op_range),
                    ),
                )
            } else {
                subject_idx
            };
            let mut new_narrow_ops =
                self.bind_pattern(match_narrowing_subject.clone(), pattern, case_subject_idx);
            self.bind_narrow_ops(
                &new_narrow_ops,
                NarrowUseLocation::Span(case_range),
                &Usage::Narrowing(None),
            );
            if let Some(mut guard) = guard {
                self.ensure_expr(&mut guard, &mut Usage::Narrowing(None));
                let guard_narrow_ops = NarrowOps::from_expr(self, Some(guard.as_ref()));
                self.bind_narrow_ops(
                    &guard_narrow_ops,
                    NarrowUseLocation::Span(guard.range()),
                    &Usage::Narrowing(None),
                );
                self.insert_binding(Key::Anon(guard.range()), Binding::Expr(None, *guard));
                new_narrow_ops.and_all(guard_narrow_ops)
            }
            negated_prev_ops.and_all(new_narrow_ops.negate());
            self.stmts(body, parent);
            self.finish_branch();
        }
        if exhaustive {
            self.finish_exhaustive_fork();
        } else {
            self.finish_non_exhaustive_fork(&negated_prev_ops);
            // Compute exhaustiveness info if we can determine the narrowing subject
            // and have accumulated narrow ops for it.
            let exhaustiveness_info = match_narrowing_subject.and_then(|narrowing_subject| {
                negated_prev_ops
                    .0
                    .get(narrowing_subject.name())
                    .map(|(op, range)| (narrowing_subject, (Box::new(op.clone()), *range)))
            });
            // Create BindingExpect only if we have the info (for exhaustiveness warnings)
            if let Some((ref narrowing_subject, ref narrow_ops_for_fall_through)) =
                exhaustiveness_info
            {
                self.insert_binding(
                    KeyExpect::MatchExhaustiveness(x.range),
                    BindingExpect::MatchExhaustiveness {
                        subject_idx,
                        narrowing_subject: narrowing_subject.clone(),
                        narrow_ops_for_fall_through: narrow_ops_for_fall_through.clone(),
                        subject_range: x.subject.range(),
                    },
                );
            }
            // Always create Key::Exhaustive binding for return analysis.
            // When exhaustiveness_info is None, the solver will conservatively
            // assume the match is not exhaustive (resolves to Type::None).
            self.insert_binding(
                Key::Exhaustive(ExhaustivenessKind::Match, x.range),
                Binding::Exhaustive {
                    kind: ExhaustivenessKind::Match,
                    subject_idx,
                    subject_range: x.subject.range(),
                    exhaustiveness_info,
                },
            );
        }
    }
}
