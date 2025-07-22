/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::Number;
use ruff_python_ast::Pattern;
use ruff_python_ast::PatternKeyword;
use ruff_python_ast::StmtMatch;
use ruff_text_size::Ranged;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::UnpackedPosition;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::expr::Usage;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::NarrowOp;
use crate::binding::narrow::NarrowOps;
use crate::binding::narrow::NarrowingSubject;
use crate::binding::narrow::expr_to_subjects;
use crate::binding::scope::FlowStyle;
use crate::config::error_kind::ErrorKind;
use crate::error::context::ErrorInfo;
use crate::graph::index::Idx;
use crate::types::facet::FacetKind;

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
        let narrowing_usage = &mut Usage::Narrowing;
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
                let mut subject = match_subject;
                if let Some(name) = &p.name {
                    self.bind_definition(name, Binding::Forward(subject_idx), FlowStyle::Other);
                    subject = Some(NarrowingSubject::Name(name.id.clone()));
                };
                if let Some(pattern) = p.pattern {
                    self.bind_pattern(subject, *pattern, subject_idx)
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
                    node_index: AtomicNodeIndex::dummy(),
                    range: x.range,
                    value: Number::Int(Int::from(num_non_star_patterns as u64)),
                });
                if let Some(subject) = &match_subject {
                    // Narrow the match subject by length
                    let narrow_op = if num_patterns == num_non_star_patterns {
                        AtomicNarrowOp::LenEq(synthesized_len)
                    } else {
                        AtomicNarrowOp::LenGte(synthesized_len)
                    };
                    subject_idx = self.insert_binding(
                        Key::PatternNarrow(x.range()),
                        Binding::Narrow(
                            subject_idx,
                            Box::new(NarrowOp::Atomic(None, narrow_op.clone())),
                            x.range(),
                        ),
                    );
                    narrow_ops.and_all(NarrowOps::from_single_narrow_op_for_subject(
                        subject.clone(),
                        narrow_op,
                        x.range,
                    ));
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
                                    Some(s.with_facet(FacetKind::Index(i)))
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
                    KeyExpect(x.range),
                    BindingExpect::UnpackedLength(subject_idx, x.range, expect),
                );
                narrow_ops
            }
            Pattern::MatchMapping(x) => {
                let mut narrow_ops = NarrowOps::new();
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
                                .map(|s| s.with_facet(FacetKind::Key(key)))
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
                let narrow_op = AtomicNarrowOp::IsInstance((*x.cls).clone());
                // Redefining subject_idx to apply the class level narrowing,
                // which is used for additional narrowing for attributes below.
                let subject_idx = self.insert_binding(
                    Key::PatternNarrow(x.range()),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(NarrowOp::Atomic(None, narrow_op.clone())),
                        x.cls.range(),
                    ),
                );
                let mut narrow_ops = if let Some(ref subject) = match_subject {
                    let mut narrow_for_subject = NarrowOps::from_single_narrow_op_for_subject(
                        subject.clone(),
                        narrow_op,
                        x.cls.range(),
                    );
                    // We're not sure whether the pattern matches all possible instances of a class, and
                    // the placeholder prevents negative narrowing from removing the class in later branches.
                    let placeholder = NarrowOps::from_single_narrow_op_for_subject(
                        subject.clone(),
                        AtomicNarrowOp::Placeholder,
                        x.cls.range(),
                    );
                    narrow_for_subject.and_all(placeholder);
                    narrow_for_subject
                } else {
                    NarrowOps::new()
                };
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
                            .map(|s| s.with_facet(FacetKind::Attribute(attr.id.clone())));
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
                let range = x.range;
                let mut branches = Vec::new();
                let n_subpatterns = x.patterns.len();
                for (idx, pattern) in x.patterns.into_iter().enumerate() {
                    if pattern.is_irrefutable() && idx != n_subpatterns - 1 {
                        self.error(
                            pattern.range(),
                            ErrorInfo::Kind(ErrorKind::MatchError),
                            "Only the last subpattern in MatchOr may be irrefutable".to_owned(),
                        )
                    }
                    let mut base = self.scopes.clone_current_flow();
                    let new_narrow_ops =
                        self.bind_pattern(match_subject.clone(), pattern, subject_idx);
                    if let Some(ref mut ops) = narrow_ops {
                        ops.or_all(new_narrow_ops)
                    } else {
                        narrow_ops = Some(new_narrow_ops);
                    }
                    self.scopes.swap_current_flow_with(&mut base);
                    branches.push(base);
                }
                self.set_current_flow_to_merged_branches(branches, range);
                narrow_ops.unwrap_or_default()
            }
            Pattern::MatchStar(_) => NarrowOps::new(),
        }
    }

    pub fn stmt_match(&mut self, mut x: StmtMatch) {
        let mut subject = self.declare_current_idx(Key::Anon(x.subject.range()));
        self.ensure_expr(&mut x.subject, subject.usage());
        let subject_idx =
            self.insert_binding_current(subject, Binding::Expr(None, *x.subject.clone()));
        let match_narrowing_subject = expr_to_subjects(&x.subject).first().cloned();
        let mut exhaustive = false;
        let range = x.range;
        let mut branches = Vec::new();
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
            let mut base = self.scopes.clone_current_flow();
            if case.pattern.is_wildcard() || case.pattern.is_irrefutable() {
                exhaustive = true;
            }
            let new_narrow_ops =
                self.bind_pattern(match_narrowing_subject.clone(), case.pattern, subject_idx);
            self.bind_narrow_ops(&negated_prev_ops, case.range);
            self.bind_narrow_ops(&new_narrow_ops, case.range);
            negated_prev_ops.and_all(new_narrow_ops.negate());
            if let Some(mut guard) = case.guard {
                self.ensure_expr(&mut guard, &mut Usage::Narrowing);
                let narrow_ops = NarrowOps::from_expr(self, Some(guard.as_ref()));
                self.bind_narrow_ops(&narrow_ops, case.range);
                self.insert_binding(Key::Anon(guard.range()), Binding::Expr(None, *guard));
            }
            self.stmts(case.body);
            self.scopes.swap_current_flow_with(&mut base);
            branches.push(base);
        }
        // If the match branches cover all possibilities, then the flow after the match
        // is just the merged branch flows.
        //
        // Otherwise, we need to merge the branches with the original `base` flow (which is current).
        if exhaustive {
            self.set_current_flow_to_merged_branches(branches, range);
        } else {
            self.merge_branches_into_current(branches, range);
        }
    }
}
