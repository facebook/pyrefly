/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprStringLiteral;
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
use crate::binding::narrow::NarrowOps;
use crate::binding::narrow::NarrowingSubject;
use crate::binding::narrow::expr_to_subjects;
use crate::binding::scope::FlowStyle;
use crate::error::kind::ErrorKind;
use crate::graph::index::Idx;
use crate::types::facet::FacetKind;

impl<'a> BindingsBuilder<'a> {
    // Traverse a pattern and bind all the names; key is the reference for the value that's being matched on
    fn bind_pattern(
        &mut self,
        match_subject: Option<NarrowingSubject>,
        pattern: Pattern,
        key: Idx<Key>,
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
                    self.bind_definition(name, Binding::Forward(key), FlowStyle::Other);
                    subject = Some(NarrowingSubject::Name(name.id.clone()));
                };
                if let Some(pattern) = p.pattern {
                    self.bind_pattern(subject, *pattern, key)
                } else {
                    NarrowOps::new()
                }
            }
            Pattern::MatchSequence(x) => {
                let mut narrow_ops = NarrowOps::new();
                let num_patterns = x.patterns.len();
                let mut unbounded = false;
                for (idx, x) in x.patterns.into_iter().enumerate() {
                    match x {
                        Pattern::MatchStar(p) => {
                            if let Some(name) = &p.name {
                                let position = UnpackedPosition::Slice(idx, num_patterns - idx - 1);
                                self.bind_definition(
                                    name,
                                    Binding::UnpackedValue(None, key, p.range, position),
                                    FlowStyle::Other,
                                );
                            }
                            unbounded = true;
                        }
                        _ => {
                            let position = if unbounded {
                                UnpackedPosition::ReverseIndex(num_patterns - idx)
                            } else {
                                UnpackedPosition::Index(idx)
                            };
                            let key = self.insert_binding(
                                Key::Anon(x.range()),
                                Binding::UnpackedValue(None, key, x.range(), position),
                            );
                            narrow_ops.and_all(self.bind_pattern(None, x, key));
                        }
                    }
                }
                let expect = if unbounded {
                    SizeExpectation::Ge(num_patterns - 1)
                } else {
                    SizeExpectation::Eq(num_patterns)
                };
                self.insert_binding(
                    KeyExpect(x.range),
                    BindingExpect::UnpackedLength(key, x.range, expect),
                );
                narrow_ops
            }
            Pattern::MatchMapping(x) => {
                let mut narrow_ops = NarrowOps::new();
                x.keys
                    .into_iter()
                    .zip(x.patterns)
                    .for_each(|(mut key_expr, pattern)| {
                        let mut key_user = self.declare_user(Key::Anon(key_expr.range()));
                        let key_name = match &key_expr {
                            Expr::StringLiteral(ExprStringLiteral { value: key, .. }) => {
                                Some(key.to_string())
                            }
                            _ => {
                                self.ensure_expr(&mut key_expr, key_user.usage());
                                None
                            }
                        };
                        let subject_for_key = key_name.and_then(|key| {
                            match_subject
                                .clone()
                                .map(|s| s.with_facet(FacetKind::Key(key)))
                        });
                        let binding_for_key = self.insert_binding_user(
                            key_user,
                            Binding::PatternMatchMapping(key_expr, key),
                        );
                        narrow_ops.and_all(self.bind_pattern(
                            subject_for_key,
                            pattern,
                            binding_for_key,
                        ))
                    });
                if let Some(rest) = x.rest {
                    self.bind_definition(&rest, Binding::Forward(key), FlowStyle::Other);
                }
                narrow_ops
            }
            Pattern::MatchClass(mut x) => {
                self.ensure_expr(&mut x.cls, narrowing_usage);
                let mut narrow_ops = if let Some(subject) = match_subject {
                    NarrowOps::from_single_narrow_op_for_subject(
                        subject,
                        AtomicNarrowOp::IsInstance((*x.cls).clone()),
                        x.cls.range(),
                    )
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
                                key,
                                pattern.range(),
                            ),
                        );
                        narrow_ops.and_all(self.bind_pattern(None, pattern.clone(), attr_key))
                    });
                x.arguments.keywords.into_iter().for_each(
                    |PatternKeyword {
                         range: _,
                         attr,
                         pattern,
                     }| {
                        let attr_key = self.insert_binding(
                            Key::Anon(attr.range()),
                            Binding::PatternMatchClassKeyword(x.cls.clone(), attr, key),
                        );
                        narrow_ops.and_all(self.bind_pattern(None, pattern, attr_key))
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
                            ErrorKind::MatchError,
                            None,
                            "Only the last subpattern in MatchOr may be irrefutable".to_owned(),
                        )
                    }
                    let mut base = self.scopes.clone_current_flow();
                    let new_narrow_ops = self.bind_pattern(match_subject.clone(), pattern, key);
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
        let mut subject_user = self.declare_user(Key::Anon(x.subject.range()));
        self.ensure_expr(&mut x.subject, subject_user.usage());
        let match_subject = *x.subject.clone();
        let key = self.insert_binding_user(subject_user, Binding::Expr(None, *x.subject.clone()));
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
            let match_narrowing_subject = expr_to_subjects(&match_subject).first().cloned();
            let new_narrow_ops = self.bind_pattern(match_narrowing_subject, case.pattern, key);
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
