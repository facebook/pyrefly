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
use ruff_text_size::TextRange;
use vec1::Vec1;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::ExhaustiveBinding;
use crate::binding::binding::ExhaustivenessKind;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::NarrowUseLocation;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::UnpackedPosition;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::expr::Usage;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::FacetOrigin;
use crate::binding::narrow::FacetSubject;
use crate::binding::narrow::NarrowOp;
use crate::binding::narrow::NarrowOps;
use crate::binding::narrow::NarrowSource;
use crate::binding::narrow::NarrowingSubject;
use crate::binding::narrow::expr_to_subjects;
use crate::binding::scope::FlowStyle;
use crate::config::error_kind::ErrorKind;
use crate::export::special::SpecialExport;
use crate::types::facet::UnresolvedFacetChain;
use crate::types::facet::UnresolvedFacetKind;

#[derive(Clone, Debug)]
enum MatchSubject {
    /// No narrowing subject available.
    None,
    /// A single match subject (e.g., `match x:`).
    Single(NarrowingSubject),
    /// A local-only subject for matching non-name expressions (e.g., `match await f():`).
    /// Python evaluates the subject once before matching, so we need a stable internal
    /// subject for branch narrowing while diagnostics still point at the source expression.
    Synthetic { display_subject_range: TextRange },
    /// Per-element subjects from a tuple match (e.g., `match x, y:`).
    Tuple(Vec<Option<NarrowingSubject>>),
}

impl MatchSubject {
    /// Extract a single narrowing subject, if available.
    fn as_single(&self) -> Option<&NarrowingSubject> {
        match self {
            MatchSubject::Single(s) => Some(s),
            _ => Option::None,
        }
    }

    fn is_synthetic(&self) -> bool {
        matches!(self, MatchSubject::Synthetic { .. })
    }

    fn subject_narrow_op(&self, op: NarrowOp, range: TextRange) -> PatternNarrowOps {
        if let Some(subject) = self.as_single() {
            let mut scope = NarrowOps::new();
            scope.and_for_subject(subject, op.for_subject(subject), range);
            PatternNarrowOps::from_scope(scope)
        } else if self.is_synthetic() {
            PatternNarrowOps::from_subject(op, range)
        } else {
            PatternNarrowOps::new()
        }
    }

    fn subject_range(&self, fallback: TextRange) -> TextRange {
        match self {
            MatchSubject::Synthetic {
                display_subject_range,
                ..
            } => *display_subject_range,
            _ => fallback,
        }
    }
}

#[derive(Debug, Default)]
struct PatternNarrowOps {
    scope: NarrowOps,
    subject: Option<(NarrowOp, TextRange)>,
    /// Narrows that apply positively to the case body but are deliberately excluded
    /// from the subject's negation. We use this to exclude any sub-patterns within
    /// a class pattern to avoid poisoning exhaustiveness checks.
    ///
    /// Example: for a class with 2 string fields, `Cls(str(), str())` is an exhaustive pattern
    /// However, the generated narrows include `isinstance(cls.x, str) & isinstance(cls.y, str)`
    /// which negates to `!isinstance(cls.x, str) or !isinstance(cls.y, str)`, causing us to
    /// infer an incorrect remainder type.
    ///
    /// Instead, we use the `ClassCoverageGate` mechanism to consider all sub-patterns of a class
    /// together when determining exhaustiveness.
    body_only: NarrowOps,
}

impl PatternNarrowOps {
    fn new() -> Self {
        Self::default()
    }

    fn from_scope(scope: NarrowOps) -> Self {
        Self {
            scope,
            subject: None,
            body_only: NarrowOps::new(),
        }
    }

    fn from_subject(op: NarrowOp, range: TextRange) -> Self {
        Self {
            scope: NarrowOps::new(),
            subject: Some((op, range)),
            body_only: NarrowOps::new(),
        }
    }

    fn and_subject(&mut self, other: Option<(NarrowOp, TextRange)>) {
        match (&mut self.subject, other) {
            (Some((op, range)), Some((other_op, other_range))) => {
                *op = NarrowOp::And(vec![op.clone(), other_op]);
                // Cover both operands so the merged range is independent of operand order.
                *range = range.cover(other_range);
            }
            (None, Some(other)) => self.subject = Some(other),
            _ => {}
        }
    }

    fn and_all(&mut self, other: Self) {
        self.scope.and_all(other.scope);
        self.body_only.and_all(other.body_only);
        self.and_subject(other.subject);
    }

    fn or_all(&mut self, other: Self) {
        self.scope.or_all(other.scope);
        self.body_only.or_all(other.body_only);
        self.subject = match (self.subject.take(), other.subject) {
            (Some((op, range)), Some((other_op, other_range))) => {
                // Cover both operands so the merged range is independent of subpattern order.
                Some((NarrowOp::Or(vec![op, other_op]), range.cover(other_range)))
            }
            _ => None,
        };
    }

    fn negate(&self) -> Self {
        Self {
            scope: self.scope.negate(),
            subject: self
                .subject
                .as_ref()
                .map(|(op, range)| (op.negate(), *range)),
            // Body-only narrows are intentionally dropped from the negation.
            body_only: NarrowOps::new(),
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    /// The atomic narrow op a leaf sub-pattern imposes on its element, if any.
    ///
    /// Used to build a subject narrowed by every sibling element constraint, so non-narrowing
    /// element captures in a sequence pattern see the narrowed parent type after applying narrowing
    /// from other elements of the sequence.
    fn sequence_element_atomic_op(pattern: &Pattern) -> Option<AtomicNarrowOp> {
        match pattern {
            Pattern::MatchClass(x) => Some(AtomicNarrowOp::IsInstance(
                (*x.cls).clone(),
                NarrowSource::Pattern,
            )),
            Pattern::MatchValue(p) => Some(AtomicNarrowOp::Eq((*p.value).clone())),
            Pattern::MatchSingleton(p) => {
                Some(AtomicNarrowOp::Is(Ast::pattern_match_singleton_to_expr(p)))
            }
            Pattern::MatchAs(p) => p
                .pattern
                .as_deref()
                .and_then(Self::sequence_element_atomic_op),
            _ => None,
        }
    }

    /// Traverse a pattern and bind all the names; key is the reference for
    /// the value that's being matched on.
    fn bind_pattern(
        &mut self,
        match_subject: MatchSubject,
        pattern: Pattern,
        subject_idx: Idx<Key>,
    ) -> PatternNarrowOps {
        // In typical code, match patterns are more like static types than normal values, so
        // we ignore match patterns for first-usage tracking.
        let narrowing_usage = &mut Usage::NonPinningValue(None);
        match pattern {
            Pattern::MatchValue(mut p) => {
                self.ensure_expr(&mut p.value, narrowing_usage);
                match_subject.subject_narrow_op(
                    NarrowOp::Atomic(None, AtomicNarrowOp::Eq((*p.value).clone())),
                    p.range(),
                )
            }
            Pattern::MatchSingleton(p) => {
                let value = Ast::pattern_match_singleton_to_expr(&p);
                match_subject
                    .subject_narrow_op(NarrowOp::Atomic(None, AtomicNarrowOp::Is(value)), p.range())
            }
            Pattern::MatchAs(p) => {
                // If there's no name for this pattern, refine the variable being matched
                // If there is a new name, refine that instead
                let original_subject = match_subject.clone();
                let alias_name = p
                    .name
                    .as_ref()
                    .filter(|name| !Ast::is_synthesized_empty_identifier(name))
                    .map(|name| name.id.clone());
                let mut subject = match_subject;
                if let Some(name) = &p.name
                    && !Ast::is_synthesized_empty_identifier(name)
                {
                    self.bind_definition(name, Binding::Forward(subject_idx), FlowStyle::Other);
                    subject = MatchSubject::Single(NarrowingSubject::Name(name.id.clone()));
                };
                if let Some(pattern) = p.pattern {
                    let mut narrow_ops = self.bind_pattern(subject, *pattern, subject_idx);
                    if let Some(alias_name) = &alias_name
                        && let Some((alias_op, range)) = narrow_ops.scope.0.get(alias_name).cloned()
                    {
                        if let Some(original_subject) = original_subject.as_single()
                            && alias_name != original_subject.name()
                        {
                            narrow_ops.scope.and_for_subject(
                                original_subject,
                                alias_op.for_subject(original_subject),
                                range,
                            );
                        } else if original_subject.is_synthetic() {
                            narrow_ops.and_subject(Some((alias_op, range)));
                        }
                    }
                    narrow_ops
                } else {
                    PatternNarrowOps::new()
                }
            }
            Pattern::MatchSequence(x) => {
                let mut narrow_ops = PatternNarrowOps::new();
                let num_patterns = x.patterns.len();
                let num_non_star_patterns = x
                    .patterns
                    .iter()
                    .filter(|x| !matches!(x, Pattern::MatchStar(_)))
                    .count();
                // If every sub-pattern is irrefutable -- i.e., patterns that always match
                // like wildcards (`_`), bare names (e.g., `x`), or `*rest` -- the structural
                // `IsSequence + LenEq/LenGte` narrow on the subject fully captures what the
                // pattern proves. Spurious Placeholders added by `and_all` for empty
                // sub-pattern narrow ops would otherwise block negative narrowing
                // (see equivalent fix in MatchClass below).
                let all_subpatterns_irrefutable = x
                    .patterns
                    .iter()
                    .all(|p| p.is_irrefutable() || p.is_wildcard());
                let mut subject_idx = subject_idx;
                let synthesized_len = Expr::NumberLiteral(ExprNumberLiteral {
                    node_index: AtomicNodeIndex::default(),
                    range: x.range,
                    value: Number::Int(Int::from(num_non_star_patterns as u64)),
                });

                // Narrow the match subject by:
                // 1. IsSequence - confirms the subject is a sequence type
                // 2. Length - confirms the sequence has the right length
                let len_narrow_op = if num_patterns == num_non_star_patterns {
                    AtomicNarrowOp::LenEq(synthesized_len)
                } else {
                    AtomicNarrowOp::LenGte(synthesized_len)
                };
                let combined_narrow_op = NarrowOp::And(vec![
                    NarrowOp::Atomic(None, AtomicNarrowOp::IsSequence),
                    NarrowOp::Atomic(None, len_narrow_op.clone()),
                ]);
                let subject_narrow_op = if num_patterns == num_non_star_patterns {
                    let mut ops = vec![
                        NarrowOp::Atomic(None, AtomicNarrowOp::IsSequence),
                        NarrowOp::Atomic(None, len_narrow_op.clone()),
                    ];
                    for (i, p) in x.patterns.iter().enumerate() {
                        if let Some(atomic) = Self::sequence_element_atomic_op(p) {
                            ops.push(NarrowOp::Atomic(
                                Some(FacetSubject {
                                    chain: UnresolvedFacetChain::new(Vec1::new(
                                        UnresolvedFacetKind::Index(i as i64),
                                    )),
                                    origin: FacetOrigin::Direct,
                                    allow_never_collapse: false,
                                }),
                                atomic,
                            ));
                        }
                    }
                    NarrowOp::And(ops)
                } else {
                    combined_narrow_op.clone()
                };
                subject_idx = self.insert_binding(
                    Key::PatternNarrow(x.range()),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(subject_narrow_op),
                        NarrowUseLocation::Span(x.range()),
                    ),
                );
                if let Some(subject) = match_subject.as_single() {
                    // Add the combined narrow op to the returned narrow_ops for
                    // scope-level narrowing propagation across cases.
                    let (name, facet) = match subject {
                        NarrowingSubject::Name(name) => (name.clone(), None),
                        NarrowingSubject::Facets(name, facets) => {
                            (name.clone(), Some(facets.clone()))
                        }
                    };
                    let scope_narrow_op = NarrowOp::And(vec![
                        NarrowOp::Atomic(facet.clone(), AtomicNarrowOp::IsSequence),
                        NarrowOp::Atomic(facet, len_narrow_op.clone()),
                    ]);
                    narrow_ops.scope.0.insert(name, (scope_narrow_op, x.range));
                } else if match_subject.is_synthetic() {
                    let subject_op = if all_subpatterns_irrefutable {
                        combined_narrow_op
                    } else {
                        NarrowOp::And(vec![
                            combined_narrow_op,
                            NarrowOp::Atomic(None, AtomicNarrowOp::Placeholder),
                        ])
                    };
                    narrow_ops.and_subject(Some((subject_op, x.range)));
                }
                // Without a star sub-pattern the sequence length is pinned exactly;
                // with one it is only a lower bound.
                let has_star = num_patterns != num_non_star_patterns;
                let mut seen_star = false;
                for (i, x) in x.patterns.into_iter().enumerate() {
                    // Process each sub-pattern in the sequence pattern
                    match x {
                        Pattern::MatchStar(p) => {
                            if let Some(name) = &p.name
                                && !Ast::is_synthesized_empty_identifier(name)
                            {
                                let position = UnpackedPosition::Slice(i, num_patterns - i - 1);
                                self.bind_definition(
                                    name,
                                    Binding::UnpackedValue(
                                        None,
                                        subject_idx,
                                        p.range,
                                        position,
                                        None,
                                    ),
                                    FlowStyle::Other,
                                );
                            }
                            seen_star = true;
                        }
                        _ => {
                            let position = if !has_star {
                                UnpackedPosition::ExactIndex(i, num_non_star_patterns)
                            } else if seen_star {
                                UnpackedPosition::ReverseIndex(
                                    num_patterns - i,
                                    num_non_star_patterns,
                                )
                            } else {
                                UnpackedPosition::Index(i, num_non_star_patterns)
                            };
                            let key_for_subpattern = self.insert_binding(
                                Key::Anon(x.range()),
                                Binding::UnpackedValue(
                                    None,
                                    subject_idx,
                                    x.range(),
                                    position,
                                    None,
                                ),
                            );
                            let subject_for_subpattern = match &match_subject {
                                // For tuple subjects, map pattern index to the
                                // correct tuple element. After a star, index from
                                // the end since the star absorbs variable elements.
                                MatchSubject::Tuple(subjects) => {
                                    let tuple_idx = if seen_star {
                                        match subjects.len().checked_sub(num_patterns - i) {
                                            Some(idx) => idx,
                                            Option::None => {
                                                // More patterns than tuple elements, skip narrowing
                                                narrow_ops.and_all(self.bind_pattern(
                                                    MatchSubject::None,
                                                    x,
                                                    key_for_subpattern,
                                                ));
                                                continue;
                                            }
                                        }
                                    } else {
                                        i
                                    };
                                    match subjects.get(tuple_idx) {
                                        Some(Some(s)) => MatchSubject::Single(s.clone()),
                                        _ => MatchSubject::None,
                                    }
                                }
                                MatchSubject::Single(subject) if !seen_star => {
                                    MatchSubject::Single(
                                        subject
                                            .clone()
                                            .with_facet(UnresolvedFacetKind::Index(i as i64)),
                                    )
                                }
                                _ => MatchSubject::None,
                            };
                            narrow_ops.and_all(self.bind_pattern(
                                subject_for_subpattern,
                                x,
                                key_for_subpattern,
                            ));
                        }
                    }
                }
                let expect = if has_star {
                    SizeExpectation::Ge(num_non_star_patterns)
                } else {
                    SizeExpectation::Eq(num_patterns)
                };
                self.insert_binding(
                    KeyExpect::UnpackedLength(x.range),
                    BindingExpect::UnpackedLength(subject_idx, x.range, expect),
                );
                if all_subpatterns_irrefutable
                    && let Some(subject) = match_subject.as_single()
                    && let Some((op, _)) = narrow_ops.scope.0.get_mut(subject.name())
                {
                    op.strip_placeholders();
                }
                narrow_ops
            }
            Pattern::MatchMapping(x) => {
                let mut narrow_ops = PatternNarrowOps::new();
                let mut subject_idx = subject_idx;
                let narrow_op = AtomicNarrowOp::IsMapping;
                subject_idx = self.insert_binding(
                    Key::PatternNarrow(x.range()),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(NarrowOp::Atomic(None, narrow_op.clone())),
                        NarrowUseLocation::Span(x.range()),
                    ),
                );
                let subject_op = if match_subject.is_synthetic() && !x.keys.is_empty() {
                    NarrowOp::And(vec![
                        NarrowOp::Atomic(None, narrow_op),
                        NarrowOp::Atomic(None, AtomicNarrowOp::Placeholder),
                    ])
                } else {
                    NarrowOp::Atomic(None, narrow_op)
                };
                narrow_ops.and_all(match_subject.subject_narrow_op(subject_op, x.range));
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
                            Binding::PatternMatchMapping(Box::new(match_key_expr), subject_idx),
                        );
                        let subject_at_key = if let (Some(key), Some(subject)) =
                            (key_name, match_subject.as_single())
                        {
                            MatchSubject::Single(
                                subject.clone().with_facet(UnresolvedFacetKind::Key(key)),
                            )
                        } else {
                            MatchSubject::None
                        };
                        narrow_ops.and_all(self.bind_pattern(
                            subject_at_key,
                            pattern,
                            match_key_idx,
                        ))
                    });
                if let Some(rest) = x.rest
                    && !Ast::is_synthesized_empty_identifier(&rest)
                {
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

                // Check whether all sub-patterns are irrefutable (e.g. wildcards like `_`).
                // If so, the class pattern matches all instances, so we don't need a
                // Placeholder that would block negative narrowing.
                let all_args_irrefutable = x
                    .arguments
                    .patterns
                    .iter()
                    .all(|p| p.is_irrefutable() || p.is_wildcard())
                    && x.arguments
                        .keywords
                        .iter()
                        .all(|kw| kw.pattern.is_irrefutable() || kw.pattern.is_wildcard());
                // Positional-only class patterns (no keywords) with at least one refutable
                // sub-pattern use a solve-time coverage gate (below) instead of a
                // `Placeholder`, so the class is narrowed away when *every* positional slot's
                // sub-pattern exhausts its slot.
                let positional_coverage = !x.arguments.patterns.is_empty()
                    && x.arguments.keywords.is_empty()
                    && !all_args_irrefutable
                    && !is_exhaustive_single_slot
                    && match_subject.as_single().is_some();

                let mut narrow_ops = match_subject
                    .subject_narrow_op(NarrowOp::Atomic(None, narrow_op.clone()), x.cls.range());
                if (!x.arguments.patterns.is_empty() || !x.arguments.keywords.is_empty())
                    && !is_exhaustive_single_slot
                    && !all_args_irrefutable
                    && !positional_coverage
                {
                    narrow_ops.and_all(match_subject.subject_narrow_op(
                        NarrowOp::Atomic(None, AtomicNarrowOp::Placeholder),
                        x.cls.range(),
                    ));
                }

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
                    if !inner_narrow_ops.scope.0.is_empty() || inner_narrow_ops.subject.is_some() {
                        narrow_ops.and_all(inner_narrow_ops);
                    }
                    return narrow_ops;
                }
                // Normal MatchClass handling
                // TODO: narrow class type vars based on pattern arguments
                let mut coverage_keys: Vec<Idx<Key>> = Vec::new();
                for (idx, pattern) in x.arguments.patterns.into_iter().enumerate() {
                    let attr_key = self.insert_binding(
                        Key::Anon(pattern.range()),
                        Binding::PatternMatchClassPositional(
                            x.cls.clone(),
                            idx,
                            subject_idx,
                            pattern.range(),
                        ),
                    );
                    // Narrow the matched attribute (`__match_args__[idx]`) as a facet
                    // of the subject, so sub-pattern narrowing flows to the parent.
                    let subject_for_slot = if let Some(subject) = match_subject.as_single() {
                        MatchSubject::Single(subject.clone().with_facet(
                            UnresolvedFacetKind::MatchArg {
                                class: x.cls.clone(),
                                index: idx,
                            },
                        ))
                    } else {
                        MatchSubject::None
                    };
                    let sub_ops = self.bind_pattern(subject_for_slot, pattern.clone(), attr_key);
                    if positional_coverage {
                        if !pattern.is_irrefutable() {
                            // Build this slot's coverage probe: narrow the subject to this class
                            // first (so other union members don't pollute the slot), then require
                            // the negated sub-pattern residual to be `Never`. Irrefutable slots are
                            // already exhausted, so only refutable slots need solve-time probes.
                            let mut coverage_scope = match_subject
                                .subject_narrow_op(
                                    NarrowOp::Atomic(None, narrow_op.clone()),
                                    x.cls.range(),
                                )
                                .scope;
                            coverage_scope.and_all(sub_ops.scope.negate());
                            let narrow_entries = self.build_narrow_entries(&coverage_scope);
                            coverage_keys.push(self.insert_binding(
                                Key::Exhaustive(
                                    ExhaustivenessKind::ClassPatternCoverage,
                                    pattern.range(),
                                ),
                                Binding::Exhaustive(Box::new(ExhaustiveBinding {
                                    kind: ExhaustivenessKind::ClassPatternCoverage,
                                    narrow_entries,
                                })),
                            ));
                        }
                        narrow_ops.body_only.and_all(sub_ops.scope);
                        narrow_ops.body_only.and_all(sub_ops.body_only);
                    } else {
                        narrow_ops.and_all(sub_ops);
                    }
                }
                if positional_coverage {
                    // The class is subtracted from later cases only when every refutable slot
                    // probe resolves to `Never` (checked by `ClassCoverageGateNeg` at solve time).
                    narrow_ops.and_all(match_subject.subject_narrow_op(
                        NarrowOp::Atomic(
                            None,
                            AtomicNarrowOp::ClassCoverageGate(coverage_keys.into()),
                        ),
                        x.cls.range(),
                    ));
                }
                x.arguments.keywords.into_iter().for_each(
                    |PatternKeyword {
                         node_index: _,
                         range: _,
                         attr,
                         pattern,
                     }| {
                        let subject_for_attr = if let Some(subject) = match_subject.as_single() {
                            MatchSubject::Single(
                                subject
                                    .clone()
                                    .with_facet(UnresolvedFacetKind::Attribute(attr.id.clone())),
                            )
                        } else {
                            MatchSubject::None
                        };
                        let attr_key = self.insert_binding(
                            Key::Anon(attr.range()),
                            Binding::PatternMatchClassKeyword(Box::new((
                                x.cls.clone(),
                                attr,
                                subject_idx,
                            ))),
                        );
                        narrow_ops.and_all(self.bind_pattern(subject_for_attr, pattern, attr_key))
                    },
                );
                // When all sub-patterns are irrefutable, strip Placeholders that `and_all`
                // added for unmerged names. These Placeholders would incorrectly block
                // negative narrowing (preventing the class from being narrowed away in
                // subsequent match cases).
                if all_args_irrefutable
                    && let Some(subject) = match_subject.as_single()
                    && let Some((op, _)) = narrow_ops.scope.0.get_mut(subject.name())
                {
                    op.strip_placeholders();
                }
                narrow_ops
            }
            Pattern::MatchOr(x) => {
                let mut narrow_ops: Option<PatternNarrowOps> = None;
                self.start_fork(x.range);
                let n_subpatterns = x.patterns.len();
                for (idx, pattern) in x.patterns.into_iter().enumerate() {
                    self.start_branch();
                    if pattern.is_irrefutable() && idx != n_subpatterns - 1 {
                        self.error(
                            pattern.range(),
                            ErrorKind::BadMatch,
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
                if let Some(name) = &p.name
                    && !Ast::is_synthesized_empty_identifier(name)
                {
                    self.bind_definition(name, Binding::Forward(subject_idx), FlowStyle::Other);
                }
                PatternNarrowOps::new()
            }
        }
    }

    pub fn stmt_match(&mut self, mut x: StmtMatch, parent: &NestingContext) {
        let mut subject = self.declare_current_idx(Key::Anon(x.subject.range()));
        self.ensure_expr(&mut x.subject, subject.usage());
        let subject_expr = x.subject.clone();
        let subject_idx =
            self.insert_binding_current(subject, Binding::Expr(None, Box::new(*x.subject.clone())));
        // When the match subject is a tuple (e.g., `match x, y:`), extract per-element
        // narrowing subjects so that sequence patterns can narrow each element individually.
        let match_subject = if let Expr::Tuple(ref tuple_expr) = *x.subject {
            MatchSubject::Tuple(
                tuple_expr
                    .elts
                    .iter()
                    .map(|elt| expr_to_subjects(elt).first().cloned())
                    .collect(),
            )
        } else {
            match expr_to_subjects(&x.subject).first() {
                Some(s) => MatchSubject::Single(s.clone()),
                None => MatchSubject::Synthetic {
                    display_subject_range: x.subject.range(),
                },
            }
        };
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
        let mut negated_prev_subject: Option<(NarrowOp, TextRange)> = None;
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
                &Usage::NonPinningValue(None),
            );
            // First try to project previous narrows directly onto the already-evaluated
            // match subject. This is required for cases like `match self.a`, where the
            // carried narrow is stored as a facet on `self` but the branch-local subject
            // is already the projected `self.a` expression.
            let case_subject_idx = if let Some(narrowing_subject) = match_subject.as_single()
                && let Some((narrow_op, op_range)) =
                    negated_prev_ops.0.get(narrowing_subject.name())
                && let Some(projected_narrow_op) = narrow_op.rebase_onto_subject(narrowing_subject)
            {
                self.insert_binding(
                    Key::PatternNarrow(case_range),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(projected_narrow_op),
                        NarrowUseLocation::Start(*op_range),
                    ),
                )
            } else if let Some((narrow_op, op_range)) = &negated_prev_subject {
                self.insert_binding(
                    Key::PatternNarrow(case_range),
                    Binding::Narrow(
                        subject_idx,
                        Box::new(narrow_op.clone()),
                        NarrowUseLocation::Start(*op_range),
                    ),
                )
            } else if match_subject.as_single().is_some() && !negated_prev_ops.0.is_empty() {
                self.insert_binding(
                    Key::PatternNarrow(case_range),
                    Binding::Expr(None, Box::new(*subject_expr.clone())),
                )
            } else {
                subject_idx
            };
            let mut new_narrow_ops =
                self.bind_pattern(match_subject.clone(), pattern, case_subject_idx);
            self.bind_narrow_ops(
                &new_narrow_ops.scope,
                NarrowUseLocation::Span(case_range),
                &Usage::NonPinningValue(None),
            );
            // Body-only narrows (e.g. sub-pattern facet narrows) apply positively to the
            // case body but are excluded from the negation accumulated below.
            self.bind_narrow_ops(
                &new_narrow_ops.body_only,
                NarrowUseLocation::Span(case_range),
                &Usage::NonPinningValue(None),
            );
            // Reachability is checked before the guard is bound (below). This is
            // intentional: if the pattern itself can never match the subject type,
            // the case is unreachable regardless of any guard condition.
            let reachability = match match_subject.as_single() {
                Some(narrowing_subject) => new_narrow_ops
                    .scope
                    .0
                    .get(narrowing_subject.name())
                    .map(|(op, range)| (Some(narrowing_subject.clone()), op.clone(), *range)),
                None if match_subject.is_synthetic() => new_narrow_ops
                    .subject
                    .as_ref()
                    .map(|(op, range)| (None, op.clone(), *range)),
                None => None,
            };
            if let Some((narrowing_subject, op, range)) = reachability {
                self.insert_binding(
                    KeyExpect::MatchCaseReachability(case_range),
                    BindingExpect::MatchCaseReachability {
                        subject_idx: case_subject_idx,
                        narrowing_subject,
                        narrow_ops_for_case: (Box::new(op), range),
                        case_range,
                    },
                );
            }
            let has_guard = guard.is_some();
            if let Some(mut guard) = guard {
                self.ensure_expr(&mut guard, &mut Usage::NonPinningValue(None));
                let guard_narrow_ops = NarrowOps::from_expr(self, Some(guard.as_ref()));
                self.bind_narrow_ops(
                    &guard_narrow_ops,
                    NarrowUseLocation::Span(guard.range()),
                    &Usage::NonPinningValue(None),
                );
                self.insert_binding(
                    Key::Anon(guard.range()),
                    Binding::Expr(None, Box::new(*guard)),
                );
                new_narrow_ops.and_all(PatternNarrowOps::from_scope(guard_narrow_ops))
            }
            // Only accumulate narrows for the match subject. Alias names
            // from MatchAs were already copied to the subject via
            // and_for_subject and would create spurious entries if they
            // shadow outer variables. When there is no narrowing subject
            // (e.g. `match make_color():`), drop all narrows so that alias
            // names don't resolve against unrelated outer variables.
            new_narrow_ops.scope.0.retain(|name, _| {
                match_subject
                    .as_single()
                    .as_ref()
                    .is_some_and(|s| name == s.name())
            });
            let mut negated_new_narrow_ops = new_narrow_ops.negate();
            // The negation of an unguarded match arm subtracts the members it covered;
            // when the arm fully characterizes its member, this subtraction may soundly
            // reduce a non-union subject to `Never` (needed for exhaustiveness across cases).
            if !has_guard {
                negated_new_narrow_ops.scope.set_allow_never_collapse();
                if let Some((op, _)) = negated_new_narrow_ops.subject.as_mut() {
                    op.set_allow_never_collapse();
                }
            }
            negated_prev_ops.and_all(negated_new_narrow_ops.scope);
            if let Some((new_op, new_range)) = negated_new_narrow_ops.subject {
                negated_prev_subject = Some(match negated_prev_subject {
                    Some((prev_op, prev_range)) => (
                        NarrowOp::And(vec![prev_op, new_op]),
                        new_range.cover(prev_range),
                    ),
                    None => (new_op, new_range),
                });
            }
            self.stmts(body, parent);
            self.finish_branch();
        }
        if exhaustive {
            self.finish_exhaustive_fork();
        } else {
            let narrow_entries = if match_subject.is_synthetic()
                && let Some((op, range)) = &negated_prev_subject
            {
                vec![(subject_idx, Box::new(op.clone()), *range)]
            } else {
                self.build_narrow_entries(&negated_prev_ops)
            };
            let fallthrough = match match_subject.as_single() {
                Some(narrowing_subject) => negated_prev_ops
                    .0
                    .get(narrowing_subject.name())
                    .map(|(op, range)| (Some(narrowing_subject.clone()), op.clone(), *range)),
                None if match_subject.is_synthetic() => negated_prev_subject
                    .as_ref()
                    .map(|(op, range)| (None, op.clone(), *range)),
                None => None,
            };
            if let Some((narrowing_subject, op, range)) = fallthrough {
                self.insert_binding(
                    KeyExpect::MatchExhaustiveness(x.range),
                    BindingExpect::MatchExhaustiveness {
                        subject_idx,
                        narrowing_subject,
                        narrow_ops_for_fall_through: (Box::new(op), range),
                        subject_range: match_subject.subject_range(x.subject.range()),
                        show_subject_expr: match_subject.is_synthetic(),
                    },
                );
            }
            // Always create Key::Exhaustive binding for return analysis and control-flow checks.
            let exhaustive_key = self.insert_binding(
                Key::Exhaustive(ExhaustivenessKind::Match, x.range),
                Binding::Exhaustive(Box::new(ExhaustiveBinding {
                    kind: ExhaustivenessKind::Match,
                    narrow_entries,
                })),
            );
            self.finish_non_exhaustive_fork(&negated_prev_ops, Some(exhaustive_key));
        }
    }
}
