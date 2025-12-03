/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::mem;
use std::sync::Arc;

use itertools::Itertools;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::assert_bytes;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use starlark_map::small_map::SmallMap;
use starlark_map::smallmap;
use vec1::Vec1;

use crate::facet::FacetKind;
use crate::types::AnyStyle;
use crate::types::Type;

assert_bytes!(TypeInfo, 40);

/// The style of a phi.
///
/// When present, the base may be used to simplify the result and to
/// eliminate narrows that we don't want included (e.g. any narrow of `Any`
/// should be dropped in flow merging).
#[derive(Clone, Debug)]
pub enum JoinStyle<T> {
    // A simple merge (including Anywhere lookup), there's no base flow to compare against.
    SimpleMerge,
    // A merge where the name was already defined in the base flow, but was reassigned.
    ReassignmentOf(T),
    // A merge where the name was already defined in the base flow, and was only narrowed.
    NarrowOf(T),
}

impl<T> JoinStyle<T> {
    pub fn map<S>(&self, f: impl FnOnce(&T) -> S) -> JoinStyle<S> {
        match self {
            JoinStyle::SimpleMerge => JoinStyle::SimpleMerge,
            JoinStyle::ReassignmentOf(x) => JoinStyle::ReassignmentOf(f(x)),
            JoinStyle::NarrowOf(x) => JoinStyle::NarrowOf(f(x)),
        }
    }

    // Flat map - used for type info joins where the base in a join style may not
    // have data in all facet subtrees.
    fn flat_map<S>(&self, f: impl FnOnce(&T) -> Option<S>) -> JoinStyle<S> {
        match self {
            JoinStyle::SimpleMerge => JoinStyle::SimpleMerge,
            JoinStyle::ReassignmentOf(x) => {
                f(x).map_or(JoinStyle::SimpleMerge, JoinStyle::ReassignmentOf)
            }
            JoinStyle::NarrowOf(x) => f(x).map_or(JoinStyle::SimpleMerge, JoinStyle::NarrowOf),
        }
    }
}

/// The `TypeInfo` datatype represents type information associated with a
/// name or expression in a control flow context.
///
/// This is distinct from `Type` because expressions and bound names can
/// track, in addition to the type of the top-level value, zero or more
/// facet narrows where we have access to additional control-flow-dependent
/// knowledge about how a chain of facet accesses will resolve.
///
/// For example:
///
/// ```python
/// x: Foo
/// if x.foo is not None x.foo.bar is None and x.baz is None:
///     # here, `x` is still `Foo` but we also can narrow
///     # `x.foo`, `x.foo.bar`, and `x.baz`.
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Visit, VisitMut, TypeEq)]
pub struct TypeInfo {
    ty: Type,
    facets: Option<Box<NarrowedFacets>>,
}

impl TypeInfo {
    pub fn of_ty(ty: Type) -> Self {
        Self { ty, facets: None }
    }

    pub fn with_ty(self, ty: Type) -> Self {
        Self {
            ty,
            facets: self.facets,
        }
    }

    pub fn map_ty(self, f: impl FnOnce(Type) -> Type) -> Self {
        Self {
            ty: f(self.ty),
            facets: self.facets,
        }
    }

    pub fn record_key_completion(&mut self, facets: &Vec1<FacetKind>, ty: Option<Type>) {
        if let Some((facet, rest)) = facets.as_slice().split_first() {
            let narrowed = self
                .facets
                .get_or_insert_with(|| Box::new(NarrowedFacets::default()));
            narrowed.ensure_completion_path(facet, rest, ty);
        }
    }

    pub fn type_at_facet(&self, facet: &FacetKind) -> Option<&Type> {
        match self.get_at_facet(facet) {
            None | Some(NarrowedFacet::WithoutRoot { .. }) => None,
            Some(NarrowedFacet::Leaf(ty)) | Some(NarrowedFacet::WithRoot(ty, _)) => Some(ty),
        }
    }

    pub fn at_facet(&self, facet: &FacetKind, fallback: impl Fn() -> Type) -> Self {
        match self.get_at_facet(facet) {
            None => TypeInfo::of_ty(fallback()),
            Some(NarrowedFacet::Leaf(ty)) => Self::of_ty(ty.clone()),
            Some(NarrowedFacet::WithoutRoot { facets, .. }) => Self {
                ty: fallback(),
                facets: Some(Box::new(facets.clone())),
            },
            Some(NarrowedFacet::WithRoot(ty, narrowed_facets)) => Self {
                ty: ty.clone(),
                facets: Some(Box::new(narrowed_facets.clone())),
            },
        }
    }

    pub fn with_narrow(&self, facets: &Vec1<FacetKind>, ty: Type) -> Self {
        if ty.is_error() {
            return self.clone();
        }
        let mut type_info = self.clone();
        type_info.add_narrow(facets, ty);
        type_info
    }

    /// Join zero or more `TypeInfo`s together:
    /// - We'll take the union of the top-level types
    /// - At facet chains where all branches narrow, take a union of the narrowed types.
    /// - Drop narrowing for facet chains where at least one branch does not narrow
    ///
    /// In the case where there are no branches, we get `Never` with no narrows.
    pub fn join(
        mut branches: Vec<Self>,
        union_types: &impl Fn(Vec<Type>) -> Type,
        is_subset_eq: &impl Fn(&Type, &Type) -> bool,
        join_style: JoinStyle<Arc<TypeInfo>>,
    ) -> Self {
        match branches.len() {
            0 => Self::of_ty(Type::never()),
            1 => branches.pop().unwrap(),
            n => {
                let (tys, facets_branches): (Vec<Type>, Vec<Option<NarrowedFacets>>) = branches
                    .into_iter()
                    .map(|TypeInfo { ty, facets }| (ty, facets.map(|x| *x)))
                    .unzip();
                let ty = join_types(
                    tys,
                    union_types,
                    is_subset_eq,
                    join_style.map(|base_type_info| base_type_info.ty.clone()),
                );
                let branches = facets_branches.into_iter().flatten().collect::<Vec<_>>();
                let facets = if branches.len() == n {
                    NarrowedFacets::join(
                        branches,
                        union_types,
                        is_subset_eq,
                        join_style.map(|base_type_info| {
                            base_type_info.facets.as_ref().map(|f| f.as_ref().clone())
                        }),
                    )
                } else {
                    // at least one branch had empty facets, we should drop facets from the join
                    None
                };
                Self {
                    ty,
                    facets: facets.map(Box::new),
                }
            }
        }
    }

    /// Add a narrow to the TypeInfo. This is used for narrowing conditions, not assignment - it
    /// only adds a new narrow (possibly overwriting any preexisting narrow), without changing subtrees.
    fn add_narrow(&mut self, facets: &Vec1<FacetKind>, ty: Type) {
        if let Some((facet, more_facets)) = facets.split_first() {
            if let Some(narrowed_facets) = &mut self.facets {
                narrowed_facets.add_narrow(facet, more_facets, ty);
            } else {
                self.facets = Some(Box::new(NarrowedFacets::of_narrow(
                    facet.clone(),
                    more_facets,
                    ty,
                )));
            }
        } else {
            unreachable!(
                "We know the Vec1 will split. But the safe API, split_off_first, is not ref-based."
            )
        }
    }

    /// Update for an assignment. This is different from `add_narrow` for two reasons:
    /// - It invalidates any existing subtree at that facet chain in addition to narrowing.
    /// - There may not be a type available for the assignment (in which case we *just* invalidate)
    pub fn update_for_assignment(&mut self, facets: &Vec1<FacetKind>, ty: Option<Type>) {
        if let Some((facet, more_facets)) = facets.split_first() {
            if let Some(narrowed_facets) = &mut self.facets {
                // If there might be an existing narrow, we need to recurse down the chain of facets and update.
                narrowed_facets.update_for_assignment(facet, more_facets, ty);
            } else if let Some(ty) = ty {
                // If there is no existing narrow and a Type is available, we should create a narrow.
                self.facets = Some(Box::new(NarrowedFacets::of_narrow(
                    facet.clone(),
                    more_facets,
                    ty,
                )));
            } // ... else we have no type nor an existing narrow, nothing to do
        } else {
            unreachable!(
                "We know the Vec1 will split. But the safe API, split_off_first, is not ref-based."
            )
        }
    }

    /// When we assign to a facet chain containing an unknown index, we don't know which index changed
    /// and have to invalidate all of them.
    pub fn invalidate_all_indexes_for_assignment(&mut self, facets: &[FacetKind]) {
        if let Some(narrowed_facets) = &mut self.facets {
            narrowed_facets.clear_index_narrows(facets);
        }
    }

    /// Return the known narrowings for dictionary-style key facets at the provided prefix.
    ///
    /// The `prefix` is the sequence of facets (attributes, indexes, and keys) that identify the
    /// container whose keys we are interested in. When the prefix is empty, this inspects the
    /// top-level facets on the `TypeInfo` itself.
    pub fn key_facets_at(&self, prefix: &[FacetKind]) -> Vec<(String, Option<Type>)> {
        fn collect_keys(facets: &NarrowedFacets) -> Vec<(String, Option<Type>)> {
            facets
                .0
                .iter()
                .filter_map(|(facet, narrowed)| {
                    if let FacetKind::Key(key) = facet {
                        let ty = match narrowed {
                            NarrowedFacet::Leaf(ty) => Some(ty.clone()),
                            NarrowedFacet::WithRoot(ty, _) => Some(ty.clone()),
                            NarrowedFacet::WithoutRoot { completion_ty, .. } => {
                                completion_ty.clone()
                            }
                        };
                        Some((key.clone(), ty))
                    } else {
                        None
                    }
                })
                .collect()
        }

        fn descend<'a>(
            mut current: &'a NarrowedFacets,
            prefix: &[FacetKind],
        ) -> Option<&'a NarrowedFacets> {
            for facet in prefix {
                let narrowed = current.0.get(facet)?;
                match narrowed {
                    NarrowedFacet::Leaf(_) => return None,
                    NarrowedFacet::WithRoot(_, nested) => {
                        current = nested;
                    }
                    NarrowedFacet::WithoutRoot { facets, .. } => {
                        current = facets;
                    }
                }
            }
            Some(current)
        }

        match &self.facets {
            Some(facets) => {
                if prefix.is_empty() {
                    collect_keys(facets.as_ref())
                } else if let Some(target) = descend(facets.as_ref(), prefix) {
                    collect_keys(target)
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        }
    }

    pub fn ty(&self) -> &Type {
        &self.ty
    }

    pub fn into_ty(self) -> Type {
        self.ty
    }

    pub fn arc_clone(self: Arc<Self>) -> Self {
        Arc::unwrap_or_clone(self)
    }

    pub fn arc_clone_ty(self: Arc<Self>) -> Type {
        self.arc_clone().into_ty()
    }

    fn get_at_facet(&self, facet: &FacetKind) -> Option<&NarrowedFacet> {
        if let Some(narrowed_facets) = &self.facets {
            narrowed_facets.get(facet)
        } else {
            None
        }
    }
}

impl Display for TypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.ty().fmt(f)?;
        if let Some(facets) = &self.facets {
            write!(f, " ({facets})")?;
        }
        Ok(())
    }
}

/// Limit on the size of [NarrowedFacets].
///
/// In order to avoid O(n^2) performance of things like x.a = 1; x.b = 2 ....
/// we cap the number of facets at one level.
///
/// Note that we don't cap the overall size of a [TypeInfo], merely the fanout at
/// each level. We may need to cap the overall size, if that becomes a problem.
const NARROWED_FACETS_LIMIT: usize = 50;

/// The facets one level down, bounded by [NARROWED_FACETS_LIMIT].
#[derive(Debug, Clone, PartialEq, Eq, TypeEq)]
struct NarrowedFacets(SmallMap<FacetKind, NarrowedFacet>);

impl Default for NarrowedFacets {
    fn default() -> Self {
        Self(SmallMap::new())
    }
}

impl NarrowedFacets {
    fn insert(&mut self, facet: FacetKind, value: NarrowedFacet) {
        // Only insert if there is space, or if the key is already present (so we overwrite)
        if self.0.len() < NARROWED_FACETS_LIMIT || self.0.contains_key(&facet) {
            self.0.insert(facet, value);
        }
    }

    fn add_narrow(&mut self, facet: &FacetKind, more_facets: &[FacetKind], ty: Type) {
        match self.0.get_mut(facet) {
            Some(narrowed_facet) => {
                narrowed_facet.add_narrow(more_facets, ty);
            }
            None => self.insert(facet.clone(), NarrowedFacet::new(more_facets, ty)),
        };
    }

    fn clear_index_narrows(&mut self, facets: &[FacetKind]) {
        match facets {
            [] => {
                self.0.retain(|k, _| !k.invalidate_on_unknown_assignment());
            }
            [facet, more_facets @ ..] => match self.0.get_mut(facet) {
                Some(narrowed_facet) => {
                    narrowed_facet.clear_index_narrows(more_facets);
                }
                _ => {}
            },
        }
    }

    fn update_for_assignment(
        &mut self,
        facet: &FacetKind,
        more_facets: &[FacetKind],
        ty: Option<Type>,
    ) {
        match more_facets {
            [] => {
                if let Some(ty) = ty {
                    self.insert(facet.clone(), NarrowedFacet::new(more_facets, ty));
                } else {
                    self.0.shift_remove(facet);
                }
            }
            [next_facet, remaining_facets @ ..] => {
                if let Some(narrowed_facet) = self.0.get_mut(facet) {
                    match narrowed_facet {
                        NarrowedFacet::Leaf(..) if let Some(ty) = ty => {
                            narrowed_facet.add_narrow(more_facets, ty);
                        }
                        NarrowedFacet::Leaf(..) => {}
                        NarrowedFacet::WithoutRoot { facets, .. }
                        | NarrowedFacet::WithRoot(_, facets) => {
                            facets.update_for_assignment(next_facet, remaining_facets, ty);
                        }
                    }
                } else if let Some(ty) = ty {
                    self.insert(facet.clone(), NarrowedFacet::new(more_facets, ty));
                }
                // ... else there is no existing narrow and no narrow type, so do nothing.
            }
        }
    }

    fn get(&self, facet: &FacetKind) -> Option<&NarrowedFacet> {
        self.0.get(facet)
    }

    fn of_narrow(facet: FacetKind, more_facets: &[FacetKind], ty: Type) -> Self {
        Self(smallmap! {facet => NarrowedFacet::new(more_facets, ty)})
    }

    fn ensure_completion_path(
        &mut self,
        facet: &FacetKind,
        rest: &[FacetKind],
        completion_ty: Option<Type>,
    ) {
        let entry = self.0.entry(facet.clone()).or_insert_with(|| {
            if rest.is_empty() {
                NarrowedFacet::WithoutRoot {
                    completion_ty: None,
                    facets: NarrowedFacets::default(),
                }
            } else {
                let mut nested = NarrowedFacets::default();
                nested.ensure_completion_path(&rest[0], &rest[1..], completion_ty.clone());
                NarrowedFacet::WithoutRoot {
                    completion_ty: None,
                    facets: nested,
                }
            }
        });

        if let Some((next, tail)) = rest.split_first() {
            entry.ensure_completion_child(next, tail, completion_ty);
        } else {
            entry.set_completion_ty(completion_ty);
        }
    }

    fn join(
        mut branches: Vec<Self>,
        union_types: &impl Fn(Vec<Type>) -> Type,
        is_subset_eq: &impl Fn(&Type, &Type) -> bool,
        join_style: JoinStyle<Option<NarrowedFacets>>,
    ) -> Option<Self> {
        match branches.len() {
            0 => None,
            1 => Some(branches.pop().unwrap()),
            n => {
                let first = branches[0].clone();
                let tail = &branches[1..];
                let facets: SmallMap<_, _> = first
                    .0
                    .into_iter()
                    .filter_map(|(facet, narrowed_facet)| {
                        let mut facet_branches = Vec::with_capacity(n);
                        facet_branches.push(narrowed_facet);
                        facet_branches
                            .extend(tail.iter().filter_map(|facets| facets.get(&facet).cloned()));
                        // If any map lacked this facet, we just drop it. Only join if all maps have it.
                        if facet_branches.len() == n {
                            NarrowedFacet::join(
                                facet_branches,
                                union_types,
                                is_subset_eq,
                                join_style.map(|base_facets| {
                                    base_facets.as_ref().and_then(|f| f.get(&facet)).cloned()
                                }),
                            )
                            .map(move |narrowed_facet| (facet, narrowed_facet))
                        } else {
                            None
                        }
                    })
                    .collect();
                if facets.is_empty() {
                    None
                } else {
                    Some(Self(facets))
                }
            }
        }
    }

    fn fmt_with_prefix(
        &self,
        prefix: &mut Vec<FacetKind>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let mut first = true;
        for (facet, value) in self.0.iter() {
            if first {
                first = false
            } else {
                write!(f, ", ")?;
            }
            match value {
                NarrowedFacet::Leaf(ty) => Self::fmt_type_with_label(prefix, facet, ty, f),
                NarrowedFacet::WithRoot(ty, facets) => {
                    Self::fmt_type_with_label(prefix, facet, ty, f)?;
                    write!(f, ", ")?;
                    facets.fmt_with_prefix_and_facet(prefix, facet, f)
                }
                NarrowedFacet::WithoutRoot { facets, .. } => {
                    facets.fmt_with_prefix_and_facet(prefix, facet, f)
                }
            }?;
        }
        Ok(())
    }

    fn fmt_with_prefix_and_facet<'a>(
        &self,
        prefix: &mut Vec<FacetKind>,
        facet: &'a FacetKind,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        prefix.push(facet.clone());
        self.fmt_with_prefix(prefix, f)?;
        prefix.pop();
        Ok(())
    }

    fn fmt_type_with_label(
        prefix: &[FacetKind],
        facet: &FacetKind,
        ty: &Type,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "_{}{}: {}", prefix.iter().join(""), facet, ty)
    }
}

impl Visit<Type> for NarrowedFacets {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        let facets = &self.0;
        facets.values().for_each(|value| {
            value.visit(f);
        })
    }
}

impl VisitMut<Type> for NarrowedFacets {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        let facets = &mut self.0;
        facets.values_mut().for_each(|value| {
            value.visit_mut(f);
        })
    }
}

impl Display for NarrowedFacets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_prefix(&mut Vec::new(), f)
    }
}

/// A `NarrowedFacet` represents a single facet within a tree of narrowed
/// facets. The facet itself may or may not be narrowed, and it may or
/// may not have any sub-facets (but at least one must be the case, or it
/// wouldn't be in the tree at all)
#[derive(Debug, Clone, Visit, VisitMut, PartialEq, Eq, TypeEq)]
enum NarrowedFacet {
    /// This facet is narrowed, and has no narrowed sub-facet (Leaf)
    Leaf(Type),
    /// This facet is narrowed, and has one or more narrowed sub-facet (WithRoot)
    WithRoot(Type, NarrowedFacets),
    /// This facet is not narrowed, and has one or more narrowed sub-facet (WithoutRoot)
    WithoutRoot {
        completion_ty: Option<Type>,
        facets: NarrowedFacets,
    },
}

impl NarrowedFacet {
    fn new(facets: &[FacetKind], ty: Type) -> Self {
        match facets {
            [] => Self::Leaf(ty),
            [facet, more_facets @ ..] => Self::WithoutRoot {
                completion_ty: None,
                facets: NarrowedFacets::of_narrow((*facet).clone(), more_facets, ty),
            },
        }
    }

    fn add_narrow(&mut self, facets: &[FacetKind], narrowed_ty: Type) {
        // Take ownership of self so we can destructure it and potentially change enum variants.
        let mut current = NarrowedFacet::Leaf(Type::None);
        mem::swap(self, &mut current);
        *self = current.with_narrow(facets, narrowed_ty)
    }

    fn clear_index_narrows(&mut self, facets: &[FacetKind]) {
        match self {
            Self::Leaf(_) => {}
            Self::WithRoot(_, narrowed_facets)
            | Self::WithoutRoot {
                facets: narrowed_facets,
                ..
            } => narrowed_facets.clear_index_narrows(facets),
        }
    }

    fn with_narrow(self, facets: &[FacetKind], narrowed_ty: Type) -> Self {
        match facets {
            [] => match self {
                Self::Leaf(_) => Self::Leaf(narrowed_ty),
                Self::WithRoot(_, narrowed_facets)
                | Self::WithoutRoot {
                    facets: narrowed_facets,
                    ..
                } => Self::WithRoot(narrowed_ty, narrowed_facets),
            },
            [facet, more_facets @ ..] => match self {
                Self::Leaf(root_ty) => {
                    let narrowed_facets =
                        NarrowedFacets::of_narrow((*facet).clone(), more_facets, narrowed_ty);
                    Self::WithRoot(root_ty, narrowed_facets)
                }
                Self::WithoutRoot { mut facets, .. } => {
                    facets.add_narrow(facet, more_facets, narrowed_ty);
                    Self::WithoutRoot {
                        completion_ty: None,
                        facets,
                    }
                }
                Self::WithRoot(root_ty, mut narrowed_facets) => {
                    narrowed_facets.add_narrow(facet, more_facets, narrowed_ty);
                    Self::WithRoot(root_ty, narrowed_facets)
                }
            },
        }
    }

    fn set_completion_ty(&mut self, ty: Option<Type>) {
        if let NarrowedFacet::WithoutRoot { completion_ty, .. } = self {
            *completion_ty = ty;
        }
    }

    fn ensure_completion_child(
        &mut self,
        facet: &FacetKind,
        rest: &[FacetKind],
        completion_ty: Option<Type>,
    ) {
        match self {
            Self::Leaf(root_ty) => {
                let mut nested = NarrowedFacets::default();
                nested.ensure_completion_path(facet, rest, completion_ty);
                *self = Self::WithRoot(root_ty.clone(), nested);
            }
            Self::WithRoot(_, nested) => {
                nested.ensure_completion_path(facet, rest, completion_ty);
            }
            Self::WithoutRoot { facets, .. } => {
                facets.ensure_completion_path(facet, rest, completion_ty);
            }
        }
    }

    fn join(
        branches: Vec<Self>,
        union_types: &impl Fn(Vec<Type>) -> Type,
        is_subset_eq: &impl Fn(&Type, &Type) -> bool,
        join_style: JoinStyle<Option<NarrowedFacet>>,
    ) -> Option<Self> {
        fn monadic_push_option<T>(acc: &mut Option<Vec<T>>, item: Option<T>) {
            match item {
                None => *acc = None,
                Some(item) => {
                    if let Some(acc) = acc {
                        acc.push(item)
                    }
                }
            }
        }
        let mut ty_branches = Some(Vec::with_capacity(branches.len()));
        let mut facets_branches = Some(Vec::with_capacity(branches.len()));
        for branch in branches {
            let (ty, facets) = match branch {
                Self::WithRoot(ty, facets) => (Some(ty), Some(facets)),
                Self::Leaf(ty) => (Some(ty), None),
                Self::WithoutRoot { facets, .. } => (None, Some(facets)),
            };
            monadic_push_option(&mut ty_branches, ty);
            monadic_push_option(&mut facets_branches, facets);
            if let (None, None) = (&ty_branches, &facets_branches) {
                return None;
            }
        }
        let ty = ty_branches.map(|tys| {
            join_types(
                tys,
                union_types,
                is_subset_eq,
                join_style.flat_map(|base_facet| {
                    base_facet.as_ref().and_then(|facet| match facet {
                        NarrowedFacet::WithRoot(ty, _) | NarrowedFacet::Leaf(ty) => {
                            Some(ty.clone())
                        }
                        NarrowedFacet::WithoutRoot { .. } => None,
                    })
                }),
            )
        });
        let facets = facets_branches.and_then(|branches| {
            NarrowedFacets::join(
                branches,
                union_types,
                is_subset_eq,
                join_style.map(|base_facet| {
                    base_facet.as_ref().and_then(|facet| match facet {
                        NarrowedFacet::WithRoot(_, facets)
                        | NarrowedFacet::WithoutRoot { facets, .. } => Some(facets.clone()),
                        NarrowedFacet::Leaf(_) => None,
                    })
                }),
            )
        });
        match (ty, facets) {
            (None, None) => None,
            (Some(ty), None) => Some(Self::Leaf(ty)),
            (Some(ty), Some(facets)) => Some(Self::WithRoot(ty, facets)),
            (None, Some(facets)) => Some(Self::WithoutRoot {
                completion_ty: None,
                facets,
            }),
        }
    }
}

/// Join types. The result is typically a union, but if we have a base type available (which
/// occurs when the join is from control flow and there was a type before the branch), we
/// may be able to get a better result.
fn join_types(
    types: Vec<Type>,
    union_types: &impl Fn(Vec<Type>) -> Type,
    is_subset_eq: &impl Fn(&Type, &Type) -> bool,
    join_style: JoinStyle<Type>,
) -> Type {
    match join_style {
        JoinStyle::SimpleMerge => union_types(types),
        JoinStyle::NarrowOf(base_ty) => {
            join_types_impl(types, base_ty, true, union_types, is_subset_eq)
        }
        JoinStyle::ReassignmentOf(base_ty) => {
            join_types_impl(types, base_ty, false, union_types, is_subset_eq)
        }
    }
}

/// Given a base flow type and a naive join of control flow branches, try to simplify
/// the join.
/// - If the base type is `Any`, and `Any` is still present in the join, just use `Any`.
///   This avoids creating union types like `Any | int` on gradual code that assigns or
///   narrows a gradually-typed variable. We only do this for explicit and implicit any,
///   not for `Any` that resulted from a type error.
/// - If the merge involves only narrows of base *and* the base type is a subset
///   of the resulting union, simplify the union. This would not be needed if we
///   had general union simplification, but it is useful today because:
///   - general union simplification is quadratic given our current architecture
///   - but this particular simplification is linear, since we have an initial guess
///   - the simplified join types are much more readable and performant downstream
fn join_types_impl(
    types: Vec<Type>,
    base_ty: Type,
    is_narrow: bool,
    union_types: &impl Fn(Vec<Type>) -> Type,
    is_subset_eq: &impl Fn(&Type, &Type) -> bool,
) -> Type {
    if matches!(base_ty, Type::Any(AnyStyle::Explicit | AnyStyle::Implicit))
        && types.iter().any(|t| t.is_any())
    {
        base_ty
    } else if is_narrow {
        // Check for the case where `base_ty` is directly in the merge before doing
        // a subset check. We do this to avoid the possibility of pinning a `Var` to
        // itself inside the join (which at one point caused a stack overflow).
        if types.iter().any(|t| t == &base_ty) {
            base_ty
        } else {
            let joined_ty = union_types(types);
            if is_subset_eq(&base_ty, &joined_ty) {
                base_ty
            } else {
                joined_ty
            }
        }
    } else {
        union_types(types)
    }
}

#[cfg(test)]
mod tests {

    use ruff_python_ast::name::Name;
    use vec1::Vec1;

    use crate::class::ClassType;
    use crate::display::tests::fake_class;
    use crate::facet::FacetKind;
    use crate::type_info::JoinStyle;
    use crate::type_info::TypeInfo;
    use crate::types::TArgs;
    use crate::types::Type;

    fn fake_class_type(class_name: &str) -> Type {
        Type::ClassType(ClassType::new(
            fake_class(class_name, "class_defs_module", 5),
            TArgs::default(),
        ))
    }

    #[test]
    fn test_type_info_one_level_only() {
        let x = || FacetKind::Attribute(Name::new_static("x"));
        let y = || FacetKind::Attribute(Name::new_static("y"));
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        assert_eq!(type_info.to_string(), "Foo");
        type_info.add_narrow(&Vec1::new(x()), fake_class_type("Bar"));
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar)");
        type_info.add_narrow(&Vec1::new(y()), fake_class_type("Baz"));
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar, _.y: Baz)");
    }

    #[test]
    fn test_type_info_adding_sub_facets() {
        let x = || FacetKind::Attribute(Name::new_static("x"));
        let y = || FacetKind::Attribute(Name::new_static("y"));
        let z = || FacetKind::Attribute(Name::new_static("z"));
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        type_info.add_narrow(&Vec1::new(x()), fake_class_type("Bar"));
        type_info.add_narrow(&Vec1::from_vec_push(vec![x()], y()), fake_class_type("Baz"));
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar, _.x.y: Baz)");
        type_info.add_narrow(&Vec1::from_vec_push(vec![x()], z()), fake_class_type("Qux"));
        assert_eq!(
            type_info.to_string(),
            "Foo (_.x: Bar, _.x.y: Baz, _.x.z: Qux)"
        );
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y()], x()),
            fake_class_type("Foo"),
        );
        assert_eq!(
            type_info.to_string(),
            "Foo (_.x: Bar, _.x.y: Baz, _.x.y.x: Foo, _.x.z: Qux)"
        );
    }

    #[test]
    fn test_type_info_creating_subtrees_and_narrowing_roots() {
        let x = || FacetKind::Attribute(Name::new_static("x"));
        let y = || FacetKind::Attribute(Name::new_static("y"));
        let z = || FacetKind::Attribute(Name::new_static("z"));
        let w = || FacetKind::Attribute(Name::new_static("w"));
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y()], z()),
            fake_class_type("Bar"),
        );
        assert_eq!(type_info.to_string(), "Foo (_.x.y.z: Bar)");
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y()], w()),
            fake_class_type("Baz"),
        );
        assert_eq!(type_info.to_string(), "Foo (_.x.y.z: Bar, _.x.y.w: Baz)");
        type_info.add_narrow(&Vec1::from_vec_push(vec![x()], y()), fake_class_type("Qux"));
        assert_eq!(
            type_info.to_string(),
            "Foo (_.x.y: Qux, _.x.y.z: Bar, _.x.y.w: Baz)"
        );
    }

    #[test]
    fn test_type_info_overwiting_existing_narrows() {
        let x = || FacetKind::Attribute(Name::new_static("x"));
        let y = || FacetKind::Attribute(Name::new_static("y"));
        let z = || FacetKind::Attribute(Name::new_static("z"));
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y()], z()),
            fake_class_type("Bar"),
        );
        type_info.add_narrow(&Vec1::from_vec_push(vec![x()], y()), fake_class_type("Qux"));
        assert_eq!(type_info.to_string(), "Foo (_.x.y: Qux, _.x.y.z: Bar)");
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x()], y()),
            fake_class_type("Qux1"),
        );
        assert_eq!(type_info.to_string(), "Foo (_.x.y: Qux1, _.x.y.z: Bar)");
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y()], z()),
            fake_class_type("Bar1"),
        );
        assert_eq!(type_info.to_string(), "Foo (_.x.y: Qux1, _.x.y.z: Bar1)");
    }

    #[test]
    fn test_type_info_empty_join() {
        let type_info = TypeInfo::join(
            Vec::new(),
            &|ts| {
                if ts.is_empty() {
                    fake_class_type("Never")
                } else {
                    fake_class_type("FakeUnionType")
                }
            },
            &|_, _| false,
            JoinStyle::SimpleMerge,
        );
        assert_eq!(type_info.to_string(), "Never");
    }

    #[test]
    fn test_type_info_invalidating_prefix() {
        let x = || FacetKind::Attribute(Name::new_static("x"));
        let y = || FacetKind::Attribute(Name::new_static("y"));
        let idx0 = || FacetKind::Index(0);
        let idx1 = || FacetKind::Index(1);
        let z = || FacetKind::Attribute(Name::new_static("z"));
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        type_info.add_narrow(&Vec1::from_vec_push(vec![x()], y()), fake_class_type("Bar"));
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y(), idx0()], z()),
            fake_class_type("Bar"),
        );
        type_info.add_narrow(
            &Vec1::from_vec_push(vec![x(), y(), idx1()], z()),
            fake_class_type("Bar"),
        );
        assert_eq!(
            type_info.to_string(),
            "Foo (_.x.y: Bar, _.x.y[0].z: Bar, _.x.y[1].z: Bar)"
        );
        // x has no narrowed indexes, so do nothing
        type_info.invalidate_all_indexes_for_assignment(&[x()]);
        assert_eq!(
            type_info.to_string(),
            "Foo (_.x.y: Bar, _.x.y[0].z: Bar, _.x.y[1].z: Bar)"
        );
        // this path doesn't have any narrowing, so do nothing
        type_info.invalidate_all_indexes_for_assignment(&[x(), z()]);
        assert_eq!(
            type_info.to_string(),
            "Foo (_.x.y: Bar, _.x.y[0].z: Bar, _.x.y[1].z: Bar)"
        );
        // this clears the narrowing for both x.y[0] and x.y[1], but not x.y
        type_info.invalidate_all_indexes_for_assignment(&[x(), y()]);
        assert_eq!(type_info.to_string(), "Foo (_.x.y: Bar, )");
    }

    #[test]
    fn test_type_info_do_not_invalidate_parent() {
        let x = || FacetKind::Attribute(Name::new_static("x"));
        let y = || FacetKind::Key("y".to_owned());
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        type_info.add_narrow(
            &Vec1::from_vec_push(Vec::new(), x()),
            fake_class_type("Bar"),
        );
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar)");
        type_info.update_for_assignment(
            &Vec1::from_vec_push(vec![x()], y()),
            Some(fake_class_type("Baz")),
        );
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar, _.x[\"y\"]: Baz)");
    }
}
