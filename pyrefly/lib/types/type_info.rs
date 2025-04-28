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
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;

use crate::assert_bytes;
use crate::types::types::Type;
use crate::util::visit::Visit;
use crate::util::visit::VisitMut;

assert_bytes!(TypeInfo, 40);

/// The `TypeInfo` datatype represents type information associated with a
/// name or expression in a control flow context.
///
/// This is distinct from `Type` because expressions and bound names can
/// track, in addition to the type of the top-level value, zero or more
/// attribute narrows where we have access to additional control-flow-dependent
/// knowledge about how a chain of attribute accesses will resolve.
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
    attrs: Option<Box<NarrowedAttrs>>,
}

impl TypeInfo {
    pub fn of_ty(ty: Type) -> Self {
        Self { ty, attrs: None }
    }

    pub fn with_ty(self, ty: Type) -> Self {
        Self {
            ty,
            attrs: self.attrs,
        }
    }

    pub fn type_at_name(&self, name: &Name) -> Option<&Type> {
        match self.get_at_name(name) {
            None | Some(NarrowedAttr::WithoutRoot(..)) => None,
            Some(NarrowedAttr::Leaf(ty)) | Some(NarrowedAttr::WithRoot(ty, _)) => Some(ty),
        }
    }

    pub fn at_name(&self, name: &Name, fallback: impl Fn() -> Type) -> Self {
        match self.get_at_name(name) {
            None => TypeInfo::of_ty(fallback()),
            Some(NarrowedAttr::Leaf(ty)) => Self::of_ty(ty.clone()),
            Some(NarrowedAttr::WithoutRoot(attrs)) => Self {
                ty: fallback(),
                attrs: Some(Box::new(attrs.clone())),
            },
            Some(NarrowedAttr::WithRoot(ty, attrs)) => Self {
                ty: ty.clone(),
                attrs: Some(Box::new(attrs.clone())),
            },
        }
    }

    pub fn with_narrow(&self, names: &Vec1<Name>, ty: Type) -> Self {
        let mut type_info = self.clone();
        type_info.add_narrow(names, ty);
        type_info
    }

    /// Join zero or more `TypeInfo`s together:
    /// - We'll take the union of the top-level types
    /// - At attribute chains where all branches narrow, take a union of the narrowed types.
    /// - Drop narrowing for attribute chains where at least one branch does not narrow
    ///
    /// In the case where there are no branches, we get `Never` with no narrows.
    pub fn join(mut branches: Vec<Self>, union_types: &impl Fn(Vec<Type>) -> Type) -> Self {
        match branches.len() {
            0 => Self::of_ty(Type::never()),
            1 => branches.pop().unwrap(),
            _ => {
                let (tys, attrs): (Vec<Type>, Vec<Option<&NarrowedAttrs>>) = branches
                    .iter()
                    .map(|TypeInfo { ty, attrs }| (ty.clone(), attrs.as_ref().map(|a| a.as_ref())))
                    .unzip();
                let ty = union_types(tys);
                let attrs = NarrowedAttrs::join(attrs.into_iter().flatten().collect(), union_types);
                Self {
                    ty,
                    attrs: attrs.map(Box::new),
                }
            }
        }
    }

    /// Add a narrow to the TypeInfo. This is used for narrowing conditions, not assignment - it
    /// only adds a new narrow (possibly overwriting any prexisting narrow), without changing subtrees.
    fn add_narrow(&mut self, names: &Vec1<Name>, ty: Type) {
        if let Some((name, more_names)) = names.split_first() {
            if let Some(attrs) = &mut self.attrs {
                attrs.add_narrow(name.clone(), more_names, ty);
            } else {
                self.attrs = Some(Box::new(NarrowedAttrs::of_narrow(
                    name.clone(),
                    more_names,
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
    /// - It invalidates any existing subtree at that attribute chain in addition to narrowing.
    /// - There may not be a type available for the assignment (in which case we *just* invalidate)
    pub fn update_for_assignment(&mut self, names: &Vec1<Name>, ty: Option<Type>) {
        if let Some((name, more_names)) = names.split_first() {
            if let Some(attrs) = &mut self.attrs {
                // If there might be an existing narrow, we need to recurse down the chain of names and update.
                attrs.update_for_assignment(name, more_names, ty);
            } else if let Some(ty) = ty {
                // If there is no existing narrow and a Type is available, we should create a narrow.
                self.attrs = Some(Box::new(NarrowedAttrs::of_narrow(
                    name.clone(),
                    more_names,
                    ty,
                )));
            } // ... else we have no type nor an existing narrow, nothing to do
        } else {
            unreachable!(
                "We know the Vec1 will split. But the safe API, split_off_first, is not ref-based."
            )
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

    fn get_at_name(&self, name: &Name) -> Option<&NarrowedAttr> {
        if let Some(box attrs) = &self.attrs {
            attrs.get(name)
        } else {
            None
        }
    }
}

impl Display for TypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.ty().fmt(f)?;
        if let Some(attrs) = &self.attrs {
            write!(f, " ({})", attrs)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, TypeEq)]
struct NarrowedAttrs(SmallMap<Name, NarrowedAttr>);

impl NarrowedAttrs {
    fn add_narrow(&mut self, name: Name, more_names: &[Name], ty: Type) {
        let attrs = &mut self.0;
        match attrs.get_mut(&name) {
            Some(attr) => {
                attr.add_narrow(more_names, ty);
            }
            None => {
                attrs.insert(name, NarrowedAttr::new(more_names, ty));
            }
        };
    }

    fn update_for_assignment(&mut self, name: &Name, more_names: &[Name], ty: Option<Type>) {
        let attrs = &mut self.0;
        match more_names {
            [] => {
                if let Some(ty) = ty {
                    attrs.insert(name.clone(), NarrowedAttr::new(more_names, ty));
                } else {
                    attrs.shift_remove(name);
                }
            }
            [name, more_names @ ..] => {
                if let Some(attr) = attrs.get_mut(name) {
                    attr.update_for_assignment(name, more_names, ty);
                } // ... else there is no existing narrow and no narrow type, so do nothing.
            }
        }
    }

    fn get(&self, name: &Name) -> Option<&NarrowedAttr> {
        self.0.get(name)
    }

    fn of_narrow(name: Name, more_names: &[Name], ty: Type) -> Self {
        let mut attrs = SmallMap::with_capacity(1);
        attrs.insert(name.clone(), NarrowedAttr::new(more_names, ty));
        Self(attrs)
    }

    fn join(mut branches: Vec<&Self>, union_types: &impl Fn(Vec<Type>) -> Type) -> Option<Self> {
        match branches.len() {
            0 => None,
            1 => Some(branches.pop().unwrap().clone()),
            n => {
                let first = branches[0].clone();
                let tail = &branches[1..];
                let attrs: SmallMap<_, _> = first
                    .0
                    .into_iter()
                    .filter_map(|(name, attr)| {
                        let mut attr_branches = Vec::with_capacity(n);
                        attr_branches.push(attr.clone());
                        attr_branches
                            .extend(tail.iter().filter_map(|attrs| attrs.get(&name).cloned()));
                        // If any map lacked this name, we just drop it. Only join if all maps have it.
                        if attr_branches.len() == n {
                            NarrowedAttr::join(attr_branches, union_types)
                                .map(move |attr| (name, attr))
                        } else {
                            None
                        }
                    })
                    .collect();
                if attrs.is_empty() {
                    None
                } else {
                    Some(Self(attrs))
                }
            }
        }
    }

    fn fmt_with_prefix(&self, prefix: &mut Vec<String>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let attrs = &self.0;
        let mut first = true;
        for (name, value) in attrs.iter() {
            if first {
                first = false
            } else {
                write!(f, ", ")?;
            }
            match value {
                NarrowedAttr::Leaf(ty) => Self::fmt_type_with_label(prefix, name, ty, f),
                NarrowedAttr::WithRoot(ty, attrs) => {
                    Self::fmt_type_with_label(prefix, name, ty, f)?;
                    write!(f, ", ")?;
                    attrs.fmt_with_prefix_and_name(prefix, name, f)
                }
                NarrowedAttr::WithoutRoot(attrs) => attrs.fmt_with_prefix_and_name(prefix, name, f),
            }?;
        }
        Ok(())
    }

    fn fmt_with_prefix_and_name<'a>(
        &self,
        prefix: &mut Vec<String>,
        name: &'a Name,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        prefix.push(name.to_string());
        self.fmt_with_prefix(prefix, f)?;
        prefix.pop();
        Ok(())
    }

    fn fmt_type_with_label(
        prefix: &[String],
        name: &Name,
        ty: &Type,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "_.{}{}{}: {}",
            prefix.iter().join("."),
            if prefix.is_empty() { "" } else { "." },
            name,
            ty
        )
    }
}

impl Visit<Type> for NarrowedAttrs {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        let attrs = &self.0;
        attrs.values().for_each(|value| {
            value.visit(f);
        })
    }
}

impl VisitMut<Type> for NarrowedAttrs {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        let attrs = &mut self.0;
        attrs.values_mut().for_each(|value| {
            value.visit_mut(f);
        })
    }
}

impl Display for NarrowedAttrs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_prefix(&mut Vec::new(), f)
    }
}

/// A `NarrowedAttr` represents a single attribute within a tree of narrowed
/// attributes. The attribute itself may or may not be narrowed, and it may or
/// may not have any sub-attributes (but at least one must be the case, or it
/// wouldn't be in the tree at all)
#[derive(Debug, Clone, Visit, VisitMut, PartialEq, Eq, TypeEq)]
enum NarrowedAttr {
    /// This attribute is narrowed, and has no narrowed sub-attributes (Leaf)
    Leaf(Type),
    /// This attribute is narrowed, and has one or more narrowed sub-attributes (WithRoot)
    WithRoot(Type, NarrowedAttrs),
    /// This attribute is not narrowed, and has one or more narrowed sub-attributes (WithoutRoot)
    WithoutRoot(NarrowedAttrs),
}

impl NarrowedAttr {
    fn new(names: &[Name], ty: Type) -> Self {
        match names {
            [] => Self::Leaf(ty),
            [name, more_names @ ..] => {
                Self::WithoutRoot(NarrowedAttrs::of_narrow((*name).clone(), more_names, ty))
            }
        }
    }

    fn add_narrow(&mut self, names: &[Name], narrowed_ty: Type) {
        // Take ownership of self so we can destructure it and potentially change enum variants.
        let mut current = NarrowedAttr::Leaf(Type::None);
        mem::swap(self, &mut current);
        *self = current.with_narrow(names, narrowed_ty)
    }

    fn with_narrow(self, names: &[Name], narrowed_ty: Type) -> Self {
        match names {
            [] => {
                // We are setting a narrow at the current node (potentially overriding an existing narrow; it is
                // up to callers to make sure this works correctly, we just take what was given).
                match self {
                    Self::Leaf(_) => Self::Leaf(narrowed_ty),
                    Self::WithRoot(_, attrs) | Self::WithoutRoot(attrs) => {
                        Self::WithRoot(narrowed_ty, attrs)
                    }
                }
            }
            [name, more_names @ ..] => {
                // We are setting a narrow in a subtree. We need to preserve any existing tree.
                match self {
                    Self::Leaf(root_ty) => {
                        let attrs =
                            NarrowedAttrs::of_narrow((*name).clone(), more_names, narrowed_ty);
                        Self::WithRoot(root_ty, attrs)
                    }
                    Self::WithoutRoot(mut attrs) => {
                        attrs.add_narrow((*name).clone(), more_names, narrowed_ty);
                        Self::WithoutRoot(attrs)
                    }
                    Self::WithRoot(root_ty, mut attrs) => {
                        attrs.add_narrow((*name).clone(), more_names, narrowed_ty);
                        Self::WithRoot(root_ty, attrs)
                    }
                }
            }
        }
    }

    fn update_for_assignment(&mut self, name: &Name, more_names: &[Name], ty: Option<Type>) {
        match self {
            Self::Leaf(..) => {}
            Self::WithoutRoot(attrs) | Self::WithRoot(_, attrs) => {
                attrs.update_for_assignment(name, more_names, ty);
            }
        }
    }

    fn join(branches: Vec<Self>, union_types: &impl Fn(Vec<Type>) -> Type) -> Option<Self> {
        fn monadic_push_option<T>(acc: &mut Option<Vec<T>>, item: Option<T>) {
            match item {
                None => *acc = None,
                Some(item) => {
                    if let Some(acc) = acc {
                        acc.push(item)
                    }
                }
            };
        }
        let mut ty_branches = Some(Vec::with_capacity(branches.len()));
        let mut attrs_branches = Some(Vec::with_capacity(branches.len()));
        for attr in branches {
            let (ty, attrs) = match attr {
                Self::WithRoot(ty, attrs) => (Some(ty), Some(attrs)),
                Self::Leaf(ty) => (Some(ty), None),
                Self::WithoutRoot(attrs) => (None, Some(attrs)),
            };
            monadic_push_option(&mut ty_branches, ty);
            monadic_push_option(&mut attrs_branches, attrs);
            if let (None, None) = (&ty_branches, &attrs_branches) {
                // Not needed for correctness, but saves some work.
                return None;
            }
        }
        let ty = ty_branches.map(union_types);
        let attrs = attrs_branches.and_then(|attrs_branches| {
            NarrowedAttrs::join(attrs_branches.iter().collect(), union_types)
        });
        match (ty, attrs) {
            (None, None) => None,
            (Some(ty), None) => Some(Self::Leaf(ty)),
            (Some(ty), Some(attrs)) => Some(Self::WithRoot(ty, attrs)),
            (None, Some(attrs)) => Some(Self::WithoutRoot(attrs)),
        }
    }
}

#[cfg(test)]
mod tests {

    use ruff_python_ast::name::Name;
    use vec1::Vec1;

    use crate::types::class::ClassType;
    use crate::types::class::TArgs;
    use crate::types::display::tests::fake_class;
    use crate::types::type_info::TypeInfo;
    use crate::types::types::Type;

    fn fake_class_type(class_name: &str) -> Type {
        Type::ClassType(ClassType::new(
            fake_class(class_name, "class_defs_module", 5, Vec::new()),
            TArgs::default(),
        ))
    }

    #[test]
    fn test_type_info_one_level_only() {
        let x = || Name::new_static("x");
        let y = || Name::new_static("y");
        let mut type_info = TypeInfo::of_ty(fake_class_type("Foo"));
        assert_eq!(type_info.to_string(), "Foo");
        type_info.add_narrow(&Vec1::new(x()), fake_class_type("Bar"));
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar)");
        type_info.add_narrow(&Vec1::new(y()), fake_class_type("Baz"));
        assert_eq!(type_info.to_string(), "Foo (_.x: Bar, _.y: Baz)");
    }

    #[test]
    fn test_type_info_adding_sub_attributes() {
        let x = || Name::new_static("x");
        let y = || Name::new_static("y");
        let z = || Name::new_static("z");
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
        let x = || Name::new_static("x");
        let y = || Name::new_static("y");
        let z = || Name::new_static("z");
        let w = || Name::new_static("w");
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
        let x = || Name::new_static("x");
        let y = || Name::new_static("y");
        let z = || Name::new_static("z");
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
        let type_info = TypeInfo::join(Vec::new(), &|ts| {
            if ts.is_empty() {
                fake_class_type("Never")
            } else {
                fake_class_type("FakeUnionType")
            }
        });
        assert_eq!(type_info.to_string(), "Never");
    }
}
