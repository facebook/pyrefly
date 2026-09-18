/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type-parameter restrictions defined by Pyrefly's experimental shape extensions.
//!
//! The private variants keep shape-specific restrictions out of the stable type-system surface.
//! Generic consumers project a restriction to ordinary types, while shape-specific code uses the
//! narrow accessors needed to implement its specialized inference policy.

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;

use super::FlagDomain;
use super::Restriction;
use crate::heap::TypeHeap;
use crate::shape_index::lower_index_type;
use crate::stdlib::Stdlib;
use crate::types::Type;

/// A type-parameter restriction whose inference policy belongs to the experimental shape system.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ShapeExtensionRestriction(ShapeExtensionRestrictionKind);

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
enum ShapeExtensionRestrictionKind {
    Flag(FlagDomain),
    Index,
}

impl ShapeExtensionRestriction {
    pub(super) fn flag(domain: FlagDomain) -> Self {
        Self(ShapeExtensionRestrictionKind::Flag(domain))
    }

    pub(super) fn index() -> Self {
        Self(ShapeExtensionRestrictionKind::Index)
    }

    fn flag_domain(&self) -> Option<FlagDomain> {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(domain) => Some(domain),
            ShapeExtensionRestrictionKind::Index => None,
        }
    }

    fn is_index(&self) -> bool {
        matches!(self.0, ShapeExtensionRestrictionKind::Index)
    }

    /// Project this restriction to its ordinary type-system upper bound.
    pub fn upper_bound(&self, stdlib: &Stdlib, heap: &TypeHeap) -> Type {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(domain) => domain.as_type(stdlib, heap),
            ShapeExtensionRestrictionKind::Index => stdlib.object().clone().to_type(),
        }
    }

    /// Materialize the nonempty members of this restriction's ordinary upper bound.
    ///
    /// Current consumers require every member to have both an attribute base and a class-instance
    /// base. New restriction kinds must preserve those invariants or give those consumers a more
    /// precise projection.
    pub fn upper_bound_members(&self, stdlib: &Stdlib) -> Vec<Type> {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(domain) => domain.types(stdlib),
            ShapeExtensionRestrictionKind::Index => vec![stdlib.object().clone().to_type()],
        }
    }

    /// Return the class names represented by this restriction's ordinary upper bound.
    pub fn upper_bound_class_names(&self) -> Vec<&'static str> {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(domain) => domain.class_names(),
            ShapeExtensionRestrictionKind::Index => vec!["builtins.object"],
        }
    }

    /// Check whether a value can specialize this restriction without losing its literal identity.
    pub fn accepts_specialization(
        &self,
        ty: &Type,
        is_str_subclass: impl FnMut(&Type) -> bool,
    ) -> bool {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(domain) => {
                domain.accepts_with_str_subclasses(ty, is_str_subclass)
            }
            ShapeExtensionRestrictionKind::Index => lower_index_type(ty).is_valid(),
        }
    }

    /// Whether a default expression is a runtime value rather than a type expression.
    ///
    /// Binding consults `TypeParameterBound::infer_default_as_value` before a semantic restriction
    /// exists; later consumers use this resolved counterpart without retaining syntax.
    pub fn infer_default_as_value(&self) -> bool {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(_) | ShapeExtensionRestrictionKind::Index => true,
        }
    }

    /// The source-level restriction name used in diagnostics.
    pub fn kind_name(&self) -> &'static str {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(_) => "Flag",
            ShapeExtensionRestrictionKind::Index => "Index",
        }
    }

    fn uses_direct_value_source(&self) -> bool {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(_) | ShapeExtensionRestrictionKind::Index => true,
        }
    }
}

impl Restriction {
    /// Construct the shape-extension restriction for a `Flag` domain.
    pub fn flag(domain: FlagDomain) -> Self {
        Self::ShapeExtension(ShapeExtensionRestriction::flag(domain))
    }

    /// Construct the shape-extension restriction for an index value.
    pub fn index() -> Self {
        Self::ShapeExtension(ShapeExtensionRestriction::index())
    }

    /// Return the `Flag` domain when this is a `Flag` restriction.
    pub fn flag_domain(&self) -> Option<FlagDomain> {
        match self {
            Self::ShapeExtension(extension) => extension.flag_domain(),
            Self::Constraints(_) | Self::Bound(_) | Self::Unrestricted => None,
        }
    }

    /// Whether this is a shape-extension `Flag` restriction.
    pub fn is_flag(&self) -> bool {
        self.flag_domain().is_some()
    }

    /// Whether this is a shape-extension `Index` restriction.
    pub fn is_index(&self) -> bool {
        matches!(self, Self::ShapeExtension(extension) if extension.is_index())
    }

    /// Whether this restriction needs one direct runtime parameter as its specialization source.
    pub fn uses_direct_value_source(&self) -> bool {
        matches!(self, Self::ShapeExtension(extension) if extension.uses_direct_value_source())
    }
}

impl Display for ShapeExtensionRestriction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ShapeExtensionRestrictionKind::Flag(domain) => write!(f, "Flag[{domain}]"),
            ShapeExtensionRestrictionKind::Index => write!(f, "Index"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use pyrefly_python::sys_info::PythonVersion;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::PrecomputedTParams;
    use crate::type_var::FlagMember;

    fn fake_class(module_name: ModuleName, name: &Name) -> Class {
        let module = Module::new(
            module_name,
            ModulePath::filesystem(PathBuf::from("stdlib.pyi")),
            Arc::new(String::new()),
        );
        Class::new(
            ClassDefIndex(0),
            Identifier::new(name.clone(), TextRange::empty(TextSize::new(0))),
            NestingContext::toplevel(),
            module,
            PrecomputedTParams::NotGeneric,
            false,
        )
    }

    #[test]
    fn flag_restriction_boundary() {
        let domain = FlagDomain::of(FlagMember::Int);
        let restriction = Restriction::flag(domain);
        assert!(restriction.is_flag());
        assert_eq!(restriction.flag_domain(), Some(domain));

        let Restriction::ShapeExtension(extension) = restriction else {
            unreachable!("a Flag domain creates a shape-extension restriction")
        };
        assert_eq!(extension.to_string(), "Flag[int]");
        assert_eq!(extension.kind_name(), "Flag");
        assert!(extension.infer_default_as_value());

        let stdlib = Stdlib::new(
            PythonVersion::default(),
            &|module, name| Some((fake_class(module, name), None)),
            &|_, _| None,
        );
        let members = extension.upper_bound_members(&stdlib);
        assert_eq!(members.len(), 1);
        assert!(matches!(&members[0], Type::ClassType(cls) if cls.is_builtin("int")));
    }

    #[test]
    fn index_restriction_projects_to_object() {
        let restriction = Restriction::index();
        assert!(restriction.is_index());

        let Restriction::ShapeExtension(extension) = restriction else {
            unreachable!("an Index bound creates a shape-extension restriction")
        };
        assert_eq!(extension.to_string(), "Index");
        assert_eq!(extension.kind_name(), "Index");
        assert!(extension.infer_default_as_value());
        assert_eq!(extension.upper_bound_class_names(), vec!["builtins.object"]);

        let stdlib = Stdlib::new(
            PythonVersion::default(),
            &|module, name| Some((fake_class(module, name), None)),
            &|_, _| None,
        );
        let members = extension.upper_bound_members(&stdlib);
        assert_eq!(members.len(), 1);
        assert!(matches!(&members[0], Type::ClassType(cls) if cls.is_builtin("object")));
    }
}
