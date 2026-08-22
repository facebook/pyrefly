/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The representation of `Flag` bounds used by Pyrefly's experimental shape extensions.
//!
//! A normal bounded type variable is generally inferred at its non-literal upper bound. A `Flag`
//! parameter instead preserves the literal supplied by its designated value parameter:
//!
//! ```python
//! def choose[B: Flag[bool]](value: B) -> B: ...
//! reveal_type(choose(True))  # Literal[True]
//! ```
//!
//! Shape stubs can pass that preserved literal to a type-level DSL function, avoiding a separate
//! overload for every option value. `FlagDomain` records the small set of builtin value categories
//! supported by that DSL. Generic type-system code projects the domain to its ordinary upper-bound
//! type whenever it does not need the shape-specific literal-preserving behavior.

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;

use crate::heap::TypeHeap;
use crate::literal::Lit;
use crate::simplify::unions;
use crate::stdlib::Stdlib;
use crate::type_var::Restriction;
use crate::types::AnyStyle;
use crate::types::Type;

/// A member of the builtin universe a `Flag` type parameter can range over. The variant
/// order is the canonical order in which a domain materializes, names, and displays its
/// members, so every consumer sees the same sequence for the same domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FlagMember {
    Int,
    Bool,
    Str,
    Tuple,
    NoneType,
}

impl FlagMember {
    const ALL: [Self; 5] = [
        Self::Int,
        Self::Bool,
        Self::Str,
        Self::Tuple,
        Self::NoneType,
    ];

    /// The fully qualified class name reported for this member. `NoneType` always uses its
    /// canonical `types` spelling, even on versions where typeshed declares it elsewhere, so
    /// that report keys do not shift with the target Python version.
    pub fn class_name(self) -> &'static str {
        match self {
            Self::Int => "builtins.int",
            Self::Bool => "builtins.bool",
            Self::Str => "builtins.str",
            Self::Tuple => "builtins.tuple",
            Self::NoneType => "types.NoneType",
        }
    }

    fn accepts(self, ty: &Type) -> bool {
        match (self, ty) {
            (Self::Int, Type::ClassType(cls)) => cls.is_builtin("int"),
            (Self::Bool, Type::ClassType(cls)) => cls.is_builtin("bool"),
            (Self::Str, Type::ClassType(cls)) => cls.is_builtin("str"),
            (Self::Tuple, Type::ClassType(cls)) => cls.is_builtin("tuple"),
            (Self::Tuple, Type::Tuple(_)) => true,
            (Self::NoneType, Type::None) => true,
            (Self::Int, Type::Int(_)) => true,
            (Self::Int, Type::Literal(lit)) => matches!(lit.value, Lit::Int(_)),
            (Self::Bool, Type::Literal(lit)) => matches!(lit.value, Lit::Bool(_)),
            (Self::Str, Type::Literal(lit)) => matches!(lit.value, Lit::Str(_)),
            _ => false,
        }
    }

    fn as_type(self, stdlib: &Stdlib) -> Type {
        match self {
            Self::Int => stdlib.int().clone().to_type(),
            Self::Bool => stdlib.bool().clone().to_type(),
            Self::Str => stdlib.str().clone().to_type(),
            Self::Tuple => stdlib.tuple(Type::Any(AnyStyle::Implicit)).to_type(),
            Self::NoneType => stdlib.none_type().clone().to_type(),
        }
    }
}

impl Display for FlagMember {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Int => "int",
            Self::Bool => "bool",
            Self::Str => "str",
            Self::Tuple => "tuple",
            Self::NoneType => "None",
        })
    }
}

/// The set of builtin types a `Flag` type parameter ranges over.
///
/// Invariant: a domain is never empty. `of` and `join` are the only constructors and both
/// yield at least one member, so materialization may assume a member exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct FlagDomain {
    /// Bitset indexed by `FlagMember` discriminant. Iteration
    /// order comes from `FlagMember::ALL`, never from construction order.
    members: u8,
}

impl FlagDomain {
    /// Parses the source domains currently supported by `Flag`.
    pub fn from_type(ty: &Type) -> Option<Self> {
        match ty {
            Type::ClassType(cls) if cls.is_builtin("int") => Some(Self::of(FlagMember::Int)),
            Type::ClassType(cls) if cls.is_builtin("bool") => Some(Self::of(FlagMember::Bool)),
            Type::ClassType(cls) if cls.is_builtin("str") => Some(Self::of(FlagMember::Str)),
            _ => None,
        }
    }

    /// Accepts the declared domain exactly, rather than applying Python subtyping.
    pub fn accepts(self, ty: &Type) -> bool {
        if ty.is_any() {
            return true;
        }
        match ty {
            Type::Quantified(q) => matches!(q.restriction(), Restriction::Flag(x) if *x == self),
            Type::TypeVar(tv) => matches!(tv.restriction(), Restriction::Flag(x) if *x == self),
            Type::Union(union) => union.members.iter().all(|member| self.accepts(member)),
            _ => self.members().any(|member| member.accepts(ty)),
        }
    }

    pub fn accepts_literal(self, ty: &Type) -> bool {
        matches!(ty, Type::Literal(_)) && self.accepts(ty)
    }

    pub const fn of(member: FlagMember) -> Self {
        Self {
            members: 1 << member as u8,
        }
    }

    /// Least upper bound: the domain admitting everything either side admits.
    pub fn join(self, other: Self) -> Self {
        Self {
            members: self.members | other.members,
        }
    }

    pub fn contains(self, member: FlagMember) -> bool {
        self.members & (1 << member as u8) != 0
    }

    fn members(self) -> impl Iterator<Item = FlagMember> {
        FlagMember::ALL
            .into_iter()
            .filter(move |member| self.contains(*member))
    }

    /// The domain's members as class types, in canonical order. Never empty.
    pub fn types(self, stdlib: &Stdlib) -> Vec<Type> {
        self.members()
            .map(|member| member.as_type(stdlib))
            .collect()
    }

    /// The fully qualified class names of the domain's members, in canonical order.
    pub fn class_names(self) -> Vec<&'static str> {
        self.members().map(FlagMember::class_name).collect()
    }

    /// The single type covering the whole domain: the member itself when there is only one,
    /// otherwise their union.
    pub fn as_type(self, stdlib: &Stdlib, heap: &TypeHeap) -> Type {
        let types = self.types(stdlib);
        assert!(
            !types.is_empty(),
            "a `Flag` domain always has at least one member"
        );
        unions(types, heap)
    }
}

impl Display for FlagDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, member) in self.members().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{member}")?;
        }
        Ok(())
    }
}

impl Visit<Type> for FlagDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for FlagDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_var::Restriction;

    #[test]
    fn passive_flag_singleton_metadata() {
        for (member, display, class_name) in [
            (FlagMember::Int, "int", "builtins.int"),
            (FlagMember::Bool, "bool", "builtins.bool"),
            (FlagMember::Str, "str", "builtins.str"),
            (FlagMember::Tuple, "tuple", "builtins.tuple"),
            (FlagMember::NoneType, "None", "types.NoneType"),
        ] {
            let domain = FlagDomain::of(member);
            assert_eq!(domain.to_string(), display);
            assert_eq!(domain.class_names(), vec![class_name]);
            assert!(domain.contains(member));
            assert!(Restriction::Flag(domain).is_restricted());
        }
    }

    /// Joining is order-independent and idempotent, and the resulting domain always reports
    /// its members in canonical order rather than join order.
    #[test]
    fn passive_flag_join_is_deterministic() {
        let members = [
            FlagMember::NoneType,
            FlagMember::Tuple,
            FlagMember::Str,
            FlagMember::Bool,
            FlagMember::Int,
        ];
        let joined = members
            .into_iter()
            .map(FlagDomain::of)
            .reduce(FlagDomain::join)
            .unwrap();

        assert_eq!(joined.to_string(), "int, bool, str, tuple, None");
        assert_eq!(
            joined.class_names(),
            vec![
                "builtins.int",
                "builtins.bool",
                "builtins.str",
                "builtins.tuple",
                "types.NoneType",
            ]
        );
        assert!(members.into_iter().all(|member| joined.contains(member)));
        assert_eq!(joined.join(FlagDomain::of(FlagMember::Str)), joined);
        assert_eq!(
            FlagDomain::of(FlagMember::Int).join(FlagDomain::of(FlagMember::Str)),
            FlagDomain::of(FlagMember::Str).join(FlagDomain::of(FlagMember::Int))
        );
    }

    /// Tuple membership uses the same representation as every other domain member.
    #[test]
    fn passive_flag_tuple_is_an_atom() {
        let tuple_only = FlagDomain::of(FlagMember::Tuple);
        assert_eq!(tuple_only.class_names(), vec!["builtins.tuple"]);
        assert!(!tuple_only.contains(FlagMember::Int));

        let scalar_only = FlagDomain::of(FlagMember::Int);
        assert!(!scalar_only.contains(FlagMember::Tuple));
        assert!(scalar_only.join(tuple_only).contains(FlagMember::Tuple));
    }
}
