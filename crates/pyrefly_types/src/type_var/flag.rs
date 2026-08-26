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
use std::slice;

use pyrefly_derive::TypeEq;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;

use crate::heap::TypeHeap;
use crate::literal::Lit;
use crate::simplify::unions;
use crate::stdlib::Stdlib;
use crate::tuple::Tuple;
use crate::type_var::Restriction;
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

/// Whether every element of `tuple` is an integer. `Type::Int` covers symbolic `Int[N]`
/// dimensions, which are legitimate entries in a shape-derived tuple.
fn is_int_tuple(tuple: &Tuple) -> bool {
    fn is_int(ty: &Type) -> bool {
        match ty {
            Type::Any(_) | Type::Int(_) => true,
            Type::ClassType(cls) => cls.is_builtin("int"),
            Type::Literal(lit) => matches!(lit.value, Lit::Int(_)),
            Type::Union(union) => union.members.iter().all(is_int),
            Type::Quantified(q) => {
                matches!(q.restriction(), Restriction::Flag(d) if d.is_subset_of(FlagDomain::of(FlagMember::Int)))
            }
            Type::TypeVar(tv) => {
                matches!(tv.restriction(), Restriction::Flag(d) if d.is_subset_of(FlagDomain::of(FlagMember::Int)))
            }
            _ => false,
        }
    }

    match tuple {
        Tuple::Concrete(elements) => elements.iter().all(is_int),
        Tuple::Unbounded(element) => is_int(element),
        Tuple::Unpacked(unpacked) => {
            let (prefix, middle, suffix) = unpacked.parts();
            prefix.iter().all(is_int)
                && suffix.iter().all(is_int)
                && match middle {
                    Type::Any(_) => true,
                    Type::Tuple(tuple) => is_int_tuple(tuple),
                    _ => false,
                }
        }
    }
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

    /// The source spelling this member is written as inside `Flag[...]`.
    fn source_spelling(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Bool => "bool",
            Self::Str => "str",
            Self::Tuple => "tuple[int, ...]",
            Self::NoneType => "None",
        }
    }

    /// The member this type spells, when written as a `Flag[...]` domain member.
    fn from_type(ty: &Type) -> Option<Self> {
        match ty {
            Type::ClassType(cls) if cls.is_builtin("int") => Some(Self::Int),
            Type::ClassType(cls) if cls.is_builtin("bool") => Some(Self::Bool),
            Type::ClassType(cls) if cls.is_builtin("str") => Some(Self::Str),
            Type::None => Some(Self::NoneType),
            Type::Tuple(Tuple::Unbounded(element)) if matches!(element.as_ref(), Type::ClassType(cls) if cls.is_builtin("int")) => {
                Some(Self::Tuple)
            }
            _ => None,
        }
    }

    fn accepts(self, ty: &Type) -> bool {
        match (self, ty) {
            (Self::Int, Type::ClassType(cls)) => cls.is_builtin("int"),
            (Self::Bool, Type::ClassType(cls)) => cls.is_builtin("bool"),
            (Self::Str, Type::ClassType(cls)) => cls.is_builtin("str"),
            (Self::Tuple, Type::ClassType(cls)) => cls.is_builtin("tuple"),
            (Self::Tuple, Type::Tuple(tuple)) => is_int_tuple(tuple),
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
            Self::Tuple => stdlib.tuple(stdlib.int().clone().to_type()).to_type(),
            Self::NoneType => stdlib.none_type().clone().to_type(),
        }
    }
}

impl Display for FlagMember {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.source_spelling())
    }
}

/// The set of builtin types a `Flag` type parameter ranges over.
///
/// Invariant: a domain is never empty. `of` and `join` are the only constructors and both
/// yield at least one member, so materialization may assume a member exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct FlagDomain {
    integer: bool,
    boolean: bool,
    string: bool,
    tuple: bool,
    none: bool,
}

impl FlagDomain {
    /// Parses a nonempty union of supported `Flag` domain members.
    pub fn from_type(ty: &Type) -> Option<Self> {
        let members = match ty {
            Type::Union(union) => union.members.as_slice(),
            _ => slice::from_ref(ty),
        };
        let mut members = members.iter();
        let first = Self::of(FlagMember::from_type(members.next()?)?);
        members.try_fold(first, |domain, member| {
            FlagMember::from_type(member).map(|member| domain.join(Self::of(member)))
        })
    }

    /// Accepts the declared domain exactly, rather than applying Python subtyping.
    pub fn accepts(self, ty: &Type) -> bool {
        if ty.is_any() {
            return true;
        }
        match ty {
            Type::Quantified(q) => {
                matches!(q.restriction(), Restriction::Flag(x) if x.is_subset_of(self))
            }
            Type::TypeVar(tv) => {
                matches!(tv.restriction(), Restriction::Flag(x) if x.is_subset_of(self))
            }
            Type::Union(union) => union.members.iter().all(|member| self.accepts(member)),
            _ => self.members().any(|member| member.accepts(ty)),
        }
    }

    /// Accepts a literal scalar, `None`, or a tuple of integer literals.
    pub fn accepts_literal(self, ty: &Type) -> bool {
        match ty {
            Type::Literal(_) | Type::None => self.accepts(ty),
            Type::Tuple(Tuple::Concrete(elements)) => {
                self.contains(FlagMember::Tuple)
                    && elements.iter().all(|element| {
                        matches!(element, Type::Literal(lit) if matches!(lit.value, Lit::Int(_)))
                    })
            }
            _ => false,
        }
    }

    pub fn of(member: FlagMember) -> Self {
        let mut domain = Self {
            integer: false,
            boolean: false,
            string: false,
            tuple: false,
            none: false,
        };
        match member {
            FlagMember::Int => domain.integer = true,
            FlagMember::Bool => domain.boolean = true,
            FlagMember::Str => domain.string = true,
            FlagMember::Tuple => domain.tuple = true,
            FlagMember::NoneType => domain.none = true,
        }
        domain
    }

    /// Least upper bound: the domain admitting everything either side admits.
    pub fn join(self, other: Self) -> Self {
        Self {
            integer: self.integer || other.integer,
            boolean: self.boolean || other.boolean,
            string: self.string || other.string,
            tuple: self.tuple || other.tuple,
            none: self.none || other.none,
        }
    }

    /// Whether everything this domain admits is also admitted by `other`.
    pub fn is_subset_of(self, other: Self) -> bool {
        self.members().all(|member| other.contains(member))
    }

    pub fn contains(self, member: FlagMember) -> bool {
        match member {
            FlagMember::Int => self.integer,
            FlagMember::Bool => self.boolean,
            FlagMember::Str => self.string,
            FlagMember::Tuple => self.tuple,
            FlagMember::NoneType => self.none,
        }
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
                write!(f, " | ")?;
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
    fn flag_singleton_metadata() {
        for (member, display, class_name) in [
            (FlagMember::Int, "int", "builtins.int"),
            (FlagMember::Bool, "bool", "builtins.bool"),
            (FlagMember::Str, "str", "builtins.str"),
            (FlagMember::Tuple, "tuple[int, ...]", "builtins.tuple"),
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
    fn flag_join_is_deterministic() {
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

        assert_eq!(
            joined.to_string(),
            "int | bool | str | tuple[int, ...] | None"
        );
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

    #[test]
    fn flag_tuple_is_a_member() {
        let tuple_only = FlagDomain::of(FlagMember::Tuple);
        assert_eq!(tuple_only.class_names(), vec!["builtins.tuple"]);
        assert!(!tuple_only.contains(FlagMember::Int));

        let scalar_only = FlagDomain::of(FlagMember::Int);
        assert!(!scalar_only.contains(FlagMember::Tuple));
        assert!(scalar_only.join(tuple_only).contains(FlagMember::Tuple));
    }

    #[test]
    fn flag_subset_covers_scalars_and_tuples() {
        let int = FlagDomain::of(FlagMember::Int);
        let tuple = FlagDomain::of(FlagMember::Tuple);
        let both = int.join(tuple);

        assert!(int.is_subset_of(both));
        assert!(tuple.is_subset_of(both));
        assert!(both.is_subset_of(both));
        assert!(!both.is_subset_of(int));
        assert!(!tuple.is_subset_of(int));
        assert!(!int.is_subset_of(tuple));
    }
}
