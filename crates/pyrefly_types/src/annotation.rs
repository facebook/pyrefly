/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the `Annotation` type.
//! See <https://typing.readthedocs.io/en/latest/spec/annotations.html#type-and-annotation-expressions>

use std::fmt;
use std::fmt::Display;

use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::intersperse_iter;

use crate::keywords::RangeConstraints;
use crate::types::AnyStyle;
use crate::types::Substitution;
use crate::types::Type;

#[derive(Debug, Clone, Default, VisitMut, TypeEq, PartialEq, Eq)]
pub struct Annotation {
    pub qualifiers: Vec<Qualifier>,
    pub ty: Option<Type>,
    pub range_constraints: RangeConstraints,
}

impl Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.qualifiers.is_empty() {
            match &self.ty {
                Some(ty) => write!(f, "{ty}"),
                None => write!(f, "_"),
            }
        } else {
            write!(f, "{}", intersperse_iter("[", || self.qualifiers.iter()))?;
            if let Some(ty) = &self.ty {
                write!(f, "[{ty}]")?;
            }
            write!(f, "{}", "]".repeat(self.qualifiers.len() - 1))
        }
    }
}

impl Annotation {
    pub fn new_type(ty: Type) -> Self {
        Self {
            qualifiers: Vec::new(),
            ty: Some(ty),
            range_constraints: RangeConstraints::default(),
        }
    }

    pub fn get_type(&self) -> &Type {
        self.ty.as_ref().unwrap_or(&Type::Any(AnyStyle::Implicit))
    }

    pub fn is_class_var(&self) -> bool {
        self.has_qualifier(&Qualifier::ClassVar)
    }

    pub fn is_final(&self) -> bool {
        self.has_qualifier(&Qualifier::Final)
    }

    pub fn is_init_var(&self) -> bool {
        self.has_qualifier(&Qualifier::InitVar)
    }

    pub fn has_qualifier(&self, qualifier: &Qualifier) -> bool {
        self.qualifiers.iter().any(|q| q == qualifier)
    }

    pub fn substitute_with(self, substitution: Substitution) -> Self {
        Self {
            qualifiers: self.qualifiers,
            ty: self.ty.map(|ty| substitution.substitute_into(ty)),
            range_constraints: self.range_constraints,
        }
    }
}

#[derive(Debug, Clone, VisitMut, TypeEq, PartialEq, Eq, Display)]
pub enum Qualifier {
    Required,
    NotRequired,
    ReadOnly,
    ClassVar,
    Final,
    InitVar,
    Annotated,
    TypeAlias,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(
            Annotation {
                qualifiers: Vec::new(),
                ty: Some(Type::None),
                range_constraints: RangeConstraints::default(),
            }
            .to_string(),
            "None"
        );
        assert_eq!(
            Annotation {
                qualifiers: vec![Qualifier::Required, Qualifier::ReadOnly],
                ty: None,
                range_constraints: RangeConstraints::default(),
            }
            .to_string(),
            "Required[ReadOnly]"
        );
        assert_eq!(
            Annotation {
                qualifiers: vec![Qualifier::Required, Qualifier::ReadOnly],
                ty: Some(Type::LiteralString),
                range_constraints: RangeConstraints::default(),
            }
            .to_string(),
            "Required[ReadOnly[LiteralString]]"
        );
    }
}
