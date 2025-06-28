/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;

use crate::types::class::Class;
use crate::types::qname::QName;
use crate::types::read_only::ReadOnlyReason;
use crate::types::types::Substitution;
use crate::types::types::TArgs;
use crate::types::types::Type;

#[derive(Clone, Debug, TypeEq, PartialEq, Eq, Hash)]
pub struct TypedDictField {
    pub ty: Type,
    pub required: bool,
    /// The reason this field is read-only. `None` indicates it is read-write.
    pub read_only_reason: Option<ReadOnlyReason>,
}

impl TypedDictField {
    /// Check if this field is read-only.
    pub fn is_read_only(&self) -> bool {
        self.read_only_reason.is_some()
    }

    pub fn substitute(self, substitution: &Substitution) -> Self {
        Self {
            ty: substitution.substitute(self.ty),
            required: self.required,
            read_only_reason: self.read_only_reason,
        }
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TypedDict {
    class: Class,
    args: TArgs,
}

impl TypedDict {
    pub fn new(class: Class, args: TArgs) -> Self {
        Self { class, args }
    }

    pub fn qname(&self) -> &QName {
        self.class.qname()
    }

    pub fn name(&self) -> &Name {
        self.class.name()
    }

    pub fn class_object(&self) -> &Class {
        &self.class
    }

    pub fn targs(&self) -> &TArgs {
        &self.args
    }
}
