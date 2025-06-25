/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;

/// Represents the specific reason why a field is read-only.
/// This provides better error messages and more precise type information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeEq, VisitMut)]
pub enum ReadOnlyReason {
    /// Field is marked with `Final` annotation
    #[allow(dead_code)] // TODO: Re-enable usage when Final+__init__ is implemented
    Final,
    /// Field is marked with `ReadOnly` annotation
    ReadOnlyAnnotation,
    /// Field belongs to a frozen dataclass
    FrozenDataclass,
    /// Field belongs to a namedtuple (immutable by design)
    NamedTupleField,
    /// Field inherited read-only status from parent class
    Inherited,
}

impl Display for ReadOnlyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadOnlyReason::Final => write!(f, "Final"),
            ReadOnlyReason::ReadOnlyAnnotation => write!(f, "ReadOnly"),
            ReadOnlyReason::FrozenDataclass => write!(f, "frozen dataclass"),
            ReadOnlyReason::NamedTupleField => write!(f, "namedtuple field"),
            ReadOnlyReason::Inherited => write!(f, "inherited read-only"),
        }
    }
}

impl ReadOnlyReason {}
