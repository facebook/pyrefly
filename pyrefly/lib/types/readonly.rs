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

impl ReadOnlyReason {
    /// Generate a specific error message for why a field cannot be assigned to.
    pub fn to_assignment_error_msg(&self, attr_name: &str, class_name: &str) -> String {
        match self {
            ReadOnlyReason::Final => {
                format!("Cannot assign to attribute `{attr_name}` of class `{class_name}`: field is marked as `Final`")
            }
            ReadOnlyReason::ReadOnlyAnnotation => {
                format!("Cannot assign to attribute `{attr_name}` of class `{class_name}`: field is marked as `ReadOnly`")
            }
            ReadOnlyReason::FrozenDataclass => {
                format!("Cannot assign to attribute `{attr_name}` of class `{class_name}`: field belongs to a frozen dataclass")
            }
            ReadOnlyReason::NamedTupleField => {
                format!("Cannot assign to attribute `{attr_name}` of class `{class_name}`: namedtuple fields are immutable")
            }
            ReadOnlyReason::Inherited => {
                format!("Cannot assign to attribute `{attr_name}` of class `{class_name}`: field is read-only")
            }
        }
    }

    /// Generate a specific error message for why a field cannot be deleted.
    pub fn to_deletion_error_msg(&self, attr_name: &str, class_name: &str) -> String {
        match self {
            ReadOnlyReason::Final => {
                format!("Key `{attr_name}` in TypedDict `{class_name}` may not be deleted: field is marked as `Final`")
            }
            ReadOnlyReason::ReadOnlyAnnotation => {
                format!("Key `{attr_name}` in TypedDict `{class_name}` may not be deleted: field is marked as `ReadOnly`")
            }
            ReadOnlyReason::FrozenDataclass => {
                format!("Key `{attr_name}` in TypedDict `{class_name}` may not be deleted: field belongs to a frozen dataclass")
            }
            ReadOnlyReason::NamedTupleField => {
                format!("Key `{attr_name}` in TypedDict `{class_name}` may not be deleted: namedtuple fields are immutable")
            }
            ReadOnlyReason::Inherited => {
                format!("Key `{attr_name}` in TypedDict `{class_name}` may not be deleted")
            }
        }
    }
}