/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;

/// Represents the specific reason why a field is read-only, to provide better error messages
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeEq, VisitMut)]
pub enum ReadOnlyReason {
    /// Field is marked with a `Final` qualifier
    Final,
    /// Field is marked with a `ReadOnly` qualifier
    ReadOnlyQualifier,
    /// Field is a frozen dataclass member
    FrozenDataclass,
    /// Field is a NamedTuple member
    NamedTuple,
    /// Field is a ClassVar
    ClassVar,
}

impl ReadOnlyReason {
    pub fn error_message(&self) -> String {
        match self {
            ReadOnlyReason::Final => "This field is marked as Final".to_owned(),
            ReadOnlyReason::ReadOnlyQualifier => "This field is marked as ReadOnly".to_owned(),
            ReadOnlyReason::FrozenDataclass => "This field is a frozen dataclass member".to_owned(),
            ReadOnlyReason::NamedTuple => "This field is a NamedTuple member".to_owned(),
            ReadOnlyReason::ClassVar => {
                "A ClassVar may not be mutated from an instance of the class".to_owned()
            }
        }
    }
}
