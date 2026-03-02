/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;

/// Indicates whether a `Final` variable was initialized at the point of declaration
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, TypeEq, Visit, VisitMut, PartialOrd, Ord
)]
pub enum IsFinalVariableInitialized {
    Yes,
    No,
}

/// Represents the specific reason why a field is read-only, to provide better error messages
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, TypeEq, Visit, VisitMut, PartialOrd, Ord
)]
pub enum ReadOnlyReason {
    /// Field is marked with a `Final` qualifier
    Final(IsFinalVariableInitialized),
    /// Field is marked with a `ReadOnly` qualifier
    ReadOnlyQualifier,
    /// Field is a frozen dataclass member
    FrozenDataclass,
    /// Field is a NamedTuple member
    NamedTuple,
    /// Field is a ClassVar
    ClassVar,
    /// Field is has a type like `type[X]` and is initialized on the body (this includes nested class defs)
    ClassObjectInitializedOnBody,
    /// Field is on a Super instance
    Super,
    /// Field is marked as frozen via a ConfigDict
    PydanticFrozen,
    /// Field is an enum member's value
    EnumMemberValue,
}

impl ReadOnlyReason {
    pub fn error_message(&self) -> String {
        match self {
            ReadOnlyReason::Final(IsFinalVariableInitialized::Yes) => {
                "This field is marked as Final, and an initial value is provided".to_owned()
            }
            ReadOnlyReason::Final(IsFinalVariableInitialized::No) => {
                "This field is marked as Final".to_owned()
            }
            ReadOnlyReason::ReadOnlyQualifier => "This field is marked as ReadOnly".to_owned(),
            ReadOnlyReason::FrozenDataclass => "This field is a frozen dataclass member".to_owned(),
            ReadOnlyReason::NamedTuple => "This field is a NamedTuple member".to_owned(),
            ReadOnlyReason::ClassVar => {
                "A ClassVar may not be mutated from an instance of the class".to_owned()
            }
            ReadOnlyReason::ClassObjectInitializedOnBody => {
                "A class object initialized in the class body is considered read-only".to_owned()
            }
            ReadOnlyReason::Super => {
                "A field accessed through `super()` is considered read-only".to_owned()
            }
            ReadOnlyReason::PydanticFrozen => {
                "This field belongs to a frozen Pydantic model".to_owned()
            }
            ReadOnlyReason::EnumMemberValue => {
                "An enum member's value may not be modified".to_owned()
            }
        }
    }
}
