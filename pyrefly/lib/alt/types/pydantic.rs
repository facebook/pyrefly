/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;

/// Flags that control whether a Pydantic model's fields are populated by their names or their aliases.
/// See https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_by_name.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq)]
pub struct PydanticValidationFlags {
    pub validate_by_name: bool,
    pub validate_by_alias: bool,
}

impl Default for PydanticValidationFlags {
    fn default() -> Self {
        Self {
            validate_by_name: false,
            validate_by_alias: true,
        }
    }
}

/// Configuration for a Pydantic model.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, VisitMut, Default)]
pub struct PydanticConfig {
    pub frozen: bool,
    pub validation_flags: PydanticValidationFlags,
    pub extra: bool,
    pub strict: bool,
    pub pydantic_model_kind: PydanticModelKind,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq, VisitMut, Default)]

pub enum PydanticModelKind {
    #[default]
    BaseModel,
    RootModel,
    BaseSettings,
}
