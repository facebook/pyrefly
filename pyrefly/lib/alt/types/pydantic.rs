/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;

use crate::binding::pydantic::PydanticAliasGenerator;

/// Options that control whether fields are populated by their names or aliases.
/// `None` means the option was not configured and its Pydantic default applies.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, Default)]
pub struct PydanticValidationFlags {
    pub validate_by_name: Option<bool>,
    pub validate_by_alias: Option<bool>,
}

impl PydanticValidationFlags {
    pub fn validate_by_name(&self) -> bool {
        self.validate_by_name.unwrap_or(false)
    }

    pub fn validate_by_alias(&self) -> bool {
        self.validate_by_alias.unwrap_or(true)
    }
}

/// Configuration for a Pydantic model.
/// For pydantic dataclasses, `frozen`, `extra`, and `strict` are `None` because
/// they come from decorator arguments via dataclass_transform, not from this config.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, VisitMut, Default)]
pub struct PydanticConfig {
    pub frozen: Option<bool>,
    pub validation_flags: PydanticValidationFlags,
    pub validation_alias_generator: Option<PydanticAliasGenerator>,
    pub extra: Option<bool>,
    pub strict: Option<bool>,
    pub pydantic_model_kind: PydanticModelKind,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq, VisitMut, Default)]
pub enum PydanticModelKind {
    #[default]
    BaseModel,
    RootModel,
    BaseSettings,
    /// A class decorated with `@pydantic.dataclasses.dataclass`.
    DataClass,
}
