/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;

// TODO Zeina: Extend this structure as we populate more metadata for pydantic classes
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, VisitMut, Default)]
pub struct PydanticMetadata {
    pub frozen: bool,
    pub validation_alias: Option<Box<str>>,
}

impl fmt::Display for PydanticMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PydanticMetadata {{ frozen: {} }}", self.frozen)
    }
}
