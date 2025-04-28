/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;

use crate::error::kind::ErrorKind;

/// Represents overrides for errors to emit when collecting/printing errors.
/// The boolean in the map represents whether the error is enabled or disabled
/// (true = show error, false = don't show error).
/// Not all error kinds are required to be defined in this map. Any that are missing
/// will be treated as `<error-kind> = true`.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(transparent)]
pub struct ErrorDisplayConfig(HashMap<ErrorKind, bool>);

impl ErrorDisplayConfig {
    pub fn new(config: HashMap<ErrorKind, bool>) -> Self {
        Self(config)
    }

    /// Gets whether the given `ErrorKind` is enabled. If the value isn't
    /// found, then assume it should be enabled.
    pub fn is_enabled(&self, kind: ErrorKind) -> bool {
        self.0.get(&kind) != Some(&false)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ErrorConfig<'a> {
    pub display_config: &'a ErrorDisplayConfig,
    pub ignore_errors_in_generated_code: bool,
}

impl<'a> ErrorConfig<'a> {
    pub fn new(
        display_config: &'a ErrorDisplayConfig,
        ignore_errors_in_generated_code: bool,
    ) -> Self {
        Self {
            display_config,
            ignore_errors_in_generated_code,
        }
    }
}
