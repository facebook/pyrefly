/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde::Deserialize;
use serde::Serialize;

use crate::config::error::ErrorDisplayConfig;
use crate::config::util::ExtraConfigs;
use crate::module::wildcard::ModuleWildcard;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
pub struct ConfigBase {
    /// Errors to silence (or not) when printing errors.
    #[serde(default)]
    pub errors: Option<ErrorDisplayConfig>,

    /// String-prefix-matched names of modules from which import errors should be ignored
    /// and the module should always be replaced with `typing.Any`
    #[serde(default)]
    pub replace_imports_with_any: Option<Vec<ModuleWildcard>>,

    /// analyze function body and infer return type
    #[serde(default)]
    pub skip_untyped_functions: Option<bool>,

    /// Whether to ignore type errors in generated code. By default this is disabled.
    /// Generated code is defined as code that contains the marker string `@` immediately followed by `generated`.
    #[serde(default)]
    pub ignore_errors_in_generated_code: Option<bool>,

    /// Any unknown config items
    #[serde(default, flatten)]
    pub extras: ExtraConfigs,
}
