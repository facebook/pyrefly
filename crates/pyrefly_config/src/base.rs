/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::ValueEnum;
use pyrefly_python::ignore::Tool;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_set::SmallSet;
use toml::Table;

use crate::error::ErrorDisplayConfig;
use crate::module_wildcard::ModuleWildcard;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Copy, Default)]
#[derive(ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum UntypedDefBehavior {
    #[default]
    CheckAndInferReturnType,
    CheckAndInferReturnAny,
    SkipAndInferReturnAny,
}

/// How to handle when recursion depth limit is exceeded.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Copy, Default)]
#[derive(ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RecursionOverflowHandler {
    /// Return a placeholder type and emit an internal error. Safe for IDE use.
    #[default]
    BreakWithPlaceholder,
    /// Dump debug info to stderr and panic. For debugging stack overflow issues.
    PanicWithDebugInfo,
}

/// Internal configuration struct combining depth limit and handler.
/// Not serialized directly - constructed from flat config fields.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct RecursionLimitConfig {
    /// Maximum recursion depth before triggering overflow protection.
    pub limit: u32,
    /// How to handle when the depth limit is exceeded.
    pub handler: RecursionOverflowHandler,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct ConfigBase {
    /// Errors to silence (or not) when printing errors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errors: Option<ErrorDisplayConfig>,

    /// Consider any ignore (including from other tools) to ignore an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissive_ignores: Option<bool>,

    /// Respect ignore directives from only these tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled_ignores: Option<SmallSet<Tool>>,

    /// Modules from which import errors should be ignored
    /// and the module should always be replaced with `typing.Any`
    #[serde(
        default,
        skip_serializing_if = "crate::util::none_or_empty",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "replace_imports_with_any"
    )]
    pub(crate) replace_imports_with_any: Option<Vec<ModuleWildcard>>,

    /// Modules from which import errors should be
    /// ignored. The module is only replaced with `typing.Any` if it can't be found.
    #[serde(default, skip_serializing_if = "crate::util::none_or_empty")]
    pub(crate) ignore_missing_imports: Option<Vec<ModuleWildcard>>,

    /// How should we handle analyzing and inferring the function signature if it's untyped?
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "untyped_def_behavior"
    )]
    pub untyped_def_behavior: Option<UntypedDefBehavior>,

    /// Whether to disable type errors in language server. By default errors will be shown in IDEs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_type_errors_in_ide: Option<bool>,

    /// Whether to ignore type errors in generated code. By default this is disabled.
    /// Generated code is defined as code that contains the marker string `@` immediately followed by `generated`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "ignore_errors_in_generated_code"
    )]
    pub ignore_errors_in_generated_code: Option<bool>,

    /// Whether to infer empty container types as Any instead of creating type variables.
    /// By default this is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub infer_with_first_use: Option<bool>,

    /// Whether to enable tensor shape type inference.
    /// When enabled, integer literals can be used as type arguments (e.g., Tensor[2, 3]),
    /// and type variables can participate in dimension arithmetic.
    /// By default this is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_shapes: Option<bool>,

    /// Maximum recursion depth before triggering overflow protection.
    /// Set to 0 to disable (default). This helps detect potential stack overflow situations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursion_depth_limit: Option<u32>,

    /// How to handle when recursion depth limit is exceeded.
    /// Only used when `recursion-depth-limit` is set to a non-zero value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursion_overflow_handler: Option<RecursionOverflowHandler>,

    /// Any unknown config items
    #[serde(default, flatten)]
    pub(crate) extras: ExtraConfigs,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(transparent)]
pub(crate) struct ExtraConfigs(pub(crate) Table);

// `Value` types in `Table` might not be `Eq`, but we don't actually care about that w.r.t. `ConfigFile`
impl Eq for ExtraConfigs {}

impl PartialEq for ExtraConfigs {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl ConfigBase {
    pub fn default_for_ide_without_config() -> Self {
        Self {
            disable_type_errors_in_ide: Some(true),
            ..Default::default()
        }
    }

    pub fn get_errors(base: &Self) -> Option<&ErrorDisplayConfig> {
        base.errors.as_ref()
    }

    pub(crate) fn get_replace_imports_with_any(base: &Self) -> Option<&[ModuleWildcard]> {
        base.replace_imports_with_any.as_deref()
    }

    pub(crate) fn get_ignore_missing_imports(base: &Self) -> Option<&[ModuleWildcard]> {
        base.ignore_missing_imports.as_deref()
    }

    pub fn get_untyped_def_behavior(base: &Self) -> Option<UntypedDefBehavior> {
        base.untyped_def_behavior
    }

    pub fn get_disable_type_errors_in_ide(base: &Self) -> Option<bool> {
        base.disable_type_errors_in_ide
    }

    pub fn get_ignore_errors_in_generated_code(base: &Self) -> Option<bool> {
        base.ignore_errors_in_generated_code
    }

    pub fn get_infer_with_first_use(base: &Self) -> Option<bool> {
        base.infer_with_first_use
    }

    pub fn get_tensor_shapes(base: &Self) -> Option<bool> {
        base.tensor_shapes
    }

    pub fn get_enabled_ignores(base: &Self) -> Option<&SmallSet<Tool>> {
        base.enabled_ignores.as_ref()
    }

    /// Get the recursion limit configuration, if enabled.
    /// Returns None if recursion_depth_limit is not set or is 0.
    pub fn get_recursion_limit_config(base: &Self) -> Option<RecursionLimitConfig> {
        base.recursion_depth_limit.and_then(|limit| {
            if limit == 0 {
                None
            } else {
                Some(RecursionLimitConfig {
                    limit,
                    handler: base
                        .recursion_overflow_handler
                        .unwrap_or(RecursionOverflowHandler::BreakWithPlaceholder),
                })
            }
        })
    }
}
