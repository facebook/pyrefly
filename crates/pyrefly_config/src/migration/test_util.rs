/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::migration::pyright::PyrightConfig;
use crate::migration::pyright::RuleOverrides;

/// Helper function to create a default PyrightConfig with all fields set to None or empty.
pub fn default_pyright_config() -> PyrightConfig {
    PyrightConfig {
        project_includes: None,
        project_excludes: None,
        search_path: None,
        python_platform: None,
        python_version: None,
        errors: RuleOverrides {
            report_missing_imports: None,
            report_missing_module_source: None,
        },
        execution_environments: vec![],
    }
}
