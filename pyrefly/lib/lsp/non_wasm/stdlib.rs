/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use pyrefly_config::config::ConfigFile;
use pyrefly_util::arc_id::ArcId;

use crate::lsp::non_wasm::type_error_display_status::TypeErrorDisplayStatus;

pub fn should_show_stdlib_error(
    config: &ArcId<ConfigFile>,
    type_error_status: TypeErrorDisplayStatus,
    path: &Path,
) -> bool {
    matches!(
        type_error_status,
        TypeErrorDisplayStatus::EnabledInIdeConfig
    ) || (config.project_includes.covers(path) && !config.project_excludes.covers(path))
}
