/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use pyrefly_config::config::ConfigFile;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_util::arc_id::ArcId;
// Re-export from pyrefly_util for use in the LSP server.
pub use pyrefly_util::stdlib::is_python_stdlib_file;

use crate::error::error::Error;
use crate::lsp::non_wasm::server::TypeErrorDisplayStatus;
use crate::state::lsp::DisplayTypeErrors;

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

/// Determines whether an error should be shown based on the display type errors mode.
///
/// When the display mode is set to `ErrorMissingImports`, only import-related errors
/// (MissingImport, MissingSource, MissingSourceForStubs) are shown. For all other
/// display modes, all errors are shown.
pub fn should_show_error_for_display_mode(
    error: &Error,
    display_mode: DisplayTypeErrors,
    display_status: TypeErrorDisplayStatus,
) -> bool {
    match display_mode {
        DisplayTypeErrors::ErrorMissingImports => {
            let error_kind = error.error_kind();
            matches!(
                error_kind,
                ErrorKind::MissingImport
                    | ErrorKind::MissingSource
                    | ErrorKind::MissingSourceForStubs
                    | ErrorKind::ParseError
                    | ErrorKind::InvalidSyntax
            )
        }
        DisplayTypeErrors::ForceOff => false,
        DisplayTypeErrors::Default
            if matches!(display_status, TypeErrorDisplayStatus::NoConfigFile) =>
        {
            let error_kind = error.error_kind();
            matches!(
                error_kind,
                ErrorKind::ParseError
                    | ErrorKind::InvalidSyntax
                    | ErrorKind::InvalidAnnotation
                    | ErrorKind::MissingImport
                    | ErrorKind::UnknownName
            )
        }
        DisplayTypeErrors::Default | DisplayTypeErrors::ForceOn => true,
    }
}

/// Returns the severity override for errors shown in NoConfigFile mode.
/// Errors that would normally be hidden (or shown as Error) are downgraded
/// to Warn so users without a config file still see critical issues without
/// being overwhelmed by error-level diagnostics.
pub fn no_config_severity_override(
    error: &Error,
    display_status: TypeErrorDisplayStatus,
) -> Option<Severity> {
    if !matches!(display_status, TypeErrorDisplayStatus::NoConfigFile) {
        return None;
    }
    let error_kind = error.error_kind();
    if matches!(
        error_kind,
        ErrorKind::InvalidAnnotation | ErrorKind::MissingImport | ErrorKind::UnknownName
    ) {
        Some(Severity::Warn)
    } else {
        None
    }
}
