/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use pyrefly_config::error_kind::Severity;
use pyrefly_util::prelude::SliceExt;
use serde::Deserialize;
use serde::Serialize;

use crate::error::error::Error;

fn severity_to_str(severity: Severity) -> String {
    match severity {
        Severity::Ignore => "ignore".to_owned(),
        Severity::Info => "info".to_owned(),
        Severity::Warn => "warn".to_owned(),
        Severity::Error => "error".to_owned(),
    }
}

fn default_severity() -> String {
    "error".to_owned()
}

/// Legacy error structure in Pyre1. Needs to be consistent with the following file:
/// <https://www.internalfb.com/code/fbsource/fbcode/tools/pyre/facebook/arc/lib/error.rs>
///
/// Used to serialize errors in a Pyre1-compatible format.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct LegacyError {
    line: usize,
    pub column: usize,
    stop_line: usize,
    stop_column: usize,
    pub path: String,
    /// This field is no longer used in Pyrefly. It is kept here for Pyre1 backward compatibility.
    code: i32,
    /// The kebab-case name of the error kind.
    pub name: String,
    description: String,
    concise_description: String,
    /// This field is not part of Pyre1 error format. But it's useful for Pyrefly clients
    #[serde(default = "default_severity")]
    severity: String,
    /// Optional notebook cell number for errors in notebook files
    #[serde(skip_serializing_if = "Option::is_none")]
    cell: Option<usize>,
}

impl LegacyError {
    pub fn from_error(relative_to: &Path, error: &Error) -> Self {
        let error_range = error.display_range();
        let error_path = error.path().as_path();
        Self {
            line: error_range.start.line_within_cell().get() as usize,
            column: error_range.start.column().get() as usize,
            stop_line: error_range.end.line_within_cell().get() as usize,
            stop_column: error_range.end.column().get() as usize,
            cell: error_range.start.cell().map(|cell| cell.get() as usize),
            path: error_path
                .strip_prefix(relative_to)
                .unwrap_or(error_path)
                .to_string_lossy()
                .into_owned(),
            // -2 is chosen because it's an unused error code in Pyre1
            code: -2, // TODO: replace this dummy value
            name: error.error_kind().to_name().to_owned(),
            description: error.msg(),
            concise_description: error.msg_header().to_owned(),
            severity: severity_to_str(error.severity()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct LegacyErrors {
    pub errors: Vec<LegacyError>,
}

impl LegacyErrors {
    pub fn from_errors(relative_to: &Path, errors: &[Error]) -> Self {
        Self {
            errors: errors.map(|e| LegacyError::from_error(relative_to, e)),
        }
    }
}
