/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use pyrefly_config::error_kind::Severity;
use pyrefly_util::absolutize::Absolutize;
use pyrefly_util::prelude::SliceExt;
use serde::Deserialize;
use serde::Serialize;

use crate::config::config::BaselineFormat;
use crate::config::config::BaselineMatchingMode;
use crate::error::error::Error;

pub(crate) fn severity_to_str(severity: Severity) -> String {
    match severity {
        Severity::Ignore => "ignore".to_owned(),
        Severity::Info => "info".to_owned(),
        Severity::Warn => "warn".to_owned(),
        Severity::Error => "error".to_owned(),
    }
}

fn default_legacy_severity() -> Severity {
    Severity::Error
}

fn default_baseline_severity() -> Option<Severity> {
    Some(default_legacy_severity())
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
    #[serde(default = "default_legacy_severity")]
    severity: Severity,
    /// Whether the error matched a configured baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    baselined: Option<bool>,
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
                .relativize_from(relative_to)
                .to_string_lossy()
                .replace('\\', "/"), // Normalize Windows backslashes so baseline files are consistent across platforms
            // -2 is chosen because it's an unused error code in Pyre1
            code: -2, // TODO: replace this dummy value
            name: error.error_kind().to_name().to_owned(),
            description: error.msg(),
            concise_description: error.msg_header().to_owned(),
            severity: error.severity(),
            baselined: error.baseline_status().legacy_baselined_flag(),
        }
    }

    pub fn severity(&self) -> Severity {
        self.severity
    }

    pub(crate) fn display_range(&self) -> String {
        if self.line == self.stop_line {
            if self.column == self.stop_column {
                format!("{}:{}", self.line, self.column)
            } else {
                format!("{}:{}-{}", self.line, self.column, self.stop_column)
            }
        } else {
            format!(
                "{}:{}-{}:{}",
                self.line, self.column, self.stop_line, self.stop_column
            )
        }
    }

    pub(crate) fn concise_description(&self) -> &str {
        &self.concise_description
    }

    pub(crate) fn is_baselined(&self) -> bool {
        self.baselined == Some(true)
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

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct BaselineError {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
    pub path: String,
    /// The kebab-case name of the error kind.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concise_description: Option<String>,
    #[serde(
        default = "default_baseline_severity",
        skip_serializing_if = "Option::is_none"
    )]
    severity: Option<Severity>,
    /// Optional notebook cell number for errors in notebook files
    #[serde(skip_serializing_if = "Option::is_none")]
    cell: Option<usize>,
}

impl BaselineError {
    fn from_error(relative_to: &Path, error: &Error) -> Self {
        let error_range = error.display_range();
        let error_path = error.path().as_path();
        Self {
            column: Some(error_range.start.column().get() as usize),
            cell: error_range.start.cell().map(|cell| cell.get() as usize),
            path: error_path
                .relativize_from(relative_to)
                .to_string_lossy()
                .replace('\\', "/"), // Normalize Windows backslashes so baseline files are consistent across platforms
            name: error.error_kind().to_name().to_owned(),
            concise_description: Some(error.msg_header().to_owned()),
            severity: Some(error.severity()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct BaselineErrors {
    pub errors: Vec<BaselineError>,
}

impl BaselineErrors {
    pub fn from_errors(relative_to: &Path, errors: &[Error]) -> Self {
        Self {
            errors: errors.map(|e| BaselineError::from_error(relative_to, e)),
        }
    }

    /// Remove fields that are not needed by a minimal baseline.
    pub fn with_format(
        mut self,
        matching_mode: BaselineMatchingMode,
        format: BaselineFormat,
    ) -> Self {
        if format == BaselineFormat::Minimal {
            for error in &mut self.errors {
                if matching_mode != BaselineMatchingMode::Column {
                    error.column = None;
                }
                if matching_mode != BaselineMatchingMode::ConciseDescription {
                    error.concise_description = None;
                }
                error.severity = None;
                error.cell = None;
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::config::error_kind::ErrorKind;
    use crate::error::error::BaselineStatus;

    #[test]
    fn test_relativize_when_error_is_not_under_relative_to() {
        let module = Module::new(
            ModuleName::from_str("foo"),
            ModulePath::filesystem(PathBuf::from("/repo/libs/foo.py")),
            Arc::new("x = 1\n".to_owned()),
        );
        let error = Error::new(
            module,
            TextRange::new(TextSize::new(0), TextSize::new(1)),
            "err".to_owned(),
            Vec::new(),
            ErrorKind::BadAssignment,
        );
        let legacy = LegacyError::from_error(Path::new("/repo/src"), &error);
        assert_eq!(legacy.path, "../libs/foo.py");
    }

    #[test]
    fn test_baseline_provenance_is_optional() {
        let module = Module::new(
            ModuleName::from_str("foo"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("x = 1\n".to_owned()),
        );
        let error = Error::new(
            module,
            TextRange::new(TextSize::new(0), TextSize::new(1)),
            "err".to_owned(),
            Vec::new(),
            ErrorKind::BadAssignment,
        );

        let without_baseline =
            serde_json::to_value(LegacyError::from_error(Path::new("/repo"), &error)).unwrap();
        assert!(without_baseline.get("baselined").is_none());

        let matched = error.clone().with_baseline_status(BaselineStatus::Matched);
        assert_eq!(
            serde_json::to_value(LegacyError::from_error(Path::new("/repo"), &matched)).unwrap()["baselined"],
            true
        );

        let not_compared = error.with_baseline_status(BaselineStatus::NotCompared);
        assert_eq!(
            serde_json::to_value(LegacyError::from_error(Path::new("/repo"), &not_compared))
                .unwrap()["baselined"],
            false
        );
    }

    #[test]
    fn test_baseline_formats() {
        let module = Module::new(
            ModuleName::from_str("foo"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("x = 1\n".to_owned()),
        );
        let error = Error::new(
            module,
            TextRange::new(TextSize::new(0), TextSize::new(1)),
            "err".to_owned(),
            Vec::new(),
            ErrorKind::BadAssignment,
        );

        let full = BaselineErrors::from_errors(Path::new("/repo"), std::slice::from_ref(&error))
            .with_format(BaselineMatchingMode::Column, BaselineFormat::Full);
        assert_eq!(
            serde_json::to_value(full).unwrap(),
            serde_json::json!({
                "errors": [{
                    "column": 1,
                    "path": "foo.py",
                    "name": "bad-assignment",
                    "concise_description": "err",
                    "severity": "error"
                }]
            })
        );

        let minimal_column =
            BaselineErrors::from_errors(Path::new("/repo"), std::slice::from_ref(&error))
                .with_format(BaselineMatchingMode::Column, BaselineFormat::Minimal);
        assert_eq!(
            serde_json::to_value(minimal_column).unwrap(),
            serde_json::json!({
                "errors": [{
                    "column": 1,
                    "path": "foo.py",
                    "name": "bad-assignment"
                }]
            })
        );

        let minimal_description =
            BaselineErrors::from_errors(Path::new("/repo"), std::slice::from_ref(&error))
                .with_format(
                    BaselineMatchingMode::ConciseDescription,
                    BaselineFormat::Minimal,
                );
        assert_eq!(
            serde_json::to_value(minimal_description).unwrap(),
            serde_json::json!({
                "errors": [{
                    "path": "foo.py",
                    "name": "bad-assignment",
                    "concise_description": "err"
                }]
            })
        );
    }
}
