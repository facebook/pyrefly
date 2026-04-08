/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::hash::DefaultHasher;
use std::hash::Hash as _;
use std::hash::Hasher as _;
use std::path::Path;

use pyrefly_config::error_kind::Severity;
use pyrefly_util::prelude::SliceExt;
use serde::Deserialize;
use serde::Serialize;

use crate::error::error::Error;

pub(crate) fn severity_to_str(severity: Severity) -> String {
    match severity {
        Severity::Ignore => "info".to_owned(), // This is the lowest valid severity level
        Severity::Info => "info".to_owned(),
        Severity::Warn => "minor".to_owned(),
        Severity::Error => "major".to_owned(),
    }
}

/// The structure for a CodeClimate issue
/// <https://github.com/codeclimate/platform/blob/master/spec/analyzers/SPEC.md#issues>.
///
/// Used to serialize errors for platforms that expect the CodeClimate format, like GitLab CI/CD's
/// Code Quality report artifact <https://docs.gitlab.com/ci/testing/code_quality>.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct CodeClimateIssue {
    #[serde(rename = "type")]
    issue_type: String,
    check_name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<CodeClimateIssueContent>,
    categories: Vec<String>,
    location: CodeClimateIssueLocation,
    severity: String,
    fingerprint: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct CodeClimateIssueContent {
    body: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct CodeClimateIssueLocation {
    path: String,
    positions: CodeClimateIssuePositions,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct CodeClimateIssuePositions {
    begin: CodeClimateIssuePosition,
    end: CodeClimateIssuePosition,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct CodeClimateIssuePosition {
    line: u32,
    column: u32,
}

impl CodeClimateIssue {
    pub fn from_error(relative_to: &Path, error: &Error) -> Self {
        let error_range = error.display_range();
        let error_path = error.path().as_path();

        let mut hasher = DefaultHasher::new();
        error.hash(&mut hasher);
        let fingerprint = format!("{:x}", hasher.finish());

        Self {
            issue_type: "issue".to_owned(),
            check_name: format!("Pyrefly/{}", error.error_kind()),
            description: error.msg_header().to_owned(),
            content: error.msg_details().map(|details| CodeClimateIssueContent {
                body: details.to_owned(),
            }),
            categories: vec!["Bug Risk".to_owned()],
            location: CodeClimateIssueLocation {
                path: error_path
                    .strip_prefix(relative_to)
                    .unwrap_or(error_path)
                    .to_string_lossy()
                    .into_owned(),
                positions: CodeClimateIssuePositions {
                    begin: CodeClimateIssuePosition {
                        line: error_range.start.line_within_cell().get(),
                        column: error_range.start.column().get(),
                    },
                    end: CodeClimateIssuePosition {
                        line: error_range.end.line_within_cell().get(),
                        column: error_range.end.column().get(),
                    },
                },
            },
            severity: severity_to_str(error.severity()),
            fingerprint,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(transparent)]
pub struct CodeClimateIssues(Vec<CodeClimateIssue>);

impl CodeClimateIssues {
    pub fn from_errors(relative_to: &Path, errors: &[Error]) -> Self {
        Self(errors.map(|e| CodeClimateIssue::from_error(relative_to, e)))
    }
}
