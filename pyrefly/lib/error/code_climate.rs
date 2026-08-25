/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::hash::Hash as _;
use std::hash::Hasher as _;
use std::path::Path;

use pyrefly_config::error_kind::Severity;
use pyrefly_util::unix_path::str_path_to_unix_string;
use serde::Deserialize;
use serde::Serialize;
use xxhash_rust::xxh64::Xxh64;

use crate::error::error::Error;

pub(crate) fn severity_to_code_climate_str(severity: Severity) -> Option<&'static str> {
    match severity {
        Severity::Ignore => None,
        Severity::Info => Some("info"),
        Severity::Warn => Some("minor"),
        Severity::Error => Some("major"),
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
    pub fn from_error(relative_to: &Path, error: &Error) -> Option<Self> {
        let severity = severity_to_code_climate_str(error.severity())?.to_owned();

        let error_range = error.display_range();
        let error_path = str_path_to_unix_string(error.path_string_with_fragment(relative_to));

        let category = if error.error_kind().is_directive() {
            "Clarity" // `reveal_type` emits a diagnostic regardless of whether there's a type error or not
        } else {
            "Bug Risk"
        };

        let mut hasher = Xxh64::new(0);
        error_path.hash(&mut hasher);
        error_range.hash(&mut hasher);
        error.msg().hash(&mut hasher);
        let fingerprint = format!("{:016x}", hasher.finish());

        Some(Self {
            issue_type: "issue".to_owned(),
            check_name: format!("pyrefly/{}", error.error_kind().to_name()),
            description: error.msg_header().to_owned(),
            content: error.msg_details().map(|details| CodeClimateIssueContent {
                body: details.to_owned(),
            }),
            categories: vec![category.to_owned()],
            location: CodeClimateIssueLocation {
                path: error_path,
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
            severity,
            fingerprint,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(transparent)]
pub struct CodeClimateIssues(Vec<CodeClimateIssue>);

impl CodeClimateIssues {
    pub fn from_errors(relative_to: &Path, errors: &[Error]) -> Self {
        Self(
            errors
                .iter()
                .filter_map(|e| CodeClimateIssue::from_error(relative_to, e))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_config::error_kind::ErrorKind;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;
    use vec1::Vec1;
    use vec1::vec1;

    use super::*;

    fn sample_error(msg: Vec1<String>) -> Error {
        let module = Module::new(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("x = 1\n".to_owned()),
        );
        Error::new(
            module,
            TextRange::new(TextSize::from(0), TextSize::from(1)),
            msg[0].clone(),
            msg.into_iter().skip(1).collect(),
            ErrorKind::BadAssignment,
        )
    }

    fn sample_directive() -> Error {
        let module = Module::new(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("reveal_type(1)\n".to_owned()),
        );
        Error::new(
            module,
            TextRange::new(TextSize::from(0), TextSize::from(14)),
            "reveal_type(1)".into(),
            vec![],
            ErrorKind::RevealType,
        )
        .with_severity(Severity::Info)
    }

    #[test]
    fn from_error_includes_full_path_and_metadata() {
        let error = sample_error(vec1![
            "Sample error message".to_owned(),
            "Additional details".to_owned()
        ]);
        let code_climate_issue =
            CodeClimateIssue::from_error(Path::new("/repo"), &error).expect("generates issue");
        assert_eq!(code_climate_issue.location.path, "foo.py");
        assert_eq!(code_climate_issue.location.positions.begin.line, 1);
        assert_eq!(code_climate_issue.location.positions.begin.column, 1);
        assert_eq!(code_climate_issue.location.positions.end.line, 1);
        assert_eq!(code_climate_issue.location.positions.end.column, 2);
        assert_eq!(code_climate_issue.severity, "major");
        assert_eq!(code_climate_issue.description, "Sample error message");
        assert_eq!(code_climate_issue.categories, vec!["Bug Risk".to_owned()]);
        assert_eq!(
            code_climate_issue.content.unwrap().body,
            "  Additional details"
        );
    }

    #[test]
    fn from_errors_respects_severity_category_mapping() {
        let warning = sample_error(vec1!["bad".into()]).with_severity(Severity::Warn);
        let notice = sample_error(vec1!["bad".into()]).with_severity(Severity::Info);
        let ignored = sample_error(vec1!["bad".into()]).with_severity(Severity::Ignore);
        let directive = sample_directive();
        assert_eq!(
            CodeClimateIssue::from_error(Path::new("/repo"), &warning)
                .unwrap()
                .severity,
            "minor"
        );
        assert_eq!(
            CodeClimateIssue::from_error(Path::new("/repo"), &notice)
                .unwrap()
                .severity,
            "info"
        );
        assert!(CodeClimateIssue::from_error(Path::new("/repo"), &ignored).is_none());
        assert_eq!(
            CodeClimateIssue::from_error(Path::new("/repo"), &directive)
                .unwrap()
                .categories,
            vec!["Clarity".to_owned()]
        );
    }

    #[test]
    fn from_errors_hash_is_deterministic() {
        let error1 = sample_error(vec1!["bad".into()]);
        let error2 = sample_error(vec1!["bad".into()]);
        let issues1 =
            CodeClimateIssue::from_error(Path::new("/repo"), &error1).expect("produce an issue");
        let issues2 =
            CodeClimateIssue::from_error(Path::new("/repo"), &error2).expect("produce an issue");
        assert_eq!(issues1.fingerprint, issues2.fingerprint);
    }
}
