/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use pyrefly_util::fs_anyhow;

use crate::error::error::Error;
use crate::error::legacy::LegacyError;
use crate::error::legacy::LegacyErrors;

/// If an error with an exactly matching path, error slug, and starting column exist in the baseline, we ignore it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BaselineKey {
    path: String,
    name: String,
    column: usize,
}

impl From<&LegacyError> for BaselineKey {
    fn from(baseline_error: &LegacyError) -> Self {
        Self {
            path: baseline_error.path.clone(),
            name: baseline_error.name.clone(),
            column: baseline_error.column,
        }
    }
}

impl BaselineKey {
    fn from_error(error: &Error, relative_to: &Path) -> Self {
        let error_path = error.path().as_path();
        Self {
            path: error_path
                .strip_prefix(relative_to)
                .unwrap_or(error_path)
                .to_string_lossy()
                .into_owned(),
            name: error.error_kind().to_name().to_owned(),
            column: error.display_range().start.column().get() as usize,
        }
    }
}

pub struct BaselineProcessor {
    baseline_keys: HashSet<BaselineKey>,
}

impl BaselineProcessor {
    pub fn from_file(baseline_path: &Path) -> Result<Self> {
        let content = fs_anyhow::read_to_string(baseline_path)?;
        let baseline_file: LegacyErrors = serde_json::from_str(&content)?;

        let baseline_keys = baseline_file.errors.iter().map(BaselineKey::from).collect();

        Ok(Self { baseline_keys })
    }

    pub fn matches_baseline(&self, error: &Error, relative_to: &Path) -> bool {
        let key = BaselineKey::from_error(error, relative_to);
        self.baseline_keys.contains(&key)
    }

    /// Baseline suppressions are processed last, after inline and config suppressions
    pub fn process_errors(
        &self,
        shown_errors: &mut Vec<Error>,
        baseline_errors: &mut Vec<Error>,
        relative_to: &Path,
    ) {
        let mut remaining_errors = Vec::new();

        for error in shown_errors.drain(..) {
            if self.matches_baseline(&error, relative_to) {
                baseline_errors.push(error);
            } else {
                remaining_errors.push(error);
            }
        }

        *shown_errors = remaining_errors;
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
    use vec1::vec1;

    use super::*;
    use crate::config::error_kind::ErrorKind;

    #[test]
    fn test_baseline_key_generation() {
        let module = Module::new(
            ModuleName::from_str("test_module"),
            ModulePath::filesystem(PathBuf::from("test/path.py")),
            Arc::new("test content".to_owned()),
        );

        let error = Error::new(
            module,
            TextRange::new(TextSize::new(0), TextSize::new(5)),
            vec1!["Test error message".to_owned()],
            ErrorKind::BadReturn,
        );

        let key = BaselineKey::from_error(&error, Path::new("/root"));

        assert_eq!(key.path, "test/path.py");
        assert_eq!(key.name, "bad-return");
        assert_eq!(key.column, 1);
    }

    #[test]
    fn test_baseline_matching() {
        let baseline_json = r#"
        {
            "errors": [
                {
                    "line": 1,
                    "column": 3,
                    "stop_line": 1,
                    "stop_column": 5,
                    "path": "test.py",
                    "code": -2,
                    "name": "bad-return",
                    "description": "Test error",
                    "concise_description": "Test error"
                }
            ]
        }
        "#;

        let baseline_file: LegacyErrors = serde_json::from_str(baseline_json).unwrap();
        let baseline_keys: HashSet<BaselineKey> =
            baseline_file.errors.iter().map(BaselineKey::from).collect();

        let processor = BaselineProcessor { baseline_keys };

        let module = Module::new(
            ModuleName::from_str("test_module"),
            ModulePath::filesystem(PathBuf::from("test.py")),
            Arc::new("test content 123456789".to_owned()),
        );
        let module2 = Module::new(
            ModuleName::from_str("test_module2"),
            ModulePath::filesystem(PathBuf::from("test2.py")),
            Arc::new("test content 123456789".to_owned()),
        );

        // This error should match (same path, error code, and column)
        let error1 = Error::new(
            module.clone(),
            TextRange::new(TextSize::new(2), TextSize::new(5)),
            vec1!["Any error message".to_owned()],
            ErrorKind::BadReturn,
        );
        assert!(processor.matches_baseline(&error1, Path::new("/")));

        // This error should not match (different column)
        let error2 = Error::new(
            module.clone(),
            TextRange::new(TextSize::new(4), TextSize::new(5)),
            vec1!["Test error".to_owned()],
            ErrorKind::BadReturn,
        );
        assert!(!processor.matches_baseline(&error2, Path::new("/")));

        // This error should not match (different error code)
        let error3 = Error::new(
            module,
            TextRange::new(TextSize::new(2), TextSize::new(5)),
            vec1!["Any error message".to_owned()],
            ErrorKind::AssertType,
        );
        assert!(!processor.matches_baseline(&error3, Path::new("/")));

        // This error should not match (different module)
        let error4 = Error::new(
            module2.clone(),
            TextRange::new(TextSize::new(2), TextSize::new(5)),
            vec1!["Any error message".to_owned()],
            ErrorKind::BadReturn,
        );
        assert!(!processor.matches_baseline(&error4, Path::new("/")));
    }
}
