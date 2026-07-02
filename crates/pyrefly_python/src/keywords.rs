/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::sys_info::PythonVersion;

/// Base Python keywords common to all supported Python versions.
const BASE_KEYWORDS: &[&str] = &[
    // Expression keywords
    "True", "False", "None", "and", "or", "not", "is", "lambda", "yield",
    // Statement keywords
    "assert", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "nonlocal", "pass", "raise", "return", "try",
    "type", "while", "with",
];

/// Additional keywords introduced in Python 3.5.
const PYTHON_3_5_KEYWORDS: &[&str] = &["async", "await"];

/// Additional keywords introduced in Python 3.10.
const PYTHON_3_10_KEYWORDS: &[&str] = &["case", "match"];

/// Subset of Python keywords known to appear as directory names in configerator
/// repos. When a directory is named with a keyword (e.g. `if`), Python module
/// names escape it with a trailing underscore (e.g. `if_`). This list matches
/// Pyright's supported subset as of 11/2024 and can be extended as needed.
const KEYWORD_ESCAPED_DIRS: &[&str] = &[
    "if", "async", "global", "import", "is", "in", "as", "or", "for", "del", "pass", "def",
];

/// All keyword-escaped directory names, stored in a HashSet for O(1) lookup.
/// Used by module resolution to detect keyword-escaped directory names
/// (e.g. `if_` → `if`).
static KEYWORD_ESCAPED_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| KEYWORD_ESCAPED_DIRS.iter().copied().collect());

/// Returns true if the given name is a Python keyword that may appear as an
/// escaped directory name in configerator repos (e.g. `if` → `if_`).
pub fn is_keyword(name: &str) -> bool {
    KEYWORD_ESCAPED_SET.contains(name)
}

/// Returns a Vec containing all Python keywords for the specified Python version.
pub fn get_keywords(version: PythonVersion) -> Vec<&'static str> {
    let mut keywords: Vec<&'static str> = BASE_KEYWORDS.to_vec();

    if version.major >= 3 && version.minor >= 5 {
        keywords.extend(PYTHON_3_5_KEYWORDS);
    }
    if version.major >= 3 && version.minor >= 10 {
        keywords.extend(PYTHON_3_10_KEYWORDS);
    }

    keywords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python35_keywords() {
        let keywords = get_keywords(PythonVersion::new(3, 5, 0));
        assert!(keywords.contains(&"def"));
        assert!(keywords.contains(&"yield"));
        assert!(keywords.contains(&"async"));
        assert!(keywords.contains(&"await"));
        assert!(!keywords.contains(&"match"));
        assert!(!keywords.contains(&"case"));
    }

    #[test]
    fn test_python310_keywords() {
        let keywords = get_keywords(PythonVersion::new(3, 10, 0));
        assert!(keywords.contains(&"def"));
        assert!(keywords.contains(&"yield"));
        assert!(keywords.contains(&"async"));
        assert!(keywords.contains(&"await"));
        assert!(keywords.contains(&"match"));
        assert!(keywords.contains(&"case"));
    }

    #[test]
    fn test_is_keyword_escaped_dirs() {
        // All 12 supported keywords should match.
        for kw in KEYWORD_ESCAPED_DIRS {
            assert!(is_keyword(kw), "{kw} should be recognized as a keyword");
        }
        // Keywords not in the configerator subset should NOT match.
        assert!(!is_keyword("while"));
        assert!(!is_keyword("class"));
        assert!(!is_keyword("return"));
        assert!(!is_keyword("try"));
        // Non-keywords should not match.
        assert!(!is_keyword("foo"));
        assert!(!is_keyword(""));
    }
}
