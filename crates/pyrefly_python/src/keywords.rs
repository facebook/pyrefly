/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

/// Returns a HashSet containing all Python keywords for the specified Python version.
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
}
