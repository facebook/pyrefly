/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::module_name::is_python_identifier;
use crate::sys_info::PythonVersion;

/// Python keywords that can appear at the start of, or within, an expression.
/// These remain valid in a nested expression position (e.g. a call argument),
/// unlike statement keywords.
const EXPRESSION_KEYWORDS: &[&str] = &[
    "True", "False", "None", "and", "or", "not", "is", "lambda", "yield", "if", "in", "for", "else",
];

/// Python keywords that can only introduce a statement, never appear in the
/// middle of an expression.
const STATEMENT_KEYWORDS: &[&str] = &[
    "assert", "break", "class", "continue", "def", "del", "elif", "except", "finally", "from",
    "global", "import", "nonlocal", "pass", "raise", "return", "try", "type", "while", "with",
];

/// Additional keywords introduced in Python 3.5.
const PYTHON_3_5_KEYWORDS: &[&str] = &["async", "await"];

/// Additional keywords introduced in Python 3.10.
const PYTHON_3_10_KEYWORDS: &[&str] = &["case", "match"];

/// Whether `name` is a reserved keyword, i.e. `keyword.iskeyword(name)`. This is
/// spelled out rather than derived from the lists above: those are grouped by
/// where a keyword may appear, and include the soft keywords (`match`, `case`,
/// `type`), which are contextual and remain usable as ordinary names.
///
/// Version-independent, unlike [`get_keywords`]: a name reserved in any supported
/// version is reserved here, so callers asking "may this be written as an
/// identifier?" get the conservative answer.
pub fn is_reserved_keyword(name: &str) -> bool {
    matches!(
        name,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield"
    )
}

/// Whether `name` can be written in source as an identifier: a keyword argument
/// name, an attribute, or a class field. Equivalent to Python's
/// `name.isidentifier() and not keyword.iskeyword(name)`.
pub fn is_valid_identifier(name: &str) -> bool {
    is_python_identifier(name) && !is_reserved_keyword(name)
}

/// Returns a Vec containing all Python keywords for the specified Python version.
pub fn get_keywords(version: PythonVersion) -> Vec<&'static str> {
    let mut keywords: Vec<&'static str> = EXPRESSION_KEYWORDS.to_vec();
    keywords.extend(STATEMENT_KEYWORDS);

    if version.major >= 3 && version.minor >= 5 {
        keywords.extend(PYTHON_3_5_KEYWORDS);
    }
    if version.major >= 3 && version.minor >= 10 {
        keywords.extend(PYTHON_3_10_KEYWORDS);
    }

    keywords
}

/// Returns the Python keywords that are valid in an expression position for the
/// specified Python version. This excludes statement-only keywords (e.g. `while`,
/// `try`, `def`) that would be invalid inside a nested expression such as a call
/// argument or the right-hand side of an assignment. `await` is the only
/// expression keyword added after the base set (Python 3.5+).
pub fn get_expression_keywords(version: PythonVersion) -> Vec<&'static str> {
    let mut keywords: Vec<&'static str> = EXPRESSION_KEYWORDS.to_vec();
    if version.major >= 3 && version.minor >= 5 {
        keywords.push("await");
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
    fn test_expression_keywords_exclude_statement_keywords() {
        let keywords = get_expression_keywords(PythonVersion::new(3, 10, 0));
        // Expression keywords are present.
        assert!(keywords.contains(&"None"));
        assert!(keywords.contains(&"lambda"));
        assert!(keywords.contains(&"not"));
        assert!(keywords.contains(&"await"));
        assert!(keywords.contains(&"for"));
        // Statement-only keywords must not appear in an expression position.
        assert!(!keywords.contains(&"while"));
        assert!(!keywords.contains(&"try"));
        assert!(!keywords.contains(&"def"));
        assert!(!keywords.contains(&"async"));
        assert!(!keywords.contains(&"match"));
    }

    #[test]
    fn test_expression_keywords_await_gated_on_version() {
        assert!(!get_expression_keywords(PythonVersion::new(3, 4, 0)).contains(&"await"));
        assert!(get_expression_keywords(PythonVersion::new(3, 5, 0)).contains(&"await"));
    }

    #[test]
    fn test_valid_identifier_excludes_reserved_keywords() {
        assert!(is_reserved_keyword("class"));
        assert!(is_reserved_keyword("async"));
        assert!(!is_valid_identifier("class"));
        assert!(!is_valid_identifier("two words"));

        for soft_keyword in ["match", "case", "type"] {
            assert!(!is_reserved_keyword(soft_keyword));
            assert!(is_valid_identifier(soft_keyword));
        }
    }
}
