/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::min;
use std::sync::Arc;

use dupe::Dupe;
use ruff_python_ast::Stmt;

#[derive(Debug, Clone, Dupe)]
pub struct Docstring(Arc<String>);

impl Docstring {
    fn new(docstring: &str) -> Self {
        let result = docstring.replace("\r", "").replace("\t", "    ");

        // Remove any string literal prefixes and suffixes
        let patterns = [
            ("\"\"\"", "\"\"\""),  // Multiline double quotes
            ("\'\'\'", "\'\'\'"),  // Multiline single quotes
            ("r\"\"\"", "\"\"\""), // Raw multiline double quotes
            ("\'", "\'"),          // Single quotes
            ("r\'", "\'"),         // Raw single quotes
            ("\"", "\""),          // Double quotes
            ("r\"", "\""),         // Raw double quotes
        ];

        let mut result = result;
        for (prefix, suffix) in patterns {
            if result.starts_with(prefix)
                && result.ends_with(suffix)
                && let Some(x) = result.strip_prefix(prefix)
                && let Some(x) = x.strip_suffix(suffix)
            {
                result = x.to_owned();
                break; // Stop after first match to avoid over-trimming
            }
        }
        result = result.replace("\r", "");
        result = result.replace("\t", "    ");

        // Remove the shortest amount of whitespace from the beginning of each line
        let min_indent = result
            .lines()
            .flat_map(|line| {
                let spaces = line.bytes().take_while(|&c| c == b' ').count();
                if spaces == line.len() {
                    None
                } else {
                    Some(spaces)
                }
            })
            .min()
            .unwrap_or(0);

        Self(Arc::new(
            result
                .lines()
                .map(|line| &line[min(min_indent, line.len())..])
                .collect::<Vec<_>>()
                .join("\n"),
        ))
    }

    pub fn from_stmts(xs: &[Stmt]) -> Option<Self> {
        xs.first().and_then(|stmt| {
            if let Stmt::Expr(expr_stmt) = stmt
                && let ruff_python_ast::Expr::StringLiteral(string_lit) = &*expr_stmt.value
            {
                return Some(Docstring::new(string_lit.value.to_str()));
            }
            None
        })
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::export::docstring::Docstring;

    #[test]
    fn test_clean_removes_double_multiline_double_quotes() {
        assert_eq!(
            Docstring::new("\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }
    #[test]
    fn test_clean_removes_multiline_single_quotes() {
        assert_eq!(
            Docstring::new("\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_single_quotes() {
        assert_eq!(
            Docstring::new("\'test docstring\'").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_raw_multiline_double_quotes() {
        assert_eq!(
            Docstring::new("r\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_raw_multiline_single_quotes() {
        assert_eq!(
            Docstring::new("r\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_double_quotes() {
        assert_eq!(
            Docstring::new("\"test docstring\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_carriage_returns() {
        assert_eq!(Docstring::new("hello\rworld").as_str(), "helloworld");
    }

    #[test]
    fn test_clean_replaces_tabs_with_spaces() {
        assert_eq!(Docstring::new("hello\tworld").as_str(), "hello    world");
    }

    #[test]
    fn test_clean_trims_shortest_whitespace() {
        assert_eq!(
            Docstring::new("  hello\n    world\n  test").as_str(),
            "hello\n  world\ntest"
        );
    }

    #[test]
    fn test_docstring_panic() {
        Docstring::new(" F\n\u{85}");
    }
}
