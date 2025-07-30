/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::min;

use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::module::Module;

#[derive(Debug, Clone)]
pub struct Docstring(pub TextRange, pub Module);

impl Docstring {
    pub fn range_from_stmts(xs: &[Stmt]) -> Option<TextRange> {
        if let Some(stmt) = xs.first()
            && let Stmt::Expr(expr_stmt) = stmt
            && let ruff_python_ast::Expr::StringLiteral(_) = &*expr_stmt.value
        {
            return Some(stmt.range());
        }
        None
    }

    /// Clean a string literal ("""...""") and turn it into a docstring.
    pub fn clean(docstring: &str) -> String {
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

        let mut result = result.as_str();
        for (prefix, suffix) in patterns {
            if let Some(x) = result.strip_prefix(prefix)
                && let Some(x) = x.strip_suffix(suffix)
            {
                result = x;
                break; // Stop after first match to avoid over-trimming
            }
        }
        let result = result.replace("\r", "").replace("\t", "    ");

        // Remove the shortest amount of whitespace from the beginning of each line
        let min_indent = result
            .lines()
            .skip(1)
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

        result
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    line
                } else {
                    &line[min(min_indent, line.len())..]
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Resolve the docstring to a string. This involves parsing the file to get the contents of the docstring and then cleaning it.
    pub fn resolve(&self) -> String {
        Self::clean(self.1.code_at(self.0))
    }
}

#[cfg(test)]
mod tests {
    use crate::docstring::Docstring;

    #[test]
    fn test_clean_removes_double_multiline_double_quotes() {
        assert_eq!(
            Docstring::clean("\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }
    #[test]
    fn test_clean_removes_multiline_single_quotes() {
        assert_eq!(
            Docstring::clean("\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_single_quotes() {
        assert_eq!(
            Docstring::clean("\'test docstring\'").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_raw_multiline_double_quotes() {
        assert_eq!(
            Docstring::clean("r\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_raw_multiline_single_quotes() {
        assert_eq!(
            Docstring::clean("r\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_double_quotes() {
        assert_eq!(
            Docstring::clean("\"test docstring\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_carriage_returns() {
        assert_eq!(Docstring::clean("hello\rworld").as_str(), "helloworld");
    }

    #[test]
    fn test_clean_replaces_tabs_with_spaces() {
        assert_eq!(Docstring::clean("hello\tworld").as_str(), "hello    world");
    }

    #[test]
    fn test_clean_trims_shortest_whitespace() {
        assert_eq!(
            Docstring::clean("\n  hello\n    world\n  test").as_str(),
            "\nhello\n  world\ntest"
        );
    }

    #[test]
    fn test_docstring_panic() {
        Docstring::clean(" F\n\u{85}");
    }

    #[test]
    fn test_docstring_multiline_starts_at_first() {
        assert_eq!(
            Docstring::clean("\"\"\"hello\n  world\n  test\"\"\"").as_str(),
            "hello\nworld\ntest"
        );
    }
}
