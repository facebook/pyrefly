/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Given a file, record which ignore statements are in it.
//!
//! Given `# type: ignore` we should ignore errors on that line.
//! Originally specified in <https://peps.python.org/pep-0484/>.
//!
//! You can also use the name of the linter, e.g. `# pyright: ignore`,
//! `# pyrefly: ignore`.
//!
//! You can specify a specific error code, e.g. `# type: ignore[invalid-type]`.
//! Note that Pyright will only honor such codes after `# pyright: ignore[code]`.
//!
//! You can also use `# mypy: ignore-errors`, `# pyrefly: ignore-errors`
//! or `# type: ignore` at the beginning of a file to suppress all errors.
//!
//! For Pyre compatibility we also allow `# pyre-ignore` and `# pyre-fixme`
//! as equivalents to `pyre: ignore`, and `# pyre-ignore-all-errors` as
//! an equivalent to `type: ignore` on its own line.
//!
//! We are permissive with whitespace, allowing `#type:ignore[code]` and
//! `#  type:  ignore  [  code  ]`, but do not allow a space before the colon.

use clap::ValueEnum;
use dupe::Dupe;
use enum_iterator::Sequence;
use pyrefly_util::lined_buffer::LineNumber;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallset;

/// Finds the byte offset of the first '#' character that starts a comment.
/// Returns None if no comment is found or if all '#' are inside strings.
/// Handles escape sequences and single/double quotes.
///
/// This is string-aware parsing that avoids treating '#' inside strings as comments.
/// For example: `x = "hello # world"  # real comment` correctly identifies the second '#'.
pub fn find_comment_start_in_line(line: &str) -> Option<usize> {
    let mut chars = line.char_indices().peekable();
    let mut in_string = None; // None, Some('"'), or Some('\'')

    while let Some((idx, ch)) = chars.next() {
        match (ch, in_string) {
            ('\\', Some(_)) => {
                chars.next();
            } // Skip next char if escaped
            ('"' | '\'', None) => in_string = Some(ch), // Enter string
            (q, Some(quote)) if q == quote => in_string = None, // Exit string
            ('#', None) => return Some(idx),            // Found comment!
            _ => {}
        }
    }
    None
}

/// The name of the tool that is being suppressed.
/// Note that the variant names and docstrings are displayed in `pyrefly check --help`.
#[derive(PartialEq, Debug, Clone, Hash, Eq, Dupe, Copy, Sequence)]
#[derive(Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Tool {
    /// Enables `# type: ignore`
    Type,
    /// Enables `# pyrefly: ignore` and `# pyrefly: ignore-errors`
    Pyrefly,
    /// Enables `# pyright: ignore`
    Pyright,
    /// Enables `# mypy: ignore-errors`
    Mypy,
    /// Enables `# ty: ignore`
    Ty,
    /// Enables `# pyre: ignore`, `# pyre-ignore`, `# pyre-fixme`, and `# pyre-ignore-all-errors`
    Pyre,
}

impl Tool {
    /// The maximum length of any tool.
    const MAX_LEN: usize = 7;

    fn from_comment(x: &str) -> Option<Self> {
        match x {
            "type" => Some(Tool::Type),
            "pyrefly" => Some(Tool::Pyrefly),
            "pyre" => Some(Tool::Pyre),
            "pyright" => Some(Tool::Pyright),
            "mypy" => Some(Tool::Mypy),
            "ty" => Some(Tool::Ty),
            _ => None,
        }
    }

    pub fn default_enabled() -> SmallSet<Self> {
        smallset! { Self::Type, Self::Pyrefly }
    }

    pub fn all() -> SmallSet<Self> {
        enum_iterator::all::<Self>().collect()
    }
}

/// A simple lexer that deals with the rules around whitespace.
/// As it consumes the string, it will move forward.
struct Lexer<'a>(&'a str);

impl<'a> Lexer<'a> {
    /// The string starts with the given string, return `true` if so.
    fn starts_with(&mut self, x: &str) -> bool {
        match self.0.strip_prefix(x) {
            Some(x) => {
                self.0 = x;
                true
            }
            None => false,
        }
    }

    /// The string starts with `tool:`, return the tool if it does.
    fn starts_with_tool(&mut self) -> Option<Tool> {
        let p = self
            .0
            .as_bytes()
            .iter()
            .take(Tool::MAX_LEN + 1)
            .position(|&c| c == b':')?;
        let tool = Tool::from_comment(&self.0[..p])?;
        self.0 = &self.0[p + 1..];
        Some(tool)
    }

    /// Trim whitespace from the start of the string.
    /// Return `true` if the string was changed.
    fn trim_start(&mut self) -> bool {
        let before = self.0;
        self.0 = self.0.trim_start();
        self.0.len() != before.len()
    }

    /// Return `true` if the string is empty or only whitespace.
    fn blank(&mut self) -> bool {
        self.0.trim_start().is_empty()
    }

    /// Return `true` if the string is at the start of a word boundary.
    /// That means the next char is not something that continues an identifier.
    fn word_boundary(&mut self) -> bool {
        self.0
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '-' && c != '_')
    }

    /// Finish and return the rest of the string.
    fn rest(self) -> &'a str {
        self.0
    }
}

#[derive(PartialEq, Debug, Clone, Hash, Eq)]
pub struct Suppression {
    tool: Tool,
    /// The permissible error kinds, use empty Vec to mean any are allowed
    kind: Vec<String>,
}

/// Record the position of lines affected by `# type: ignore[valid-type]` suppressions.
/// For now we don't record the content of the ignore, but we could.
#[derive(Debug, Clone, Default)]
pub struct Ignore {
    /// The line number here represents the line that the suppression applies to,
    /// not the line of the suppression comment.
    ignores: SmallMap<LineNumber, Vec<Suppression>>,
    /// All the tools with an ignore-all directive, with the line number that the directive is on.
    ignore_all: SmallMap<Tool, LineNumber>,
}

impl Ignore {
    pub fn new(code: &str) -> Self {
        Self {
            ignores: Self::parse_ignores(code),
            ignore_all: Self::parse_ignore_all(code),
        }
    }

    /// All the errors that were ignored, and the line number that ignore happened.
    fn parse_ignore_all(code: &str) -> SmallMap<Tool, LineNumber> {
        // process top level comments
        let mut res = SmallMap::new();
        let mut prev_ignore = None;
        for (line, x) in code
            .lines()
            .map(|x| x.trim())
            .take_while(|x| x.is_empty() || x.starts_with('#'))
            .enumerate()
        {
            let line = LineNumber::from_zero_indexed(line as u32);
            if let Some((tool, line)) = prev_ignore {
                // We consider any `# type: ignore` followed by a line with code to be a
                // normal suppression, not an ignore-all directive.
                res.entry(tool).or_insert(line);
                prev_ignore = None;
            }

            let mut lex = Lexer(x);
            if !lex.starts_with("#") {
                continue;
            }
            lex.trim_start();
            if lex.starts_with("pyre-ignore-all-errors") {
                res.entry(Tool::Pyre).or_insert(line);
            } else if let Some(tool) = lex.starts_with_tool() {
                lex.trim_start();
                if lex.starts_with("ignore-errors") && lex.blank() {
                    res.entry(tool).or_insert(line);
                } else if lex.starts_with("ignore") && lex.blank() {
                    prev_ignore = Some((tool, line));
                }
            }
        }
        res
    }

    fn parse_ignores(code: &str) -> SmallMap<LineNumber, Vec<Suppression>> {
        let mut ignores: SmallMap<LineNumber, Vec<Suppression>> = SmallMap::new();
        // If we see a comment on a non-code line, apply it to the next non-comment line.
        let mut pending = Vec::new();
        let mut line = LineNumber::default();
        for (idx, line_str) in code.lines().enumerate() {
            line = LineNumber::from_zero_indexed(idx as u32);
            let mut xs = line_str.split('#');
            let first = xs.next().unwrap_or("");
            if !pending.is_empty() && (line_str.is_empty() || !first.trim_start().is_empty()) {
                ignores.entry(line).or_default().append(&mut pending);
            }
            for x in xs {
                if let Some(supp) = Self::parse_ignore_comment(x) {
                    if first.trim_start().is_empty() {
                        pending.push(supp);
                    } else {
                        ignores.entry(line).or_default().push(supp);
                    }
                }
            }
        }
        if !pending.is_empty() {
            ignores
                .entry(line.increment())
                .or_default()
                .append(&mut pending);
        }
        ignores
    }

    /// Given the content of a comment, parse it as a suppression.
    fn parse_ignore_comment(l: &str) -> Option<Suppression> {
        let mut lex = Lexer(l);
        lex.trim_start();

        let mut tool = None;
        if let Some(t) = lex.starts_with_tool() {
            lex.trim_start();
            if lex.starts_with("ignore") {
                tool = Some(t);
            }
        } else if lex.starts_with("pyre-ignore") || lex.starts_with("pyre-fixme") {
            tool = Some(Tool::Pyre);
        }
        let tool = tool?;

        // We have seen `type: ignore` or `pyre-ignore`. Now look for `[code]` or the end.
        let gap = lex.trim_start();
        if lex.starts_with("[") {
            let rest = lex.rest();
            let inside = rest.split_once(']').map_or(rest, |x| x.0);
            return Some(Suppression {
                tool,
                kind: inside.split(',').map(|x| x.trim().to_owned()).collect(),
            });
        } else if gap || lex.word_boundary() {
            return Some(Suppression {
                tool,
                kind: Vec::new(),
            });
        }
        None
    }

    pub fn is_ignored(
        &self,
        start_line: LineNumber,
        kind: &str,
        enabled_ignores: &SmallSet<Tool>,
    ) -> bool {
        if enabled_ignores
            .iter()
            .any(|tool| self.ignore_all.contains_key(tool))
        {
            return true;
        }
        if let Some(suppressions) = self.ignores.get(&start_line)
            && suppressions.iter().any(|supp| {
                enabled_ignores.contains(&supp.tool)
                    && match supp.tool {
                        // We only check the subkind if they do `# pyrefly: ignore`
                        Tool::Pyrefly => {
                            supp.kind.is_empty() || supp.kind.iter().any(|x| x == kind)
                        }
                        _ => true,
                    }
            })
        {
            return true;
        }
        false
    }

    /// Similar to `is_ignored`, but it only returns true if the error is ignored
    /// by a suppression that targets a specific line.
    pub fn is_ignored_by_suppression_line(
        &self,
        suppression_line: LineNumber,
        start_line: LineNumber,
        end_line: LineNumber,
        kind: &str,
        enabled_ignores: &SmallSet<Tool>,
    ) -> bool {
        // If the error does not overlap the range, skip the more expensive check
        if start_line > suppression_line || end_line < suppression_line {
            return false;
        }
        let Some(suppressions) = self.ignores.get(&suppression_line) else {
            return false;
        };
        if suppressions.iter().any(|supp| {
            enabled_ignores.contains(&supp.tool)
                && match supp.tool {
                    // We only check the subkind if they do `# pyrefly: ignore`
                    Tool::Pyrefly => supp.kind.is_empty() || supp.kind.iter().any(|x| x == kind),
                    _ => true,
                }
        }) {
            return true;
        }
        false
    }

    // gets either just pyrefly ignores or pyrefly and type: ignore comments
    pub fn get_pyrefly_ignores(&self, all: bool) -> SmallSet<LineNumber> {
        let ignore_iter = self.ignores.iter();
        let filtered_ignores: Box<dyn Iterator<Item = (&LineNumber, &Vec<Suppression>)>> = if all {
            Box::new(ignore_iter.filter(|ignore| {
                ignore
                    .1
                    .iter()
                    .any(|s| s.tool == Tool::Pyrefly || s.tool == Tool::Type)
            }))
        } else {
            Box::new(ignore_iter.filter(|ignore| ignore.1.iter().any(|s| s.tool == Tool::Pyrefly)))
        };
        filtered_ignores.map(|(line, _)| *line).collect()
    }

    /// Returns an iterator over all suppressions in the file.
    /// Each item is a (line_number, suppressions) pair where line_number is where the suppression applies.
    pub fn iter(&self) -> impl Iterator<Item = (&LineNumber, &Vec<Suppression>)> {
        self.ignores.iter()
    }

    /// Gets the suppressions for a specific line.
    pub fn get(&self, line: &LineNumber) -> Option<&Vec<Suppression>> {
        self.ignores.get(line)
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::prelude::SliceExt;

    use super::*;

    #[test]
    fn test_parse_ignores() {
        fn f(x: &str, expect: &[(Tool, u32)]) {
            assert_eq!(
                &Ignore::parse_ignores(x)
                    .into_iter()
                    .flat_map(|(line, xs)| xs.map(|x| (x.tool, line.get())))
                    .collect::<Vec<_>>(),
                expect,
                "{x:?}"
            );
        }

        f("stuff # type: ignore # and then stuff", &[(Tool::Type, 1)]);
        f("more # stuff # type: ignore", &[(Tool::Type, 1)]);
        f(" pyrefly: ignore", &[]);
        f("normal line", &[]);
        f(
            "code # pyright: ignore\n# pyre-fixme\nmore code",
            &[(Tool::Pyright, 1), (Tool::Pyre, 3)],
        );
        f(
            "# type: ignore\n# pyright: ignore\n# bad\n\ncode",
            &[(Tool::Type, 4), (Tool::Pyright, 4)],
        );
    }

    #[test]
    fn test_parse_ignore_comment() {
        fn f(x: &str, tool: Option<Tool>, kind: &[&str]) {
            assert_eq!(
                Ignore::parse_ignore_comment(x),
                tool.map(|tool| Suppression {
                    tool,
                    kind: kind.map(|x| (*x).to_owned()),
                }),
                "{x:?}"
            );
        }

        f("ignore: pyrefly", None, &[]);
        f("pyrefly: ignore", Some(Tool::Pyrefly), &[]);
        f(
            "pyrefly: ignore[bad-return]",
            Some(Tool::Pyrefly),
            &["bad-return"],
        );
        f("pyrefly: ignore[]", Some(Tool::Pyrefly), &[""]);
        f("pyrefly: ignore[bad-]", Some(Tool::Pyrefly), &["bad-"]);

        // Check spacing
        f(" type: ignore ", Some(Tool::Type), &[]);
        f("type:ignore", Some(Tool::Type), &[]);
        f("type :ignore", None, &[]);

        // Check extras
        // Mypy rejects that, Pyright accepts it
        f("type: ignore because it is wrong", Some(Tool::Type), &[]);
        f("type: ignore_none", None, &[]);
        f("type: ignore1", None, &[]);
        f("type: ignore?", Some(Tool::Type), &[]);

        f("pyright: ignore", Some(Tool::Pyright), &[]);
        f(
            "pyright: ignore[something]",
            Some(Tool::Pyright),
            &["something"],
        );

        f("pyre-ignore", Some(Tool::Pyre), &[]);
        f("pyre-ignore[7]", Some(Tool::Pyre), &["7"]);
        f("pyre-fixme[7]", Some(Tool::Pyre), &["7"]);
        f(
            "pyre-fixme[61]: `x` may not be initialized here.",
            Some(Tool::Pyre),
            &["61"],
        );
        f("pyre-fixme: core type error", Some(Tool::Pyre), &[]);

        // For a malformed comment, at least do something with it (works well incrementally)
        f("type: ignore[hello", Some(Tool::Type), &["hello"]);
    }

    #[test]
    fn test_find_comment_start_in_line() {
        // Test basic comment finding
        assert_eq!(find_comment_start_in_line("x = 1  # comment"), Some(7));
        assert_eq!(find_comment_start_in_line("no comment here"), None);

        // Test string-aware parsing
        assert_eq!(
            find_comment_start_in_line(r#"x = "hello # world"  # real"#),
            Some(21)
        );
        assert_eq!(
            find_comment_start_in_line(r#"x = 'hello # world'  # real"#),
            Some(21)
        );

        // Test escaped quotes
        assert_eq!(
            find_comment_start_in_line(r#"x = "she said \"hi\" # not" # real"#),
            Some(28)
        );

        // Test multiple hashes
        assert_eq!(find_comment_start_in_line("# first # second"), Some(0));
    }

    #[test]
    fn test_parse_ignore_all() {
        fn f(x: &str, ignores: &[(Tool, u32)]) {
            assert_eq!(
                Ignore::parse_ignore_all(x),
                ignores
                    .iter()
                    .map(|x| (x.0, LineNumber::new(x.1).unwrap()))
                    .collect(),
                "{x:?}"
            );
        }

        f("# pyrefly: ignore-errors\nx = 5", &[(Tool::Pyrefly, 1)]);
        f(
            "# comment\n# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 2)],
        );
        f(
            "#comment\n  # indent\n# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 3)],
        );
        f("x = 5\n# pyrefly: ignore-errors", &[]);
        f("# type: ignore\n\nx = 5", &[(Tool::Type, 1)]);
        f(
            "# comment\n# type: ignore\n# comment\nx = 5",
            &[(Tool::Type, 2)],
        );
        f("# type: ignore\nx = 5", &[]);
        f("# pyre-ignore-all-errors\nx = 5", &[(Tool::Pyre, 1)]);
        f(
            "# mypy: ignore-errors\n#pyrefly:ignore-errors",
            &[(Tool::Mypy, 1), (Tool::Pyrefly, 2)],
        );

        // Anything else on the line (other than space) makes it invalid
        f("# pyrefly: ignore-errors because I want to\nx = 5", &[]);
        f("# pyrefly: ignore-errors # because I want to\nx = 5", &[]);
        f("# pyrefly: ignore-errors \nx = 5", &[(Tool::Pyrefly, 1)]);
    }
}
