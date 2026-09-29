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
//! You can specify a specific error code, e.g. `# pyrefly: ignore[bad-return]`.
//! Within a `# type: ignore[...]` comment, Pyrefly always treats codes prefixed
//! with `pyrefly:` selectively, e.g. `# type: ignore[pyrefly:bad-return]` only
//! suppresses `bad-return`. How other (unknown) codes are treated depends on the
//! `type-ignore-unknown-tag-behavior` config: by default (`suppress`) they blanket
//! suppress every Pyrefly diagnostic on the line, `downgrade-to-warning` caps
//! their severity at warning, and `no-effect` leaves Pyrefly diagnostics unchanged.
//! Note that Pyright will only honor such codes after `# pyright: ignore[code]`.
//!
//! You can also use `# mypy: ignore-errors`, `# pyrefly: ignore-errors`
//! or `# type: ignore` at the beginning of a file to suppress all errors.
//! `# pyrefly: ignore-errors[invalid-type]` suppresses only the listed error
//! codes across the file rather than all errors.
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
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::token::Tokens;
use ruff_python_parser::Mode;
use ruff_python_parser::lexer::lex;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallset;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Comment {
    range: TextRange,
    offset: usize,
}

#[derive(Debug, Clone, Copy)]
struct PhysicalLineRange {
    start: usize,
    content_end: usize,
    end: usize,
}

fn physical_line_ranges(code: &str) -> Vec<PhysicalLineRange> {
    let bytes = code.as_bytes();
    let mut lines = Vec::new();
    let mut line_start = 0;
    let mut pos = 0;
    while pos < bytes.len() {
        if bytes[pos] == b'\r' || bytes[pos] == b'\n' {
            let content_end = pos;
            if bytes[pos] == b'\r' && bytes.get(pos + 1) == Some(&b'\n') {
                pos += 1;
            }
            let end = pos + 1;
            lines.push(PhysicalLineRange {
                start: line_start,
                content_end,
                end,
            });
            line_start = end;
        }
        pos += 1;
    }
    if line_start < code.len() {
        lines.push(PhysicalLineRange {
            start: line_start,
            content_end: code.len(),
            end: code.len(),
        });
    }
    lines
}

/// A physical source line with its original line terminator.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalLine<'a> {
    text: &'a str,
    ending: &'a str,
}

impl<'a> PhysicalLine<'a> {
    pub fn text(self) -> &'a str {
        self.text
    }

    pub fn ending(self) -> &'a str {
        self.ending
    }
}

/// Splits source using Python's universal-newline rules while retaining each
/// line's original terminator.
pub fn physical_lines_with_endings(code: &str) -> Vec<PhysicalLine<'_>> {
    physical_lines_iter(code).collect()
}

fn physical_lines_iter(code: &str) -> impl Iterator<Item = PhysicalLine<'_>> {
    physical_line_ranges(code)
        .into_iter()
        .map(move |line| PhysicalLine {
            text: &code[line.start..line.content_end],
            ending: &code[line.content_end..line.end],
        })
}

/// Splits source using Python's universal-newline rules.
pub fn physical_lines(code: &str) -> Vec<&str> {
    physical_lines_iter(code).map(PhysicalLine::text).collect()
}

/// Records comment positions from their absolute source ranges.
fn comments_from_ranges(
    ranges: impl IntoIterator<Item = TextRange>,
    lines: &[PhysicalLineRange],
) -> SmallMap<LineNumber, Comment> {
    let mut comments = SmallMap::new();
    let mut line = 0;
    for range in ranges {
        let start = range.start().to_usize();
        while line + 1 < lines.len() && lines[line + 1].start <= start {
            line += 1;
        }
        let offset = start - lines[line].start;
        debug_assert!(offset <= lines[line].content_end - lines[line].start);
        let line_number = LineNumber::from_zero_indexed(line as u32);
        assert!(
            comments
                .insert(line_number, Comment { range, offset })
                .is_none(),
            "the Python lexer must emit at most one comment token per physical line"
        );
    }
    comments
}

/// Lexes comment positions without building a Python syntax tree.
fn comments_from_source(code: &str, lines: &[PhysicalLineRange]) -> SmallMap<LineNumber, Comment> {
    let mut lexer = lex(code, Mode::Module);
    comments_from_ranges(
        std::iter::from_fn(|| {
            loop {
                let kind = lexer.next_token();
                if kind.is_eof() {
                    return None;
                }
                if kind == TokenKind::Comment {
                    return Some(lexer.current_range());
                }
            }
        }),
        lines,
    )
}

/// Records comment positions from tokens produced by the canonical parse.
fn comments_from_tokens(
    tokens: &Tokens,
    lines: &[PhysicalLineRange],
) -> SmallMap<LineNumber, Comment> {
    comments_from_ranges(
        tokens
            .iter()
            .filter(|token| token.kind() == TokenKind::Comment)
            .map(|token| token.range()),
        lines,
    )
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
    /// Enables `# zuban: ignore`
    Zuban,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    ValueEnum,
    Default
)]
#[serde(rename_all = "kebab-case")]
pub enum TypeIgnoreUnknownTagBehavior {
    /// Unknown tags have no effect on Pyrefly diagnostics.
    NoEffect,
    /// Unknown tags cap Pyrefly diagnostics on the same line at warning severity.
    DowngradeToWarning,
    /// Unknown tags suppress all Pyrefly diagnostics on the line.
    #[default]
    Suppress,
}

/// The effect a suppression has on a diagnostic. The derived `Ord` ordering
/// (`None` < `DowngradeToWarning` < `Suppress`) encodes suppression precedence:
/// call sites combine the effects of multiple applicable suppressions with
/// `.max()` to pick the strongest one, so the variant order here is load-bearing
/// and must remain weakest-to-strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SuppressionEffect {
    None,
    DowngradeToWarning,
    Suppress,
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
            "zuban" => Some(Tool::Zuban),
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
    /// The line number where the suppression comment is located.
    /// This may differ from the line the suppression applies to
    /// (e.g., when the comment is on the line above).
    comment_line: LineNumber,
    /// Byte offset within `comment_line` of the `#` that starts the comment.
    comment_offset: usize,
}

impl Suppression {
    /// A blanket suppression for `tool` that matches every error code.
    fn blanket(tool: Tool, comment_line: LineNumber, comment_offset: usize) -> Self {
        Self {
            tool,
            kind: Vec::new(),
            comment_line,
            comment_offset,
        }
    }

    /// Returns the line number where the suppression comment is located.
    pub fn comment_line(&self) -> LineNumber {
        self.comment_line
    }

    /// Returns the byte offset of the comment's `#` within `comment_line`.
    pub fn comment_offset(&self) -> usize {
        self.comment_offset
    }

    /// Returns the error codes that this suppression applies to.
    /// An empty slice means the suppression applies to all error codes.
    pub fn error_codes(&self) -> &[String] {
        &self.kind
    }

    /// Returns the tool that this suppression is for.
    pub fn tool(&self) -> Tool {
        self.tool
    }

    fn effect(
        &self,
        kind: &str,
        type_ignore_unknown_tag_behavior: TypeIgnoreUnknownTagBehavior,
    ) -> SuppressionEffect {
        match self.tool {
            Tool::Pyrefly => {
                if self.kind.is_empty() || self.kind.iter().any(|x| x == kind) {
                    SuppressionEffect::Suppress
                } else {
                    SuppressionEffect::None
                }
            }
            Tool::Type => {
                if self.kind.is_empty()
                    || self
                        .kind
                        .iter()
                        .any(|x| x.strip_prefix("pyrefly:") == Some(kind))
                {
                    SuppressionEffect::Suppress
                } else if self.kind.iter().any(|x| !x.starts_with("pyrefly:")) {
                    match type_ignore_unknown_tag_behavior {
                        TypeIgnoreUnknownTagBehavior::NoEffect => SuppressionEffect::None,
                        TypeIgnoreUnknownTagBehavior::DowngradeToWarning => {
                            SuppressionEffect::DowngradeToWarning
                        }
                        TypeIgnoreUnknownTagBehavior::Suppress => SuppressionEffect::Suppress,
                    }
                } else {
                    SuppressionEffect::None
                }
            }
            _ => SuppressionEffect::Suppress,
        }
    }
}

/// Record the position of lines affected by ignore suppressions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ignore {
    /// The line number here represents the line that the suppression applies to,
    /// not the line of the suppression comment.
    ignores: SmallMap<LineNumber, Vec<Suppression>>,
    comments: SmallMap<LineNumber, Comment>,
}

impl Ignore {
    pub fn new(code: &str) -> Self {
        let lines = physical_line_ranges(code);
        let comments = comments_from_source(code, &lines);
        Self {
            ignores: Self::parse(code, &lines, &comments),
            comments,
        }
    }

    /// Builds suppressions from parser tokens produced from the same source text.
    pub fn from_tokens(code: &str, tokens: &Tokens) -> Self {
        let lines = physical_line_ranges(code);
        let comments = comments_from_tokens(tokens, &lines);
        let ignores = Self::parse(code, &lines, &comments);
        Self { ignores, comments }
    }

    fn parse(
        code: &str,
        lines: &[PhysicalLineRange],
        comments: &SmallMap<LineNumber, Comment>,
    ) -> SmallMap<LineNumber, Vec<Suppression>> {
        let mut ignores: SmallMap<LineNumber, Vec<Suppression>> = SmallMap::new();
        // If we see a comment on a non-code line, apply it to the next non-comment line.
        let mut pending = Vec::new();
        let mut line = LineNumber::default();
        for (idx, source_line) in lines.iter().enumerate() {
            let line_str = &code[source_line.start..source_line.content_end];
            line = LineNumber::from_zero_indexed(idx as u32);
            let comment_start = comments.get(&line).map(|comment| comment.offset);
            let is_comment_only_line = comment_start
                .is_some_and(|comment_start| line_str[..comment_start].trim_start().is_empty());
            if !pending.is_empty() && (line_str.is_empty() || !is_comment_only_line) {
                ignores.entry(line).or_default().append(&mut pending);
            }
            let Some(comment_start) = comment_start else {
                continue;
            };
            // The lexer guarantees that comment_start points at the first hash.
            for comment in line_str[comment_start..].split('#').skip(1) {
                if let Some(suppression) = Self::parse_ignore_comment(comment, line, comment_start)
                {
                    if is_comment_only_line {
                        pending.push(suppression);
                    } else {
                        ignores.entry(line).or_default().push(suppression);
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
    /// `comment_line` and `comment_offset` locate the `#` starting the comment.
    fn parse_ignore_comment(
        l: &str,
        comment_line: LineNumber,
        comment_offset: usize,
    ) -> Option<Suppression> {
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
                kind: parse_error_codes(inside),
                comment_line,
                comment_offset,
            });
        } else if gap || lex.word_boundary() {
            return Some(Suppression::blanket(tool, comment_line, comment_offset));
        }
        None
    }

    pub fn is_ignored(
        &self,
        start_line: LineNumber,
        kind: &str,
        enabled_ignores: &SmallSet<Tool>,
    ) -> bool {
        self.suppression_effect(
            start_line,
            kind,
            enabled_ignores,
            TypeIgnoreUnknownTagBehavior::default(),
        ) == SuppressionEffect::Suppress
    }

    pub fn suppression_effect(
        &self,
        start_line: LineNumber,
        kind: &str,
        enabled_ignores: &SmallSet<Tool>,
        type_ignore_unknown_tag_behavior: TypeIgnoreUnknownTagBehavior,
    ) -> SuppressionEffect {
        if let Some(suppressions) = self.ignores.get(&start_line)
            && let Some(effect) = suppressions
                .iter()
                .filter(|supp| enabled_ignores.contains(&supp.tool))
                .map(|supp| supp.effect(kind, type_ignore_unknown_tag_behavior))
                .max()
        {
            return effect;
        }
        SuppressionEffect::None
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
        type_ignore_unknown_tag_behavior: TypeIgnoreUnknownTagBehavior,
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
                && supp.effect(kind, type_ignore_unknown_tag_behavior)
                    == SuppressionEffect::Suppress
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

    /// Returns the byte offset of the comment on a physical line.
    pub fn comment_start(&self, line: LineNumber) -> Option<usize> {
        self.comments.get(&line).map(|comment| comment.offset)
    }

    /// Returns the absolute source range of the comment on a physical line.
    pub fn comment_range(&self, line: LineNumber) -> Option<TextRange> {
        self.comments.get(&line).map(|comment| comment.range)
    }

    /// Returns all Python comment ranges in source order.
    pub fn comment_ranges(&self) -> impl Iterator<Item = TextRange> + '_ {
        self.comments.iter().map(|(_, comment)| comment.range)
    }

    /// Returns true if there are no suppressions.
    pub fn is_empty(&self) -> bool {
        self.ignores.is_empty()
    }
}

/// Returns true if `line` falls inside one of the sorted multiline string ranges.
fn is_in_multiline_string(
    multiline_string_ranges: &[(LineNumber, LineNumber)],
    line: LineNumber,
) -> bool {
    let idx = multiline_string_ranges.partition_point(|(start, _)| *start <= line);
    idx > 0 && {
        let (start, end) = multiline_string_ranges[idx - 1];
        line >= start && line <= end
    }
}

/// Parse top-level `ignore-errors` / `ignore-all-errors` / `type: ignore` directives.
///
/// Scans the beginning of the file for comment-only lines (including blank lines
/// and lines inside multiline strings like docstrings). Returns the file-level
/// suppressions found; Pyrefly entries may carry specific error codes
/// (`# pyrefly: ignore-errors[code]`), while other tools are blanket-only.
///
/// After a docstring, only `ignore-errors` directives are recognized — bare
/// `# type: ignore` is not, since it could plausibly be meant as a per-line
/// suppression for code that follows.
pub fn parse_ignore_all(
    code: &str,
    multiline_string_ranges: &[(LineNumber, LineNumber)],
) -> Vec<Suppression> {
    let mut res = Vec::new();
    let mut prev_ignore = None;
    let mut seen_docstring = false;

    for (idx, source_line) in physical_line_ranges(code).into_iter().enumerate() {
        let line = LineNumber::from_zero_indexed(idx as u32);
        let raw_line = &code[source_line.start..source_line.content_end];
        let trimmed = raw_line.trim();

        // Lines inside a multiline string (e.g. a module docstring) are not
        // code — skip them but record that we've passed through a docstring.
        if is_in_multiline_string(multiline_string_ranges, line) {
            seen_docstring = true;
            continue;
        }

        // Lines that open/close a triple-quoted string are also part of the
        // preamble — skip them.
        if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
            seen_docstring = true;
            continue;
        }

        // Stop at the first non-empty, non-comment line (i.e. actual code).
        // A pending `# type: ignore` followed directly by code is a per-line
        // suppression, not an ignore-all directive, so we discard it.
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            break;
        }

        if let Some((tool, prev_line, prev_offset)) = prev_ignore {
            // The previous `# type: ignore` was followed by another comment or
            // blank line, so it is a whole-file suppression.
            res.push(Suppression::blanket(tool, prev_line, prev_offset));
            prev_ignore = None;
        }

        let mut lex = Lexer(trimmed);
        if !lex.starts_with("#") {
            continue;
        }
        // `trimmed` starts with `#`, so its offset is the line's leading whitespace.
        let comment_offset = raw_line.len() - raw_line.trim_start().len();
        lex.trim_start();
        if lex.starts_with("pyre-ignore-all-errors") {
            res.push(Suppression::blanket(Tool::Pyre, line, comment_offset));
        } else if let Some(tool) = lex.starts_with_tool() {
            lex.trim_start();
            if lex.0.starts_with("ignore-errors") {
                // A file-level directive must have nothing after it (unlike the
                // misplaced-directive scan, which tolerates a trailing comment).
                // Only Pyrefly honors specific codes; other tools are blanket-only.
                if let Some((kind, tail)) = parse_ignore_errors_body(&mut lex)
                    && Lexer(tail).blank()
                    && (tool == Tool::Pyrefly || kind.is_empty())
                {
                    res.push(Suppression {
                        tool,
                        kind,
                        comment_line: line,
                        comment_offset,
                    });
                }
            } else if !seen_docstring && lex.starts_with("ignore") && lex.blank() {
                // After a docstring, bare `# type: ignore` is not recognized
                // as an ignore-all directive.
                prev_ignore = Some((tool, line, comment_offset));
            }
        }
    }
    res
}

/// Split the comma-separated error codes inside a `[...]` suppression into trimmed names.
fn parse_error_codes(inside: &str) -> Vec<String> {
    inside.split(',').map(|x| x.trim().to_owned()).collect()
}

/// Recognize an `ignore-errors` / `ignore-errors[code]` directive body, with the
/// leading `#` and `<tool>:` already consumed. Returns the error codes it names
/// (empty = blanket) together with the unparsed remainder after the directive,
/// or `None` if this is not an `ignore-errors` directive or its `[` bracket is
/// unclosed. The caller decides whether the remainder is acceptable — the
/// file-level parser requires it blank, while the misplaced-directive scan
/// tolerates a trailing comment.
fn parse_ignore_errors_body<'a>(lex: &mut Lexer<'a>) -> Option<(Vec<String>, &'a str)> {
    if !lex.starts_with("ignore-errors") {
        return None;
    }
    lex.trim_start();
    if lex.starts_with("[") {
        // Drop empty entries so `[]`/trailing commas act as a blanket ignore
        // rather than a directive that matches nothing.
        let (inside, after) = lex.0.split_once(']')?;
        let codes = parse_error_codes(inside)
            .into_iter()
            .filter(|code| !code.is_empty())
            .collect();
        Some((codes, after))
    } else {
        Some((Vec::new(), lex.0))
    }
}

/// Returns `true` if `comment` (a single `#…` comment, starting at its leading
/// `#`) is a pyrefly `ignore-errors` / `ignore-errors[code]` directive.
///
/// Shares directive recognition with `parse_ignore_all` via
/// `parse_ignore_errors_body`; the parsed codes are discarded since callers only
/// care whether the directive is present. A trailing explanatory `# …` comment
/// is tolerated (so tests can append `# E:` markers); any other trailing content
/// (e.g. prose) is not a directive. Other tools and the line-level
/// `# pyrefly: ignore` form are intentionally excluded.
fn is_pyrefly_ignore_errors(comment: &str) -> bool {
    let mut lex = Lexer(comment);
    if !lex.starts_with("#") {
        return false;
    }
    lex.trim_start();
    if lex.starts_with_tool() != Some(Tool::Pyrefly) {
        return false;
    }
    lex.trim_start();
    let Some((_, tail)) = parse_ignore_errors_body(&mut lex) else {
        return false;
    };
    let tail = tail.trim_start();
    tail.is_empty() || tail.starts_with('#')
}

/// Find the lines of pyrefly `ignore-errors` directives that appear *after* the
/// preamble, where a file-level suppression is silently inert.
///
/// The preamble is the leading run of blank lines, comments, and docstrings.
/// A directive there is honored by `parse_ignore_all`, so it is not reported.
/// Once the first real code line is seen, every subsequent comment-only line is
/// checked for a pyrefly `ignore-errors` directive (blanket or typed). Line
/// classification mirrors `parse_ignore_all` so the two functions partition
/// directives into "honored" (preamble) and "misplaced" (after code).
pub fn misplaced_ignore_errors(
    code: &str,
    multiline_string_ranges: &[(LineNumber, LineNumber)],
) -> Vec<LineNumber> {
    let mut res = Vec::new();
    let mut seen_code = false;

    for (idx, source_line) in physical_line_ranges(code).into_iter().enumerate() {
        let line = LineNumber::from_zero_indexed(idx as u32);
        let raw_line = &code[source_line.start..source_line.content_end];
        let trimmed = raw_line.trim();

        // Lines inside a multiline string (docstring, multi-line assignment) and
        // triple-quote boundary lines are neither code nor comment — skip them,
        // matching `parse_ignore_all`.
        if is_in_multiline_string(multiline_string_ranges, line)
            || trimmed.starts_with("\"\"\"")
            || trimmed.starts_with("'''")
            || trimmed.is_empty()
        {
            continue;
        }

        if !trimmed.starts_with('#') {
            // A non-empty, non-comment line is real code: it ends the preamble.
            seen_code = true;
            continue;
        }

        if seen_code && is_pyrefly_ignore_errors(trimmed) {
            res.push(line);
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use pyrefly_util::prelude::SliceExt;

    use super::*;

    #[test]
    fn test_parse_ignores() {
        fn f(x: &str, expect: &[(Tool, u32)]) {
            assert_eq!(
                &Ignore::new(x)
                    .ignores
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

        // Ignore `# pyrefly: ignore` inside a string but not before/after
        f("x = 1 + '# pyrefly: ignore'", &[]);
        f("x = ''  # pyrefly: ignore", &[(Tool::Pyrefly, 1)]);
        f("x = '''# pyrefly: ignore'''", &[]);
        f(
            r#"
x = """
x = 1  # pyrefly: ignore
"""
        "#,
            &[],
        );
        f(
            r#"
import textwrap
textwrap.dedent("""\
x = 1  # pyrefly: ignore
""")
        "#,
            &[],
        );
        f(
            r#"
x = """  # pyrefly: ignore
"""
        "#,
            &[],
        );
        f(
            r#"
x = """
# pyrefly: ignore"""
        "#,
            &[],
        );
        f(
            r#"
x = """
"""  # pyrefly: ignore
        "#,
            &[(Tool::Pyrefly, 3)],
        );
        f("x = ''''''  # pyrefly: ignore", &[(Tool::Pyrefly, 1)]);
        // A triple-quoted expression inside an f-string must not leave the
        // following line looking like part of a multiline string.
        f(
            "x = f'start{\"\"\"message\n\"\"\"}end'\ny: int = \"hello\"  # pyrefly: ignore[bad-assignment]",
            &[(Tool::Pyrefly, 3)],
        );
        f(
            "x = fr'start{\"\"\"# pyrefly: ignore\n\"\"\"}end'\ny: int = \"hello\"  # pyrefly: ignore[bad-assignment]",
            &[(Tool::Pyrefly, 3)],
        );
        f(
            "x = f'start{\"\"\"message\r\n\"\"\"}end'\r\ny: int = \"hello\"  # pyrefly: ignore[bad-assignment]",
            &[(Tool::Pyrefly, 3)],
        );
        f(
            "x = t'start{\"\"\"message\n\"\"\"}end'\ny: int = \"hello\"  # pyrefly: ignore[bad-assignment]",
            &[(Tool::Pyrefly, 3)],
        );
        f(r##"x = f"{"# pyrefly: ignore"}""##, &[]);
        f(
            r##"y: int = f"{"a#b"}" # pyrefly: ignore[bad-assignment]"##,
            &[(Tool::Pyrefly, 1)],
        );
        f(
            "a = \"\"\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\ryyy\"\"\" # comment\nb = f'{1}'  # pyrefly: ignore",
            &[(Tool::Pyrefly, 3)],
        );
    }

    #[test]
    fn test_suppression_comment_offset() {
        fn f(x: &str, expect: &[(u32, usize)]) {
            assert_eq!(
                &Ignore::new(x)
                    .ignores
                    .into_iter()
                    .flat_map(|(_, xs)| xs.map(|x| (x.comment_line.get(), x.comment_offset)))
                    .collect::<Vec<_>>(),
                expect,
                "{x:?}"
            );
        }

        f("x = 1  # type: ignore", &[(1, 7)]);
        // Not the `#` inside the string literal
        f(r##"x: str = "#hash"  # type: ignore"##, &[(1, 18)]);
        // Line starts inside a triple-quoted string that closes mid-line
        f("x = \"\"\"\n#fake\"\"\" # type: ignore", &[(2, 9)]);
        // A comment above code keeps its own line and offset
        f("  # type: ignore\nx = 1", &[(1, 2)]);
    }

    #[test]
    fn test_parse_ignore_comment() {
        fn f(x: &str, tool: Option<Tool>, kind: &[&str]) {
            let dummy_line = LineNumber::default();
            assert_eq!(
                Ignore::parse_ignore_comment(x, dummy_line, 0),
                tool.map(|tool| Suppression {
                    tool,
                    kind: kind.map(|x| (*x).to_owned()),
                    comment_line: dummy_line,
                    comment_offset: 0,
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

        f("zuban: ignore", Some(Tool::Zuban), &[]);
        f(
            "zuban: ignore[something]",
            Some(Tool::Zuban),
            &["something"],
        );

        // For a malformed comment, at least do something with it (works well incrementally)
        f("type: ignore[hello", Some(Tool::Type), &["hello"]);
    }

    #[test]
    fn test_type_ignore_specific_codes_require_pyrefly_prefix() {
        let enabled = Tool::default_enabled();
        let line = LineNumber::from_zero_indexed(0);

        let blanket = Ignore::new("x: int = ''  # type: ignore");
        assert!(blanket.is_ignored(line, "bad-assignment", &enabled));

        let mypy_code = Ignore::new("x: int = ''  # type: ignore[assignment]");
        assert!(mypy_code.is_ignored(line, "bad-assignment", &enabled));

        let pyrefly_code = Ignore::new("x: int = ''  # type: ignore[pyrefly:bad-assignment]");
        assert!(pyrefly_code.is_ignored(line, "bad-assignment", &enabled));
        assert!(!pyrefly_code.is_ignored(line, "bad-return", &enabled));

        let mixed_codes =
            Ignore::new("x: int = ''  # type: ignore[assignment, pyrefly:bad-assignment]");
        assert!(mixed_codes.is_ignored(line, "bad-assignment", &enabled));

        assert_eq!(
            mypy_code.suppression_effect(
                line,
                "bad-assignment",
                &enabled,
                TypeIgnoreUnknownTagBehavior::DowngradeToWarning,
            ),
            SuppressionEffect::DowngradeToWarning
        );
        assert_eq!(
            mypy_code.suppression_effect(
                line,
                "bad-assignment",
                &enabled,
                TypeIgnoreUnknownTagBehavior::Suppress,
            ),
            SuppressionEffect::Suppress
        );

        let mismatched_pyrefly_code =
            Ignore::new("x: int = ''  # type: ignore[pyrefly:bad-return]");
        assert_eq!(
            mismatched_pyrefly_code.suppression_effect(
                line,
                "bad-assignment",
                &enabled,
                TypeIgnoreUnknownTagBehavior::Suppress,
            ),
            SuppressionEffect::None
        );

        let mixed_mismatched_codes =
            Ignore::new("x: int = ''  # type: ignore[assignment, pyrefly:bad-return]");
        assert_eq!(
            mixed_mismatched_codes.suppression_effect(
                line,
                "bad-assignment",
                &enabled,
                TypeIgnoreUnknownTagBehavior::Suppress,
            ),
            SuppressionEffect::Suppress
        );
    }

    #[test]
    fn test_parse_ignore_all() {
        fn f(x: &str, ignores: &[(Tool, u32, &[&str])]) {
            assert_eq!(
                parse_ignore_all(x, &[]),
                ignores
                    .iter()
                    .map(|x| Suppression {
                        tool: x.0,
                        kind: x.2.iter().map(|x| (*x).to_owned()).collect(),
                        comment_line: LineNumber::new(x.1).unwrap(),
                        comment_offset: 0,
                    })
                    .collect::<Vec<_>>(),
                "{x:?}"
            );
        }

        f(
            "# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 1, &[])],
        );
        f(
            "# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[(Tool::Pyrefly, 1, &["bad-assignment"])],
        );
        f(
            "# pyrefly: ignore-errors [ bad-assignment, bad-return ]\nx = 5",
            &[(Tool::Pyrefly, 1, &["bad-assignment", "bad-return"])],
        );
        // Empty brackets and trailing commas drop empty entries, acting as a blanket ignore.
        f(
            "# pyrefly: ignore-errors[]\nx = 5",
            &[(Tool::Pyrefly, 1, &[])],
        );
        f(
            "# pyrefly: ignore-errors[bad-assignment,]\nx = 5",
            &[(Tool::Pyrefly, 1, &["bad-assignment"])],
        );
        // A missing closing bracket is malformed and rejected, not silently accepted.
        f("# pyrefly: ignore-errors[bad-assignment\nx = 5", &[]);
        f(
            "# comment\n# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 2, &[])],
        );
        f(
            "#comment\n  # indent\n# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 3, &[])],
        );
        f("x = 5\n# pyrefly: ignore-errors", &[]);
        // Directives are only recognized in the preamble; once real code (including an
        // import) appears the scan stops, so a later typed directive is inert — whether
        // it trails code, trails an import, or is sandwiched between code lines.
        f("x = 5\n# pyrefly: ignore-errors[bad-assignment]", &[]);
        f(
            "import os\n# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[],
        );
        f(
            "x = 5\n# pyrefly: ignore-errors[bad-assignment]\ny = 6",
            &[],
        );
        f("# type: ignore\n\nx = 5", &[(Tool::Type, 1, &[])]);
        f(
            "# comment\n# type: ignore\n# comment\nx = 5",
            &[(Tool::Type, 2, &[])],
        );
        f("# type: ignore\nx = 5", &[]);
        f("# pyre-ignore-all-errors\nx = 5", &[(Tool::Pyre, 1, &[])]);
        f(
            "# mypy: ignore-errors\n#pyrefly:ignore-errors",
            &[(Tool::Mypy, 1, &[]), (Tool::Pyrefly, 2, &[])],
        );
        f("# mypy: ignore-errors[bad-assignment]\nx = 5", &[]);
        f(
            "\r# pyrefly: ignore-errors\rx = 5",
            &[(Tool::Pyrefly, 2, &[])],
        );

        // Anything else on the line (other than space) makes it invalid
        f("# pyrefly: ignore-errors because I want to\nx = 5", &[]);
        f("# pyrefly: ignore-errors # because I want to\nx = 5", &[]);
        f(
            "# pyrefly: ignore-errors[bad-assignment] # because I want to\nx = 5",
            &[],
        );
        f(
            "# pyrefly: ignore-errors \nx = 5",
            &[(Tool::Pyrefly, 1, &[])],
        );
    }

    #[test]
    fn test_parse_ignore_all_with_docstring() {
        fn f(x: &str, ranges: &[(LineNumber, LineNumber)], ignores: &[(Tool, u32, &[&str])]) {
            assert_eq!(
                parse_ignore_all(x, ranges),
                ignores
                    .iter()
                    .map(|x| Suppression {
                        tool: x.0,
                        kind: x.2.iter().map(|x| (*x).to_owned()).collect(),
                        comment_line: LineNumber::new(x.1).unwrap(),
                        comment_offset: 0,
                    })
                    .collect::<Vec<_>>(),
                "{x:?}"
            );
        }

        // ignore-errors after a docstring should work
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# pyrefly: ignore-errors\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[(Tool::Pyrefly, 4, &[])],
        );

        // typed ignore-errors[code] after a docstring should also work
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[(Tool::Pyrefly, 4, &["bad-assignment"])],
        );

        // bare `# type: ignore` after docstring should NOT be recognized
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# type: ignore\n\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[],
        );

        // ignore-errors before a docstring should still work
        f(
            "# pyrefly: ignore-errors\n\"\"\"\nmodule docstring\n\"\"\"\nx = 5",
            &[(
                LineNumber::from_zero_indexed(1),
                LineNumber::from_zero_indexed(3),
            )],
            &[(Tool::Pyrefly, 1, &[])],
        );
    }

    #[test]
    fn test_misplaced_ignore_errors() {
        fn f(x: &str, expect: &[u32]) {
            assert_eq!(
                misplaced_ignore_errors(x, &[]),
                expect
                    .iter()
                    .map(|line| LineNumber::new(*line).unwrap())
                    .collect::<Vec<_>>(),
                "{x:?}"
            );
        }

        // A directive after code, after an import, or sandwiched between code
        // lines is inert and therefore misplaced.
        f("x = 5\n# pyrefly: ignore-errors", &[2]);
        f("x = 5\n# pyrefly: ignore-errors[bad-assignment]", &[2]);
        f(
            "import os\n# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[2],
        );
        f("x = 5\n# pyrefly: ignore-errors\ny = 6", &[2]);
        f("x = 5\r# pyrefly: ignore-errors", &[2]);
        // Multiple misplaced directives are all reported.
        f(
            "x = 5\n# pyrefly: ignore-errors\n# pyrefly: ignore-errors[bad-return]",
            &[2, 3],
        );
        // A trailing explanatory comment (or a test `# E:` marker) is tolerated.
        f("x = 5\n# pyrefly: ignore-errors  # E:", &[2]);
        f("x = 5\n# pyrefly: ignore-errors[bad-return]  # note", &[2]);

        // A directive in the preamble is honored by `parse_ignore_all`, so it is
        // never misplaced — whether it is the first line or follows comments.
        f("# pyrefly: ignore-errors\nx = 5", &[]);
        f("# comment\n# pyrefly: ignore-errors\nx = 5", &[]);
        f("# pyrefly: ignore-errors[bad-assignment]\nx = 5", &[]);

        // Other tools and the line-level form are not flagged.
        f("x = 5\n# mypy: ignore-errors", &[]);
        f("x = 5\n# pyre-ignore-all-errors", &[]);
        f("x = 5\n# pyrefly: ignore", &[]);
        f("x = 5\n# pyrefly: ignore[bad-return]", &[]);
        // Trailing prose (no `#`) is not a directive.
        f("x = 5\n# pyrefly: ignore-errors because I want to", &[]);
        // A same-line trailing directive is not on a comment-only line.
        f("x = 5  # pyrefly: ignore-errors", &[]);
    }

    #[test]
    fn test_misplaced_ignore_errors_with_docstring() {
        fn f(x: &str, ranges: &[(LineNumber, LineNumber)], expect: &[u32]) {
            assert_eq!(
                misplaced_ignore_errors(x, ranges),
                expect
                    .iter()
                    .map(|line| LineNumber::new(*line).unwrap())
                    .collect::<Vec<_>>(),
                "{x:?}"
            );
        }

        // A directive after a docstring is still in the preamble: not misplaced.
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# pyrefly: ignore-errors\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[],
        );
        // A directive after code that follows a docstring is misplaced.
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\nx = 5\n# pyrefly: ignore-errors",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[5],
        );
    }
}
