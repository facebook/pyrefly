/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::module::Module;

const STRING_LITERAL_PATTERNS: [(&str, &str); 8] = [
    ("\"\"\"", "\"\"\""),  // Multiline double quotes
    ("\'\'\'", "\'\'\'"),  // Multiline single quotes
    ("r\"\"\"", "\"\"\""), // Raw multiline double quotes
    ("r\'\'\'", "\'\'\'"), // Raw multiline single quotes
    ("\'", "\'"),          // Single quotes
    ("r\'", "\'"),         // Raw single quotes
    ("\"", "\""),          // Double quotes
    ("r\"", "\""),         // Raw double quotes
];

fn strip_literal_quotes<'a>(mut text: &'a str) -> &'a str {
    for (prefix, suffix) in STRING_LITERAL_PATTERNS {
        if let Some(x) = text.strip_prefix(prefix)
            && let Some(x) = x.strip_suffix(suffix)
        {
            text = x;
            break;
        }
    }
    text
}

#[derive(Debug, Clone)]
pub struct Docstring(pub TextRange, pub Module);

impl Docstring {
    pub fn range_from_stmts(xs: &[Stmt]) -> Option<TextRange> {
        if let Some(stmt) = xs.first()
            && let Stmt::Expr(expr_stmt) = stmt
            && let Expr::StringLiteral(_) = &*expr_stmt.value
        {
            return Some(stmt.range());
        }
        None
    }

    /// Clean a string literal ("""...""") and turn it into a docstring.
    pub fn clean(docstring: &str) -> String {
        let result = normalize_literal(docstring);
        let lines: Vec<&str> = result.lines().collect();

        if lines.is_empty() {
            return String::new();
        }

        // Remove the shortest amount of whitespace from the beginning of each line
        let min_indent = minimal_indentation(lines.iter().skip(1).copied());

        format_docstring_lines(&lines, min_indent)
    }

    /// Resolve the docstring to a string. This involves parsing the file to get the contents of the docstring and then cleaning it.
    pub fn resolve(&self) -> String {
        Self::clean(self.1.code_at(self.0))
    }
}

/// Render cleaned docstring lines into markdown, adding fences for code blocks.
fn format_docstring_lines(lines: &[&str], min_indent: usize) -> String {
    let mut state = DocstringRenderState::new();
    for (i, line) in lines.iter().enumerate() {
        state.handle_line(line, i == 0, min_indent);
    }
    state.finish()
}

struct DocstringRenderState {
    output: Vec<String>,
    pending_literal_block: bool,
    pending_literal_block_indent: usize,
    code_block: Option<CodeBlockKind>,
    code_block_indent: usize,
    literal_block_marker_indent: usize,
    fence_backticks: usize,
}

impl DocstringRenderState {
    /// Start a new rendering state for cleaned docstrings.
    fn new() -> Self {
        Self {
            output: Vec::new(),
            pending_literal_block: false,
            pending_literal_block_indent: 0,
            code_block: None,
            code_block_indent: 0,
            literal_block_marker_indent: 0,
            fence_backticks: 0,
        }
    }

    /// Process one normalized docstring line, updating fence state and output.
    fn handle_line(&mut self, line: &str, is_first: bool, min_indent: usize) {
        let raw_leading_spaces = leading_space_count(line);
        let base_line = dedent_docstring_line(line, min_indent, is_first);
        let mut current = base_line.to_owned();

        // Inside a fenced code block, a line ending in `::` is code, not a literal-block marker.
        let saw_literal_marker = self.code_block != Some(CodeBlockKind::Fenced)
            && self.apply_literal_block_marker(&mut current, raw_leading_spaces);

        let trimmed_start = current.trim_start();
        let is_blank = trimmed_start.is_empty();
        let is_doctest_prompt = is_doctest_prompt(trimmed_start);
        let leading_spaces = leading_space_count(&current);

        if self.write_code_block_line(is_blank, is_doctest_prompt, raw_leading_spaces, &current) {
            return;
        }

        if self.maybe_start_code_block(
            is_blank,
            is_doctest_prompt,
            raw_leading_spaces,
            leading_spaces,
            &current,
            saw_literal_marker,
        ) {
            return;
        }

        self.output
            .push(format_non_code_line(&current, is_doctest_prompt));
    }

    /// Record a literal-block marker and normalize the line if needed.
    fn apply_literal_block_marker(
        &mut self,
        current: &mut String,
        raw_leading_spaces: usize,
    ) -> bool {
        if let Some(updated) = strip_literal_block_marker(current) {
            *current = updated;
            self.pending_literal_block = true;
            self.pending_literal_block_indent = raw_leading_spaces;
            return true;
        }
        false
    }

    /// Write this line as part of the open code block, if there is one.
    ///
    /// Returns true if the line has been written. Returns false if no code block is open, or if
    /// this line ends a doctest or literal block. Those blocks end at the first line that is not
    /// part of them, so the caller still has to render that line. A fenced block ends at its own
    /// closing fence, which is written here.
    fn write_code_block_line(
        &mut self,
        is_blank: bool,
        is_doctest_prompt: bool,
        raw_leading_spaces: usize,
        current: &str,
    ) -> bool {
        let Some(kind) = self.code_block else {
            return false;
        };

        match kind {
            CodeBlockKind::Doctest => {
                if !is_blank && !is_doctest_prompt {
                    self.output.push("```".to_owned());
                    self.code_block = None;
                    false
                } else {
                    self.output
                        .push(strip_code_indent(current, self.code_block_indent));
                    true
                }
            }
            CodeBlockKind::Literal => {
                if !is_blank && raw_leading_spaces <= self.literal_block_marker_indent {
                    self.output.push("```".to_owned());
                    self.code_block = None;
                    false
                } else {
                    self.output
                        .push(strip_code_indent(current, self.code_block_indent));
                    true
                }
            }
            CodeBlockKind::Fenced => {
                if is_closing_code_fence(current, self.fence_backticks) {
                    self.code_block = None;
                    self.output.push(current.trim_start().to_owned());
                } else {
                    self.output
                        .push(strip_code_indent(current, self.code_block_indent));
                }
                true
            }
        }
    }

    /// Open a doctest or literal-block fence when the line starts one.
    fn maybe_start_code_block(
        &mut self,
        is_blank: bool,
        is_doctest_prompt: bool,
        raw_leading_spaces: usize,
        leading_spaces: usize,
        current: &str,
        saw_literal_marker: bool,
    ) -> bool {
        if is_doctest_prompt {
            self.code_block = Some(CodeBlockKind::Doctest);
            self.code_block_indent = leading_spaces;
            self.output.push("```python".to_owned());
            self.output
                .push(strip_code_indent(current, self.code_block_indent));
            self.pending_literal_block = false;
            return true;
        }

        if self.pending_literal_block
            && !is_blank
            && raw_leading_spaces > self.pending_literal_block_indent
        {
            self.code_block = Some(CodeBlockKind::Literal);
            self.code_block_indent = leading_spaces;
            self.literal_block_marker_indent = self.pending_literal_block_indent;
            self.output.push("```".to_owned());
            self.output
                .push(strip_code_indent(current, self.code_block_indent));
            self.pending_literal_block = false;
            return true;
        }

        if let Some(backticks) = code_fence_opening(current) {
            self.code_block = Some(CodeBlockKind::Fenced);
            self.code_block_indent = leading_spaces;
            self.fence_backticks = backticks;
            self.output.push(current.trim_start().to_owned());
            self.pending_literal_block = false;
            return true;
        }

        if self.pending_literal_block
            && !is_blank
            && raw_leading_spaces <= self.pending_literal_block_indent
            && !saw_literal_marker
        {
            self.pending_literal_block = false;
        }

        false
    }

    /// Close any open fences and join output lines with markdown line breaks.
    fn finish(mut self) -> String {
        match self.code_block {
            Some(CodeBlockKind::Fenced) => self.output.push("`".repeat(self.fence_backticks)),
            Some(CodeBlockKind::Doctest | CodeBlockKind::Literal) => {
                self.output.push("```".to_owned())
            }
            None => {}
        }

        // Note: markdown doesn't break on just `\n`
        self.output.join("  \n")
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CodeBlockKind {
    /// A doctest example: lines starting with a `>>>` or `...` prompt, and the blank lines
    /// between them.
    ///
    /// Counts: `>>> add(1, 2)`, `... more`, and a bare `>>>`.
    /// Doesn't count: an output line such as `3`, which ends the block, or `>>>>` and `....`.
    Doctest,
    /// A reStructuredText literal block: the lines indented more than a preceding line that
    /// ends in `::`, and the blank lines between them.
    ///
    /// Counts: an indented `x = 1` after a line `Example::`.
    /// Doesn't count: a line no more indented than `Example::`, which ends the block; an
    /// indented line with no `::` line before it; a `::` line inside a fenced code block.
    Literal,
    /// A Markdown fenced code block written by the docstring author. It starts at a line of
    /// three or more backticks, optionally followed by an info string such as `python`, and ends
    /// at a line of at least as many backticks and nothing else.
    Fenced,
}

/// Return true if this line looks like a doctest prompt.
fn is_doctest_prompt(line: &str) -> bool {
    if line.starts_with(">>>") {
        return line.as_bytes().get(3).is_none_or(|b| *b != b'>');
    }
    if line.starts_with("...") {
        return line.as_bytes().get(3).is_none_or(|b| *b != b'.');
    }
    false
}

/// If this line opens a Markdown code fence, return its number of backticks: three or more,
/// optionally followed by an info string such as `python`. The info string cannot contain
/// backticks, so a line like ```` ```x = 1``` ```` is inline code rather than a fence.
fn code_fence_opening(line: &str) -> Option<usize> {
    let line = line.trim_start();
    let info = line.trim_start_matches('`');
    let backticks = line.len() - info.len();
    (backticks >= 3 && !info.contains('`')).then_some(backticks)
}

/// Return true if this line closes a fenced code block opened with `backticks` backticks. As in
/// CommonMark, the closing fence has at least as many backticks and nothing else.
fn is_closing_code_fence(line: &str, backticks: usize) -> bool {
    let line = line.trim();
    line.len() >= backticks && line.bytes().all(|b| b == b'`')
}

/// Strip a reStructuredText literal-block marker (`::`) and return the updated line.
fn strip_literal_block_marker(line: &str) -> Option<String> {
    let trimmed = line.trim_end();
    if trimmed == "::" {
        if line.trim() == "::" {
            return Some(String::new());
        }
        return Some(strip_one_trailing_colon(line));
    }
    if trimmed.ends_with("::") {
        return Some(strip_one_trailing_colon(line));
    }
    None
}

/// Remove a single trailing colon while preserving trailing whitespace.
fn strip_one_trailing_colon(line: &str) -> String {
    let trimmed = line.trim_end();
    let trimmed_len = trimmed.len();
    if trimmed_len == 0 {
        return line.to_owned();
    }
    if !trimmed.ends_with(':') {
        return line.to_owned();
    }
    let trailing = &line[trimmed_len..];
    let before_colon = &line[..trimmed_len - 1];
    format!("{before_colon}{trailing}")
}

/// Drop up to `indent` leading spaces for lines inside code fences.
fn strip_code_indent(line: &str, indent: usize) -> String {
    if line.trim().is_empty() {
        return String::new();
    }
    strip_leading_spaces(line, indent).to_owned()
}

/// Dedent a docstring line while keeping the first line intact.
fn dedent_docstring_line<'a>(line: &'a str, min_indent: usize, is_first: bool) -> &'a str {
    if is_first {
        return line;
    }
    strip_leading_spaces(line, min_indent)
}

/// Remove at most `max` leading spaces. A line may have fewer leading spaces than the indentation
/// being removed (e.g. a code block line that is less indented than the line that opened the
/// block), and since spaces are single-byte, the cut always lands on a char boundary.
fn strip_leading_spaces(line: &str, max: usize) -> &str {
    &line[leading_space_count(line).min(max)..]
}

/// Format a non-code line by handling blockquotes and preserving leading spaces.
fn format_non_code_line(line: &str, is_doctest_prompt: bool) -> String {
    // Handle potential leading blockquote (`> `) for non-doctest lines.
    let mut content = line;
    if !is_doctest_prompt {
        while let Some(rest) = content.strip_prefix('>') {
            content = rest.strip_prefix(' ').unwrap_or(rest);
        }
    }

    // Replace remaining leading spaces with &nbsp; or they might be ignored in markdown parsers.
    let leading_spaces = content.bytes().take_while(|&c| c == b' ').count();
    if leading_spaces > 0 {
        format!(
            "{}{}",
            "&nbsp;".repeat(leading_spaces),
            &content[leading_spaces..]
        )
    } else {
        content.to_owned()
    }
}

fn normalize_literal(docstring: &str) -> String {
    let normalized = docstring.replace("\r", "").replace("\t", "    ");
    let stripped = strip_literal_quotes(&normalized);
    stripped.replace("\r", "").replace("\t", "    ")
}

fn dedented_lines_for_parsing(docstring: &str) -> Vec<String> {
    let stripped = normalize_literal(docstring);
    let stripped = stripped.trim_matches('\n');
    if stripped.is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = stripped.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let min_indent = minimal_indentation(lines.iter().copied());

    lines
        .into_iter()
        .map(|line| {
            if line.trim_end().is_empty() {
                String::new()
            } else {
                let start = min_indent.min(line.len());
                line[start..].to_owned()
            }
        })
        .collect()
}

/// Dedent a block of text while preserving blank lines, similar to how we handle docstrings.
pub fn dedent_block_preserving_layout(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let min_indent = minimal_indentation(lines.iter().copied());
    let mut dedented = String::new();
    for line in lines {
        if line.trim().is_empty() {
            dedented.push('\n');
            continue;
        }
        let start = min_indent.min(line.len());
        dedented.push_str(&line[start..]);
        dedented.push('\n');
    }
    if !text.ends_with('\n') {
        dedented.push('\n');
    }
    Some(dedented)
}

fn leading_space_count(line: &str) -> usize {
    line.as_bytes().iter().take_while(|c| **c == b' ').count()
}

fn minimal_indentation<'a, I>(lines: I) -> usize
where
    I: Iterator<Item = &'a str>,
{
    lines
        .filter_map(|line| {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                None
            } else {
                Some(leading_space_count(line))
            }
        })
        .min()
        .unwrap_or(0)
}

/// Persist the documentation collected so far for the current parameters.
fn commit_parameter_doc(
    current_params: &mut Vec<String>,
    current_lines: &mut Vec<String>,
    docs: &mut HashMap<String, String>,
) {
    let content = current_lines.join("\n").trim().to_owned();
    current_lines.clear();
    if content.is_empty() {
        current_params.clear();
        return;
    }
    for name in current_params.drain(..) {
        docs.entry(name).or_insert_with(|| content.clone());
    }
}

/// Parse [`Sphinx`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
/// style `:param foo: description` blocks into a map of parameter docs.
fn parse_sphinx_params(lines: &[String], docs: &mut HashMap<String, String>) {
    let mut current_params = Vec::new();
    let mut current_lines = Vec::new();
    let mut base_indent = 0usize;

    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with(":param") {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);

            let rest = trimmed.trim_start_matches(":param").trim_start();
            let (name_part, desc_part) = match rest.split_once(':') {
                Some(parts) => parts,
                None => continue,
            };
            let name = name_part
                .split_whitespace()
                .last()
                .unwrap_or("")
                .trim_matches(',')
                .trim_end_matches(':')
                .trim_start_matches('*')
                .trim_start_matches('*')
                .to_owned();
            if name.is_empty() {
                continue;
            }
            base_indent = leading_space_count(line);
            current_params = vec![name];
            let desc = desc_part.trim_start();
            if !desc.is_empty() {
                current_lines.push(desc.to_owned());
            }
            continue;
        }

        if trimmed.starts_with(':') {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            continue;
        }

        if !current_params.is_empty() {
            if trimmed.is_empty() {
                commit_parameter_doc(&mut current_params, &mut current_lines, docs);
                continue;
            }
            let indent = leading_space_count(line);
            if indent > base_indent {
                current_lines.push(line.trim_start().to_owned());
            } else {
                commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            }
        }
    }

    commit_parameter_doc(&mut current_params, &mut current_lines, docs);
}

fn is_google_section(header: &str) -> bool {
    matches!(
        header,
        "Args" | "Arguments" | "Parameters" | "Keyword Args" | "Keyword Arguments"
    )
}

fn extract_google_param_name(header: &str) -> Option<String> {
    let token = header
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('(')
        .next()
        .unwrap_or("")
        .trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
}

/// Parse Google-style `Args:`/`Arguments:` sections of the form:
///
/// ```text
/// Args:
///     foo (int): description
///     bar: another description
/// ```
///
/// See <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>.
fn parse_google_params(lines: &[String], docs: &mut HashMap<String, String>) {
    let mut in_section = false;
    let mut section_indent = 0usize;
    let mut current_params = Vec::new();
    let mut current_lines = Vec::new();
    let mut param_indent = 0usize;

    for line in lines {
        let indent = leading_space_count(line);
        let trimmed = line.trim();

        if trimmed.is_empty() {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            continue;
        }

        if trimmed.ends_with(':') {
            let header = trimmed.trim_end_matches(':');
            if is_google_section(header) {
                commit_parameter_doc(&mut current_params, &mut current_lines, docs);
                in_section = true;
                section_indent = indent;
                continue;
            }
            if in_section && indent <= section_indent {
                commit_parameter_doc(&mut current_params, &mut current_lines, docs);
                in_section = false;
            }
        } else if in_section && indent <= section_indent {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            in_section = false;
        }

        if !in_section {
            continue;
        }

        if indent <= section_indent {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            in_section = false;
            continue;
        }

        let content = line[section_indent.min(line.len())..].trim_start();
        if let Some((header, rest)) = content.split_once(':')
            && let Some(name) = extract_google_param_name(header.trim())
        {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            current_params = vec![name];
            param_indent = indent;
            let desc = rest.trim_start();
            if !desc.is_empty() {
                current_lines.push(desc.to_owned());
            }
            continue;
        }

        if !current_params.is_empty() {
            if indent > param_indent {
                current_lines.push(content.trim_start().to_owned());
            } else {
                commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            }
        }
    }

    commit_parameter_doc(&mut current_params, &mut current_lines, docs);
}

/// Return true if `line` is the dashed underline of a numpydoc section header.
fn is_numpydoc_underline(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 3 && trimmed.bytes().all(|byte| byte == b'-')
}

fn is_numpydoc_section(header: &str) -> bool {
    matches!(header, "Parameters" | "Other Parameters")
}

/// Extract the names from a numpydoc parameter declaration such as
/// `x, y : int` or `*args`. The type is optional.
fn extract_numpydoc_param_names(line: &str) -> Option<Vec<String>> {
    let names = line.split_once(':').map_or(line, |(names, _)| names);
    let names = names
        .split(',')
        .map(|name| name.trim().trim_matches('`').trim_start_matches('*'))
        .filter(|name| {
            let mut chars = name.chars();
            chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
                && chars.all(|c| c.is_alphanumeric() || c == '_' || c == '.')
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    (!names.is_empty()).then_some(names)
}

/// Parse numpydoc-style `Parameters` sections of the form:
///
/// ```text
/// Parameters
/// ----------
/// foo : int
///     Description of foo.
/// bar, baz : str
///     Description of bar and baz.
/// ```
///
/// See <https://numpydoc.readthedocs.io/en/latest/format.html#parameters>.
fn parse_numpydoc_params(lines: &[String], docs: &mut HashMap<String, String>) {
    let mut in_section = false;
    let mut section_indent = 0usize;
    let mut current_params = Vec::new();
    let mut current_lines = Vec::new();
    let mut param_indent = 0usize;

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let indent = leading_space_count(line);

        // Within a parameter section, deeper-indented headings belong to descriptions.
        if !trimmed.is_empty()
            && (!in_section || indent <= section_indent)
            && lines.get(index + 1).is_some_and(|next| {
                leading_space_count(next) == indent && is_numpydoc_underline(next)
            })
        {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            in_section = is_numpydoc_section(trimmed);
            section_indent = indent;
            continue;
        }

        if indent == section_indent && is_numpydoc_underline(line) {
            continue;
        }

        if !in_section {
            continue;
        }

        if trimmed.is_empty() {
            if !current_params.is_empty()
                && current_lines.last().is_some_and(|line| !line.is_empty())
            {
                current_lines.push(String::new());
            }
            continue;
        }

        if let Some(names) = extract_numpydoc_param_names(trimmed)
            && (current_params.is_empty() || indent <= param_indent)
        {
            commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            current_params = names;
            param_indent = indent;
            continue;
        }

        if !current_params.is_empty() {
            if indent > param_indent {
                current_lines.push(trimmed.to_owned());
            } else {
                commit_parameter_doc(&mut current_params, &mut current_lines, docs);
            }
        }
    }

    commit_parameter_doc(&mut current_params, &mut current_lines, docs);
}

/// Extract a map of `parameter -> markdown` documentation snippets from the
/// supplied docstring, supporting Sphinx (`:param foo:`), Google-style
/// (`Args:`), and numpydoc (`Parameters`) formats.
pub fn parse_parameter_documentation(docstring: &str) -> HashMap<String, String> {
    let lines = dedented_lines_for_parsing(docstring);
    let mut docs = HashMap::new();
    if lines.is_empty() {
        return docs;
    }
    parse_sphinx_params(&lines, &mut docs);
    parse_google_params(&lines, &mut docs);
    parse_numpydoc_params(&lines, &mut docs);
    docs
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::docstring::CodeBlockKind;
    use crate::docstring::Docstring;
    use crate::docstring::DocstringRenderState;
    use crate::docstring::parse_parameter_documentation;

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
    fn test_clean_trims_shortest_whitespace_and_replaces_space_with_nbsp() {
        assert_eq!(
            Docstring::clean("\n  hello\n    world\n  test").as_str(),
            "  \nhello  \n&nbsp;&nbsp;world  \ntest"
        );
    }

    #[test]
    fn test_docstring_panic() {
        Docstring::clean(" F\n\u{85}");
    }

    #[test]
    fn test_docstring_code_block_line_less_indented_than_fence_starts_with_multibyte_char() {
        assert_eq!(
            Docstring::clean("\"\"\"Tree::\n\n      root\n    ├── a\"\"\"").as_str(),
            "Tree:  \n  \n```  \nroot  \n├── a  \n```"
        );
    }

    #[test]
    fn test_docstring_whitespace_only_line_with_multibyte_whitespace() {
        assert_eq!(
            Docstring::clean("\"\"\"x\n    a\n  \u{3000}\"\"\"").as_str(),
            "x  \na  \n\u{3000}"
        );
    }

    #[test]
    fn test_docstring_multiline_starts_at_first() {
        assert_eq!(
            Docstring::clean("\"\"\"hello\n  world\n  test\"\"\"").as_str(),
            "hello  \nworld  \ntest"
        );
    }

    #[test]
    fn test_docstring_preserves_doctest_prompt() {
        assert_eq!(
            Docstring::clean("\"\"\"Example\n>>> foo()\"\"\"").as_str(),
            "Example  \n```python  \n>>> foo()  \n```"
        );
    }

    #[test]
    fn test_docstring_literal_block_uses_code_fence() {
        assert_eq!(
            Docstring::clean(
                "\"\"\"Example::\n\n    >>> app = Flask(__name__)\n    >>> api = Api()\"\"\""
            )
            .as_str(),
            "Example:  \n  \n```python  \n>>> app = Flask(__name__)  \n>>> api = Api()  \n```"
        );
    }

    #[test]
    fn test_docstring_fenced_code_block_indentation() {
        let docstring = r#""""
    **Example:**
    ```python
    test = Test(
        name = "Test"
    )
    ```
    """"#;
        assert_eq!(
            Docstring::clean(docstring).replace("  \n", "\n"),
            r#"
**Example:**
```python
test = Test(
    name = "Test"
)
```
"#
        );
    }

    /// Return the line ranges that `DocstringRenderState` identifies as each kind of code block.
    fn code_block_ranges(lines: &[&str]) -> Vec<(CodeBlockKind, Range<usize>)> {
        let mut state = DocstringRenderState::new();
        let mut ranges: Vec<(CodeBlockKind, Range<usize>)> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let before = state.code_block;
            state.handle_line(line, i == 0, 0);
            // A closing fence belongs to the block it closes.
            let fenced = before.filter(|kind| *kind == CodeBlockKind::Fenced);
            let Some(kind) = state.code_block.or(fenced) else {
                continue;
            };
            if before == Some(kind) {
                ranges.last_mut().expect("an open block has a range").1.end = i + 1;
            } else {
                ranges.push((kind, i..i + 1));
            }
        }
        ranges
    }

    #[test]
    fn test_docstring_code_block_ranges() {
        let lines = [
            "Summary",
            ">>> f()",
            "Example::",
            "",
            "    x = 1",
            "After",
            "```python",
            ">>> g()",
            "```",
            "```",
            "y = 2",
            "```",
            "Done",
        ];
        assert_eq!(
            code_block_ranges(&lines),
            vec![
                (CodeBlockKind::Doctest, 1..2),
                (CodeBlockKind::Literal, 4..5),
                (CodeBlockKind::Fenced, 6..9),
                (CodeBlockKind::Fenced, 9..12),
            ]
        );
    }

    #[test]
    fn test_docstring_fenced_code_block_edge_cases() {
        // An unclosed fence runs to the end of the docstring.
        assert_eq!(
            code_block_ranges(&["```", "x = 1"]),
            vec![(CodeBlockKind::Fenced, 0..2)]
        );
        // An empty fence is still a code block.
        assert_eq!(
            code_block_ranges(&["```", "```", "after"]),
            vec![(CodeBlockKind::Fenced, 0..2)]
        );
        // Backticks that open and close on the same line are inline code, not a fence.
        assert_eq!(
            code_block_ranges(&["```x = 1```", "```", "y = 2", "```"]),
            vec![(CodeBlockKind::Fenced, 1..4)]
        );
    }

    #[test]
    fn test_docstring_fenced_code_block_next_to_other_kinds() {
        // A `::` line inside a fence is code, so no literal block follows the fence.
        assert_eq!(
            code_block_ranges(&["```rst", "Example::", "```", "    indented"]),
            vec![(CodeBlockKind::Fenced, 0..3)]
        );
        // A fence right after a `::` marker wins, and the marker doesn't apply after it.
        assert_eq!(
            code_block_ranges(&["Example::", "", "```", "x", "```", "    after"]),
            vec![(CodeBlockKind::Fenced, 2..5)]
        );
        // Backticks inside a literal block are literal text.
        assert_eq!(
            code_block_ranges(&["Example::", "", "    ```", "    x", "Done"]),
            vec![(CodeBlockKind::Literal, 2..4)]
        );
        assert_eq!(
            code_block_ranges(&["Example::", "", "    x", "    ```", "    y", "Done"]),
            vec![(CodeBlockKind::Literal, 2..5)]
        );
        // An indented fence, as in a parameter description.
        assert_eq!(
            code_block_ranges(&[
                "Args:",
                "    x: an arg",
                "    ```",
                "    code",
                "    ```",
                "After"
            ]),
            vec![(CodeBlockKind::Fenced, 2..5)]
        );
        // A `>>>` line right after a fence starts a doctest block.
        assert_eq!(
            code_block_ranges(&["```", "x", "```", ">>> f()"]),
            vec![
                (CodeBlockKind::Fenced, 0..3),
                (CodeBlockKind::Doctest, 3..4)
            ]
        );
    }

    #[test]
    fn test_docstring_fenced_code_block_closing_fence() {
        // The closing fence needs at least as many backticks as the opening one.
        assert_eq!(
            code_block_ranges(&["````", "```", "x", "````"]),
            vec![(CodeBlockKind::Fenced, 0..4)]
        );
        assert_eq!(
            code_block_ranges(&["```", "x", "````", "after"]),
            vec![(CodeBlockKind::Fenced, 0..3)]
        );
        // A line with an info string doesn't close a fence.
        assert_eq!(
            code_block_ranges(&["```", "```python", "x", "```"]),
            vec![(CodeBlockKind::Fenced, 0..4)]
        );
    }

    #[test]
    fn test_docstring_fenced_code_block_rendering() {
        // A line ending in `::` inside a fence keeps both colons.
        assert_eq!(
            Docstring::clean("\"\"\"Summary\n```rst\nExample::\n```\n\"\"\"").as_str(),
            "Summary  \n```rst  \nExample::  \n```"
        );
        // An unclosed fence gets exactly one closing fence, as long as the opening one.
        assert_eq!(
            Docstring::clean("\"\"\"Summary\n```\nx = 1\n\"\"\"").as_str(),
            "Summary  \n```  \nx = 1  \n```"
        );
        assert_eq!(
            Docstring::clean("\"\"\"Summary\n````\nx = 1\n\"\"\"").as_str(),
            "Summary  \n````  \nx = 1  \n````"
        );
    }

    #[test]
    fn test_docstring_fenced_code_block_under_header() {
        let docstring = r#""""
    Examples:
        ```python
        test = Test(
            name = "Test"
        )
        ```
    """"#;
        assert_eq!(
            Docstring::clean(docstring).replace("  \n", "\n"),
            r#"
Examples:
```python
test = Test(
    name = "Test"
)
```
"#
        );
    }

    #[test]
    fn test_parse_sphinx_param_docs() {
        let doc = r#"
:param foo: first line
    second line
:param str bar: another
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first line\nsecond line");
        assert_eq!(docs.get("bar").unwrap(), "another");
    }

    #[test]
    fn test_parse_google_param_docs() {
        let doc = r#"
Args:
    foo (int): first line
        second line
    bar: final
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first line\nsecond line");
        assert_eq!(docs.get("bar").unwrap(), "final");
    }

    #[test]
    fn test_parse_sphinx_empty_param() {
        let doc = r#"
:param foo:
:param bar: has description
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo"), None); // Empty params should not be included
        assert_eq!(docs.get("bar").unwrap(), "has description");
    }

    #[test]
    fn test_parse_sphinx_with_type_annotations() {
        let doc = r#"
:param int foo: an integer
:param str bar: a string
:param Optional[Dict[str, int]] baz: complex type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "an integer");
        assert_eq!(docs.get("bar").unwrap(), "a string");
        assert_eq!(docs.get("baz").unwrap(), "complex type");
    }

    #[test]
    fn test_parse_sphinx_multiple_continuation_lines() {
        let doc = r#"
:param foo: line one
    line two
    line three
    line four
:param bar: single line
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("foo").unwrap(),
            "line one\nline two\nline three\nline four"
        );
        assert_eq!(docs.get("bar").unwrap(), "single line");
    }

    #[test]
    fn test_parse_sphinx_with_other_directives() {
        let doc = r#"
:param foo: the foo parameter
:param bar: the bar parameter
:return: the return value
:raises ValueError: when invalid
:type foo: int
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "the foo parameter");
        assert_eq!(docs.get("bar").unwrap(), "the bar parameter");
        // Other directives should not be included as parameters
        assert_eq!(docs.get("return"), None);
        assert_eq!(docs.get("raises"), None);
    }

    #[test]
    fn test_parse_sphinx_with_varargs() {
        let doc = r#"
:param *args: positional arguments
:param **kwargs: keyword arguments
:param regular: regular param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("args").unwrap(), "positional arguments");
        assert_eq!(docs.get("kwargs").unwrap(), "keyword arguments");
        assert_eq!(docs.get("regular").unwrap(), "regular param");
    }

    #[test]
    fn test_parse_sphinx_indented_in_docstring() {
        let doc = r#"""
    Summary of function.

    :param foo: first param
        with continuation
    :param bar: second param
    """#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first param\nwith continuation");
        assert_eq!(docs.get("bar").unwrap(), "second param");
    }

    #[test]
    fn test_parse_google_different_headers() {
        let doc = r#"
Arguments:
    foo: using Arguments header
    bar: second arg

def another_func():
    """
    Keyword Arguments:
        baz: keyword arg
    """
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "using Arguments header");
        assert_eq!(docs.get("bar").unwrap(), "second arg");
    }

    #[test]
    fn test_parse_google_parameters_header() {
        let doc = r#"
Parameters:
    foo (int): first param
    bar (str): second param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first param");
        assert_eq!(docs.get("bar").unwrap(), "second param");
    }

    #[test]
    fn test_parse_google_keyword_args_header() {
        let doc = r#"
Keyword Args:
    foo: keyword arg one
    bar: keyword arg two
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "keyword arg one");
        assert_eq!(docs.get("bar").unwrap(), "keyword arg two");
    }

    #[test]
    fn test_parse_google_deeply_indented_continuation() {
        let doc = r#"
Args:
    foo: first line
        second line
            third line deeply indented
        fourth line
    bar: simple
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("foo").unwrap(),
            "first line\nsecond line\nthird line deeply indented\nfourth line"
        );
        assert_eq!(docs.get("bar").unwrap(), "simple");
    }

    #[test]
    fn test_parse_google_no_type_annotation() {
        let doc = r#"
Args:
    foo: no type annotation
    bar (int): with type annotation
    baz: also no type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "no type annotation");
        assert_eq!(docs.get("bar").unwrap(), "with type annotation");
        assert_eq!(docs.get("baz").unwrap(), "also no type");
    }

    #[test]
    fn test_parse_google_complex_type_annotations() {
        let doc = r#"
Args:
    foo (Optional[List[Dict[str, int]]]): complex type
    bar (Callable[[int, str], bool]): callable type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "complex type");
        assert_eq!(docs.get("bar").unwrap(), "callable type");
    }

    #[test]
    fn test_parse_google_section_ends_with_other_section() {
        let doc = r#"
Args:
    foo: the foo param
    bar: the bar param

Returns:
    int: the return value

Raises:
    ValueError: when things go wrong
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "the foo param");
        assert_eq!(docs.get("bar").unwrap(), "the bar param");
        // Returns and Raises should not be parsed as params
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_parse_google_empty_parameter_description() {
        let doc = r#"
Args:
    foo:
    bar: has description
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo"), None); // Empty description should not be included
        assert_eq!(docs.get("bar").unwrap(), "has description");
    }

    #[test]
    fn test_parse_mixed_sphinx_and_google() {
        let doc = r#"
:param sphinx_param: using Sphinx style
    with continuation

Args:
    google_param: using Google style
    another_google: second Google param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("sphinx_param").unwrap(),
            "using Sphinx style\nwith continuation"
        );
        assert_eq!(docs.get("google_param").unwrap(), "using Google style");
        assert_eq!(docs.get("another_google").unwrap(), "second Google param");
    }

    #[test]
    fn test_parse_sphinx_param_with_comma_in_type() {
        let doc = r#"
:param Dict[str, int] foo: dict param
:param Tuple[int, str, bool] bar: tuple param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "dict param");
        assert_eq!(docs.get("bar").unwrap(), "tuple param");
    }

    #[test]
    fn test_parse_numpydoc_param_docs() {
        let doc = r#"
Parameters
----------
foo : int
    first line
    second line
bar : str
    another
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first line\nsecond line");
        assert_eq!(docs.get("bar").unwrap(), "another");
    }

    #[test]
    fn test_parse_numpydoc_multiple_names_and_varargs() {
        let doc = r#"
Parameters
----------
foo, bar : int
    shared description
*args : tuple
    positional arguments
**kwargs : dict
    keyword arguments
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "shared description");
        assert_eq!(docs.get("bar").unwrap(), "shared description");
        assert_eq!(docs.get("args").unwrap(), "positional arguments");
        assert_eq!(docs.get("kwargs").unwrap(), "keyword arguments");
    }

    #[test]
    fn test_parse_numpydoc_other_parameters_and_section_end() {
        let doc = r#"
Parameters
----------
foo : int
    foo description

Other Parameters
----------------
bar : str
    bar description

Returns
-------
int
    the return value
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "foo description");
        assert_eq!(docs.get("bar").unwrap(), "bar description");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_parse_numpydoc_indented_in_docstring() {
        let doc = r#"""
    Summary.

    Parameters
    ----------
    foo : int
        first line
        second line
    """#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first line\nsecond line");
    }

    #[test]
    fn test_parse_numpydoc_type_with_default() {
        let doc = r#"
Parameters
----------
foo, bar : int, default: 1
    shared description
baz : Dict[str, int]
    complex type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "shared description");
        assert_eq!(docs.get("bar").unwrap(), "shared description");
        assert_eq!(docs.get("baz").unwrap(), "complex type");
    }

    #[test]
    fn test_parse_numpydoc_without_description_is_ignored() {
        let doc = r#"
Parameters
----------
foo : int
bar : str
    bar description
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo"), None);
        assert_eq!(docs.get("bar").unwrap(), "bar description");
    }

    #[test]
    fn test_parse_numpydoc_untyped_params() {
        let doc = r#"
Parameters
----------
foo, bar
    shared description
*args
    positional arguments
**kwargs
    keyword arguments
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "shared description");
        assert_eq!(docs.get("bar").unwrap(), "shared description");
        assert_eq!(docs.get("args").unwrap(), "positional arguments");
        assert_eq!(docs.get("kwargs").unwrap(), "keyword arguments");
    }

    #[test]
    fn test_parse_numpydoc_preserves_multiple_paragraphs() {
        let doc = r#"
Parameters
----------
foo : int
    First paragraph.

    Second paragraph.
bar : str
    another
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("foo").unwrap(),
            "First paragraph.\n\nSecond paragraph."
        );
        assert_eq!(docs.get("bar").unwrap(), "another");
    }

    #[test]
    fn test_parse_numpydoc_literal_dashes() {
        let doc = r#"
Summary.

    Parameters
    ----------
    separator : str
        Example separator::

            ----

        Used between records.
    count : int
        Number of records.
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("separator").unwrap(),
            "Example separator::\n\n----\n\nUsed between records."
        );
        assert_eq!(docs.get("count").unwrap(), "Number of records.");
    }

    #[test]
    fn test_parse_numpydoc_nested_heading() {
        let doc = r#"
Parameters
----------
separator : str
    First paragraph.

    Details
    -------
    Used between records.
count : int
    Number of records.
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("separator").unwrap(),
            "First paragraph.\n\nDetails\n-------\nUsed between records."
        );
        assert_eq!(docs.get("count").unwrap(), "Number of records.");
    }

    #[test]
    fn test_parse_numpydoc_misaligned_underline() {
        let doc = r#"
Parameters
    ----------
separator : str
    Used between records.
"#;
        let docs = parse_parameter_documentation(doc);
        assert!(docs.is_empty());
    }

    #[test]
    fn test_parse_numpydoc_indented_section_boundary() {
        let doc = r#"
Summary.

    Parameters
    ----------
    separator : str
        Used between records.

    Other Parameters
    ----------------
    count : int
        Number of records.

    Returns
    -------
    result : str
        Joined records.
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("separator").unwrap(), "Used between records.");
        assert_eq!(docs.get("count").unwrap(), "Number of records.");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_parse_numpydoc_dedented_section_boundary() {
        let doc = r#"
Notes
-----
    Parameters
    ----------
    separator : str
        Used between records.

Returns
-------
result : str
    Joined records.
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("separator").unwrap(), "Used between records.");
        assert_eq!(docs.len(), 1);
    }
}
