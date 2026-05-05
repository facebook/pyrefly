/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A buffer that tracks line numbers, and deals with positional information.

use std::fmt;
use std::fmt::Display;
use std::num::NonZeroU32;
use std::ops::Deref;
use std::ops::Range;
use std::str::Lines;
use std::sync::Arc;

use parse_display::Display;
use ruff_notebook::Notebook;
use ruff_python_ast::Expr;
use ruff_source_file::LineColumn;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::PositionEncoding;
use ruff_source_file::SourceLocation;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct LinedBuffer {
    buffer: Arc<String>,
    lines: LineIndex,
}

impl LinedBuffer {
    pub fn new(buffer: Arc<String>) -> Self {
        let lines = LineIndex::from_source_text(&buffer);
        Self { buffer, lines }
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub fn contents(&self) -> &Arc<String> {
        &self.buffer
    }

    pub fn line_index(&self) -> &LineIndex {
        &self.lines
    }

    pub fn lines(&self) -> Lines<'_> {
        self.buffer.lines()
    }

    /// Clamp a byte offset so it is safe to pass to `LineIndex::source_location`.
    /// Handles offsets past EOF (#1698) and offsets inside multi-byte UTF-8
    /// characters (#3041).
    pub fn clamp_position(&self, offset: TextSize) -> TextSize {
        let buffer_len = self.buffer.len();
        let mut pos = offset.to_usize();
        if pos > buffer_len {
            pos = buffer_len;
        }
        while pos > 0 && !self.buffer.is_char_boundary(pos) {
            pos -= 1;
        }
        TextSize::try_from(pos).unwrap()
    }

    pub fn display_pos(&self, offset: TextSize, notebook: Option<&Notebook>) -> DisplayPos {
        let offset = self.clamp_position(offset);
        let LineColumn { line, column } = self.lines.line_column(offset, &self.buffer);
        if let Some(notebook) = notebook
            && let Some((cell, cell_line)) = self.get_cell_and_line_from_concatenated_line(
                notebook,
                LineNumber::from_one_indexed(line),
            )
        {
            DisplayPos::Notebook {
                cell: NonZeroU32::new(cell.get() as u32).unwrap(),
                cell_line,
                line: LineNumber::from_one_indexed(line),
                column: NonZeroU32::new(column.get() as u32).unwrap(),
            }
        } else {
            DisplayPos::Source {
                line: LineNumber::from_one_indexed(line),
                column: NonZeroU32::new(column.get() as u32).unwrap(),
            }
        }
    }

    pub fn display_range(&self, range: TextRange, notebook: Option<&Notebook>) -> DisplayRange {
        DisplayRange {
            start: self.display_pos(range.start(), notebook),
            end: self.display_pos(range.end(), notebook),
        }
    }

    pub fn code_at(&self, range: TextRange) -> &str {
        let range = TextRange::new(
            self.clamp_position(range.start()),
            self.clamp_position(range.end()),
        );
        match self.buffer.get(Range::<usize>::from(range)) {
            Some(code) => code,
            None => panic!(
                "`range` is invalid, got {range:?}, but file is {} bytes long",
                self.buffer.len()
            ),
        }
    }

    /// Convert an expression's range into a `PythonASTRange`.
    ///
    /// Generator expressions receive special handling to match the
    /// parenthesized range that CPython's `ast` module reports (ruff
    /// uses the un-parenthesized range for non-parenthesized generators).
    pub fn python_ast_range_for_expr(
        &self,
        original_range: TextRange,
        expr: &Expr,
        parent_expr: Option<&Expr>,
    ) -> PythonASTRange {
        let expression_range = if let Expr::Generator(e) = expr {
            if e.parenthesized {
                original_range
            } else if let Some(Expr::Call(p)) = parent_expr
                && p.arguments.len() == 1
                && p.arguments.inner_range().contains_range(original_range)
            {
                TextRange::new(
                    p.arguments.l_paren_range().start(),
                    p.arguments.r_paren_range().end(),
                )
            } else {
                original_range
                    .sub_start(TextSize::new(1))
                    .add_end(TextSize::new(1))
            }
        } else {
            original_range
        };

        let start_location = self.lines.source_location(
            expression_range.start(),
            &self.buffer,
            PositionEncoding::Utf8,
        );
        let end_location = self.lines.source_location(
            expression_range.end(),
            &self.buffer,
            PositionEncoding::Utf8,
        );

        PythonASTRange {
            start_line: LineNumber::new(start_location.line.get() as u32).unwrap(),
            start_col: start_location.character_offset.to_zero_indexed() as u32,
            end_line: LineNumber::new(end_location.line.get() as u32).unwrap(),
            end_col: end_location.character_offset.to_zero_indexed() as u32,
        }
    }

    /// Convert from a user position to a `TextSize`.
    /// Doesn't take account of a leading BOM, so should be used carefully.
    pub fn from_display_pos(&self, pos: DisplayPos) -> TextSize {
        self.lines.offset(
            SourceLocation {
                line: pos.line_within_file().to_one_indexed(),
                character_offset: OneIndexed::new(pos.column().get() as usize).unwrap(),
            },
            &self.buffer,
            PositionEncoding::Utf32,
        )
    }

    /// Convert from a user range to a `TextRange`.
    /// Doesn't take account of a leading BOM, so should be used carefully.
    pub fn from_display_range(&self, source_range: &DisplayRange) -> TextRange {
        TextRange::new(
            self.from_display_pos(source_range.start),
            self.from_display_pos(source_range.end),
        )
    }

    /// Gets the content from the beginning of start_line to the end of end_line.
    pub fn content_in_line_range(&self, start_line: LineNumber, end_line: LineNumber) -> &str {
        debug_assert!(start_line <= end_line);
        let start = self
            .lines
            .line_start(start_line.to_one_indexed(), &self.buffer);
        let end = self.lines.line_end(end_line.to_one_indexed(), &self.buffer);
        &self.buffer[start.to_usize()..end.to_usize()]
    }

    pub fn line_start(&self, line: LineNumber) -> TextSize {
        self.lines.line_start(line.to_one_indexed(), &self.buffer)
    }

    /// Translates a text range to a LSP range.
    /// For notebook, the input range is relative to the concatenated contents of the whole notebook
    /// and the output range is relative to a specific cell.
    pub fn to_lsp_range(&self, x: TextRange, notebook: Option<&Notebook>) -> lsp_types::Range {
        let start_cell = self.to_cell_for_lsp(x.start(), notebook);
        let end_cell = self.to_cell_for_lsp(x.end(), notebook);
        let start = self.to_lsp_position(x.start(), notebook);
        let mut end = self.to_lsp_position(x.end(), notebook);
        if let Some(start_cell) = start_cell
            && let Some(end_cell) = end_cell
            && end_cell != start_cell
        {
            // If the range spans multiple cells, as can happen when a parse error reaches the next line
            // We should return the "next" line in the same cell, instead of line 0 in the next cell
            end = lsp_types::Position {
                line: start.line + 1,
                character: end.character,
            }
        };
        lsp_types::Range::new(start, end)
    }

    /// Translates a text size to a LSP position.
    /// For notebook, the input position is relative to the concatenated contents of the whole notebook
    /// and the output position is relative to a specific cell.
    pub fn to_lsp_position(&self, x: TextSize, notebook: Option<&Notebook>) -> lsp_types::Position {
        let x = self.clamp_position(x);
        let loc = self
            .lines
            .source_location(x, &self.buffer, PositionEncoding::Utf16);
        if let Some(notebook) = notebook
            && let Some((_, cell_line)) = self.get_cell_and_line_from_concatenated_line(
                notebook,
                LineNumber::from_one_indexed(loc.line),
            )
        {
            lsp_types::Position {
                line: cell_line.to_zero_indexed(),
                character: loc.character_offset.to_zero_indexed() as u32,
            }
        } else {
            lsp_types::Position {
                line: loc.line.to_zero_indexed() as u32,
                character: loc.character_offset.to_zero_indexed() as u32,
            }
        }
    }

    /// If the module is a notebook, take an input position relative to the concatenated contents
    /// and return the index of the corresponding notebook cell.
    pub fn to_cell_for_lsp(&self, x: TextSize, notebook: Option<&Notebook>) -> Option<usize> {
        let x = self.clamp_position(x);
        let loc = self
            .lines
            .source_location(x, &self.buffer, PositionEncoding::Utf16);
        if let Some(notebook) = notebook
            && let Some((cell, _)) = self.get_cell_and_line_from_concatenated_line(
                notebook,
                LineNumber::from_one_indexed(loc.line),
            )
        {
            Some(cell.to_zero_indexed())
        } else {
            None
        }
    }

    /// Translates an LSP position to a text size.
    /// For notebooks, the input position is relative to a notebook cell and the output
    /// position is relative to the concatenated contents of the notebook.
    ///
    /// Per the LSP spec, if the character value is greater than the line length,
    /// it defaults back to the line length.
    pub fn from_lsp_position(
        &self,
        position: lsp_types::Position,
        notebook_and_cell: Option<(&Notebook, usize)>,
    ) -> TextSize {
        let line = if let Some((notebook, cell)) = notebook_and_cell
            && let Some(concatenated_line) = self.get_concatenated_line_from_cell_and_range(
                notebook,
                cell,
                position.line as usize,
            ) {
            concatenated_line.to_one_indexed()
        } else {
            OneIndexed::from_zero_indexed(position.line as usize)
        };
        // Clamp line number to the valid range. The LSP client may send a line
        // number beyond EOF (e.g., when the editor's view is stale after a file
        // truncation). LineIndex::line_start() only handles row_index ==
        // line_count (one-past-the-end) but panics for anything beyond that.
        let max_line = OneIndexed::from_zero_indexed(self.lines.line_count());
        let line = std::cmp::min(line, max_line);
        // Clamp character offset to the line length per the LSP specification:
        // "If the character value is greater than the line length it defaults
        // back to the line length."
        let line_start = self.lines.line_start(line, &self.buffer);
        let line_end = self.lines.line_end(line, &self.buffer);
        let requested = self.lines.offset(
            SourceLocation {
                line,
                character_offset: OneIndexed::from_zero_indexed(position.character as usize),
            },
            &self.buffer,
            PositionEncoding::Utf16,
        );
        // line_end includes the trailing newline. Clamp to the content end
        // (excluding the newline) so that out-of-bounds positions land on the
        // last real character rather than spilling into the next line.
        let content_end = if line_end > line_start
            && self
                .buffer
                .as_bytes()
                .get(line_end.to_usize().saturating_sub(1))
                == Some(&b'\n')
        {
            line_end - TextSize::from(1)
        } else {
            line_end
        };
        std::cmp::min(requested, content_end)
    }

    /// Translates an LSP position to a text range.
    /// For notebooks, the input range is relative to a notebook cell and the output
    /// position is range to the concatenated contents of the notebook.
    pub fn from_lsp_range(
        &self,
        position: lsp_types::Range,
        notebook_and_cell: Option<(&Notebook, usize)>,
    ) -> TextRange {
        TextRange::new(
            self.from_lsp_position(position.start, notebook_and_cell),
            self.from_lsp_position(position.end, notebook_and_cell),
        )
    }

    pub fn is_ascii(&self) -> bool {
        self.lines.is_ascii()
    }

    /// Given a one-indexed row in the concatenated source,
    /// return the cell number and the row in the cell.
    fn get_cell_and_line_from_concatenated_line(
        &self,
        notebook: &Notebook,
        line: LineNumber,
    ) -> Option<(OneIndexed, LineNumber)> {
        let index = notebook.index();
        let one_indexed = line.to_one_indexed();
        let cell = index.cell(one_indexed)?;
        let cell_row = index.cell_row(one_indexed).unwrap_or(OneIndexed::MIN);
        Some((cell, LineNumber::from_one_indexed(cell_row)))
    }

    // Given a zero-indexed cell and zero-indexed line within the cell,
    // return the line number in the concatenated notebook source.
    fn get_concatenated_line_from_cell_and_range(
        &self,
        notebook: &Notebook,
        cell: usize,
        cell_line: usize,
    ) -> Option<LineNumber> {
        let cell_start_offset = notebook.cell_offsets().deref().get(cell)?;
        let cell_start_loc =
            self.lines
                .source_location(*cell_start_offset, &self.buffer, PositionEncoding::Utf16);
        let cell_start_line = cell_start_loc.line.to_zero_indexed();
        Some(LineNumber::from_zero_indexed(
            (cell_start_line + cell_line) as u32,
        ))
    }
}

/// A range in a file, with a start and end, both containing line and column.
/// Stored in terms of characters, not including any BOM.
#[derive(Debug, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Default)]
pub struct DisplayRange {
    pub start: DisplayPos,
    pub end: DisplayPos,
}

impl Serialize for DisplayRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("DisplayRange", 4)?;
        if let Some(start_cell) = &self.start.cell() {
            state.serialize_field("start_cell", &start_cell.get())?;
        }
        state.serialize_field("start_line", &self.start.line_within_cell().0.get())?;
        state.serialize_field("start_col", &self.start.column().get())?;
        if let Some(end_cell) = &self.end.cell() {
            state.serialize_field("end_cell", &end_cell.get())?;
        }
        state.serialize_field("end_line", &self.end.line_within_cell().0.get())?;
        state.serialize_field("end_col", &self.end.column().get())?;
        state.end()
    }
}

impl Display for DisplayRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start.line_within_cell() == self.end.line_within_cell() {
            if self.start.column() == self.end.column() {
                write!(
                    f,
                    "{}:{}",
                    self.start.line_within_cell(),
                    self.start.column()
                )
            } else {
                write!(
                    f,
                    "{}:{}-{}",
                    self.start.line_within_cell(),
                    self.start.column(),
                    self.end.column()
                )
            }
        } else {
            write!(
                f,
                "{}:{}-{}:{}",
                self.start.line_within_cell(),
                self.start.column(),
                self.end.line_within_cell(),
                self.end.column()
            )
        }
    }
}

/// A line number in a file.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Display)]
pub struct LineNumber(NonZeroU32);

impl Default for LineNumber {
    fn default() -> Self {
        Self(NonZeroU32::MIN)
    }
}

impl LineNumber {
    pub fn new(x: u32) -> Option<Self> {
        Some(LineNumber(NonZeroU32::new(x)?))
    }

    pub fn from_zero_indexed(x: u32) -> Self {
        Self(NonZeroU32::MIN.saturating_add(x))
    }

    pub fn to_zero_indexed(self) -> u32 {
        self.0.get() - 1
    }

    pub fn from_one_indexed(x: OneIndexed) -> Self {
        Self(NonZeroU32::new(x.get().try_into().unwrap()).unwrap())
    }

    pub fn to_one_indexed(self) -> OneIndexed {
        OneIndexed::new(self.0.get() as usize).unwrap()
    }

    pub fn decrement(&self) -> Option<Self> {
        Self::new(self.0.get() - 1)
    }

    pub fn increment(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// Source location in Python AST conventions: 1-indexed lines, 0-indexed columns.
///
/// Matches the `lineno`/`col_offset`/`end_lineno`/`end_col_offset` fields that
/// CPython's `ast` module exposes on expression nodes.
#[derive(Debug, Clone)]
pub struct PythonASTRange {
    pub start_line: LineNumber,
    pub start_col: u32,
    pub end_line: LineNumber,
    pub end_col: u32,
}

impl Serialize for PythonASTRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PythonASTRange", 4)?;
        state.serialize_field("start_line", &self.start_line.get())?;
        state.serialize_field("start_col", &self.start_col)?;
        state.serialize_field("end_line", &self.end_line.get())?;
        state.serialize_field("end_col", &self.end_col)?;
        state.end()
    }
}

/// The line and column of an offset in a source file.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DisplayPos {
    Source {
        /// The line in the source text.
        line: LineNumber,
        /// The column (UTF scalar values) relative to the start of the line except any
        /// potential BOM on the first line.
        column: NonZeroU32,
    },
    Notebook {
        cell: NonZeroU32,
        // The line within the cell
        cell_line: LineNumber,
        // The line within the concatenated source
        line: LineNumber,
        column: NonZeroU32,
    },
}

impl Default for DisplayPos {
    fn default() -> Self {
        Self::Source {
            line: LineNumber::default(),
            column: NonZeroU32::MIN,
        }
    }
}

impl Display for DisplayPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source { line, column } => {
                write!(f, "{}:{}", line, column)
            }
            Self::Notebook {
                cell,
                cell_line,
                column,
                ..
            } => {
                write!(f, "{}:{}:{}", cell, cell_line, column)
            }
        }
    }
}

impl DisplayPos {
    // Get the line number within the file, or the line number within the cell
    // for notebooks
    pub fn line_within_cell(self) -> LineNumber {
        match self {
            Self::Source { line, .. } => line,
            Self::Notebook { cell_line, .. } => cell_line,
        }
    }

    // Get the line number within the file, using the position in the
    // concatenated source for notebooks
    pub fn line_within_file(self) -> LineNumber {
        match self {
            Self::Source { line, .. } => line,
            Self::Notebook { line, .. } => line,
        }
    }

    pub fn column(self) -> NonZeroU32 {
        match self {
            Self::Source { column, .. } => column,
            Self::Notebook { column, .. } => column,
        }
    }

    pub fn cell(self) -> Option<NonZeroU32> {
        match self {
            Self::Notebook { cell, .. } => Some(cell),
            Self::Source { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_line_buffer_unicode() {
        // Test with a mix of ASCII, accented characters, and emoji
        let contents =
            "def greet(name: str) -> str:\n    return f\"Bonjour {name}! 👋 Café? ☕\"\n# done\n";
        let lined_buffer = LinedBuffer::new(Arc::new(contents.to_owned()));

        assert_eq!(lined_buffer.line_count(), 4);

        let range = |l1, c1, l2, c2| DisplayRange {
            start: DisplayPos::Source {
                line: LineNumber::from_zero_indexed(l1),
                column: NonZeroU32::new(c1 + 1u32).unwrap(),
            },
            end: DisplayPos::Source {
                line: LineNumber::from_zero_indexed(l2),
                column: NonZeroU32::new(c2 + 1u32).unwrap(),
            },
        };

        assert_eq!(
            lined_buffer.code_at(lined_buffer.from_display_range(&range(1, 4, 2, 0))),
            "return f\"Bonjour {name}! 👋 Café? ☕\"\n"
        );

        assert_eq!(
            lined_buffer.code_at(lined_buffer.from_display_range(&range(1, 29, 1, 36))),
            "👋 Café?"
        );
        assert_eq!(
            lined_buffer.code_at(lined_buffer.from_display_range(&range(2, 2, 2, 4))),
            "do"
        );
    }

    #[test]
    fn test_display_pos_clamps_out_of_range_offset() {
        let contents = Arc::new("i:\"\"\"".to_owned());
        let lined_buffer = LinedBuffer::new(Arc::clone(&contents));
        let eof = TextSize::new(contents.len() as u32);
        let past_eof = eof.checked_add(TextSize::from(1)).unwrap();
        assert_eq!(
            lined_buffer.display_pos(eof, None),
            lined_buffer.display_pos(past_eof, None)
        );
    }

    /// Regression test: `to_lsp_position` and `to_cell_for_lsp` must not panic
    /// when given an offset past the end of the buffer. `display_pos` already
    /// clamps, but these two methods call `source_location` without clamping,
    /// which triggers an out-of-bounds panic in `LineIndex::source_location`.
    /// See the `workspace_symbols` panic in ruff_source_file/line_index.rs.
    ///
    /// The content must include non-ASCII characters because the panic occurs
    /// in the non-ASCII code path of `source_location`, which slices the text
    /// with `&text[line_start..offset]`. The ASCII fast-path only does
    /// arithmetic and does not slice, so it would not trigger the panic.
    #[test]
    fn test_to_lsp_position_out_of_range_offset() {
        // Non-ASCII content to trigger the UTF-16 string-slicing code path
        let contents = Arc::new("def café():\n    pass\n".to_owned());
        let lined_buffer = LinedBuffer::new(Arc::clone(&contents));
        let past_eof = TextSize::new(contents.len() as u32 + 100);
        // This should not panic - it should clamp to the end of the buffer.
        let _pos = lined_buffer.to_lsp_position(past_eof, None);
    }

    /// Same as above but for `to_cell_for_lsp`. Even for non-notebook files,
    /// `to_cell_for_lsp` calls `source_location` unconditionally before
    /// checking whether a notebook is present.
    #[test]
    fn test_to_cell_for_lsp_out_of_range_offset() {
        let contents = Arc::new("def café():\n    pass\n".to_owned());
        let lined_buffer = LinedBuffer::new(Arc::clone(&contents));
        let past_eof = TextSize::new(contents.len() as u32 + 100);
        // This should not panic - it should clamp to the end of the buffer.
        let _cell = lined_buffer.to_cell_for_lsp(past_eof, None);
    }

    /// Same as above but for `to_lsp_range`, which calls `to_lsp_position`
    /// for both start and end of the range.
    #[test]
    fn test_to_lsp_range_out_of_range_offset() {
        let contents = Arc::new("def café():\n    pass\n".to_owned());
        let lined_buffer = LinedBuffer::new(Arc::clone(&contents));
        let past_eof = TextSize::new(contents.len() as u32 + 100);
        let range = TextRange::new(TextSize::new(0), past_eof);
        // This should not panic - it should clamp to the end of the buffer.
        let _lsp_range = lined_buffer.to_lsp_range(range, None);
    }

    /// Regression test: `from_lsp_position` must not panic when the LSP client
    /// sends a position with a line number beyond the end of the buffer. This
    /// can happen when the editor's view of the file is stale (e.g., after a
    /// DidChangeTextDocument race where the file was truncated). The LSP spec
    /// says out-of-range positions should be clamped, not crash the server.
    ///
    /// This reproduces the crash reported in Pyrefly 0.60 where a
    /// `textDocument/codeAction` request triggered:
    ///   "index out of bounds: the len is 13 but the index is 14"
    /// in `LineIndex::line_start()` via `LinedBuffer::from_lsp_position()`.
    #[test]
    fn test_from_lsp_position_out_of_range_line() {
        let contents = Arc::new("def foo():\n    pass\n".to_owned());
        let lined_buffer = LinedBuffer::new(Arc::clone(&contents));
        let position = lsp_types::Position {
            line: 100,
            character: 0,
        };
        // Should clamp to EOF, not panic.
        let offset = lined_buffer.from_lsp_position(position, None);
        assert_eq!(offset, TextSize::new(contents.len() as u32));
    }

    /// Bug: `LspNotebook::get_code_cell_index` returns an index among ALL cells
    /// (code + markdown), but `Notebook::cell_offsets()` is indexed by valid
    /// CODE cells only. When a notebook has markdown cells interspersed with
    /// code cells, the all-cells index doesn't match the code-cell index,
    /// causing `from_lsp_position` to resolve to the wrong offset or panic.
    ///
    /// For a notebook [code_0, markdown, code_1]:
    ///   - `get_code_cell_index` returns 2 for code_1 (its position among all cells)
    ///   - `cell_offsets` has 3 entries: [0, end_of_code_0, total_len]
    ///   - `cell_offsets[2]` is the trailing sentinel (source length), not the
    ///     start of code_1 — giving the wrong concatenated line
    #[test]
    fn test_from_lsp_position_notebook_cell_index_mismatch() {
        use ruff_notebook::Cell;
        use ruff_notebook::CellMetadata;
        use ruff_notebook::CodeCell;
        use ruff_notebook::MarkdownCell;
        use ruff_notebook::Notebook;
        use ruff_notebook::RawNotebook;
        use ruff_notebook::RawNotebookMetadata;
        use ruff_notebook::SourceValue;

        let raw = RawNotebook {
            cells: vec![
                Cell::Code(CodeCell {
                    execution_count: None,
                    id: None,
                    metadata: CellMetadata::default(),
                    outputs: vec![],
                    source: SourceValue::String("x = 1".to_owned()),
                }),
                Cell::Markdown(MarkdownCell {
                    attachments: None,
                    id: None,
                    metadata: CellMetadata::default(),
                    source: SourceValue::String("# heading".to_owned()),
                }),
                Cell::Code(CodeCell {
                    execution_count: None,
                    id: None,
                    metadata: CellMetadata::default(),
                    outputs: vec![],
                    source: SourceValue::String("y = 2".to_owned()),
                }),
            ],
            metadata: RawNotebookMetadata::default(),
            nbformat: 4,
            nbformat_minor: 5,
        };
        let notebook = Notebook::from_raw_notebook(raw, false).unwrap();
        // source_code() concatenates only code cells: "x = 1\ny = 2\n"
        let source = notebook.source_code().to_owned();
        assert_eq!(source, "x = 1\ny = 2\n");
        let lined_buffer = LinedBuffer::new(Arc::new(source.clone()));

        let position = lsp_types::Position {
            line: 0,
            character: 0,
        };

        // Correct: code_1 is at code-cell index 1, so cell_offsets[1] points
        // to the start of "y = 2".
        let correct_offset = lined_buffer.from_lsp_position(position, Some((&notebook, 1)));
        assert_eq!(correct_offset, TextSize::new(6)); // offset of 'y'
        assert_eq!(
            &source[correct_offset.to_usize()..correct_offset.to_usize() + 5],
            "y = 2"
        );

        // If the all-cells index (2) were used instead of the code-cell index
        // (1), cell_offsets[2] would be the trailing sentinel (12 = source
        // length) and from_lsp_position would resolve to EOF. This is
        // prevented by LspNotebook::get_code_cell_index translating to the
        // code-cell index.
        let wrong_offset = lined_buffer.from_lsp_position(position, Some((&notebook, 2)));
        assert_ne!(
            correct_offset, wrong_offset,
            "all-cells index 2 must differ from code-cell index 1 — \
             callers must translate via get_code_cell_index"
        );
    }
}
