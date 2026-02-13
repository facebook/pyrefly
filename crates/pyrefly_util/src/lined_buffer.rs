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

    /// The parser can emit ranges whose end extends past EOF (e.g. unterminated
    /// triple-quoted strings). Clamp to the maximum valid offset so we can still
    /// render a useful location instead of panicking (see #1698).
    pub fn clamp_position(&self, offset: TextSize) -> TextSize {
        let buffer_len = self.buffer.len();
        if offset.to_usize() > buffer_len {
            TextSize::try_from(buffer_len).unwrap()
        } else {
            offset
        }
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
        self.lines.offset(
            SourceLocation {
                line,
                character_offset: OneIndexed::from_zero_indexed(position.character as usize),
            },
            &self.buffer,
            PositionEncoding::Utf16,
        )
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
}
