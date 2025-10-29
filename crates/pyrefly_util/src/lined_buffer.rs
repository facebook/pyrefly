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

    pub fn display_pos(&self, offset: TextSize, notebook: Option<&Notebook>) -> DisplayPos {
        assert!(
            offset.to_usize() <= self.buffer.len(),
            "offset out of range, expected {} <= {}",
            offset.to_usize(),
            self.buffer.len()
        );
        let LineColumn { line, column } = self.lines.line_column(offset, &self.buffer);
        if let Some(notebook) = notebook
            && let Some((cell, cell_line)) =
                map_notebook_line(notebook, LineNumber::from_one_indexed(line))
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

    pub fn to_lsp_range(&self, x: TextRange) -> lsp_types::Range {
        lsp_types::Range::new(
            self.to_lsp_position(x.start()),
            self.to_lsp_position(x.end()),
        )
    }

    pub fn to_lsp_position(&self, x: TextSize) -> lsp_types::Position {
        let loc = self
            .lines
            .source_location(x, &self.buffer, PositionEncoding::Utf16);
        lsp_types::Position {
            line: loc.line.to_zero_indexed() as u32,
            character: loc.character_offset.to_zero_indexed() as u32,
        }
    }

    pub fn from_lsp_position(&self, position: lsp_types::Position) -> TextSize {
        self.lines.offset(
            SourceLocation {
                line: OneIndexed::from_zero_indexed(position.line as usize),
                character_offset: OneIndexed::from_zero_indexed(position.character as usize),
            },
            &self.buffer,
            PositionEncoding::Utf16,
        )
    }

    pub fn from_lsp_range(&self, position: lsp_types::Range) -> TextRange {
        TextRange::new(
            self.from_lsp_position(position.start),
            self.from_lsp_position(position.end),
        )
    }

    pub fn is_ascii(&self) -> bool {
        self.lines.is_ascii()
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

/// Given a one-indexed row in the concatenated source,
/// return the cell number and the row in the cell.
pub fn map_notebook_line(
    notebook: &Notebook,
    line: LineNumber,
) -> Option<(OneIndexed, LineNumber)> {
    let index = notebook.index();
    let one_indexed = line.to_one_indexed();
    let cell = index.cell(one_indexed)?;
    let cell_row = index.cell_row(one_indexed).unwrap_or(OneIndexed::MIN);
    Some((cell, LineNumber::from_one_indexed(cell_row)))
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
}
