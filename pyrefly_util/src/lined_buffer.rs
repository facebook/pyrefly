/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A buffer that tracks line numbers, and deals with positional information.

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;

use ruff_source_file::LineColumn;
use ruff_source_file::LineIndex;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct LinedBuffer {
    pub buffer: Arc<String>,
    pub lines: LineIndex,
}

impl LinedBuffer {
    pub fn new(buffer: Arc<String>) -> Self {
        let lines = LineIndex::from_source_text(&buffer);
        Self { buffer, lines }
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Default)]
pub struct SourceRange {
    pub start: LineColumn,
    pub end: LineColumn,
}

impl Serialize for SourceRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SourceRange", 4)?;
        state.serialize_field("start_line", &self.start.line.get())?;
        state.serialize_field("start_col", &self.start.column.get())?;
        state.serialize_field("end_line", &self.end.line.get())?;
        state.serialize_field("end_col", &self.end.column.get())?;
        state.end()
    }
}

impl Display for SourceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start.line == self.end.line {
            if self.start.column == self.end.column {
                write!(f, "{}:{}", self.start.line, self.start.column)
            } else {
                write!(
                    f,
                    "{}:{}-{}",
                    self.start.line, self.start.column, self.end.column
                )
            }
        } else {
            write!(
                f,
                "{}:{}-{}:{}",
                self.start.line, self.start.column, self.end.line, self.end.column
            )
        }
    }
}
