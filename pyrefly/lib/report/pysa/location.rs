/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::num::NonZeroU32;

use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_text_size::TextRange;
use serde::Serialize;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PysaLocation(DisplayRange);

impl PysaLocation {
    #[cfg(test)]
    pub fn new(range: DisplayRange) -> Self {
        Self(range)
    }

    pub fn as_key(&self) -> String {
        format!(
            "{}:{}-{}:{}",
            self.0.start.line_within_file(),
            self.0.start.column(),
            self.0.end.line_within_file(),
            self.0.end.column()
        )
    }

    pub fn from_text_range(location: TextRange, module: &pyrefly_python::module::Module) -> Self {
        let encoding = ruff_source_file::PositionEncoding::Utf8;
        let lined_buffer = module.lined_buffer();
        let text = lined_buffer.contents();
        let start = lined_buffer
            .line_index()
            .source_location(location.start(), text, encoding);
        let end = lined_buffer
            .line_index()
            .source_location(location.end(), text, encoding);
        let location = DisplayRange {
            start: DisplayPos::Source {
                line: LineNumber::from_one_indexed(start.line),
                column: NonZeroU32::new(start.character_offset.get() as u32).unwrap(),
            },
            end: DisplayPos::Source {
                line: LineNumber::from_one_indexed(end.line),
                column: NonZeroU32::new(end.character_offset.get() as u32).unwrap(),
            },
        };
        Self(location)
    }
}

impl std::fmt::Debug for PysaLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PysaLocation({})", self.as_key())
    }
}

impl Serialize for PysaLocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.as_key())
    }
}
