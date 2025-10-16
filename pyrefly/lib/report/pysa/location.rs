/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(test)]
use std::num::NonZeroU32;

#[cfg(test)]
use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
#[cfg(test)]
use pyrefly_util::lined_buffer::LineNumber;
use serde::Serialize;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PysaLocation(DisplayRange);

impl PysaLocation {
    pub fn new(range: DisplayRange) -> Self {
        Self(range)
    }

    pub fn as_key(&self) -> String {
        format!(
            "{}:{}-{}:{}",
            self.0.start.line, self.0.start.column, self.0.end.line, self.0.end.column
        )
    }

    #[cfg(test)]
    pub fn from_key(key: &str) -> Option<Self> {
        let mut parts = key.split('-');
        let mut start = parts.next()?.split(':');
        let mut end = parts.next()?.split(':');
        if parts.next().is_some() {
            return None;
        }

        let start_line = LineNumber::new(start.next()?.parse::<u32>().ok()?)?;
        let start_column = start.next()?.parse::<NonZeroU32>().ok()?;
        if start.next().is_some() {
            return None;
        }

        let end_line = LineNumber::new(end.next()?.parse::<u32>().ok()?)?;
        let end_column = end.next()?.parse::<NonZeroU32>().ok()?;
        if end.next().is_some() {
            return None;
        }

        Some(Self(DisplayRange {
            start: DisplayPos {
                line: start_line,
                column: start_column,
            },
            end: DisplayPos {
                line: end_line,
                column: end_column,
            },
        }))
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
