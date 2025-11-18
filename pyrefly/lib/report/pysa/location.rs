/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::lined_buffer::DisplayRange;
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
            self.0.start.line_within_file(),
            self.0.start.column(),
            self.0.end.line_within_file(),
            self.0.end.column()
        )
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
