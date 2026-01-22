/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Support for comment-based sections in Python files.
//!
//! This module provides support for structured comment headers that can be used
//! for code organization, similar to the convention in R:
//! - `# Section 1 ----`
//! - `## Section 1.1 ----`
//! - `### Section 1.1.1 ----`
//!
//! These comments are recognized by editors to provide:
//! - Code folding
//! - Document outline/symbols
//!
//! The pattern matches lines that:
//! - Start with one or more '#' (the count determines the level)
//! - Followed by whitespace and some content
//! - End with 4 or more '-', '#', or '=' characters

use std::sync::LazyLock;

use regex::Regex;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::module::Module;

/// Regular expression to match comment sections.
/// Captures:
/// - Group 1: The leading '#' characters (determines level)
/// - Group 2: The title text (without trailing delimiters)
/// - Group 3: The trailing delimiter characters (-----, ####, or =====)
static RE_COMMENT_SECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(#+)\s+(.+?)\s+([-#=]{4,})\s*$").expect("Invalid regex pattern")
});

/// Represents a parsed comment section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentSection {
    /// The hierarchical level based on number of '#' characters (1-based)
    pub level: usize,
    /// The title/name of the section
    pub title: String,
    /// The text range of the comment line in the source
    pub range: TextRange,
    /// The line number (0-based)
    pub line_number: u32,
}

impl CommentSection {
    /// Parse a single line to check if it's a comment section.
    /// Returns Some(CommentSection) if the line matches the pattern.
    ///
    /// Note: Returns None if the line length exceeds TextSize limits (extremely rare
    /// for comment lines, which are typically short).
    pub fn parse(line: &str, line_number: u32, line_start: TextSize) -> Option<Self> {
        let captures = RE_COMMENT_SECTION.captures(line)?;

        let hashes = captures.get(1)?.as_str();
        let level = hashes.len();
        let title = captures.get(2)?.as_str().trim().to_owned();

        // Calculate the range for the entire line
        // If the line is too long to fit in TextSize (> u32::MAX bytes), we skip it.
        // This is extremely unlikely for a comment line.
        let line_len = TextSize::try_from(line.len()).ok()?;
        let range = TextRange::new(line_start, line_start + line_len);

        Some(CommentSection {
            level,
            title,
            range,
            line_number,
        })
    }

    /// Extract all comment sections from a module.
    pub fn extract_from_module(module: &Module) -> Vec<CommentSection> {
        let mut sections = Vec::new();
        let lined_buffer = module.lined_buffer();
        for (line_number, line) in lined_buffer.lines().enumerate() {
            let line_number = line_number as u32;
            let line_start = lined_buffer.line_start(
                pyrefly_util::lined_buffer::LineNumber::from_zero_indexed(line_number),
            );
            if let Some(section) = Self::parse(line, line_number, line_start) {
                sections.push(section);
            }
        }
        sections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_hash_section() {
        let line = "# Section 1 ----";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_some());
        let section = section.unwrap();
        assert_eq!(section.level, 1);
        assert_eq!(section.title, "Section 1");
        assert_eq!(section.line_number, 0);
    }

    #[test]
    fn test_parse_double_hash_section() {
        let line = "## Section 1.1 ----";
        let section = CommentSection::parse(line, 5, TextSize::from(100));
        assert!(section.is_some());
        let section = section.unwrap();
        assert_eq!(section.level, 2);
        assert_eq!(section.title, "Section 1.1");
        assert_eq!(section.line_number, 5);
    }

    #[test]
    fn test_parse_with_different_delimiters() {
        // Test with hashes
        let line = "# Section ####";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_some());
        assert_eq!(section.unwrap().title, "Section");

        // Test with equals
        let line = "# Section ====";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_some());
        assert_eq!(section.unwrap().title, "Section");

        // Test with dashes
        let line = "# Section ----";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_some());
        assert_eq!(section.unwrap().title, "Section");
    }

    #[test]
    fn test_parse_requires_at_least_four_delimiters() {
        let line = "# Section ---"; // Only 3 dashes
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_none());

        let line = "# Section ----"; // 4 dashes - should work
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_some());
    }

    #[test]
    fn test_parse_non_section_comments() {
        // Regular comment without delimiters
        let line = "# This is just a regular comment";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_none());

        // Comment with delimiter but no content
        let line = "# ----";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_none());

        // Not a comment at all
        let line = "def foo(): pass";
        let section = CommentSection::parse(line, 0, TextSize::from(0));
        assert!(section.is_none());
    }

    #[test]
    fn test_parse_triple_hash_section() {
        let line = "### Section 1.1.1 ----";
        let section = CommentSection::parse(line, 10, TextSize::from(200));
        assert!(section.is_some());
        let section = section.unwrap();
        assert_eq!(section.level, 3);
        assert_eq!(section.title, "Section 1.1.1");
    }
}
