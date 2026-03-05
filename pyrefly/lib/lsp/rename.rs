/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::BytesLiteral;
use ruff_python_ast::Expr;
use ruff_python_ast::Mod;
use ruff_python_ast::StringFlags;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_parser::Mode;
use ruff_python_parser::ParseOptions;
use ruff_python_parser::TokenKind;
use ruff_python_parser::parse_unchecked;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

pub(crate) fn comment_and_string_content_ranges(source: &str) -> Vec<TextRange> {
    if source.is_empty() {
        return Vec::new();
    }

    let parsed = parse_unchecked(source, ParseOptions::from(Mode::Module));
    let mut ranges = Vec::new();

    for token in parsed.tokens() {
        if token.kind() == TokenKind::Comment {
            ranges.push(token.range());
        }
    }

    let mut collector = StringContentRangeCollector::default();
    match parsed.syntax() {
        Mod::Module(module) => {
            for stmt in &module.body {
                collector.visit_stmt(stmt);
            }
        }
        Mod::Expression(expr) => collector.visit_expr(&expr.body),
    }
    ranges.extend(collector.ranges);

    ranges
}

#[derive(Default)]
struct StringContentRangeCollector {
    ranges: Vec<TextRange>,
}

impl<'a> Visitor<'a> for StringContentRangeCollector {
    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::StringLiteral(expr) => {
                for literal in expr.value.iter() {
                    self.ranges.push(literal.content_range());
                }
            }
            Expr::BytesLiteral(expr) => {
                for literal in expr.value.iter() {
                    if let Some(range) = bytes_literal_content_range(literal) {
                        self.ranges.push(range);
                    }
                }
            }
            Expr::FString(expr) => {
                for element in expr.value.elements() {
                    if let Some(literal) = element.as_literal() {
                        self.ranges.push(literal.range);
                    }
                }
            }
            Expr::TString(expr) => {
                for element in expr.value.elements() {
                    if let Some(literal) = element.as_literal() {
                        self.ranges.push(literal.range);
                    }
                }
            }
            _ => {}
        }

        walk_expr(self, expr);
    }
}

fn bytes_literal_content_range(literal: &BytesLiteral) -> Option<TextRange> {
    let start = literal.range.start() + literal.flags.opener_len();
    let end = literal
        .range
        .end()
        .saturating_sub(literal.flags.closer_len());
    if start >= end {
        return None;
    }
    Some(TextRange::new(start, end))
}

pub(crate) fn find_word_occurrences(source: &str, word: &str) -> Vec<TextRange> {
    if word.is_empty() {
        return Vec::new();
    }
    let full_range = TextRange::new(TextSize::from(0), TextSize::from(source.len() as u32));
    find_word_occurrences_in_ranges(source, word, &[full_range])
}

pub(crate) fn find_word_occurrences_in_ranges(
    source: &str,
    word: &str,
    ranges: &[TextRange],
) -> Vec<TextRange> {
    if word.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let word_len = word.len();

    for range in ranges {
        let start = range.start().to_usize();
        let end = range.end().to_usize().min(source.len());
        if start >= end {
            continue;
        }
        let slice = &source[start..end];
        for (rel_idx, _) in slice.match_indices(word) {
            let absolute = start + rel_idx;
            let before = source[..absolute].chars().last();
            let after = source[absolute + word_len..].chars().next();
            if before.is_some_and(is_identifier_char) || after.is_some_and(is_identifier_char) {
                continue;
            }
            matches.push(TextRange::new(
                TextSize::from(absolute as u32),
                TextSize::from((absolute + word_len) as u32),
            ));
        }
    }

    matches
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::comment_and_string_content_ranges;
    use super::find_word_occurrences_in_ranges;

    #[test]
    fn find_occurrences_in_comments_and_strings() {
        let source = "foo = 1\n# foo\ns = \"foo\"\nf\"{foo} foo\"\n";
        let ranges = comment_and_string_content_ranges(source);
        let hits = find_word_occurrences_in_ranges(source, "foo", &ranges);

        assert_eq!(hits.len(), 3);
        for range in hits {
            let start = range.start().to_usize();
            let end = range.end().to_usize();
            assert_eq!(&source[start..end], "foo");
        }
    }

    #[test]
    fn ignores_identifier_substrings() {
        let source = "# foobar foo\n\"foo_bar\"\n";
        let ranges = comment_and_string_content_ranges(source);
        let hits = find_word_occurrences_in_ranges(source, "foo", &ranges);

        assert_eq!(hits.len(), 1);
        let range = hits[0];
        let start = range.start().to_usize();
        let end = range.end().to_usize();
        assert_eq!(&source[start..end], "foo");
    }

    #[test]
    fn skips_string_prefix_and_interpolations() {
        let source = "s = r\"foo\"\nf\"{foo} bar\"\n";
        let ranges = comment_and_string_content_ranges(source);
        let r_hits = find_word_occurrences_in_ranges(source, "r", &ranges);
        let foo_hits = find_word_occurrences_in_ranges(source, "foo", &ranges);

        assert!(r_hits.is_empty());
        assert_eq!(foo_hits.len(), 1);
        let range = foo_hits[0];
        let start = range.start().to_usize();
        let end = range.end().to_usize();
        assert_eq!(&source[start..end], "foo");
    }
}
