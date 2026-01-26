/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;

use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use tree_sitter::Node;
use tree_sitter::Parser;

use crate::module::module_info::ModuleInfo;

const CYTHON_EXTENSIONS: &[&str] = &["pyx", "pxd", "pxi"];

pub(crate) fn is_cython_module(module: &ModuleInfo) -> bool {
    module
        .path()
        .as_path()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| CYTHON_EXTENSIONS.contains(&ext))
}

pub(crate) fn syntax_error_ranges(contents: &str) -> Vec<TextRange> {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_cython::language())
        .is_err()
    {
        return Vec::new();
    }
    let Some(tree) = parser.parse(contents, None) else {
        return Vec::new();
    };
    let mut ranges = BTreeSet::new();
    collect_error_ranges(tree.root_node(), &mut ranges);
    ranges
        .into_iter()
        .map(|(start, end)| TextRange::new(start, end))
        .collect()
}

fn collect_error_ranges(node: Node<'_>, ranges: &mut BTreeSet<(TextSize, TextSize)>) {
    if node.is_error() || node.is_missing() {
        let start = TextSize::new(node.start_byte() as u32);
        let end = TextSize::new(node.end_byte() as u32);
        if start < end {
            ranges.insert((start, end));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_error_ranges(child, ranges);
    }
}
