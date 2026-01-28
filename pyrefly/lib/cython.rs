/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
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

pub(crate) fn completion_items(module: &ModuleInfo, position: TextSize) -> Vec<CompletionItem> {
    let contents = module.contents();
    let Some(base) = attribute_base_at(contents, position) else {
        return keyword_completion_items();
    };
    let index = build_index(contents);
    let members = index
        .member_map_for_base(&base)
        .or_else(|| index.member_map_for_type(&base));
    let Some(members) = members else {
        return Vec::new();
    };
    members
        .iter()
        .map(|(name, kind)| CompletionItem {
            label: name.clone(),
            kind: Some(*kind),
            ..Default::default()
        })
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

#[derive(Default)]
struct CythonIndex {
    class_members: HashMap<String, BTreeMap<String, CompletionItemKind>>,
    struct_members: HashMap<String, BTreeMap<String, CompletionItemKind>>,
    var_types: HashMap<String, String>,
}

impl CythonIndex {
    fn member_map_for_base(&self, base: &str) -> Option<&BTreeMap<String, CompletionItemKind>> {
        let ty = self.var_types.get(base)?;
        self.member_map_for_type(ty)
    }

    fn member_map_for_type(&self, ty: &str) -> Option<&BTreeMap<String, CompletionItemKind>> {
        self.class_members
            .get(ty)
            .or_else(|| self.struct_members.get(ty))
    }
}

fn build_index(contents: &str) -> CythonIndex {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_cython::language())
        .is_err()
    {
        return CythonIndex::default();
    }
    let Some(tree) = parser.parse(contents, None) else {
        return CythonIndex::default();
    };
    let mut index = CythonIndex::default();
    collect_symbols(tree.root_node(), contents, &mut index);
    index
}

fn collect_symbols(node: Node<'_>, source: &str, index: &mut CythonIndex) {
    match node.kind() {
        "class_definition" => {
            if let Some(name) = node_text(source, node.child_by_field_name("name")) {
                let mut members = BTreeMap::new();
                if let Some(body) = node.child_by_field_name("body") {
                    collect_class_members(body, source, &mut members);
                }
                if !members.is_empty() {
                    index.class_members.insert(name, members);
                }
            }
            return;
        }
        "struct" => {
            if let Some(name) = first_identifier_child(source, node) {
                let mut members = BTreeMap::new();
                if let Some(suite) = first_child_kind(node, "struct_suite") {
                    collect_struct_members(suite, source, &mut members);
                }
                if !members.is_empty() {
                    index.struct_members.insert(name, members);
                }
            }
            return;
        }
        "cvar_def" => {
            if let Some((type_name, names)) = cvar_def_type_and_names(node, source) {
                for name in names {
                    index.var_types.insert(name, type_name.clone());
                }
            }
        }
        "cvar_decl" => {
            if let Some((type_name, names)) = cvar_decl_type_and_names(node, source) {
                for name in names {
                    index.var_types.insert(name, type_name.clone());
                }
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_symbols(child, source, index);
    }
}

fn collect_class_members(
    node: Node<'_>,
    source: &str,
    members: &mut BTreeMap<String, CompletionItemKind>,
) {
    match node.kind() {
        "cvar_def" => {
            if let Some((_type_name, names)) = cvar_def_type_and_names(node, source) {
                for name in names {
                    members.entry(name).or_insert(CompletionItemKind::FIELD);
                }
            }
        }
        "cvar_decl" => {
            for name in cvar_decl_names(node, source) {
                members.entry(name).or_insert(CompletionItemKind::FIELD);
            }
        }
        "function_definition" | "c_function_definition" => {
            if let Some(name) = node_text(source, node.child_by_field_name("name")) {
                members.entry(name).or_insert(CompletionItemKind::METHOD);
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(child.kind(), "class_definition" | "struct") {
            continue;
        }
        collect_class_members(child, source, members);
    }
}

fn collect_struct_members(
    node: Node<'_>,
    source: &str,
    members: &mut BTreeMap<String, CompletionItemKind>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "cvar_decl" {
            for name in cvar_decl_names(child, source) {
                members.entry(name).or_insert(CompletionItemKind::FIELD);
            }
        }
    }
}

fn cvar_def_type_and_names(node: Node<'_>, source: &str) -> Option<(String, Vec<String>)> {
    let mut cursor = node.walk();
    let mut maybe_typed_name = None;
    for child in node.children(&mut cursor) {
        if child.kind() == "maybe_typed_name" {
            maybe_typed_name = Some(child);
            break;
        }
    }
    let maybe_typed_name = maybe_typed_name?;
    let type_node = maybe_typed_name.child_by_field_name("type")?;
    let type_name = node_text(source, Some(type_node))?;
    let mut names = Vec::new();
    if let Some(name_node) = maybe_typed_name.child_by_field_name("name") {
        if let Some(name) = node_text(source, Some(name_node)) {
            names.push(name);
        }
    }
    let mut child_cursor = node.walk();
    for child in node.children(&mut child_cursor) {
        if child.kind() == "identifier" {
            if let Some(name) = node_text(source, Some(child)) {
                names.push(name);
            }
        }
    }
    names.sort();
    names.dedup();
    if names.is_empty() {
        None
    } else {
        Some((type_name, names))
    }
}

fn cvar_decl_names(node: Node<'_>, source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            if let Some(name) = node_text(source, Some(child)) {
                names.push(name);
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

fn cvar_decl_type_and_names(node: Node<'_>, source: &str) -> Option<(String, Vec<String>)> {
    let mut cursor = node.walk();
    let mut type_node = None;
    for child in node.children(&mut cursor) {
        if child.kind() == "c_type" {
            type_node = Some(child);
            break;
        }
    }
    let type_name = node_text(source, type_node)?;
    let mut names = cvar_decl_names(node, source);
    names.retain(|name| name != &type_name);
    if names.is_empty() {
        None
    } else {
        Some((type_name, names))
    }
}

fn attribute_base_at(contents: &str, position: TextSize) -> Option<String> {
    let mut pos = position.to_usize().min(contents.len());
    let bytes = contents.as_bytes();
    if pos == 0 {
        return None;
    }
    if pos < contents.len() && bytes[pos] == b'.' {
        pos += 1;
    }
    let mut end = pos;
    while end > 0 && is_ident_char(bytes[end - 1]) {
        end -= 1;
    }
    if end == 0 || bytes[end - 1] != b'.' {
        return None;
    }
    let mut base_end = end - 1;
    while base_end > 0 && bytes[base_end - 1].is_ascii_whitespace() {
        base_end -= 1;
    }
    let mut base_start = base_end;
    while base_start > 0 && is_ident_char(bytes[base_start - 1]) {
        base_start -= 1;
    }
    if base_start == base_end {
        return None;
    }
    Some(contents[base_start..base_end].to_owned())
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn keyword_completion_items() -> Vec<CompletionItem> {
    const KEYWORDS: &[&str] = &[
        "cdef", "cpdef", "cimport", "ctypedef", "cclass", "nogil", "gil", "inline", "extern",
        "public", "api", "readonly", "fused", "except", "noexcept", "struct", "union", "cppclass",
        "enum",
    ];
    KEYWORDS
        .iter()
        .map(|keyword| CompletionItem {
            label: (*keyword).to_owned(),
            kind: Some(CompletionItemKind::KEYWORD),
            ..Default::default()
        })
        .collect()
}

fn node_text(source: &str, node: Option<Node<'_>>) -> Option<String> {
    let node = node?;
    let start = node.start_byte();
    let end = node.end_byte();
    source.get(start..end).map(|s| s.to_owned())
}

fn first_identifier_child(source: &str, node: Node<'_>) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            return node_text(source, Some(child));
        }
    }
    None
}

fn first_child_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == kind)
}
