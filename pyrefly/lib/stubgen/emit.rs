/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Generates `.pyi` stub file text from a `ModuleStub`.

use crate::stubgen::extract::ModuleStub;
use crate::stubgen::extract::StubClass;
use crate::stubgen::extract::StubFunction;
use crate::stubgen::extract::StubItem;
use crate::stubgen::extract::StubParam;
use crate::stubgen::extract::StubVariable;

/// Generate the full text of a `.pyi` stub file from a `ModuleStub`.
pub fn emit_stub(stub: &ModuleStub) -> String {
    let mut out = String::new();

    if stub.uses_self {
        out.push_str("from typing import Self\n");
    }

    if stub.uses_incomplete {
        out.push_str("from _typeshed import Incomplete\n");
    }

    if stub.uses_self || stub.uses_incomplete {
        out.push('\n');
    }

    emit_items(&stub.items, &mut out, "");

    if !out.ends_with('\n') {
        out.push('\n');
    }

    out
}

/// Emit a list of stub items at a given indentation level.
fn emit_items(items: &[StubItem], out: &mut String, indent: &str) {
    let mut prev_kind: Option<ItemKind> = None;

    for item in items {
        let kind = item_kind(item);

        if let Some(prev) = prev_kind {
            let blanks = blank_lines_between(prev, kind, indent);
            for _ in 0..blanks {
                out.push('\n');
            }
        }

        match item {
            StubItem::Import(imp) => {
                out.push_str(indent);
                out.push_str(&imp.text);
                out.push('\n');
            }
            StubItem::Function(func) => {
                emit_function(func, out, indent);
            }
            StubItem::Class(cls) => {
                emit_class(cls, out, indent);
            }
            StubItem::Variable(var) => {
                emit_variable(var, out, indent);
            }
            StubItem::TypeAlias(ta) => {
                out.push_str(indent);
                out.push_str(&ta.text);
                out.push('\n');
            }
        }

        prev_kind = Some(kind);
    }
}

fn emit_function(func: &StubFunction, out: &mut String, indent: &str) {
    for dec in &func.decorators {
        out.push_str(indent);
        out.push_str(dec);
        out.push('\n');
    }

    out.push_str(indent);
    if func.is_async {
        out.push_str("async ");
    }
    out.push_str("def ");
    out.push_str(&func.name);
    if let Some(tp) = &func.type_params {
        out.push_str(tp);
    }
    out.push('(');
    emit_params(&func.params, out);
    out.push(')');

    if let Some(ret) = &func.return_type {
        out.push_str(" -> ");
        out.push_str(ret);
    }

    if let Some(ds) = &func.docstring {
        out.push_str(":\n");
        let body_indent = format!("{}    ", indent);
        out.push_str(&body_indent);
        out.push_str(ds);
        out.push('\n');
        out.push_str(&body_indent);
        out.push_str("...\n");
    } else {
        out.push_str(": ...\n");
    }
}

fn emit_params(params: &[StubParam], out: &mut String) {
    let mut first = true;
    for param in params {
        if !first {
            out.push_str(", ");
        }
        first = false;

        if param.name == "*" || param.name == "/" {
            out.push_str(&param.name);
            continue;
        }

        out.push_str(param.prefix);
        out.push_str(&param.name);
        if let Some(ann) = &param.annotation {
            out.push_str(": ");
            out.push_str(ann);
        }
        if let Some(default) = &param.default {
            if param.annotation.is_some() {
                out.push_str(" = ");
            } else {
                out.push('=');
            }
            out.push_str(default);
        }
    }
}

fn emit_class(cls: &StubClass, out: &mut String, indent: &str) {
    for dec in &cls.decorators {
        out.push_str(indent);
        out.push_str(dec);
        out.push('\n');
    }

    out.push_str(indent);
    out.push_str("class ");
    out.push_str(&cls.name);
    if let Some(tp) = &cls.type_params {
        out.push_str(tp);
    }
    if !cls.bases.is_empty() {
        out.push('(');
        out.push_str(&cls.bases);
        out.push(')');
    }
    out.push_str(":\n");

    let body_indent = format!("{}    ", indent);

    if let Some(ds) = &cls.docstring {
        out.push_str(&body_indent);
        out.push_str(ds);
        out.push('\n');
    }

    if cls.body.is_empty() && cls.docstring.is_none() {
        out.push_str(&body_indent);
        out.push_str("...\n");
    } else {
        emit_items(&cls.body, out, &body_indent);

        // If the class body only had items that were filtered out,
        // emit `...` as the body.
        if cls.body.is_empty() {
            out.push_str(&body_indent);
            out.push_str("...\n");
        }
    }
}

fn emit_variable(var: &StubVariable, out: &mut String, indent: &str) {
    out.push_str(indent);
    out.push_str(&var.name);
    if let Some(ann) = &var.annotation {
        out.push_str(": ");
        out.push_str(ann);
    }
    if let Some(val) = &var.value {
        out.push_str(" = ");
        out.push_str(val);
    }
    out.push('\n');
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Import,
    Function,
    Class,
    Variable,
    TypeAlias,
}

fn item_kind(item: &StubItem) -> ItemKind {
    match item {
        StubItem::Import(_) => ItemKind::Import,
        StubItem::Function(_) => ItemKind::Function,
        StubItem::Class(_) => ItemKind::Class,
        StubItem::Variable(_) => ItemKind::Variable,
        StubItem::TypeAlias(_) => ItemKind::TypeAlias,
    }
}

fn blank_lines_between(prev: ItemKind, next: ItemKind, indent: &str) -> usize {
    let at_top_level = indent.is_empty();
    match (prev, next) {
        (ItemKind::Import, ItemKind::Import) => 0,
        (ItemKind::Variable, ItemKind::Variable) => 0,
        (_, ItemKind::Function) | (_, ItemKind::Class) if at_top_level => 2,
        (ItemKind::Function, _) | (ItemKind::Class, _) if at_top_level => 2,
        _ => 1,
    }
}
