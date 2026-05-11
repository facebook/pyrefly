/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Human-readable rendering of CinderX type reports.
//!
//! Produces `.txt` files alongside the JSON type reports, with each
//! expression location followed by its inlined type (indices resolved to
//! their full structures). Mirrors the logic of `view_types.py`.

use crate::report::cinderx::types::LocatedType;
use crate::report::cinderx::types::StructuredType;
use crate::report::cinderx::types::TypeTableEntry;

/// Format a single type by recursively inlining all referenced type table entries.
fn format_type(entries: &[TypeTableEntry], idx: usize) -> String {
    match &entries[idx].ty {
        StructuredType::Class {
            qname,
            args,
            traits,
        } => {
            let mut s = qname.clone();
            if !args.is_empty() {
                let args_str = args
                    .iter()
                    .map(|&a| format_type(entries, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                s.push('[');
                s.push_str(&args_str);
                s.push(']');
            }
            if !traits.is_empty() {
                s.push_str(" <");
                s.push_str(&traits.join(", "));
                s.push('>');
            }
            s
        }
        StructuredType::Callable {
            params,
            return_type,
            defining_func,
        } => {
            let params_str = params
                .iter()
                .map(|&p| format_type(entries, p))
                .collect::<Vec<_>>()
                .join(", ");
            let ret = format_type(entries, *return_type);
            match defining_func {
                Some(df) => format!("{}({}) -> {}", df, params_str, ret),
                None => format!("({}) -> {}", params_str, ret),
            }
        }
        StructuredType::OtherForm { qname, args } => {
            let mut s = qname.clone();
            if !args.is_empty() {
                let args_str = args
                    .iter()
                    .map(|&a| format_type(entries, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                s.push('[');
                s.push_str(&args_str);
                s.push(']');
            }
            s
        }
        StructuredType::BoundMethod {
            self_type,
            func_type,
            defining_class,
        } => {
            let self_str = format_type(entries, *self_type);
            let func_str = format_type(entries, *func_type);
            let dc = defining_class.as_deref().unwrap_or("?");
            format!("BoundMethod[{}]({}, {})", dc, self_str, func_str)
        }
        StructuredType::Variable { name, bounds } => {
            let mut s = name.clone();
            if !bounds.is_empty() {
                let bounds_str = bounds
                    .iter()
                    .map(|&b| format_type(entries, b))
                    .collect::<Vec<_>>()
                    .join(", ");
                s.push_str(": ");
                s.push_str(&bounds_str);
            }
            s
        }
        StructuredType::Literal {
            value,
            promoted_type,
        } => {
            format!(
                "Literal[{}] -> {}",
                value,
                format_type(entries, *promoted_type)
            )
        }
    }
}

/// Format the source location of a located type as `line:start_col-end_col`.
///
/// Uses the compact single-line form `line:start-end` when the range fits on
/// one line, or `start_line:col-end_line:col` for multi-line ranges.
/// Omits the end column when it equals the start (i.e., a point location).
fn format_location(loc: &LocatedType) -> String {
    let r = &loc.location;
    if r.start_line == r.end_line {
        if r.start_col == r.end_col {
            format!("{}:{}", r.start_line, r.start_col)
        } else {
            format!("{}:{}-{}", r.start_line, r.start_col, r.end_col)
        }
    } else {
        format!(
            "{}:{}-{}:{}",
            r.start_line, r.start_col, r.end_line, r.end_col
        )
    }
}

/// Render the per-module type data as a human-readable string.
///
/// Each located expression produces one line `  LOC: TYPE`, with optional
/// indented lines for `unnarrowed` and `contextual` types when present.
pub(crate) fn format_module_types(entries: &[TypeTableEntry], locations: &[LocatedType]) -> String {
    let mut out = String::new();
    for loc in locations {
        let loc_str = format_location(loc);
        let type_str = format_type(entries, loc.type_index);
        out.push_str(&format!("  {}: {}\n", loc_str, type_str));
        if let Some(unnarrowed_idx) = loc.unnarrowed_type {
            let unnarrowed_str = format_type(entries, unnarrowed_idx);
            if loc.is_narrowed_mismatch {
                out.push_str(&format!("    unnarrowed: {} (mismatch)\n", unnarrowed_str));
            } else {
                out.push_str(&format!("    unnarrowed: {}\n", unnarrowed_str));
            }
        }
        if let Some(ctx_idx) = loc.contextual_type {
            let ctx_str = format_type(entries, ctx_idx);
            out.push_str(&format!("    contextual: {}\n", ctx_str));
        }
    }
    out
}
