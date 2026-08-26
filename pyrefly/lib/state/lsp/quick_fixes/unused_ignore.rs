/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::ignore::apply_suppression_edits_to_line;
use pyrefly_python::module::Module;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::ModuleInfo;
use crate::error::error::Error;
use crate::error::error::ErrorQuickFix;

pub(crate) fn remove_unused_ignore_code_action(
    module_info: &ModuleInfo,
    error: &Error,
) -> Option<(String, Module, TextRange, String)> {
    if module_info.is_notebook() || module_info.is_generated() {
        return None;
    }

    let edit = error.quick_fixes().iter().find_map(|fix| match fix {
        ErrorQuickFix::RemoveUnusedSuppression(edit) => Some(edit),
        ErrorQuickFix::ReplaceWithEnumMember { .. } | ErrorQuickFix::AssertNotNone => None,
    })?;
    let title = if edit.replacement.is_empty() {
        "Remove suppression comment"
    } else {
        "Remove unused suppression codes"
    };
    let buffer = module_info.lined_buffer();
    let line = buffer
        .display_pos(edit.range.start(), None)
        .line_within_file();
    let line_start = buffer.line_start(line);
    let line_text_full = buffer.content_in_line_range(line, line);
    let line_text = line_text_full
        .strip_suffix("\r\n")
        .or_else(|| line_text_full.strip_suffix('\n'))
        .or_else(|| line_text_full.strip_suffix('\r'))
        .unwrap_or(line_text_full);
    let replacement = apply_suppression_edits_to_line(
        line_text,
        std::iter::once((
            (edit.range.start() - line_start).to_usize()
                ..(edit.range.end() - line_start).to_usize(),
            edit.replacement.as_str(),
        )),
    );
    let edit_range = TextRange::new(
        line_start,
        line_start
            + TextSize::of(if replacement.is_empty() {
                line_text_full
            } else {
                line_text
            }),
    );

    Some((
        title.to_owned(),
        module_info.dupe(),
        edit_range,
        replacement,
    ))
}
