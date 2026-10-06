/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::ignore::Tool;
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
    let ErrorQuickFix::RemoveUnusedSuppression(edit) = error.quick_fixes().first()? else {
        return None;
    };
    if module_info.code_at(edit.range) != edit.expected {
        return None;
    }

    let title = if edit.replacement.is_empty() {
        match edit.tool {
            Tool::Type => "Remove unused `# type: ignore` comment",
            Tool::Pyrefly => "Remove unused `# pyrefly: ignore` comment",
            Tool::Pyre => "Remove unused Pyre suppression",
            Tool::Pyright | Tool::Mypy | Tool::Ty | Tool::Zuban => {
                unreachable!(
                    "unused-ignore diagnostics are not emitted for {:?}",
                    edit.tool
                )
            }
        }
    } else {
        "Remove unused suppression codes"
    };

    let mut edit_range = edit.range;
    if edit.replacement.is_empty() {
        let line = error.display_range().start.line_within_file();
        let line_start = module_info.lined_buffer().line_start(line);
        let line_text_full = module_info.lined_buffer().content_in_line_range(line, line);
        let line_text = line_text_full
            .strip_suffix("\r\n")
            .or_else(|| line_text_full.strip_suffix('\n'))
            .unwrap_or(line_text_full);
        let line_end = line_start
            + TextSize::try_from(line_text.len()).expect("source line length must fit in u32");
        let prefix = module_info.code_at(TextRange::new(line_start, edit.range.start()));
        let suffix = module_info.code_at(TextRange::new(edit.range.end(), line_end));

        if prefix.trim().is_empty() && suffix.trim().is_empty() {
            // Avoid leaving a blank line when the suppression is the only comment.
            edit_range = TextRange::new(
                line_start,
                line_start
                    + TextSize::try_from(line_text_full.len())
                        .expect("source line length must fit in u32"),
            );
        } else if suffix.is_empty() {
            // Avoid leaving trailing whitespace after an inline suppression.
            edit_range = TextRange::new(
                line_start
                    + TextSize::try_from(prefix.trim_end().len())
                        .expect("source line length must fit in u32"),
                edit.range.end(),
            );
        }
    }

    Some((
        title.to_owned(),
        module_info.dupe(),
        edit_range,
        edit.replacement.clone(),
    ))
}
