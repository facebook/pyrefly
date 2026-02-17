/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::module::Module;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::ModuleInfo;
use crate::config::error_kind::ErrorKind;
use crate::error::error::Error;
use crate::error::suppress::merge_error_codes;
use crate::error::suppress::parse_ignore_comment;
use crate::error::suppress::replace_ignore_comment;

pub(crate) fn add_pyrefly_ignore_code_action(
    module_info: &ModuleInfo,
    error: &Error,
) -> Option<(String, Module, TextRange, String)> {
    if !should_offer_pyrefly_ignore(module_info, error) {
        return None;
    }
    let error_code = error.error_kind().to_name();
    let title = format!("Add `# pyrefly: ignore [{error_code}]`");
    let error_line = error.display_range().start.line_within_file();
    let (line_range, line_text) = get_line_text_and_range(module_info, error_line)?;

    if has_blank_pyrefly_ignore_comment(line_text) {
        return None;
    }
    if let Some(existing_codes) = parse_ignore_comment(line_text) {
        if existing_codes.iter().any(|code| code == error_code) {
            return None;
        }
        let new_comment = merge_error_codes(existing_codes, &[error_code.to_owned()]);
        let updated_line = replace_ignore_comment(line_text, &new_comment);
        return Some((title, module_info.dupe(), line_range, updated_line));
    }

    if let Some(above_line) = error_line.decrement()
        && let Some((above_range, above_text)) = get_line_text_and_range(module_info, above_line)
        && above_text.trim_start().starts_with('#')
        && !has_blank_pyrefly_ignore_comment(above_text)
        && let Some(existing_codes) = parse_ignore_comment(above_text)
    {
        if existing_codes.iter().any(|code| code == error_code) {
            return None;
        }
        let new_comment = merge_error_codes(existing_codes, &[error_code.to_owned()]);
        let updated_line = replace_ignore_comment(above_text, &new_comment);
        return Some((title, module_info.dupe(), above_range, updated_line));
    }

    let insert_range = TextRange::new(line_range.end(), line_range.end());
    let insert_text = if find_comment_start_in_line(line_text).is_some() {
        format!("  pyrefly: ignore [{error_code}]")
    } else {
        format!("  # pyrefly: ignore [{error_code}]")
    };
    Some((title, module_info.dupe(), insert_range, insert_text))
}

fn should_offer_pyrefly_ignore(module_info: &ModuleInfo, error: &Error) -> bool {
    !module_info.is_notebook()
        && !module_info.is_generated()
        && error.error_kind() != ErrorKind::UnusedIgnore
}

fn get_line_text_and_range(
    module_info: &ModuleInfo,
    line: LineNumber,
) -> Option<(TextRange, &str)> {
    let line_index = line.to_zero_indexed() as usize;
    if line_index >= module_info.lined_buffer().line_count() {
        return None;
    }
    let line_text_full = module_info.lined_buffer().content_in_line_range(line, line);
    let line_text = line_text_full
        .strip_suffix("\r\n")
        .or_else(|| line_text_full.strip_suffix('\n'))
        .unwrap_or(line_text_full);
    let start = module_info.lined_buffer().line_start(line);
    let end = start + TextSize::from(line_text.len() as u32);
    Some((TextRange::new(start, end), line_text))
}

/// Returns true if the line already has a `# pyrefly: ignore` comment without any codes.
fn has_blank_pyrefly_ignore_comment(line: &str) -> bool {
    let Some(comment_start) = find_comment_start_in_line(line) else {
        return false;
    };
    let mut rest = line[comment_start..].trim_start();
    if !rest.starts_with('#') {
        return false;
    }
    rest = rest[1..].trim_start();
    if !rest.starts_with("pyrefly") {
        return false;
    }
    rest = rest["pyrefly".len()..].trim_start();
    if !rest.starts_with(':') {
        return false;
    }
    rest = rest[1..].trim_start();
    if !rest.starts_with("ignore") {
        return false;
    }
    rest = &rest["ignore".len()..];
    let mut chars = rest.chars();
    let Some(first) = chars.next() else {
        return true;
    };
    if first.is_alphanumeric() || first == '-' || first == '_' {
        return false;
    }
    if first.is_whitespace() {
        let trimmed = rest.trim_start();
        return !trimmed.starts_with('[');
    }
    first != '['
}
