/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::LazyLock;

use dupe::Dupe;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::module::Module;
use pyrefly_util::lined_buffer::LineNumber;
use regex::Regex;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::ModuleInfo;
use crate::config::error_kind::ErrorKind;
use crate::error::error::Error;

static PYREFLY_IGNORE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"#\s*pyrefly:\s*ignore\s*\[([^\]]*)\]").unwrap());

pub(crate) fn add_pyrefly_ignore_code_action(
    module_info: &ModuleInfo,
    error: &Error,
) -> Option<(String, Module, TextRange, String)> {
    if module_info.is_notebook() || module_info.is_generated() {
        return None;
    }
    if error.error_kind() == ErrorKind::UnusedIgnore {
        return None;
    }
    let error_code = error.error_kind().to_name();
    let title = format!("Add `# pyrefly: ignore [{error_code}]`");
    let error_line = error.display_range().start.line_within_file();
    let (line_range, line_text) = line_text_and_range(module_info, error_line)?;

    if let Some(existing_codes) = pyrefly_ignore_codes(line_text) {
        if existing_codes.iter().any(|code| code == error_code) {
            return None;
        }
        let merged = merge_pyrefly_ignore_codes(existing_codes, error_code);
        let new_comment = format!("# pyrefly: ignore [{}]", merged.join(", "));
        let updated_line = replace_pyrefly_ignore_comment(line_text, &new_comment)?;
        return Some((title, module_info.dupe(), line_range, updated_line));
    }

    if let Some(above_line) = error_line.decrement()
        && let Some((above_range, above_text)) = line_text_and_range(module_info, above_line)
        && above_text.trim_start().starts_with('#')
        && let Some(existing_codes) = pyrefly_ignore_codes(above_text)
    {
        if existing_codes.iter().any(|code| code == error_code) {
            return None;
        }
        let merged = merge_pyrefly_ignore_codes(existing_codes, error_code);
        let new_comment = format!("# pyrefly: ignore [{}]", merged.join(", "));
        let updated_line = replace_pyrefly_ignore_comment(above_text, &new_comment)?;
        return Some((title, module_info.dupe(), above_range, updated_line));
    }

    let insert_range = TextRange::new(line_range.end(), line_range.end());
    let insert_text = format!("  # pyrefly: ignore [{error_code}]");
    Some((title, module_info.dupe(), insert_range, insert_text))
}

fn line_text_and_range(module_info: &ModuleInfo, line: LineNumber) -> Option<(TextRange, &str)> {
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

fn pyrefly_ignore_codes(line: &str) -> Option<Vec<String>> {
    let comment_start = find_comment_start_in_line(line)?;
    let comment_part = &line[comment_start..];
    let captures = PYREFLY_IGNORE_REGEX.captures(comment_part)?;
    let inside = captures.get(1)?.as_str();
    let codes = inside
        .split(',')
        .map(|code| code.trim())
        .filter(|code| !code.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    Some(codes)
}

fn merge_pyrefly_ignore_codes(existing: Vec<String>, new_code: &str) -> Vec<String> {
    let mut all_codes = SmallSet::new();
    for code in existing {
        all_codes.insert(code);
    }
    all_codes.insert(new_code.to_owned());
    let mut merged: Vec<_> = all_codes.into_iter().collect();
    merged.sort();
    merged
}

fn replace_pyrefly_ignore_comment(line: &str, new_comment: &str) -> Option<String> {
    let comment_start = find_comment_start_in_line(line)?;
    let code_part = &line[..comment_start];
    let comment_part = &line[comment_start..];
    if !PYREFLY_IGNORE_REGEX.is_match(comment_part) {
        return None;
    }
    let updated_comment = PYREFLY_IGNORE_REGEX.replace(comment_part, new_comment);
    Some(format!("{code_part}{updated_comment}"))
}
