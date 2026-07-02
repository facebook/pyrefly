/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// Capturing group metadata inferred from a literal regex pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexGroup {
    pub name: Option<String>,
    pub required: bool,
}

/// Parse enough of Python's regex syntax to identify capturing groups and
/// whether they may be skipped. This intentionally mirrors basedmypy's
/// lightweight analysis instead of using Rust's regex parser, whose accepted
/// syntax differs from Python's.
pub fn parse_groups(value: &str, mut verbose: bool) -> Result<Vec<RegexGroup>, String> {
    let bytes = value.as_bytes();
    let mut group_required = Vec::new();
    let mut group_names = Vec::new();
    let mut working = Vec::new();
    let mut depth = 0usize;
    let mut escape = false;
    let mut union: Option<usize> = None;
    let mut character_set = 0u8;
    let mut comment_group = false;
    let mut comment = false;
    let mut backreference = false;
    let mut i = 0;

    while i < bytes.len() {
        let char = bytes[i] as char;
        if escape {
            escape = false;
            i += 1;
            continue;
        }
        if backreference {
            if char == ')' {
                backreference = false;
            }
            i += 1;
            continue;
        }
        if comment_group && char == ')' {
            comment_group = false;
            i += 1;
            continue;
        }
        if char == '\n' && comment {
            comment = false;
            i += 1;
            continue;
        }
        if char == '\\' {
            if character_set != 0 {
                character_set = 2;
            }
            escape = true;
            i += 1;
            continue;
        }
        if comment_group || comment {
            i += 1;
            continue;
        }
        if char == '^' && character_set == 1 {
            i += 1;
            continue;
        }
        if char == ']' && character_set == 2 {
            character_set = 0;
            i += 1;
            continue;
        }
        if character_set != 0 {
            character_set = 2;
            i += 1;
            continue;
        }
        if char == '[' {
            character_set = 1;
            i += 1;
            continue;
        }
        if char == '|' && union.is_none_or(|u| u > depth) {
            union = Some(depth);
            i += 1;
            continue;
        }
        if verbose && char == '#' {
            comment = true;
            i += 1;
            continue;
        }
        match char {
            '(' => {
                depth += 1;
                if bytes.get(i + 1) == Some(&b'?') {
                    if bytes.get(i + 2) == Some(&b'P') && bytes.get(i + 3) == Some(&b'<') {
                        let name_start = i + 4;
                        let Some(name_end) = value[name_start..].find('>').map(|j| name_start + j)
                        else {
                            return Err(format!("missing >, unterminated name at position {i}"));
                        };
                        working.push(Some((group_required.len() + working.len(), true)));
                        group_names.push(Some(value[name_start..name_end].to_owned()));
                    } else if bytes.get(i + 2) == Some(&b'(') {
                        backreference = true;
                        working.push(None);
                    } else if bytes.get(i + 2) == Some(&b'!') {
                        working.push(Some((usize::MAX, false)));
                    } else if bytes.get(i + 2) == Some(&b':') {
                        working.push(None);
                    } else if bytes.get(i + 2) == Some(&b'#') {
                        comment_group = true;
                    } else if let Some(end) = value[i + 2..].find(')') {
                        if value[i + 2..i + 2 + end].contains('x') {
                            verbose = true;
                        } else {
                            working.push(None);
                        }
                    } else {
                        return Err(format!(
                            "missing ), unterminated subpattern at position {i}"
                        ));
                    }
                } else {
                    working.push(Some((group_required.len() + working.len(), true)));
                    group_names.push(None);
                }
            }
            ')' => {
                if depth == 0 {
                    return Err(format!("unbalanced parenthesis at position {i}"));
                }
                depth -= 1;
                if matches!(bytes.get(i + 1), Some(b'*' | b'?')) {
                    for item in working.iter_mut().skip(depth) {
                        if let Some((_, required)) = item {
                            *required = false;
                        }
                    }
                }
                if union == Some(depth + 1)
                    || working
                        .get(depth)
                        .is_some_and(|item| matches!(item, Some((usize::MAX, false))))
                {
                    for item in working.iter_mut().skip(depth + 1) {
                        if let Some((_, required)) = item {
                            *required = false;
                        }
                    }
                    union = None;
                }
                if depth == 0 {
                    for item in working.drain(..).flatten() {
                        if item.0 != usize::MAX {
                            group_required.push(item.1);
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return Err("missing ), unterminated subpattern at position 0".to_owned());
    }
    if union == Some(0) {
        group_required.fill(false);
    }
    Ok(group_required
        .into_iter()
        .enumerate()
        .map(|(i, required)| RegexGroup {
            name: group_names.get(i).cloned().flatten(),
            required,
        })
        .collect())
}
