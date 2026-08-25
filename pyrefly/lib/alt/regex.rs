/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RegexValidationError {
    Invalid(&'static str),
    Unsupported,
}

enum InlineFlags {
    Global { end: usize, verbose: bool },
    Scoped { end: usize, verbose: bool },
}

/// Validate structural errors in the subset of Python regex syntax understood here.
/// Unknown extensions are left to Python at runtime rather than risking a false positive.
pub(crate) fn validate_pattern(
    pattern: &[u8],
    mut verbose: bool,
) -> Result<(), RegexValidationError> {
    let mut open_groups = Vec::new();
    let mut character_set_can_close = None;
    let mut escaped = false;
    let mut i = 0;

    while i < pattern.len() {
        let byte = pattern[i];
        if escaped {
            escaped = false;
            if let Some(can_close) = &mut character_set_can_close {
                *can_close = true;
            }
            i += 1;
            continue;
        }
        if byte == b'\\' {
            escaped = true;
            i += 1;
            continue;
        }
        if let Some(can_close) = character_set_can_close {
            if byte == b']' && can_close {
                character_set_can_close = None;
            } else if byte != b'^' || can_close {
                character_set_can_close = Some(true);
            }
            i += 1;
            continue;
        }
        if byte == b'[' {
            character_set_can_close = Some(false);
            i += 1;
            continue;
        }
        if verbose && byte == b'#' {
            i = pattern[i..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(pattern.len(), |offset| i + offset + 1);
            continue;
        }

        match byte {
            b'(' if pattern.get(i + 1) != Some(&b'?') => {
                open_groups.push(verbose);
                i += 1;
            }
            b'(' => match pattern.get(i + 2) {
                Some(b'#') => {
                    let Some(end) = pattern[i + 3..].iter().position(|byte| *byte == b')') else {
                        return Err(RegexValidationError::Invalid(
                            "missing ), unterminated comment",
                        ));
                    };
                    i += end + 4;
                }
                Some(b'P') if pattern.get(i + 3) == Some(&b'<') => {
                    let Some(end) = pattern[i + 4..].iter().position(|byte| *byte == b'>') else {
                        return Err(RegexValidationError::Invalid(
                            "missing >, unterminated name",
                        ));
                    };
                    open_groups.push(verbose);
                    i += end + 5;
                }
                Some(b'P') if pattern.get(i + 3) == Some(&b'=') => {
                    open_groups.push(verbose);
                    i += 4;
                }
                Some(b'(') => {
                    let Some(end) = pattern[i + 3..].iter().position(|byte| *byte == b')') else {
                        return Err(RegexValidationError::Invalid(
                            "missing ), unterminated conditional",
                        ));
                    };
                    open_groups.push(verbose);
                    i += end + 4;
                }
                Some(b':' | b'=' | b'!' | b'>') => {
                    open_groups.push(verbose);
                    i += 3;
                }
                Some(b'<') if matches!(pattern.get(i + 3), Some(b'=' | b'!')) => {
                    open_groups.push(verbose);
                    i += 4;
                }
                Some(_) => match parse_inline_flags(pattern, i, verbose) {
                    Some(InlineFlags::Global {
                        end,
                        verbose: new_verbose,
                    }) => {
                        verbose = new_verbose;
                        i = end + 1;
                    }
                    Some(InlineFlags::Scoped {
                        end,
                        verbose: new_verbose,
                    }) => {
                        open_groups.push(verbose);
                        verbose = new_verbose;
                        i = end + 1;
                    }
                    None => return Err(RegexValidationError::Unsupported),
                },
                None => {
                    return Err(RegexValidationError::Invalid(
                        "missing ), unterminated subpattern",
                    ));
                }
            },
            b')' => {
                let Some(previous_verbose) = open_groups.pop() else {
                    return Err(RegexValidationError::Invalid("unbalanced parenthesis"));
                };
                verbose = previous_verbose;
                i += 1;
            }
            _ => i += 1,
        }
    }

    if escaped {
        Err(RegexValidationError::Invalid("bad escape (end of pattern)"))
    } else if character_set_can_close.is_some() {
        Err(RegexValidationError::Invalid("unterminated character set"))
    } else if !open_groups.is_empty() {
        Err(RegexValidationError::Invalid(
            "missing ), unterminated subpattern",
        ))
    } else {
        Ok(())
    }
}

fn parse_inline_flags(pattern: &[u8], start: usize, verbose: bool) -> Option<InlineFlags> {
    let mut i = start + 2;
    let mut saw_enabled = false;
    let mut saw_disabled = false;
    let mut disabling = false;
    let mut new_verbose = verbose;

    while let Some(byte) = pattern.get(i) {
        match byte {
            b'a' | b'i' | b'L' | b'm' | b's' | b'u' | b'x' if !disabling => {
                saw_enabled = true;
                if *byte == b'x' {
                    new_verbose = true;
                }
            }
            b'i' | b'm' | b's' | b'x' if disabling => {
                saw_disabled = true;
                if *byte == b'x' {
                    new_verbose = false;
                }
            }
            b'-' if !disabling => disabling = true,
            b')' if saw_enabled && !disabling => {
                return Some(InlineFlags::Global {
                    end: i,
                    verbose: new_verbose,
                });
            }
            b':' if saw_enabled || saw_disabled => {
                return Some(InlineFlags::Scoped {
                    end: i,
                    verbose: new_verbose,
                });
            }
            _ => return None,
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::RegexValidationError;
    use super::validate_pattern;

    #[test]
    fn test_validate_pattern_parentheses() {
        assert_eq!(
            validate_pattern(b"a(b(c)", false),
            Err(RegexValidationError::Invalid(
                "missing ), unterminated subpattern"
            ))
        );
        assert_eq!(
            validate_pattern(b"a)b", false),
            Err(RegexValidationError::Invalid("unbalanced parenthesis"))
        );
        assert_eq!(validate_pattern(b"[(]", false), Ok(()));
        assert_eq!(validate_pattern(br"\(\)", false), Ok(()));
    }

    #[test]
    fn test_validate_pattern_verbose() {
        assert_eq!(validate_pattern(b"(\n# ignored )\na)", true), Ok(()));
        assert_eq!(validate_pattern(b"(?x:(\n# ignored )\na))", false), Ok(()));
        assert_eq!(validate_pattern(b"(?x)(\n# ignored )\na)", false), Ok(()));
    }

    #[test]
    fn test_validate_pattern_extensions() {
        for pattern in [
            b"(?:x)".as_slice(),
            b"(?=x)".as_slice(),
            b"(?!x)".as_slice(),
            b"(?<=x)".as_slice(),
            b"(?<!x)".as_slice(),
            b"(?>x)".as_slice(),
            b"(?P<name>x)".as_slice(),
            b"(?# comment)".as_slice(),
        ] {
            assert_eq!(validate_pattern(pattern, false), Ok(()));
        }
        assert_eq!(
            validate_pattern(b"(?z:x)", false),
            Err(RegexValidationError::Unsupported)
        );
    }
}
