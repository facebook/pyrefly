/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Range;

use ruff_annotate_snippets::Level as SnippetLevel;
use ruff_annotate_snippets::Renderer as SnippetRenderer;
use ruff_annotate_snippets::Snippet as SnippetBlock;

/// Parse byte ranges for parameters and return type from a signature string
/// produced by `TypeDisplayContext` with `SignatureHelp` mode.
///
/// Expects format like `def foo(self: T, a: int) -> None: ...`.
/// Returns `(params_range, return_type_range)` as byte ranges, or `None`
/// if the format doesn't match. Callers should fall back gracefully on `None`
/// since the display format could change or contain edge cases we don't handle.
fn signature_parts(sig: &str) -> Option<(Range<usize>, Range<usize>)> {
    let open = sig.find('(')?;

    // Find closing ')' respecting nested parentheses
    let mut balance = 0;
    let mut close = None;
    for (i, c) in sig[open..].char_indices() {
        match c {
            '(' => balance += 1,
            ')' => {
                balance -= 1;
                if balance == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;

    let params = (open + 1)..close;

    // Find " -> " after close
    let arrow_search_start = close;
    let arrow = sig[arrow_search_start..].find(" -> ")? + arrow_search_start;

    let ret_start = arrow + " -> ".len();
    let ret_end = if let Some(pos) = sig[ret_start..].rfind(": ...") {
        ret_start + pos
    } else if let Some(pos) = sig[ret_start..].rfind(':') {
        ret_start + pos
    } else {
        sig.len()
    };
    Some((params, ret_start..ret_end))
}

/// Find the byte ranges where two strings differ, using longest common
/// prefix and suffix. Returns `None` if the strings are equal.
///
/// The returned ranges highlight the "differing middle" of each string.
/// When one string is a strict prefix/suffix of the other, a minimal
/// single-byte range is returned to ensure there's always something to annotate.
///
/// Note: operates on raw bytes, which is correct for ASCII type names but
/// could produce ranges that split multi-byte UTF-8 characters for non-ASCII
/// identifiers.
fn diff_ranges(expected: &str, found: &str) -> Option<(Range<usize>, Range<usize>)> {
    if expected == found {
        return None;
    }
    let expected_bytes = expected.as_bytes();
    let found_bytes = found.as_bytes();
    let mut lcp = 0;
    while lcp < expected_bytes.len()
        && lcp < found_bytes.len()
        && expected_bytes[lcp] == found_bytes[lcp]
    {
        lcp += 1;
    }
    let mut lcs = 0;
    while expected_bytes.len() > lcp + lcs
        && found_bytes.len() > lcp + lcs
        && expected_bytes[expected_bytes.len() - 1 - lcs]
            == found_bytes[found_bytes.len() - 1 - lcs]
    {
        lcs += 1;
    }
    let expected_end = expected_bytes.len().saturating_sub(lcs);
    let found_end = found_bytes.len().saturating_sub(lcs);
    let expected_span = if expected_end > lcp {
        lcp..expected_end
    } else {
        // The expected params are a prefix of the found params (or vice versa).
        // Point at the first character after the shared prefix, which in the
        // full source corresponds to the closing `)` or `,` — indicating where
        // parameters are missing or extra. Clamp to the string length to avoid
        // producing an out-of-bounds range when the entire string is a prefix
        // (e.g., for Callable types whose return type ends at the string boundary).
        lcp..(lcp + 1).min(expected_bytes.len())
    };
    let found_span = if found_end > lcp {
        lcp..found_end
    } else {
        lcp..(lcp + 1).min(found_bytes.len())
    };
    Some((expected_span, found_span))
}

/// Render a visual diff between two function signatures, highlighting where
/// the parameters and/or return types differ.
///
/// Uses `ruff_annotate_snippets` to produce caret-underline annotations, then
/// strips the line-number gutter from the rendered output. Returns `None` if
/// signature parsing fails or if there are no differences to annotate.
///
/// The gutter-stripping logic depends on the current output format of
/// `SnippetRenderer::plain()`. If that format changes upstream, this
/// post-processing may need to be updated.
pub fn render_signature_diff(expected: &str, found: &str) -> Option<Vec<String>> {
    let (expected_params, expected_ret) = signature_parts(expected)?;
    let (found_params, found_ret) = signature_parts(found)?;

    let expected_prefix = "expected: ";
    let found_prefix = "found:    ";
    let expected_line_text = format!("{expected_prefix}{expected}");
    let found_line_text = format!("{found_prefix}{found}");

    // We strip line numbers from the output, so the starting line number doesn't matter
    // for the final text, but it's used by the renderer.
    let line_start = 1;
    let mut source = expected_line_text.clone();
    source.push('\n');
    source.push_str(&found_line_text);
    let found_offset = expected_line_text.len() + 1;

    let mut annotations = Vec::new();
    if let Some((exp_span, found_span)) = diff_ranges(
        &expected[expected_params.clone()],
        &found[found_params.clone()],
    ) {
        annotations.push(
            SnippetLevel::Error
                .span(
                    (expected_prefix.len() + expected_params.start + exp_span.start)
                        ..(expected_prefix.len() + expected_params.start + exp_span.end),
                )
                .label("parameters"),
        );
        annotations.push(
            SnippetLevel::Error
                .span(
                    (found_offset + found_prefix.len() + found_params.start + found_span.start)
                        ..(found_offset + found_prefix.len() + found_params.start + found_span.end),
                )
                .label("parameters"),
        );
    }
    if let Some((exp_span, found_span)) =
        diff_ranges(&expected[expected_ret.clone()], &found[found_ret.clone()])
    {
        annotations.push(
            SnippetLevel::Error
                .span(
                    (expected_prefix.len() + expected_ret.start + exp_span.start)
                        ..(expected_prefix.len() + expected_ret.start + exp_span.end),
                )
                .label("return type"),
        );
        annotations.push(
            SnippetLevel::Error
                .span(
                    (found_offset + found_prefix.len() + found_ret.start + found_span.start)
                        ..(found_offset + found_prefix.len() + found_ret.start + found_span.end),
                )
                .label("return type"),
        );
    }

    if annotations.is_empty() {
        return None;
    }

    let mut snippet = SnippetBlock::source(&source).line_start(line_start);
    for ann in annotations {
        snippet = snippet.annotation(ann);
    }
    let message = SnippetLevel::None.title("").snippet(snippet);
    let rendered = SnippetRenderer::plain().render(message).to_string();
    let mut lines: Vec<String> = Vec::new();
    lines.push("Signature mismatch:".to_owned());
    for line in rendered.lines() {
        if let Some(idx) = line.find('|') {
            let (left, right) = line.split_at(idx);
            if left.trim().is_empty() || left.trim().chars().all(|c| c.is_ascii_digit()) {
                let mut trimmed = right.trim_start_matches('|');
                if trimmed.starts_with(' ') {
                    trimmed = &trimmed[1..];
                }
                if trimmed.is_empty() {
                    continue;
                }
                lines.push(trimmed.to_owned());
                continue;
            }
        }
        if !line.trim().is_empty() {
            lines.push(line.to_owned());
        }
    }
    Some(lines)
}

#[cfg(test)]
mod tests {
    use crate::test::util::TestEnv;

    /// Run a single-module type check and return the error messages.
    fn error_messages(code: &str) -> Vec<String> {
        let (state, handle) = TestEnv::one("main", code).to_state();
        state
            .transaction()
            .get_errors(&[handle("main")])
            .collect_errors()
            .shown
            .iter()
            .map(|e| e.msg().to_string())
            .collect()
    }

    /// Integration test verifying the full error message produced by the
    /// override checker includes the signature diff annotation.
    #[test]
    fn test_override_signature_diff_full_message() {
        let messages = error_messages(
            r#"
from abc import ABC

class A(ABC):
    def foo(self, a: int, b: int, c: int):
        raise NotImplementedError()

class B(A):
    def foo(self):
        x = 1
        print(x)
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: B) -> None`, which is not assignable to `(self: B, a: int, b: int, c: int) -> Unknown`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, a: int, b: int, c: int) -> Unknown: ...
                           ^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^ return type
                           |
                           parameters
  found:    def foo(self: B) -> None: ...
                           ^    ^^^^ return type
                           |
                           parameters"#;
        assert_eq!(messages[0], expected);
    }

    /// Override has too many arguments (inverse of the basic test).
    #[test]
    fn test_signature_diff_too_many_args() {
        let messages = error_messages(
            r#"
class A:
    def foo(self) -> None:
        pass

class B(A):
    def foo(self, x: int, y: str) -> None:
        pass
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: B, x: int, y: str) -> None`, which is not assignable to `(self: B) -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B) -> None: ...
                           ^ parameters
  found:    def foo(self: B, x: int, y: str) -> None: ...
                           ^^^^^^^^^^^^^^^^ parameters"#;
        assert_eq!(messages[0], expected);
    }

    /// Simple single argument type mismatch.
    #[test]
    fn test_signature_diff_single_param_mismatch() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> None:
        pass

class B(A):
    def foo(self, x: str) -> None:
        pass
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: B, x: str) -> None`, which is not assignable to `(self: B, x: int) -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> None: ...
                                ^^^ parameters
  found:    def foo(self: B, x: str) -> None: ...
                                ^^^ parameters"#;
        assert_eq!(messages[0], expected);
    }

    /// Simple return type only mismatch (parameters are identical).
    #[test]
    fn test_signature_diff_return_type_only() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> int:
        return x

class B(A):
    def foo(self, x: int) -> str:
        return ""
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: B, x: int) -> str`, which is not assignable to `(self: B, x: int) -> int`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                                        ^^^ return type
  found:    def foo(self: B, x: int) -> str: ...
                                        ^^^ return type"#;
        assert_eq!(messages[0], expected);
    }

    /// Overloaded parent method: signature diff should not appear since
    /// overloads have multiple signatures.
    #[test]
    fn test_signature_diff_overloads() {
        let messages = error_messages(
            r#"
from typing import overload

class A:
    @overload
    def foo(self, x: int) -> int: ...
    @overload
    def foo(self, x: str) -> str: ...
    def foo(self, x):
        return x

class B(A):
    def foo(self, x: float) -> float:
        return x
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        // Overloads have multiple signatures, so no signature diff is shown.
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: B, x: float) -> float`, which is not assignable to `Overload[
  (self: B, x: int) -> int
  (self: B, x: str) -> str
]`, the type of `A.foo`"#;
        assert_eq!(messages[0], expected);
    }

    /// Inferred (unannotated) return types: verifies signature diff works
    /// when return types have no explicit annotation.
    #[test]
    fn test_signature_diff_inferred_return() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int):
        return x

class B(A):
    def foo(self, x: str):
        return x
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: B, x: str) -> str`, which is not assignable to `(self: B, x: int) -> int`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                                ^^^     ^^^ return type
                                |
                                parameters
  found:    def foo(self: B, x: str) -> str: ...
                                ^^^     ^^^ return type
                                |
                                parameters"#;
        assert_eq!(messages[0], expected);
    }

    /// Lambda override: when the arity differs, the override falls through
    /// to the general BadOverride path, showing the signature diff instead
    /// of a misleading parameter name mismatch.
    #[test]
    fn test_signature_diff_lambda() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> int:
        return x

class B(A):
    foo = lambda self: None
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: Unknown) -> None`, which is not consistent with `(self: B, x: int) -> int` in `A.foo` (the type of read-write attributes cannot be changed)
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                          ^^^^^^^^^     ^^^ return type
                          |
                          parameters
  found:    (self: Unknown) -> None
                   ^^^^^^^     ^^^^ return type
                   |
                   parameters"#;
        assert_eq!(messages[0], expected);
    }

    /// Named function override: verifies signature diff normalizes the
    /// function name to the attribute name for readability.
    #[test]
    fn test_signature_diff_named_function() {
        let messages = error_messages(
            r#"
def helper(self) -> None:
    pass

class A:
    def method(self, x: int) -> int:
        return x

class B(A):
    method = helper
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.method` overrides parent class `A` in an inconsistent manner
  `B.method` has type `(self: Unknown) -> None`, which is not assignable to `(self: B, x: int) -> int`, the type of `A.method`
  Signature mismatch:
  expected: def method(self: B, x: int) -> int: ...
                             ^^^^^^^^^     ^^^ return type
                             |
                             parameters
  found:    def method(self: Unknown) -> None: ...
                             ^^^^^^^     ^^^^ return type
                             |
                             parameters"#;
        assert_eq!(messages[0], expected);
    }

    /// Callable-style signatures (no `def`, no `: ...` suffix) where the
    /// shorter return type is a prefix of the longer one. Previously panicked
    /// because `diff_ranges` produced a span past the end of the source string.
    #[test]
    fn test_render_signature_diff_callable_prefix_return_type() {
        use super::render_signature_diff;
        let expected = "() -> EdgeFoo | EdgeBar";
        let found = "() -> Edge";
        // Should not panic. The diff should highlight the return type difference.
        let result = render_signature_diff(expected, found);
        assert!(
            result.is_some(),
            "Expected a signature diff for differing return types"
        );
    }
}
