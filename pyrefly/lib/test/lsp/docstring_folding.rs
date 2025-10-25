/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;

use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_no_cursor;

fn get_docstring_ranges_report(state: &State, handle: &Handle) -> String {
    let transaction = state.transaction();
    let Some(module) = transaction.get_module_info(handle) else {
        return "[]".to_owned();
    };
    let ranges = transaction.docstring_ranges(handle).unwrap_or_default();
    let lines: Vec<(u32, u32)> = ranges
        .into_iter()
        .map(|text_range| {
            let range = module.lined_buffer().to_lsp_range(text_range);
            (range.start.line, range.end.line)
        })
        .collect();
    serde_json::to_string_pretty(&lines).unwrap()
}

#[test]
fn docstring_ranges_cover_nested_definitions() {
    let code = r#"
"""Module doc"""

class Foo:
    """Class doc"""

    def method(self):
        """Method doc"""
        if True:
            def inner():
                """Inner doc"""
                pass

    async def async_method(self):
        """Async method doc"""
        pass

def outer():
    """Outer doc"""
    def inner_outer():
        """Inner outer doc"""
        pass
    return inner_outer

if True:
    class Conditional:
        """Conditional class doc"""
        pass
"#;

    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_docstring_ranges_report);

    assert_eq!(
        r#"# main.py

[
  [
    1,
    1
  ],
  [
    4,
    4
  ],
  [
    7,
    7
  ],
  [
    10,
    10
  ],
  [
    14,
    14
  ],
  [
    18,
    18
  ],
  [
    20,
    20
  ],
  [
    26,
    26
  ]
]"#
        .trim(),
        report.trim(),
    );
}
