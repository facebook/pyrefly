/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools as _;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::TextRangeWithModule;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

fn get_declaration_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let defs = state.transaction().goto_declaration(handle, position);
    if !defs.is_empty() {
        defs.into_iter()
            .map(
                |TextRangeWithModule {
                     module: module_info,
                     range,
                 }| {
                    format!(
                        "Declaration Result:\n{}",
                        code_frame_of_source_at_range(module_info.contents(), range)
                    )
                },
            )
            .join("\n")
    } else {
        "Declaration Result: None".to_owned()
    }
}

#[test]
fn goto_declaration_stops_at_import() {
    let code_import_provider: &str = r#"
def f():
    pass
"#;
    let code_test: &str = r#"
from .import_provider import f
bar = f()
#     ^
"#;

    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_declaration_report,
    );
    assert_eq!(
        r#"
# main.py
3 | bar = f()
          ^
Declaration Result:
2 | from .import_provider import f
                                 ^


# import_provider.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn goto_declaration_aliased_import() {
    let code_import_provider: &str = r#"
# top of module
class Foo: pass
    "#;
    let code_test: &str = r#"
from import_provider import Foo as F
#                                  ^

def f(y: F):
#        ^
    return y
"#;

    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_declaration_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from import_provider import Foo as F
                                       ^
Declaration Result:
2 | from import_provider import Foo as F
                                       ^

5 | def f(y: F):
             ^
Declaration Result:
2 | from import_provider import Foo as F
                                       ^


# import_provider.py
"#
        .trim(),
        report.trim()
    );
}
