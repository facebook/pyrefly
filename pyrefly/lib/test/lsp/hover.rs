/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Hover;
use lsp_types::HoverContents;
use pretty_assertions::assert_eq;
use ruff_text_size::TextSize;

use crate::lsp::features::hover::get_hover;
use crate::state::handle::Handle;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    match get_hover(&state.transaction(), handle, position) {
        Some(Hover {
            contents: HoverContents::Markup(markup),
            ..
        }) => markup.value,
        _ => "None".to_owned(),
    }
}

#[test]
fn bound_methods_test() {
    let code = r#"
class Foo:
   def meth(self):
        pass

foo = Foo()
foo.meth()
#   ^
xyz = [foo.meth]
#^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
7 | foo.meth()
        ^
```python
(attribute) meth: (self: Self@Foo) -> None
```

9 | xyz = [foo.meth]
     ^
```python
(variable) xyz: list[(self: Self@Foo) -> None]
```
"#
        .trim(),
        report.trim(),
    );
}
