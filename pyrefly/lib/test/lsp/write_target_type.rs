/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for hover (`get_type_at`) and the TSP `getComputedType` endpoint
//! (`get_computed_type_at_range`) on an attribute *target* — the `x` in
//! `c.x = 1`, `del c.x` and `c.x: int`. The common shapes of assignment target
//! are covered by `hover_on_attribute_assignment_target` in `hover.rs`.
//!
//! A target is not a read, so these report what is being written to the
//! attribute rather than what reading it would return. A target that is only
//! declared reports its declared type, since nothing is written to it.

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;
use crate::test::util::mk_multi_file_state;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let transaction = state.transaction();
    let hover = match transaction.get_type_at(handle, position) {
        Some(t) => format!("`{t}`"),
        None => "None".to_owned(),
    };
    let computed = match transaction.get_computed_type_at_range(handle, TextRange::empty(position))
    {
        Some(t) => format!("`{t}`"),
        None => "None".to_owned(),
    };
    format!("Hover: {hover}\nComputed: {computed}")
}

/// A store target reports the type that is being assigned to it. When the value
/// does not fit the attribute's declared type, the assignment recovers with the
/// declared type and the target reports that.
#[test]
fn test_store_target() {
    let code = r#"
class C:
    x: int

c = C()
c.x = "oops"
# ^
for c.x in [1, 2]:
#     ^
    pass

class Outer:
    inner: C
o = Outer()
o.inner.x = 5
#       ^

def g() -> C: ...
g().x = 6
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | c.x = "oops"
      ^
Hover: `int`
Computed: `int`

8 | for c.x in [1, 2]:
          ^
Hover: `int`
Computed: `int`

15 | o.inner.x = 5
             ^
Hover: `Literal[5]`
Computed: `Literal[5]`

19 | g().x = 6
         ^
Hover: `Literal[6]`
Computed: `Literal[6]`
"#
        .trim(),
        report.trim(),
    );
}

/// A target that is only declared has nothing assigned to it, so it reports its
/// declared type.
#[test]
fn test_declared_target() {
    let code = r#"
class C:
    def __init__(self) -> None:
        self.w: bytes
#            ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |         self.w: bytes
                 ^
Hover: `bytes`
Computed: `bytes`
"#
        .trim(),
        report.trim(),
    );
}

/// A delete target has no type: `check_attr_delete` only validates the deletion,
/// so there is no delete-side type to report and we report nothing rather than
/// falling back to what reading the attribute would return.
#[test]
fn test_delete_target() {
    let code = r#"
class C:
    x: int

c = C()
del c.x
#     ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | del c.x
          ^
Hover: None
Computed: None
"#
        .trim(),
        report.trim(),
    );
}

/// Writing through a property, a descriptor or `__setattr__` runs a setter, which
/// produces no type Pyrefly can attribute to the target, so the target reports
/// nothing. In particular the getter is never consulted: reading `p.p` and `d.d`
/// gives `int`, which must not leak into the store. A `__slots__` attribute is a
/// plain attribute, so its store reports the assigned type.
#[test]
fn test_property_and_descriptor_target() {
    let code = r#"
from typing import Any

class P:
    @property
    def p(self) -> int: ...
    @p.setter
    def p(self, v: str) -> None: ...

p = P()
p.p = "s"
# ^
print(p.p)
#       ^

class Desc[T]:
    def __get__(self, obj: object, objtype: Any = None) -> T: ...
    def __set__(self, obj: object, value: T) -> None: ...

class D:
    d: Desc[int]

d = D()
d.d = 1
# ^
print(d.d)
#       ^

class S:
    __slots__ = ("a",)
    a: int
s = S()
s.a = 1
# ^

class Setattr:
    def __setattr__(self, name: str, value: int) -> None: ...
sa = Setattr()
sa.anything = 1
#  ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
11 | p.p = "s"
       ^
Hover: None
Computed: None

13 | print(p.p)
             ^
Hover: `int`
Computed: `int`

24 | d.d = 1
       ^
Hover: None
Computed: None

26 | print(d.d)
             ^
Hover: `int`
Computed: `int`

33 | s.a = 1
       ^
Hover: `Literal[1]`
Computed: `Literal[1]`

39 | sa.anything = 1
        ^
Hover: None
Computed: None
"#
        .trim(),
        report.trim(),
    );
}

/// A store target reports the same narrowed type as the read that follows it.
#[test]
fn test_narrowed_target() {
    let code = r#"
class C:
    y: int | None

def f(c: C) -> None:
    if c.y is not None:
        c.y = 1
#         ^
        print(c.y)
#               ^
    c.y = 2
#     ^
    print(c.y)
#           ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
7 |         c.y = 1
              ^
Hover: `Literal[1]`
Computed: `Literal[1]`

9 |         print(c.y)
                    ^
Hover: `Literal[1]`
Computed: `Literal[1]`

11 |     c.y = 2
           ^
Hover: `Literal[2]`
Computed: `Literal[2]`

13 |     print(c.y)
                 ^
Hover: `Literal[2]`
Computed: `Literal[2]`
"#
        .trim(),
        report.trim(),
    );
}

/// The range of `needle`, which must appear exactly once in `code`.
fn range_of(code: &str, needle: &str) -> TextRange {
    let start = code.find(needle).unwrap();
    assert_eq!(code.rfind(needle), Some(start), "`{needle}` is not unique");
    TextRange::at(
        TextSize::try_from(start).unwrap(),
        TextSize::try_from(needle.len()).unwrap(),
    )
}

/// `getComputedType` takes a range, so cover the shapes a client can send for an
/// attribute expression: an empty range (a point query), the whole expression, and
/// just the attribute name. A type belongs to the whole expression, so a name-only
/// range reports nothing for a target exactly as it does for a read.
#[test]
fn test_computed_type_at_range() {
    let code = r#"
class C:
    x: int

c = C()
c.x = 2
d = C()
print(d.x)

class E:
    def __init__(self) -> None:
        self.w: bytes
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::Exports, false);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let report = |label: &str, range: TextRange| {
        let ty = match transaction.get_computed_type_at_range(handle, range) {
            Some(t) => format!("`{t}`"),
            None => "None".to_owned(),
        };
        format!("{label}: {ty}\n")
    };
    let store = range_of(code, "c.x");
    let read = range_of(code, "d.x");
    let declaration = range_of(code, "self.w");
    let name_only = |range: TextRange| TextRange::at(range.end() - TextSize::new(1), 1.into());
    let mut actual = String::new();
    actual.push_str(&report("store `c.x`", store));
    actual.push_str(&report("store `x`", name_only(store)));
    actual.push_str(&report(
        "store point",
        TextRange::empty(name_only(store).start()),
    ));
    actual.push_str(&report("read `d.x`", read));
    actual.push_str(&report("read `x`", name_only(read)));
    actual.push_str(&report(
        "read point",
        TextRange::empty(name_only(read).start()),
    ));
    actual.push_str(&report("declaration `self.w`", declaration));
    actual.push_str(&report("declaration `w`", name_only(declaration)));
    actual.push_str(&report(
        "declaration point",
        TextRange::empty(name_only(declaration).start()),
    ));
    assert_eq!(
        r#"
store `c.x`: `Literal[2]`
store `x`: None
store point: `Literal[2]`
read `d.x`: `int`
read `x`: None
read point: `int`
declaration `self.w`: `bytes`
declaration `w`: None
declaration point: `bytes`
"#
        .trim(),
        actual.trim(),
    );
}
