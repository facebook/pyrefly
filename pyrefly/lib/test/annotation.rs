/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_class_scope_quoted_annotation_bypasses_shadowing,
    r#"
from typing import assert_type
class D:
    def int(self) -> None:
        ...
    x: "int" = 0  # E: Expected a type form
assert_type(D.x, int)  # E: assert_type(Unknown, int) failed
"#,
);

testcase!(
    test_union_operator_with_bare_string_literal,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from typing import assert_type, Any, TypeVar, Generic, Literal, Callable, LiteralString
T = TypeVar("T")
class C(Generic[T]): ...
bad1: int | "str" = "foo"  # E: `|` union syntax does not work with string literals
bad2: int | "str" | T = "foo"  # E: `|` union syntax does not work with string literals  # E: Type variable `T` is not in scope
bad3: "str" | int = "foo"  # E: `|` union syntax does not work with string literals
bad4: "str" | int | T = "foo"  # E: `|` union syntax does not work with string literals  # E: Type variable `T` is not in scope
bad5: C | "str" = "foo"  # E: `|` union syntax does not work with string literals
bad6: "str" | None = "foo"  # E: `|` union syntax does not work with string literals
bad7: None | "str" = "foo"  # E: `|` union syntax does not work with string literals
bad8: "str" | Any = "foo"  # E: `|` union syntax does not work with string literals
bad9: Any | "str" = "foo"  # E: `|` union syntax does not work with string literals
ok1: T | "str" = "foo"  # E: Type variable `T` is not in scope
ok2: "str" | T = "foo"  # E: Type variable `T` is not in scope
ok3 = list["str" | T]
ok4 = (int) | (str)
ok5: "str" | C[int] = "foo"
ok6: C[int] | "str" = "foo"
ok7: "str" | Literal[1] = "foo"
ok8: "str" | Callable[[int], str] = "foo"
ok9: "str" | LiteralString = "foo"
"#,
);

testcase!(
    test_union_of_plain_type_and_complex_forward_ref,
    r#"
bad1: int | "list[str]" = []  # E: `|` union syntax does not work with string literals
bad2: "list[str]" | int = []  # E: `|` union syntax does not work with string literals
    "#,
);

testcase!(
    test_union_of_forward_refs,
    r#"
bad: "int" | "list[str]" = 1  # E: `|` union syntax does not work with string literals
    "#,
);

testcase!(
    test_union_forward_ref_with_special_classes,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from typing import TypedDict, NamedTuple, Protocol

class TD(TypedDict):
    x: int

class NT(NamedTuple):
    x: int

class P(Protocol):
    def f(self) -> None: ...

bad1: "str" | TD = "foo"  # E: `|` union syntax does not work with string literals
bad2: "str" | NT = NT(x=1)  # E: `|` union syntax does not work with string literals
bad3: "str" | P = "foo"  # E: `|` union syntax does not work with string literals
"#,
);

testcase!(
    test_union_forward_ref_with_none,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from dataclasses import dataclass

def test() -> "Foo" | None:  # E: `|` union syntax does not work with string literals
    return Foo(bar=1)

@dataclass
class Foo:
    bar: int
"#,
);

testcase!(
    bug = "list[int] (types.GenericAlias) errors at runtime but List[int] (typing._GenericAlias) does not; pyrefly can't distinguish them",
    test_union_forward_ref_with_builtin_generics,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from typing import List, Dict, Type, Any
ok_but_runtime_error1: "str" | list[int] = "foo"
ok_but_runtime_error2: "str" | dict[str, int] = "foo"
ok_but_runtime_error3: "str" | tuple[int, ...] = "foo"
ok_but_runtime_error4: "str" | type[int] = "foo"
ok1: "str" | List[int] = "foo"
ok2: "str" | Dict[str, int] = "foo"
ok3: "str" | Type[Any] = "foo"
"#,
);

// Test that the error is NOT raised for Python 3.14+
// In Python 3.14+, annotations are not evaluated at runtime by default (PEP 649)
testcase!(
    test_union_type_with_bare_string_literal_py314,
    TestEnv::new_with_version(PythonVersion::new(3, 14, 0)),
    r#"
ok1: int | "str" = "foo"
ok2: "str" | int = "foo"
"#,
);

// Test that the error is NOT raised when `from __future__ import annotations` is used
// With future annotations, annotations are not evaluated at runtime
testcase!(
    test_union_type_with_bare_string_literal_future_annotations,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from __future__ import annotations
ok1: int | "str" = "foo"
ok2: "str" | int = "foo"
"#,
);

// Test legacy type alias with forward reference string literal
testcase!(
    test_union_operator_with_legacy_type_alias,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from typing import TypeAlias

class C[T]: pass

IntAlias1 = C[int]
IntAlias2: TypeAlias = C[int]
IntAlias3 = int
IntAlias4: TypeAlias = int

ok1: IntAlias1 | "str" = "foo"
ok2: IntAlias2 | "str" = "foo"
bad1: IntAlias3 | "str" = "foo"  # E: `|` union syntax does not work with string literals
bad2: IntAlias4 | "str" = "foo"  # E: `|` union syntax does not work with string literals
"#,
);

// Test scoped type alias with forward reference string literal
// TypeAliasType is a plain type, so this is a runtime error
testcase!(
    test_union_operator_with_scoped_type_alias,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
class C[T]: pass
type IntAlias = C[int]
bad: IntAlias | "str" = "foo"  # E: `|` union syntax does not work with string literals
"#,
);

fn env_3_13_with_stub() -> TestEnv {
    let mut env = TestEnv::new_with_version(PythonVersion::new(3, 13, 0));
    env.add_with_path("foo", "foo.pyi", "x: int | 'str'");
    env
}

testcase!(
    test_union_forward_ref_ok_in_stub,
    env_3_13_with_stub(),
    "import foo",
);
