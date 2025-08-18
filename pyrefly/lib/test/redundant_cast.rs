/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_redundant_cast_exact_equality,
    r#"
from typing import cast
# These should warn because the types are exactly equal
x: int = 42
y = cast(int, x)  # E: Redundant cast: `int` is the same type as `int`

s: str = "hello"
t = cast(str, s)  # E: Redundant cast: `str` is the same type as `str`

n: None = None
m = cast(None, n)  # E: Redundant cast: `None` is the same type as `None`
"#,
);

testcase!(
    test_no_warning_literals,
    r#"
from typing import cast
x = cast(int, 5)  # No warning - Literal[5] ≠ int
y = cast(str, "hello")  # No warning - Literal['hello'] ≠ str
z = cast(bool, True)  # No warning - Literal[True] ≠ bool
w = cast(float, 3.14)  # E: Redundant cast: `float` is the same type as `float`
"#,
);

testcase!(
    test_no_warning_subclasses,
    r#"
from typing import cast
class Base: pass
class Derived(Base): pass

base_obj: Base = Base()
derived_obj: Derived = Derived()
obj: object = object()

# No warnings - subclass relationships are not exact equality (both directions)
x = cast(Base, derived_obj)  # No warning - Derived ≠ Base
y = cast(object, base_obj)   # No warning - Base ≠ object
z = cast(object, derived_obj)  # No warning - Derived ≠ object
"#,
);

testcase!(
    test_redundant_cast_same_subclass_types,
    r#"
from typing import cast
class Base: pass
class Derived(Base): pass

base_obj: Base = Base()
derived_obj: Derived = Derived()
obj: object = object()

same_base = cast(Base, base_obj)         # E: Redundant cast: `Base` is the same type as `Base`
same_derived = cast(Derived, derived_obj) # E: Redundant cast: `Derived` is the same type as `Derived`
same_object = cast(object, obj)          # E: Redundant cast: `object` is the same type as `object`
"#,
);

testcase!(
    test_no_warning_any_unknown,
    r#"
from typing import Any, cast
def deserialize(data: Any) -> int:
    return cast(int, data)  # No warning - Any ≠ int

def process(obj: object) -> str:
    return cast(str, obj)  # No warning - object ≠ str

def handle_unknown(x) -> bool:
    return cast(bool, x)  # No warning - Unknown ≠ bool
"#,
);

testcase!(
    test_no_warning_union_types,
    r#"
from typing import cast
x: int | str = 42
y = cast(int, x)  # No warning - int | str ≠ int

z: str | None = "test"
w = cast(str, z)  # No warning - str | None ≠ str
"#,
);

testcase!(
    test_no_warning_generic_types,
    r#"
from typing import cast, List
int_list: list[int] = [1, 2, 3]
str_list: list[str] = ["a", "b"]

# Even though both are lists, the generic parameters make them different types
x = cast(list[str], int_list)  # No warning - list[int] ≠ list[str]
y = cast(List[int], int_list)  # E: Redundant cast: `list[int]` is the same type as `list[int]`
"#,
);

testcase!(
    test_redundant_cast_complex_types,
    r#"
from typing import cast, Dict, Optional
# Complex types that are exactly equal should still warn
d: dict[str, int] = {"a": 1}
x = cast(dict[str, int], d)  # E: Redundant cast: `dict[str, int]` is the same type as `dict[str, int]`

opt: str | None = "test"
y = cast(str | None, opt)  # E: Redundant cast: `str | None` is the same type as `str | None`
"#,
);

testcase!(
    test_type_aliases_resolved,
    r#"
from typing import cast
IntAlias = int
StringAlias = str

x: int = 42
y = cast(IntAlias, x)  # E: Redundant cast: `int` is the same type as `int`

s: str = "hello"
t = cast(StringAlias, s)  # E: Redundant cast: `str` is the same type as `str`
"#,
);
