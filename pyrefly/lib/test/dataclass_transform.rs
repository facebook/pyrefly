/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_function_basic,
    r#"
from typing import dataclass_transform

@dataclass_transform()
def create[T](cls: type[T]) -> type[T]: ...

@create
class C:
    x: int
C(x=0)
C(x="oops")  # E: `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_class_basic,
    r#"
from typing import dataclass_transform

@dataclass_transform()
class C: ...

class D(C):
    x: int
D(x=0)
D(x="oops")  # E: `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    bug = "We fail to pick up dataclass field `y`",
    test_class_inheritance,
    r#"
from typing import dataclass_transform

@dataclass_transform()
class C: ...

class D(C):
    x: int
class E(D):
    y: str
E(x=0, y="")  # E: Unexpected keyword argument `y`
    "#,
);

testcase!(
    test_metaclass_basic,
    r#"
from typing import dataclass_transform

@dataclass_transform()
class Meta(type): ...
class C(metaclass=Meta): ...

class D(C):
    x: int
D(x=0)
D(x="oops")  # E: `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);
