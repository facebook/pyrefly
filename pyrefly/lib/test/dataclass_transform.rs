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
    test_class_inheritance,
    r#"
from typing import dataclass_transform

@dataclass_transform()
class C: ...

class D(C):
    x: int
class E(D):
    y: str
E(x=0, y="")
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

testcase!(
    test_call_transform,
    r#"
from typing import dataclass_transform

@dataclass_transform()
def build(**kwargs): ...

@build()
class C:
    x: int
C(x=0)
C(x="oops")  # E: `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_frozen_default,
    r#"
from typing import dataclass_transform, Any
@dataclass_transform()
def build_default_mutable(**kwargs) -> Any: ...
@dataclass_transform(frozen_default=True)
def build_default_frozen(**kwargs) -> Any: ...

@build_default_mutable()
class Mutable1:
    x: int
@build_default_mutable(frozen=True)
class Frozen1:
    x: int
@build_default_frozen()
class Frozen2:
    x: int
@build_default_frozen(frozen=False)
class Mutable2:
    x: int

def f(mut1: Mutable1, mut2: Mutable2, froz1: Frozen1, froz2: Frozen2):
    mut1.x = 42
    mut2.x = 42
    froz1.x = 42  # E: frozen dataclass member
    froz2.x = 42  # E: frozen dataclass member
    "#,
);

testcase!(
    test_class_keyword,
    r#"
from typing import dataclass_transform
@dataclass_transform()
class Base: ...
class Data(Base, frozen=True):
    x: int
data = Data(x=0)
data.x = 42  # E: frozen dataclass member
    "#,
);

testcase!(
    test_field_specifier,
    r#"
from typing import dataclass_transform, Any
def field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(field,))
def build(x): ...
@build
class C:
    x: int = field()
C(x=0)
C()  # E: Missing argument `x`
    "#,
);

testcase!(
    test_factory,
    r#"
from typing import dataclass_transform, Any
def field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(field,))
def build(x): ...
@build
class C:
    x: int = field(factory=int)
C(x=0)
C()  # OK because `factory` gives `x` a default
    "#,
);

testcase!(
    test_alias,
    r#"
from typing import dataclass_transform, Any, assert_type
def my_field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(my_field,))
def build(x): ...
@build
class C:
    x: int = my_field(alias="not_my_x")
c = C(not_my_x=0)
assert_type(c.x, int)
    "#,
);

testcase!(
    test_set_init_through_overload,
    r#"
from typing import dataclass_transform, overload, Any, Literal

@overload
def field(name: None = None, init: Literal[False] = False) -> Any: ...
@overload
def field(name: str, init: Literal[True] = True) -> Any: ...
def field(name: str | None = None, init: bool = False) -> Any: ...

@dataclass_transform(field_specifiers=(field,))
def build(x): ...
@build
class C:
    x: int = field()
    y: str = field(name='y')
C(y='hello world')
    "#,
);

testcase!(
    test_field_default,
    r#"
from typing import dataclass_transform, Any
def field(kw_only: bool = True) -> Any: ...
@dataclass_transform(field_specifiers=(field,))
def build(x): ...
@build
class C:
    x: int = field()
C(x=0)
C(0)  # E: Expected argument `x` to be passed by name
    "#,
);

testcase!(
    test_converter,
    r#"
from typing import dataclass_transform, Any
def my_field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(my_field,))
def build(x): ...
def int_to_str(x: int) -> str:
    return str(x)
@build
class C:
    x: str = my_field(converter=int_to_str)
C(x=0)
C(x="oops")  # E: `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_set_attr_with_converter,
    r#"
from typing import dataclass_transform, Any
def my_field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(my_field,))
def build(x): ...
def int_to_str(x: int) -> str:
    return str(x)
@build
class C:
    x: str = my_field(converter=int_to_str)
c = C(x=0)
c.x = 42
c.x = "oops"  # E: `Literal['oops']` is not assignable to attribute `x` with type `int`
    "#,
);

testcase!(
    test_class_converter,
    r#"
from typing import dataclass_transform, Any
def my_field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(my_field,))
def build(x): ...

class Converter:
    def __init__(self, x: int) -> None: ...

@build
class Data:
    x: Converter = my_field(converter=Converter)
Data(x=0)
Data(x=Converter(0))  # E: `Converter` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_overloaded_converter,
    r#"
from typing import dataclass_transform, overload, Any
def my_field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(my_field,))
def build(x): ...

class Converter1:
    @overload
    def __init__(self, x: int) -> None: ...
    @overload
    def __init__(self, x: str) -> None: ...
    def __init__(self, x): ...

@build
class Data1:
    x: Converter1 = my_field(converter=Converter1)
Data1(x=0)
Data1(x="")
Data1(x=Converter1(0))  # E: `Converter1` is not assignable to parameter `x` with type `int | str`

@overload
def converter2(x: bytes) -> int: ...
@overload
def converter2(x: str) -> int: ...
def converter2(x): return int(x)

@build
class Data2:
    x: int = my_field(converter=converter2)
Data2(x=b"")
Data2(x="")
Data2(x=0)  # E: `Literal[0]` is not assignable to parameter `x` with type `bytes | str`
    "#,
);

testcase!(
    test_builtin_class_converter,
    r#"
from typing import dataclass_transform, Any
def my_field(**kwargs) -> Any: ...
@dataclass_transform(field_specifiers=(my_field,))
def build(x): ...
@build
class Data:
    x: int = my_field(converter=int)

class NotConvertibleToInt: ...

Data(x="42")
Data(x=NotConvertibleToInt())  # E: `NotConvertibleToInt` is not assignable to parameter `x`
    "#,
);

testcase!(
    test_existing_init_decorator,
    r#"
from typing import dataclass_transform, reveal_type
@dataclass_transform()
def build[T](cls: type[T]) -> type[T]: ...

@build
class A:
    def __init__(self, x: int) -> None: ...
    y: str
# `__init__` is not synthesized when one already exists
reveal_type(A.__init__)  # E: (self: A, x: int) -> None

class B(A): ...
# inheriting from `A` still does not synthesize an `__init__`
reveal_type(B.__init__)  # E: (self: B, x: int) -> None

@build
class C(A): ...
# inheriting from `A` and reapplying `@build` synthesizes an `__init__`
reveal_type(C.__init__)  # E: (self: C, y: str) -> None
    "#,
);

testcase!(
    test_existing_init_base_class,
    r#"
from typing import dataclass_transform, reveal_type
@dataclass_transform()
class Base:
    def __init__(self, x: int) -> None: ...

class A(Base): ...
# __init__ in `Base` is ignored
reveal_type(A.__init__)  # E: (self: A) -> None

class B(Base):
    def __init__(self, y: str) -> None: ...
    z: bool
# __init__ is not synthesized when one already exists
reveal_type(B.__init__)  # E: (self: B, y: str) -> None

class C(B): ...
# inheriting from `B` synthesizes an `__init__`
reveal_type(C.__init__)  # E: (self: C, z: bool) -> None
    "#,
);

fn overloaded_transform_env() -> TestEnv {
    TestEnv::one_with_path(
        "foo",
        "foo.pyi",
        r#"
from typing import Any, dataclass_transform, overload
@overload
def create() -> Any: ...
@overload
@dataclass_transform()
def create[T](cls: type[T]) -> type[T]: ...
    "#,
    )
}

testcase!(
    test_overloaded_transform,
    overloaded_transform_env(),
    r#"
import foo
@foo.create
class C:
    x: int
C(x=0)
    "#,
);

testcase!(
    test_lower_version,
    TestEnv::new_with_version(PythonVersion::new(3, 10, 0)),
    r#"
from typing_extensions import dataclass_transform
@dataclass_transform()
def decorate(cls):
    return cls
@decorate
class A:
    x: int
A(x=0)
    "#,
);
