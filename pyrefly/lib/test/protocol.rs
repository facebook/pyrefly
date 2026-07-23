/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn proxy_method_env() -> TestEnv {
    TestEnv::one_with_path(
        "shape_extensions",
        "shape_extensions/__init__.pyi",
        r#"
class ProxyMethod[T]: ...
"#,
    )
}

testcase!(
    test_protocol,
    r#"
from typing import Final, Protocol
class P(Protocol):
    x: int
    y: str
class C1:
    x: int
    y: str
class C2:
    x: str
class C3(P, C1): ...
class C4(P):
    y: int # E: Class member `C4.y` overrides parent class `P` in an inconsistent manner
class C5:
    x: int
    y: int
def f(proto: P) -> None: ...
def g(p: P, c1: C1, c2: C2, c3: C3, c4: C4, c5: C5) -> None:
    f(c1)
    f(c2)  # E: Argument `C2` is not assignable to parameter `proto` with type `P`
    f(c3)
    f(c4)
    f(c5)  # E: Argument `C5` is not assignable to parameter `proto` with type `P`
    c: C1 = p  # E: `P` is not assignable to `C1`
 "#,
);

testcase!(
    test_proxy_method_direct_call_and_attribute_access,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]

class A(Base):
    def forward(self, x: int) -> str: ...

a = A()
assert_type(a(1), str)
assert_type(a.__call__(1), str)
a("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_proxy_method_forward_ref_annotation,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: "ProxyMethod['forward']"
    def forward(self, x: int) -> str: ...

base = Base()
assert_type(base(1), str)
base("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_proxy_method_import_alias_is_direct_annotation,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod as PM

class Base:
    __call__: PM["forward"]
    def forward(self, x: int) -> str: ...

base = Base()
assert_type(base(1), str)
"#,
);

testcase!(
    test_proxy_method_special_dunder_lookup,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class ValidGetitem:
    __getitem__: ProxyMethod["get"]
    def get(self, x: int) -> str: ...

class MissingGetitem:
    __getitem__: ProxyMethod["get"]

valid = ValidGetitem()
assert_type(valid[0], str)
valid["bad"]  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
MissingGetitem()[0]  # E: Proxy method `__getitem__` of class `MissingGetitem` cannot resolve target method `get`
"#,
);

testcase!(
    test_proxy_method_len_dunder_lookup,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class ValidLen:
    __len__: ProxyMethod["length"]
    def length(self) -> int: ...

class MissingLen:
    __len__: ProxyMethod["length"]

assert_type(len(ValidLen()), int)
len(MissingLen())  # E: Argument `MissingLen` is not assignable to parameter `obj` with type `Sized`
"#,
);

testcase!(
    test_proxy_method_super_access,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def forward(self, x: int) -> str: ...

class Child(Base):
    def call_super(self) -> str:
        return super().__call__(1)

assert_type(Child().call_super(), str)
"#,
);

testcase!(
    test_proxy_method_source_form_uses_target_not_body,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def __call__(self, x: str) -> int: ...
    def forward(self, x: int) -> str: ...

base = Base()
assert_type(base(1), str)
assert_type(base.__call__(1), str)
base("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_proxy_method_valid_self_call,
    proxy_method_env(),
    r#"
from typing import Self, assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def forward(self: Self, x: int) -> str: ...
    def call_self(self: Self) -> str:
        return self(1)

base = Base()
assert_type(base.call_self(), str)
"#,
);

testcase!(
    test_proxy_method_protocol_matching,
    proxy_method_env(),
    r#"
from collections.abc import Callable
from typing import Protocol
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]

class A(Base):
    def forward(self, x: int) -> str: ...

class Callback(Protocol):
    def __call__(self, x: int) -> str: ...

class BadCallback(Protocol):
    def __call__(self, x: str) -> str: ...

ok: Callback = A()
bad: BadCallback = A()  # E: `A` is not assignable to `BadCallback`
callable_ok: Callable[[int], str] = A()
callable_bad: Callable[[str], str] = A()  # E: `A` is not assignable to `(str) -> str`
"#,
);

testcase!(
    bug = "`ProxyMethod` override checks should compare against the forwarded signature",
    test_proxy_method_subclass_call_override_wins,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]

class Child(Base):
    def __call__(self, x: str) -> int: ...
    def forward(self, x: int) -> str: ...

child = Child()
assert_type(child("x"), int)
child(1)  # E: Argument `Literal[1]` is not assignable to parameter `x` with type `str`
"#,
);

testcase!(
    test_proxy_method_quoted_annotation,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Base:
    __call__: "ProxyMethod['forward']"
    def forward(self, x: int) -> str: ...

base = Base()
assert_type(base(1), str)
base("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_proxy_method_target_lookup_uses_receiver_class,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class ForwardBase:
    def forward(self, x: int) -> object: ...

class DeclaresProxy(ForwardBase):
    __call__: ProxyMethod["forward"]

class OverridesTarget(DeclaresProxy):
    def forward(self, x: int) -> str: ...

declares = DeclaresProxy()
assert_type(declares(1), object)
declares("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`

overrides = OverridesTarget()
assert_type(overrides(1), str)
overrides("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_proxy_method_non_call_dunder,
    proxy_method_env(),
    r#"
from collections.abc import Iterator
from typing import assert_type
from shape_extensions import ProxyMethod

class IterProxy:
    __iter__: ProxyMethod["iter_impl"]
    def iter_impl(self) -> Iterator[int]: ...

class BrokenIterProxy:
    __iter__: ProxyMethod["missing"]

assert_type(iter(IterProxy()), Iterator[int])
BrokenIterProxy().__iter__  # E: Proxy method `__iter__` of class `BrokenIterProxy` cannot resolve target method `missing`
"#,
);

testcase!(
    test_proxy_method_generic_target_substitution,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class Box[T]:
    __call__: ProxyMethod["forward"]
    def forward(self, x: T) -> T: ...

box = Box[int]()
assert_type(box(1), int)
box("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_proxy_method_generic_protocol_matching,
    proxy_method_env(),
    r#"
from typing import Protocol
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def forward[T](self, x: T) -> T: ...

class GenericCallback(Protocol):
    def __call__[T](self, x: T) -> T: ...

class BadCallback(Protocol):
    def __call__(self, x: str) -> int: ...

ok: GenericCallback = Base()
bad: BadCallback = Base()  # E: `Base` is not assignable to `BadCallback`
"#,
);

testcase!(
    test_proxy_method_target_method_kinds,
    proxy_method_env(),
    r#"
from typing import assert_type
from shape_extensions import ProxyMethod

class StaticTarget:
    __call__: ProxyMethod["forward"]
    @staticmethod
    def forward(x: int) -> str: ...

class ClassTarget:
    __call__: ProxyMethod["forward"]
    @classmethod
    def forward(cls, x: int) -> str: ...

class PropertyTarget:
    __call__: ProxyMethod["forward"]
    @property
    def forward(self) -> int: ...

StaticTarget()()  # E: Proxy method `__call__` of class `StaticTarget` cannot resolve target method `forward`
ClassTarget()()  # E: Proxy method `__call__` of class `ClassTarget` cannot resolve target method `forward`
PropertyTarget()()  # E: Proxy method `__call__` of class `PropertyTarget` cannot resolve target method `forward`
"#,
);

testcase!(
    test_proxy_method_overloaded_target,
    proxy_method_env(),
    r#"
from typing import assert_type, overload
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    @overload
    def forward(self, x: int) -> str: ...
    @overload
    def forward(self, x: str) -> int: ...
    def forward(self, x: int | str) -> str | int: ...

base = Base()
assert_type(base(1), str)
assert_type(base("x"), int)
base(None)  # E: No matching overload found for function `Base.forward`
"#,
);

testcase!(
    test_proxy_method_non_method_target,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    forward: int

Base().__call__  # E: Proxy method `__call__` of class `Base` cannot resolve target method `forward`
Base()()  # E: Proxy method `__call__` of class `Base` cannot resolve target method `forward`
"#,
);

testcase!(
    test_proxy_method_rejects_proxy_chain_and_self_reference,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod

class Chain:
    __call__: ProxyMethod["forward"]
    forward: ProxyMethod["other"]
    def other(self, x: int) -> str: ...

class SelfReference:
    __call__: ProxyMethod["__call__"]

Chain()()  # E: Proxy method `__call__` of class `Chain` cannot resolve target method `forward`
SelfReference()()  # E: Proxy method `__call__` of class `SelfReference` cannot resolve target method `__call__`
"#,
);

testcase!(
    test_proxy_method_missing_target_is_not_getattr_fallback,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def __getattr__(self, name: str) -> object: ...

Base()()  # E: Proxy method `__call__` of class `Base` cannot resolve target method `forward`
Base().__call__  # E: Proxy method `__call__` of class `Base` cannot resolve target method `forward`
"#,
);

testcase!(
    test_proxy_method_invalid_target_union_receiver_reports_proxy_error,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod

class Broken:
    __call__: ProxyMethod["forward"]

class Other:
    pass

def f(x: Broken | Other) -> object:
    return x()  # E: Proxy method `__call__` of class `Broken` cannot resolve target method `forward`  # E: Expected a callable, got `Other`
"#,
);

testcase!(
    test_proxy_method_invalid_self_call_reports_proxy_error,
    proxy_method_env(),
    r#"
from typing import Self
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def call_self(self: Self) -> object:
        return self()  # E: Proxy method `__call__` of class `Base` cannot resolve target method `forward`
"#,
);

testcase!(
    test_proxy_method_valid_target_class_access_is_instance_only,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod

class Base:
    __call__: ProxyMethod["forward"]
    def forward(self, x: int) -> str: ...

Base.__call__  # E: Proxy method `__call__` of class `Base` can only be accessed on instances
"#,
);

testcase!(
    test_proxy_method_invalid_annotations,
    proxy_method_env(),
    r#"
from typing import Final
from shape_extensions import ProxyMethod

class NonString:
    __call__: ProxyMethod[123]  # E: `ProxyMethod` target must be a string literal

class MultipleTargets:
    __call__: ProxyMethod["forward", "other"]  # E: `ProxyMethod` requires exactly one string literal target

class Bare:
    __call__: ProxyMethod  # E: `ProxyMethod` target must be a string literal

class InvalidIdentifier:
    __call__: ProxyMethod["not-an-id"]  # E: `ProxyMethod` target must be a non-empty ASCII identifier

class Wrapped:
    __call__: Final[ProxyMethod["forward"]]  # E: Final attribute declared in class body must be initialized with a value or in `__init__`  # E: `ProxyMethod` may not be wrapped in another annotation
"#,
);

testcase!(
    test_proxy_method_rejects_protocol_initializer_and_alias,
    proxy_method_env(),
    r#"
from typing import Final, Protocol
from shape_extensions import ProxyMethod

type AliasProxy = ProxyMethod["forward"]  # E: `ProxyMethod` is only valid as a direct class member annotation
CallProxy = ProxyMethod["forward"]

class P(Protocol):
    __call__: ProxyMethod["forward"]  # E: `ProxyMethod` cannot be declared in protocols

class Meta(type):
    __call__: ProxyMethod["make"]  # E: `ProxyMethod` cannot be declared in metaclasses
    def make(cls) -> object: ...

class Initialized:
    __call__: ProxyMethod["forward"] = ProxyMethod()  # E: `ProxyMethod` class member annotations may not have class-body initializers
    def forward(self, x: int) -> str: ...

class Aliased:
    __call__: CallProxy  # E: `ProxyMethod` must be used directly as a class member annotation
    def forward(self, x: int) -> str: ...

class QualifiedAlias:
    __call__: Final[CallProxy]  # E: Final attribute declared in class body must be initialized with a value or in `__init__`  # E: `ProxyMethod` must be used directly as a class member annotation
    def forward(self, x: int) -> str: ...
"#,
);

testcase!(
    test_proxy_method_rejects_non_class_member_annotations,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod, ProxyMethod as PM

module_value: ProxyMethod["forward"]  # E: `ProxyMethod` is only valid as a direct class member annotation
module_bare: ProxyMethod  # E: `ProxyMethod` is only valid as a direct class member annotation

def takes_proxy(x: ProxyMethod["forward"]) -> None: ...  # E: `ProxyMethod` is only valid as a direct class member annotation
def takes_proxy_bare(x: PM) -> None: ...  # E: `ProxyMethod` is only valid as a direct class member annotation

def returns_proxy() -> ProxyMethod["forward"]: ...  # E: `ProxyMethod` is only valid as a direct class member annotation
def returns_proxy_bare() -> PM: ...  # E: `ProxyMethod` is only valid as a direct class member annotation

def local() -> None:
    local_value: ProxyMethod["forward"]  # E: `ProxyMethod` is only valid as a direct class member annotation

CallProxy = ProxyMethod["forward"]
alias_hidden: CallProxy  # E: `ProxyMethod` is only valid as a direct class member annotation

class Base:
    def method(self) -> None:
        self.__call__: ProxyMethod["forward"]  # E: `ProxyMethod` is only valid as a direct class member annotation
        self.other: list[ProxyMethod["forward"]]  # E: `ProxyMethod` is only valid as a direct class member annotation

Base().method()
"#,
);

testcase!(
    test_proxy_method_attribute_annotation_rejection_preserves_other_attribute_annotation_context,
    proxy_method_env(),
    r#"
from typing import ClassVar, Final

class Base:
    def method(self) -> None:
        self.class_var: ClassVar[int]
        self.final_no_value: Final
"#,
);

testcase!(
    test_proxy_method_name_is_not_special_by_itself,
    proxy_method_env(),
    r#"
class ProxyMethod[T]: ...

module_value: ProxyMethod[int]

class Base:
    value: ProxyMethod[int]

def takes_proxy(x: ProxyMethod[int]) -> None: ...

def returns_proxy() -> ProxyMethod[int]: ...

def local() -> None:
    local_value: ProxyMethod[int]
"#,
);

testcase!(
    test_proxy_method_rejects_wrappers_and_decorated_source_form,
    proxy_method_env(),
    r#"
import functools
from typing import Annotated
from shape_extensions import ProxyMethod, ProxyMethod as PM

class Wrapped:
    __call__: Annotated[ProxyMethod["forward"], "meta"]  # E: `ProxyMethod` may not be wrapped in another annotation
    def forward(self, x: int) -> str: ...

class ListWrapped:
    __call__: list[ProxyMethod["forward"]]  # E: `ProxyMethod` may not be wrapped in another annotation
    def forward(self, x: int) -> str: ...

class AliasListWrapped:
    __call__: list[PM["forward"]]  # E: `ProxyMethod` may not be wrapped in another annotation
    def forward(self, x: int) -> str: ...

class StaticSource:
    __call__: ProxyMethod["forward"]  # E: `ProxyMethod` source-form declarations require an ordinary instance method body
    @staticmethod
    def __call__(x: str) -> int: ...
    def forward(self, x: int) -> str: ...

class ClassSource:
    __call__: ProxyMethod["forward"]  # E: `ProxyMethod` source-form declarations require an ordinary instance method body
    @classmethod
    def __call__(cls, x: str) -> int: ...
    def forward(self, x: int) -> str: ...

class PropertySource:
    __call__: ProxyMethod["forward"]  # E: `ProxyMethod` source-form declarations require an ordinary instance method body
    @property
    def __call__(self) -> int: ...
    def forward(self, x: int) -> str: ...

class CachedPropertySource:
    __call__: ProxyMethod["forward"]  # E: `ProxyMethod` source-form declarations require an ordinary instance method body
    @functools.cached_property
    def __call__(self) -> int: ...
    def forward(self, x: int) -> str: ...
"#,
);

testcase!(
    test_proxy_method_rejects_constructor_and_post_init_names,
    proxy_method_env(),
    r#"
from shape_extensions import ProxyMethod

class InitProxy:
    __init__: ProxyMethod["init_impl"]  # E: `ProxyMethod` cannot be declared on `__init__`
    def init_impl(self) -> None: ...

class NewProxy:
    __new__: ProxyMethod["new_impl"]  # E: `ProxyMethod` cannot be declared on `__new__`
    def new_impl(cls) -> object: ...

class PostInitProxy:
    __post_init__: ProxyMethod["post_init_impl"]  # E: `ProxyMethod` cannot be declared on `__post_init__`
    def post_init_impl(self) -> None: ...
"#,
);

testcase!(
    test_proxy_method_rejects_namedtuple_and_typeddict,
    proxy_method_env(),
    r#"
from typing import NamedTuple, TypedDict
from shape_extensions import ProxyMethod

class NT(NamedTuple):
    call: ProxyMethod["forward"]  # E: `ProxyMethod` cannot be declared in typed dictionaries or named tuples

class TD(TypedDict):
    call: ProxyMethod["forward"]  # E: `ProxyMethod` cannot be declared in typed dictionaries or named tuples
"#,
);

testcase!(
    test_proxy_method_dataclass_is_not_field,
    proxy_method_env(),
    r#"
from dataclasses import dataclass
from shape_extensions import ProxyMethod

@dataclass
class Base:
    __call__: ProxyMethod["forward"]
    def forward(self, x: int) -> str: ...

Base()
Base(1)  # E: Expected 0 positional arguments, got 1
"#,
);

testcase!(
    test_protocol_base,
    r#"
from typing import Final, Protocol
class C1:
    x: int
    y: str
class P1(Protocol, C1):  # E: If `Protocol` is included as a base class, all other bases must be protocols
    x: int
class P2(Protocol):
    x: int
class P3(Protocol, P2):
    y: str
 "#,
);

testcase!(
    test_callable_protocol,
    r#"
from typing import Callable, Protocol
class P(Protocol):
    def __call__(self, x: int) -> str: ...
def f1(x: int) -> str: ...
def f2(x: str) -> str: ...

p1: P = f1
p2: P = f2  # E: `(x: str) -> str` is not assignable to `P`

def g(p: P) -> None:
    c1: Callable[[int], str] = p
    c2: Callable[[str], str] = p  # E: `P` is not assignable to `(str) -> str`
 "#,
);

testcase!(
    test_protocol_variance,
    r#"
from typing import Protocol
# read-write attributes
class P1(Protocol):
    x: int
class P2(Protocol):
    x: object
# read-only properties
class P3(Protocol):
    @property
    def x(self) -> int: ...
class P4(Protocol):
    @property
    def x(self) -> object: ...
def f(p1: P1, p2: P2, p3: P3, p4: P4):
    # read-write attributes are invariant
    x1: P1 = p2  # E: `P2` is not assignable to `P1`
    x2: P2 = p1  # E: `P1` is not assignable to `P2`
    # properties are covariant w/ the getter/setter types
    x3: P3 = p4  # E: `P4` is not assignable to `P3`
    x4: P4 = p3
    x5: P3 = p1
    x6: P3 = p2  # E: `P2` is not assignable to `P3`
    x7: P4 = p1
    x8: P4 = p2
 "#,
);

testcase!(
    test_protocol_attr_subtype,
    r#"
from typing import Protocol
class P1(Protocol):
    @property
    def x(self) -> int:
        return 1
    @x.setter
    def x(self, y: int) -> None:
        pass
class P2(Protocol):
    x: int
class P3(Protocol):
    @property
    def x(self) -> int:
        return 1
class P4(Protocol):
    @property
    def x(self) -> int:
        return 1
    @x.setter
    def x(self, y: object) -> None:
        pass
class P5(Protocol):
    @property
    def x(self) -> int:
        return 1
    @x.setter
    def x(self, y: str) -> None:
        pass
class ExtendsInt(int): ...
class P6(Protocol):
    @property
    def x(self) -> int:
        return 1
    @x.setter
    def x(self, y: ExtendsInt) -> None:
        pass
def f(p1: P1, p2: P2, p3: P3, p4: P4):
    x1: P1 = p2
    # read-only properties can't be used as read-write properties
    x2: P1 = p3  # E: `P3` is not assignable to `P1`
    # properties can't be used as regular attributes
    x3: P2 = p1  # E: `P1` is not assignable to `P2`
    x4: P2 = p3  # E: `P3` is not assignable to `P2`
    x5: P3 = p1
    x6: P3 = p2
    # setter type compatibility: P2 (x: int) must accept everything the setter promises.
    # P4 setter accepts object, but P2 only accepts int.
    x7: P4 = p2  # E: `P2` is not assignable to `P4`
    # P5 setter accepts str, P2 only accepts int.
    x8: P5 = p2  # E: `P2` is not assignable to `P5`
    # P6 setter accepts ExtendsInt, and ExtendsInt <: int, so P2 can handle it.
    x9: P6 = p2
"#,
);

testcase!(
    test_generic_protocol,
    r#"
from typing import Protocol, TypeVar
T = TypeVar("T")
class P(Protocol[T]):
   x: T
class C1:
   x: int
   y: str
class C2:
   x: str
   y: str
def f(proto: P[str]) -> None: ...
def g(c1: C1, c2: C2) -> None:
    f(c1)  # E: Argument `C1` is not assignable to parameter `proto` with type `P[str]`
    f(c2)
"#,
);

testcase!(
    test_protocol_property,
    r#"
from typing import Protocol
class P1(Protocol):
    @property
    def foo(self) -> object:
        return 1
class C1:
    @property
    def foo(self) -> int:
        return 1
a: P1 = C1()

class P2(Protocol):
    @property
    def foo(self) -> int:
        return 1
class C2:
    @property
    def foo(self) -> object:
        return 1
b: P2 = C2()  # E: `C2` is not assignable to `P2`

class P3(Protocol):
    @property
    def foo(self) -> object:
        return 1
    @foo.setter
    def foo(self, val: object) -> None:
        pass
class C3:
    @property
    def foo(self) -> int:
        return 1
    @foo.setter
    def foo(self, val: int) -> None:
        pass
c: P3 = C3()  # E: `C3` is not assignable to `P3`

class P4(Protocol):
    @property
    def foo(self) -> object:
        return 1
    @foo.setter
    def foo(self, val: int) -> None:
        pass
class C4:
    @property
    def foo(self) -> int:
        return 1
    @foo.setter
    def foo(self, val: object) -> None:
        pass
d: P4 = C4()

class P5(Protocol):
    @property
    def foo(self) -> object:
        return 1
class C5:
    @property
    def foo(self) -> int:
        return 1
    @foo.setter
    def foo(self, val: object) -> None:
        pass
e: P5 = C5()

class P6(Protocol):
    @property
    def foo(self) -> object:
        return 1
    @foo.setter
    def foo(self, val: object) -> None:
        pass
class C6:
    @property
    def foo(self) -> int:
        return 1
f: P6 = C6()  # E: `C6` is not assignable to `P6`
"#,
);

testcase!(
    test_protocol_overload,
    r#"
from typing import Protocol, overload

class P(Protocol):
    @overload
    def foo(self, x: int) -> int: ...
    @overload
    def foo(self, x: str) -> str: ...

class C1:
    @overload
    def foo(self, x: int) -> int: ...
    @overload
    def foo(self, x: str) -> str: ...
    def foo(self, x: int | str) -> int | str:
        return x

x1: P = C1() # OK
"#,
);

testcase!(
    test_hashable,
    r#"
from typing import ClassVar, Hashable
class A:
    pass
class B:
    __hash__: ClassVar[None]
def f(x: Hashable):
    pass
f(A())
f(B())  # E: Argument `B` is not assignable to parameter `x` with type `Hashable`
    "#,
);

testcase!(
    test_protocol_instantiation,
    r#"
from typing import Protocol
class A(Protocol):
    pass
a: A = A()  # E: Cannot instantiate `A` because it is a protocol

class B(A):
    pass
type_a: type[A] = B
a = type_a()  # This is OK because it's not a bare class name
    "#,
);

testcase!(
    test_protocol_stub_method_instantiation_error,
    r#"
from typing import Protocol

class Proto(Protocol):
    def method(self) -> int: ...

class Concrete(Proto):
    pass

Concrete()  # E: Cannot instantiate `Concrete`
"#,
);

testcase!(
    test_protocol_getattr,
    r#"
from typing import Protocol
class P(Protocol):
    x: int
def f(proto: P) -> None: ...

class C:
    def __getattr__(self, name: str) -> int: ...

f(C()) # E: Argument `C` is not assignable to parameter `proto` with type `P`
    "#,
);

testcase!(
    bug = "The conformance tests require that we accept this, and mypy and pyright do so, but it is unsound. Consider emitting an error.",
    test_self_param,
    r#"
from typing import Protocol, Self
class P(Protocol):
    def f(self, x: Self):
        pass
class C:
    def f(self, x: Self):
        pass
def f(x: P):
    pass
f(C())
    "#,
);

testcase!(
    test_call_protocol_with_other_attr,
    r#"
from typing import Protocol, assert_type
class P(Protocol):
    x: int
    def __call__(self, x: int) -> str: ...
def decorate(func) -> P: ...
@decorate
def f():
    pass
assert_type(f.x, int)
    "#,
);

testcase!(
    test_protocol_with_implicit_attr_assigned_in_method,
    r#"
from typing import Protocol
class P(Protocol):
    x: int
    def __init__(self, x: int, z: str):
        self.x = x  # ok
        self.z = z  # E: Protocol variables must be explicitly declared in the class body
    def f(self, x: int, y: str):
        self.x = x  # ok
        self.y = y  # E: Protocol variables must be explicitly declared in the class body
    "#,
);

testcase!(
    test_protocol_runtime_checkable_isinstance,
    r#"
from typing import Protocol, runtime_checkable

# Protocol without @runtime_checkable
class NonRuntimeProtocol(Protocol):
    def method(self) -> int: ...

# Protocol with @runtime_checkable
@runtime_checkable
class RuntimeProtocol(Protocol):
    def method(self) -> int: ...

class ConcreteClass:
    def method(self) -> int:
        return 42

obj = ConcreteClass()

# These should fail - protocol not decorated with @runtime_checkable
isinstance(obj, NonRuntimeProtocol)  # E: Protocol `NonRuntimeProtocol` is not decorated with @runtime_checkable and cannot be used with isinstance()
issubclass(ConcreteClass, NonRuntimeProtocol)  # E: Protocol `NonRuntimeProtocol` is not decorated with @runtime_checkable and cannot be used with issubclass()

# These should work - protocol is decorated with @runtime_checkable
isinstance(obj, RuntimeProtocol)
issubclass(ConcreteClass, RuntimeProtocol)
"#,
);

testcase!(
    test_protocol_union_isinstance,
    r#"
from typing import Protocol, runtime_checkable, Union

@runtime_checkable
class Protocol1(Protocol):
    def method1(self) -> int: ...

class NonRuntimeProtocol(Protocol):
    def method2(self) -> str: ...

@runtime_checkable
class DataProtocol(Protocol):
    x: int

class ConcreteClass:
    x: int = 42
    def method1(self) -> int:
        return 1
    def method2(self) -> str:
        return "hello"

obj = ConcreteClass()

# Union with non-runtime-checkable protocol should fail
isinstance(obj, (Protocol1, NonRuntimeProtocol))  # E: Protocol `NonRuntimeProtocol` is not decorated with @runtime_checkable and cannot be used with isinstance()

# issubclass with data protocol in union should fail
issubclass(ConcreteClass, (Protocol1, DataProtocol))  # E: Protocol `DataProtocol` has non-method members and cannot be used with issubclass()
"#,
);

testcase!(
    test_protocol_narrowing_behavior_unions,
    r#"
from typing import Protocol, runtime_checkable, Union

@runtime_checkable
class ReadableProtocol(Protocol):
    def read(self) -> str: ...

@runtime_checkable
class WritableProtocol(Protocol):
    def write(self, data: str) -> None: ...

class File:
    def read(self) -> str:
        return "data"
    def write(self, data: str) -> None:
        pass

class ReadOnlyFile:
    def read(self) -> str:
        return "data"

def process_file(f: Union[ReadableProtocol, WritableProtocol]) -> None:
    if isinstance(f, ReadableProtocol):
        data = f.read()
    if isinstance(f, WritableProtocol):
        f.write("test")

# These should work
process_file(File())
process_file(ReadOnlyFile())
"#,
);

testcase!(
    test_protocol_difference_data_vs_non_data,
    r#"
from typing import Protocol, runtime_checkable

# Data protocol with both data and methods
@runtime_checkable
class MixedDataProtocol(Protocol):
    name: str
    value: int
    def process(self) -> None: ...

# Non-data protocol with only methods
@runtime_checkable
class MethodOnlyProtocol(Protocol):
    def process(self) -> None: ...
    def validate(self) -> bool: ...

# Protocol with only data (no methods)
@runtime_checkable
class DataOnlyProtocol(Protocol):
    name: str
    value: int

class Implementation:
    name: str = "test"
    value: int = 42

    def process(self) -> None:
        pass

    def validate(self) -> bool:
        return True

# isinstance should work for all
isinstance(Implementation(), MixedDataProtocol)
isinstance(Implementation(), MethodOnlyProtocol)
isinstance(Implementation(), DataOnlyProtocol)

# issubclass should only work for non-data protocols
issubclass(Implementation, MethodOnlyProtocol)  # OK - only methods
issubclass(Implementation, MixedDataProtocol)   # E: Protocol `MixedDataProtocol` has non-method members and cannot be used with issubclass()
issubclass(Implementation, DataOnlyProtocol)   # E: Protocol `DataOnlyProtocol` has non-method members and cannot be used with issubclass()
"#,
);

testcase!(
    test_runtime_checkable_non_protocol,
    r#"
from typing import runtime_checkable

# Applying @runtime_checkable to a non-protocol class should fail
@runtime_checkable  # E: @runtime_checkable can only be applied to Protocol classes
class RegularClass:
    def method(self) -> int:
        return 42

# This should also fail
@runtime_checkable  # E: @runtime_checkable can only be applied to Protocol classes
class AnotherClass:
    x: int = 5
"#,
);

testcase!(
    test_union_as_protocol,
    r#"
from typing import Protocol
class P(Protocol):
    x: int
class A:
    x: int
class B:
    y: int
def f[T: A | B](direct: A | B, quantified: T) -> None:
    p: P = direct  # E: `A | B` is not assignable to `P`
    p: P = quantified  # E: `T` is not assignable to `P`
    "#,
);

testcase!(
    test_callback_protocol_generic,
    r#"
from typing import Protocol
class C(Protocol):
    def __call__[T](self, x: T) -> T:
        return x
def f[T](x: T) -> T:
    return x
def g(x: int) -> int:
    return x
c: C = f
c: C = g  # E: `(x: int) -> int` is not assignable to `C`
    "#,
);

testcase!(
    test_protocol_return_self,
    r#"
from typing import Protocol, Self, runtime_checkable

@runtime_checkable
class CanAddSelf(Protocol):
    def __add__(self, other: Self, /) -> Self: ...

assert isinstance(42, CanAddSelf)
    "#,
);

testcase!(
    test_protocol_self_tvar,
    r#"
from typing import Protocol

class P(Protocol):
    def f[T: 'P'](self: T) -> T:
        return self

class C:
    def f[T: 'C'](self: T) -> T:
        return self

x: P = C() # OK
    "#,
);

testcase!(
    test_assign_to_type_protocol,
    r#"
from typing import Protocol

class CanFly(Protocol):
    def fly(self) -> None: ...

class A:
    def __init__(self, wingspan: float) -> None: ...
    def fly(self) -> None: ...

cls1: type[CanFly] = CanFly # E: `type[CanFly]` is not assignable to `type[CanFly]`
cls2: type[CanFly] = A      # OK
    "#,
);

testcase!(
    test_runtime_checkable_unsafe_overlap,
    r#"
from typing import Protocol, runtime_checkable
@runtime_checkable
class UnsafeProtocol(Protocol):
    def foo(self) -> int: ...
class No:
    def foo(self) -> str:
        return "not an int"
isinstance(No(), UnsafeProtocol) # E: Runtime checkable protocol `UnsafeProtocol` has an unsafe overlap with type `No`
issubclass(No, UnsafeProtocol) # E: Runtime checkable protocol `UnsafeProtocol` has an unsafe overlap with type `No`
    "#,
);

testcase!(
    test_runtime_checkable_unsafe_overlap_with_inheritance,
    r#"
from typing import Protocol, runtime_checkable
@runtime_checkable
class UnsafeProtocol(Protocol):
    def foo(self) -> int: ...
@runtime_checkable
class ChildUnsafeProtocol(UnsafeProtocol, Protocol):
    def bar(self) -> str: ...
class No:
    def foo(self) -> str:
        return "not an int"
    def bar(self) -> int:
        return 42
isinstance(No(), ChildUnsafeProtocol) # E: Runtime checkable protocol `ChildUnsafeProtocol` has an unsafe overlap with type `No`
issubclass(No, ChildUnsafeProtocol) # E: Runtime checkable protocol `ChildUnsafeProtocol` has an unsafe overlap with type `No`
    "#,
);

testcase!(
    test_unsafe_overlap_with_abc,
    r#"
from collections.abc import Sized
class X:
    def __len__(self) -> str:
        return "42"
isinstance(X(), Sized) # E: Runtime checkable protocol `Sized` has an unsafe overlap with type `X`
issubclass(X, Sized) # E: Runtime checkable protocol `Sized` has an unsafe overlap with type `X`
"#,
);

testcase!(
    test_runtime_checkable_protocol_never_no_unsafe_overlap,
    r#"
from collections.abc import Iterable
from typing import Never

def f(x: Never) -> None:
    if isinstance(x, Iterable):
        pass
"#,
);

testcase!(
    test_runtime_checkable_missing_members_do_not_overlap,
    r#"
from typing import Any, Generator, Iterable, Protocol, runtime_checkable

def test1(a: Iterable[Any]) -> None:
    isinstance(a, Generator)

@runtime_checkable
class P(Protocol):
    x: int
    y: int

class C:
    x: str

def test2(x: C):
    isinstance(x, P)
"#,
);

testcase!(
    test_runtime_checkable_generics_no_error,
    r#"
from typing import Protocol, runtime_checkable, TypeVar
T = TypeVar('T')
@runtime_checkable
class GenericProtocol(Protocol[T]):
    def get(self, x: T) -> T: ...

class IntImpl:
    def get(self, x: int) -> int:
        return 42
isinstance(IntImpl(), GenericProtocol)
issubclass(IntImpl, GenericProtocol)
"#,
);

testcase!(
    test_runtime_checkable_generics_unsafe_overlap_inconsistent_within_method,
    r#"
from typing import Protocol, runtime_checkable, TypeVar
T = TypeVar('T')
@runtime_checkable
class GenericProtocol(Protocol[T]):
    def get(self, x: T) -> T: ...

class IntImpl:
    def get(self, x: str) -> int:
        return 42
isinstance(IntImpl(), GenericProtocol)  # E: Runtime checkable protocol `GenericProtocol` has an unsafe overlap with type `IntImpl`
issubclass(IntImpl, GenericProtocol)  # E: Runtime checkable protocol `GenericProtocol` has an unsafe overlap with type `IntImpl`
"#,
);

testcase!(
    test_runtime_checkable_generics_unsafe_overlap_inconsistent_across_methods,
    r#"
from typing import Protocol, runtime_checkable, TypeVar
T = TypeVar('T')
@runtime_checkable
class GenericProtocol(Protocol[T]):
    def get(self, x: T) -> T: ...
    def get2(self, x: T) -> T: ...

class Impl:
    def get(self, x: str) -> str:
        return ""
    def get2(self, x: int) -> int:
        return 42
isinstance(Impl(), GenericProtocol)  # E: Runtime checkable protocol `GenericProtocol` has an unsafe overlap with type `Impl`
issubclass(Impl, GenericProtocol)  # E: Runtime checkable protocol `GenericProtocol` has an unsafe overlap with type `Impl`
"#,
);

testcase!(
    test_runtime_checkable_protocol_bound_violation,
    r#"
from typing import Protocol, runtime_checkable, TypeVar

T = TypeVar('T', bound=str)
@runtime_checkable
class GenericProtocol(Protocol[T]):
    def get(self, x: T) -> T: ...

class IntImpl:
    def get(self, x: int) -> int:
        return x

isinstance(IntImpl(), GenericProtocol)  # E: `int` is not assignable to upper bound `str` of type variable `T`
    "#,
);

testcase!(
    test_protocol_with_uninit_classvar,
    r#"
from typing import Protocol, ClassVar, final
class P(Protocol):
    x: ClassVar[int]

@final
class C(P): # E: Final class `C` cannot have unimplemented abstract members: `x`
    pass

c = C()  # E: Cannot instantiate `C` because the following members are abstract: `x`
"#,
);

testcase!(
    test_check_protocol_upper_bound,
    r#"
from typing import Protocol
class A(Protocol):
    x: int
class B:
    x: int
class C:
    pass
def f[T: A](a: T) -> T:
    return a
def g(b: B, c: C):
    f(b)
    f(c)  # E: `C` is not assignable to upper bound `A`
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1905
testcase!(
    test_functor_protocol_and_impl,
    r#"
from typing import Generic, TypeVar, Protocol, Callable

T = TypeVar('T')
U = TypeVar('U')

class Functor(Protocol[T]):  # E: Type variable `T` in class `Functor` is declared as invariant, but could be covariant based on its usage
    """A Functor protocol - common in functional programming."""
    def map(self, f: Callable[[T], U]) -> Functor[U]: ...

class Maybe(Generic[T]):
    """A Maybe/Option type that should implement Functor."""
    value: T | None
    def map(self, f: Callable[[T], U]) -> Maybe[U]: ...

def test():
    m: Maybe[int] = ...  # type: ignore
    f: Functor[int] = m  # Should work now!
"#,
);

// Regression test for a case an early implementation of https://github.com/facebook/pyrefly/issues/1905 got wrong
testcase!(
    test_second_order_protocol_subset_failure,
    r#"
from typing import Generic, TypeVar, Protocol, Callable

T = TypeVar('T')
U = TypeVar('U')

class TrickyProtocol(Protocol[T]):  # E: Type variable `T` in class `TrickyProtocol` is declared as invariant, but could be covariant based on its usage
    def recurse(self, f: Callable[[T], U]) -> "TrickyProtocol[U]": ...
    def check(self) -> T: ...

class TrickyImpl(Generic[T]):
    def recurse(self, f: Callable[[T], U]) -> "TrickyImpl[U]": ...
    def check(self) -> int: ...

def test():
    t: TrickyImpl[int] = TrickyImpl()
    # Invalid because p.recurse(lambda i: str(i)).check() returns int, but
    # it should return `str` if we fully implemented the protocol
    p: TrickyProtocol[int] = t  # E:
"#,
);

testcase!(
    test_protocols_class_objects_conformance,
    r#"
from typing import ClassVar, Protocol

class ProtoC1(Protocol):
    attr1: ClassVar[int]

class ProtoC2(Protocol):
    attr1: int

class ConcreteC1:
    attr1: ClassVar[int] = 1

class ConcreteC2:
    attr1: int = 1

class CMeta(type):
    attr1: int
    def __init__(self, attr1: int) -> None:
        self.attr1 = attr1

class ConcreteC3(metaclass=CMeta): ...

pc1: ProtoC1 = ConcreteC1  # E: `type[ConcreteC1]` is not assignable to `ProtoC1`
pc3: ProtoC1 = ConcreteC2  # E: `type[ConcreteC2]` is not assignable to `ProtoC1`
pc4: ProtoC2 = ConcreteC2  # E: `type[ConcreteC2]` is not assignable to `ProtoC2`
pc5: ProtoC1 = ConcreteC3  # E: `type[ConcreteC3]` is not assignable to `ProtoC1`
"#,
);

testcase!(
    test_protocols_definition_conformance,
    r#"
from typing import ClassVar, Protocol, Sequence

class Template(Protocol):
    name: str
    value: int = 0
    def method(self) -> None:
        self.temp: list[int] = []  # E: Protocol variables must be explicitly declared in the class body

class Concrete:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value
    def method(self) -> None:
        return

var: Template = Concrete("value", 42)

class Template2(Protocol):
    val1: ClassVar[Sequence[int]]

class Concrete2_Bad3:
    val1: list[int] = [2]

class Concrete2_Bad4:
    val1: Sequence[int] = [2]

v2_bad3: Template2 = Concrete2_Bad3()  # E: `Concrete2_Bad3` is not assignable to `Template2`
v2_bad4: Template2 = Concrete2_Bad4()  # E: `Concrete2_Bad4` is not assignable to `Template2`
"#,
);

fn env_protocols_modules() -> TestEnv {
    TestEnv::one(
        "_protocols_modules1",
        r#"
"""
Support file for protocols_modules.py test.
"""

timeout = 100
one_flag = True
other_flag = False
"#,
    )
}

testcase!(
    test_protocols_modules_conformance,
    env_protocols_modules(),
    r#"
import _protocols_modules1
from typing import Protocol

class Options1(Protocol):
    timeout: int
    one_flag: bool
    other_flag: bool

op1: Options1 = _protocols_modules1
"#,
);

testcase!(
    test_protocol_any_args_kwargs,
    r#"
from typing import Generic, Sized, Iterable, Any, TypeVar, Protocol, Self

class NativeSeries(Protocol):
    def filter(self, *args: Any, **kwargs: Any) -> Any: ...

class MySeries:
    def filter(self, _predicate: Iterable[bool]) -> Self:
        return self

T = TypeVar('T', bound=NativeSeries)

class Foo(Generic[T]):
    ...

def to_foo() -> Foo[MySeries]:
    ...
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1202
testcase!(
    test_assert_type_structurally_identical_protocols,
    r#"
from typing import Protocol, assert_type

class F1(Protocol):
    a: int

class F2(Protocol):
    a: int

def f(f2: F2):
    assert_type(f2, F1)
"#,
);

// https://github.com/facebook/pyrefly/issues/2925
testcase!(
    test_protocol_ambiguous_member,
    r#"
from typing import Protocol

class Ambiguous(Protocol):
    x = None  # E: Protocol member `x` must have an explicit type annotation
    y = ...  # E: Protocol member `y` must have an explicit type annotation

class Ok(Protocol):
    x: int
    y: str = "default"
"#,
);

testcase!(
    test_protocol_overloaded_method_filtered_by_self,
    r#"
from __future__ import annotations
from datetime import timedelta
from typing import Generic, TypeVar, Protocol, overload, assert_type

T_contra = TypeVar("T_contra", contravariant=True)
S1_co = TypeVar("S1_co", bound=timedelta | int | float, covariant=True)
S2_co = TypeVar("S2_co", bound=timedelta | int | float, covariant=True)

class SupportsProtoTrueDiv(Protocol[T_contra, S2_co]):
    def _proto_truediv(self, other: T_contra, /) -> ElementOpsMixin[S2_co]: ...

class ElementOpsMixin(Protocol, Generic[S2_co]):
    @overload
    def _proto_truediv(self: ElementOpsMixin[int], other: int, /) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_truediv(self: ElementOpsMixin[timedelta], other: timedelta, /) -> ElementOpsMixin[float]: ...

class Series(ElementOpsMixin[S2_co], Protocol):
    def __truediv__(self: SupportsProtoTrueDiv[T_contra, S1_co], other: T_contra) -> Series[S1_co]: ...

def main2(s: Series[timedelta]) -> None:
    td = timedelta(1)
    assert_type(s / td, Series[float])
"#,
);

testcase!(
    test_protocol_overloaded_generic_self_referencing_protocol_terminates,
    r#"
from typing import Protocol, TypeVar, overload

S = TypeVar("S", covariant=True)
R = TypeVar("R", covariant=True)


class Lens(Protocol[S, R]):
    @overload
    def __call__[S2, R2](self: Lens[S2, R2], state: S2, /, value: R2) -> S2: ...
    @overload
    def __call__[S2, R2](self: Lens[S2, R2], state: S2, /) -> R2: ...


class BaseLens(Lens[S, R], Protocol):
    def at(self) -> Lens[S, R]:
        return self
"#,
);

testcase!(
    test_protocol_overloaded_generic_self_mutual_recursion_terminates,
    r#"
from typing import Protocol, TypeVar, overload

S = TypeVar("S", covariant=True)


class A(Protocol[S]):
    @overload
    def __call__[S2, R2](self: B[S2], state: S2, /, value: R2) -> S2: ...
    @overload
    def __call__[S2, R2](self: B[S2], state: S2, /) -> R2: ...


class B(Protocol[S]):
    @overload
    def __call__[S2, R2](self: A[S2], state: S2, /, value: R2) -> S2: ...
    @overload
    def __call__[S2, R2](self: A[S2], state: S2, /) -> R2: ...


class Impl(A[S], B[S], Protocol):
    def at(self) -> A[S]:
        return self
"#,
);

testcase!(
    test_protocol_overloaded_generic_self_non_conforming_still_rejected,
    r#"
from typing import Protocol, TypeVar, overload

S = TypeVar("S", covariant=True)
R = TypeVar("R", covariant=True)


class Lens(Protocol[S, R]):
    @overload
    def __call__[S2, R2](self: Lens[S2, R2], state: S2, /, value: R2) -> S2: ...
    @overload
    def __call__[S2, R2](self: Lens[S2, R2], state: S2, /) -> R2: ...
    def extra(self) -> int: ...


class HasCallNoExtra(Protocol[S, R]):
    @overload
    def __call__[S2, R2](self: HasCallNoExtra[S2, R2], state: S2, /, value: R2) -> S2: ...
    @overload
    def __call__[S2, R2](self: HasCallNoExtra[S2, R2], state: S2, /) -> R2: ...


def f(x: HasCallNoExtra[int, str]) -> Lens[int, str]:
    return x  # E: not assignable to declared return type
"#,
);

testcase!(
    test_protocol_overload_typevar_self_kept,
    r#"
from typing import Protocol, overload

class C:
    @overload
    def f[S](self: S, x: int) -> S: ...
    @overload
    def f(self, x: str) -> str: ...
    def f(self, x: int | str) -> object: ...

class P(Protocol):
    def f(self, x: int) -> object: ...

p: P = C()  # the `self: S` overload binds S=C, satisfying P
"#,
);

testcase!(
    test_protocol_overload_parameterized_self_kept,
    r#"
from typing import Protocol, overload

class Arr[S, T]:
    @overload
    def astype[S2, T2](self: Arr[S2, T2], dtype: int) -> Arr[S2, T2]: ...
    @overload
    def astype(self, dtype: str) -> bytes: ...
    def astype(self, dtype: object) -> object: ...

class HasAstype(Protocol):
    def astype(self, dtype: int) -> object: ...

def g(a: Arr[int, float]) -> HasAstype:
    return a  # the `self: Arr[S2, T2]` overload binds to the receiver, satisfying P
"#,
);

testcase!(
    test_protocol_overload_partially_concrete_self_filtered,
    r#"
from typing import Protocol, overload

class Arr[S, T]:
    @overload
    def m[S2](self: Arr[S2, int], x: int) -> int: ...
    @overload
    def m(self, x: str) -> str: ...
    def m(self, x: object) -> object: ...

class HasM(Protocol):
    def m(self, x: int) -> int: ...

def g(a: Arr[float, str]) -> HasM:
    # `self: Arr[S2, int]` pins T=int; receiver has T=str, so that overload does not
    # apply and `Arr[float, str]` does not satisfy `HasM`.
    return a  # E: not assignable to declared return type
"#,
);
