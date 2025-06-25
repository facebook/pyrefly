/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_override_any,
    r#"
from typing import override, Any
 
class ParentB(Any):
    pass


class ChildB(ParentB):
    @override
    def method1(self) -> None:
        pass        
 "#,
);

testcase!(
    test_override_basic_method,
    r#"
 
class A:
    def f(self, x:str, y:str) -> str:
        return x + y

class B(A):
    def f(self, x:int, y:int) -> int: # E: Class member `B.f` overrides parent class `A` in an inconsistent manner
        return x + y        
 "#,
);

testcase!(
    test_override_basic_field,
    r#"
class A:
    x: int
    y: bool
    z: bool

class B(A):
    pass

class C(B):
    x: int
    y: str # E: Class member `C.y` overrides parent class `B` in an inconsistent manner
 "#,
);

testcase!(
    test_override_class_var,
    r#"
from typing import ClassVar
class A:
    x: int = 1
class B:
    x: ClassVar[int] = 1
class C(A):
    x: ClassVar[int] = 1  # E: ClassVar `C.x` overrides instance variable of the same name in parent class `A`
class D(B):
    x: ClassVar[int] = 1  # OK
class E(B):
    x: int = 1  # E: Instance variable `E.x` overrides ClassVar of the same name in parent class `B`
 "#,
);

testcase!(
    test_override_final_var,
    r#"
from typing import Final
class A:
    x: Final = 1
    y: Final[int] = 1
class B(A):
    x = 1  # E: `x` is declared as final in parent class `A`
    y = 1  # E: `y` is declared as final in parent class `A`
 "#,
);

testcase!(
    test_overload_override,
    r#"
from typing import overload

class A:

    @overload
    def method(self, x: int) -> int:
        ...

    @overload
    def method(self, x: str) -> str:
        ...

    def method(self, x: int | str) -> int | str:
        return 0


class B(A):

    @overload
    def method(self, x: int) -> int:
        ...

    @overload
    def method(self, x: str) -> str:
        ...

    def method(self, x: int | str) -> int | str:
        return 0
 "#,
);

testcase!(
    test_no_base_override,
    r#"
from typing import override

class A:
    def method1(self) -> int:
        return 1


class B(A):
    @override
    def method2(self) -> int: # E: Class member `B.method2` is marked as an override, but no parent class has a matching attribute
        return 1
 "#,
);

testcase!(
    test_default_value_consistent,
    r#"
class A:
    x: int

class B(A):
    def __init__(self) -> None:
        self.x = 0
 "#,
);

testcase!(
    test_default_value_inconsistent,
    r#"
class A:
    x: str

class B(A):
    def __init__(self) -> None:
        self.x = 0 # E: `Literal[0]` is not assignable to attribute `x` with type `str`
 "#,
);

testcase!(
    test_override_decorators,
    r#"
from typing import override

class ParentA:
    pass

class ChildA(ParentA):
    @staticmethod
    @override
    def static_method1() -> int: # E: Class member `ChildA.static_method1` is marked as an override, but no parent class has a matching attribute
        return 1

    @classmethod
    @override
    def class_method1(cls) -> int: # E: Class member `ChildA.class_method1` is marked as an override, but no parent class has a matching attribute
        return 1

    @property
    @override
    def property1(self) -> int: # E: Class member `ChildA.property1` is marked as an override, but no parent class has a matching attribute
        return 1
    
 "#,
);

testcase!(
    test_override_decorators_switched,
    r#"
from typing import override

class ParentA:
    pass

class ChildA(ParentA):
    @override
    @staticmethod
    def static_method1() -> int: # E: Class member `ChildA.static_method1` is marked as an override, but no parent class has a matching attribute
        return 1
    
 "#,
);

testcase!(
    test_override_custom_wrapper,
    r#"
from typing import Any, Callable, override

def wrapper(func: Callable[..., Any], /) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    return wrapped


class ParentA:

    @staticmethod
    def static_method1() -> int:
        return 1

class ChildA(ParentA):

    @wrapper
    @override
    @staticmethod
    def static_method1() -> bool: 
        return True
 "#,
);

testcase!(
    test_override_duplicate_decorator,
    r#"
from typing import  override

class ParentA:

    @staticmethod
    def static_method1() -> int:
        return 1

class ChildA(ParentA):

    @staticmethod
    @override
    @staticmethod
    def static_method1() -> int:
        return 1    
 "#,
);

testcase!(
    bug = "TODO: method4 should be marked as an error since it doesn't exist in the parent class",
    test_overload_override_error,
    r#"

from typing import overload, override

class ParentA:
    ...

class ChildA(ParentA):
    @overload
    def method4(self, x: int) -> int:
        ...

    @overload
    def method4(self, x: str) -> str:
        ...

    @override
    def method4(self, x: int | str) -> int | str: 
        return 0
 "#,
);

testcase!(
    test_override_final_method,
    r#"
from typing import final

class Parent:
    @final
    def a(self): ...

class Child(Parent):
    def a(self): ...  # E: `a` is declared as final in parent class `Parent`
 "#,
);

testcase!(
    test_override_literal_attr,
    r#"
from typing import Literal
class A:
    X: Literal[0] = 0
class B(A):
    X = 0
    "#,
);

testcase!(
    test_bad_override_literal_attr,
    r#"
from typing import Literal
class A:
    x: Literal[0] = 0
class B(A):
    x = 1 # E: `Literal[1]` is not assignable to attribute `x` with type `Literal[0]`
    "#,
);

testcase!(
    test_override_context_manager,
    r#"
import contextlib
import abc

class Parent:
    @contextlib.asynccontextmanager
    @abc.abstractmethod
    async def run(self):
        yield

class Child(Parent):
    @contextlib.asynccontextmanager
    async def run(self):
        yield
    "#,
);

testcase!(
    test_override_with_args_and_kwargs,
    r#"
from typing import *

class A:
    def test1(self): ...
    def test2(self, x: int, /): ...
    def test3(self, x: int = 1, /): ...
    def test4(self, x: int): ...
    def test5(self, x: int = 1): ...

class B(A):
    def test1(self, *args: int, **kwargs: int): ...
    def test2(self, *args: int, **kwargs: int): ...
    def test3(self, *args: int, **kwargs: int): ...
    def test4(self, *args: int, **kwargs: int): ...
    "#,
);

testcase!(
    test_override_with_unannotated_args_and_kwargs,
    r#"
from typing import *

class A:
    def test1(self): ...
    def test2(self, x: int, /): ...
    def test3(self, x: int = 1, /): ...
    def test4(self, x: int): ...
    def test5(self, x: int = 1): ...

class B(A):
    def test1(self, *args, **kwargs): ...
    def test2(self, *args, **kwargs): ...
    def test3(self, *args, **kwargs): ...
    def test4(self, *args, **kwargs): ...
    "#,
);

testcase!(
    test_override_with_args_and_kwonly,
    r#"
from typing import *

class A:
    def test1(self, x: int, /): ...
    def test2(self, x: int = 1, /): ...
    def test3(self, x: int): ...
    def test4(self, x: int): ...

class B(A):
    def test1(self, *args: int): ...
    def test2(self, *args: int): ...
    # This one is not OK because the kw-only x is required
    # If A.test3 passes x positionally then it will crash
    def test3(self, *args: int, x: int): ...  # E: Class member `B.test3` overrides parent class `A` in an inconsistent manner
    def test4(self, *args: int, x: int = 1): ...
    "#,
);
