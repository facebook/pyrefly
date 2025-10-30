/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_abstract_class_instantiation_error,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

# This should error - cannot instantiate
shape = Shape()  # E: Cannot instantiate `Shape`
"#,
);

testcase!(
    test_concrete_subclass_instantiation_ok,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def area(self) -> float:
        return 3.14

# This should work - concrete subclass can be instantiated
circle = Circle()
"#,
);

testcase!(
    test_polymorphic_calls_ok,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def area(self) -> float:
        return 3.14

def calculate_area(shape: Shape) -> float:
    # This should work - polymorphic call is allowed
    return shape.area()

circle = Circle()
area = calculate_area(circle)
"#,
);

testcase!(
    test_multiple_abstract_methods,
    r#"
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def erase(self) -> None:
        pass

# This should error - class has multiple abstract methods
drawable = Drawable()  # E: Cannot instantiate `Drawable`
"#,
);

testcase!(
    test_inherited_abstract_method,
    TestEnv::new().enable_implicit_abstract_class_error(),
    r#"
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def method(self) -> None:
        pass

class Child(Base):  # E: Class `Child` has unimplemented abstract members: `method`
    # Child doesn't implement method, so it's still abstract
    pass

# This should error - child class is still abstract
child = Child()  # E: Cannot instantiate `Child`
"#,
);

testcase!(
    test_final_class_with_abstract_methods,
    r#"
from typing import final, Protocol
from abc import ABC, abstractmethod

@final
class BadClass(ABC):  # E: Final class `BadClass` cannot have unimplemented abstract members: `method`
    @abstractmethod
    def method(self) -> None:
        pass

x = BadClass()  # E: Cannot instantiate `BadClass`

@final
class AbstractProtocol(Protocol):
    @abstractmethod
    def method(self) -> None:
        pass
"#,
);

testcase!(
    test_partial_implementation,
    TestEnv::new().enable_implicit_abstract_class_error(),
    r#"
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def method1(self) -> None:
        pass

    @abstractmethod
    def method2(self) -> None:
        pass

class Partial(Base):  # E: Class `Partial` has unimplemented abstract members: `method2`
    def method1(self) -> None:
        print("implemented")

    # method2 is not implemented

# Should error - not all abstract methods are implemented
p = Partial()  # E: Cannot instantiate `Partial`
"#,
);

testcase!(
    test_overloaded_abstract_method,
    r#"
from abc import ABC, abstractmethod
from typing import overload

class Base(ABC):
    @overload
    @abstractmethod
    def method(self, x: int) -> int: ...

    @overload
    @abstractmethod
    def method(self, x: str) -> str: ...

    @abstractmethod
    def method(self, x):
        # Abstract method, but needs to match overload signatures for type checking
        return x

# Should error - has abstract overloaded method
b = Base()  # E: Cannot instantiate `Base`

class Concrete(Base):
    @overload
    def method(self, x: int) -> int: ...

    @overload
    def method(self, x: str) -> str: ...

    def method(self, x):
        return x

# Should work - overloaded method is implemented
c = Concrete()
"#,
);

testcase!(
    test_super_abstract_call,
    r#"
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def method(self) -> str:
        pass

class Child(Base):
    def method(self) -> str:
        # Calling abstract method via super() should be allowed (no error)
        # Even though it would fail at runtime, type checkers don't error here
        super().method()
        return "child"

# Child is concrete, so this works
c = Child()
"#,
);

testcase!(
    test_abstract_property,
    TestEnv::new().enable_implicit_abstract_class_error(),
    r#"
from typing import *
from abc import ABC, abstractmethod
class Base(ABC):
    def __init__(self) -> None: pass

    @property
    @abstractmethod
    def processor(self) -> bool: pass

class Child(Base):  # E: Class `Child` has unimplemented abstract members: `processor`
    def __init__(self) -> None:
        super().__init__()

x = Child()  # E: Cannot instantiate `Child`
"#,
);

testcase!(
    test_no_error_for_type_of_class,
    r#"
from abc import ABC, abstractmethod

class A(ABC):
    @abstractmethod
    def m(self) -> None: ...

    @classmethod
    def classm(cls) -> None:
        cls() # should not error

def test(cls: type[A]):
    cls() # should not error
"#,
);

testcase!(
    test_abstract_async_iterator,
    TestEnv::new().enable_implicit_abstract_class_error(),
    r#"
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

# error
class A(ABC):
    @abstractmethod
    async def foo(self) -> AsyncIterator[int]:  # E: Abstract methods for async generators should use `def`, not `async def`
        pass

class B(A):
    async def foo(self) -> AsyncIterator[int]:
        yield 1

# ok
class C(ABC):
    @abstractmethod
    def foo(self) -> AsyncIterator[int]:
        pass

    @abstractmethod
    async def bar(self) -> Any:
        pass

class D(C):  # E: Class `D` has unimplemented abstract members: `bar`
    async def foo(self) -> AsyncIterator[int]:
        yield 1
    "#,
);
