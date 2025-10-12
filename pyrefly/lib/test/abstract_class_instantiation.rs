/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    r#"
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def method(self) -> None:
        pass

class Child(Base):
    # Child doesn't implement method, so it's still abstract
    pass

# This should error - child class is still abstract
child = Child()  # E: Cannot instantiate `Child`
"#,
);

testcase!(
    bug = "We should consider erroring on the class definition too",
    test_final_class_with_abstract_methods,
    r#"
from typing import final
from abc import ABC, abstractmethod

@final
class BadClass(ABC):
    @abstractmethod
    def method(self) -> None:
        pass

x = BadClass()  # E: Cannot instantiate `BadClass`
"#,
);

testcase!(
    test_partial_implementation,
    r#"
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def method1(self) -> None:
        pass

    @abstractmethod
    def method2(self) -> None:
        pass

class Partial(Base):
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
    r#"
from typing import *
from abc import ABC, abstractmethod
class Base(ABC):
    def __init__(self) -> None: pass

    @property
    @abstractmethod
    def processor(self) -> bool: pass

class Child(Base):
    def __init__(self) -> None:
        super().__init__()

x = Child()  # E: Cannot instantiate `Child`
"#,
);
