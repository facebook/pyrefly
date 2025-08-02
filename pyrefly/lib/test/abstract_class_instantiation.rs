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

# This should error - cannot instantiate abstract class
shape = Shape()  # E: Cannot instantiate abstract class `Shape`
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
drawable = Drawable()  # E: Cannot instantiate abstract class `Drawable`
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
child = Child()  # E: Cannot instantiate abstract class `Child`
"#,
);
