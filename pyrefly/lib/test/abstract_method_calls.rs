/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_abstract_method_call_error,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

shape = Shape()
shape.area()  # E: Cannot call abstract method `Shape.area` - must be implemented in a subclass
"#,
);

testcase!(
    test_abstract_method_call_direct,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

Shape.area(Shape())  # E: Cannot call abstract method `Shape.area` - must be implemented in a subclass
"#,
);

testcase!(
    test_abstract_static_method_call,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @staticmethod
    @abstractmethod
    def create() -> "Shape":
        pass

Shape.create()  # E: Cannot call abstract method `Shape.create` - must be implemented in a subclass
"#,
);

testcase!(
    test_abstract_class_method_call,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @classmethod
    @abstractmethod
    def default_shape(cls) -> "Shape":
        pass

Shape.default_shape()  # E: Cannot call abstract method `Shape.default_shape` - must be implemented in a subclass
"#,
);

testcase!(
    test_abstract_method_call_super,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def area(self) -> float:
        return super().area()  # E: Cannot call abstract method `Shape.area` - must be implemented in a subclass

circle = Circle()
"#,
);

testcase!(
    test_concrete_method_call_ok,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def area(self) -> float:
        return 3.14

circle = Circle()
circle.area()  # This should be fine
"#,
);

testcase!(
    test_multiple_abstract_methods,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass

shape = Shape()
shape.area()  # E: Cannot call abstract method `Shape.area` - must be implemented in a subclass
shape.perimeter()  # E: Cannot call abstract method `Shape.perimeter` - must be implemented in a subclass
"#,
);

testcase!(
    test_abstract_property,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

shape = Shape()
_ = shape.name  # E: Cannot call abstract method `Shape.name` - must be implemented in a subclass
"#,
);

testcase!(
    test_inheritance_chain,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Polygon(Shape):
    # Still abstract, doesn't implement area
    pass

polygon = Polygon()
polygon.area()  # E: Cannot call abstract method `Shape.area` - must be implemented in a subclass
"#,
);
