/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_abstract_method_direct_call,
    r#"
import abc

class AbstractClass:
    @abc.abstractmethod
    def abstract_method(self) -> str:
        pass

obj = AbstractClass()
obj.abstract_method()  # E: Cannot call abstract method `AbstractClass.abstract_method` - must be implemented in a subclass
    "#,
);

testcase!(
    test_abstract_method_concrete_implementation,
    r#"
import abc

class AbstractClass:
    @abc.abstractmethod
    def abstract_method(self) -> str:
        pass

class ConcreteClass(AbstractClass):
    def abstract_method(self) -> str:
        return "implemented"

obj = ConcreteClass()
obj.abstract_method()  # No error - concrete implementation
    "#,
);

testcase!(
    test_abstract_static_method_call,
    r#"
import abc

class AbstractClass:
    @abc.abstractmethod
    @staticmethod
    def abstract_static_method() -> str:
        pass

AbstractClass.abstract_static_method()  # E: Cannot call abstract method `AbstractClass.abstract_static_method` - must be implemented in a subclass
    "#,
);

testcase!(
    test_abstract_class_method_call,
    r#"
import abc

class AbstractClass:
    @abc.abstractmethod
    @classmethod
    def abstract_class_method(cls) -> str:
        pass

AbstractClass.abstract_class_method()  # E: Cannot call abstract method `AbstractClass.abstract_class_method` - must be implemented in a subclass
    "#,
);

testcase!(
    test_abstract_method_super_call,
    r#"
import abc

class AbstractClass:
    @abc.abstractmethod
    def abstract_method(self) -> str:
        pass

class ChildClass(AbstractClass):
    def abstract_method(self) -> str:
        return super().abstract_method()  # E: Cannot call abstract method `AbstractClass.abstract_method` - must be implemented in a subclass

obj = ChildClass()
obj.abstract_method()
    "#,
);

testcase!(
    test_abstract_method_multiple_inheritance,
    r#"
import abc

class AbstractBase1:
    @abc.abstractmethod
    def method1(self) -> str:
        pass

class AbstractBase2:
    @abc.abstractmethod
    def method2(self) -> str:
        pass

class ConcreteChild(AbstractBase1, AbstractBase2):
    def method1(self) -> str:
        return "implemented1"
    
    def method2(self) -> str:
        return "implemented2"

obj = ConcreteChild()
obj.method1()  # No error - concrete implementation
obj.method2()  # No error - concrete implementation
    "#,
);

testcase!(
    test_abstract_method_partial_implementation,
    r#"
import abc

class AbstractBase:
    @abc.abstractmethod
    def method1(self) -> str:
        pass
    
    @abc.abstractmethod
    def method2(self) -> str:
        pass

class PartialChild(AbstractBase):
    def method1(self) -> str:
        return "implemented1"
    
    def call_abstract_method(self) -> str:
        return self.method2()  # E: Cannot call abstract method `AbstractBase.method2` - must be implemented in a subclass

obj = PartialChild()
obj.method1()  # No error - concrete implementation
obj.method2()  # E: Cannot call abstract method `AbstractBase.method2` - must be implemented in a subclass
    "#,
);

testcase!(
    test_abstract_method_protocol,
    r#"
import abc
from typing import Protocol

class AbstractProtocol(Protocol):
    @abc.abstractmethod
    def abstract_method(self) -> str:
        pass

class ConcreteImpl:
    def abstract_method(self) -> str:
        return "implemented"

def use_protocol(obj: AbstractProtocol) -> str:
    return obj.abstract_method()  # E: Cannot call abstract method `AbstractProtocol.abstract_method` - must be implemented in a subclass

impl = ConcreteImpl()
use_protocol(impl)
    "#,
);

testcase!(
    test_abstract_method_decorator_order,
    r#"
import abc

class AbstractClass:
    @staticmethod
    @abc.abstractmethod
    def abstract_static_method() -> str:
        pass
    
    @classmethod
    @abc.abstractmethod
    def abstract_class_method(cls) -> str:
        pass

AbstractClass.abstract_static_method()  # E: Cannot call abstract method `AbstractClass.abstract_static_method` - must be implemented in a subclass
AbstractClass.abstract_class_method()   # E: Cannot call abstract method `AbstractClass.abstract_class_method` - must be implemented in a subclass
    "#,
);

testcase!(
    test_abstract_method_mixed_decorators,
    r#"
import abc
from typing import override

class AbstractClass:
    @abc.abstractmethod
    def abstract_method(self) -> str:
        pass

class ChildClass(AbstractClass):
    @override
    def abstract_method(self) -> str:
        return "implemented"

obj = ChildClass()
obj.abstract_method()  # No error - concrete implementation with override
    "#,
);
