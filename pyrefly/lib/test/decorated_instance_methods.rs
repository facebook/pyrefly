/// Test decorated instance methods from stubs (e.g., PySpark @dispatch_col_method)
/// This test verifies that decorated methods that become Type::Callable in stubs
/// are properly recognized as bound methods when called on instances.
/// See: https://github.com/facebook/pyrefly/issues/3465

use crate::test_helpers::check;

#[test]
fn test_pyspark_decorated_column_methods() {
    // Simulates PySpark Column methods like isNull(), asc(), etc.
    // When loaded from stubs, these may be decorated with @dispatch_col_method
    // which can cause them to be represented as Callable types instead of Function types.
    // The fix ensures that such Callable types are still recognized as bound methods.
    check(
        r#"
class Column:
    def isNull(self) -> 'Column': ...
    def isNotNull(self) -> 'Column': ...
    def asc(self) -> 'Column': ...
    def desc(self) -> 'Column': ...
    def eqNullSafe(self, other: 'Column') -> 'Column': ...

def col(x: str) -> Column: ...

# These should all be valid - no bad-argument-count errors
col("x").isNull()
col("x").isNotNull()
col("x").asc()
col("x").desc()
col("x").eqNullSafe(col("y"))
        "#,
        &[],
    );
}

#[test]
fn test_regular_instance_methods_still_work() {
    // Ensure that regular instance methods (not from stubs) still work correctly
    check(
        r#"
class MyClass:
    def method1(self) -> int: ...
    def method2(self, x: int) -> str: ...
    def method3(self, x: int, y: str) -> bool: ...

obj = MyClass()
obj.method1()
obj.method2(42)
obj.method3(42, "hello")
        "#,
        &[],
    );
}

#[test]
fn test_callable_class_attributes_not_bound() {
    // Ensure that Callable class attributes that are not methods are NOT bound
    // (e.g., callbacks stored as class variables)
    check(
        r#"
class MyClass:
    handler: Callable[[int], None]  # This should NOT be treated as a bound method

obj = MyClass()
# Calling handler should not bind self - we need to pass obj as first arg if it's a Callable
obj.handler(42)  # This should check if handler is callable with int arg
        "#,
        &[],
    );
}

#[test]
fn test_union_with_decorated_method() {
    // Test that methods in unions are handled correctly
    check(
        r#"
from typing import Union

class ColumnA:
    def isNull(self) -> 'ColumnA': ...

class ColumnB:
    def isNull(self) -> 'ColumnB': ...

def col() -> Union[ColumnA, ColumnB]: ...

result = col().isNull()
        "#,
        &[],
    );
}

#[test]
fn test_factory_callable_as_class_method() {
    // Test the PySpark 3.4+ factory pattern where Callable types from factory
    // functions are assigned as class attributes and should bind self at runtime
    check(
        r#"
from typing import Callable, Union

def _unary_op(name: str) -> Callable[["Column"], "Column"]: ...
def _bin_op(name: str) -> Callable[["Column", Union["Column", int]], "Column"]: ...

class Column:
    isNull = _unary_op("isNull")
    asc = _unary_op("asc")
    eqNullSafe = _bin_op("eqNullSafe")

def col(x: str) -> Column: ...

# At runtime these work because the returned functions are descriptors.
# Pyrefly should now recognize them as methods.
col("x").isNull()  # No error
col("x").asc()  # No error
col("x").eqNullSafe(col("y"))  # No error
        "#,
        &[],
    );
}

#[test]
fn test_callable_attribute_without_matching_class_not_method() {
    // Verify that Callable attributes with parameters that DON'T match the
    // containing class are NOT treated as methods. This prevents false positives.
    check(
        r#"
from typing import Callable

class Config:
    # Callable with different parameter type - should NOT bind self
    validator: Callable[[str], bool] = lambda x: len(x) > 0

cfg = Config()
# Should work: calling validator(string) without passing self
result: bool = cfg.validator("test")
        "#,
        &[],
    );
}

#[test]
fn test_strategy_pattern_not_method() {
    // Strategy pattern with Callable - should NOT be treated as a method
    check(
        r#"
from typing import Callable

class Calculator:
    # Strategy pattern - Callable but not a method
    operation: Callable[[int, int], int] = lambda a, b: a + b

calc = Calculator()
# This should work: calling operation(int, int) without self binding
result: int = calc.operation(5, 3)
        "#,
        &[],
    );
}

#[test]
fn test_callable_with_unrelated_type_not_method() {
    // Callable with parameters of completely different type - should NOT be method
    check(
        r#"
from typing import Callable

class Handler:
    # This is a callback handler - not a method
    callback: Callable[[str], None] = lambda x: print(x)

h = Handler()
# Should work without self binding
h.callback("message")
        "#,
        &[],
    );
}
