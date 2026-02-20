/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_specified_variance_gets_respected,
    r#"
from typing import Generic, TypeVar

T = TypeVar("T", contravariant=True)

# Intentionally set up 2 type variables:
# - U needs to has its variance inferred (to be covariant)
# - T has its variance specified incorrectly -- but downstream logic is expected to respect it.
class Foo[U](Generic[T]):  # E: Type parameter T is not included in the type parameter list
    def m0(self) -> T: ...  # E: Type variable `T` is Contravariant but is used in covariant position
    def m1(self) -> U: ...

t_good: Foo[int, int] = Foo[int, float]()
t_bad: Foo[int, float] = Foo[int, int]()  # E:

u_good: Foo[float, int] = Foo[int, int]()
u_bad: Foo[int, int] = Foo[float, int]()  # E:
"#,
);

testcase!(
    test_covariance_inference_class,
    r#"
from typing import Sequence, Any
class ShouldBeCovariant[T](Sequence[T]):
    def __getitem__(self, *args, **kwargs) -> Any: ...
    def __len__(self) -> int: ...

vco2_1: ShouldBeCovariant[float] = ShouldBeCovariant[int]()
vco2_2: ShouldBeCovariant[int] = ShouldBeCovariant[float]()  # E:
"#,
);

testcase!(
    bug = "T2 and T3 should be resolved when we traverse methods. They will be bivariant until then. For T1, we raise an error because we already know it's invariant in list.",
    test_general_variance,
    r#"

class ClassA[T1, T2, T3](list[T1]):
    def method1(self, a: T2) -> None:
        ...

    def method2(self) -> T3:
            ...

def func_a(p1: ClassA[float, int, int], p2: ClassA[int, float, float]):
    v1: ClassA[int, int, int] = p1  # E:
    v2: ClassA[float, float, int] = p1 # E:
    v3: ClassA[float, int, float] = p1

    v4: ClassA[int, int, int] = p2 # E:
    v5: ClassA[int, int, float] = p2
"#,
);

testcase!(
    test_bivariant,
    r#"
class A[T]:
    def f(self, x: B[T]) -> B[T]:
        return x

class B[U]:
    def g(self, x: A[U]) -> A[U]:
        return x

a = A[int]()
b = B[int]()

# We follow mypy and pyright's lead in treating bivariant type parameters as invariant.
x: A[float] = b.g(a)  # E: `A[int]` is not assignable to `A[float]`
"#,
);

testcase!(
    test_invariant_callable,
    r#"
from typing import Callable

class ShouldBeInvariant[T]:

    def f (self, x: Callable[[T], T]):
        return x

square: Callable[[int], int] = lambda x: x ** 2

a: Callable[[int], int] = ShouldBeInvariant[int]().f(square)
b: Callable[[float], int]= ShouldBeInvariant[float]().f(square)  # E: # E:
"#,
);

testcase!(
    test_invariant_dict,
    r#"
class ShouldBeInvariant[K, V](dict[K, V]):
    pass

vinv3_1: ShouldBeInvariant[float, str] = ShouldBeInvariant[int, str]()  # E:
"#,
);

testcase!(
    test_infer_variance,
    r#"
from typing import Sequence


class ShouldBeCovariant2[T](Sequence[T]):
    pass

class ShouldBeCovariant3[U]:
    def method(self) -> ShouldBeCovariant2[U]:
        ...

vco3_1: ShouldBeCovariant3[float] = ShouldBeCovariant3[int]()  # OK
vco3_2: ShouldBeCovariant3[int] = ShouldBeCovariant3[float]()  # E:

"#,
);

testcase!(
    test_attrs,
    r#"
class ShouldBeInvariant5[T]:
    def __init__(self, x: T) -> None:
        self.x = x

vinv5_1: ShouldBeInvariant5[float] = ShouldBeInvariant5[int](1)  # E:

"#,
);

testcase!(
    test_attrs_set_and_get,
    r#"
class ShouldBeCovariant1[T]:
    def __getitem__(self, index: int) -> T:
        ...

vco1_1: ShouldBeCovariant1[float] = ShouldBeCovariant1[int]()  # OK
vco1_2: ShouldBeCovariant1[int] = ShouldBeCovariant1[float]()  # E:


class ShouldBeContravariant2[T]:
    def __init__(self, value: T) -> None:
        pass

    def set_value(self, value: T):
        pass


vcontra1_1: ShouldBeContravariant2[float] = ShouldBeContravariant2[int](1)  # E:
vcontra1_2: ShouldBeContravariant2[int] = ShouldBeContravariant2[float](1.2)  # OK


"#,
);

testcase!(
    test_infer_variance_and_private_field,
    r#"
from typing import Generic, TypeVar, Iterator

T = TypeVar("T", infer_variance=True)


class ShouldBeCovariant1(Generic[T]):
    def __getitem__(self, index: int) -> T:
        ...

    def __iter__(self) -> Iterator[T]:
        ...


vco1_1: ShouldBeCovariant1[float] = ShouldBeCovariant1[int]()  # OK
vco1_2: ShouldBeCovariant1[int] = ShouldBeCovariant1[float]()  # E:



K = TypeVar("K", infer_variance=True)


class ShouldBeCovariant5(Generic[K]):
    def __init__(self, x: K) -> None:
        self._x = x

    def x(self) -> K:
        return self._x

vo5_1: ShouldBeCovariant5[float] = ShouldBeCovariant5[int](1)  # OK
vo5_2: ShouldBeCovariant5[int] = ShouldBeCovariant5[float](1.0)  # E:

# we are making sure we don't treat __dunder__ attributes as private.
class ShouldBeInvariant6(Generic[K]):
    def __init__(self, x: K) -> None:
        self.__x__ = x

    def x(self) -> K:
        return self.__x__


vo6_1: ShouldBeInvariant6[float] = ShouldBeInvariant6[int](1)  # E:
vo6_2: ShouldBeInvariant6[int] = ShouldBeInvariant6[float](1.0)  # E:

"#,
);

testcase!(
    test_private_field,
    r#"
class ShouldBeCovariant5[K]:
    def __init__(self, x: K) -> None:
        self._x = x

    def x(self) -> K:
        return self._x


vo5_1: ShouldBeCovariant5[float] = ShouldBeCovariant5[int](1)  # OK
vo5_2: ShouldBeCovariant5[int] = ShouldBeCovariant5[float](1.0)  # E:

"#,
);

testcase!(
    test_dataclass_frozen_variance,
    r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class ShouldBeCovariant4[T]:
    x: T


vo4_1: ShouldBeCovariant4[float] = ShouldBeCovariant4[int](1)  # OK
vo4_4: ShouldBeCovariant4[int] = ShouldBeCovariant4[float](1.0)  # E:
"#,
);

testcase!(
    test_property,
    r#"
from typing import *
class ShouldBeInvariant1[K]:
    def __init__(self, value: K) -> None:
        self._value = value

    @property
    def value(self) -> K:
        return self._value

    @value.setter
    def value(self, value: K) -> None:
        self._value = value

vinv1_1: ShouldBeInvariant1[float] = ShouldBeInvariant1[int](1)  # E:
vinv1_2: ShouldBeInvariant1[int] = ShouldBeInvariant1[float](1.1)  # E:
"#,
);

testcase!(
    bug = "For properties with both a getter and setter, the stored type is the setter function. This is causing us to miss visiting the getter.",
    test_protocol_property_invariant,
    r#"
from typing import Protocol, TypeVar

TypeT = TypeVar("TypeT")

class HasP(Protocol[TypeT]): # E: Type variable `TypeT` in class `HasP` is declared as invariant, but could be contravariant based on its usage
    @property
    def p(self) -> TypeT: ...
    @p.setter
    def p(self, p: TypeT, /) -> None: ...
"#,
);

testcase!(
    test_sequence_inheritance,
    r#"
from typing import Sequence

class A[T](B[Sequence[T]]):
    ...

class B[T]:
    def f(self, x:T) -> T:
        return x

b = B[int]()

y = b.f(3)
z = b.f(3.0) # E:
"#,
);

testcase!(
    test_self_referential_covariance,
    r#"
class FooInferred[Node]:
    def __init__(self, *options: FooInferred[Node]) -> None: ...
    def __or__[OtherNode](self, other: FooInferred[OtherNode]) -> FooInferred[Node | OtherNode]:
        # Node should be inferred as covariant since it only appears in covariant positions
        # (the return type of __or__, and __init__ is skipped for variance inference)
        return FooInferred[Node | OtherNode](self, other)

# If FooInferred is covariant, this should work:
# FooInferred[int] <: FooInferred[int | str] because int <: int | str
foo_int: FooInferred[int] = FooInferred[int]()
foo_str: FooInferred[str] = FooInferred[str]()
foo_union: FooInferred[int | str] = foo_int | foo_str
"#,
);

// Test variance inference with stdlib generic that has covariant type parameter
testcase!(
    test_class_variance_with_mapping,
    r#"
from collections.abc import Mapping

class Container[T]:
    def get(self) -> Mapping[str, T]:
        ...

def widen(c: Container[int]) -> Container[float]:
    return c  # OK - T is covariant since Mapping's V type is covariant
"#,
);

testcase!(
    test_variance_enforcement_in_base_classes,
    r#"
from typing import TypeVar, Generic

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Co(Generic[T_co]): ...
class Contra(Generic[T_contra]): ...
class Inv(Generic[T]): ...
class CoContra(Generic[T_co, T_contra]): ...

class Class1(
    Inv[T_co]  # E: Type variable `T_co` is Covariant but is used in invariant position
): ...
class Class2(
    Inv[T_contra]  # E: Type variable `T_contra` is Contravariant but is used in invariant position
): ...

class Co_Child3(
    Co[T_contra]  # E: Type variable `T_contra` is Contravariant but is used in covariant position
): ...
class Contra_Child3(
    Contra[T_co]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...
class Contra_Child5(
    Contra[Co[T_co]]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...

class CoContra_Child2(
    CoContra[T_co, T_co]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...
class CoContra_Child3(
    CoContra[T_contra, T_contra]  # E: Type variable `T_contra` is Contravariant but is used in covariant position
): ...
class CoContra_Child5(
    CoContra[Co[T_co], Co[T_co]]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...

class CoToContraToContra(
    Contra[Co[Contra[T_contra]]]  # E: Type variable `T_contra` is Contravariant but is used in covariant position
): ...
class ContraToContraToContra(
    Contra[Contra[Contra[T_co]]]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...

Co_TA = Co[T_co]
Contra_TA = Contra[T_contra]

class CoToContraToContra_WithTA(
    Contra_TA[Co_TA[Contra_TA[T_contra]]]  # E: Type variable `T_contra` is Contravariant but is used in covariant position
): ...
class ContraToContraToContra_WithTA(
    Contra_TA[Contra_TA[Contra_TA[T_co]]]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...
"#,
);

testcase!(
    test_protocols_variance_conformance,
    r#"
from typing import Protocol, TypeVar

T1 = TypeVar("T1")
T1_co = TypeVar("T1_co", covariant=True)
T1_contra = TypeVar("T1_contra", contravariant=True)

class AnotherBox(Protocol[T1]):  # E: Type variable `T1` in class `AnotherBox` is declared as invariant, but could be covariant based on its usage
    def content(self) -> T1: ...

class Protocol4(Protocol[T1]):  # E: Type variable `T1` in class `Protocol4` is declared as invariant, but could be contravariant based on its usage
    def m1(self, p0: T1) -> None: ...

class Protocol5(Protocol[T1_co]):
    def m1(self, p0: T1_co) -> None: ...  # E: Type variable `T1_co` is Covariant but is used in contravariant position

class Protocol6(Protocol[T1]):  # E: Type variable `T1` in class `Protocol6` is declared as invariant, but could be covariant based on its usage
    def m1(self) -> T1: ...

class Protocol7(Protocol[T1_contra]):
    def m1(self) -> T1_contra: ...  # E: Type variable `T1_contra` is Contravariant but is used in covariant position

class Protocol12(Protocol[T1]):  # E: Type variable `T1` in class `Protocol12` is declared as invariant, but could be covariant based on its usage
    def __init__(self, x: T1) -> None: ...
"#,
);

testcase!(
    test_shallow_covariant_in_param,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)

class Foo(Generic[T_co]):
    def f(self, x: T_co) -> None: ...  # E: Type variable `T_co` is Covariant but is used in contravariant position
"#,
);

testcase!(
    test_shallow_contravariant_in_return,
    r#"
from typing import TypeVar, Generic
T_contra = TypeVar("T_contra", contravariant=True)

class Foo(Generic[T_contra]):
    def f(self) -> T_contra: ...  # E: Type variable `T_contra` is Contravariant but is used in covariant position
"#,
);

// Deep check: we should NOT raise an error here
testcase!(
    test_deep_covariant_in_contra_return,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Contra(Generic[T_contra]): ...

class Foo(Generic[T_co]):  
    def f(self) -> Contra[T_co]: ...
"#,
);

// Deep check: we should NOT raise an error here
testcase!(
    test_deep_covariant_in_co_param,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)

class Co(Generic[T_co]): ...

class Foo(Generic[T_co]):  
    def f(self, x: Co[T_co]) -> None: ...
"#,
);

// Deep check: we should NOT raise an error here
testcase!(
    test_deep_callable_param_in_return,
    r#"
from typing import TypeVar, Generic, Callable
T_co = TypeVar("T_co", covariant=True)

class Foo(Generic[T_co]):  
    def f(self) -> Callable[[T_co], None]: ...
"#,
);

// Deep check: we should NOT raise an error here
testcase!(
    test_deep_callable_return_in_param,
    r#"
from typing import TypeVar, Generic, Callable
T_co = TypeVar("T_co", covariant=True)

class Foo(Generic[T_co]):  
    def f(self, x: Callable[[], T_co]) -> None: ...
"#,
);

// Deep check: we should NOT raise an error here
testcase!(
    test_deep_double_callable,
    r#"
from typing import TypeVar, Generic, Callable
T_contra = TypeVar("T_contra", contravariant=True)

class Foo(Generic[T_contra]):
    def f(self) -> Callable[[Callable[[T_contra], None]], None]: ...
"#,
);

// We skip checking fields
testcase!(
    test_field_covariant_in_mutable,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)

class Foo(Generic[T_co]):
    x: T_co 
"#,
);

// We skip checking fields
testcase!(
    test_field_contravariant_in_mutable,
    r#"
from typing import TypeVar, Generic
T_contra = TypeVar("T_contra", contravariant=True)

class Foo(Generic[T_contra]):
    x: T_contra  
"#,
);

testcase!(
    test_base_covariant_in_invariant,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

class Inv(Generic[T]): ...

class Foo(
    Inv[T_co]  # E: Type variable `T_co` is Covariant but is used in invariant position
): ...
"#,
);

testcase!(
    test_base_contravariant_in_invariant,
    r#"
from typing import TypeVar, Generic
T_contra = TypeVar("T_contra", contravariant=True)
T = TypeVar("T")

class Inv(Generic[T]): ...

class Foo(
    Inv[T_contra]  # E: Type variable `T_contra` is Contravariant but is used in invariant position
): ...
"#,
);

testcase!(
    test_base_covariant_in_contravariant,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Contra(Generic[T_contra]): ...

class Foo(
    Contra[T_co]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...
"#,
);

testcase!(
    test_base_contravariant_in_covariant,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Co(Generic[T_co]): ...

class Foo(
    Co[T_contra]  # E: Type variable `T_contra` is Contravariant but is used in covariant position
): ...
"#,
);

testcase!(
    test_base_nested_double,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Co(Generic[T_co]): ...
class Contra(Generic[T_contra]): ...

# pyright errors and mypy does not
class Foo(
    Contra[Co[T_co]]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...
"#,
);

testcase!(
    test_base_nested_triple_error,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Contra(Generic[T_contra]): ...

# pyright errors and mypy does not
class Foo(
    Contra[Contra[Contra[T_co]]]  # E: Type variable `T_co` is Covariant but is used in contravariant position
): ...
"#,
);

testcase!(
    test_base_nested_triple_ok,
    r#"
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Co(Generic[T_co]): ...
class Contra(Generic[T_contra]): ...

# contra * co * contra = co, so T_co in covariant position - OK
class Foo(Contra[Co[Contra[T_co]]]): ...
"#,
);
