/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    bug = "T is only used in covariant positions so it should be inferred as covariant",
    test_covariance_inference,
    r#"
from typing import Sequence, reveal_type
def id[T](x: Sequence[T]) -> Sequence[T]:
    return x
def test(x: Sequence[int] | Sequence[str]):
    reveal_type(id(x))  # E: revealed type: Sequence[int] | int # E: Argument `Sequence[int] | Sequence[str]` is not assignable to parameter `x` 
"#,
);

testcase!(
    test_covariance_inference_class,
    r#"
from typing import Sequence
class ShouldBeCovariant[T](Sequence[T]):
    pass

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

x : A[float] = b.g(a)
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
