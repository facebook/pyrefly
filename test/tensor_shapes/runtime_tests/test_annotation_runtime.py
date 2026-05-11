# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Runtime behavior of tensor type annotations.

Tests how pyrefly's native tensor syntax (Tensor[N, 3], Tensor[N+1, M]) behaves
at Python runtime. Two independent issues with standard Python typing:

1. torch.Tensor is not natively subscriptable — Tensor[3, 4] fails.
   SOLVED: torch_shapes patches __class_getitem__ on import.

2. PEP 695 TypeVar doesn't support arithmetic — N + 1, N * 2, etc. fail.
   SOLVED: torch_shapes.TypeVar provides arithmetic operators that return self.

This file tests both the remaining problems (PEP 695 TypeVar arithmetic)
and the solutions (torch_shapes patches, torch_shapes.TypeVar, Generic integration).
"""

import unittest
from typing import Generic, TypedDict

import torch
from torch_shapes import Dim, TypeVar, TypeVarTuple


class TestSubscriptRuntime(unittest.TestCase):
    """torch.Tensor subscripting behavior at runtime.

    torch.Tensor is not natively subscriptable, but importing torch_shapes
    patches __class_getitem__ so that Tensor[3, 4] evaluates to Tensor
    (a no-op) instead of crashing.
    """

    def test_concrete_subscript(self):
        """Tensor[3, 4] — works after torch_shapes patches __class_getitem__."""

        def f(x: torch.Tensor[3, 4]) -> torch.Tensor[3, 4]:
            return x

        self.assertTrue(callable(f))

    def test_typevar_subscript(self):
        """Tensor[N, 3] — TypeVar in subscript, works after patch."""

        def f[N](x: torch.Tensor[N, 3]) -> torch.Tensor[N, 3]:
            return x

        self.assertTrue(callable(f))


class TestTypeVarArithmetic(unittest.TestCase):
    """TypeVar doesn't support arithmetic operators."""

    def test_typevar_add(self):
        """N + 1 in an annotation."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: N + 1) -> None:  # type: ignore[valid-type]
                pass

    def test_typevar_mul(self):
        """N * 2 in an annotation."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: N * 2) -> None:  # type: ignore[valid-type]
                pass

    def test_typevar_sub(self):
        """N - 1 in an annotation."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for -: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: N - 1) -> None:  # type: ignore[valid-type]
                pass

    def test_typevar_floordiv(self):
        """N // 2 in an annotation."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for //: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: N // 2) -> None:  # type: ignore[valid-type]
                pass

    def test_two_typevars_add(self):
        """N + M in an annotation."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'typing.TypeVar'",
        ):

            def f[N, M](x: N + M) -> None:  # type: ignore[valid-type]
                pass


class TestCombined(unittest.TestCase):
    """TypeVar arithmetic inside Tensor subscript."""

    def test_tensor_typevar_arithmetic(self):
        """Tensor[N+1, 3] — arithmetic is evaluated before subscript, so
        the error is 'unsupported operand' not 'not subscriptable'."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: torch.Tensor[N + 1, 3]) -> torch.Tensor[N, 3]:
                return x


class TestClassAnnotationRuntime(unittest.TestCase):
    """Classes with class-level and method-level TypeVars."""

    def test_class_concrete_subscript(self):
        """Class method with Tensor[3, 4] — works after torch_shapes patch."""

        class Layer:
            def forward(self, x: torch.Tensor[3, 4]) -> torch.Tensor[3, 4]:
                return x

        self.assertTrue(hasattr(Layer, "forward"))

    def test_class_typevars_no_arithmetic(self):
        """Class-level (N, M) and method-level (B) TypeVars — works after patch."""

        class Layer[N, M]:
            def forward[B](self, x: torch.Tensor[B, N]) -> torch.Tensor[B, M]:
                return x  # type: ignore[return-value]

        self.assertTrue(hasattr(Layer, "forward"))

    def test_class_typevar_arithmetic(self):
        """Class-level TypeVar with arithmetic in method annotation.
        Tensor[N, 3] works after patch, but N+1 still crashes because
        PEP 695 TypeVar doesn't support arithmetic."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            class PadLayer[N]:
                def forward(self, x: torch.Tensor[N, 3]) -> torch.Tensor[N + 1, 3]:
                    return x  # type: ignore[return-value]


class TestDimRuntime(unittest.TestCase):
    """Dim[...] behavior at runtime.

    Dim natively supports __class_getitem__, so Dim[3] and Dim[N] work.
    But Dim[N+1] crashes because Python evaluates N+1 before passing it
    to __class_getitem__, and PEP 695 TypeVar doesn't support arithmetic."""

    def test_dim_concrete(self):
        """Dim[3] — concrete integer, works fine."""

        def f(x: Dim[3]) -> Dim[3]:
            return x

        f(42)

    def test_dim_typevar(self):
        """Dim[N] — bare TypeVar, works fine."""

        def f[N](x: Dim[N]) -> Dim[N]:
            return x

        f(42)

    def test_dim_arithmetic(self):
        """Dim[N+1] — TypeVar arithmetic crashes."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: Dim[N]) -> Dim[N + 1]:
                return x

    def test_dim_two_typevars(self):
        """Dim[N+M] — two TypeVars in arithmetic crashes."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'typing.TypeVar'",
        ):

            def f[N, M](x: Dim[N]) -> Dim[N + M]:
                return x


class TestTypeVarRuntime(unittest.TestCase):
    """torch_shapes.TypeVar provides arithmetic operators that don't crash."""

    def test_add(self):
        """N + 1 doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = N + 1
        self.assertIsNotNone(result)

    def test_radd(self):
        """1 + N doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = 1 + N
        self.assertIsNotNone(result)

    def test_sub(self):
        """N - 1 doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = N - 1
        self.assertIsNotNone(result)

    def test_rsub(self):
        """1 - N doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = 1 - N
        self.assertIsNotNone(result)

    def test_mul(self):
        """N * 2 doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = N * 2
        self.assertIsNotNone(result)

    def test_rmul(self):
        """2 * N doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = 2 * N
        self.assertIsNotNone(result)

    def test_floordiv(self):
        """N // 2 doesn't crash with torch_shapes.TypeVar."""
        N = TypeVar("N")
        result = N // 2
        self.assertIsNotNone(result)

    def test_chained(self):
        """(N + 1) * 2 doesn't crash — chained arithmetic."""
        N = TypeVar("N")
        result = (N + 1) * 2
        self.assertIsNotNone(result)

    def test_two_vars(self):
        """N + M doesn't crash with two torch_shapes.TypeVars."""
        N = TypeVar("N")
        M = TypeVar("M")
        result = N + M
        self.assertIsNotNone(result)

    def test_repr(self):
        """torch_shapes.TypeVar repr shows the name."""
        N = TypeVar("N")
        self.assertEqual(repr(N), "N")

    def test_in_dim(self):
        """Dim[N] with torch_shapes.TypeVar."""
        N = TypeVar("N")

        def f(x: Dim[N]) -> Dim[N]:
            return x

        f(42)

    def test_arithmetic_in_dim(self):
        """Dim[N+1] with torch_shapes.TypeVar — N+1 returns self, which Dim accepts."""
        N = TypeVar("N")

        def f(x: Dim[N]) -> Dim[N + 1]:
            return x

        f(42)


class TestGenericRuntime(unittest.TestCase):
    """Generic works with torch_shapes.TypeVar at runtime thanks to __class__ = typing.TypeVar."""

    def test_generic_subscript(self):
        """Generic[N, M] works with torch_shapes.TypeVar."""
        N = TypeVar("N")
        M = TypeVar("M")

        class Foo(Generic[N, M]):
            pass

        self.assertTrue(issubclass(Foo, object))

    def test_generic_as_base_class(self):
        """class Foo(Generic[N, M]) with method annotations works."""
        N = TypeVar("N")
        M = TypeVar("M")

        class Foo(Generic[N, M]):
            def forward(self, x: Dim[N]) -> Dim[M]:
                return x

        foo = Foo()
        result = foo.forward(42)
        self.assertEqual(result, 42)

    def test_generic_accepts_intvar(self):
        """Generic[N] works when N is torch_shapes.TypeVar — it sets
        __class__ = typing.TypeVar so isinstance(N, typing.TypeVar) returns True."""
        N = TypeVar("N")

        class Layer(Generic[N]):
            def forward(self, x: Dim[N]) -> Dim[N + 1]:
                return x

        layer = Layer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_generic_with_dim_arithmetic(self):
        """Generic[N] with Dim[N+1] in a method works."""
        N = TypeVar("N")

        class PadLayer(Generic[N]):
            def forward(self, x: Dim[N]) -> Dim[N + 1]:
                return x

        layer = PadLayer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_typeddict_generic_intvar(self):
        """TypedDict + Generic[N] works with torch_shapes.TypeVar."""
        N = TypeVar("N")
        M = TypeVar("M")

        class MyDict(TypedDict, Generic[N, M]):
            x: Dim[N]
            y: Dim[M]

        self.assertTrue(issubclass(MyDict, dict))


class TestTypeVarTupleRuntime(unittest.TestCase):
    """torch_shapes.TypeVarTuple supports star-unpacking at runtime."""

    def test_iter(self):
        """*Ns unpacking works — __iter__ yields self."""
        Ns = TypeVarTuple("Ns")
        items = list(Ns)
        self.assertEqual(len(items), 1)
        self.assertIs(items[0], Ns)

    def test_in_dim(self):
        """Dim[*Ns] — star-unpacking in subscript works."""
        Ns = TypeVarTuple("Ns")

        def f(x: Dim[*Ns]) -> Dim[*Ns]:
            return x

        f(42)

    def test_generic(self):
        """Generic[*Ns] — variadic class generic works."""
        Ns = TypeVarTuple("Ns")

        class Layer(Generic[*Ns]):
            def forward(self, x: Dim[*Ns]) -> Dim[*Ns]:
                return x

        layer = Layer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_mixed_with_typevar(self):
        """Generic[*Ns, N] — variadic + fixed dim works."""
        Ns = TypeVarTuple("Ns")
        N = TypeVar("N")

        class Layer(Generic[*Ns, N]):
            def forward(self, x: Dim[*Ns]) -> Dim[N + 1]:
                return x

        layer = Layer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_repr(self):
        """torch_shapes.TypeVarTuple repr shows *name."""
        Ns = TypeVarTuple("Ns")
        self.assertEqual(repr(Ns), "*Ns")


if __name__ == "__main__":
    unittest.main()
