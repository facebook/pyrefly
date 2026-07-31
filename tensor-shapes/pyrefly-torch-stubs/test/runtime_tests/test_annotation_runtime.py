# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Runtime behavior of tensor type annotations.

Tests how pyrefly's native tensor syntax (Tensor[[N, 3]], Tensor[[N+1, M]]) behaves
at Python runtime. Two independent issues with standard Python typing:

1. torch.Tensor is not natively subscriptable — Tensor[[3, 4]] fails.
   SOLVED: shape_extensions patches __class_getitem__ on import.

2. PEP 695 TypeVar doesn't support arithmetic — N + 1, N * 2, etc. fail.
   SOLVED: shape_extensions.IntVar provides arithmetic operators that return self.

This file tests both the remaining problems (PEP 695 TypeVar arithmetic)
and the solutions (shape_extensions patches, shape_extensions.IntVar, Generic integration).
"""

import unittest
from typing import Generic, TypedDict

import torch
from shape_extensions import (
    assert_shape,
    D,
    defines_assert_shape,
    enable_torchscript_runtime_compat,
    Int,
    IntTuple,
    IntVar,
    TypeVarTuple,
)


class TestSubscriptRuntime(unittest.TestCase):
    """torch.Tensor subscripting behavior at runtime.

    torch.Tensor is not natively subscriptable, but importing shape_extensions
    patches __class_getitem__ so that Tensor[[3, 4]] evaluates to Tensor
    (a no-op) instead of crashing.
    """

    def test_concrete_subscript(self):
        """Tensor[[3, 4]] — works after shape_extensions patches __class_getitem__."""

        def f(x: torch.Tensor[[3, 4]]) -> torch.Tensor[[3, 4]]:
            return x

        self.assertTrue(callable(f))

    def test_tensor_subscript_erases_to_tensor(self):
        self.assertIs(torch.Tensor[[3, 4]], torch.Tensor)

    def test_typevar_subscript(self):
        """Tensor[[N, 3]] — TypeVar in subscript, works after patch."""

        def f[N](x: torch.Tensor[[N, 3]]) -> torch.Tensor[[N, 3]]:
            return x

        self.assertTrue(callable(f))


class TestTorchScriptRuntimeCompat(unittest.TestCase):
    """TorchScript compatibility mode erases shape-only runtime annotations."""

    def setUp(self):
        super().setUp()
        original_class_getitem = Int.__dict__.get("__class_getitem__")

        def restore_int_class_getitem():
            if original_class_getitem is None:
                if "__class_getitem__" in Int.__dict__:
                    delattr(Int, "__class_getitem__")
            else:
                Int.__class_getitem__ = original_class_getitem

        self.addCleanup(restore_int_class_getitem)

    def test_int_subscript_erases_to_int(self):
        enable_torchscript_runtime_compat()

        def f[N]() -> None:
            self.assertIs(Int[N], int)

        f()

    def test_tensor_subscript_still_erases_to_tensor(self):
        enable_torchscript_runtime_compat()

        self.assertIs(torch.Tensor[[3, 4]], torch.Tensor)


class TestIntTupleRuntime(unittest.TestCase):
    def test_call_constructs_tuple(self):
        value = IntTuple([1, 2])

        self.assertEqual(value, (1, 2))
        self.assertIs(type(value), tuple)

    def test_subscript_erases_to_runtime_helper(self):
        self.assertIs(IntTuple[1, 2], IntTuple)


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
        """Tensor[[N+1, 3]] — arithmetic is evaluated before subscript, so
        the error is 'unsupported operand' not 'not subscriptable'."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: torch.Tensor[[N + 1, 3]]) -> torch.Tensor[[N, 3]]:
                return x


class TestSymbolicArithExprRuntime(unittest.TestCase):
    """D wraps PEP 695 TypeVars so runtime arithmetic can build an expression."""

    def test_bracket_and_call_forms_match(self):
        def f[N]() -> None:
            self.assertEqual(D[N], D(N))

        f()

    def test_arithmetic_expression_tree(self):
        def f[N, M]() -> None:
            expr = (D[N] + 1) * (2 ** D(M)) // -D[N]
            self.assertEqual(str(expr), "((N + 1) * (2 ** M)) // -N")

        f()

    def test_reverse_operators(self):
        def f[N]() -> None:
            self.assertEqual(str(1 + D[N]), "1 + N")
            self.assertEqual(str(1 - D[N]), "1 - N")
            self.assertEqual(str(2 * D[N]), "2 * N")
            self.assertEqual(str(4 // D[N]), "4 // N")

        f()

    def test_tensor_annotations_with_bracket_form(self):
        def f[N, M](x: torch.Tensor[[D[N] + D[M], 3]]) -> torch.Tensor[[D[N] * 2, 3]]:
            return x

        self.assertTrue(callable(f))

    def test_tensor_annotations_with_call_form(self):
        def f[N, M](x: torch.Tensor[[D(N) + D(M), 3]]) -> torch.Tensor[[D(N) // 2, 3]]:
            return x

        self.assertTrue(callable(f))


class TestAssertShapeRuntime(unittest.TestCase):
    """Default assert_shape checks concrete shapes and symbolic ranks."""

    class Array:
        __slots__ = ("shape",)

        def __init__(self, shape):
            self.shape = shape

    def test_concrete_shape_matches(self):
        x = self.Array((2, 3))
        self.assertIs(assert_shape(x, (2, 3)), x)

    def test_concrete_shape_mismatch(self):
        with self.assertRaisesRegex(
            AssertionError, r"expected shape \(2, 3\), got \(2, 4\)"
        ):
            assert_shape(self.Array((2, 4)), (2, 3))

    def test_symbolic_shape_checks_rank_only(self):
        def f[N]() -> None:
            x = self.Array((2, 4))
            self.assertIs(assert_shape(x, (2, D[N] + 1)), x)

        f()

    def test_symbolic_shape_rank_mismatch(self):
        def f[N]() -> None:
            with self.assertRaisesRegex(AssertionError, r"expected rank 2"):
                assert_shape(self.Array((2, 4, 5)), (2, D[N] + 1))

        f()

    def test_defines_assert_shape_returns_function(self):
        def custom(x, shape):
            return (x, shape)

        self.assertIs(defines_assert_shape(custom), custom)


class TestClassAnnotationRuntime(unittest.TestCase):
    """Classes with class-level and method-level TypeVars."""

    def test_class_concrete_subscript(self):
        """Class method with Tensor[[3, 4]] — works after shape_extensions patch."""

        class Layer:
            def forward(self, x: torch.Tensor[[3, 4]]) -> torch.Tensor[[3, 4]]:
                return x

        self.assertTrue(hasattr(Layer, "forward"))

    def test_class_typevars_no_arithmetic(self):
        """Class-level (N, M) and method-level (B) TypeVars — works after patch."""

        class Layer[N, M]:
            def forward[B](self, x: torch.Tensor[[B, N]]) -> torch.Tensor[[B, M]]:
                return x  # type: ignore[return-value]

        self.assertTrue(hasattr(Layer, "forward"))

    def test_class_typevar_arithmetic(self):
        """Class-level TypeVar with arithmetic in method annotation.
        Tensor[[N, 3]] works after patch, but N+1 still crashes because
        PEP 695 TypeVar doesn't support arithmetic."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            class PadLayer[N]:
                def forward(self, x: torch.Tensor[[N, 3]]) -> torch.Tensor[[N + 1, 3]]:
                    return x  # type: ignore[return-value]


class TestDimRuntime(unittest.TestCase):
    """Int[...] behavior at runtime.

    Int natively supports __class_getitem__, so Int[3] and Int[N] work.
    But Int[N+1] crashes because Python evaluates N+1 before passing it
    to __class_getitem__, and PEP 695 TypeVar doesn't support arithmetic."""

    def test_dim_concrete(self):
        """Int[3] — concrete integer, works fine."""

        def f(x: Int[3]) -> Int[3]:
            return x

        f(42)

    def test_dim_typevar(self):
        """Int[N] — bare TypeVar, works fine."""

        def f[N](x: Int[N]) -> Int[N]:
            return x

        f(42)

    def test_dim_arithmetic(self):
        """Int[N+1] — TypeVar arithmetic crashes."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
        ):

            def f[N](x: Int[N]) -> Int[N + 1]:
                return x

    def test_dim_two_typevars(self):
        """Int[N+M] — two TypeVars in arithmetic crashes."""
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'typing.TypeVar'",
        ):

            def f[N, M](x: Int[N]) -> Int[N + M]:
                return x


class TestIntVarRuntime(unittest.TestCase):
    """shape_extensions.IntVar provides arithmetic operators that don't crash."""

    def test_add(self):
        """N + 1 doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = N + 1
        self.assertIsNotNone(result)

    def test_radd(self):
        """1 + N doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = 1 + N
        self.assertIsNotNone(result)

    def test_sub(self):
        """N - 1 doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = N - 1
        self.assertIsNotNone(result)

    def test_rsub(self):
        """1 - N doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = 1 - N
        self.assertIsNotNone(result)

    def test_mul(self):
        """N * 2 doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = N * 2
        self.assertIsNotNone(result)

    def test_rmul(self):
        """2 * N doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = 2 * N
        self.assertIsNotNone(result)

    def test_floordiv(self):
        """N // 2 doesn't crash with shape_extensions.IntVar."""
        N = IntVar("N")
        result = N // 2
        self.assertIsNotNone(result)

    def test_chained(self):
        """(N + 1) * 2 doesn't crash — chained arithmetic."""
        N = IntVar("N")
        result = (N + 1) * 2
        self.assertIsNotNone(result)

    def test_two_vars(self):
        """N + M doesn't crash with two shape_extensions.IntVars."""
        N = IntVar("N")
        M = IntVar("M")
        result = N + M
        self.assertIsNotNone(result)

    def test_repr(self):
        """shape_extensions.IntVar repr shows the name."""
        N = IntVar("N")
        self.assertEqual(repr(N), "N")

    def test_in_dim(self):
        """Int[N] with shape_extensions.IntVar."""
        N = IntVar("N")

        def f(x: Int[N]) -> Int[N]:
            return x

        f(42)

    def test_arithmetic_in_dim(self):
        """Int[N+1] with shape_extensions.IntVar — N+1 returns self, which Int accepts."""
        N = IntVar("N")

        def f(x: Int[N]) -> Int[N + 1]:
            return x

        f(42)


class TestGenericRuntime(unittest.TestCase):
    """Generic works with shape_extensions.IntVar at runtime thanks to __class__ = typing.TypeVar."""

    def test_generic_subscript(self):
        """Generic[N, M] works with shape_extensions.IntVar."""
        N = IntVar("N")
        M = IntVar("M")

        class Foo(Generic[N, M]):
            pass

        self.assertTrue(issubclass(Foo, object))

    def test_generic_as_base_class(self):
        """class Foo(Generic[N, M]) with method annotations works."""
        N = IntVar("N")
        M = IntVar("M")

        class Foo(Generic[N, M]):
            def forward(self, x: Int[N]) -> Int[M]:
                return x

        foo = Foo()
        result = foo.forward(42)
        self.assertEqual(result, 42)

    def test_generic_accepts_intvar(self):
        """Generic[N] works when N is shape_extensions.IntVar — it sets
        __class__ = typing.TypeVar so isinstance(N, typing.TypeVar) returns True."""
        N = IntVar("N")

        class Layer(Generic[N]):
            def forward(self, x: Int[N]) -> Int[N + 1]:
                return x

        layer = Layer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_generic_with_dim_arithmetic(self):
        """Generic[N] with Int[N+1] in a method works."""
        N = IntVar("N")

        class PadLayer(Generic[N]):
            def forward(self, x: Int[N]) -> Int[N + 1]:
                return x

        layer = PadLayer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_typeddict_generic_intvar(self):
        """TypedDict + Generic[N] works with shape_extensions.IntVar."""
        N = IntVar("N")
        M = IntVar("M")

        class MyDict(TypedDict, Generic[N, M]):
            x: Int[N]
            y: Int[M]

        self.assertTrue(issubclass(MyDict, dict))


class TestTypeVarTupleRuntime(unittest.TestCase):
    """shape_extensions.TypeVarTuple supports star-unpacking at runtime."""

    def test_iter(self):
        """*Ns unpacking works — __iter__ yields self."""
        Ns = TypeVarTuple("Ns")
        items = list(Ns)
        self.assertEqual(len(items), 1)
        self.assertIs(items[0], Ns)

    def test_in_dim(self):
        """Int[*Ns] — star-unpacking in subscript works."""
        Ns = TypeVarTuple("Ns")

        def f(x: Int[*Ns]) -> Int[*Ns]:
            return x

        f(42)

    def test_generic(self):
        """Generic[*Ns] — variadic class generic works."""
        Ns = TypeVarTuple("Ns")

        class Layer(Generic[*Ns]):
            def forward(self, x: Int[*Ns]) -> Int[*Ns]:
                return x

        layer = Layer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_mixed_with_typevar(self):
        """Generic[*Ns, N] — variadic + fixed dim works."""
        Ns = TypeVarTuple("Ns")
        N = IntVar("N")

        class Layer(Generic[*Ns, N]):
            def forward(self, x: Int[*Ns]) -> Int[N + 1]:
                return x

        layer = Layer()
        result = layer.forward(42)
        self.assertEqual(result, 42)

    def test_repr(self):
        """shape_extensions.TypeVarTuple repr shows *name."""
        Ns = TypeVarTuple("Ns")
        self.assertEqual(repr(Ns), "*Ns")


if __name__ == "__main__":
    unittest.main()
