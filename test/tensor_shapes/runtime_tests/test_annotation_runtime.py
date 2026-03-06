# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Runtime impact of native tensor type annotations.

Pyrefly's native tensor syntax (Tensor[N, 3], Tensor[N+1, M]) works at
type-checking time but crashes at Python runtime. Two independent problems:

1. torch.Tensor is not subscriptable — even Tensor[3, 4] fails.
2. TypeVar doesn't support arithmetic — N + 1, N * 2, etc. fail.

Both are blockers for runtime evaluation of these annotations.
"""

import unittest

import torch
from torch_shapes import Dim


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
    """Dim[...] isolates the arithmetic crash from the subscripting crash.

    Unlike torch.Tensor, Dim supports __class_getitem__, so Dim[3] and
    Dim[N] work fine. But Dim[N+1] still crashes because Python evaluates
    N+1 before passing it to __class_getitem__."""

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


if __name__ == "__main__":
    unittest.main()
