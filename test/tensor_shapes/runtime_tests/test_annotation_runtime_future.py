# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Same annotation patterns as test_annotation_runtime.py, but with
`from __future__ import annotations` (PEP 563).

With postponed evaluation, annotations are stored as strings and never
evaluated at definition time. This additionally avoids the TypeVar
arithmetic crash (N + 1 etc.), which remains a problem without future
annotations when using PEP 695 TypeVar (torch_shapes.TypeVar solves it differently).
"""

from __future__ import annotations

import unittest
from typing import assert_type, Generic

import torch
from torch_shapes import Dim, TypeVar


class TestSubscriptRuntime(unittest.TestCase):
    """torch.Tensor subscript — works both with and without future annotations
    (torch_shapes patches __class_getitem__)."""

    def test_concrete_subscript(self):
        """Tensor[3, 4] — concrete integer dims, no TypeVars."""

        def f(x: torch.Tensor[3, 4]) -> torch.Tensor[3, 4]:
            return x

        t = torch.randn(3, 4)
        result = f(t)
        self.assertEqual(result.shape, (3, 4))

    def test_typevar_subscript(self):
        """Tensor[N, 3] — TypeVar in subscript, no arithmetic."""

        def f[N](x: torch.Tensor[N, 3]) -> torch.Tensor[N, 3]:
            return x

        t = torch.randn(4, 3)
        result = f(t)
        self.assertEqual(result.shape, (4, 3))


class TestTypeVarArithmetic(unittest.TestCase):
    """PEP 695 TypeVar arithmetic — crashes without future annotations,
    works with (annotations become strings, arithmetic is never evaluated)."""

    def test_typevar_add(self):
        """N + 1 in an annotation."""

        def f[N](x: N + 1) -> None:  # type: ignore[valid-type]
            pass

        f(42)

    def test_typevar_mul(self):
        """N * 2 in an annotation."""

        def f[N](x: N * 2) -> None:  # type: ignore[valid-type]
            pass

        f(42)

    def test_typevar_sub(self):
        """N - 1 in an annotation."""

        def f[N](x: N - 1) -> None:  # type: ignore[valid-type]
            pass

        f(42)

    def test_typevar_floordiv(self):
        """N // 2 in an annotation."""

        def f[N](x: N // 2) -> None:  # type: ignore[valid-type]
            pass

        f(42)

    def test_two_typevars_add(self):
        """N + M in an annotation."""

        def f[N, M](x: N + M) -> None:  # type: ignore[valid-type]
            pass

        f(42)


class TestCombined(unittest.TestCase):
    """TypeVar arithmetic inside Tensor subscript."""

    def test_tensor_typevar_arithmetic(self):
        """Tensor[N+1, 3] — both problems at once, works with future annotations."""

        def f[N](x: torch.Tensor[N + 1, 3]) -> torch.Tensor[N, 3]:
            return x

        t = torch.randn(4, 3)
        result = f(t)
        self.assertEqual(result.shape, (4, 3))


class TestClassAnnotationRuntime(unittest.TestCase):
    """Classes with class-level and method-level TypeVars."""

    def test_class_concrete_subscript(self):
        """Class method with Tensor[3, 4] — concrete dims."""

        class Layer:
            def forward(self, x: torch.Tensor[3, 4]) -> torch.Tensor[3, 4]:
                return x

        result = Layer().forward(torch.randn(3, 4))
        self.assertEqual(result.shape, (3, 4))

    def test_class_typevars_no_arithmetic(self):
        """Class-level (N, M) and method-level (B) TypeVars, no arithmetic."""

        class Layer[N, M]:
            def forward[B](self, x: torch.Tensor[B, N]) -> torch.Tensor[B, M]:
                return x  # type: ignore[return-value]

        result = Layer().forward(torch.randn(2, 5))
        self.assertEqual(result.shape, (2, 5))

    def test_class_typevar_arithmetic(self):
        """Class-level TypeVar with arithmetic in method annotation."""

        class PadLayer[N]:
            def forward(self, x: torch.Tensor[N, 3]) -> torch.Tensor[N + 1, 3]:
                return x  # type: ignore[return-value]

        result = PadLayer().forward(torch.randn(4, 3))
        self.assertEqual(result.shape, (4, 3))


class TestDimRuntime(unittest.TestCase):
    """Dim[...] with future annotations — all work since annotations are strings."""

    def test_dim_concrete(self):
        """Dim[3] — works."""

        def f(x: Dim[3]) -> Dim[3]:
            return x

        f(42)

    def test_dim_typevar(self):
        """Dim[N] — works."""

        def f[N](x: Dim[N]) -> Dim[N]:
            return x

        f(42)

    def test_dim_arithmetic(self):
        """Dim[N+1] — works with future annotations (annotation is a string)."""

        def f[N](x: Dim[N]) -> Dim[N + 1]:
            return x

        f(42)

    def test_dim_two_typevars(self):
        """Dim[N+M] — works with future annotations."""

        def f[N, M](x: Dim[N]) -> Dim[N + M]:
            return x

        f(42)


class TestAssertTypeRuntime(unittest.TestCase):
    """assert_type's second argument is a regular expression, not an annotation.
    from __future__ import annotations does NOT postpone its evaluation."""

    def test_assert_type_concrete(self):
        """assert_type(x, Tensor[3, 4]) — works after torch_shapes patch."""
        t = torch.randn(3, 4)
        assert_type(t, torch.Tensor[3, 4])

    def test_assert_type_typevar(self):
        """assert_type(result, Tensor[N, 3]) — works after torch_shapes patch."""

        def f[N](x: torch.Tensor[N, 3]) -> torch.Tensor[N, 3]:
            assert_type(x, torch.Tensor[N, 3])
            return x

        f(torch.randn(4, 3))

    def test_assert_type_arithmetic(self):
        """assert_type(result, Tensor[N+1, 3]) — arithmetic in assert_type."""

        def f[N](x: torch.Tensor[N, 3]) -> torch.Tensor[N + 1, 3]:
            with self.assertRaisesRegex(
                TypeError,
                r"unsupported operand type\(s\) for \+: 'typing.TypeVar' and 'int'",
            ):
                assert_type(x, torch.Tensor[N + 1, 3])
            return x

        f(torch.randn(4, 3))


class TestTypeVarWithFutureAnnotations(unittest.TestCase):
    """torch_shapes.TypeVar combined with future annotations — everything works."""

    def test_in_annotation(self):
        """torch_shapes.TypeVar in annotations with future annotations."""
        N = TypeVar("N")
        M = TypeVar("M")

        def f(x: torch.Tensor[N, M]) -> torch.Tensor[N, M]:
            return x

        t = torch.randn(3, 4)
        result = f(t)
        self.assertEqual(result.shape, (3, 4))

    def test_arithmetic_in_annotation(self):
        """torch_shapes.TypeVar arithmetic in annotations with future annotations."""
        N = TypeVar("N")

        def f(x: torch.Tensor[N, 3]) -> torch.Tensor[N + 1, 3]:
            return x

        t = torch.randn(4, 3)
        result = f(t)
        self.assertEqual(result.shape, (4, 3))

    def test_generic_class(self):
        """Generic class with future annotations."""
        N = TypeVar("N")
        M = TypeVar("M")

        class Layer(Generic[N, M]):
            def forward(self, x: torch.Tensor[N]) -> torch.Tensor[M]:
                return x

        layer = Layer()
        t = torch.randn(5)
        result = layer.forward(t)
        self.assertEqual(result.shape, (5,))


if __name__ == "__main__":
    unittest.main()
