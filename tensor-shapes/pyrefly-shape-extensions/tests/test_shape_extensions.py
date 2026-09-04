# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing
import unittest

from shape_extensions import shape_vars


class ShapeVarsRuntimeTest(unittest.TestCase):
    def test_empty_declaration_preserves_functions(self) -> None:
        def identity(value: int) -> int:
            return value

        self.assertIs(shape_vars("")(identity), identity)

    def test_empty_declaration_preserves_generic_type_arguments(self) -> None:
        @shape_vars("")
        class Box[T]:
            pass

        specialized = Box[int]
        self.assertIs(typing.get_origin(specialized), Box)
        self.assertEqual(typing.get_args(specialized), (int,))
