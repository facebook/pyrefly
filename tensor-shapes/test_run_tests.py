#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from run_tests import shaped_array_reference_lines


class ShapedArrayCorpusGuardTest(unittest.TestCase):
    def test_finds_every_reference(self) -> None:
        source = "\n".join(
            [
                "from shape_extensions import shaped_array",
                "@shaped_array(shape='Shape')",
                "# shaped_array must not remain in comments either",
                "class Ordinary: ...",
            ]
        )

        self.assertEqual(shaped_array_reference_lines(source), [1, 2, 3])
