# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# A non-generic subclass of a shape-parameterized base has nowhere to retain
# constructor-inferred type arguments, so the base defaults apply and conflicting
# explicit values are correctly rejected. Accepting them would be unsound: the
# object is statically the default specialization.

from typing import assert_type

from shape_extensions import Int, IntVar


class Scaled[S: IntVar = 1, P: IntVar = 0]:
    factor: Int[S]
    padding: Int[P]

    def __init__(self, factor: Int[S] = 1, padding: Int[P] = 0) -> None: ...


class FixedScale(Scaled): ...


def test_non_generic_subclass_binds_defaults() -> None:
    default = FixedScale()
    assert_type(default.factor, Int[1])
    assert_type(default.padding, Int[0])

    FixedScale(
        factor=2  # E: not assignable to parameter `factor` with type `Int[1]`
    )
    FixedScale(
        padding=1  # E: not assignable to parameter `padding` with type `Int[0]`
    )
