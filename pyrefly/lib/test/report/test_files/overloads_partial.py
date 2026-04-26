# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Overload merging edge cases.

from typing import overload


# Partial: one overload typed, one missing return → worst wins.
@overload
def partial(x: int) -> int: ...


@overload
def partial(x: str): ...


def partial(x):
    return x


# Different param counts → union of keys.
@overload
def different_params(x: int) -> int: ...


@overload
def different_params(x: int, y: str) -> str: ...


def different_params(x, y=None):
    return x


# Non-overloaded, unaffected by merge.
def standalone(a: int, b: str) -> bool:
    return True
