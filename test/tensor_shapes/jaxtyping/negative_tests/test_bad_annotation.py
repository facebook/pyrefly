# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative test: malformed jaxtyping annotations."""

from jaxtyping import Shaped
from torch import Tensor


def too_few_args(x: Shaped[Tensor]) -> None:
    """Jaxtyping annotations require exactly 2 arguments."""
    pass


def too_many_args(x: Shaped[Tensor, "3", "extra"]) -> None:  # noqa: F821
    """Jaxtyping annotations require exactly 2 arguments."""
    pass


def non_string_second_arg(x: Shaped[Tensor, 42]) -> None:
    """Second argument must be a string literal."""
    pass
