# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test rejection behavior for filled top-level Torch gaps."""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def test_unknown_top_level_name() -> None:
    torch.not_a_real_api()  # E: No attribute `not_a_real_api` in module `torch`


def test_new_zeros_rejects_mixed_size() -> None:
    x = torch.zeros(2, 3)
    x.new_zeros(
        (2, 3),  # E: not assignable
        4,
    )


def test_new_zeros_requires_size() -> None:
    x = torch.zeros(2, 3)
    x.new_zeros()  # E: No matching overload found


def test_creation_options_are_keyword_only() -> None:
    torch.tensor([1, 2], torch.float32)  # E: Expected at most 1 positional argument
    torch.Tensor([1, 2], "cpu")  # E: Unpacked argument
    torch.full((2, 2), 0.0, torch.float32)  # E: Expected at most 2 positional
    torch.rand((2, 2), torch.float32)  # E: Unpacked argument
    torch.eye(2, 3, torch.float32)  # E: No matching overload found


def test_scalar_options_reject_unrelated_values() -> None:
    x = torch.zeros(2, 3)
    torch.mul(x, object())  # E: is not assignable to parameter `other`
    torch.allclose(x, x, equal_nan="yes")  # E: is not assignable


def test_context_does_not_override_constructor_scalar_shape(
    _x: torch.Tensor[[2, 2]],
) -> None:
    _x = torch.tensor(1)  # E: is not assignable to variable `_x`
    _x = torch.Tensor(1)  # E: is not assignable to variable `_x`
    assert_type(_x, torch.Tensor[[2, 2]])
