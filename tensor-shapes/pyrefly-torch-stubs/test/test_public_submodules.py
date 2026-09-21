# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that submodules imported by `torch/__init__.py` resolve as attributes.

Without the re-exports these resolve only through Pyrefly's implicit-import
fallback, which reports `implicit-import` at every use.
"""

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple

if TYPE_CHECKING:
    from torch.optim import Adafactor, Adam, Muon
    from torch.quantization import default_eval_fn


def test_torch_submodules_are_attributes() -> None:
    assert_type(torch.cuda.is_available(), bool)
    assert_type(torch.backends.mps.is_available(), bool)
    _ = torch.testing.assert_close, torch.special.erf, torch.optim.Adam
    _ = torch.nn.utils.clip_grad_norm_, torch.nn.parameter.Parameter
    # TODO: BUG: Add a shape-aware overlay for `torch.special`.
    assert_shape(
        torch.special.erf(torch.ones((2, 3))).shape,
        IntTuple,
        runtime=(2, 3),
    )


if TYPE_CHECKING:
    _ = Adafactor, Adam, Muon, default_eval_fn
    torch.not_a_real_api()  # E: No attribute `not_a_real_api` in module `torch`
