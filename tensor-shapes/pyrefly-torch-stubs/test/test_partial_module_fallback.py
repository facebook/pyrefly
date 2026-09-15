# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import Adafactor, Adam, Muon
from torch.quantization import default_eval_fn, quantize_dynamic


def test_unshipped_modules_fall_back_to_installed_torch() -> None:
    _ = Adafactor, Adam, Muon, default_eval_fn, quantize_dynamic
