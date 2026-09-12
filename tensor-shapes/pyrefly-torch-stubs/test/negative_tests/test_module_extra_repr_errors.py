# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that `nn.Module.extra_repr` overrides are checked."""

from typing import override

import torch.nn as nn


class BadExtraRepr(nn.Module):
    @override
    def extra_repr(self) -> int:  # E: inconsistent manner
        return 0
