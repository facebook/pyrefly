# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_axis_keyword() -> None:
    # TODO: BUG: `axis` is a supported alias for `dim`.
    # E: cat expects all tensor sizes to match outside the concatenated dimension
    torch.concatenate(
        (torch.zeros((2, 3)), torch.zeros((2, 4))),
        axis=1,  # E: Unexpected keyword argument `axis`
    )
