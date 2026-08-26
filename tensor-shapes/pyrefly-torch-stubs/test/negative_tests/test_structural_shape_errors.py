# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn import functional as F


def check_invalid_structural_controls(
    x: Tensor[[2, 3]],
    cube: Tensor[[2, 3, 4]],
    indices: Tensor[[4]],
    scalar: Tensor[[]],
    scalar_index: Tensor[[]],
    matrix_index: Tensor[[2, 2]],
) -> None:
    # E: Cannot evaluate type-level shape DSL call: squeeze dimension out of range
    torch.squeeze(x, 2)
    # E: Cannot evaluate type-level shape DSL call: squeeze dimension out of range
    scalar.squeeze(1)
    # E: Cannot evaluate type-level shape DSL call: unsqueeze dimension out of range
    x.unsqueeze(-4)
    # E: Cannot evaluate type-level shape DSL call: transpose dimension out of range
    torch.transpose(x, 0, 2)
    # E: Cannot evaluate type-level shape DSL call: select dimension out of range
    x.select(-3, 0)
    # E: Cannot evaluate type-level shape DSL call: select dimension out of range
    scalar.select(-1, 0)
    # E: Cannot evaluate type-level shape DSL call: index_select dimension out of range
    torch.index_select(x, 2, indices)
    # E: Cannot evaluate type-level shape DSL call: index_select dimension out of range
    scalar.index_select(-1, indices)
    x.index_select(0, scalar_index)
    # E: Cannot evaluate type-level shape DSL call: index_select index must be 0D or 1D
    torch.index_select(x, 0, matrix_index)
    # E: Cannot evaluate type-level shape DSL call: unbind dimension out of range
    x.unbind(2)
    # E: Cannot evaluate type-level shape DSL call: unbind dimension out of range
    torch.unbind(x, -3)
    # E: Cannot evaluate type-level shape DSL call: unbind dimension out of range
    scalar.unbind()
    # E: Cannot evaluate type-level shape DSL call: unbind dimension out of range
    torch.unbind(scalar, -1)
    # E: Cannot evaluate type-level shape DSL call: dimension out of range
    torch.narrow(x, 2, 0, 1)
    # E: Cannot evaluate type-level shape DSL call: dimension out of range
    x.narrow(-3, 0, 1)
    # E: Cannot evaluate type-level shape DSL call: dimension out of range
    scalar.narrow(0, 0, 1)
    # E: Cannot evaluate type-level shape DSL call: dimension out of range
    torch.topk(x, 1, dim=2)
    # E: Cannot evaluate type-level shape DSL call: multinomial expects 1D or 2D input
    torch.multinomial(scalar, 1)
    # E: Cannot evaluate type-level shape DSL call: multinomial expects 1D or 2D input
    scalar.multinomial(1)
    # E: Cannot evaluate type-level shape DSL call: multinomial expects 1D or 2D input
    torch.multinomial(cube, 1)
    # E: Cannot evaluate type-level shape DSL call: multinomial expects 1D or 2D input
    cube.multinomial(1)
    # E: Cannot evaluate type-level shape DSL call: size dimension out of range
    x.size(2)
    # E: Cannot evaluate type-level shape DSL call: size dimension out of range
    x.size(-3)
    # E: Cannot evaluate type-level shape DSL call: size dimension out of range
    scalar.size(0)
    # E: Cannot evaluate type-level shape DSL call: size dimension out of range
    scalar.size(-1)
    # E: Cannot evaluate type-level shape DSL call: unfold dimension out of range
    torch.unfold(x, 2, 1, 1)
    # E: Cannot evaluate type-level shape DSL call: unfold dimension out of range
    x.unfold(-3, 1, 1)
    # E: Cannot evaluate type-level shape DSL call: unfold dimension out of range
    torch.unfold(scalar, 1, 0, 1)
    # E: Cannot evaluate type-level shape DSL call: unfold size must not exceed the selected dimension
    x.unfold(0, 3, 1)
    # E: Cannot evaluate type-level shape DSL call: unfold size must not exceed the selected dimension
    scalar.unfold(0, 2, 1)
    # E: Cannot evaluate type-level shape DSL call: unfold size must be non-negative
    x.unfold(0, -1, 1)
    # E: Cannot evaluate type-level shape DSL call: unfold step must be greater than zero
    x.unfold(0, 1, 0)
    # E: Cannot evaluate type-level shape DSL call: diag_embed input must have at least one dimension
    torch.diag_embed(scalar)
    # E: Cannot evaluate type-level shape DSL call: diag_embed dimensions must be different
    torch.diag_embed(x, dim1=1, dim2=-2)
    # E: Cannot evaluate type-level shape DSL call: diag_embed dimension out of range
    torch.diag_embed(x, dim1=-4)


def check_invalid_cosine_similarity_controls(
    x: Tensor[[2, 3]], incompatible: Tensor[[4, 5]], scalar: Tensor[[]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: cosine_similarity dimension out of range
    F.cosine_similarity(x, x, dim=2)
    # E: Cannot evaluate type-level shape DSL call: cosine_similarity dimension out of range
    F.cosine_similarity(x, x, dim=-3)
    # E: Cannot evaluate type-level shape DSL call: cosine_similarity dimension out of range
    F.cosine_similarity(scalar, scalar, dim=1)
    # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[3] with dimension Int[5] at position 1
    F.cosine_similarity(x, incompatible, dim=0)
