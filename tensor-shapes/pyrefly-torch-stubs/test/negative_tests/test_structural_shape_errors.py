# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import reveal_type

import torch
import torch.nn as nn
from shape_extensions import IntTuple
from torch import Tensor
from torch.nn import functional as F


def test_invalid_constructor_control_module_shapes() -> None:
    rank_two: Tensor[[8, 4]] = torch.randn(8, 4)
    bad_channels: Tensor[[2, 10, 4, 4]] = torch.randn(2, 10, 4, 4)
    glu_input: Tensor[[2, 5, 4]] = torch.randn(2, 5, 4)
    pad_rank_two: Tensor[[4, 4]] = torch.randn(4, 4)
    pad_rank_five: Tensor[[2, 3, 4, 4, 4]] = torch.randn(2, 3, 4, 4, 4)

    nn.PixelShuffle(2)(rank_two)  # E: PixelShuffle requires at least 3D input
    nn.PixelShuffle(0)(bad_channels)  # E: PixelShuffle upscale_factor must be positive
    nn.PixelShuffle(3)(bad_channels)  # E: PixelShuffle input channels must be divisible
    nn.GLU(3)(glu_input)  # E: GLU dimension out of range
    nn.GLU(1)(glu_input)  # E: GLU input dimension must be even
    nn.ReflectionPad2d(1)(pad_rank_two)  # E: 2D padding requires 3D or 4D input
    nn.ReflectionPad2d(1)(pad_rank_five)  # E: 2D padding requires 3D or 4D input
    nn.ReplicationPad2d(1)(pad_rank_two)  # E: 2D padding requires 3D or 4D input
    nn.ReplicationPad2d(1)(pad_rank_five)  # E: 2D padding requires 3D or 4D input


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


def check_invalid_permute_controls(x: Tensor[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: permute dimensions must match the input rank
    x.permute(0, 1)
    # E: Cannot evaluate type-level shape DSL call: permute dimension out of range
    x.permute(0, 1, 3)
    # E: Cannot evaluate type-level shape DSL call: permute dimensions must be unique
    x.permute(0, 0, 1)
    # E: Cannot evaluate type-level shape DSL call: permute dimensions must be unique
    torch.permute(x, (0, -1, 2))


def check_gradual_permute_controls[Shape: IntTuple, Dims: IntTuple](
    x: Tensor[Shape], dims: Dims, broad: tuple[int, int, int]
) -> None:
    # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(x.permute(dims))
    # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(torch.empty(2, 3, 4).permute(broad))


def check_repeat_interleave_controls(broad_dim: int, broad_repeats: int) -> None:
    concrete: Tensor[[2, 3]] = torch.empty(2, 3)

    # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(concrete.repeat_interleave(2, broad_dim))
    # E: revealed type: Tensor[[2, int]]
    reveal_type(concrete.repeat_interleave(broad_repeats, dim=1))

    # Zero repeats is an empty but valid result at runtime; a negative count is not.
    # E: revealed type: Tensor[[2, 0]]
    reveal_type(concrete.repeat_interleave(0, dim=1))
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave repeats must be non-negative
    torch.repeat_interleave(concrete, -1, dim=-1)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave repeats must be non-negative
    concrete.repeat_interleave(-2)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave output_size must be non-negative
    concrete.repeat_interleave(2, dim=0, output_size=-1)
    tensor_repeats = torch.tensor([2, 3])
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave output_size must be non-negative
    concrete.repeat_interleave(tensor_repeats, dim=0, output_size=-1)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave output_size does not match the result
    concrete.repeat_interleave(99, dim=1, output_size=5)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave output_size does not match the result
    torch.repeat_interleave(concrete, 99, output_size=5)

    # E: Cannot evaluate type-level shape DSL call: repeat_interleave dimension out of range
    concrete.repeat_interleave(2, dim=2)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave dimension out of range
    torch.repeat_interleave(concrete, 2, dim=-3)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave dimension out of range
    concrete.repeat_interleave(tensor_repeats, dim=2, output_size=5)

    # A rank-0 input only admits the synthesized axis named by dim 0 or -1.
    scalar: Tensor[[]] = torch.tensor(1)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave dimension out of range
    scalar.repeat_interleave(2, dim=1)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave dimension out of range
    torch.repeat_interleave(scalar, 2, dim=-2, output_size=2)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave dimension out of range
    torch.repeat_interleave(scalar, tensor_repeats, dim=-2, output_size=5)
    # E: Cannot evaluate type-level shape DSL call: repeat_interleave output_size does not match the result
    scalar.repeat_interleave(3, dim=0, output_size=4)

    concrete.repeat_interleave(1.5)  # E: No matching overload


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


def check_invalid_tile_parameters(x: Tensor[[2, 3]]) -> None:
    # PyTorch rejects negative repeats at runtime. The type-level DSL preserves
    # the corresponding arithmetic until tuple-wide validation is available.
    # E: revealed type: Tensor[[2, -3]]
    reveal_type(torch.tile(x, (1, -1)))
    # E: revealed type: Tensor[[2, 0]]
    reveal_type(x.tile((1, 0)))
    # E: `list[int]` is not assignable to upper bound `IntTuple` of type variable `Repeats`
    torch.tile(x, [2, 3])
    # E: Argument `tuple[Literal[2], float]` is not assignable to parameter `dims`
    x.tile((2, 3.0))


def check_invalid_repeat_parameters(x: Tensor[[2, 3]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor
    x.repeat(2)

    # PyTorch rejects negative repeats at runtime. As with tile, the type-level
    # DSL preserves the corresponding arithmetic until validation is available.
    # E: revealed type: Tensor[[2, -3]]
    reveal_type(x.repeat((1, -1)))
    # E: revealed type: Tensor[[2, 0]]
    reveal_type(x.repeat(1, 0))

    # E: No matching overload found for function `torch.Tensor.repeat`
    x.repeat([2, 3])


def check_invalid_expand_controls(x: Tensor[[2, 3]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: expand target rank cannot be smaller than input rank
    x.expand(2)
    # E: Cannot evaluate type-level shape DSL call: expand target rank cannot be smaller than input rank
    x.expand()
    # E: Cannot evaluate type-level shape DSL call: expand cannot use -1 for a new leading dimension
    x.expand(-1, 2, 3)

    # E: Cannot evaluate type-level shape DSL call: expand target dimension cannot be less than -1
    x.expand(-2, 3)
    # E: Cannot evaluate type-level shape DSL call: expand target dimension cannot be less than -1
    x.expand((-2, 3))
    # E: Cannot evaluate type-level shape DSL call: expand cannot resize a non-singleton dimension
    x.expand(4, 3)

    # Zero-size dimensions are preserved, although explicit zero shape
    # annotations are rejected elsewhere.
    # E: revealed type: Tensor[[0, 4]]
    reveal_type(torch.empty(0, 1).expand(0, 4))

    # E: No matching overload found for function `torch.Tensor.expand`
    x.expand([2, 3])
    # E: No matching overload found for function `torch.Tensor.expand`
    x.expand((2, 3.0))
    # E: No matching overload found for function `torch.Tensor.expand`
    x.expand((True, 2))
