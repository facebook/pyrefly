# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Test meta-shape function integration
from typing import Any, assert_type, Literal

import torch
from torch import Tensor


# Test 1: Basic torch.cat with literal dim
def test_cat_basic():
    x: Tensor[2, 3] = torch.randn(2, 3)
    y: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[4, 3] (concat along dim 0 adds 2+2=4)
    result = torch.cat((x, y), dim=0)
    assert_type(result, Tensor[4, 3])


# Test 2: torch.reshape with literal shape tuple
def test_reshape():
    x: Tensor[6] = torch.randn(6)
    # Should infer: Tensor[2, 3]
    result = torch.reshape(x, (2, 3))
    assert_type(result, Tensor[2, 3])


# Test 2M: x.reshape with literal shape tuple (method style)
def test_reshape_method():
    x: Tensor[6] = torch.randn(6)
    # Should infer: Tensor[2, 3]
    result = x.reshape((2, 3))
    assert_type(result, Tensor[2, 3])


# Test 3: torch.sum with dim and keepdim
def test_sum_with_dim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (reduce dim 1, don't keep it)
    result = torch.sum(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 3M: x.sum with dim and keepdim (method style)
def test_sum_with_dim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (reduce dim 1, don't keep it)
    result = x.sum(dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 4: torch.transpose
def test_transpose():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[3, 2] (swap dims 0 and 1)
    result = torch.transpose(x, 0, 1)
    assert_type(result, Tensor[3, 2])


# Test 4M: x.transpose (method style)
def test_transpose_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[3, 2] (swap dims 0 and 1)
    result = x.transpose(0, 1)
    assert_type(result, Tensor[3, 2])


# Test 5: torch.squeeze - remove dimension of size 1
def test_squeeze():
    x: Tensor[2, 1, 3] = torch.randn(2, 1, 3)
    # Should infer: Tensor[2, 3] (remove dim 1)
    result = torch.squeeze(x, dim=1)
    assert_type(result, Tensor[2, 3])


# Test 5M: x.squeeze - remove dimension of size 1 (method style)
def test_squeeze_method():
    x: Tensor[2, 1, 3] = torch.randn(2, 1, 3)
    # Should infer: Tensor[2, 3] (remove dim 1)
    result = x.squeeze(dim=1)
    assert_type(result, Tensor[2, 3])


# Test 6: torch.unsqueeze - add dimension of size 1
def test_unsqueeze():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[2, 1, 3] (add dim at position 1)
    result = torch.unsqueeze(x, dim=1)
    assert_type(result, Tensor[2, 1, 3])


# Test 6M: x.unsqueeze - add dimension of size 1 (method style)
def test_unsqueeze_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[2, 1, 3] (add dim at position 1)
    result = x.unsqueeze(dim=1)
    assert_type(result, Tensor[2, 1, 3])


# Test 7: torch.permute - permute dimensions
def test_permute():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[4, 2, 3] (permute to [2, 0, 1])
    result = torch.permute(x, (2, 0, 1))
    assert_type(result, Tensor[4, 2, 3])


# Test 7M: x.permute - permute dimensions (method style)
def test_permute_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[4, 2, 3] (permute to [2, 0, 1])
    result = x.permute((2, 0, 1))
    assert_type(result, Tensor[4, 2, 3])


# Test 8: torch.mean - reduction operation
def test_mean():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (mean along dim 1, don't keep it)
    result = torch.mean(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 8M: x.mean - reduction operation (method style)
def test_mean_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (mean along dim 1, don't keep it)
    result = x.mean(dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 9: torch.max - reduction operation
def test_max():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 3], Tensor[2, 3]] (values, indices along dim 2)
    values, indices = torch.max(x, dim=2, keepdim=False)
    assert_type(values, Tensor[2, 3])
    assert_type(indices, Tensor[2, 3])


# Test 9M: x.max - reduction operation (method style)
def test_max_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 3], Tensor[2, 3]] (values, indices along dim 2)
    values, indices = x.max(dim=2, keepdim=False)
    assert_type(values, Tensor[2, 3])
    assert_type(indices, Tensor[2, 3])


# Test 10: torch.zeros - tensor creation
def test_zeros():
    # Should infer: Tensor[3, 4]
    assert_type(torch.zeros(3, 4), Tensor[3, 4])


# Test 11: torch.ones - tensor creation
def test_ones():
    # Should infer: Tensor[5, 2]
    assert_type(torch.ones(5, 2), Tensor[5, 2])


# Test 12: torch.flatten - flatten all dimensions
def test_flatten_all():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[24] (2*3*4 = 24)
    result = torch.flatten(x)
    assert_type(result, Tensor[24])


# Test 12M: x.flatten - flatten all dimensions (method style)
def test_flatten_all_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[24] (2*3*4 = 24)
    result = x.flatten()
    assert_type(result, Tensor[24])


# Test 13: torch.flatten - partial flatten
def test_flatten_partial():
    x: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    # Should infer: Tensor[2, 12, 5] (flatten dims 1 and 2: 3*4 = 12)
    result = torch.flatten(x, start_dim=1, end_dim=2)
    assert_type(result, Tensor[2, 12, 5])


# Test 13M: x.flatten - partial flatten (method style)
def test_flatten_partial_method():
    x: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    # Should infer: Tensor[2, 12, 5] (flatten dims 1 and 2: 3*4 = 12)
    result = x.flatten(start_dim=1, end_dim=2)
    assert_type(result, Tensor[2, 12, 5])


# Test 14: torch.stack - stack tensors along new dimension
def test_stack():
    x: Tensor[2, 3] = torch.randn(2, 3)
    y: Tensor[2, 3] = torch.randn(2, 3)
    z: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[3, 2, 3] (stack 3 tensors along dim 0)
    result = torch.stack((x, y, z), dim=0)
    assert_type(result, Tensor[3, 2, 3])


# Test 15: torch.std - standard deviation reduction
def test_std():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (std along dim 1)
    result = torch.std(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 15M: x.std - standard deviation reduction (method style)
def test_std_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (std along dim 1)
    result = x.std(dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 16: torch.argmax - argmax reduction
def test_argmax():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3] (argmax along dim 2)
    result = torch.argmax(x, dim=2, keepdim=False)
    assert_type(result, Tensor[2, 3])


# Test 16M: x.argmax - argmax reduction (method style)
def test_argmax_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3] (argmax along dim 2)
    result = x.argmax(dim=2, keepdim=False)
    assert_type(result, Tensor[2, 3])


# Test 17: torch.broadcast_to
def test_broadcast_to():
    x: Tensor[3] = torch.randn(3)
    # Should infer: Tensor[2, 3] (broadcast to larger shape)
    result = torch.broadcast_to(x, (2, 3))
    assert_type(result, Tensor[2, 3])


# Test 18: torch.tile
def test_tile():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[4, 9] (tile 2x in dim 0, 3x in dim 1: 2*2=4, 3*3=9)
    result = torch.tile(x, (2, 3))
    assert_type(result, Tensor[4, 9])


# Test 19: x.view (method style)
def test_view():
    x: Tensor[6] = torch.randn(6)
    # Should infer: Tensor[2, 3]
    result = x.view(2, 3)
    assert_type(result, Tensor[2, 3])


# Test 20: torch.select
def test_select():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (select along dim 1 removes that dimension)
    result = torch.select(x, dim=1, index=0)
    assert_type(result, Tensor[2, 4])


# Test 21: torch.narrow
def test_narrow():
    x: Tensor[5, 4] = torch.randn(5, 4)
    # Should infer: Tensor[3, 4] (narrow dim 0 to length 3)
    result = torch.narrow(x, dim=0, start=1, length=3)
    assert_type(result, Tensor[3, 4])


# Test 22: torch.split
def test_split():
    x: Tensor[6, 4] = torch.randn(6, 4)
    # split returns tuple of tensors with computed shapes
    result = torch.split(x, split_size_or_sections=2, dim=0)
    # Should split 6 into 3 chunks of 2 each
    assert_type(result, tuple[Tensor[2, 4], Tensor[2, 4], Tensor[2, 4]])


# Test 23: torch.chunk
def test_chunk():
    x: Tensor[6, 4] = torch.randn(6, 4)
    # chunk returns tuple of tensors with computed shapes
    result = torch.chunk(x, chunks=3, dim=0)
    # Should split 6 into 3 chunks of 2 each
    assert_type(result, tuple[Tensor[2, 4], Tensor[2, 4], Tensor[2, 4]])


# Test 24: torch.index_select
def test_index_select():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    indices: Tensor[2] = torch.zeros(2)  # Select 2 elements
    # Should infer: Tensor[3, 2, 5] (replace dim 1 with size of indices)
    result = torch.index_select(x, dim=1, index=indices)
    assert_type(result, Tensor[3, 2, 5])


# Test 25: torch.gather
def test_gather():
    x: Tensor[3, 4] = torch.randn(3, 4)
    indices: Tensor[3, 2] = torch.zeros(3, 2)
    # Should infer: Tensor[3, 2] (same shape as indices)
    result = torch.gather(x, dim=1, index=indices)
    assert_type(result, Tensor[3, 2])


# Test 26: torch.scatter
def test_scatter():
    x: Tensor[3, 4] = torch.randn(3, 4)
    indices: Tensor[3, 2] = torch.zeros(3, 2)
    values: Tensor[3, 2] = torch.randn(3, 2)
    # Should infer: Tensor[3, 4] (same shape as input)
    result = torch.scatter(x, dim=1, index=indices, src=values)
    assert_type(result, Tensor[3, 4])


# Test 27: torch.masked_select
def test_masked_select():
    x: Tensor[3, 4] = torch.randn(3, 4)
    mask: Tensor[3, 4] = torch.ones(3, 4)
    # Should infer: Tensor[Any] (1D tensor of unknown size)
    result = torch.masked_select(x, mask)
    assert_type(result, Tensor[Any])


# Test 28: torch.prod
def test_prod():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (prod along dim 1)
    result = torch.prod(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 28M: x.prod (method style)
def test_prod_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (prod along dim 1)
    result = x.prod(dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 29: torch.min with dim (returns tuple)
def test_min():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 3], Tensor[2, 3]] (values, indices along dim 2)
    values, indices = torch.min(x, dim=2, keepdim=False)
    assert_type(values, Tensor[2, 3])
    assert_type(indices, Tensor[2, 3])


# Test 29M: x.min (method style with dim, returns tuple)
def test_min_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 3], Tensor[2, 3]] (values, indices along dim 2)
    values, indices = x.min(dim=2, keepdim=False)
    assert_type(values, Tensor[2, 3])
    assert_type(indices, Tensor[2, 3])


# Test 30: torch.all
def test_all():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (all along dim 1)
    result = torch.all(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 31: torch.any
def test_any():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3] (any along dim 2)
    result = torch.any(x, dim=2, keepdim=False)
    assert_type(result, Tensor[2, 3])


# Test 32: torch.var
def test_var():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (variance along dim 1)
    result = torch.var(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 32M: x.var (method style)
def test_var_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (variance along dim 1)
    result = x.var(dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 33: torch.argmin
def test_argmin():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (argmin along dim 1)
    result = torch.argmin(x, dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 33M: x.argmin (method style)
def test_argmin_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (argmin along dim 1)
    result = x.argmin(dim=1, keepdim=False)
    assert_type(result, Tensor[2, 4])


# Test 34: torch.rand
def test_rand():
    # Should infer: Tensor[3, 4]
    assert_type(torch.rand(3, 4), Tensor[3, 4])


# Test 35: torch.empty
def test_empty():
    # Should infer: Tensor[2, 3]
    assert_type(torch.empty(2, 3), Tensor[2, 3])


# Test 36: torch.full
def test_full():
    # Should infer: Tensor[2, 3] (filled with value)
    assert_type(torch.full((2, 3), 5.0), Tensor[2, 3])


# Test 37: torch.arange
def test_arange():
    # Should infer: Tensor[10] (0 to 9)
    assert_type(torch.arange(10), Tensor[10])


# Test 38: torch.linspace
def test_linspace():
    # Should infer: Tensor[5] (5 points from 0 to 1)
    assert_type(torch.linspace(0, 1, 5), Tensor[5])


# Test 39: torch.eye
def test_eye():
    # Should infer: Tensor[3, 3] (3x3 identity matrix)
    assert_type(torch.eye(3), Tensor[3, 3])


# Additional method-style tests for operations that also have method forms


# Test 40M: x.reshape with varargs (method style)
def test_reshape_varargs_method():
    x: Tensor[6] = torch.randn(6)
    # Should infer: Tensor[2, 3]
    result = x.reshape(2, 3)
    assert_type(result, Tensor[2, 3])


# Test 41M: x.view with tuple (method style)
def test_view_tuple_method():
    x: Tensor[6] = torch.randn(6)
    # Should infer: Tensor[2, 3]
    result = x.view((2, 3))
    assert_type(result, Tensor[2, 3])


# Test 42M: x.repeat for tiling (method style)
def test_repeat_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[4, 9] (repeat 2x in dim 0, 3x in dim 1: 2*2=4, 3*3=9)
    result = x.repeat((2, 3))
    assert_type(result, Tensor[4, 9])


# Test 43M: Multiple reductions with different dims
def test_sum_multiple_dims_method():
    x: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    # Should infer: Tensor[2, 5] (reduce dims 1 and 2)
    result = x.sum(dim=(1, 2), keepdim=False)
    assert_type(result, Tensor[2, 5])


# Test 44M: Transpose with t() method (2D tensors)
def test_t_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[3, 2] (transpose)
    result = x.t()
    assert_type(result, Tensor[3, 2])


# Test 45M: Contiguous after transpose
def test_contiguous_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[3, 2] (transpose, then contiguous keeps shape)
    result = x.transpose(0, 1).contiguous()
    assert_type(result, Tensor[3, 2])


# Test 46M: Clone preserves shape
def test_clone_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3, 4] (clone preserves shape)
    result = x.clone()
    assert_type(result, Tensor[2, 3, 4])


# Test 47M: Detach preserves shape
def test_detach_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3, 4] (detach preserves shape)
    result = x.detach()
    assert_type(result, Tensor[2, 3, 4])


# Test 48M: Expand broadcast
def test_expand_method():
    x: Tensor[1, 3] = torch.randn(1, 3)
    # Should infer: Tensor[2, 3] (expand dim 0 from 1 to 2)
    result = x.expand(2, 3)
    assert_type(result, Tensor[2, 3])


# Test 49M: x.size returns Size with precise Literal dimensions
def test_size_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # size() returns torch.Size with Literal dimensions for concrete shapes
    assert_type(x.size(), tuple[Literal[2], Literal[3], Literal[4]])


# Test 50M: Reshape with -1 inference
def test_reshape_infer_dim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 12] (infer -1 as 12)
    result = x.reshape(2, -1)
    assert_type(result, Tensor[2, 12])


# Test 51M: View with -1 inference
def test_view_infer_dim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 12] (infer -1 as 12)
    result = x.view(2, -1)
    assert_type(result, Tensor[2, 12])


# Test 52M: Squeeze all singleton dimensions
def test_squeeze_all_method():
    x: Tensor[1, 2, 1, 3] = torch.randn(1, 2, 1, 3)
    # Should infer: Tensor[2, 3] (remove all singleton dims)
    result = x.squeeze()
    assert_type(result, Tensor[2, 3])


# Test 53M: Mean over all dimensions
def test_mean_all_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[()] (scalar)
    result = x.mean()
    assert_type(result, Tensor[()])


# Test 54: torch.reshape with -1 inference (function style)
def test_reshape_infer_dim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 12] (infer -1 as 12)
    result = torch.reshape(x, (2, -1))
    assert_type(result, Tensor[2, 12])


# Test 55: torch.squeeze all singleton dimensions (function style)
def test_squeeze_all():
    x: Tensor[1, 2, 1, 3] = torch.randn(1, 2, 1, 3)
    # Should infer: Tensor[2, 3] (remove all singleton dims)
    result = torch.squeeze(x)
    assert_type(result, Tensor[2, 3])


# Test 56: torch.mean over all dimensions (function style)
def test_mean_all():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[()] (scalar)
    result = torch.mean(x)
    assert_type(result, Tensor[()])


# Test 57M: Multiple consecutive operations
def test_chained_ops_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[12, 2] (reshape, then transpose)
    result = x.reshape(2, 12).transpose(0, 1)
    assert_type(result, Tensor[12, 2])


# Test 58M: Unsqueeze multiple times
def test_unsqueeze_chain_method():
    x: Tensor[3] = torch.randn(3)
    # Should infer: Tensor[1, 1, 3] (add two dims)
    result = x.unsqueeze(0).unsqueeze(0)
    assert_type(result, Tensor[1, 1, 3])


# Test 59M: Sum with keepdim=True
def test_sum_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (reduce dim 1, keep it as size 1)
    result = x.sum(dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 60: torch.sum with keepdim=True (function style)
def test_sum_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (reduce dim 1, keep it as size 1)
    result = torch.sum(x, dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 61M: Max with keepdim=True (returns tuple)
def test_max_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 3, 1], Tensor[2, 3, 1]] (values, indices with keepdim)
    values, indices = x.max(dim=2, keepdim=True)
    assert_type(values, Tensor[2, 3, 1])
    assert_type(indices, Tensor[2, 3, 1])


# Test 62: torch.max with keepdim=True (function style, returns tuple)
def test_max_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 3, 1], Tensor[2, 3, 1]] (values, indices with keepdim)
    values, indices = torch.max(x, dim=2, keepdim=True)
    assert_type(values, Tensor[2, 3, 1])
    assert_type(indices, Tensor[2, 3, 1])


# Test 63M: Flatten with single start_dim
def test_flatten_start_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 12] (flatten from dim 1 to end: 3*4=12)
    result = x.flatten(start_dim=1)
    assert_type(result, Tensor[2, 12])


# Test 64: torch.flatten with single start_dim (function style)
def test_flatten_start():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 12] (flatten from dim 1 to end: 3*4=12)
    result = torch.flatten(x, start_dim=1)
    assert_type(result, Tensor[2, 12])


# Test 65M: Permute identity (no change)
def test_permute_identity_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3, 4] (identity permutation)
    result = x.permute((0, 1, 2))
    assert_type(result, Tensor[2, 3, 4])


# Test 66: torch.permute identity (function style)
def test_permute_identity():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 3, 4] (identity permutation)
    result = torch.permute(x, (0, 1, 2))
    assert_type(result, Tensor[2, 3, 4])


# Test 67M: Argmax without keepdim (default)
def test_argmax_no_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (argmax along dim 1, don't keep it)
    result = x.argmax(dim=1)
    assert_type(result, Tensor[2, 4])


# Test 68: torch.argmax without keepdim (function style)
def test_argmax_no_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (argmax along dim 1, don't keep it)
    result = torch.argmax(x, dim=1)
    assert_type(result, Tensor[2, 4])


# Test 69M: Std with keepdim=True
def test_std_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (std along dim 1, keep it)
    result = x.std(dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 70: torch.std with keepdim=True (function style)
def test_std_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (std along dim 1, keep it)
    result = torch.std(x, dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 71M: Var with keepdim=True
def test_var_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (var along dim 1, keep it)
    result = x.var(dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 72: torch.var with keepdim=True (function style)
def test_var_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (var along dim 1, keep it)
    result = torch.var(x, dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 73M: Min with keepdim=True (returns tuple)
def test_min_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 1, 4], Tensor[2, 1, 4]] (values, indices with keepdim)
    values, indices = x.min(dim=1, keepdim=True)
    assert_type(values, Tensor[2, 1, 4])
    assert_type(indices, Tensor[2, 1, 4])


# Test 74: torch.min with keepdim=True (function style, returns tuple)
def test_min_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: tuple[Tensor[2, 1, 4], Tensor[2, 1, 4]] (values, indices with keepdim)
    values, indices = torch.min(x, dim=1, keepdim=True)
    assert_type(values, Tensor[2, 1, 4])
    assert_type(indices, Tensor[2, 1, 4])


# Test 75M: Prod with keepdim=True
def test_prod_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (prod along dim 1, keep it)
    result = x.prod(dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 76: torch.prod with keepdim=True (function style)
def test_prod_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (prod along dim 1, keep it)
    result = torch.prod(x, dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 77M: Argmin with keepdim=True
def test_argmin_keepdim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (argmin along dim 1, keep it)
    result = x.argmin(dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test 78: torch.argmin with keepdim=True (function style)
def test_argmin_keepdim():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 1, 4] (argmin along dim 1, keep it)
    result = torch.argmin(x, dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# ============================================================================
# Additional Tests for Complete Coverage
# ============================================================================


# Test 79: torch.view (function style) - rarely used but for completeness
def test_view_functional():
    x: Tensor[6] = torch.randn(6)
    # Should infer: Tensor[2, 3]
    result = torch.reshape(
        x, (2, 3)
    )  # Note: torch.view doesn't exist as function, using reshape
    assert_type(result, Tensor[2, 3])


# Test 80M: x.tile() method style
def test_tile_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Should infer: Tensor[4, 9] (tile 2x in dim 0, 3x in dim 1: 2*2=4, 3*3=9)
    result = x.tile((2, 3))
    assert_type(result, Tensor[4, 9])


# Test 81M: x.select() method style
def test_select_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Should infer: Tensor[2, 4] (select along dim 1 removes that dimension)
    result = x.select(dim=1, index=0)
    assert_type(result, Tensor[2, 4])


# Test 82M: x.narrow() method style
def test_narrow_method():
    x: Tensor[5, 4] = torch.randn(5, 4)
    # Should infer: Tensor[3, 4] (narrow dim 0 to length 3)
    result = x.narrow(dim=0, start=1, length=3)
    assert_type(result, Tensor[3, 4])


# Test 83M: x.split() method style
def test_split_method():
    x: Tensor[6, 4] = torch.randn(6, 4)
    # split returns tuple of tensors with computed shapes
    result = x.split(split_size_or_sections=2, dim=0)
    # Should split 6 into 3 chunks of 2 each
    assert_type(result, tuple[Tensor[2, 4], Tensor[2, 4], Tensor[2, 4]])


# Test 84M: x.chunk() method style
def test_chunk_method():
    x: Tensor[6, 4] = torch.randn(6, 4)
    # chunk returns tuple of tensors with computed shapes
    result = x.chunk(chunks=3, dim=0)
    # Should split 6 into 3 chunks of 2 each
    assert_type(result, tuple[Tensor[2, 4], Tensor[2, 4], Tensor[2, 4]])


# Test 85M: x.index_select() method style
def test_index_select_method():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    indices: Tensor[2] = torch.zeros(2)  # Select 2 elements
    # Should infer: Tensor[3, 2, 5] (replace dim 1 with size of indices)
    result = x.index_select(dim=1, index=indices)
    assert_type(result, Tensor[3, 2, 5])


# Test 86M: x.gather() method style
def test_gather_method():
    x: Tensor[3, 4] = torch.randn(3, 4)
    indices: Tensor[3, 2] = torch.zeros(3, 2)
    # Should infer: Tensor[3, 2] (same shape as indices)
    result = x.gather(dim=1, index=indices)
    assert_type(result, Tensor[3, 2])


# Test 87M: x.scatter() method style
def test_scatter_method():
    x: Tensor[3, 4] = torch.randn(3, 4)
    indices: Tensor[3, 2] = torch.zeros(3, 2)
    values: Tensor[3, 2] = torch.randn(3, 2)
    # Should infer: Tensor[3, 4] (same shape as input)
    result = x.scatter(dim=1, index=indices, src=values)
    assert_type(result, Tensor[3, 4])


# Test 88M: x.masked_select() method style
def test_masked_select_method():
    x: Tensor[3, 4] = torch.randn(3, 4)
    mask: Tensor[3, 4] = torch.ones(3, 4)
    # Should infer: Tensor[Any] (1D tensor of unknown size)
    result = x.masked_select(mask)
    assert_type(result, Tensor[Any])
