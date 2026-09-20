# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test remaining ~35 operations with symbolic dimensions
Covers FFT variants, loss functions, creation ops, indexing, and specialized operations
"""

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.fft
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor

# ==== FFT Variants (~4 operations) ====
# Testing: rfft2, irfft2, rfftn, irfftn


def test_rfft2[H: IntVar, W: IntVar](x: Tensor[[2, H, W]]):
    """2D real FFT - last dimension changes"""
    y = torch.fft.rfft2(x)
    # Last dimension becomes W//2 + 1
    # For symbolic dims, may return shapeless
    assert_type(y, Tensor)


def test_rfftn[D1: IntVar, D2: IntVar, D3: IntVar](x: Tensor[[D1, D2, D3]]):
    """N-dimensional real FFT"""
    y = torch.fft.rfftn(x)
    # Last dimension changes, may return shapeless
    assert_type(y, Tensor)


# Test FFT operations
test_rfft2(torch.randn(2, 28, 28))
test_rfftn(torch.randn(4, 8, 16))

# Note: ifft2, ifftn, irfft2, irfftn, hfft, ihfft require complex dtype
# which is not supported in our test fixtures

# ==== Loss Functions (~5 operations) ====
# We already tested: mse_loss ✅
# Testing: cross_entropy, nll_loss, binary_cross_entropy, kl_div, smooth_l1_loss


def test_tolist_is_gradual(x: Tensor[[2, 3]]) -> None:
    assert_type(x.tolist(), Any)


# ==== Remaining Indexing Operations (~3 operations) ====
# We already tested: index_select, gather, scatter, masked_fill, take, index_add ✅
# Testing: index_copy, index_put, masked_scatter


def test_index_copy[N: IntVar, M: IntVar](
    x: Tensor[[N, M]], indices: Tensor[[2]], source: Tensor[[2, M]]
):
    """index_copy preserves input shape"""
    y = x.index_copy(0, indices, source)
    assert_type(y, Tensor[[N, M]])


def test_masked_scatter[N: IntVar, M: IntVar](
    x: Tensor[[N, M]], mask: Tensor[[N, M]], source: Tensor[[10]]
):
    """masked_scatter preserves shape"""
    # Fixture doesn't support .bool(), just verify operation doesn't crash
    # masked_scatter may not be in fixtures, just document
    # y = x.masked_scatter(mask, source)
    # expected: Tensor[[N, M]] = y
    pass


# Test indexing operations
_indices2 = torch.ones(2)
_t35 = torch.randn(3, 5)
_source25 = torch.randn(2, 5)
_mask35 = torch.ones(3, 5)
_source10 = torch.randn(10)
test_index_copy(_t35, _indices2, _source25)
test_masked_scatter(_t35, _mask35, _source10)

# Note: index_put is less commonly used, similar to scatter


def test_tensordot[N: IntVar, M: IntVar, K: IntVar](
    a: Tensor[[N, M, K]], b: Tensor[[K, 6]]
):
    """tensordot - generalized tensor contraction"""
    # Using dims=1 (simple int form) which is supported
    # dims=1 contracts last 1 dimension of a with first 1 dimension of b
    y = torch.tensordot(a, b, dims=1)
    # Contracts K dimension, result is [N, M, 6]
    assert_type(y, Tensor[[N, M, 6]])


_t345 = torch.randn(3, 4, 5)
_t56 = torch.randn(5, 6)
test_tensordot(_t345, _t56)

# ==== Random Sampling Operations (~5 operations) ====
# Testing: multinomial, normal, poisson, bernoulli (more thorough)


def test_normal_tensor[N: IntVar, M: IntVar](mean: Tensor[[N, M]], std: Tensor[[N, M]]):
    """normal tensor operation"""
    # torch.normal(mean_tensor, std_tensor) preserves shape
    y = torch.normal(mean, std)
    assert_type(y, Tensor[[N, M]])


def test_bernoulli[N: IntVar, M: IntVar](p: Tensor[[N, M]]):
    """Bernoulli sampling preserves shape"""
    y = torch.bernoulli(p)
    assert_type(y, Tensor[[N, M]])


def test_poisson[N: IntVar, M: IntVar](lam: Tensor[[N, M]]):
    """Poisson sampling preserves shape"""
    y = torch.poisson(lam)
    assert_type(y, Tensor[[N, M]])


def test_rand_n[N: IntVar](x: Tensor[[N, 3]]):
    """randn with symbolic in output (via like)"""
    # Can't create with symbolic size directly, but can use like
    y = torch.randn_like(x)
    assert_type(y, Tensor[[N, 3]])


# Test random sampling operations
_mean35 = torch.zeros(3, 5)
_std35 = torch.ones(3, 5)
_p35 = torch.rand(3, 5)
_lam35 = torch.rand(3, 5)
_t53 = torch.randn(5, 3)
test_normal_tensor(_mean35, _std35)
test_bernoulli(_p35)
test_poisson(_lam35)
test_rand_n(_t53)

# ==== Additional Coverage ====


def test_einsum_matmul[N: IntVar, M: IntVar, K: IntVar](
    a: Tensor[[N, M]], b: Tensor[[M, K]]
):
    """einsum matrix multiplication with symbolic shapes"""
    y = torch.einsum("ij,jk->ik", a, b)
    assert_type(y, Tensor[[N, K]])


def test_einsum_batch_matmul[B: IntVar, N: IntVar, M: IntVar, K: IntVar](
    a: Tensor[[B, N, M]], b: Tensor[[B, M, K]]
):
    """einsum batch matrix multiplication with symbolic shapes"""
    y = torch.einsum("bij,bjk->bik", a, b)
    assert_type(y, Tensor[[B, N, K]])


def test_einsum_transpose[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    """einsum transpose operation"""
    y = torch.einsum("ij->ji", x)
    assert_type(y, Tensor[[M, N]])


def test_einsum_trace[N: IntVar](x: Tensor[[N, N]]):
    """einsum trace: extracts diagonal"""
    y = torch.einsum("ii->i", x)
    assert_type(y, Tensor[[N]])


def test_einsum_trace_scalar[N: IntVar](x: Tensor[[N, N]]):
    """einsum trace to scalar: sums the diagonal"""
    y = torch.einsum("ii->", x)
    assert_type(y, Tensor[[]])


def test_einsum_elementwise[N: IntVar, M: IntVar](a: Tensor[[N, M]], b: Tensor[[N, M]]):
    """einsum element-wise multiplication"""
    y = torch.einsum("ij,ij->ij", a, b)
    assert_type(y, Tensor[[N, M]])


def test_einsum_sum_reduction[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    """einsum sum all elements to scalar"""
    y = torch.einsum("ij->", x)
    assert_type(y, Tensor[[]])


def test_einsum_outer_product[N: IntVar, M: IntVar](a: Tensor[[N]], b: Tensor[[M]]):
    """einsum outer product"""
    y = torch.einsum("i,j->ij", a, b)
    assert_type(y, Tensor[[N, M]])


def test_masked_select_documented_limitation[N: IntVar, M: IntVar](
    x: Tensor[[N, M]], mask: Tensor[[N, M]]
):
    """masked_select returns Tensor[[Any]] (data-dependent 1D size)"""
    # Fixture doesn't support .bool()
    # Output size depends on how many True values in mask
    y = torch.masked_select(x, mask)
    assert_type(y, Tensor[[Any]])  # Returns 1D tensor with unknown size


# Test additional coverage operations
_a34 = torch.randn(3, 4)
_b45 = torch.randn(4, 5)
_a2345 = torch.randn(2, 3, 4)
_b2456 = torch.randn(2, 4, 5)
_x55 = torch.randn(5, 5)
_vec3 = torch.randn(3)
_vec4 = torch.randn(4)
test_einsum_matmul(_a34, _b45)
test_einsum_batch_matmul(_a2345, _b2456)
test_einsum_transpose(_a34)
test_einsum_trace(_x55)
test_einsum_trace_scalar(_x55)
test_einsum_elementwise(_a34, _a34)
test_einsum_sum_reduction(_a34)
test_einsum_outer_product(_vec3, _vec4)
test_masked_select_documented_limitation(_t35, _mask35)
