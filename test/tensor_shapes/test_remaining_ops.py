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
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

# ==== FFT Variants (~10 operations) ====
# We already tested: fft, ifft, rfft ✅
# Testing: fft2, ifft2, fftn, ifftn, rfft2, irfft2, rfftn, irfftn, hfft, ihfft


def test_fft2[H, W](x: Tensor[2, H, W]):
    """2D FFT preserves spatial dimensions"""
    y = torch.fft.fft2(x)
    # Should preserve shape
    assert_type(y, Tensor[2, H, W])


def test_fftn[D1, D2, D3](x: Tensor[D1, D2, D3]):
    """N-dimensional FFT preserves dimensions"""
    y = torch.fft.fftn(x)
    assert_type(y, Tensor[D1, D2, D3])


def test_rfft2[H, W](x: Tensor[2, H, W]):
    """2D real FFT - last dimension changes"""
    y = torch.fft.rfft2(x)
    # Last dimension becomes W//2 + 1
    # For symbolic dims, may return shapeless
    assert_type(y, Tensor)


def test_rfftn[D1, D2, D3](x: Tensor[D1, D2, D3]):
    """N-dimensional real FFT"""
    y = torch.fft.rfftn(x)
    # Last dimension changes, may return shapeless
    assert_type(y, Tensor)


# Test FFT operations
test_fft2(torch.randn(2, 28, 28))
test_fftn(torch.randn(4, 8, 16))
test_rfft2(torch.randn(2, 28, 28))
test_rfftn(torch.randn(4, 8, 16))

# Note: ifft2, ifftn, irfft2, irfftn, hfft, ihfft require complex dtype
# which is not supported in our test fixtures

# ==== Loss Functions (~5 operations) ====
# We already tested: mse_loss ✅
# Testing: cross_entropy, nll_loss, binary_cross_entropy, kl_div, smooth_l1_loss


def test_cross_entropy[N, C](input: Tensor[N, C], target: Tensor[N]):
    """Cross entropy loss - returns scalar by default"""
    loss = F.cross_entropy(input, target)
    # Default reduction='mean' → scalar
    assert_type(loss, Tensor[()])


def test_cross_entropy_no_reduction[N, C](input: Tensor[N, C], target: Tensor[N]):
    """Cross entropy with no reduction - returns scalar with default reduction='mean'"""
    loss = F.cross_entropy(input, target, reduction="none")
    # Loss function with reduction="none" should return element-wise loss Tensor[N]
    # but meta-shape currently returns scalar for loss functions
    assert_type(loss, Tensor[()])  # Currently returns scalar


def test_nll_loss[N, C](input: Tensor[N, C], target: Tensor[N]):
    """Negative log likelihood loss"""
    loss = F.nll_loss(input, target)
    assert_type(loss, Tensor[()])


def test_binary_cross_entropy[N](input: Tensor[N], target: Tensor[N]):
    """Binary cross entropy loss"""
    loss = F.binary_cross_entropy(input, target)
    assert_type(loss, Tensor[()])


def test_kl_div[N, C](input: Tensor[N, C], target: Tensor[N, C]):
    """KL divergence loss"""
    loss = F.kl_div(input, target)
    assert_type(loss, Tensor[()])


def test_smooth_l1_loss[N](input: Tensor[N], target: Tensor[N]):
    """Smooth L1 loss (Huber loss)"""
    loss = F.smooth_l1_loss(input, target)
    assert_type(loss, Tensor[()])


# Test loss functions
_input_810 = torch.randn(8, 10)
_target_8 = torch.ones(8)
_target_zeros_8 = torch.zeros(8)
_input_prob_8 = torch.rand(8)
_input_8 = torch.randn(8)
test_cross_entropy(_input_810, _target_8)
test_cross_entropy_no_reduction(_input_810, _target_zeros_8)
test_nll_loss(_input_810, _target_zeros_8)
test_binary_cross_entropy(_input_prob_8, _target_8)
test_kl_div(_input_810, _input_810)
test_smooth_l1_loss(_input_8, _input_8)

# ==== Creation Operations with Symbolic (~7 operations) ====
# These take explicit sizes, but we can test with symbolic in the output


def test_zeros_like_symbolic[N, M](x: Tensor[N, M]):
    """zeros_like with symbolic dimensions"""
    y = torch.zeros_like(x)
    assert_type(y, Tensor[N, M])


def test_ones_like_symbolic[N, M](x: Tensor[N, M]):
    """ones_like with symbolic dimensions"""
    y = torch.ones_like(x)
    assert_type(y, Tensor[N, M])


def test_randn_like_symbolic[N, M](x: Tensor[N, M]):
    """randn_like with symbolic dimensions"""
    y = torch.randn_like(x)
    assert_type(y, Tensor[N, M])


def test_rand_like[N, M](x: Tensor[N, M]):
    """rand_like with symbolic dimensions"""
    y = torch.rand_like(x)
    assert_type(y, Tensor[N, M])


def test_full_like[N, M](x: Tensor[N, M]):
    """full_like with symbolic dimensions"""
    y = torch.full_like(x, 3.14)
    assert_type(y, Tensor[N, M])


def test_empty_like_symbolic[N, M](x: Tensor[N, M]):
    """empty_like with symbolic dimensions"""
    y = torch.empty_like(x)
    assert_type(y, Tensor[N, M])


# Test creation operations
_t35 = torch.randn(3, 5)
test_zeros_like_symbolic(_t35)
test_ones_like_symbolic(_t35)
test_randn_like_symbolic(_t35)
test_rand_like(_t35)
test_full_like(_t35)
test_empty_like_symbolic(_t35)

# Note: zeros, ones, randn, rand, empty, full, arange take literal sizes
# These are less relevant for symbolic dimension testing

# ==== Remaining Indexing Operations (~3 operations) ====
# We already tested: index_select, gather, scatter, masked_fill, take, index_add ✅
# Testing: index_copy, index_put, masked_scatter


def test_index_copy[N, M](x: Tensor[N, M], indices: Tensor[2], source: Tensor[2, M]):
    """index_copy preserves input shape"""
    y = x.index_copy(0, indices, source)
    assert_type(y, Tensor[N, M])


def test_masked_scatter[N, M](x: Tensor[N, M], mask: Tensor[N, M], source: Tensor[10]):
    """masked_scatter preserves shape"""
    # Fixture doesn't support .bool(), just verify operation doesn't crash
    # masked_scatter may not be in fixtures, just document
    # y = x.masked_scatter(mask, source)
    # expected: Tensor[N, M] = y
    pass


# Test indexing operations
_indices2 = torch.ones(2)
_source25 = torch.randn(2, 5)
_mask35 = torch.ones(3, 5)
_source10 = torch.randn(10)
test_index_copy(_t35, _indices2, _source25)
test_masked_scatter(_t35, _mask35, _source10)

# Note: index_put is less commonly used, similar to scatter

# ==== Specialized Operations (~5 operations) ====
# Testing: linspace, eye, tensordot, broadcast_to, unbind


def test_linspace_symbolic():
    """linspace creates 1D tensor of specified size"""
    y = torch.linspace(0, 1, 10)
    # Creates literal size, not symbolic
    assert_type(y, Tensor[10])


def test_eye_symbolic():
    """eye creates identity matrix"""
    y = torch.eye(5)
    assert_type(y, Tensor[5, 5])


def test_tensordot[N, M, K](a: Tensor[N, M, K], b: Tensor[K, 6]):
    """tensordot - generalized tensor contraction"""
    # Using dims=1 (simple int form) which is supported
    # dims=1 contracts last 1 dimension of a with first 1 dimension of b
    y = torch.tensordot(a, b, dims=1)
    # Contracts K dimension, result is [N, M, 6]
    assert_type(y, Tensor[N, M, 6])


def test_broadcast_to_symbolic[N](x: Tensor[N, 1]):
    """broadcast_to with symbolic dimensions"""
    y = torch.broadcast_to(x, (3, 5))
    # Broadcasts to literal target shape
    assert_type(y, Tensor[3, 5])


def test_unbind[N, M](x: Tensor[3, N, M]):
    """unbind splits tensor along dimension"""
    tensors = torch.unbind(x, dim=0)
    # Returns tuple[Tensor[N, M], ...] (unbounded tuple)
    # Each element has shape [N, M] (removed dim=0)
    assert_type(tensors, tuple[Tensor[N, M], ...])


# Test specialized operations
test_linspace_symbolic()
test_eye_symbolic()
_t345 = torch.randn(3, 4, 5)
_t56 = torch.randn(5, 6)
_t31 = torch.randn(3, 1)
_t345_b = torch.randn(3, 4, 5)
test_tensordot(_t345, _t56)
test_broadcast_to_symbolic(_t31)
test_unbind(_t345_b)

# ==== Random Sampling Operations (~5 operations) ====
# Testing: multinomial, normal, poisson, bernoulli (more thorough)


def test_multinomial[N](weights: Tensor[N, 10]):
    """multinomial sampling"""
    samples = torch.multinomial(weights, num_samples=5, replacement=True)
    # Returns [N, 5]
    assert_type(samples, Tensor[N, 5])


def test_normal_tensor[N, M](mean: Tensor[N, M], std: Tensor[N, M]):
    """normal tensor operation"""
    # torch.normal(mean_tensor, std_tensor) preserves shape
    y = torch.normal(mean, std)
    assert_type(y, Tensor[N, M])


def test_bernoulli[N, M](p: Tensor[N, M]):
    """Bernoulli sampling preserves shape"""
    y = torch.bernoulli(p)
    assert_type(y, Tensor[N, M])


def test_poisson[N, M](lam: Tensor[N, M]):
    """Poisson sampling preserves shape"""
    y = torch.poisson(lam)
    assert_type(y, Tensor[N, M])


def test_rand_n[N](x: Tensor[N, 3]):
    """randn with symbolic in output (via like)"""
    # Can't create with symbolic size directly, but can use like
    y = torch.randn_like(x)
    assert_type(y, Tensor[N, 3])


# Test random sampling operations
_weights210 = torch.rand(2, 10)
_mean35 = torch.zeros(3, 5)
_std35 = torch.ones(3, 5)
_p35 = torch.rand(3, 5)
_lam35 = torch.rand(3, 5)
_t53 = torch.randn(5, 3)
test_multinomial(_weights210)
test_normal_tensor(_mean35, _std35)
test_bernoulli(_p35)
test_poisson(_lam35)
test_rand_n(_t53)

# ==== Additional Coverage ====


def test_einsum_matmul[N, M, K](a: Tensor[N, M], b: Tensor[M, K]):
    """einsum - shape inference not yet working for symbolic shapes"""
    # einsum('ij,jk->ik') performs matrix multiplication
    # Contracts index 'j' (middle dimension), keeps 'i' and 'k'
    y = torch.einsum("ij,jk->ik", a, b)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_batch_matmul[B, N, M, K](a: Tensor[B, N, M], b: Tensor[B, M, K]):
    """einsum batch matrix multiplication - shape inference not yet working"""
    # 'bij,bjk->bik' performs batched matmul
    # Keeps batch dimension 'b', contracts 'j', keeps 'i' and 'k'
    y = torch.einsum("bij,bjk->bik", a, b)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_transpose[N, M](x: Tensor[N, M]):
    """einsum transpose operation - shape inference not yet working"""
    # 'ij->ji' swaps dimensions
    y = torch.einsum("ij->ji", x)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_trace[N](x: Tensor[N, N]):
    """einsum trace operation - shape inference not yet working"""
    # 'ii->i' extracts diagonal (same index appears twice)
    y = torch.einsum("ii->i", x)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_trace_scalar[N](x: Tensor[N, N]):
    """einsum trace to scalar - shape inference not yet working"""
    # 'ii->' sums the diagonal to scalar
    y = torch.einsum("ii->", x)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_elementwise[N, M](a: Tensor[N, M], b: Tensor[N, M]):
    """einsum element-wise multiplication - shape inference not yet working"""
    # 'ij,ij->ij' element-wise multiply (all indices preserved)
    y = torch.einsum("ij,ij->ij", a, b)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_sum_reduction[N, M](x: Tensor[N, M]):
    """einsum sum all elements to scalar - shape inference not yet working"""
    # 'ij->' sums all elements
    y = torch.einsum("ij->", x)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_einsum_outer_product[N, M](a: Tensor[N], b: Tensor[M]):
    """einsum outer product - shape inference not yet working"""
    # 'i,j->ij' creates outer product
    y = torch.einsum("i,j->ij", a, b)
    # TODO: Fix einsum binding to capture tensor operands
    assert_type(y, Tensor)  # Currently returns shapeless


def test_masked_select_documented_limitation[N, M](x: Tensor[N, M], mask: Tensor[N, M]):
    """masked_select returns Tensor[Any] (data-dependent 1D size)"""
    # Fixture doesn't support .bool()
    # Output size depends on how many True values in mask
    y = torch.masked_select(x, mask)
    assert_type(y, Tensor[Any])  # Returns 1D tensor with unknown size


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
