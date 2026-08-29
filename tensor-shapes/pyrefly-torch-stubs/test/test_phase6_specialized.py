# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 6: Specialized operations tests (FFT, Loss, Padding, Random, Properties)
from typing import Any, assert_type, Literal

import torch
import torch.fft
import torch.nn
import torch.nn.functional as F
from shape_extensions import Int, IntTuple, IntVar
from torch import Tensor

# ==== Loss Functions ====
# Note: Loss functions approximate shape behavior (default returns scalar)


def test_mse_loss_reduced():
    """MSE loss with default reduction"""
    input: Tensor[[3, 4]] = torch.randn(3, 4)
    target: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.nn.functional.mse_loss(input, target)
    # Default reduction='mean' returns scalar
    assert_type(result, Tensor[[]])


def test_l1_loss():
    """L1 loss"""
    input: Tensor[[2, 5]] = torch.randn(2, 5)
    target: Tensor[[2, 5]] = torch.randn(2, 5)
    result = torch.nn.functional.l1_loss(input, target)
    # Default reduction returns scalar
    assert_type(result, Tensor[[]])


def test_cross_entropy():
    """Cross entropy loss"""
    input: Tensor[[3, 10]] = torch.randn(3, 10)  # 3 samples, 10 classes
    target: Tensor[[3]] = torch.randn(3)
    result = torch.nn.functional.cross_entropy(input, target)
    # Returns scalar
    assert_type(result, Tensor[[]])


def test_binary_cross_entropy():
    """Binary cross entropy"""
    input: Tensor[[4, 5]] = torch.randn(4, 5)
    target: Tensor[[4, 5]] = torch.randn(4, 5)
    result = torch.nn.functional.binary_cross_entropy(input, target)
    # Returns scalar
    assert_type(result, Tensor[[]])


def test_kl_div():
    """KL divergence"""
    input: Tensor[[2, 3]] = torch.randn(2, 3)
    target: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.nn.functional.kl_div(input, target)
    # Returns scalar
    assert_type(result, Tensor[[]])


def test_smooth_l1_loss():
    """Smooth L1 loss"""
    input: Tensor[[3, 4]] = torch.randn(3, 4)
    target: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.nn.functional.smooth_l1_loss(input, target)
    # Returns scalar
    assert_type(result, Tensor[[]])


def test_huber_loss():
    """Huber loss"""
    input: Tensor[[2, 5]] = torch.randn(2, 5)
    target: Tensor[[2, 5]] = torch.randn(2, 5)
    result = torch.nn.functional.huber_loss(input, target)
    # Returns scalar
    assert_type(result, Tensor[[]])


def test_elementwise_loss_smoke[Shape: IntTuple](
    input: Tensor[Shape], target: Tensor[Shape]
) -> None:
    """Elementwise losses score each element, so an unreduced result keeps the input
    shape whatever its rank."""
    assert_type(F.mse_loss(input, target, reduce=False), Tensor[Shape])
    assert_type(F.l1_loss(input, target, reduce=False), Tensor[Shape])
    assert_type(F.binary_cross_entropy(input, target, reduce=False), Tensor[Shape])
    assert_type(
        F.binary_cross_entropy_with_logits(input, target, reduce=False), Tensor[Shape]
    )
    assert_type(F.kl_div(input, target, reduce=False), Tensor[Shape])
    assert_type(F.smooth_l1_loss(input, target, reduce=False), Tensor[Shape])
    assert_type(F.huber_loss(input, target, reduction="none"), Tensor[Shape])
    assert_type(F.poisson_nll_loss(input, target, reduce=False), Tensor[Shape])
    assert_type(
        F.margin_ranking_loss(input, target, target, reduce=False), Tensor[Shape]
    )
    assert_type(F.hinge_embedding_loss(input, target, reduce=False), Tensor[Shape])


def test_unreduced_elementwise_losses_broadcast() -> None:
    input: Tensor[[2, 1]] = torch.randn(2, 1)
    target: Tensor[[2, 3]] = torch.randn(2, 3)

    assert_type(F.mse_loss(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.l1_loss(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.kl_div(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.smooth_l1_loss(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.huber_loss(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.poisson_nll_loss(input, target, reduction="none"), Tensor[[2, 3]])

    other: Tensor[[1, 3]] = torch.randn(1, 3)
    assert_type(
        F.margin_ranking_loss(input, other, target, reduction="none"), Tensor[[2, 3]]
    )
    assert_type(F.hinge_embedding_loss(input, target, reduction="none"), Tensor[[2, 3]])


def test_classification_loss_drops_class_dim(target: Tensor) -> None:
    """NLL and cross-entropy score `(N, C, *D)` down to `(N, *D)`."""
    two_d: Tensor[[3, 10]] = torch.randn(3, 10)
    four_d: Tensor[[3, 10, 8, 6]] = torch.randn(3, 10, 8, 6)
    unbatched: Tensor[[10]] = torch.randn(10)

    assert_type(F.cross_entropy(two_d, target, reduction="none"), Tensor[[3]])
    assert_type(F.nll_loss(two_d, target, reduce=False), Tensor[[3]])
    assert_type(F.cross_entropy(four_d, target, reduction="none"), Tensor[[3, 8, 6]])
    assert_type(F.nll_loss(four_d, target, reduce=False), Tensor[[3, 8, 6]])
    # An unbatched `(C,)` input has nothing left once the class dimension goes.
    assert_type(F.nll_loss(unbatched, target, reduction="none"), Tensor[[]])
    # Reducing collapses to a scalar regardless of rank.
    assert_type(F.cross_entropy(four_d, target), Tensor[[]])


def test_classification_loss_symbolic[N: IntVar, C: IntVar](
    input: Tensor[[N, C]], target: Tensor
) -> None:
    assert_type(F.cross_entropy(input, target, reduction="none"), Tensor[[N]])
    assert_type(F.nll_loss(input, target, reduce=False), Tensor[[N]])


def test_cosine_embedding_loss_ranks() -> None:
    vector: Tensor[[10]] = torch.randn(10)
    scalar_target: Tensor[[]] = torch.randn(())
    two_d: Tensor[[3, 10]] = torch.randn(3, 10)
    batch_target: Tensor[[3]] = torch.randn(3)
    broadcast_input: Tensor[[1, 10]] = torch.randn(1, 10)
    singleton_target: Tensor[[1]] = torch.randn(1)

    assert_type(
        F.cosine_embedding_loss(vector, vector, scalar_target, reduction="none"),
        Tensor[[]],
    )
    assert_type(
        F.cosine_embedding_loss(two_d, two_d, batch_target, reduction="none"),
        Tensor[[3]],
    )
    assert_type(F.cosine_embedding_loss(two_d, two_d, batch_target), Tensor[[]])
    assert_type(
        F.cosine_embedding_loss(
            two_d, broadcast_input, singleton_target, reduction="none"
        ),
        Tensor[[3]],
    )


def test_triplet_margin_loss_drops_feature_dim() -> None:
    """Triplet-margin loss compares along the trailing dimension."""
    two_d: Tensor[[3, 10]] = torch.randn(3, 10)
    four_d: Tensor[[3, 10, 8, 6]] = torch.randn(3, 10, 8, 6)
    anchor: Tensor[[2, 1]] = torch.randn(2, 1)
    positive: Tensor[[2, 3]] = torch.randn(2, 3)
    negative: Tensor[[2, 4]] = torch.randn(2, 4)

    assert_type(F.triplet_margin_loss(two_d, two_d, two_d, reduce=False), Tensor[[3]])
    assert_type(
        F.triplet_margin_loss(four_d, four_d, four_d, reduction="none"),
        Tensor[[3, 10, 8]],
    )
    assert_type(
        F.triplet_margin_loss(anchor, positive, negative, reduction="none"), Tensor[[2]]
    )


def test_loss_first_parameter_keywords(target: Tensor) -> None:
    """The first parameter is spelled as PyTorch spells it, so keyword calls work."""
    input: Tensor[[2, 3]] = torch.randn(2, 3)
    elementwise_target: Tensor[[2, 3]] = torch.randn(2, 3)

    assert_type(
        F.l1_loss(input=input, target=elementwise_target, reduction="none"),
        Tensor[[2, 3]],
    )
    assert_type(
        F.mse_loss(input=input, target=elementwise_target, reduction="none"),
        Tensor[[2, 3]],
    )
    assert_type(
        F.huber_loss(input=input, target=elementwise_target, reduction="none"),
        Tensor[[2, 3]],
    )
    assert_type(F.kl_div(input=input, target=target, reduction="batchmean"), Tensor[[]])
    assert_type(
        F.cross_entropy(input=input, target=target, reduction="none"), Tensor[[2]]
    )
    cosine_target: Tensor[[2]] = torch.randn(2)
    assert_type(
        F.cosine_embedding_loss(
            input1=input, input2=input, target=cosine_target, reduction="none"
        ),
        Tensor[[2]],
    )
    assert_type(
        F.triplet_margin_loss(
            anchor=input, positive=input, negative=input, reduction="none"
        ),
        Tensor[[2]],
    )


def test_l1_loss_legacy_reduction_precedence(
    input: Tensor[[2, 3]],
    target: Tensor[[2, 3]],
    size_average: bool | None,
    reduce: bool | None,
    reduction: str,
) -> None:
    assert_type(F.l1_loss(input, target), Tensor[[]])
    assert_type(F.l1_loss(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.l1_loss(input, target, reduce=False), Tensor[[2, 3]])
    assert_type(
        F.l1_loss(input, target, size_average=False, reduce=False, reduction="sum"),
        Tensor[[2, 3]],
    )
    assert_type(
        F.l1_loss(input, target, size_average=False, reduce=True, reduction="none"),
        Tensor[[]],
    )
    assert_type(
        F.l1_loss(input, target, size_average=False, reduction="none"), Tensor[[]]
    )
    assert_type(
        F.l1_loss(input, target, size_average=True, reduction="none"), Tensor[[]]
    )
    assert_type(F.l1_loss(input, target, reduce=True, reduction="none"), Tensor[[]])
    assert_type(
        F.l1_loss(input, target, size_average=False, reduction="invalid"), Tensor[[]]
    )
    assert_type(
        F.l1_loss(input, target, reduce=False, reduction="invalid"), Tensor[[2, 3]]
    )
    assert_type(
        F.l1_loss(input, target, size_average=size_average, reduce=False),
        Tensor[[2, 3]],
    )
    assert_type(F.l1_loss(input, target, size_average=size_average), Tensor[IntTuple])
    assert_type(
        F.l1_loss(input, target, size_average=False, reduce=reduce), Tensor[IntTuple]
    )
    assert_type(F.l1_loss(input, target, reduction=reduction), Tensor[IntTuple])
    assert_type(F.kl_div(input, target, reduction="batchmean"), Tensor[[]])
    assert_type(F.huber_loss(input, target, reduction="none"), Tensor[[2, 3]])
    assert_type(F.huber_loss(input, target, reduction="mean"), Tensor[[]])
    assert_type(F.huber_loss(input, target, reduction="sum"), Tensor[[]])
    assert_type(F.huber_loss(input, target, reduction=reduction), Tensor[IntTuple])
    assert_type(
        F.mse_loss(input=input, target=target, reduction="none"),
        Tensor[[2, 3]],
    )


def test_loss_gradual_input(input: Tensor[IntTuple], target: Tensor) -> None:
    assert_type(F.l1_loss(input, target, reduction="none"), Tensor[IntTuple])
    assert_type(F.l1_loss(input, target), Tensor[[]])
    # An unknown rank hides the class and feature dimensions.
    assert_type(F.cross_entropy(input, target, reduction="none"), Tensor[IntTuple])
    assert_type(
        F.cosine_embedding_loss(input, target, target, reduce=False), Tensor[IntTuple]
    )


def test_loss_nonliteral_flags_stay_gradual(
    input: Tensor[[2, 3]], target: Tensor, reduction: str, reduce: bool | None
) -> None:
    """A Flag value that is not a literal leaves the result gradual for every family."""
    assert_type(F.cross_entropy(input, target, reduction=reduction), Tensor[IntTuple])
    assert_type(F.nll_loss(input, target, reduce=reduce), Tensor[IntTuple])
    assert_type(
        F.cosine_embedding_loss(input, target, target, reduction=reduction),
        Tensor[IntTuple],
    )
    assert_type(
        F.triplet_margin_loss(input, target, target, reduce=reduce), Tensor[IntTuple]
    )


# ==== Padding Operations ====
# Note: Simplified implementation - pad parameter handling is complex


def test_pad_1d():
    """Pad 1D tensor"""
    x: Tensor[[10]] = torch.randn(10)
    # Pad operations type check but shape inference needs pad parameter
    _ = torch.nn.functional.pad(x, (2, 3))


def test_pad_2d():
    """Pad 2D tensor"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    _ = torch.nn.functional.pad(x, (1, 1, 2, 2))


def test_pad_3d():
    """Pad 3D tensor"""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    _ = torch.nn.functional.pad(x, (1, 1))


# ==== FFT Operations ====


def test_fft_1d():
    """1D FFT"""
    x: Tensor[[10]] = torch.randn(10)
    result = torch.fft.fft(x)
    # Preserves shape
    assert_type(result, Tensor[[10]])


def test_ifft_1d():
    """1D inverse FFT"""
    x: Tensor[[8]] = torch.randn(8)
    result = torch.fft.ifft(x)
    # Preserves shape
    assert_type(result, Tensor[[8]])


def test_fft2_2d():
    """2D FFT"""
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    result = torch.fft.fft2(x)
    # Preserves shape
    assert_type(result, Tensor[[4, 5]])


def test_fftn_3d():
    """ND FFT"""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = torch.fft.fftn(x)
    # Preserves shape
    assert_type(result, Tensor[[2, 3, 4]])


def test_rfft():
    """Real FFT (dimension changes)"""
    x: Tensor[[10]] = torch.randn(10)
    result = torch.fft.rfft(x)
    # Real FFT: [10] -> [6] (n//2 + 1 = 10//2 + 1 = 6)
    assert_type(result, Tensor[[6]])


def test_rfft_2d():
    """Real FFT on 2D tensor"""
    x: Tensor[[4, 8]] = torch.randn(4, 8)
    result = torch.fft.rfft(x, dim=1)
    # Real FFT along dim 1: [4, 8] -> [4, 5] (8//2 + 1 = 5)
    assert_type(result, Tensor[[4, 5]])


def test_irfft():
    """Inverse real FFT (dimension changes)"""
    x: Tensor[[6]] = torch.randn(6)
    result = torch.fft.irfft(x)
    # Inverse real FFT: [6] -> [10] (2*(n-1) = 2*(6-1) = 10)
    assert_type(result, Tensor[[10]])


def test_real_fft_axes_and_lengths():
    x: Tensor[[4, 10, 6]] = torch.randn(4, 10, 6)
    assert_type(torch.fft.rfft(x, dim=-2), Tensor[[4, 6, 6]])
    assert_type(torch.fft.rfft(x, n=8, dim=0), Tensor[[5, 10, 6]])
    assert_type(torch.fft.irfft(x, n=12, dim=1), Tensor[[4, 12, 6]])
    assert_type(torch.fft.hfft(x, dim=0), Tensor[[6, 10, 6]])
    assert_type(torch.fft.ihfft(x, dim=1), Tensor[[4, 6, 6]])


def test_real_fft_symbolic_n[N: IntVar](x: Tensor[[3, 7]], n: Int[N]):
    assert_type(torch.fft.rfft(x, n=n, dim=0), Tensor[[N // 2 + 1, 7]])
    assert_type(torch.fft.irfft(x, n=n, dim=-1), Tensor[[3, N]])
    assert_type(torch.fft.hfft(x, n=n, dim=-1), Tensor[[3, N]])


def test_real_fft_known_shape_gradual_n(
    x: Tensor[[4, 10, 6]], n: int, optional_n: int | None
) -> None:
    # Forward arithmetic and inverse transforms both preserve gradual dimensions.
    assert_type(torch.fft.rfft(x, n=n, dim=0), Tensor[[int, 10, 6]])
    assert_type(torch.fft.ihfft(x, n=n, dim=0), Tensor[[int, 10, 6]])
    assert_type(torch.fft.irfft(x, n=n, dim=1), Tensor[[4, int, 6]])
    assert_type(torch.fft.hfft(x, n=n, dim=1), Tensor[[4, int, 6]])
    assert_type(
        torch.fft.hfft(x, n=optional_n, dim=1),
        Tensor[[4, 18, 6]] | Tensor[[4, int, 6]],
    )
    assert_type(
        torch.fft.ihfft(x, n=optional_n, dim=0),
        Tensor[[3, 10, 6]] | Tensor[[int, 10, 6]],
    )


def test_real_fft_gradual_dim(x: Tensor[[4, 10, 6]], dim: int) -> None:
    assert_type(torch.fft.rfft(x, dim=dim), Tensor)
    assert_type(torch.fft.irfft(x, n=12, dim=dim), Tensor)


def test_real_fft_any(x: Tensor[[4, 10, 6]], value: Any) -> None:
    assert_type(torch.fft.rfft(x, n=value), Tensor)
    assert_type(torch.fft.irfft(x, dim=value), Tensor)


def test_real_fft_gradual(x: Tensor, n: int):
    assert_type(torch.fft.rfft(x), Tensor)
    assert_type(torch.fft.rfft(x, dim=0), Tensor)
    assert_type(torch.fft.irfft(x, n=n), Tensor)


def test_fftshift():
    """FFT shift"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.fft.fftshift(x)
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_ifftshift():
    """Inverse FFT shift"""
    x: Tensor[[5, 6]] = torch.randn(5, 6)
    result = torch.fft.ifftshift(x)
    # Preserves shape
    assert_type(result, Tensor[[5, 6]])


# ==== Random Sampling Operations ====


def test_bernoulli():
    """Bernoulli sampling"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.bernoulli(x)
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_bernoulli_method():
    """Bernoulli sampling as method"""
    x: Tensor[[2, 5]] = torch.randn(2, 5)
    result = x.bernoulli()
    # Preserves shape
    assert_type(result, Tensor[[2, 5]])


def test_bernoulli_inplace():
    """Bernoulli sampling in-place"""
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    result = x.bernoulli_()
    # Preserves shape
    assert_type(result, Tensor[[4, 3]])


def test_multinomial_1d():
    """Multinomial sampling from 1D"""
    x: Tensor[[5]] = torch.randn(5)
    # Note: num_samples is positional, meta-shape may not receive it as kwarg
    _ = torch.multinomial(x, 3)


def test_multinomial_2d():
    """Multinomial sampling from 2D"""
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    _ = torch.multinomial(x, 3)


def test_multinomial_method():
    """Multinomial as method"""
    x: Tensor[[3, 10]] = torch.randn(3, 10)
    _ = x.multinomial(5)


def test_normal_inplace():
    """Normal distribution sampling in-place"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = x.normal_()
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_poisson():
    """Poisson sampling"""
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.poisson(x)
    # Preserves shape
    assert_type(result, Tensor[[2, 3]])


def test_random_inplace():
    """Random integers in-place"""
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    result = x.random_()
    # Preserves shape
    assert_type(result, Tensor[[3, 3]])


def test_uniform_inplace():
    """Uniform distribution in-place"""
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    result = x.uniform_()
    # Preserves shape
    assert_type(result, Tensor[[4, 5]])


# ==== Tensor Property Operations ====


def test_numel():
    """Number of elements"""
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    result = torch.numel(x)
    # Returns int (symbolic multiplication of dimensions)
    assert_type(result, Literal[60])


# ==== Tier 3: torch.normal Overloads ====


def test_normal_tensor_tensor():
    """Normal with both tensor parameters"""
    mean: Tensor[[3, 4]] = torch.randn(3, 4)
    std: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.normal(mean, std)
    assert_type(result, Tensor[[3, 4]])


def test_normal_tensor_tensor_mean_shape():
    mean: Tensor[[2, 3]] = torch.randn(2, 3)
    std: Tensor[[6]] = torch.randn(6)
    assert_type(torch.normal(mean, std), Tensor[[2, 3]])


def test_normal_tensor_scalar():
    """Normal with tensor mean, scalar std"""
    mean: Tensor[[2, 5]] = torch.randn(2, 5)
    result = torch.normal(mean, 0.5)
    assert_type(result, Tensor[[2, 5]])


def test_normal_scalar_tensor():
    """Normal with scalar mean, tensor std"""
    std: Tensor[[4, 3]] = torch.randn(4, 3)
    result = torch.normal(0.0, std)
    assert_type(result, Tensor[[4, 3]])


def test_normal_scalar_scalar_size():
    """Normal with scalar mean/std and size parameter"""
    result = torch.normal(0.0, 1.0, size=(3, 4))
    assert_type(result, Tensor[[3, 4]])


def test_normal_scalar_scalar_shape[N: IntVar](n: Int[N], plain: int):
    assert_type(torch.normal(0.0, 1.0, size=()), Tensor[[]])
    assert_type(torch.normal(0.0, 1.0, size=(n, plain)), Tensor[[N, int]])
