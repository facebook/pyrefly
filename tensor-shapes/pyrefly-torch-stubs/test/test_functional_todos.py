# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_adaptive_max_pool_with_indices_shapes() -> None:
    sequence = torch.randn((2, 3, 12))
    values, indices = F.adaptive_max_pool1d_with_indices(sequence, (5,))
    assert_shape(values.shape, (2, 3, 5))
    assert_shape(indices.shape, (2, 3, 5))

    image = torch.randn((2, 3, 8, 10))
    values, indices = F.adaptive_max_pool2d_with_indices(image, (4, 6))
    assert_shape(values.shape, (2, 3, 4, 6))
    assert_shape(indices.shape, (2, 3, 4, 6))
    values, indices = F.adaptive_max_pool2d_with_indices(image, (4, None))
    assert tuple(values.shape) == (2, 3, 4, 10)
    assert tuple(indices.shape) == (2, 3, 4, 10)

    volume = torch.randn((2, 3, 8, 10, 12))
    values, indices = F.adaptive_max_pool3d_with_indices(volume, (2, 3, 4))
    assert_shape(values.shape, (2, 3, 2, 3, 4))
    assert_shape(indices.shape, (2, 3, 2, 3, 4))


def test_max_pool_with_indices_shapes() -> None:
    sequence = torch.randn((2, 3, 12))
    values, indices = F.max_pool1d_with_indices(sequence, 2)
    assert_shape(values.shape, (2, 3, 6))
    assert_shape(indices.shape, (2, 3, 6))

    image = torch.randn((2, 3, 8, 12))
    values, indices = F.max_pool2d_with_indices(
        image, (2, 3), stride=(2, 3), return_indices=False
    )
    assert_shape(values.shape, (2, 3, 4, 4))
    assert_shape(indices.shape, (2, 3, 4, 4))

    volume = torch.randn((2, 3, 8, 10, 12))
    values, indices = F.max_pool3d_with_indices(volume, (2, 2, 3), (2, 2, 3))
    assert_shape(values.shape, (2, 3, 4, 5, 4))
    assert_shape(indices.shape, (2, 3, 4, 5, 4))


def test_max_unpool_shapes() -> None:
    sequence = torch.randn((2, 3, 11))
    values, indices = F.max_pool1d_with_indices(sequence, 2)
    assert_shape(
        F.max_unpool1d(values, indices, 2, output_size=(11,)).shape, (2, 3, 11)
    )

    image = torch.randn((2, 3, 9, 11))
    values, indices = F.max_pool2d_with_indices(image, 2)
    assert_shape(
        F.max_unpool2d(values, indices, 2, output_size=(2, 3, 9, 11)).shape,
        (2, 3, 9, 11),
    )
    even_values, even_indices = F.max_pool2d_with_indices(torch.randn((2, 3, 8, 10)), 2)
    assert tuple(F.max_unpool2d(even_values, even_indices, 2).shape) == (2, 3, 8, 10)

    volume = torch.randn((2, 3, 9, 11, 13))
    values, indices = F.max_pool3d_with_indices(volume, 2)
    assert_shape(
        F.max_unpool3d(values, indices, 2, output_size=(9, 11, 13)).shape,
        (2, 3, 9, 11, 13),
    )


def test_lp_pool_shapes() -> None:
    sequence = torch.randn((2, 3, 12))
    assert_shape(F.lp_pool1d(sequence, 2, 3, stride=2, ceil_mode=True).shape, (2, 3, 6))

    image = torch.randn((2, 3, 8, 9))
    assert_shape(F.lp_pool2d(image, 2, (2, 3), stride=(2, 3)).shape, (2, 3, 4, 3))

    volume = torch.randn((2, 3, 8, 10, 12))
    assert_shape(
        F.lp_pool3d(volume, 2, (2, 2, 3), stride=(2, 2, 3)).shape,
        (2, 3, 4, 5, 4),
    )


def test_margin_loss_shapes() -> None:
    scores = torch.randn((2, 3))
    classes = torch.tensor([0, 2])
    labels = torch.tensor([[0, 2, -1], [1, -1, -1]])

    assert_shape(F.multi_margin_loss(scores, classes, reduction="none").shape, (2,))
    assert_shape(F.multi_margin_loss(scores, classes).shape, ())
    assert_shape(F.multilabel_margin_loss(scores, labels, reduction="none").shape, (2,))
    assert_shape(F.multilabel_margin_loss(scores, labels).shape, ())
    assert_shape(
        F.soft_margin_loss(scores, torch.ones((2, 3)), reduction="none").shape,
        (2, 3),
    )
    assert_shape(F.soft_margin_loss(scores, torch.ones((2, 3))).shape, ())


def test_fractional_max_pool_with_indices_shapes() -> None:
    image = torch.randn((2, 3, 8, 10))
    values, indices = F.fractional_max_pool2d_with_indices(image, 2, (4, 5))
    assert_shape(values.shape, (2, 3, 4, 5))
    assert_shape(indices.shape, (2, 3, 4, 5))
    values, indices = F.fractional_max_pool2d_with_indices(image[0], 2, 3)
    assert_shape(values.shape, (3, 3, 3))
    assert_shape(indices.shape, (3, 3, 3))
    values, indices = F.fractional_max_pool2d_with_indices(
        image, 2, output_ratio=(0.5, 0.5)
    )
    assert tuple(values.shape) == (2, 3, 4, 5)
    assert tuple(indices.shape) == (2, 3, 4, 5)

    volume = torch.randn((2, 3, 8, 10, 12))
    values, indices = F.fractional_max_pool3d_with_indices(volume, 2, (4, 5, 6))
    assert_shape(values.shape, (2, 3, 4, 5, 6))
    assert_shape(indices.shape, (2, 3, 4, 5, 6))


def test_fractional_max_pool_dispatch_shapes() -> None:
    image = torch.randn((2, 3, 8, 10))
    assert_shape(F.fractional_max_pool2d(image, 2, (4, 5)).shape, (2, 3, 4, 5))
    values, indices = F.fractional_max_pool2d(image, 2, 3, return_indices=True)
    assert_shape(values.shape, (2, 3, 3, 3))
    assert_shape(indices.shape, (2, 3, 3, 3))
    assert_shape(F.fractional_max_pool2d(image[0], 2, (4, 5)).shape, (3, 4, 5))

    volume = torch.randn((2, 3, 8, 10, 12))
    assert_shape(F.fractional_max_pool3d(volume, 2, 3).shape, (2, 3, 3, 3, 3))
    values, indices = F.fractional_max_pool3d(
        volume, 2, output_ratio=0.5, return_indices=True
    )
    assert tuple(values.shape) == (2, 3, 4, 5, 6)
    assert tuple(indices.shape) == (2, 3, 4, 5, 6)


def test_grid_sample_modes_and_int_pair() -> None:
    assert F.GRID_SAMPLE_INTERPOLATION_MODES["bilinear"] == 0
    assert F.GRID_SAMPLE_PADDING_MODES["reflection"] == 2
    F.assert_int_or_pair(2, "size", "invalid {}")
    F.assert_int_or_pair((2, 3), "size", "invalid {}")
    with assert_raises(AssertionError):
        F.assert_int_or_pair([2], "size", "invalid {}")


def test_gumbel_softmax_and_embedding_bag_shapes() -> None:
    logits = torch.randn((2, 3, 4))
    assert_shape(F.gumbel_softmax(logits).shape, (2, 3, 4))
    assert_shape(F.gumbel_softmax(logits, hard=True, dim=1).shape, (2, 3, 4))

    weight = torch.randn((8, 4))
    bags = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert_shape(F.embedding_bag(bags, weight).shape, (2, 4))
    flattened = torch.tensor([0, 1, 2, 3, 4, 5])
    assert_shape(
        F.embedding_bag(flattened, weight, torch.tensor([0, 3])).shape,
        (2, 4),
    )
    assert_shape(
        F.embedding_bag(
            flattened,
            weight,
            torch.tensor([0, 3, 6]),
            include_last_offset=True,
        ).shape,
        (2, 4),
    )


def test_affine_grid_and_fold_shapes() -> None:
    theta_2d = torch.randn((2, 2, 3))
    assert_shape(F.affine_grid(theta_2d, (2, 3, 4, 5), False).shape, (2, 4, 5, 2))
    assert tuple(F.affine_grid(theta_2d, [2, 3, 4, 5], False).shape) == (2, 4, 5, 2)
    with assert_raises(ValueError):
        F.affine_grid(  # E: affine_grid size must match theta rank
            theta_2d, (2, 3, 4, 5, 6), False
        )
    with assert_raises(ValueError):
        F.affine_grid(  # E: affine_grid size must match theta rank
            theta_2d, [2, 3, 4, 5, 6], False
        )

    theta_3d = torch.randn((2, 3, 4))
    assert_shape(
        F.affine_grid(theta_3d, (2, 3, 4, 5, 6), False).shape,
        (2, 4, 5, 6, 3),
    )
    with assert_raises(ValueError):
        F.affine_grid(  # E: affine_grid size must match theta rank
            theta_3d, (2, 3, 4, 5), False
        )

    columns = torch.randn((2, 12, 20))
    assert_shape(F.fold(columns, (8, 10), 2, stride=2).shape, (2, 3, 8, 10))
    assert tuple(F.fold(columns, [8, 10], [2, 2], stride=2).shape) == (2, 3, 8, 10)
    assert tuple(F.fold(columns, (8, 10), [2, 2], stride=2).shape) == (2, 3, 8, 10)
    assert_shape(
        F.fold(columns[0], (8, 10), (2, 2), stride=2).shape,
        (3, 8, 10),
    )
    with assert_raises(RuntimeError):
        F.fold(  # E: fold input channels must be divisible
            torch.randn((2, 5, 20)), (8, 10), 2, stride=2
        )
    square_columns = torch.randn((2, 12, 16))
    assert_shape(F.fold(square_columns, 8, 2, stride=2).shape, (2, 3, 8, 8))
    assert_shape(F.fold(square_columns[0], 8, 2, stride=2).shape, (3, 8, 8))


def test_unfold_channel_and_window_shapes() -> None:
    image = torch.randn((2, 3, 8, 10))
    columns = F.unfold(image, 2, stride=2)
    assert_shape(columns.shape[:2], (2, 12))
    assert tuple(columns.shape) == (2, 12, 20)

    columns = F.unfold(image, (2, 3), stride=(2, 3))
    assert_shape(columns.shape[:2], (2, 18))
    assert tuple(columns.shape) == (2, 18, 12)

    unbatched = F.unfold(image[0], 2, stride=2)
    assert_shape(unbatched.shape[:1], (12,))
    assert tuple(unbatched.shape) == (12, 20)


def test_ctc_loss_reductions() -> None:
    log_probs = F.log_softmax(torch.randn((4, 2, 3)), dim=-1)
    targets = torch.tensor([[1, 2], [1, 1]])
    input_lengths = [4, 4]
    target_lengths = [2, 2]

    assert_shape(
        F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, reduction="none"
        ).shape,
        (2,),
    )
    assert_shape(
        F.ctc_loss(log_probs, targets, input_lengths, target_lengths).shape, ()
    )
    assert_shape(
        F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, reduction="sum"
        ).shape,
        (),
    )
    assert_shape(
        F.ctc_loss(
            log_probs[:, 0, :],
            torch.tensor([1, 2]),
            torch.tensor(4),
            torch.tensor(2),
            reduction="none",
        ).shape,
        (),
    )


def test_gaussian_nll_loss_reductions() -> None:
    prediction = torch.ones((2, 1))
    target = torch.zeros((1, 3))

    assert_shape(
        F.gaussian_nll_loss(prediction, target, 1.0, reduction="none").shape,
        (2, 3),
    )
    assert_shape(F.gaussian_nll_loss(prediction, target, 1.0).shape, ())
    assert_shape(
        F.gaussian_nll_loss(
            prediction, target, torch.ones((2, 1)), reduction="sum"
        ).shape,
        (),
    )


def test_multilabel_soft_margin_loss_reductions() -> None:
    scores = torch.ones((2, 4, 3))
    labels = torch.zeros((2, 4, 3))
    weight = torch.ones(3)

    assert_shape(
        F.multilabel_soft_margin_loss(scores, labels, weight, reduction="none").shape,
        (2, 4),
    )
    assert_shape(F.multilabel_soft_margin_loss(scores, labels).shape, ())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_shape(
            F.multilabel_soft_margin_loss(
                scores, labels, weight, reduce=False, reduction="sum"
            ).shape,
            (2, 4),
        )
        assert_shape(
            F.multilabel_soft_margin_loss(
                scores,
                labels,
                weight,
                size_average=False,
                reduce=True,
                reduction="none",
            ).shape,
            (),
        )
    assert_shape(
        F.multilabel_soft_margin_loss(
            scores[0, 0], labels[0, 0], reduction="none"
        ).shape,
        (),
    )
    assert_shape(
        F.multilabel_soft_margin_loss(
            scores[:, 0, :], labels[:, 0, :], torch.ones((4, 2, 3)), reduction="none"
        ).shape,
        IntTuple,
        runtime=(4, 3),
    )


def test_triplet_margin_with_distance_loss_reductions() -> None:
    anchor = torch.ones((2, 4, 3))
    positive = torch.ones((2, 4, 3))
    negative = torch.zeros((2, 4, 3))

    assert_shape(
        F.triplet_margin_with_distance_loss(
            anchor, positive, negative, reduction="none"
        ).shape,
        (2, 4),
    )
    assert_shape(
        F.triplet_margin_with_distance_loss(anchor, positive, negative).shape, ()
    )
    assert_shape(
        F.triplet_margin_with_distance_loss(
            anchor, positive, negative, reduction="sum"
        ).shape,
        (),
    )
    assert_shape(
        F.triplet_margin_with_distance_loss(
            anchor[0, 0], positive[0, 0], negative[0, 0], reduction="none"
        ).shape,
        (),
    )
    assert_shape(
        F.triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=lambda x, y: (x - y).abs(),
            reduction="none",
        ).shape,
        IntTuple,
        runtime=(2, 4, 3),
    )


def test_deprecated_upsample_shapes() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sequence = torch.randn((2, 3, 4))
        assert_shape(F.upsample_nearest(sequence, size=8).shape, (2, 3, 8))

        image = torch.randn((2, 3, 4, 5))
        assert_shape(F.upsample_nearest(image, size=(8, 9)).shape, (2, 3, 8, 9))
        assert_shape(F.upsample_bilinear(image, size=(8, 9)).shape, (2, 3, 8, 9))
        assert_shape(F.upsample_bilinear(image, scale_factor=2).shape, (2, 3, 8, 10))
        assert tuple(F.upsample_bilinear(image, scale_factor=2.0).shape) == (
            2,
            3,
            8,
            10,
        )

        volume = torch.randn((2, 3, 4, 5, 6))
        assert_shape(
            F.upsample_nearest(volume, scale_factor=2).shape,
            (2, 3, 8, 10, 12),
        )


def test_grouped_mm_cpu_shapes() -> None:
    mat_a_2d = torch.ones((16, 32), dtype=torch.bfloat16)
    mat_b_2d = torch.ones((32, 48), dtype=torch.bfloat16)
    mat_a_3d = torch.ones((2, 16, 32), dtype=torch.bfloat16)
    mat_b_3d = torch.ones((2, 32, 48), dtype=torch.bfloat16)
    offs = torch.tensor([8, 15], dtype=torch.int32)

    assert_shape(F.grouped_mm(mat_a_3d, mat_b_3d).shape, (2, 16, 48))
    assert_shape(
        F.grouped_mm(mat_a_2d, mat_b_3d, offs=offs).shape,
        (16, 48),
    )
    assert_shape(
        F.grouped_mm(mat_a_3d, mat_b_2d, offs=offs).shape,
        (16, 48),
    )
    assert_shape(
        F.grouped_mm(mat_a_2d, mat_b_2d, offs=offs).shape,
        (2, 16, 48),
    )


def test_scaled_mm_cpu_shapes() -> None:
    # The installed Torch exposes this dtype, but the top-level stub does not yet.
    float8_dtype = getattr(torch, "float8_e4m3fn")  # noqa: B009
    mat_a = torch.randn((2, 3)).to(float8_dtype)
    mat_b = torch.randn((3, 4)).to(float8_dtype)
    scale = torch.ones((), dtype=torch.float32)
    recipe = F.ScalingType.TensorWise

    assert_shape(
        F.scaled_mm(
            mat_a,
            mat_b,
            scale,
            recipe,
            scale,
            recipe,
            swizzle_a=None,
            swizzle_b=None,
            bias=torch.ones((4,)),
            output_dtype=torch.float32,
            contraction_dim=(),
            use_fast_accum=False,
        ).shape,
        (2, 4),
    )
    assert_shape(
        F.scaled_mm(
            mat_a,
            mat_b,
            [scale],
            [recipe],
            [scale],
            [recipe],
            output_dtype=torch.float32,
        ).shape,
        (2, 4),
    )


def test_multi_head_attention_forward_shapes() -> None:
    query = torch.ones((3, 2, 8))
    key = torch.ones((5, 2, 8))
    in_proj_weight = torch.cat((torch.eye(8),) * 3)
    out_proj_weight = torch.ones((6, 8))

    output, averaged = F.multi_head_attention_forward(
        query,
        key,
        key,
        8,
        2,
        in_proj_weight,
        None,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        None,
        training=False,
    )
    assert_shape(output.shape, (3, 2, 6))
    assert averaged is not None
    assert averaged.shape == (2, 3, 5)

    output, per_head = F.multi_head_attention_forward(
        torch.ones((3, 8)),
        torch.ones((5, 8)),
        torch.ones((5, 8)),
        8,
        2,
        in_proj_weight,
        None,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        None,
        training=False,
        average_attn_weights=False,
    )
    assert_shape(output.shape, (3, 6))
    assert per_head is not None
    assert per_head.shape == (2, 3, 5)

    output, weights = F.multi_head_attention_forward(
        query,
        key,
        key,
        8,
        2,
        in_proj_weight,
        None,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        None,
        training=False,
        need_weights=False,
    )
    assert_shape(output.shape, (3, 2, 6))
    assert weights is None


if TYPE_CHECKING:  # noqa: C901

    def check_multi_head_attention_forward_shapes[
        Target: IntVar,
        Batch: IntVar,
        Embedding: IntVar,
        Source: IntVar,
        Output: IntVar,
    ](
        query: Tensor[[Target, Batch, Embedding]],
        key: Tensor[[Source, Batch, Embedding]],
        in_proj_weight: Tensor,
        out_proj_weight: Tensor[[Output, Embedding]],
        embedding_dim: int,
        heads: int,
    ) -> None:
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            key,
            embedding_dim,
            heads,
            in_proj_weight,
            None,
            None,
            None,
            False,
            0.0,
            out_proj_weight,
            None,
        )
        assert_type(output, Tensor[[Target, Batch, Output]])
        assert_type(weights, Tensor | None)

    def check_adaptive_max_pool_with_indices_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 12]],
        image: Tensor[[B, 3, 8, 10]],
        volume: Tensor[[B, 3, 8, 10, 12]],
        output_size: int,
    ) -> None:
        assert_type(
            F.adaptive_max_pool1d_with_indices(sequence, 5),
            tuple[Tensor[[B, 3, 5]], Tensor[[B, 3, 5]]],
        )
        assert_type(
            F.adaptive_max_pool2d_with_indices(image, (4, 6)),
            tuple[Tensor[[B, 3, 4, 6]], Tensor[[B, 3, 4, 6]]],
        )
        assert_type(
            F.adaptive_max_pool3d_with_indices(volume, 2),
            tuple[Tensor[[B, 3, 2, 2, 2]], Tensor[[B, 3, 2, 2, 2]]],
        )
        assert_type(
            F.adaptive_max_pool2d_with_indices(image, output_size),
            tuple[Tensor[[B, 3, int, int]], Tensor[[B, 3, int, int]]],
        )
        assert_type(
            F.adaptive_max_pool2d_with_indices(image, (4, None)),
            tuple[Tensor[[B, 3, int, int]], Tensor[[B, 3, int, int]]],
        )

    def check_max_pool_with_indices_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 12]],
        image: Tensor[[B, 3, 8, 12]],
        volume: Tensor[[B, 3, 8, 10, 12]],
    ) -> None:
        assert_type(
            F.max_pool1d_with_indices(sequence, 2),
            tuple[Tensor[[B, 3, 6]], Tensor[[B, 3, 6]]],
        )
        assert_type(
            F.max_pool2d_with_indices(image, (2, 3), stride=(2, 3)),
            tuple[Tensor[[B, 3, 4, 4]], Tensor[[B, 3, 4, 4]]],
        )
        assert_type(
            F.max_pool3d_with_indices(volume, (2, 2, 3), (2, 2, 3)),
            tuple[Tensor[[B, 3, 4, 5, 4]], Tensor[[B, 3, 4, 5, 4]]],
        )

    def check_max_unpool_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 5]],
        sequence_indices: Tensor[[B, 3, 5]],
        image: Tensor[[B, 3, 4, 5]],
        image_indices: Tensor[[B, 3, 4, 5]],
        volume: Tensor[[B, 3, 4, 5, 6]],
        volume_indices: Tensor[[B, 3, 4, 5, 6]],
        output_size: tuple[int, ...],
    ) -> None:
        assert_type(
            F.max_unpool1d(sequence, sequence_indices, 2, output_size=(11,)),
            Tensor[[B, 3, 11]],
        )
        assert_type(
            F.max_unpool2d(image, image_indices, 2, output_size=(B, 3, 9, 11)),
            Tensor[[B, 3, 9, 11]],
        )
        assert_type(
            F.max_unpool3d(volume, volume_indices, 2, output_size=(9, 11, 13)),
            Tensor[[B, 3, 9, 11, 13]],
        )
        assert_type(
            F.max_unpool2d(image, image_indices, 2),
            Tensor[[B, 3, int, int]],
        )
        assert_type(
            F.max_unpool3d(volume, volume_indices, 2, output_size=output_size),
            Tensor[[B, 3, int, int, int]],
        )

    def check_lp_pool_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 12]],
        image: Tensor[[B, 3, 8, 9]],
        volume: Tensor[[B, 3, 8, 10, 12]],
    ) -> None:
        assert_type(
            F.lp_pool1d(sequence, 2, 3, stride=2, ceil_mode=True),
            Tensor[[B, 3, 6]],
        )
        assert_type(
            F.lp_pool2d(image, 2, (2, 3), stride=(2, 3)),
            Tensor[[B, 3, 4, 3]],
        )
        assert_type(
            F.lp_pool3d(volume, 2, (2, 2, 3), stride=(2, 2, 3)),
            Tensor[[B, 3, 4, 5, 4]],
        )

    def check_margin_loss_shapes[B: IntVar, C: IntVar](
        scores: Tensor[[B, C]],
        classes: Tensor[[B]],
        labels: Tensor[[B, C]],
    ) -> None:
        assert_type(F.multi_margin_loss(scores, classes, reduction="none"), Tensor[[B]])
        assert_type(F.multi_margin_loss(scores, classes), Tensor[[]])
        assert_type(
            F.multilabel_margin_loss(scores, labels, reduction="none"), Tensor[[B]]
        )
        assert_type(F.multilabel_margin_loss(scores, labels), Tensor[[]])
        assert_type(
            F.soft_margin_loss(scores, labels, reduction="none"), Tensor[[B, C]]
        )
        assert_type(F.soft_margin_loss(scores, labels), Tensor[[]])

    def check_fractional_max_pool_with_indices_shapes[B: IntVar](
        image: Tensor[[B, 3, 8, 10]],
        volume: Tensor[[B, 3, 8, 10, 12]],
        size: tuple[int, int],
        depth: int,
    ) -> None:
        assert_type(
            F.fractional_max_pool2d_with_indices(image, 2, (4, 5)),
            tuple[Tensor[[B, 3, 4, 5]], Tensor[[B, 3, 4, 5]]],
        )
        assert_type(
            F.fractional_max_pool2d_with_indices(image, 2, 3),
            tuple[Tensor[[B, 3, 3, 3]], Tensor[[B, 3, 3, 3]]],
        )
        assert_type(
            F.fractional_max_pool3d_with_indices(volume, 2, (4, 5, 6)),
            tuple[Tensor[[B, 3, 4, 5, 6]], Tensor[[B, 3, 4, 5, 6]]],
        )
        assert_type(
            F.fractional_max_pool3d_with_indices(volume, 2, output_ratio=0.5),
            tuple[Tensor[[B, 3, int, int, int]], Tensor[[B, 3, int, int, int]]],
        )
        assert_type(
            F.fractional_max_pool2d_with_indices(image, 2, size),
            tuple[Tensor[[B, 3, int, int]], Tensor[[B, 3, int, int]]],
        )
        assert_type(
            F.fractional_max_pool3d_with_indices(volume[0], 2, depth),
            tuple[Tensor[[3, int, int, int]], Tensor[[3, int, int, int]]],
        )

    def check_fractional_max_pool_dispatch_shapes[B: IntVar](
        image: Tensor[[B, 3, 8, 10]],
        volume: Tensor[[B, 3, 8, 10, 12]],
        return_indices: bool,
        size: tuple[int, int],
        depth: int,
    ) -> None:
        assert_type(F.fractional_max_pool2d(image, 2, (4, 5)), Tensor[[B, 3, 4, 5]])
        assert_type(
            F.fractional_max_pool2d(image, 2, (4, 5), return_indices=True),
            tuple[Tensor[[B, 3, 4, 5]], Tensor[[B, 3, 4, 5]]],
        )
        assert_type(
            F.fractional_max_pool2d(image, 2, output_ratio=(0.5, 0.5)),
            Tensor[[B, 3, int, int]],
        )
        assert_type(F.fractional_max_pool3d(volume, 2, 3), Tensor[[B, 3, 3, 3, 3]])
        assert_type(
            F.fractional_max_pool3d(volume, 2, (4, 5, 6), return_indices=True),
            tuple[Tensor[[B, 3, 4, 5, 6]], Tensor[[B, 3, 4, 5, 6]]],
        )
        assert_type(
            F.fractional_max_pool3d(
                volume, 2, output_ratio=0.5, return_indices=return_indices
            ),
            Tensor[[B, 3, int, int, int]]
            | tuple[Tensor[[B, 3, int, int, int]], Tensor[[B, 3, int, int, int]]],
        )
        assert_type(
            F.fractional_max_pool2d(image, 2, size, return_indices=return_indices),
            Tensor[[B, 3, int, int]]
            | tuple[Tensor[[B, 3, int, int]], Tensor[[B, 3, int, int]]],
        )
        assert_type(
            F.fractional_max_pool3d(volume[0], 2, depth),
            Tensor[[3, int, int, int]],
        )

    def check_grid_sample_modes_and_int_pair() -> None:
        assert_type(F.GRID_SAMPLE_INTERPOLATION_MODES, dict[str, int])
        assert_type(F.GRID_SAMPLE_PADDING_MODES, dict[str, int])
        assert_type(F.assert_int_or_pair(2, "size", "invalid {}"), None)

    def check_gumbel_softmax_and_embedding_bag_shapes[B: IntVar, N: IntVar, D: IntVar](
        logits: Tensor[[B, N, D]],
        bags: Tensor[[B, N]],
        flat: Tensor[[N]],
        offsets: Tensor[[B]],
        weight: Tensor[[8, D]],
        include_last_offset: bool,
    ) -> None:
        assert_type(F.gumbel_softmax(logits, hard=True), Tensor[[B, N, D]])
        assert_type(F.embedding_bag(bags, weight), Tensor[[B, D]])
        assert_type(F.embedding_bag(flat, weight, offsets), Tensor[[B, D]])
        assert_type(
            F.embedding_bag(flat, weight, offsets, include_last_offset=True),
            Tensor[[B - 1, D]],
        )
        assert_type(
            F.embedding_bag(
                flat, weight, offsets, include_last_offset=include_last_offset
            ),
            Tensor[[int, D]],
        )

    def check_affine_grid_and_fold_shapes[B: IntVar](
        theta_2d: Tensor[[B, 2, 3]],
        theta_3d: Tensor[[B, 3, 4]],
        columns: Tensor[[B, 12, 20]],
        square_columns: Tensor[[B, 12, 16]],
        batch: Int[B],
        size: list[int],
        kernel_size: int,
        kernel_list: list[int],
    ) -> None:
        assert_type(F.affine_grid(theta_2d, (batch, 3, 4, 5)), Tensor[[B, 4, 5, 2]])
        assert_type(F.affine_grid(theta_2d, [batch, 3, 4, 5]), Tensor[[B, 4, 5, 2]])
        assert_type(
            F.affine_grid(theta_3d, (batch, 3, 4, 5, 6)), Tensor[[B, 4, 5, 6, 3]]
        )
        assert_type(
            F.affine_grid(theta_3d, [batch, 3, 4, 5, 6]), Tensor[[B, 4, 5, 6, 3]]
        )
        assert_type(F.affine_grid(theta_2d, size), Tensor[[B, int, int, 2]])
        assert_type(F.fold(columns, (8, 10), 2, stride=2), Tensor[[B, 3, 8, 10]])
        assert_type(
            F.fold(columns, [8, 10], [2, 2], stride=2),
            Tensor[[B, 3, 8, 10]],
        )
        assert_type(
            F.fold(columns, (8, 10), [2, 2], stride=2),
            Tensor[[B, 3, 8, 10]],
        )
        assert_type(F.fold(columns, [8, 10], 2, stride=2), Tensor[[B, 3, 8, 10]])
        assert_type(F.fold(columns, size, 2, stride=2), Tensor[[B, int, int, int]])
        assert_type(
            F.fold(columns, (8, 10), kernel_list, stride=2),
            Tensor[[B, int, 8, 10]],
        )
        assert_type(F.fold(columns[0], (8, 10), (2, 2), stride=2), Tensor[[3, 8, 10]])
        assert_type(
            F.fold(columns, (8, 10), kernel_size, stride=2),
            Tensor[[B, int, 8, 10]],
        )
        assert_type(
            F.fold(columns[0], (8, 10), kernel_size, stride=2),
            Tensor[[int, 8, 10]],
        )
        assert_type(F.fold(square_columns, 8, 2, stride=2), Tensor[[B, 3, 8, 8]])
        assert_type(
            F.fold(square_columns, 8, kernel_size, stride=2),
            Tensor[[B, int, 8, 8]],
        )
        assert_type(
            F.fold(square_columns[0], 8, kernel_size, stride=2),
            Tensor[[int, 8, 8]],
        )

    def check_unfold_shapes[B: IntVar](
        image: Tensor[[B, 3, 8, 10]], kernel_size: int
    ) -> None:
        assert_type(F.unfold(image, 2, stride=2), Tensor[[B, 12, int]])
        assert_type(F.unfold(image, (2, 3), stride=(2, 3)), Tensor[[B, 18, int]])
        assert_type(F.unfold(image[0], 2, stride=2), Tensor[[12, int]])
        assert_type(F.unfold(image, kernel_size), Tensor[[B, int, int]])

    def check_ctc_loss_reductions[B: IntVar, T: IntVar, C: IntVar](
        log_probs: Tensor[[T, B, C]],
        targets: Tensor,
        input_lengths: Tensor[[B]],
        target_lengths: Tensor[[B]],
        reduction: str,
    ) -> None:
        assert_type(
            F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, reduction="none"
            ),
            Tensor[[B]],
        )
        assert_type(
            F.ctc_loss(log_probs, targets, input_lengths, target_lengths), Tensor[[]]
        )
        assert_type(
            F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, reduction=reduction
            ),
            Tensor[IntTuple],
        )
        assert_type(
            F.ctc_loss(
                log_probs[:, 0, :],
                targets,
                input_lengths[0],
                target_lengths[0],
                reduction="none",
            ),
            Tensor[[]],
        )
        F.ctc_loss(  # E: loss reduction must be 'none', 'mean', or 'sum'
            log_probs, targets, input_lengths, target_lengths, reduction="median"
        )
        F.ctc_loss(  # E: ctc_loss requires 2D or 3D log probabilities
            torch.ones((2, 3, 4, 5)), targets, input_lengths, target_lengths
        )

    def check_gaussian_nll_loss_reductions[B: IntVar, C: IntVar](
        prediction: Tensor[[B, 1]],
        target: Tensor[[1, C]],
        reduction: str,
    ) -> None:
        assert_type(
            F.gaussian_nll_loss(prediction, target, 1.0, reduction="none"),
            Tensor[[B, C]],
        )
        assert_type(F.gaussian_nll_loss(prediction, target, 1.0), Tensor[[]])
        assert_type(
            F.gaussian_nll_loss(prediction, target, 1.0, reduction=reduction),
            Tensor[IntTuple],
        )
        F.gaussian_nll_loss(  # E: loss reduction must be 'none', 'mean', or 'sum'
            prediction, target, 1.0, reduction="median"
        )

    def check_multilabel_soft_margin_loss_shapes[B: IntVar, C: IntVar](
        scores: Tensor[[B, 4, C]],
        labels: Tensor[[B, 4, C]],
        class_weights: Tensor[[C]],
        extra_weight: Tensor[[5, B, 4, C]],
        reduction: str,
    ) -> None:
        assert_type(
            F.multilabel_soft_margin_loss(
                scores, labels, class_weights, reduction="none"
            ),
            Tensor[[B, 4]],
        )
        assert_type(F.multilabel_soft_margin_loss(scores, labels), Tensor[[]])
        assert_type(
            F.multilabel_soft_margin_loss(
                scores, labels, reduce=False, reduction="sum"
            ),
            Tensor[[B, 4]],
        )
        assert_type(
            F.multilabel_soft_margin_loss(
                scores, labels, class_weights, reduction=reduction
            ),
            Tensor[IntTuple],
        )
        assert_type(
            F.multilabel_soft_margin_loss(
                scores, labels, extra_weight, reduction="none"
            ),
            Tensor[IntTuple],
        )
        F.multilabel_soft_margin_loss(  # E: loss reduction must be 'none', 'mean', or 'sum'
            scores, labels, extra_weight, reduction="median"
        )

    def check_triplet_margin_with_distance_shapes[B: IntVar, C: IntVar](
        anchor: Tensor[[B, 4, C]],
        positive: Tensor[[B, 4, C]],
        negative: Tensor[[B, 4, C]],
        reduction: str,
    ) -> None:
        assert_type(
            F.triplet_margin_with_distance_loss(
                anchor, positive, negative, reduction="none"
            ),
            Tensor[[B, 4]],
        )
        assert_type(
            F.triplet_margin_with_distance_loss(anchor, positive, negative), Tensor[[]]
        )
        assert_type(
            F.triplet_margin_with_distance_loss(
                anchor, positive, negative, reduction=reduction
            ),
            Tensor[IntTuple],
        )
        assert_type(
            F.triplet_margin_with_distance_loss(
                anchor,
                positive,
                negative,
                distance_function=lambda x, y: (x - y).abs(),
                reduction="none",
            ),
            Tensor[IntTuple],
        )
        F.triplet_margin_with_distance_loss(  # E: loss reduction must be 'none', 'mean', or 'sum'
            anchor,
            positive,
            negative,
            distance_function=lambda x, y: (x - y).abs(),
            reduction="median",
        )

    def check_deprecated_upsample_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 4]],
        image: Tensor[[B, 3, 4, 5]],
        volume: Tensor[[B, 3, 4, 5, 6]],
    ) -> None:
        assert_type(F.upsample_nearest(sequence, size=8), Tensor[[B, 3, 8]])
        assert_type(F.upsample_nearest(image, size=(8, 9)), Tensor[[B, 3, 8, 9]])
        assert_type(
            F.upsample_nearest(image, scale_factor=2.0), Tensor[[B, 3, int, int]]
        )
        assert_type(F.upsample_bilinear(image, size=(8, 9)), Tensor[[B, 3, 8, 9]])
        assert_type(F.upsample_bilinear(image, scale_factor=2), Tensor[[B, 3, 8, 10]])
        assert_type(
            F.upsample_bilinear(image, scale_factor=2.0), Tensor[[B, 3, int, int]]
        )
        assert_type(
            F.upsample_nearest(volume, scale_factor=2),
            Tensor[[B, 3, 8, 10, 12]],
        )
        F.upsample_nearest(image)  # E: interpolate requires size or scale_factor
        F.upsample_nearest(image, size=8, scale_factor=2.0)  # E: No matching overload
        F.upsample_bilinear(image, size=8, scale_factor=2.0)  # E: No matching overload
        F.upsample_nearest(  # E: interpolate requires rank 3, 4, or 5
            torch.ones((3, 4)), size=8
        )
        F.upsample_nearest(  # E: interpolate size must match the spatial rank
            image, size=(2, 3, 4)
        )

    def check_grouped_mm_shapes[G: IntVar, M: IntVar, K: IntVar, L: IntVar, N: IntVar](
        mat_a_2d: Tensor[[M, K]],
        mat_b_2d: Tensor[[K, N]],
        mat_b_jagged: Tensor[[L, N]],
        mat_a_3d: Tensor[[G, M, K]],
        mat_b_3d: Tensor[[G, K, N]],
        offs: Tensor[[G]],
    ) -> None:
        assert_type(F.grouped_mm(mat_a_3d, mat_b_3d), Tensor[[G, M, N]])
        assert_type(F.grouped_mm(mat_a_2d, mat_b_3d, offs=offs), Tensor[[M, N]])
        assert_type(F.grouped_mm(mat_a_3d, mat_b_2d, offs=offs), Tensor[[M, N]])
        assert_type(F.grouped_mm(mat_a_2d, mat_b_jagged, offs=offs), Tensor[[G, M, N]])

    def check_scaled_mm_shapes[M: IntVar, K: IntVar, N: IntVar](
        mat_a: Tensor[[M, K]],
        mat_b: Tensor[[K, N]],
        scale: Tensor,
        recipe: F.ScalingType,
        bias: Tensor[[N]],
    ) -> None:
        assert_type(
            F.scaled_mm(
                mat_a,
                mat_b,
                scale,
                recipe,
                scale,
                recipe,
                swizzle_a=None,
                swizzle_b=None,
                bias=bias,
                output_dtype=torch.float32,
                contraction_dim=(),
                use_fast_accum=False,
            ),
            Tensor[[M, N]],
        )
        assert_type(
            F.scaled_mm(mat_a, mat_b, [scale], [recipe], [scale], [recipe]),
            Tensor[[M, N]],
        )

    def check_scaled_grouped_mm_shapes[
        G: IntVar,
        M: IntVar,
        K: IntVar,
        L: IntVar,
        N: IntVar,
    ](
        mat_a_2d: Tensor[[M, K]],
        mat_b_2d: Tensor[[K, N]],
        mat_b_jagged: Tensor[[L, N]],
        mat_a_3d: Tensor[[G, M, K]],
        mat_b_3d: Tensor[[G, K, N]],
        offs: Tensor[[G]],
        scale: Tensor,
        recipe: F.ScalingType,
        bias: Tensor[[N]],
    ) -> None:
        assert_type(
            F.scaled_grouped_mm(
                mat_a_3d,
                mat_b_3d,
                [scale],
                [recipe],
                [scale],
                [recipe],
                swizzle_a=None,
                swizzle_b=None,
                bias=bias,
                offs=None,
                output_dtype=torch.float32,
                contraction_dim=(),
                use_fast_accum=False,
            ),
            Tensor[[G, M, N]],
        )
        assert_type(
            F.scaled_grouped_mm(
                mat_a_2d, mat_b_3d, scale, recipe, scale, recipe, offs=offs
            ),
            Tensor[[M, N]],
        )
        assert_type(
            F.scaled_grouped_mm(
                mat_a_3d, mat_b_2d, scale, recipe, scale, recipe, offs=offs
            ),
            Tensor[[M, N]],
        )
        assert_type(
            F.scaled_grouped_mm(
                mat_a_2d, mat_b_jagged, scale, recipe, scale, recipe, offs=offs
            ),
            Tensor[[G, M, N]],
        )
