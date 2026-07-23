# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from shape_extensions import assert_shape


def test_arange_from_array_length() -> None:
    targets = np.zeros(5, dtype=np.intp)
    indices = np.arange(len(targets))

    assert_shape(indices, (5,))
    assert_type(indices.dtype, np.dtype[np.intp])
    assert indices.dtype == np.dtype(np.intp)


def test_paired_row_column_indexing() -> None:
    logits = np.ones((5, 3))
    targets = np.zeros(5, dtype=np.intp)
    selected: np.ndarray[[5], np.dtype[np.float64]] = logits[
        np.arange(len(targets)), targets
    ]

    assert_shape(selected, (5,))
    assert_type(selected.dtype, np.dtype[np.float64])


def test_paired_row_column_indexing_accepts_integer_dtypes() -> None:
    logits = np.ones((5, 3))
    int64_targets = np.zeros(5, dtype=np.int64)
    int32_targets = np.zeros(5, dtype=np.int32)
    selected_int64: np.ndarray[[5], np.dtype[np.float64]] = logits[
        np.arange(len(int64_targets)), int64_targets
    ]
    selected_int32: np.ndarray[[5], np.dtype[np.float64]] = logits[
        np.arange(len(int32_targets)), int32_targets
    ]

    assert_shape(selected_int64, (5,))
    assert_shape(selected_int32, (5,))


def test_paired_row_column_indexing_uses_index_shape() -> None:
    logits = np.ones((5, 3))
    rows = np.arange(2)
    columns = np.zeros(2, dtype=np.int64)
    selected: np.ndarray[[2], np.dtype[np.float64]] = logits[rows, columns]

    assert_shape(selected, (2,))


def test_none_indexing_for_nbody_broadcasting() -> None:
    positions = np.ones((5, 3))
    masses = np.ones(5)
    pairwise_deltas = positions[:, None, :] - positions[None, :, :]
    source_masses = masses[None, :, None]

    assert_shape(positions[:, None, :], (5, 1, 3))
    assert_shape(positions[None, :, :], (1, 5, 3))
    assert_shape(pairwise_deltas, (5, 5, 3))
    assert_shape(source_masses, (1, 5, 1))


def test_projecting_3d_slice_for_fill_diagonal() -> None:
    distances = np.expand_dims(np.ones((5, 5)), axis=-1)
    diagonal_view = distances[:, :, 0]
    result = np.fill_diagonal(diagonal_view, 1.0)

    assert_shape(diagonal_view, (5, 5))
    assert result is None


def test_fill_diagonal_rejects_vector() -> None:
    vector = np.ones(5)

    assert_shape(vector, (5,))
    try:
        # E: Tensor rank mismatch
        np.fill_diagonal(vector, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject a one-dimensional diagonal")
