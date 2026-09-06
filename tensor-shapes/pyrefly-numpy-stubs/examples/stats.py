# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape, IntVar

N = IntVar("N")
P = IntVar("P")
K = IntVar("K")
C = IntVar("C")


def ordinary_least_squares(
    x: np.ndarray[[N, P]],
    y: np.ndarray[[N, 1]],
) -> np.ndarray[[P, 1]]:
    return np.linalg.solve(x.T @ x, x.T @ y)


def ridge_regression(
    x: np.ndarray[[N, P]],
    y: np.ndarray[[N, 1]],
    penalty_matrix: np.ndarray[[P, P]],
) -> np.ndarray[[P, 1]]:
    return np.linalg.solve(x.T @ x + penalty_matrix, x.T @ y)


def logistic_irls_step(
    x: np.ndarray[[N, P]],
    y: np.ndarray[[N, 1]],
    beta: np.ndarray[[P, 1]],
) -> np.ndarray[[P, 1]]:
    eta = x @ beta
    probability = 1.0 / (1.0 + np.exp(-eta))
    weight = probability * (1.0 - probability)
    adjusted_response = eta + (y - probability) / weight
    return np.linalg.solve(x.T @ (x * weight), x.T @ (weight * adjusted_response))


def pca_full_basis_projection(
    x: np.ndarray[[N, P]],
) -> np.ndarray[[N, P]]:
    x_centered = x - x.mean(axis=0)
    scatter = x_centered.T @ x_centered
    _u, _s, vt = np.linalg.svd(scatter, full_matrices=False)
    return x_centered @ vt.T


def nearest_centroid_labels(
    x: np.ndarray[[N, P]],
    centroids: np.ndarray[[K, P]],
) -> np.ndarray[[N]]:
    point_vectors = np.expand_dims(x, axis=-2)
    centroid_vectors = np.expand_dims(centroids, axis=-3)
    deltas = point_vectors - centroid_vectors
    squared_distances = np.sum(deltas**2, axis=-1)
    return np.argmin(squared_distances, axis=-1)


def cross_entropy_loss(
    logits: np.ndarray[[N, C]],
    targets: np.ndarray[[N], np.dtype[np.intp]],
) -> np.ndarray[[]]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(targets)), targets].mean()


def test_ordinary_least_squares() -> None:
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 1)

    assert_shape(x.shape, (5, 3))
    assert_shape(y.shape, (5, 1))
    assert_shape(ordinary_least_squares(x, y).shape, (3, 1))


def test_ridge_regression() -> None:
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 1)
    lam = 0.1
    penalty_matrix = lam * np.identity(3)

    assert_shape(x.shape, (5, 3))
    assert_shape(y.shape, (5, 1))
    assert_shape(penalty_matrix.shape, (3, 3))
    assert_shape(ridge_regression(x, y, penalty_matrix).shape, (3, 1))


def test_logistic_irls_step() -> None:
    x = np.random.randn(5, 3)
    y = np.ones((5, 1))
    beta = np.ones((3, 1))
    eta = x @ beta
    probability = 1.0 / (1.0 + np.exp(-eta))
    weight = probability * (1.0 - probability)
    adjusted_response = eta + (y - probability) / weight

    assert_shape(x.shape, (5, 3))
    assert_shape(y.shape, (5, 1))
    assert_shape(beta.shape, (3, 1))
    assert_shape(eta.shape, (5, 1))
    assert_shape(probability.shape, (5, 1))
    assert_shape(weight.shape, (5, 1))
    assert_shape(adjusted_response.shape, (5, 1))
    assert_shape(logistic_irls_step(x, y, beta).shape, (3, 1))


def test_pca_full_basis_projection() -> None:
    x = np.random.randn(5, 3)
    x_centered = x - x.mean(axis=0)
    scatter = x_centered.T @ x_centered
    u, s, vt = np.linalg.svd(scatter, full_matrices=False)
    projection = pca_full_basis_projection(x)

    assert_shape(x.shape, (5, 3))
    assert_shape(x_centered.shape, (5, 3))
    assert_shape(scatter.shape, (3, 3))
    assert_shape(u.shape, (3, 3))
    assert_shape(s.shape, (3,))
    assert_shape(vt.shape, (3, 3))
    assert_shape(projection.shape, (5, 3))


def test_nearest_centroid_labels() -> None:
    x = np.random.randn(5, 3)
    centroids = np.random.randn(4, 3)
    point_vectors = np.expand_dims(x, axis=-2)
    centroid_vectors = np.expand_dims(centroids, axis=-3)
    deltas = point_vectors - centroid_vectors
    squared_distances = np.sum(deltas**2, axis=-1)
    labels = nearest_centroid_labels(x, centroids)

    assert_shape(x.shape, (5, 3))
    assert_shape(centroids.shape, (4, 3))
    assert_shape(point_vectors.shape, (5, 1, 3))
    assert_shape(centroid_vectors.shape, (1, 4, 3))
    assert_shape(deltas.shape, (5, 4, 3))
    assert_shape(squared_distances.shape, (5, 4))
    assert_shape(labels.shape, (5,))


def test_cross_entropy_loss() -> None:
    logits = np.random.randn(5, 3)
    targets = np.zeros(5, dtype=np.intp)
    shifted = logits - logits.max(axis=1, keepdims=True)
    normalizers = np.exp(shifted).sum(axis=1, keepdims=True)
    log_probs = shifted - np.log(normalizers)
    target_log_probs: np.ndarray[[5], np.dtype[np.float64]] = log_probs[
        np.arange(len(targets)), targets
    ]
    loss = cross_entropy_loss(logits, targets)

    assert_shape(logits.shape, (5, 3))
    assert_shape(targets.shape, (5,))
    assert_shape(shifted.shape, (5, 3))
    assert_shape(normalizers.shape, (5, 1))
    assert_shape(log_probs.shape, (5, 3))
    assert_shape(target_log_probs.shape, (5,))
    assert_shape(loss.shape, ())
