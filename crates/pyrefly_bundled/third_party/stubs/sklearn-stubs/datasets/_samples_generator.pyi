from collections.abc import Iterable, Sequence
from typing import Literal

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike

# Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Nothman
# License: BSD 3 clause

def make_classification(
    n_samples: Int = 100,
    n_features: Int = 20,
    *,
    n_informative: Int = 2,
    n_redundant: Int = 2,
    n_repeated: Int = 0,
    n_classes: Int = 2,
    n_clusters_per_class: Int = 2,
    weights: None | ArrayLike = None,
    flip_y: Float = 0.01,
    class_sep: Float = 1.0,
    hypercube: bool = True,
    shift: float | None | ArrayLike = 0.0,
    scale: float | None | ArrayLike = 1.0,
    shuffle: bool = True,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_multilabel_classification(
    n_samples: Int = 100,
    n_features: Int = 20,
    *,
    n_classes: Int = 5,
    n_labels: Int = 2,
    length: Int = 50,
    allow_unlabeled: bool = True,
    sparse: bool = False,
    return_indicator: Literal["dense", "sparse"] | bool = "dense",
    return_distributions: bool = False,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray | spmatrix, ndarray, ndarray]: ...
def make_hastie_10_2(n_samples: Int = 12000, *, random_state: RandomState | None | Int = None) -> tuple[ndarray, ndarray]: ...
def make_regression(
    n_samples: Int = 100,
    n_features: Int = 100,
    *,
    n_informative: Int = 10,
    n_targets: Int = 1,
    bias: Float = 0.0,
    effective_rank: None | Int = None,
    tail_strength: Float = 0.5,
    noise: Float = 0.0,
    shuffle: bool = True,
    coef: bool = False,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray, ndarray]: ...
def make_circles(
    n_samples: int | tuple[int, int] = 100,
    *,
    shuffle: bool = True,
    noise: None | Float = None,
    random_state: RandomState | None | Int = None,
    factor: Float = 0.8,
) -> tuple[ndarray, ndarray]: ...
def make_moons(
    n_samples: int | tuple[int, int] = 100,
    *,
    shuffle: bool = True,
    noise: None | Float = None,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_blobs(
    n_samples: ArrayLike | int = 100,
    n_features: Int = 2,
    *,
    centers: None | MatrixLike | int = None,
    cluster_std: float | Sequence[float] = 1.0,
    center_box: tuple[float, float] = ...,
    shuffle: bool = True,
    random_state: RandomState | None | Int = None,
    return_centers: bool = False,
) -> tuple[ndarray, ndarray, ndarray] | tuple[ndarray, ndarray]: ...
def make_friedman1(
    n_samples: Int = 100,
    n_features: Int = 10,
    *,
    noise: Float = 0.0,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_friedman2(
    n_samples: Int = 100,
    *,
    noise: Float = 0.0,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_friedman3(
    n_samples: Int = 100,
    *,
    noise: Float = 0.0,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_low_rank_matrix(
    n_samples: Int = 100,
    n_features: Int = 100,
    *,
    effective_rank: Int = 10,
    tail_strength: Float = 0.5,
    random_state: RandomState | None | Int = None,
) -> ndarray: ...
def make_sparse_coded_signal(
    n_samples: Int,
    *,
    n_components: Int,
    n_features: Int,
    n_nonzero_coefs: Int,
    random_state: RandomState | None | Int = None,
    data_transposed: str | bool = "warn",
) -> map | tuple[ndarray, ndarray, ndarray]: ...
def make_sparse_uncorrelated(
    n_samples: Int = 100,
    n_features: Int = 10,
    *,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_spd_matrix(n_dim: Int, *, random_state: RandomState | None | Int = None) -> ndarray: ...
def make_sparse_spd_matrix(
    dim: Int = 1,
    *,
    alpha: Float = 0.95,
    norm_diag: bool = False,
    smallest_coef: Float = 0.1,
    largest_coef: Float = 0.9,
    random_state: RandomState | None | Int = None,
) -> ndarray | spmatrix: ...
def make_swiss_roll(
    n_samples: Int = 100,
    *,
    noise: Float = 0.0,
    random_state: RandomState | None | Int = None,
    hole: bool = False,
) -> tuple[ndarray, ndarray]: ...
def make_s_curve(
    n_samples: Int = 100,
    *,
    noise: Float = 0.0,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_gaussian_quantiles(
    *,
    mean: None | ArrayLike = None,
    cov: Float = 1.0,
    n_samples: Int = 100,
    n_features: Int = 2,
    n_classes: Int = 3,
    shuffle: bool = True,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def make_biclusters(
    shape: Iterable[MatrixLike] | tuple[int, int],
    n_clusters: Int,
    *,
    noise: Float = 0.0,
    minval: Int = 10,
    maxval: Int = 100,
    shuffle: bool = True,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray, ndarray]: ...
def make_checkerboard(
    shape: tuple | tuple[int, int],
    n_clusters: MatrixLike | Int | tuple[int, int],
    *,
    noise: Float = 0.0,
    minval: Int = 10,
    maxval: Int = 100,
    shuffle: bool = True,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray, ndarray]: ...
