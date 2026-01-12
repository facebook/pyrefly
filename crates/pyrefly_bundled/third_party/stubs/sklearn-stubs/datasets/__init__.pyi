import textwrap as textwrap

from ._base import (
    clear_data_home as clear_data_home,
    get_data_home as get_data_home,
    load_breast_cancer as load_breast_cancer,
    load_diabetes as load_diabetes,
    load_digits as load_digits,
    load_files as load_files,
    load_iris as load_iris,
    load_linnerud as load_linnerud,
    load_sample_image as load_sample_image,
    load_sample_images as load_sample_images,
    load_wine as load_wine,
)
from ._california_housing import fetch_california_housing as fetch_california_housing
from ._covtype import fetch_covtype as fetch_covtype
from ._kddcup99 import fetch_kddcup99 as fetch_kddcup99
from ._lfw import fetch_lfw_pairs as fetch_lfw_pairs, fetch_lfw_people as fetch_lfw_people
from ._olivetti_faces import fetch_olivetti_faces as fetch_olivetti_faces
from ._openml import fetch_openml as fetch_openml
from ._rcv1 import fetch_rcv1 as fetch_rcv1
from ._samples_generator import (
    make_biclusters as make_biclusters,
    make_blobs as make_blobs,
    make_checkerboard as make_checkerboard,
    make_circles as make_circles,
    make_classification as make_classification,
    make_friedman1 as make_friedman1,
    make_friedman2 as make_friedman2,
    make_friedman3 as make_friedman3,
    make_gaussian_quantiles as make_gaussian_quantiles,
    make_hastie_10_2 as make_hastie_10_2,
    make_low_rank_matrix as make_low_rank_matrix,
    make_moons as make_moons,
    make_multilabel_classification as make_multilabel_classification,
    make_regression as make_regression,
    make_s_curve as make_s_curve,
    make_sparse_coded_signal as make_sparse_coded_signal,
    make_sparse_spd_matrix as make_sparse_spd_matrix,
    make_sparse_uncorrelated as make_sparse_uncorrelated,
    make_spd_matrix as make_spd_matrix,
    make_swiss_roll as make_swiss_roll,
)
from ._species_distributions import fetch_species_distributions as fetch_species_distributions
from ._svmlight_format_io import (
    dump_svmlight_file as dump_svmlight_file,
    load_svmlight_file as load_svmlight_file,
    load_svmlight_files as load_svmlight_files,
)
from ._twenty_newsgroups import (
    fetch_20newsgroups as fetch_20newsgroups,
    fetch_20newsgroups_vectorized as fetch_20newsgroups_vectorized,
)

__all__ = [
    "clear_data_home",
    "dump_svmlight_file",
    "fetch_20newsgroups",
    "fetch_20newsgroups_vectorized",
    "fetch_lfw_pairs",
    "fetch_lfw_people",
    "fetch_olivetti_faces",
    "fetch_species_distributions",
    "fetch_california_housing",
    "fetch_covtype",
    "fetch_rcv1",
    "fetch_kddcup99",
    "fetch_openml",
    "get_data_home",
    "load_diabetes",
    "load_digits",
    "load_files",
    "load_iris",
    "load_breast_cancer",
    "load_linnerud",
    "load_sample_image",
    "load_sample_images",
    "load_svmlight_file",
    "load_svmlight_files",
    "load_wine",
    "make_biclusters",
    "make_blobs",
    "make_circles",
    "make_classification",
    "make_checkerboard",
    "make_friedman1",
    "make_friedman2",
    "make_friedman3",
    "make_gaussian_quantiles",
    "make_hastie_10_2",
    "make_low_rank_matrix",
    "make_moons",
    "make_multilabel_classification",
    "make_regression",
    "make_s_curve",
    "make_sparse_coded_signal",
    "make_sparse_spd_matrix",
    "make_sparse_uncorrelated",
    "make_spd_matrix",
    "make_swiss_roll",
]
