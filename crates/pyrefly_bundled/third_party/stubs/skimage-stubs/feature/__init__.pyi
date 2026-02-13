from .._shared.utils import deprecated as deprecated
from ._basic_features import multiscale_basic_features as multiscale_basic_features
from ._canny import canny as canny
from ._cascade import Cascade as Cascade
from ._daisy import daisy as daisy
from ._hog import hog as hog
from .blob import blob_dog as blob_dog, blob_doh as blob_doh, blob_log as blob_log
from .brief import BRIEF as BRIEF
from .censure import CENSURE as CENSURE
from .corner import (
    corner_fast as corner_fast,
    corner_foerstner as corner_foerstner,
    corner_harris as corner_harris,
    corner_kitchen_rosenfeld as corner_kitchen_rosenfeld,
    corner_moravec as corner_moravec,
    corner_orientations as corner_orientations,
    corner_peaks as corner_peaks,
    corner_shi_tomasi as corner_shi_tomasi,
    corner_subpix as corner_subpix,
    hessian_matrix as hessian_matrix,
    hessian_matrix_det as hessian_matrix_det,
    hessian_matrix_eigvals as hessian_matrix_eigvals,
    shape_index as shape_index,
    structure_tensor as structure_tensor,
    structure_tensor_eigenvalues as structure_tensor_eigenvalues,
    structure_tensor_eigvals as structure_tensor_eigvals,
)
from .haar import (
    draw_haar_like_feature as draw_haar_like_feature,
    haar_like_feature as haar_like_feature,
    haar_like_feature_coord as haar_like_feature_coord,
)
from .match import match_descriptors as match_descriptors
from .orb import ORB as ORB
from .peak import peak_local_max as peak_local_max
from .sift import SIFT as SIFT
from .template import match_template as match_template
from .texture import (
    draw_multiblock_lbp as draw_multiblock_lbp,
    graycomatrix as graycomatrix,
    graycoprops as graycoprops,
    local_binary_pattern as local_binary_pattern,
    multiblock_lbp as multiblock_lbp,
)
from .util import plot_matches as plot_matches

@deprecated(alt_func="skimage.feature.graycomatrix", removed_version="1.0")
def greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False): ...
@deprecated(alt_func="skimage.feature.graycoprops", removed_version="1.0")
def greycoprops(P, prop="contrast"): ...

__all__ = [
    "canny",
    "Cascade",
    "daisy",
    "hog",
    "graycomatrix",
    "graycoprops",
    "greycomatrix",
    "greycoprops",
    "local_binary_pattern",
    "multiblock_lbp",
    "draw_multiblock_lbp",
    "peak_local_max",
    "structure_tensor",
    "structure_tensor_eigenvalues",
    "structure_tensor_eigvals",
    "hessian_matrix",
    "hessian_matrix_det",
    "hessian_matrix_eigvals",
    "shape_index",
    "corner_kitchen_rosenfeld",
    "corner_harris",
    "corner_shi_tomasi",
    "corner_foerstner",
    "corner_subpix",
    "corner_peaks",
    "corner_moravec",
    "corner_fast",
    "corner_orientations",
    "match_template",
    "BRIEF",
    "CENSURE",
    "ORB",
    "SIFT",
    "match_descriptors",
    "plot_matches",
    "blob_dog",
    "blob_doh",
    "blob_log",
    "haar_like_feature",
    "haar_like_feature_coord",
    "draw_haar_like_feature",
    "multiscale_basic_features",
]
