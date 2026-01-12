import numpy as np
from scipy.ndimage import convolve, maximum_filter, minimum_filter

from .._shared.utils import check_nD
from ..feature import structure_tensor
from ..feature.util import FeatureDetector, _mask_border_keypoints, _prepare_grayscale_input_2D
from ..morphology import octagon, star
from ..transform import integral_image

# The paper(Reference [1]) mentions the sizes of the Octagon shaped filter
# kernel for the first seven scales only. The sizes of the later scales
# have been extrapolated based on the following statement in the paper.
# "These octagons scale linearly and were experimentally chosen to correspond
# to the seven DOBs described in the previous section."
OCTAGON_OUTER_SHAPE: list = ...
OCTAGON_INNER_SHAPE: list = ...

# The sizes for the STAR shaped filter kernel for different scales have been
# taken from the OpenCV implementation.
STAR_SHAPE: list = ...
STAR_FILTER_SHAPE: list = ...

def _filter_image(image, min_scale, max_scale, mode): ...
def _octagon_kernel(mo, no, mi, ni): ...
def _star_kernel(m, n): ...
def _suppress_lines(feature_mask, image, sigma, line_threshold): ...

class CENSURE(FeatureDetector):
    def __init__(
        self,
        min_scale=1,
        max_scale=7,
        mode="DoB",
        non_max_threshold=0.15,
        line_threshold=10,
    ): ...
    def detect(self, image): ...
