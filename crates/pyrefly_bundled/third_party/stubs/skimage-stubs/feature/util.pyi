from typing import Literal

import numpy as np

from .._shared.utils import _supported_float_type, check_nD

class FeatureDetector:
    def __init__(self): ...
    def detect(self, image): ...

class DescriptorExtractor:
    def __init__(self): ...
    def extract(self, image, keypoints): ...

def plot_matches(
    ax,
    image1,
    image2,
    keypoints1,
    keypoints2,
    matches,
    keypoints_color="k",
    matches_color=None,
    only_matches: bool = False,
    alignment: Literal["horizontal", "vertical"] = "horizontal",
): ...
def _prepare_grayscale_input_2D(image): ...
def _prepare_grayscale_input_nD(image): ...
def _mask_border_keypoints(image_shape, keypoints, distance): ...
