from typing import Literal

import numpy as np

from .._shared.filters import gaussian
from .._shared.utils import check_nD
from .util import DescriptorExtractor, _mask_border_keypoints, _prepare_grayscale_input_2D

class BRIEF(DescriptorExtractor):
    def __init__(
        self,
        descriptor_size: int = 256,
        patch_size: int = 49,
        mode: Literal["normal", "uniform"] = "normal",
        sigma: float = 1,
        sample_seed=1,
    ): ...
    def extract(self, image, keypoints): ...
