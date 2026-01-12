import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from .._typing import Scalar

def integral_image(image: NDArray, *, dtype=None) -> NDArray: ...
def integrate(ii: NDArray, start, end) -> Scalar | np.ndarray: ...
