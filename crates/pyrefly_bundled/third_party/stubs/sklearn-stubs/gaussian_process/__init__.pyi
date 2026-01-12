from . import kernels as kernels
from ._gpc import GaussianProcessClassifier as GaussianProcessClassifier
from ._gpr import GaussianProcessRegressor as GaussianProcessRegressor

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Vincent Dubourg <vincent.dubourg@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

__all__ = ["GaussianProcessRegressor", "GaussianProcessClassifier", "kernels"]
