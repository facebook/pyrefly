from numpy import ndarray

from .._typing import ArrayLike, MatrixLike
from ..base import TransformerMixin

# Author: V. Michel, A. Gramfort
# License: BSD 3 clause

###############################################################################
# Mixin class for feature agglomeration.

class AgglomerationTransform(TransformerMixin):
    def transform(self, X: MatrixLike) -> ndarray: ...
    def inverse_transform(self, Xred: MatrixLike | ArrayLike) -> ndarray: ...
