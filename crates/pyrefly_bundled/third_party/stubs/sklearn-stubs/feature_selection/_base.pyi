from abc import ABCMeta

from numpy import ndarray
from scipy.sparse._csr import csr_matrix

from .._typing import ArrayLike, MatrixLike
from ..base import TransformerMixin

# Authors: G. Varoquaux, A. Gramfort, L. Buitinck, J. Nothman
# License: BSD 3 clause

class SelectorMixin(TransformerMixin, metaclass=ABCMeta):
    def get_support(self, indices: bool = False) -> ndarray: ...
    def transform(self, X: MatrixLike) -> csr_matrix | ndarray: ...
    def inverse_transform(self, X: MatrixLike) -> ndarray: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
