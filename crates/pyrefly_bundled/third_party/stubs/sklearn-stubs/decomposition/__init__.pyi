from ..utils.extmath import randomized_svd as randomized_svd
from ._dict_learning import (
    DictionaryLearning as DictionaryLearning,
    MiniBatchDictionaryLearning as MiniBatchDictionaryLearning,
    SparseCoder as SparseCoder,
    dict_learning as dict_learning,
    dict_learning_online as dict_learning_online,
    sparse_encode as sparse_encode,
)
from ._factor_analysis import FactorAnalysis as FactorAnalysis
from ._fastica import FastICA as FastICA, fastica as fastica
from ._incremental_pca import IncrementalPCA as IncrementalPCA
from ._kernel_pca import KernelPCA as KernelPCA
from ._lda import LatentDirichletAllocation as LatentDirichletAllocation
from ._nmf import NMF as NMF, MiniBatchNMF as MiniBatchNMF, non_negative_factorization as non_negative_factorization
from ._pca import PCA as PCA
from ._sparse_pca import MiniBatchSparsePCA as MiniBatchSparsePCA, SparsePCA as SparsePCA
from ._truncated_svd import TruncatedSVD as TruncatedSVD

__all__ = [
    "DictionaryLearning",
    "FastICA",
    "IncrementalPCA",
    "KernelPCA",
    "MiniBatchDictionaryLearning",
    "MiniBatchNMF",
    "MiniBatchSparsePCA",
    "NMF",
    "PCA",
    "SparseCoder",
    "SparsePCA",
    "dict_learning",
    "dict_learning_online",
    "fastica",
    "non_negative_factorization",
    "randomized_svd",
    "sparse_encode",
    "FactorAnalysis",
    "TruncatedSVD",
    "LatentDirichletAllocation",
]
