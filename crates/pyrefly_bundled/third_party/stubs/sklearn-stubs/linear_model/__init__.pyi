from ._base import LinearRegression as LinearRegression
from ._bayes import ARDRegression as ARDRegression, BayesianRidge as BayesianRidge
from ._coordinate_descent import (
    ElasticNet as ElasticNet,
    ElasticNetCV as ElasticNetCV,
    Lasso as Lasso,
    LassoCV as LassoCV,
    MultiTaskElasticNet as MultiTaskElasticNet,
    MultiTaskElasticNetCV as MultiTaskElasticNetCV,
    MultiTaskLasso as MultiTaskLasso,
    MultiTaskLassoCV as MultiTaskLassoCV,
    enet_path as enet_path,
    lasso_path as lasso_path,
)
from ._glm import GammaRegressor as GammaRegressor, PoissonRegressor as PoissonRegressor, TweedieRegressor as TweedieRegressor
from ._huber import HuberRegressor as HuberRegressor
from ._least_angle import (
    Lars as Lars,
    LarsCV as LarsCV,
    LassoLars as LassoLars,
    LassoLarsCV as LassoLarsCV,
    LassoLarsIC as LassoLarsIC,
    lars_path as lars_path,
    lars_path_gram as lars_path_gram,
)
from ._logistic import LogisticRegression as LogisticRegression, LogisticRegressionCV as LogisticRegressionCV
from ._omp import (
    OrthogonalMatchingPursuit as OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV as OrthogonalMatchingPursuitCV,
    orthogonal_mp as orthogonal_mp,
    orthogonal_mp_gram as orthogonal_mp_gram,
)
from ._passive_aggressive import (
    PassiveAggressiveClassifier as PassiveAggressiveClassifier,
    PassiveAggressiveRegressor as PassiveAggressiveRegressor,
)
from ._perceptron import Perceptron as Perceptron
from ._quantile import QuantileRegressor as QuantileRegressor
from ._ransac import RANSACRegressor as RANSACRegressor
from ._ridge import (
    Ridge as Ridge,
    RidgeClassifier as RidgeClassifier,
    RidgeClassifierCV as RidgeClassifierCV,
    RidgeCV as RidgeCV,
    ridge_regression as ridge_regression,
)
from ._sgd_fast import Hinge as Hinge, Huber as Huber, Log as Log, ModifiedHuber as ModifiedHuber, SquaredLoss as SquaredLoss
from ._stochastic_gradient import SGDClassifier as SGDClassifier, SGDOneClassSVM as SGDOneClassSVM, SGDRegressor as SGDRegressor
from ._theil_sen import TheilSenRegressor as TheilSenRegressor

__all__ = [
    "ARDRegression",
    "BayesianRidge",
    "ElasticNet",
    "ElasticNetCV",
    "Hinge",
    "Huber",
    "HuberRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LinearRegression",
    "Log",
    "LogisticRegression",
    "LogisticRegressionCV",
    "ModifiedHuber",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
    "QuantileRegressor",
    "Ridge",
    "RidgeCV",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SGDRegressor",
    "SGDOneClassSVM",
    "SquaredLoss",
    "TheilSenRegressor",
    "enet_path",
    "lars_path",
    "lars_path_gram",
    "lasso_path",
    "orthogonal_mp",
    "orthogonal_mp_gram",
    "ridge_regression",
    "RANSACRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]
