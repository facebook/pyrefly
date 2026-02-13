from ._bagging import BaggingClassifier as BaggingClassifier, BaggingRegressor as BaggingRegressor
from ._base import BaseEnsemble as BaseEnsemble
from ._forest import (
    ExtraTreesClassifier as ExtraTreesClassifier,
    ExtraTreesRegressor as ExtraTreesRegressor,
    RandomForestClassifier as RandomForestClassifier,
    RandomForestRegressor as RandomForestRegressor,
    RandomTreesEmbedding as RandomTreesEmbedding,
)
from ._gb import GradientBoostingClassifier as GradientBoostingClassifier, GradientBoostingRegressor as GradientBoostingRegressor
from ._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier as HistGradientBoostingClassifier,
    HistGradientBoostingRegressor as HistGradientBoostingRegressor,
)
from ._iforest import IsolationForest as IsolationForest
from ._stacking import StackingClassifier as StackingClassifier, StackingRegressor as StackingRegressor
from ._voting import VotingClassifier as VotingClassifier, VotingRegressor as VotingRegressor
from ._weight_boosting import AdaBoostClassifier as AdaBoostClassifier, AdaBoostRegressor as AdaBoostRegressor

__all__ = [
    "BaseEnsemble",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomTreesEmbedding",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "IsolationForest",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
]
