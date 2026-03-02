import numpy as np

def predict_stages(estimators: np.ndarray, X, scale: float, out: np.ndarray) -> None:
    """Add predictions of ``estimators`` to ``out``.
    Each estimator is scaled by ``scale`` before its prediction
    is added to ``out``.
    """

def predict_stage(estimators: np.ndarray, stage: int, X, scale: float, out: np.ndarray) -> None:
    """Add predictions of ``estimators[stage]`` to ``out``.
    Each estimator in the stage is scaled by ``scale`` before
    its prediction is added to ``out``.
    """
