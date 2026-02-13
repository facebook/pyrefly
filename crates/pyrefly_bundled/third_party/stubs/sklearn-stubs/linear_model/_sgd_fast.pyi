class LossFunction:
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y) -> float: ...

class Regression(LossFunction):
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y) -> float: ...

class Classification(LossFunction):
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y) -> float: ...

class ModifiedHuber(Classification):
    """Modified Huber loss for binary classification with y in {-1, 1}

    This is equivalent to quadratically smoothed SVM with gamma = 2.

    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """

    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class Hinge(Classification):
    """Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    """

    threshold: float
    def __init__(self, threshold: float = 1.0) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class SquaredHinge(Classification):
    """Squared Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by
        (quadratically penalized) SVM.
    """

    threshold: float
    def __init__(self, threshold: float = 1.0) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class Log(Classification):
    """Logistic regression loss for binary classification with y in {-1, 1}"""

    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class SquaredLoss(Regression):
    """Squared loss traditional used in linear regression."""

    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class Huber(Regression):
    """Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    https://en.wikipedia.org/wiki/Huber_Loss_Function
    """

    c: float
    def __init__(self, c: float) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class EpsilonInsensitive(Regression):
    """Epsilon-Insensitive loss (used by SVR).

    loss = max(0, |y - p| - epsilon)
    """

    epsilon: float
    def __init__(self, epsilon: float) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...

class SquaredEpsilonInsensitive(Regression):
    """Epsilon-Insensitive loss.

    loss = max(0, |y - p| - epsilon)^2
    """

    epsilon: float
    def __init__(self, epsilon: float) -> None: ...
    def loss(self, p: float, y: float) -> float: ...
    def dloss(self, p: float, y: float) -> float: ...
