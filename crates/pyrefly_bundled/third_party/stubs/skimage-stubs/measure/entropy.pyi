from numpy import unique
from scipy.stats import entropy as scipy_entropy

def shannon_entropy(image, base: float = 2) -> float: ...
