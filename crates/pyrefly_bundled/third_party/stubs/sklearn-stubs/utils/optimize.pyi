import warnings

import numpy as np

from ..exceptions import ConvergenceWarning as ConvergenceWarning

# This is a modified file from scipy.optimize
# Original authors: Travis Oliphant, Eric Jones
# Modifications by Gael Varoquaux, Mathieu Blondel and Tom Dupre la Tour
# License: BSD

class _LineSearchError(RuntimeError): ...
