from typing import Literal, overload
from typing_extensions import deprecated

from numpy import ndarray

from .._typing import ArrayLike, Float, MatrixLike

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Karan Desai <karandesai281196@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
#          Konstantin Shmelkov <konstantin.shmelkov@polytechnique.edu>
#          Christian Lorentzen <lorentzen.ch@gmail.com>
#          Ashutosh Hathidara <ashutoshhathidara98@gmail.com>
#          Uttam kumar <bajiraouttamsinha@gmail.com>
#          Sylvain Marie <sylvain.marie@se.com>
#          Ohad Michel <ohadmich@gmail.com>
# License: BSD 3 clause

__ALL__: list = ...

@overload
def mean_absolute_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def mean_absolute_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
@overload
def mean_pinball_loss(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    alpha: float = 0.5,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def mean_pinball_loss(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    alpha: float = 0.5,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> Float: ...
@overload
def mean_absolute_percentage_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def mean_absolute_percentage_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
@overload
def mean_squared_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def mean_squared_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
@deprecated(
    "`squared` is deprecated in 1.4 and will be removed in 1.6. Use `root_mean_squared_error` instead to calculate the root mean squared error."
)
@overload
def mean_squared_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
    squared: bool,
) -> ndarray: ...
@deprecated(
    "`squared` is deprecated in 1.4 and will be removed in 1.6. Use `root_mean_squared_error` instead to calculate the root mean squared error."
)
@overload
def mean_squared_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
    squared: bool,
) -> float: ...
@overload
def mean_squared_log_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def mean_squared_log_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
@deprecated(
    "`squared` is deprecated in 1.4 and will be removed in 1.6. Use `root_mean_squared_log_error` instead to calculate the root mean squared logarithmic error."
)
@overload
def mean_squared_log_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
    squared: bool,
) -> ndarray: ...
@deprecated(
    "`squared` is deprecated in 1.4 and will be removed in 1.6. Use `root_mean_squared_log_error` instead to calculate the root mean squared logarithmic error."
)
@overload
def mean_squared_log_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
    squared: bool,
) -> float: ...
@overload
def median_absolute_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    multioutput: Literal["raw_values"],
    sample_weight: None | ArrayLike = None,
) -> ndarray: ...
@overload
def median_absolute_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
    sample_weight: None | ArrayLike = None,
) -> Float: ...
@overload
def explained_variance_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
    force_finite: bool = True,
) -> ndarray: ...
@overload
def explained_variance_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average", "variance_weighted"] | ArrayLike = "uniform_average",
    force_finite: bool = True,
) -> float: ...
@overload
def r2_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
    force_finite: bool = True,
) -> ndarray: ...
@overload
def r2_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average", "variance_weighted"] | ArrayLike | None = "uniform_average",
    force_finite: bool = True,
) -> float: ...
def max_error(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def mean_tweedie_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    power: Float = 0,
) -> float: ...
def mean_poisson_deviance(y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: None | ArrayLike = None) -> Float: ...
def mean_gamma_deviance(y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: None | ArrayLike = None) -> float: ...
def d2_tweedie_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    power: Float = 0,
) -> float: ...
@overload
def d2_pinball_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    alpha: Float = 0.5,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def d2_pinball_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    alpha: Float = 0.5,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> Float: ...
@overload
def d2_absolute_error_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def d2_absolute_error_score(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> Float: ...
@overload
def root_mean_squared_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def root_mean_squared_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
@overload
def root_mean_squared_log_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["raw_values"],
) -> ndarray: ...
@overload
def root_mean_squared_log_error(
    y_true: MatrixLike | ArrayLike,
    y_pred: MatrixLike | ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
