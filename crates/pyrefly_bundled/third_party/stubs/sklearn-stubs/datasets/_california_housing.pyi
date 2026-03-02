from ..utils import Bunch

# The original data can be found at:
# https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz
ARCHIVE = ...

logger = ...

def fetch_california_housing(
    *, data_home: None | str = None, download_if_missing: bool = True, return_X_y: bool = False, as_frame: bool = False
) -> tuple[Bunch, tuple]: ...
