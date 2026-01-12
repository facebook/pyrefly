from typing import Literal

from numpy.random import RandomState

from .._typing import Int
from ..utils import Bunch

# Author: Tom Dupre la Tour
# License: BSD 3 clause

# The original vectorized data can be found at:
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt0.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt1.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt2.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt3.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz
# while the original stemmed token files can be found
# in the README, section B.12.i.:
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
XY_METADATA = ...

# The original data can be found at:
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
TOPICS_METADATA = ...

logger = ...

def fetch_rcv1(
    *,
    data_home: None | str = None,
    subset: Literal["train", "test", "all"] = "all",
    download_if_missing: bool = True,
    random_state: RandomState | None | Int = None,
    shuffle: bool = False,
    return_X_y: bool = False,
) -> tuple[Bunch, tuple]: ...
