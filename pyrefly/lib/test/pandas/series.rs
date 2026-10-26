/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

/// The bundled pandas stubs need numpy's generic array types for Series overloads.
fn env_with_pandas_ndarray_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "numpy",
        r#"
from typing import Any, Generic, TypeVar

_ShapeT = TypeVar("_ShapeT", covariant=True)
_DTypeT = TypeVar("_DTypeT", covariant=True)

class generic: ...
class number(generic): ...
class floating(number): ...
class float32(floating): ...
class complexfloating(number): ...
class integer(number): ...
class unsignedinteger(integer): ...
class signedinteger(integer): ...
class int_(signedinteger): ...
class int8(signedinteger): ...
class intp(signedinteger): ...
class int64(signedinteger): ...
class bool_(generic): ...
class str_(generic): ...
class datetime64(generic): ...
class timedelta64(generic): ...
class dtype(Generic[_DTypeT]): ...
class ndarray(Generic[_ShapeT, _DTypeT]): ...

def __getattr__(name: str) -> Any: ...
"#,
    );
    env.add(
        "numpy.typing",
        r#"
from typing import Any, TypeAlias, TypeVar
from numpy import dtype, ndarray

_DTypeT = TypeVar("_DTypeT")
NDArray: TypeAlias = ndarray[tuple[Any, ...], dtype[_DTypeT]]
"#,
    );
    env
}

testcase!(
    test_series_ndarray_multiplication,
    env_with_pandas_ndarray_stubs(),
    r#"
from typing import Any, assert_type
import numpy as np
import pandas as pd  # pyrefly: ignore[missing-source-for-stubs]

def multiply(
    array: np.ndarray[tuple[Any, ...], np.dtype[np.float32]],
    series: pd.Series[float],
) -> None:
    assert_type(array * series, pd.Series[float])
"#,
);
