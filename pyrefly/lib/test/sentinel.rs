/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_sentinel_construction_success,
    r#"
from typing_extensions import Sentinel

A = Sentinel("A")
    "#,
);

testcase!(
    test_sentinel_construction_second_positional_arg,
    r#"
from typing_extensions import Sentinel

A = Sentinel("A", 123)  # E: Sentinel only takes one positional argument
    "#,
);

testcase!(
    test_sentinel_construction_with_repr_success,
    r#"
from typing_extensions import Sentinel

A = Sentinel("A", repr="some text")
    "#,
);

testcase!(
    test_sentinel_construction_with_repr_str_success,
    r#"
from typing_extensions import Sentinel

text: str = "some other text"
A = Sentinel("A", repr=text)
    "#,
);

testcase!(
    test_sentinel_construction_with_repr_int_error,
    r#"
from typing_extensions import Sentinel

text = 5
A = Sentinel("A", repr=text)  # E: Invalid type for Sentinel `repr` Literal[5]
    "#,
);

testcase!(
    test_sentinel_construction_name_kwarg_error,
    r#"
from typing_extensions import Sentinel

A = Sentinel(name="A")  # E: Sentinel requires a name as the first argument # E: Unexpected keyword argument `name` to Sentinel
    "#,
);

testcase!(
    test_sentinel_construction_different_names,
    r#"
from typing_extensions import Sentinel

A = Sentinel("B")  # E: Sentinel must be assigned to a variable named `B`
    "#,
);

testcase!(
    test_sentinel_typehint_success,
    r#"
from typing_extensions import Sentinel

A: A = Sentinel("A")
    "#,
);

testcase!(
    test_sentinel_typehint_different_sentinel,
    r#"
from typing_extensions import Sentinel

A = Sentinel("A")
B: A = Sentinel("B")  # E: `Sentinel` is not assignable to `A`
    "#,
);

testcase!(
    test_sentinel_typehint_any,
    r#"
from typing import Any, assert_type
from typing_extensions import Sentinel

A: Any = Sentinel("A")
assert_type(A, Any)
    "#,
);

testcase!(
    test_sentinel_violates_annotation,
    r#"
from typing_extensions import Sentinel

MISSING: int = 0
MISSING = Sentinel('MISSING')  # E: `MISSING` is not assignable to variable `MISSING` with type `int`
    "#,
);

testcase!(
    test_sentinel_no_args_error,
    r#"
from typing_extensions import Sentinel

MISSING = Sentinel()  # E: Sentinel requires a name as the first argument
    "#,
);
