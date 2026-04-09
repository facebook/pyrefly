/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_newtype_none_is_nominal,
    r#"
from typing import NewType
from types import NoneType

NewNoneType = NewType("NewNoneType", NoneType)
NewNone = NewNoneType(None)

def test(x: int | NewNoneType) -> None:
    pass

test(None)  # E: Argument `None` is not assignable to parameter `x` with type `NewNoneType | int` in function `test`
test(NewNone)
test(1)
    "#,
);