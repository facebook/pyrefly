/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    bug = "Fails to catch type error in x4",
    test_basic,
    r#"
from typing import Union
X = Union[int, list["X"]]

x1: X = 1
x2: X = [1]
x3: X = [[1]]

x4: X = ["oops"]
    "#,
);
