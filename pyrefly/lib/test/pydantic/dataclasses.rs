/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    test_basic,
    r#"
from pydantic.dataclasses import dataclass
@dataclass
class A:
    x: int
A(x=0)
    "#,
);

pydantic_testcase!(
    bug = "We should accept this because pydantic coerces '0' to 0 at runtime",
    test_lax_mode,
    r#"
from pydantic.dataclasses import dataclass
@dataclass
class A:
    x: int
A(x='0')  # E: `Literal['0']` is not assignable to parameter `x` with type `int`
    "#,
);
