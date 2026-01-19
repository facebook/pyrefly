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
    test_lax_mode_default,
    r#"
from pydantic.dataclasses import dataclass
@dataclass
class A:
    x: int
# Pydantic dataclasses default to strict=False (lax mode), so coercion is allowed
A(x='0')
    "#,
);

pydantic_testcase!(
    test_dataclass_literal_strictness,
    r#"
from typing import Literal
from pydantic.dataclasses import dataclass

@dataclass
class Example:
    value: Literal["MyLiteral"]

Example(value="MyLiteral")
Example(value=2)  # E: Argument `Literal[2]` is not assignable to parameter `value` with type `Literal['MyLiteral']` in function `Example.__init__`
    "#,
);
