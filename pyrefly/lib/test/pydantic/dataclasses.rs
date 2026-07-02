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
    test_property_override_init_false_field,
    r#"
from pydantic.dataclasses import dataclass
from dataclasses import field
@dataclass
class A:
    foo: int = field(init=False)
@dataclass
class B(A):
    @property
    def foo(self) -> int:  # E: Class member `B.foo` overrides parent class `A` in an inconsistent manner
        return 1
# `foo` is `init=False`, inherited from `A`: not a constructor parameter.
B()
B(foo=2)  # E: Unexpected keyword argument `foo`
    "#,
);
