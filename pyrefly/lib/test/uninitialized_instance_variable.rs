/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::pydantic::util::pydantic_env;
use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_uninitialized_instance_variable,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
class C:
    x: int  # E: Instance attribute `x` is declared but never initialized
"#,
);

// Cases that must not fire: initialized in `__init__`, given a class-body value,
// `ClassVar`, `Protocol` members, `NamedTuple` fields, and ordinary dataclass fields
// (whose synthesized `__init__` initializes them).
testcase!(
    test_uninitialized_instance_variable_no_error,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from typing import ClassVar, NamedTuple, Protocol
from dataclasses import dataclass
class InInit:
    a: int
    def __init__(self) -> None:
        self.a = 1
class ClassBodyValue:
    b: int = 0
class HasClassVar:
    c: ClassVar[int]
class Proto(Protocol):
    d: int
class NT(NamedTuple):
    e: int
@dataclass
class DC:
    f: int
"#,
);

testcase!(
    test_uninitialized_instance_variable_pydantic,
    pydantic_env().enable_uninitialized_instance_variable_error(),
    r#"
from pydantic import BaseModel
class M(BaseModel):
    x: int
"#,
);

// A `Final` field with no initializer is reported by the dedicated `Final` check, not by
// this one; asserting a single error confirms we do not emit a duplicate diagnostic.
testcase!(
    test_uninitialized_instance_variable_final,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from typing import Final
class C:
    x: Final[int]  # E: Final attribute declared in class body must be initialized with a value or in `__init__`
"#,
);

testcase!(
    test_uninitialized_instance_variable_disabled_by_default,
    r#"
class C:
    x: int
"#,
);

testcase!(
    test_uninitialized_instance_variable_dataclass_init_false,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from dataclasses import dataclass
@dataclass(init=False)
class C:
    x: int  # E: Instance attribute `x` is declared but never initialized
"#,
);

testcase!(
    test_uninitialized_instance_variable_dataclass_default_init,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int
"#,
);

testcase!(
    test_uninitialized_instance_variable_dataclass_init_false_class_body_value,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from dataclasses import dataclass
@dataclass(init=False)
class C:
    x: int = 0
"#,
);

// https://github.com/facebook/pyrefly/issues/5280
testcase!(
    test_uninitialized_instance_variable_set_in_new,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from typing import Self
class Label(str):
    flag: bool
    def __new__(cls, text: str, flag: bool) -> Self:
        result = super().__new__(cls, text)
        result.flag = flag
        return result
class Partial:
    x: int
    y: int  # E: Instance attribute `y` is declared but never initialized
    def __new__(cls) -> Self:
        result = super().__new__(cls)
        result.x = 1
        return result
"#,
);

testcase!(
    test_uninitialized_instance_variable_not_set_in_new,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
from typing import Self
class Other:
    x: int = 0
class NeverSet:
    x: int  # E: Instance attribute `x` is declared but never initialized
    def __new__(cls) -> Self:
        return super().__new__(cls)
class SetOnOtherObject:
    x: int  # E: Instance attribute `x` is declared but never initialized
    def __new__(cls) -> Self:
        obj = Other()
        obj.x = 1
        return super().__new__(cls)
"#,
);

// https://github.com/facebook/pyrefly/issues/5278
// Metaclass annotations describe attributes supplied by classes that use the
// metaclass (via type.__new__), not fields that must be set in metaclass __init__.
testcase!(
    test_uninitialized_instance_variable_metaclass_attrs,
    TestEnv::new().enable_uninitialized_instance_variable_error(),
    r#"
class Metadata(type):
    label: str

class Example(metaclass=Metadata):
    label = "example"

def read_label(cls: Metadata) -> str:
    return cls.label
"#,
);
