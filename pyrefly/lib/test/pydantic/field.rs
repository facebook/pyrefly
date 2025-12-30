/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::pydantic_testcase;
use crate::test::pydantic::util::pydantic_env;
use crate::test::util::TestEnv;
use crate::testcase;

pydantic_testcase!(
    bug = "we could support ranges, but this is not for v1",
    test_field_right_type,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
   x: int = Field(gt=0, lt=10)

Model(x=5) 
Model(x=0)  
Model(x=15)
"#,
);

pydantic_testcase!(
    test_field_wrong_type,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt="A", lt="B") # E:  Pydantic `gt` value is of type `Literal['A']` but the field is annotated with `int` # E: Pydantic `lt` value is of type `Literal['B']` but the field is annotated with `int`

Model(x=5)
"#,
);

pydantic_testcase!(
    test_field_ge,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge="B") # E: Pydantic `ge` value is of type `Literal['B']` but the field is annotated with `int`

Model(x=5)
"#,
);

pydantic_testcase!(
    test_field_optional,
    r#"
from pydantic import BaseModel, Field

class Example(BaseModel):
    id: str
    attribute_1: str = Field(..., description="A required attribute")
    optional_attribute1: str | None = Field(None, description="An optional attribute")
    optional_attribute2: int = Field(0, description="Another optional attribute")

Example(id="123", attribute_1="value1")
Example(id="123")  # E: Missing argument `attribute_1`
"#,
);

pydantic_testcase!(
    bug = "consider erroring on invalid5 and invalid6",
    test_discriminated_unions,
    r#"
from typing import Literal, Union
from pydantic import BaseModel, Field

class A(BaseModel):
    kind: Literal["a"]
    val: int

class B(BaseModel):
    kind: Literal["b"]
    msg: str

class Wrapper(BaseModel):
    item: Union[A, B] = Field(discriminator="kind")

valid1 = Wrapper(item=A(kind="a", val=123))
valid2 = Wrapper(item=B(kind="b", msg="Bob"))

invalid1 = Wrapper(item=A(kind="a")) # E: Missing argument `val` in function `A.__init__` 
invalid2 = Wrapper(item=B(kind="b", val=123)) # E: Missing argument `msg` in function `B.__init__` 

valid3 = Wrapper.model_validate({"item": A(kind="a", val=123)})
valid4 = Wrapper.model_validate({"item": B(kind="b", msg="Bob")})

invalid3 = Wrapper.model_validate({"item": A(kind="a")}) # E: Missing argument `val` in function `A.__init__`
invalid4 =  Wrapper.model_validate({"item": B(kind="b", val=123)}) # E: Missing argument `msg` in function `B.__init__`

valid5 = Wrapper.model_validate({"item": {"kind": "a", "val": 123}})
valid6 = Wrapper.model_validate({"item": {"kind": "b", "msg": "Bob"}})

invalid5 = Wrapper.model_validate({"item": {"kind": "a"}})  
invalid6 = Wrapper.model_validate({"item": {"kind": "b", "name": 123}})  

    "#,
);

pydantic_testcase!(
    test_discriminated_unions_annotated,
    r#"
from typing import Annotated, Literal
from pydantic import BaseModel, Field

class A(BaseModel): 
    input_type: Literal["A"] = "A"
class B(BaseModel):
    input_type: Literal["B"] = "B"

T = Annotated[A | B, Field(discriminator="input_type")]

def foo(ts: list[T]) -> list[A]:
    return [
        t for t in ts
        if t.input_type == "A"
    ]

print(foo([A(), B()]))
    "#,
);

pydantic_testcase!(
    test_required_field,
    r#"
from pydantic import BaseModel
class A(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int
A()  # E: Missing argument `x`
    "#,
);

pydantic_testcase!(
    test_field_default_gt_violation,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    value: int = Field(0, gt=0)  # E: Default value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `value`

class Model2(BaseModel):
    value: int = Field(default=0, gt=0)  # E: violates Pydantic `gt` constraint

class Model3(BaseModel):
    value: int = Field(default_factory=lambda: 0, gt=0)  # E: violates Pydantic `gt` constraint
    "#,
);

pydantic_testcase!(
    test_field_default_gt_ok,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    value: int = Field(1, gt=0)

class Model2(BaseModel):
    value: int = Field(default=1, gt=0)

class Model3(BaseModel):
    value: int = Field(default_factory=lambda: 1, gt=0)

def f() -> int: ...
class Model4(BaseModel):
    value: int = Field(f(), gt=0)
    "#,
);

fn pydantic_env_3_10() -> TestEnv {
    let env = pydantic_env();
    env.with_version(PythonVersion::new(3, 10, 0))
}

testcase!(
    test_model_3_10,
    pydantic_env_3_10(),
    r#"
from pydantic import BaseModel
class A(BaseModel, strict=True):
    x: int
A(x='')  # E: `Literal['']` is not assignable to parameter `x` with type `int`
    "#,
);

pydantic_testcase!(
    test_default_keywords,
    r#"
from pydantic import BaseModel, Field
class A(BaseModel):
    x: int = Field(default='oops')  # E: `str` is not assignable to `int`
class B(BaseModel):
    x: int = Field(default_factory=lambda: 'oops')  # E: `str` is not assignable to `int`
    "#,
);

pydantic_testcase!(
    test_annotated_field_with_defaults,
    r#"
from typing import Annotated
from pydantic import BaseModel, Field

class House(BaseModel):
    street: str
    city: str
    zipcode: str
    notes: Annotated[str, Field(default="")]
    extra_fields: Annotated[dict, Field(default_factory=dict)]
    something: dict = Field(default_factory=dict)

house = House(
    street="House Street",
    city="House City",
    zipcode="House Zipcode",
)
    "#,
);
