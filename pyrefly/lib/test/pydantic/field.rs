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
    test_field_right_type,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
   x: int = Field(gt=0, lt=10)

Model(x=5) 
Model(x=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(x=15)  # E: Argument value `Literal[15]` violates Pydantic `lt` constraint `Literal[10]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_ge_le,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge=0, le=10)

Model(x=0)
Model(x=10)
Model(x=-1)  # E: Argument value `Literal[-1]` violates Pydantic `ge` constraint `Literal[0]` for field `x`
Model(x=11)  # E: Argument value `Literal[11]` violates Pydantic `le` constraint `Literal[10]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_positional,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt=0, kw_only=False)
    y: int = Field(lt=3, kw_only=False)

Model(1, 2)
Model(0, 2)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(1, 3)  # E: Argument value `Literal[3]` violates Pydantic `lt` constraint `Literal[3]` for field `y`
"#,
);

pydantic_testcase!(
    test_field_range_kw_only,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge=1, kw_only=True)

Model(x=1)
Model(x=0)  # E: Argument value `Literal[0]` violates Pydantic `ge` constraint `Literal[1]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_alias,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int = Field(gt=0, validation_alias="y")

Model(x=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(y=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_alias_only,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel, validate_by_name=False, validate_by_alias=True):
    x: int = Field(gt=0, validation_alias="y")

Model(y=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(x=0)  # E: Missing argument `y`
"#,
);

pydantic_testcase!(
    test_field_range_multiple,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt=1, kw_only=False)
    y: int = Field(lt=0, kw_only=False)

Model(2, -1)
Model(y=-1, x=2)

Model(2, 0)  # E: violates Pydantic `lt` constraint `Literal[0]` for field `y`
Model(y=-1, x=1)  # E: violates Pydantic `gt` constraint `Literal[1]` for field `x`
    "#,
);

pydantic_testcase!(
    test_field_wrong_type,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt="A", lt="B") # E:  Pydantic `gt` value has type `Literal['A']`, which is not assignable to field type `int` # E: Pydantic `lt` value has type `Literal['B']`, which is not assignable to field type `int`

Model(x=5)
"#,
);

pydantic_testcase!(
    test_field_ge,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge="B") # E: Pydantic `ge` value has type `Literal['B']`, which is not assignable to field type `int`

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
