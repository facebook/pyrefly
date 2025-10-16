/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    test_config_conditional_extra,
    r#"
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

u = User(name="Alice", age=30)
print(u)
"#,
);

pydantic_testcase!(
    test_model_config,
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int = 42


m = Model()
m.x = 10 # E: Cannot set field `x`
"#,
);

pydantic_testcase!(
    test_model_config_dict_func,
    r#"
from pydantic import BaseModel

class Model(BaseModel):
    model_config = dict(frozen=True)

    x: int = 42

m = Model()
m.x = 10 # E: Cannot set field `x`
"#,
);

pydantic_testcase!(
    test_model_config_dict_display,
    r#"
from pydantic import BaseModel

class Model(BaseModel):
    model_config = {'frozen': True}

    x: int = 42

m = Model()
m.x = 10 # E: Cannot set field `x`
"#,
);

pydantic_testcase!(
    test_not_a_pydantic_model,
    r#"
from pydantic import ConfigDict

class Model:
    model_config = ConfigDict(frozen=True)
    x: int = 42


m = Model()
m.x = 10 
"#,
);

// This is a corner case, but since y is annotated, we consider it a field
pydantic_testcase!(
    test_model_config_alias,
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    y: ConfigDict = ConfigDict(frozen=True)
    model_config = y
    x: int = 42

m = Model()
m.x = 10

"#,
);

pydantic_testcase!(
    bug = "We should raise an error on y because all model fields require an annotation.",
    test_model_config_y,
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    y = ConfigDict(frozen=True)
    x: int = 42

m = Model()
m.x = 10 
"#,
);

// Only the last config is considered
pydantic_testcase!(
    test_model_two_config,
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_config = ConfigDict(frozen=False)
    x: int = 42

m = Model()
m.x = 10
"#,
);

pydantic_testcase!(
    test_frozen_model_subclass,
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int = 42

class Model2(Model):
    model_config = ConfigDict(frozen=False)

m = Model2()
m.x = 10
"#,
);

pydantic_testcase!(
    test_frozen_model_default,
    r#"
from pydantic import BaseModel
class A(BaseModel, frozen=True):
    x: int

a = A(x=0)
a.x = 1 # E: Cannot set field `x`

class B(A):
    pass

b = B(x=0)
b.x = 1 # E: Cannot set field `x`
"#,
);

pydantic_testcase!(
    test_config_inheritance,
    r#"
from pydantic import BaseModel, ConfigDict

class Base(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Sub(Base):
    a: int

class Sub2(Sub):
    a: int
    
Sub(a=1, y=2) # E: Unexpected keyword argument `y` in function `Sub.__init__`
Sub2(a=1, y=2) # E: Unexpected keyword argument `y` in function `Sub2.__init__`


class Base3(BaseModel):
    model_config = ConfigDict(extra='allow')

class Sub3(Base3):
    a: int
    
Sub3(a=1, y=2) 

"#,
);

pydantic_testcase!(
    test_two_config_values,
    r#"
from pydantic import BaseModel, ConfigDict
class A(BaseModel):
    x: int
    model_config = ConfigDict(frozen=True, strict=True)
a = A(x=0)
a.x = 1  # E: Cannot set field `x`
A(x='oops')  # E: `Literal['oops']` is not assignable to parameter `x`
    "#,
);
