/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    test_alias,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(alias="y")

m = Model(y=0)
m.x
m.y # E: Object of class `Model` has no attribute `y`
"#,
);

pydantic_testcase!(
    test_validation_alias,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(validation_alias="y", alias="z")

m = Model(y=0)
m = Model(z=0) # E: Missing argument `y` in function `Model.__init__`

m.x
m.y  # E: Object of class `Model` has no attribute `y`
m.z  # E: Object of class `Model` has no attribute `z`
"#,
);

pydantic_testcase!(
    test_validation_by_alias_and_name,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int = Field(alias='y')
Model(x=0) 
Model(y=0)
"#,
);

pydantic_testcase!(
    test_validation_by_alias,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_alias=True):
    x: int = Field(alias='y')
Model(x=0) # E: Missing argument `y` in function `Model.__init__` 
Model(y=0)
"#,
);

pydantic_testcase!(
    test_validation_by_name,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True):
    x: int = Field(alias='y')
Model(x=0)
Model(y=0)
"#,
);

pydantic_testcase!(
    test_validation_by_name_only,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True, validate_by_alias=False):
    x: int = Field(alias='y')
Model(x=0)
Model(y=0) # E: Missing argument `x` in function `Model.__init__` 
"#,
);

pydantic_testcase!(
    bug = "when both validation keys are true, aliases should be required. mypy doesn't support the right behavior here either. This requires generating overloads. We might shelve this for v2",
    test_validation_defaults,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int = Field(alias='y')
Model()
"#,
);

pydantic_testcase!(
    test_configdict_validate_by_name,
    r#"
from pydantic import BaseModel, Field, ConfigDict
class Model(BaseModel):
    x: str = Field(..., alias="y")
    model_config = ConfigDict(validate_by_name=True)
Model(y="123")
Model(x="123")  
    "#,
);

pydantic_testcase!(
    test_configdict_validate_by_alias,
    r#"
from pydantic import BaseModel, Field, ConfigDict
class Model(BaseModel):
    x: str = Field(..., alias="y")
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=False)
Model(y="123") # E: Missing argument `x` in function `Model.__init__`
Model(x="123")   
    "#,
);

pydantic_testcase!(
    test_configdict_validate_by_alias_optional,
    r#"
from pydantic import BaseModel, Field, ConfigDict
class Example(BaseModel):
    id: str
    some_attribute: str = Field("", alias="someAttribute")
    optional_attribute: str | None = Field(None, alias="optionalAttribute")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

x1 = Example(id="1", some_attribute="value")
x2 = Example(id="1", someAttribute="value")  
x3 = Example(id="1", someAttribute123="value")  
    "#,
);

pydantic_testcase!(
    test_validation_inheritance,
    r#"
from pydantic import BaseModel, ConfigDict, Field

class Model(BaseModel):
    x: str = Field(..., alias="y")
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=False)

class Model2(Model): ...

Model2(y="123") # E: Missing argument `x` in function `Model2.__init__`
Model2(x="123")

class Model3(Model2):
    x: str = Field(..., alias="y")

Model3(y="123") # E: Missing argument `x` in function `Model3.__init__`
Model3(x="123") 
    "#,
);
