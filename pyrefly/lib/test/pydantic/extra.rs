/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    test_extra_allow,
    r#"
from pydantic import BaseModel

class ModelAllow(BaseModel, extra="allow"):
    x: int

ModelAllow(x=1, y=2)

"#,
);

pydantic_testcase!(
    test_extra_forbid,
    r#"
from pydantic import BaseModel

class ModelForbid(BaseModel, extra="forbid"):
    x: int

ModelForbid(x=1, y=2) # E: Unexpected keyword argument `y` in function `ModelForbid.__init__`
"#,
);

pydantic_testcase!(
    test_extra_ignore,
    r#"
from pydantic import BaseModel

class ModelForbid(BaseModel, extra="ignore"):
    x: int

ModelForbid(x=1, y=2)
"#,
);

pydantic_testcase!(
    test_extra_default,
    r#"
from pydantic import BaseModel

class ModelForbid(BaseModel):
    x: int

ModelForbid(x=1, y=2)
"#,
);

pydantic_testcase!(
    test_extra_wrong_type,
    r#"
from pydantic import BaseModel

class ModelForbid(BaseModel, extra=True): # E: Invalid value for `extra`. Expected one of 'allow', 'ignore', or 'forbid'
    x: int

ModelForbid(x=1, y=2)
"#,
);

pydantic_testcase!(
    test_extra_wrong_literal,
    r#"
from pydantic import BaseModel

class ModelForbid(BaseModel, extra="123"): # E: Invalid value for `extra`. Expected one of 'allow', 'ignore', or 'forbid'
    x: int

ModelForbid(x=1, y=2)
"#,
);

pydantic_testcase!(
    test_extra_allow_config_dict,
    r#"
from pydantic import BaseModel, ConfigDict

class ModelAllow(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: int

ModelAllow(x=1, y=2)

"#,
);

pydantic_testcase!(
    test_extra_forbid_config_dict,
    r#"
from pydantic import BaseModel, ConfigDict

class ModelForbid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int

ModelForbid(x=1, y=2) # E: Unexpected keyword argument `y` in function `ModelForbid.__init__`
"#,
);

pydantic_testcase!(
    test_extra_ignore_config_dict,
    r#"
from pydantic import BaseModel, ConfigDict

class ModelForbid(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x: int

ModelForbid(x=1, y=2)
"#,
);

pydantic_testcase!(
    test_extra_default_config_dict,
    r#"
from pydantic import BaseModel, ConfigDict

class ModelForbid(BaseModel):
    model_config = ConfigDict()
    x: int

ModelForbid(x=1, y=2)
"#,
);

// Note, Zeina: For ConfigDict, we extra can take any value and it won't raise an error
// the list of valid keywords is still 'allow', 'forbid' and 'ignore' (as well as None).
// Values like True/False seem legitimate and won't cause an error but won't actually apply to the model
// The errors here are raised directly from the stub definitions
pydantic_testcase!(
    test_extra_wrong_type_config_dict,
    r#"
from pydantic import BaseModel, ConfigDict

class ModelForbid(BaseModel):
    model_config = ConfigDict(extra=False) # E: No matching overload found for function `pydantic.config.ConfigDict.__init__`
    x: int

ModelForbid(x=1, y=2)
"#,
);

pydantic_testcase!(
    test_extra_wrong_literal_config_dict,
    r#"
from pydantic import BaseModel, ConfigDict

class ModelForbid(BaseModel):
    model_config = ConfigDict(extra="123") # E: No matching overload found for function `pydantic.config.ConfigDict.__init__`
    x: int

ModelForbid(x=1, y=2)
"#,
);
