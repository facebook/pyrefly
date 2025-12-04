/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    test_pyrefly_strict,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
    x: int = Field(strict=False)  # this is the default
    y: int = Field(strict=True)
Model(x='0', y=1) 
Model(x='0', y='1') # E: Argument `Literal['1']` is not assignable to parameter `y` with type `int` in function `Model.__init__` 
"#,
);

// Note: mypy does not support strict=false. Everything is strict.
pydantic_testcase!(
    test_pyrefly_strict_default,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
    x: int = Field()
    y: int = Field(strict=True)
Model(x='0', y=1) 
Model(x='0', y='1') # E: Argument `Literal['1']` is not assignable to parameter `y` with type `int` in function `Model.__init__` 
"#,
);

pydantic_testcase!(
    test_class_keyword,
    r#"
from pydantic import BaseModel, Field

class Model1(BaseModel, strict=True):
    x: int = Field(0)
    y: int = Field(0, strict=False)
# `x` is strict
Model1(x=0)
Model1(x='0')  # E: `Literal['0']` is not assignable to parameter `x`
# `y` is lax
Model1(y=0)
Model1(y='0')

class Model2(BaseModel, strict=False):
    x: int = Field(0)
    y: int = Field(0, strict=True)
# `x` is lax
Model2(x=0)
Model2(x='0')
# `y` is strict
Model2(y=0)
Model2(y='0')  # E: `Literal['0']` is not assignable to parameter `y`
    "#,
);

pydantic_testcase!(
    test_configdict,
    r#"
from pydantic import BaseModel, ConfigDict
class Model(BaseModel):
    x: int
    model_config = ConfigDict(strict=True)
Model(x=0)
Model(x='0')  # E: `Literal['0']` is not assignable to parameter `x`
    "#,
);

pydantic_testcase!(
    test_multiple_strict_values,
    r#"
from pydantic import BaseModel, ConfigDict

# When `strict` appears in both the class keywords and model_config, the keyword wins
class Model1(BaseModel, strict=True):
    x: int
    model_config = ConfigDict(strict=False)
Model1(x=0)
Model1(x='0')  # E: `Literal['0']` is not assignable to parameter `x`
class Model2(BaseModel, strict=False):
    x: int
    model_config = ConfigDict(strict=True)
Model2(x=0)
Model2(x='0')
    "#,
);

pydantic_testcase!(
    test_inherit,
    r#"
from pydantic import BaseModel
class Model1(BaseModel, strict=True):
    pass
class Model2(Model1):
    x: int
Model2(x=0)
Model2(x='0')  # E: `Literal['0']` is not assignable to parameter `x`
    "#,
);

pydantic_testcase!(
    test_lax_mode_coercion,
    r#"
from pydantic import BaseModel
from typing import reveal_type
from decimal import Decimal

class Model(BaseModel):
    x: int = 0

reveal_type(Model.__init__)  # E: revealed type: (self: Model, *, x: Decimal | bool | bytes | float | int | str = ..., **Unknown) -> None

# int field accepts: int, bool, float, str, bytes, Decimal
Model(x=1)
Model(x=True)
Model(x=1.0)
Model(x='123')
Model(x=b'123')
Model(x=Decimal('123'))
    "#,
);

pydantic_testcase!(
    bug = "An error should be raised here",
    test_lax_mode_coercion_literals,
    r#"
from typing import Literal
from pydantic import BaseModel

class Model1(BaseModel):
    status: Literal[1]

m = Model1(status="1")
    "#,
);

pydantic_testcase!(
    bug = "Revealed type should be more informative and conversion table should be applied to containers",
    test_lax_mode_coercion_container,
    r#"
from typing import List, reveal_type

from pydantic import BaseModel

class Model(BaseModel):
    x: List[int] = [0, 1]

reveal_type(Model.__init__) # E: revealed type: (self: Model, *, x: Any = ..., **Unknown) -> None
    "#,
);
