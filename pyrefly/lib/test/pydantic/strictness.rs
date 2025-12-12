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
from typing import Callable, reveal_type
from decimal import Decimal
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import UUID

class Model(BaseModel):
    x: int = 0

reveal_type(Model.__init__)  # E: revealed type: (self: Model, *, x: LaxInt = ..., **Unknown) -> None

# int field accepts: int, bool, float, str, bytes, Decimal
Model(x=1)
Model(x=True)
Model(x=1.0)
Model(x='123')
Model(x=b'123')
Model(x=Decimal('123'))


class Model2(BaseModel):
    x: bytes

reveal_type(Model2.__init__)  # E: revealed type: (self: Model2, *, x: LaxBytes, **Unknown) -> None

class Model3(BaseModel):
    func: Callable[[int], str]

reveal_type(Model3.__init__)  # E: revealed type: (self: Model3, *, func: Any, **Unknown) -> None

class Model4(BaseModel):
    d: date

reveal_type(Model4.__init__)  # E: revealed type: (self: Model4, *, d: LaxDate, **Unknown) -> None

class Model5(BaseModel):
    dt: datetime

reveal_type(Model5.__init__)  # E: revealed type: (self: Model5, *, dt: LaxDatetime, **Unknown) -> None

class Model6(BaseModel):
    t: time

reveal_type(Model6.__init__)  # E: revealed type: (self: Model6, *, t: LaxTime, **Unknown) -> None

class Model7(BaseModel):
    td: timedelta

reveal_type(Model7.__init__)  # E: revealed type: (self: Model7, *, td: LaxTimedelta, **Unknown) -> None

class Model8(BaseModel):
    dec: Decimal

reveal_type(Model8.__init__)  # E: revealed type: (self: Model8, *, dec: LaxDecimal, **Unknown) -> None

class Model9(BaseModel):
    p: Path

reveal_type(Model9.__init__)  # E: revealed type: (self: Model9, *, p: LaxPath, **Unknown) -> None

class Model10(BaseModel):
    u: UUID

reveal_type(Model10.__init__)  # E: revealed type: (self: Model10, *, u: LaxUUID, **Unknown) -> None
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
    test_lax_mode_coercion_container,
    r#"
from typing import List, Sequence, Iterable, reveal_type
from collections import deque

from pydantic import BaseModel

class Model(BaseModel):
    x: List[int] = [0, 1]

reveal_type(Model.__init__) # E: revealed type: (self: Model, *, x: Iterable[LaxInt] = ..., **Unknown) -> None

class Model2(BaseModel):
    q: deque[int]

reveal_type(Model2.__init__) # E: revealed type: (self: Model2, *, q: Iterable[LaxInt], **Unknown) -> None

class Model3(BaseModel):
    d: dict[str, int]

reveal_type(Model3.__init__) # E: revealed type: (self: Model3, *, d: Mapping[str, LaxInt], **Unknown) -> None

class Model4(BaseModel):
    f: frozenset[int]

reveal_type(Model4.__init__) # E: revealed type: (self: Model4, *, f: Iterable[LaxInt], **Unknown) -> None

class Model5(BaseModel):
    s: set[int]

reveal_type(Model5.__init__) # E: revealed type: (self: Model5, *, s: Iterable[LaxInt], **Unknown) -> None

class Model6(BaseModel):
    t: Iterable[int]

reveal_type(Model6.__init__) # E: revealed type: (self: Model6, *, t: Iterable[LaxInt], **Unknown) -> None

class Model7(BaseModel):
    seq: Sequence[int]

reveal_type(Model7.__init__) # E: revealed type: (self: Model7, *, seq: Iterable[LaxInt], **Unknown) -> None

class Model8(BaseModel):
    t: tuple[int, ...]

reveal_type(Model8.__init__) # E: revealed type: (self: Model8, *, t: Iterable[LaxInt], **Unknown) -> None

class Model9(BaseModel):
    fixed: tuple[int, str, bool]

reveal_type(Model9.__init__) # E: revealed type: (self: Model9, *, fixed: Iterable[Decimal | bool | bytearray | bytes | float | int | str], **Unknown) -> None
    "#,
);

pydantic_testcase!(
    test_lax_mode_coercion_union,
    r#"
from typing import List, reveal_type
from decimal import Decimal

from pydantic import BaseModel

class Model(BaseModel):
    y: int | bool

reveal_type(Model.__init__)  # E: revealed type: (self: Model, *, y: Decimal | bool | bytes | float | int | str, **Unknown) -> None
    "#,
);

pydantic_testcase!(
    test_lax_mode_list_and_set_invariance,
    r#"
from pydantic import BaseModel
from collections import deque
from typing import reveal_type

class TestModel(BaseModel):
    name: str
    things: list[str]
    tags: set[str]

list_of_things = ["thing1", "thing2"]
set_of_tags = {"tag1", "tag2"}
a = TestModel(name="test", things=list_of_things, tags=set_of_tags)

deque_of_bytes: deque[bytes] = deque([b"thing1", b"thing2"])
b = TestModel(name="test", things=deque_of_bytes, tags=set_of_tags)

# When reading the field back, you get the original declared type (list[str]), not Iterable[LaxStr]
reveal_type(a.things)  # E: revealed type: list[str]
reveal_type(a.tags)    # E: revealed type: set[str]
    "#,
);

pydantic_testcase!(
    test_lax_mode_dict_invariance,
    r#"
from pydantic import BaseModel
from typing import reveal_type

class TestModel(BaseModel):
    name: str
    metadata: dict[str, str]

my_dict = {"key1": "value1", "key2": "value2"}
a = TestModel(name="test", metadata=my_dict)
    "#,
);

pydantic_testcase!(
    test_lax_mode_frozenset_invariance,
    r#"
from pydantic import BaseModel

class TestModel(BaseModel):
    items: frozenset[str]

my_frozenset = frozenset({"a", "b"})
a = TestModel(items=my_frozenset)

my_list = ["a", "b"]
b = TestModel(items=my_list)

my_set = {"a", "b"}
c = TestModel(items=my_set)
    "#,
);

pydantic_testcase!(
    test_lax_mode_other,
    r#"
from pydantic import BaseModel
from typing import Any, reveal_type

class Model1(BaseModel):
    x: None

reveal_type(Model1.__init__)  # E: revealed type: (self: Model1, *, x: None, **Unknown) -> None

class Model2(BaseModel):
    y: Any

reveal_type(Model2.__init__)  # E: revealed type: (self: Model2, *, y: Any, **Unknown) -> None
    "#,
);

pydantic_testcase!(
    test_lax_mode_type_expansion,
    r#"
from pydantic import BaseModel
from typing import reveal_type

class Model1(BaseModel):
    t: type[int]

reveal_type(Model1.__init__)  # E: revealed type: (self: Model1, *, t: type[LaxInt], **Unknown) -> None

    "#,
);
