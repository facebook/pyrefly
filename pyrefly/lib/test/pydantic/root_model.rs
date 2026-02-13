/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    test_root_model_basic,
    r#"
from pydantic import RootModel
class IntRootModel(RootModel[int]):
   pass
m1 = IntRootModel(123)
m2 = IntRootModel("abc")
m3 = IntRootModel(root=123)
m4 = IntRootModel()  # E: Missing argument `root`
m5 = IntRootModel(123, 456)  # E: Expected 1 positional argument, got 2
"#,
);

pydantic_testcase!(
    test_root_model_generic,
    r#"
from pydantic import RootModel
class GenericRootModel[T](RootModel[T]):
   pass
m1 = GenericRootModel(123)
m2 = GenericRootModel("abc")
"#,
);

pydantic_testcase!(
    test_root_model_wrong_args,
    r#"
from pydantic import RootModel
class TwoArgRootModel[F, G](RootModel[F, G]): # E: Expected 1 type argument for `RootModel`, got 2
    pass
m1 = TwoArgRootModel(123, "abc") # E: Expected 1 positional argument, got 2 in function `TwoArgRootModel.__init__`
"#,
);

pydantic_testcase!(
    test_zero_to_one_args,
    r#"
from pydantic import RootModel

class ZeroArgRootModel(RootModel):
    pass
m1 = ZeroArgRootModel()
m2 = ZeroArgRootModel(123)
m3 = ZeroArgRootModel(root=123)
m4 = ZeroArgRootModel(123, 456)  # E: Expected 1 positional argument, got 2
"#,
);

pydantic_testcase!(
    test_fallback,
    r#"
from pydantic import RootModel

class FallBackRootModel(RootModel):
    pass

m1 = FallBackRootModel(123)
"#,
);

pydantic_testcase!(
    test_inheritance,
    r#"
from pydantic import RootModel

class A(RootModel[int]):
    pass

class B(A):
    pass

m1 = B(3)
m2 = B("abc")
"#,
);

pydantic_testcase!(
    test_inheritance_kwarg,
    r#"
from pydantic import RootModel

class A(RootModel[int]):
    pass

class B(A):
    pass

m1 = B(root=3)
m2 = B(root="abc")
m3 = B(3)
"#,
);

pydantic_testcase!(
    bug =
        "we should not error on the call with a str argument because it could be coercible to int ",
    test_directly_use_root_model,
    r#"
from typing import Any, assert_type
from pydantic import RootModel

m1 = RootModel()
assert_type(m1, RootModel[Any])
m2 = RootModel(5)
assert_type(m2, RootModel[int])
RootModel(5, extra=6)  # E: Unexpected keyword argument `extra`

m3 = RootModel[int](5)
assert_type(m3, RootModel[int])
RootModel[int]("") # E: Argument `Literal['']` is not assignable to parameter `root` with type `int` in function `pydantic.root_model.RootModel.__init__` 
RootModel[int](5, extra=6)  # E: Unexpected keyword argument `extra`
    "#,
);

pydantic_testcase!(
    test_root_model_field,
    r#"
from typing import assert_type
from pydantic import BaseModel, RootModel

class RootModel1(RootModel[int]):
    pass

class Model1(BaseModel, strict=True):
    x: RootModel1
m1 = Model1(x=0)
assert_type(m1.x, RootModel1)
m2 = Model1(x=RootModel1(0))
assert_type(m2.x, RootModel1)
Model1(x='oops')  # E: `Literal['oops']` is not assignable to parameter `x`

class Model2(BaseModel, strict=True):
    x: RootModel
m3 = Model2(x=0)
assert_type(m3.x, RootModel)
m4 = Model2(x=RootModel(0))
assert_type(m4.x, RootModel)

class Model3(BaseModel, strict=True):
    x: RootModel[int]
m5 = Model3(x=0)
assert_type(m5.x, RootModel[int])
m6 = Model3(x=RootModel(0))
assert_type(m6.x, RootModel[int])
Model3(x='oops')  # E: `Literal['oops']` is not assignable to parameter `x`
Model3(x=RootModel('oops')) # E: Argument `Literal['oops']` is not assignable to parameter `root` with type `int` in function `pydantic.root_model.RootModel.__init__`
    "#,
);

pydantic_testcase!(
    test_strict_basic,
    r#"
from pydantic import RootModel, StrictInt 

class A(RootModel[StrictInt]):
    pass

class B(A):
    pass

m1 = B(3)
m2 = B("1") # E: Argument `Literal['1']` is not assignable to parameter `root` with type `int` in function `B.__init__`
m3 = A("1") # E: Argument `Literal['1']` is not assignable to parameter `root` with type `int` in function `A.__init__
"#,
);

pydantic_testcase!(
    test_root_model_strict,
    r#"
from pydantic import BaseModel, RootModel

class IntRootModel(RootModel[int]):
    pass

class Model(BaseModel, strict=True):
    x: IntRootModel

m1 = Model(x=IntRootModel(123))
m2 = Model(x=123) 
m3 = Model(x=IntRootModel(123))
    "#,
);

pydantic_testcase!(
    test_multiple_root_models_in_union,
    r#"
from pydantic import BaseModel, RootModel

class IntRootModel(RootModel[int]):
    pass

class StrRootModel(RootModel[str]):
    pass

class Model(BaseModel, strict=True):
    x: IntRootModel | StrRootModel

# valid types
m1 = Model(x=IntRootModel(5))
m2 = Model(x=StrRootModel("hello"))
m3 = Model(x=5)  
m4 = Model(x="hello") 

# Invalid types 
Model(x=5.5)  # E: Argument `float` is not assignable to parameter `x` with type `IntRootModel | StrRootModel | int | str` in function `Model.__init__`
Model(x=[])   # E: Argument `list[@_]` is not assignable to parameter `x` with type `IntRootModel | StrRootModel | int | str` in function `Model.__init__
    "#,
);

pydantic_testcase!(
    test_optional_root_model_fields,
    r#"
from typing import Optional
from pydantic import BaseModel, RootModel

class IntRootModel(RootModel[int]):
    pass

class Model(BaseModel):
    x: Optional[IntRootModel] = None

m1 = Model()
m2 = Model(x=None)
m3 = Model(x=5)
m4 = Model(x=IntRootModel(5))
    "#,
);

pydantic_testcase!(
    test_nested_root_models,
    r#"
from pydantic import BaseModel, RootModel

class InnerModel(RootModel[int]):
    pass

class OuterModel(RootModel[InnerModel]):
    pass

class Model(BaseModel, strict=True):
    x: OuterModel

m1 = Model(x=OuterModel(InnerModel(5)))
m2 = Model(x=InnerModel(5))  
m3 = Model(x=5)
    "#,
);
