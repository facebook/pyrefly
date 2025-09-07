/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn pydantic_env() -> TestEnv {
    let path = std::env::var("PYDANTIC_TEST_PATH").expect("PYDANTIC_TEST_PATH must be set");
    TestEnv::new_with_site_package_path(&path)
}

testcase!(
    test_root_model_basic,
    pydantic_env(),
    r#"
from pydantic import RootModel
class IntRootModel(RootModel[int]):
   pass
m1 = IntRootModel(123) 
m2 = IntRootModel("abc") # E: Argument `Literal['abc']` is not assignable to parameter `root` with type `int` in function `IntRootModel.__init__` 
"#,
);

testcase!(
    test_root_model_generic,
    pydantic_env(),
    r#"
from pydantic import RootModel
class GenericRootModel[T](RootModel[T]):
   pass
m1 = GenericRootModel(123)
m2 = GenericRootModel("abc")
"#,
);

testcase!(
    test_root_model_wrong_args,
    pydantic_env(),
    r#"
from pydantic import RootModel
class TwoArgRootModel[F, G](RootModel[F, G]): # E: Expected 1 type argument for `RootModel`, got 2
    pass
m1 = TwoArgRootModel(123, "abc") # E: Expected 1 positional argument, got 2 in function `TwoArgRootModel.__init__`
"#,
);

testcase!(
    test_no_args,
    pydantic_env(),
    r#"
from pydantic import RootModel

class ZeroArgRootModel(RootModel):
    pass
m1 = ZeroArgRootModel()
"#,
);

testcase!(
    test_fallback,
    pydantic_env(),
    r#"
from pydantic import RootModel

class FallBackRootModel(RootModel):
    pass

m1 = FallBackRootModel(123)
"#,
);

testcase!(
    test_inheritance,
    pydantic_env(),
    r#"
from pydantic import RootModel

class A(RootModel[int]):
    pass

class B(A):
    pass

m1 = B(3)
m2 = B("abc") # E: Argument `Literal['abc']` is not assignable to parameter `root` with type `int` in function `B.__init__`
"#,
);

testcase!(
    test_inheritance_kwarg,
    pydantic_env(),
    r#"
from pydantic import RootModel

class A(RootModel[int]):
    pass

class B(A):
    pass

m1 = B(root=3)
m2 = B(root="abc") # E: Argument `Literal['abc']` is not assignable to parameter `root` with type `int` in function `B.__init__`
m3 = B(3)
"#,
);
