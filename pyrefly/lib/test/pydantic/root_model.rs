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
    bug = "We should error m2 only because the argument type is inconsistent with int",
    test_root_model_basic,
    pydantic_env(),
    r#"
from pydantic import RootModel
class IntRootModel(RootModel[int]):
   pass
m1 = IntRootModel(123) # E: Expected argument `root` to be passed by name in function `IntRootModel.__init__`
m2 = IntRootModel("abc") # E: Expected argument `root` to be passed by name in function `IntRootModel.__init__`
"#,
);

testcase!(
    bug = "We should not error on anything here since we have a generic type param",
    test_root_model_generic,
    pydantic_env(),
    r#"
from pydantic import RootModel
class GenericRootModel[T](RootModel[T]):
   pass
m1 = GenericRootModel(123) # E: Expected argument `root` to be passed by name in function `GenericRootModel.__init__`
m2 = GenericRootModel("abc") # E: Expected argument `root` to be passed by name in function `GenericRootModel.__init__`
"#,
);

testcase!(
    bug = "We should expect one positional argument",
    test_root_model_wrong_args,
    pydantic_env(),
    r#"
from pydantic import RootModel
class TwoArgRootModel[F, G](RootModel[F, G]): # E: Expected 1 type argument for `RootModel`, got 2 
    pass
m1 = TwoArgRootModel(123, "abc") # E: Expected 0 positional arguments, got 2 in function `TwoArgRootModel.__init__` # E: Expected argument `root` to be passed by name in function `TwoArgRootModel.__init__` 
"#,
);

testcase!(
    bug = "Zero arguments is acceptable",
    test_no_args,
    pydantic_env(),
    r#"
from pydantic import RootModel

class ZeroArgRootModel(RootModel):
    pass
m1 = ZeroArgRootModel() # E:  Missing argument `root` in function `ZeroArgRootModel.__init__`
"#,
);

testcase!(
    bug = "when no args are specified, fallback to the default generic arg",
    test_fallback,
    pydantic_env(),
    r#"
from pydantic import RootModel

class FallBackRootModel(RootModel):
    pass

m1 = FallBackRootModel(123) # E: Expected argument `root` to be passed by name in function `FallBackRootModel.__init__` 
"#,
);

testcase!(
    bug = "Rootmodel info should be propagated through the class heirchy",
    test_inheritance,
    pydantic_env(),
    r#"
from pydantic import RootModel

class A(RootModel[int]):
    pass

class B(A):
    pass

m1 = B(3) # E: Expected argument `root` to be passed by name in function `B.__init__`
m2 = B("abc") # E: Expected argument `root` to be passed by name in function `B.__init__`
"#,
);
