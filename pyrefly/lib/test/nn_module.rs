/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn torch_env() -> TestEnv {
    TestEnv::one(
        "torch.nn",
        r#"
class Tensor:
    def tolist(self) -> list[int]: ...

class Parameter(Tensor): ...

class Module:
    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None: ...
    def register_parameter(self, name: str, param: Parameter | None) -> None: ...
"#,
    )
}

testcase!(
    test_nn_module_register_buffer_and_parameter,
    torch_env(),
    r#"
from torch.nn import Module, Parameter, Tensor
from typing import assert_type

class Model(Module):
    def __init__(self) -> None:
        self.register_buffer("values", Tensor())
        self.register_parameter(name="weight", param=Parameter())

model = Model()
assert_type(model.values, Tensor)
assert_type(model.weight, Parameter)
model.values.tolist()
Model.values  # E: Instance-only attribute `values` of class `Model` is not visible on the class
"#,
);

testcase!(
    test_register_buffer_is_only_special_for_nn_module,
    torch_env(),
    r#"
from torch.nn import Tensor

class NotAModule:
    def register_buffer(self, name: str, value: Tensor) -> None: ...

    def __init__(self) -> None:
        self.register_buffer("values", Tensor())

NotAModule().values  # E: Object of class `NotAModule` has no attribute `values`
"#,
);
