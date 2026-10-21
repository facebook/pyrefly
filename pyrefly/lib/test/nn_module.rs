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

// Regression test for https://github.com/facebook/pyrefly/issues/5208
// The registration signature acts like an attribute declaration when the value is rejected.
testcase!(
    test_nn_module_registration_rejected_value_falls_back_to_parameter_type,
    torch_env(),
    r#"
from torch.nn import Module, Parameter, Tensor
from typing import assert_type

class Model(Module):
    def __init__(
        self,
        values: tuple[int, ...] | None,
        weight: tuple[int, ...] | None,
    ) -> None:
        self.register_buffer("values", values)  # E: Argument `tuple[int, ...] | None` is not assignable to parameter `tensor` with type `Tensor | None`
        self.register_parameter("weight", weight)  # E: Argument `tuple[int, ...] | None` is not assignable to parameter `param` with type `Parameter | None`

class NonOptionalModel(Module):
    def __init__(self, values: list[int]) -> None:
        self.register_buffer("values", values)  # E: Argument `list[int]` is not assignable to parameter `tensor` with type `Tensor | None`

model = Model(None, None)
assert_type(model.values, Tensor | None)
assert_type(model.weight, Parameter | None)
assert_type(NonOptionalModel([]).values, Tensor | None)
"#,
);

testcase!(
    test_nn_module_registration_uses_overridden_method_signature,
    torch_env(),
    r#"
from torch.nn import Module, Tensor
from typing import assert_type

class WiderModule(Module):
    def register_buffer(
        self,
        name: str,
        tensor: Tensor | list[int] | None,
        persistent: bool = True,
    ) -> None: ...

class Model(WiderModule):
    def __init__(self) -> None:
        self.register_buffer("values", [1, 2, 3])

assert_type(Model().values, list[int])
"#,
);

testcase!(
    test_nn_module_registration_allows_renamed_value_parameter,
    TestEnv::one(
        "torch.nn",
        r#"
class Parameter: ...

class Module:
    def register_parameter(
        self, name: str, parameter: Parameter | None
    ) -> None: ...
"#,
    ),
    r#"
from torch.nn import Module, Parameter
from typing import assert_type

class Model(Module):
    def __init__(self, value: tuple[int, ...] | None) -> None:
        self.register_parameter("value", value)  # E: Argument `tuple[int, ...] | None` is not assignable to parameter `parameter` with type `Parameter | None`

assert_type(Model(None).value, Parameter | None)
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
