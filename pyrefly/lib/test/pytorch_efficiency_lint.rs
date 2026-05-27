/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn env() -> TestEnv {
    let mut e = TestEnv::new();
    e.add(
        "torch",
        r#"
class device: ...

class Size(tuple[int, ...]): ...

class Tensor:
    shape: Size
    def item(self) -> int | float: ...
    def sum(self) -> "Tensor": ...
    def to(self, device: device) -> "Tensor": ...
    def cuda(self) -> "Tensor": ...

def zeros(*size: int, device: device | None = None) -> Tensor: ...
def ones(*size: int, device: device | None = None) -> Tensor: ...
def empty(*size: int, device: device | None = None) -> Tensor: ...
def randn(*size: int, device: device | None = None) -> Tensor: ...
def rand(*size: int, device: device | None = None) -> Tensor: ...
def full(size: tuple[int, ...], fill_value: float, device: device | None = None) -> Tensor: ...
def arange(start: float, end: float | None = None, step: float = 1, device: device | None = None) -> Tensor: ...
"#,
    );
    e
}

fn env_with_lint() -> TestEnv {
    env().enable_pytorch_efficiency_lint_error()
}

testcase!(
    test_tensor_item_call,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    v = x.item()  # E: `Tensor.item()` causes implicit GPU-to-CPU synchronization
"#,
);

testcase!(
    test_tensor_item_call_disabled_by_default,
    env(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    v = x.item()
"#,
);

testcase!(
    test_non_tensor_item_call_ok,
    env_with_lint(),
    r#"
class Foo:
    def item(self) -> int:
        return 0

def f(x: Foo) -> None:
    v = x.item()
"#,
);

testcase!(
    test_tensor_item_with_args_ok,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    v = x.item(0)  # E: Expected 0 positional argument
"#,
);

testcase!(
    test_tensor_other_method_ok,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    v = x.sum()
"#,
);

// --- C3: Redundant .to(device) on tensor factory calls ---

testcase!(
    test_redundant_to_on_zeros,
    env_with_lint(),
    r#"
import torch

def f(device: torch.device) -> None:
    x = torch.zeros(3, 4).to(device)  # E: `torch.zeros(...).to(device)` creates the tensor on CPU first, then copies it
"#,
);

testcase!(
    test_redundant_to_on_randn,
    env_with_lint(),
    r#"
import torch

def f(device: torch.device) -> None:
    x = torch.randn(3, 4).to(device)  # E: `torch.randn(...).to(device)` creates the tensor on CPU first, then copies it
"#,
);

testcase!(
    test_to_on_non_factory_ok,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor, device: torch.device) -> None:
    y = x.to(device)
"#,
);

testcase!(
    test_redundant_to_disabled_by_default,
    env(),
    r#"
import torch

def f(device: torch.device) -> None:
    x = torch.zeros(3, 4).to(device)
"#,
);

testcase!(
    test_direct_device_arg_ok,
    env_with_lint(),
    r#"
import torch

def f(device: torch.device) -> None:
    x = torch.zeros(3, 4, device=device)
"#,
);

testcase!(
    test_to_dtype_on_factory_with_device_ok,
    env_with_lint(),
    r#"
import torch

def f(device: torch.device, other_device: torch.device) -> None:
    x = torch.randn(3, 4, device=device).to(other_device)
"#,
);

// --- C7: Deprecated .cuda() calls ---

testcase!(
    test_tensor_cuda_call,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    y = x.cuda()  # E: `Tensor.cuda()` hard-codes the target device
"#,
);

testcase!(
    test_non_tensor_cuda_call_ok,
    env_with_lint(),
    r#"
class Foo:
    def cuda(self) -> "Foo":
        return self

def f(x: Foo) -> None:
    y = x.cuda()
"#,
);

testcase!(
    test_tensor_cuda_disabled_by_default,
    env(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    y = x.cuda()
"#,
);

// --- C5: Printing tensors ---

testcase!(
    test_print_tensor,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    print(x)  # E: printing a `Tensor` causes implicit GPU-to-CPU synchronization
"#,
);

testcase!(
    test_print_non_tensor_ok,
    env_with_lint(),
    r#"
def f(x: int) -> None:
    print(x)
"#,
);

testcase!(
    test_print_tensor_shape_ok,
    env_with_lint(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    print(x.shape)
"#,
);

testcase!(
    test_print_tensor_disabled_by_default,
    env(),
    r#"
import torch

def f(x: torch.Tensor) -> None:
    print(x)
"#,
);
