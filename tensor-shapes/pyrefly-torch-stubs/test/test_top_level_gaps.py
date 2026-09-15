# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for newly filled top-level Torch stub gaps."""

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import IntVar

if TYPE_CHECKING:
    import torch._tensor
    import torch.linalg
    import torch.nn
    import torch.nn.modules.module
    import torch.nn.parameter
    from torch import Tensor


@torch.inference_mode()
def inference_identity[N: IntVar](x: Tensor[[N]]) -> Tensor[[N]]:
    return x


def test_pi[N: IntVar](x: Tensor[[N]]):
    assert_type(torch.pi, float)
    y = x * torch.pi
    assert_type(y, Tensor[[N]])


def test_inverse_trig[N: IntVar](x: Tensor[[N]]):
    assert_type(torch.asin(x), Tensor[[N]])
    assert_type(torch.arcsin(x), Tensor[[N]])
    assert_type(torch.atan(x), Tensor[[N]])


def test_eye_kwargs():
    assert_type(torch.eye(3, dtype=torch.float32, device="cpu"), Tensor[[3, 3]])
    assert_type(torch.eye(3, 4), Tensor[[3, 4]])


def test_numpy():
    assert_type(torch.zeros(2, 2).numpy(), Any)


def test_rand_generator():
    g = torch.Generator()
    assert_type(torch.rand(2, 2, generator=g), Tensor[[2, 2]])
    assert_type(torch.rand((2, 2), generator=g), Tensor[[2, 2]])
    assert_type(torch.randint(0, 10, (3,), generator=g), Tensor[[3]])


def test_random_method_generator():
    x = torch.zeros(2, 2)
    assert_type(x.random_(generator=torch.Generator()), Tensor[[2, 2]])


def test_randperm():
    assert_type(torch.randperm(5), Tensor)
    assert_type(torch.randperm(5, generator=torch.Generator()), Tensor)


def test_new_zeros_size():
    x = torch.zeros(2, 3)
    assert_type(
        x.new_zeros(
            x.shape + (3, 3),
            layout=None,
            pin_memory=False,
        ),
        Tensor,
    )
    assert_type(x.new_zeros(2, 3, dtype=torch.float32, device="cpu"), Tensor)


def test_new_ones_size():
    x = torch.zeros(2, 3)
    assert_type(x.new_ones((2, 3), layout=None, pin_memory=False), Tensor)
    assert_type(x.new_ones(2, 3), Tensor)


def test_where_scalar_on_either_side():
    cond = torch.zeros(2, 2) > 0
    x = torch.zeros(2, 2)
    assert_type(torch.where(cond, x, 0.0), Tensor)
    assert_type(torch.where(cond, 0.0, x), Tensor)
    assert_type(torch.where(cond, 0.0, 1.0), Tensor)


def test_sequence_creation_sizes(size: list[int]):
    assert_type(torch.rand(size), Tensor)
    assert_type(torch.zeros(size), Tensor)
    assert_type(torch.ones(size), Tensor)
    assert_type(torch.full(size, 1.0), Tensor)


def test_gradual_creation_and_stack(x: Tensor, tensors: list[Tensor]):
    assert_type(torch.zeros_like(x, dtype=torch.float32), Tensor)
    assert_type(torch.stack(tensors), Tensor)


def test_linalg_norm():
    x = torch.zeros(3, 3)
    assert_type(torch.linalg.norm(x, dim=-1), Tensor)


def test_deg2rad_inplace():
    x = torch.zeros(3)
    assert_type(x.deg2rad_(), Tensor[[3]])


def test_tensor_mod():
    x = torch.zeros(4)
    assert_type(x % 2, Tensor[[4]])


def test_reduction_methods[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    assert_type(x.all(dim=0), Tensor[[M]])
    assert_type(x.any(dim=0, keepdim=True), Tensor[[1, M]])


def test_scalar_arithmetic[N: IntVar](x: Tensor[[N]]):
    assert_type(x.mul(2.0), Tensor[[N]])
    assert_type(torch.pow(2.0, x), Tensor[[N]])


def test_gradual_ordering[N: IntVar](x: Tensor[[N]]):
    assert_type(x > 0, Any)


def test_tensor_clamp_bounds[N: IntVar](x: Tensor[[N]]):
    assert_type(x.clamp(min=x, max=x), Tensor[[N]])
    assert_type(torch.clamp(x, min=x, max=x), Tensor[[N]])


def test_uniform_generator[N: IntVar](x: Tensor[[N]]):
    assert_type(x.uniform_(generator=torch.Generator()), Tensor[[N]])


def test_gradual_interpolate_options(
    x: Tensor,
    size: tuple[int, int] | None,
    scale_factor: float | None,
):
    assert_type(F.interpolate(x, size=size, scale_factor=scale_factor), Tensor)


def test_module_containers(
    modules: torch.nn.ModuleList[torch.nn.Module],
    sequential: torch.nn.Sequential,
):
    assert_type(modules[1:], torch.nn.ModuleList[torch.nn.Module])
    for module in sequential:
        assert_type(module, torch.nn.Module)


def test_common_explicit_apis[N: IntVar](x: Tensor[[N]]):
    assert_type(torch.allclose(x, x, rtol=1e-4, equal_nan=True), bool)
    assert_type(torch.mul(x, 2), Tensor[[N]])
    assert_type(torch.isnan(x), Tensor[[N]])
    assert_type(torch.argsort(x), Tensor[[N]])
    assert_type(torch.expm1(x), Tensor[[N]])
    assert_type(torch.log10(x), Tensor[[N]])
    assert_type(torch.sign(x), Tensor[[N]])
    assert_type(torch.quantile(x, 0.5), Tensor)
    assert_type(torch.diagonal(torch.zeros(2, 2)), Tensor)
    assert_type(torch.linalg.vector_norm(x), Tensor)
    assert_type(inference_identity(x), Tensor[[N]])
    assert_type(torch.__version__, str)


def test_common_explicit_tensor_members[N: IntVar](x: Tensor[[N]]):
    assert_type(x.zero_(), Tensor[[N]])
    assert_type(x.add_(1.0), Tensor[[N]])
    assert_type(x.pin_memory(), Tensor[[N]])
    assert_type(x.byte(), Tensor[[N]])
    assert_type(x.reshape_as(torch.zeros(2, 3)), Tensor[[2, 3]])
    assert_type(x.unique(), Any)
    assert_type(torch.zeros(2, 2).diagonal(), Tensor)
    assert_type(x.data_ptr(), int)
    assert_type(x.is_contiguous(), bool)
    assert_type(x.grad, Any)


def test_private_tensor_and_parameter_are_usable_types(
    x: torch._tensor.Tensor,
    parameter: torch.nn.Parameter,
    module: torch.nn.modules.module.Module,
):
    assert_type(x, Tensor)
    assert_type(parameter, torch.nn.Parameter)
    assert_type(module, torch.nn.Module)
    assert_type(torch.nn.DataParallel, Any)
    assert_type(torch._tensor._convert, Any)
    assert_type(torch.nn.modules.module.register_module_forward_hook, Any)
    assert_type(torch.nn.parameter.UninitializedParameter, Any)


def test_out_of_memory_error_is_a_type(error: torch.OutOfMemoryError):
    assert_type(error, torch.OutOfMemoryError)


def test_concatenate():
    x = torch.zeros(2, 2)
    assert_type(torch.concatenate([x, x], dim=0), Tensor[[4, 2]])


def test_as_tensor_from_numpy():
    assert_type(torch.as_tensor([1.0, 2.0], dtype=torch.float32), Tensor)
    assert_type(torch.from_numpy(torch.zeros(2).numpy()), Tensor)


def test_tensor_float():
    assert_type(float(torch.zeros(())), float)
    assert_type(float(torch.zeros(1)), float)
    assert_type(float(torch.zeros(1, 1)), float)
