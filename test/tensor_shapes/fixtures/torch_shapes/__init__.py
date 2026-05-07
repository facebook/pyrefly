# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Runtime implementation of shape typing constructs.

The .pyi stub provides full type information to pyrefly. This .py file
provides minimal runtime classes so that annotations using these types
don't crash when evaluated by Python.
"""

import typing

import torch
import torch.nn as nn

# Make torch types subscriptable at runtime so that annotations like
# Tensor[B, T, N] or nn.Linear[In, Out] evaluate as no-ops instead of
# crashing with "type is not subscriptable".
# All nn.Module subclasses and Tensor that need to be subscriptable at runtime.
_subscriptable_classes = [
    torch.Tensor,
    nn.Embedding,
    nn.Linear,
    nn.ModuleList,
    # Convolution modules
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    # Pooling modules
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
]
for _cls in _subscriptable_classes:
    if not hasattr(_cls, "__class_getitem__"):
        _cls.__class_getitem__ = classmethod(lambda cls, params: cls)


class Dim[T]:
    """Symbolic integer type for dimension values.

    At runtime this is a no-op generic class. The type checker uses the
    .pyi stub for shape inference.
    """

    pass


class TypeVar:
    """TypeVar with arithmetic support for tensor shape dimensions.

    Like typing.TypeVar but arithmetic operations (N + 1, N * 2, etc.)
    return self instead of raising TypeError. Setting
    __class__ = typing.TypeVar makes isinstance(x, typing.TypeVar)
    return True, so Generic[N] and TypedDict + Generic[N] both work.

    In pyrefly, torch_shapes.TypeVar is treated identically to
    typing.TypeVar.
    """

    __class__ = typing.TypeVar

    def __init__(self, name: str):
        self.__name__ = name
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __floordiv__(self, other):
        return self

    def __typing_subst__(self, arg):
        return arg


class TypeVarTuple:
    """TypeVarTuple with support for integer shape dimensions.

    Like typing.TypeVarTuple but for use in tensor shape annotations.
    Setting __class__ = typing.TypeVarTuple and providing
    __typing_is_unpacked_typevartuple__ makes Generic[*Ns] work.

    In pyrefly, torch_shapes.TypeVarTuple is treated identically to
    typing.TypeVarTuple.

    __iter__ yields self so that *Ns unpacking works in subscripts
    like Generic[*Ns] or Tensor[*Ns, 3]. Python's star-unpacking
    calls __iter__ on the object.
    """

    __class__ = typing.TypeVarTuple

    def __init__(self, name: str):
        self.__name__ = name
        self.name = name

    def __repr__(self):
        return f"*{self.name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other

    def __iter__(self):
        yield self

    @property
    def __typing_is_unpacked_typevartuple__(self):
        return True
