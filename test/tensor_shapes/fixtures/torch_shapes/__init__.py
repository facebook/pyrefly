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

import torch
import torch.nn as nn

# Make torch types subscriptable at runtime so that annotations like
# Tensor[B, T, N] or nn.Linear[In, Out] evaluate as no-ops instead of
# crashing with "type is not subscriptable".
for _cls in (torch.Tensor, nn.Embedding, nn.Linear, nn.ModuleList):
    if not hasattr(_cls, "__class_getitem__"):
        _cls.__class_getitem__ = classmethod(lambda cls, params: cls)


class Dim[T]:
    """Symbolic integer type for dimension values.

    At runtime this is a no-op generic class. The type checker uses the
    .pyi stub for shape inference.
    """

    pass
