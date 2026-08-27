# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Shape annotations with TorchScript-compatible runtime behavior enabled.

Importing this module erases shape-only runtime annotations to types
TorchScript understands. This must happen before any annotated class body is
evaluated, because TorchScript reads class attribute annotations back out of
`__annotations__`, so a call made after import would be too late.

`Int` here is the same class object as `shape_extensions.Int`, so the change is
process-global and one-way: it also affects code that imports `shape_extensions`
directly. There is intentionally no way to undo it.
"""

from . import *  # noqa: F401, F403
from . import __all__ as _shape_extensions_all, _return_int, Int as _Int

__all__ = _shape_extensions_all

_Int.__class_getitem__ = classmethod(_return_int)
