# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""DSL internals for shape typing.

Only used inside DSL definition files (e.g. torch/_shapes.pyi), not in
normal stubs or user code.
"""

import typing


def shape_dsl_function(fn: typing.Callable) -> typing.Callable:
    """Marks a function as a shape DSL function.

    At runtime this is a no-op: the decorated function is returned unchanged.
    Pyrefly uses this decorator at type-checking time to convert the function
    body to DSL IR via convert_fndef.
    """
    return fn
