# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Foundational shape typing constructs for tensor shape inference."""

class Dim[T]:
    """Symbolic integer type for dimension values.

    Represents integer values that can be symbolic (type variables) or concrete.
    Used to pass dimension values to constructors and functions.

    Examples:
        Dim[3]      # Concrete dimension value 3
        Dim[N]      # Symbolic dimension
        Dim[N + 1]  # Symbolic expression N + 1
    """

    pass

__all__ = ["Dim"]
