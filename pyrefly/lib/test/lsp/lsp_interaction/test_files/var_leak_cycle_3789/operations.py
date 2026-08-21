# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expr import Expr


class Basic: ...


class AssocOp(Basic):
    @classmethod
    def make_args(cls, expr: "Expr") -> tuple["Expr", ...]: ...
