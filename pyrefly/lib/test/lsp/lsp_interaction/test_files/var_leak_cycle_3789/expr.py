# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Regression fixture for https://github.com/facebook/pyrefly/issues/3789.
# `expr` -> `add` -> `operations` -> `expr` form an import cycle, and `expr`
# contains a lambda. Opening this file in the IDE used to panic with "a variable
# has leaked from one module to another".

from typing import Any


class Basic: ...


class EvalfMixin: ...


func: Any = ...
k: Any = ...
o: Any = ...
exp: Any = ...
logw: Any = ...


class Expr(Basic, EvalfMixin):
    def aseries(self, x=None, n=6, bound=0):
        s = func.series(k, 0, n)
        terms = sorted(
            Add.make_args(s.removeO()), key=lambda i: int(i.as_coeff_exponent(k)[1])
        )
        for t in terms:
            coeff, expo = t.as_coeff_exponent(k)
            snew = coeff.aseries(x, n, bound=bound - 1)
            s += snew * k**expo
        return (s + o).subs(k, exp(logw))

    def as_coeff_exponent(self, x) -> tuple["Expr", "Expr"]:
        return self, self


from add import Add
