# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# gh-3576: module-scope `del` names must not appear in the report.

for ta in ["a", "b"]:
    pass
del ta


tmp = 1
del tmp


keep_me: int = 42  # must survive a function-local `del` of a same-named local.


def helper() -> None:
    keep_me = 1
    del keep_me


lst: list[int] = [1]
del lst[0]


def setup() -> None: ...


setup()
del setup


class _Tmp: ...


del _Tmp
