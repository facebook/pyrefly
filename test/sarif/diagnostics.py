# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Protocol, reveal_type

x: str = 0
one = 1 / 0
reveal_type(x)


class Comparable(Protocol):
    def compare(self) -> int: ...


class Mismatched:
    def compare(self) -> str: ...


# A diagnostic whose message has both a header and details.
comparable: Comparable = Mismatched()


def takes_int(value: int) -> None:
    pass


takes_int("ignored by configuration")
inline_ignored = 1 + ""  # pyrefly: ignore[unsupported-operation]
value: int = 1
redundant = cast(int, value)
