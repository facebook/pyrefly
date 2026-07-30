# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

# Container types with Any type args should count as typed (not any).
# Only a bare `Any` annotation is "100% any".
x: list[Any] = []
y: dict[str, Any] = {}
z: tuple[int, Any] = (1, None)

# Bare Any annotation: should count as any
w: Any = None

# Concrete annotation: should count as typed
v: int = 0


def container_any(a: list[Any]) -> dict[str, Any]:
    return {}


def pure_any(a: Any) -> Any:
    return a


def concrete(a: int) -> str:
    return ""
