# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# All instance attributes only initialized in __init__ — no class-body
# declarations. These should be recognized and emitted, since traditionally
# in Python `__init__` is the source of truth for available attributes.
class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


# Mix of class-body declarations and __init__-only attributes. The stub
# should preserve the class-body annotation for `name` and add `created_at`
# discovered from __init__.
class User:
    name: str

    def __init__(self, name: str, created_at: int) -> None:
        self.name = name
        self.created_at = created_at


# Annotated self assignments (`self.x: T = ...`) carry an explicit type.
class Config:
    def __init__(self) -> None:
        self.host: str = "localhost"
        self.port: int = 8080


# Attributes initialized in non-__init__ recognized methods (e.g. __post_init__,
# setUp, async __init__) should also be picked up.
class Service:
    def __init__(self) -> None:
        self.client = make_client()

    def setUp(self) -> None:
        self.cache: dict[str, int] = {}


def make_client() -> "Client":
    return Client()


class Client:
    pass
