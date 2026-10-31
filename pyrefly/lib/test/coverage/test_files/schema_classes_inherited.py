# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TypedDict and dataclass subclasses: all schema class fields are IMPLICIT.
# Inherited fields appear only under the defining class.

from dataclasses import dataclass
from typing import TypedDict


class BaseConfig(TypedDict):
    name: str


class ExtendedConfig(BaseConfig):
    value: int


@dataclass
class Base:
    x: int


@dataclass
class Derived(Base):
    y: float
