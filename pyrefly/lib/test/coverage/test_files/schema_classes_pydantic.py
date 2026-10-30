# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Pydantic BaseModel fields are IMPLICIT (0 typable), matching typestats behavior.

from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


class AdminUser(User):
    role: str
