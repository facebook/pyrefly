# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for decorator-related reporting:
# @staticmethod, @classmethod, @type_check_only.
# Ported from typestats TestClassDescriptorAlias, TestTypeCheckOnly.


class WithDecorators:
    @staticmethod
    def static_method(x: int) -> bool:
        return True

    @classmethod
    def class_method(cls, s: str) -> None:
        pass

    def regular_method(self, a: int) -> str:
        return ""
