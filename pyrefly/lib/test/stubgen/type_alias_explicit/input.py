# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc
from typing import Literal, TypeAlias, TypeVar

T = TypeVar("T")

TemplateId: TypeAlias = str
TemplateKind: TypeAlias = Literal["abc", "def"]
TemplateLookup: TypeAlias = dict[TemplateId, str]
TemplateSequence: TypeAlias = collections.abc.Sequence[tuple[TemplateId, str]]
GenericTemplateSequence: TypeAlias = collections.abc.Sequence[tuple[T, str]]

type ModernTemplateId = str
type ModernTemplateKind = Literal["abc", "def"]
type ModernTemplateLookup = dict[ModernTemplateId, str]
type ModernTemplateSequence = collections.abc.Sequence[tuple[ModernTemplateId, str]]
type ModernGenericTemplateSequence[T] = collections.abc.Sequence[tuple[T, str]]

# Complex runtime values on ordinary annotated variables should still be omitted.
template_names: collections.abc.Sequence[str] = ("thumbnail", "document")
