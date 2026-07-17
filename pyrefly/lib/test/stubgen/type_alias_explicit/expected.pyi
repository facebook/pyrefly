# @generated
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

template_names: collections.abc.Sequence[str]
