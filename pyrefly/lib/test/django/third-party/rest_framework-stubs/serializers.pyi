from collections.abc import Sequence
from typing import ClassVar, Generic, Literal, TypeVar

from django.db.models import Model
from rest_framework.fields import CharField as CharField, Field as Field

_MT = TypeVar("_MT", bound=Model)

class Serializer(Field, Generic[_MT]): ...

class ModelSerializer(Serializer[_MT]):
    class Meta:
        model: ClassVar[type[_MT]]
        fields: ClassVar[Sequence[str] | Literal["__all__"]]
