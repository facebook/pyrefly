from typing import TypeVar

from factory.base import Factory

T = TypeVar("T")

class DjangoModelFactory(Factory[T]):
    class Meta:
        abstract: bool
