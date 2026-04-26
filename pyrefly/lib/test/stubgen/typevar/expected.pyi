# @generated
from typing import NamedTuple, ParamSpec, TypeVar

T = TypeVar("T")

P = ParamSpec("P")


class Employee(NamedTuple):
    name: str
    age: int
