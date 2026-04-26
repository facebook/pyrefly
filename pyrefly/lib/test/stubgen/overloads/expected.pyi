# @generated
import typing
from typing import overload


@overload
def process(x: int) -> int: ...


@overload
def process(x: str) -> str: ...


@typing.overload
def convert(x: int) -> str: ...


@typing.overload
def convert(x: str) -> int: ...


def normal_function(x: int) -> int: ...
