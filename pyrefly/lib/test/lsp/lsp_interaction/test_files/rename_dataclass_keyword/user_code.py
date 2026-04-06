from dataclasses import dataclass


@dataclass
class Foo:
    a: int
    b: str


Foo(
    a=123,
    b="abc",
)
