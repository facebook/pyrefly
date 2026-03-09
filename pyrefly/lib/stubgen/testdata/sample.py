"""A sample module for testing stub generation."""

from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

__all__ = [
    "VERSION",
    "Config",
    "process",
    "make_config",
    "Stack",
    "apply_fn",
    "Coordinates",
    "greet",
    "load_items",
]

VERSION: str = "1.0.0"

_INTERNAL_COUNTER: int = 0

T = TypeVar("T")

Coordinates = Tuple[float, float]


class Config:
    """Application configuration."""

    DEFAULT_NAME: ClassVar[str] = "default"
    debug: bool
    name: str

    def __init__(self, name: str, debug: bool = False) -> None:
        self.name = name
        self.debug = debug

    @property
    def display_name(self) -> str:
        return self.name.upper()

    @display_name.setter
    def display_name(self, value: str) -> None:
        self.name = value.lower()


class Stack(Generic[T]):
    """A generic stack implementation."""

    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("empty stack")
        return self._items.pop()

    def peek(self) -> Optional[T]:
        return self._items[-1] if self._items else None


@overload
def process(x: int) -> int: ...


@overload
def process(x: str) -> str: ...


def process(x: Union[int, str]) -> Union[int, str]:
    return x


def make_config(name: str, debug: bool = False) -> Config:
    """Create a new Config instance."""
    return Config(name, debug)


def apply_fn(fn: Callable[[int], int], values: List[int]) -> List[int]:
    return [fn(v) for v in values]


def greet(name: Optional[str] = None) -> str:
    return f"Hello, {name or 'World'}!"


def load_items(path: str) -> Dict[str, List[Tuple[int, ...]]]:
    return {}


def _private_helper(x: int) -> int:
    return x * 2
