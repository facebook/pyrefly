# @generated
from dataclasses import dataclass, field


@dataclass
class Example:
    required: int
    default_none: str | None = ...
    default_factory: list[int] = ...
    literal_default: int = 1
