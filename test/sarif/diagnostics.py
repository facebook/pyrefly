from typing import cast, reveal_type

x: str = 0
one = 1 / 0
reveal_type(x)


def takes_int(value: int) -> None:
    pass


takes_int("ignored by configuration")
inline_ignored = 1 + ""  # pyrefly: ignore[unsupported-operation]
value: int = 1
redundant = cast(int, value)
