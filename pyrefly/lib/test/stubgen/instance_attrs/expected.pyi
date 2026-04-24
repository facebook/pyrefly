# @generated
class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None: ...


class User:
    name: str
    created_at: int

    def __init__(self, name: str, created_at: int) -> None: ...


class Config:
    host: str
    port: int

    def __init__(self) -> None: ...


class Service:
    client: Client
    cache: dict[str, int]

    def __init__(self) -> None: ...

    def setUp(self) -> None: ...


def make_client() -> "Client": ...


class Client:
    ...
