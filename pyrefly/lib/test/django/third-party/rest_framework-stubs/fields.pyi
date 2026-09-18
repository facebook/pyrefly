class Field:
    label: str | None
    source: str | None
    context: dict[str, object]
    data: object

class CharField(Field): ...
