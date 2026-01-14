"""
Minimal marshmallow stubs for testing.
"""

from typing import Any

class SchemaMeta(type):
    """Metaclass for Schema."""
    ...

class Schema(metaclass=SchemaMeta):
    """Base schema class for marshmallow serialization."""

    class Meta:
        """Options class for Schema. Child classes typically override this."""
        ...

    def load(self, data: Any) -> Any:
        ...

    def dump(self, obj: Any) -> Any:
        ...
