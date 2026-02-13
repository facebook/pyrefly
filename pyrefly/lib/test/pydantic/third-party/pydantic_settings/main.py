"""
Minimal pydantic_settings.main stub for pyrefly tests.
"""

from pydantic import BaseModel


class BaseSettings(BaseModel):
    """
    Minimal BaseSettings stub.
    In real pydantic_settings, this class reads fields from environment variables.
    For pyrefly type checking, we just need the class structure.
    """

    pass
