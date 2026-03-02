"""
Type annotations for botocore.plugin module.

Copyright 2025 Vlad Emelianov
"""

from collections.abc import Mapping
from contextvars import Token
from dataclasses import dataclass
from logging import Logger

from botocore.client import BaseClient

log: Logger = ...

@dataclass
class PluginContext:
    plugins: str | None = ...

def get_plugin_context() -> PluginContext | None: ...
def set_plugin_context(ctx: PluginContext) -> Token[PluginContext]: ...
def reset_plugin_context(token: Token[PluginContext]) -> None: ...
def get_botocore_plugins() -> str | None: ...
def load_client_plugins(client: BaseClient, plugins: Mapping[str, str]) -> None: ...
