from typing import Any

WARN_NOT_INSTALLED = ...
WARN_OLD_VERSION = ...
_component_re = ...

def version_tuple(vstring) -> tuple[Any, ...]: ...
def import_module(
    module,
    min_module_version=...,
    min_python_version=...,
    warn_not_installed=...,
    warn_old_version=...,
    module_version_attr=...,
    module_version_attr_call_args=...,
    import_kwargs=...,
    catch=...,
): ...
