from _typeshed import Incomplete
from collections.abc import Generator
from conans.errors import ConanException as ConanException
from conans.util.files import normalize as normalize

sh_activate: Incomplete
sh_deactivate: Incomplete
bat_activate: Incomplete
bat_deactivate: Incomplete
ps1_activate: Incomplete
ps1_deactivate: Incomplete
BAT_FLAVOR: str
PS1_FLAVOR: str
SH_FLAVOR: str

def _variable_placeholder(flavor, name, append_with_spaces): ...
def _format_values(flavor, variables, append_with_spaces) -> Generator[Incomplete, None, None]: ...
def _files(env_vars, vars_with_spaces, flavor, activate_tpl, deactivate_tpl, venv_name, env_filepath): ...
def env_files(env_vars, vars_with_spaces, flavor, folder, name, venv_name): ...
