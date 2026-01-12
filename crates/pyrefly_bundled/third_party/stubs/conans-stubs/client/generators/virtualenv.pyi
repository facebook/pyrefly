from _typeshed import Incomplete
from conans.client.envvars.environment import BAT_FLAVOR as BAT_FLAVOR, PS1_FLAVOR as PS1_FLAVOR, SH_FLAVOR as SH_FLAVOR, env_files as env_files
from conans.client.tools.oss import OSInfo as OSInfo
from conans.model import Generator as Generator

class VirtualEnvGenerator(Generator):
    append_with_spaces: Incomplete
    suffix: str
    venv_name: str
    conanfile: Incomplete
    env: Incomplete
    normalize: bool
    def __init__(self, conanfile) -> None: ...
    @property
    def filename(self) -> None: ...
    @property
    def content(self): ...
