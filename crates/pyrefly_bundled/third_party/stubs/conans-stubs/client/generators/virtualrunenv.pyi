from _typeshed import Incomplete
from conans.client.generators.virtualenv import VirtualEnvGenerator as VirtualEnvGenerator
from conans.client.run_environment import RunEnvironment as RunEnvironment

class VirtualRunEnvGenerator(VirtualEnvGenerator):
    suffix: str
    venv_name: str
    env: Incomplete
    def __init__(self, conanfile) -> None: ...
