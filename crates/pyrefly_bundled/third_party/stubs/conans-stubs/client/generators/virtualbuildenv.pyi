from _typeshed import Incomplete
from conans.client.build.autotools_environment import AutoToolsBuildEnvironment as AutoToolsBuildEnvironment
from conans.client.build.visual_environment import VisualStudioBuildEnvironment as VisualStudioBuildEnvironment
from conans.client.generators.virtualenv import VirtualEnvGenerator as VirtualEnvGenerator
from conans.client.tools.win import vcvars_dict as vcvars_dict

class VirtualBuildEnvGenerator(VirtualEnvGenerator):
    suffix: str
    venv_name: str
    env: Incomplete
    def __init__(self, conanfile) -> None: ...
