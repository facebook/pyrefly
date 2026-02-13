from _typeshed import Incomplete
from conans.client.build.autotools_environment import AutoToolsBuildEnvironment as AutoToolsBuildEnvironment
from conans.client.build.cmake import CMake as CMake
from conans.client.build.meson import Meson as Meson
from conans.client.build.msbuild import MSBuild as MSBuild
from conans.client.build.visual_environment import VisualStudioBuildEnvironment as VisualStudioBuildEnvironment
from conans.client.run_environment import RunEnvironment as RunEnvironment
from conans.model.conan_file import ConanFile as ConanFile
from conans.model.options import Options as Options
from conans.model.settings import Settings as Settings
from conans.util.files import load as load

COMPLEX_SEARCH_CAPABILITY: str
CHECKSUM_DEPLOY: str
REVISIONS: str
ONLY_V2: str
MATRIX_PARAMS: str
OAUTH_TOKEN: str
SERVER_CAPABILITIES: Incomplete
DEFAULT_REVISION_V1: str
__version__: str
