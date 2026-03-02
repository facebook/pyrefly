from ..tools import chdir as chdir
from .b2 import B2Generator as B2Generator
from .boostbuild import BoostBuildGenerator as BoostBuildGenerator
from .cmake import CMakeGenerator as CMakeGenerator
from .cmake_multi import CMakeMultiGenerator as CMakeMultiGenerator
from .cmake_paths import CMakePathsGenerator as CMakePathsGenerator
from .deploy import DeployGenerator as DeployGenerator
from .gcc import GCCGenerator as GCCGenerator
from .json_generator import JsonGenerator as JsonGenerator
from .make import MakeGenerator as MakeGenerator
from .markdown import MarkdownGenerator as MarkdownGenerator
from .premake import PremakeGenerator as PremakeGenerator
from .qbs import QbsGenerator as QbsGenerator
from .qmake import QmakeGenerator as QmakeGenerator
from .scons import SConsGenerator as SConsGenerator
from .text import TXTGenerator as TXTGenerator
from .virtualbuildenv import VirtualBuildEnvGenerator as VirtualBuildEnvGenerator
from .virtualenv import VirtualEnvGenerator as VirtualEnvGenerator
from .virtualenv_python import VirtualEnvPythonGenerator as VirtualEnvPythonGenerator
from .virtualrunenv import VirtualRunEnvGenerator as VirtualRunEnvGenerator
from .visualstudio import VisualStudioGenerator as VisualStudioGenerator
from .visualstudio_multi import VisualStudioMultiGenerator as VisualStudioMultiGenerator
from .visualstudiolegacy import VisualStudioLegacyGenerator as VisualStudioLegacyGenerator
from .xcode import XCodeGenerator as XCodeGenerator
from .ycm import YouCompleteMeGenerator as YouCompleteMeGenerator
from _typeshed import Incomplete
from conans.client.generators.cmake_find_package import CMakeFindPackageGenerator as CMakeFindPackageGenerator
from conans.client.generators.cmake_find_package_multi import CMakeFindPackageMultiGenerator as CMakeFindPackageMultiGenerator
from conans.client.generators.compiler_args import CompilerArgsGenerator as CompilerArgsGenerator
from conans.client.generators.pkg_config import PkgConfigGenerator as PkgConfigGenerator
from conans.client.subsystems import deduce_subsystem as deduce_subsystem, subsystem_path as subsystem_path
from conans.errors import ConanException as ConanException, conanfile_exception_formatter as conanfile_exception_formatter
from conans.util.env_reader import get_env as get_env
from conans.util.files import mkdir as mkdir, normalize as normalize, save as save

class GeneratorManager:
    _generators: Incomplete
    _new_generators: Incomplete
    def __init__(self) -> None: ...
    def add(self, name, generator_class, custom: bool = ...) -> None: ...
    def __contains__(self, name) -> bool: ...
    def __getitem__(self, key): ...
    def _new_generator(self, generator_name, output): ...
    def write_generators(self, conanfile, old_gen_folder, new_gen_folder, output) -> None: ...

def _receive_conf(conanfile) -> None: ...
def write_toolchain(conanfile, path, output) -> None: ...
def _generate_aggregated_env(conanfile): ...
