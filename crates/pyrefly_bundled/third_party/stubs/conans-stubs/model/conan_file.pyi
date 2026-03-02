""" Conan type stubs for version 1.59.0 """

from _typeshed import Incomplete
from collections.abc import Generator
from conans.client.graph.python_requires import PyRequire
from conans.client.output import ScopedOutput
from conans.model.build_info import DepsCppInfo, CppInfo
from conans.model.conf import Conf as Conf
from conans.model.dependencies import ConanFileDependencies
from conans.model.env_info import DepsEnvInfo, EnvInfo
from conans.model.layout import Folders, Infos, Layouts
from conans.model.new_build_info import NewCppInfo
from conans.model.requires import Requirements as Requirements
from conans.model.ref import ConanFileReference
from conans.model.settings import Settings
from conans.model.user_info import DepsUserInfo, UserInfo
from conans.model.info import ConanInfo

from conan.tools.env import Environment
from pathlib import Path

from typing import Any, Callable, Dict, Iterable, List, Literal, \
        Optional, Tuple, TypeAlias, TypedDict, Union

def create_options(conanfile): ...
def create_requirements(conanfile): ...
def create_settings(conanfile, settings): ...
def _env_and_python(conanfile) -> Generator[None, None, None]: ...
def get_env_context_manager(conanfile, without_python: bool = ...): ...


class ConanSCM(TypedDict, total=False):
  type: Literal["git", "svn"]
  url: str | Literal["auto"] # this does resolve to str, but at least it's notated
  revision:  str | Literal["auto"]
  subfolder: str # FIXME: From here on needs NotRequired, but is only available from 3.11
  username: Optional[str]
  password: Optional[str]
  verify_ssl : bool
  shallow: bool # shallow clone for git
  submodule: Literal["shallow", "recursive"]

class ConanFile:
    name: str 
    version: str
    url: str
    license: str
    author: str
    description: str
    topics: Iterable[str] # TODO 
    homepage: str
    build_policy: Literal["missing", "always"]
    upload_policy: Optional[Literal["skip"]]
    short_paths: bool
    apply_env: bool # When True (Default), the values from self.deps_env_info (corresponding to the declared env_info in the requires and tool_requires) will be automatically applied to the os.environ.
    exports: Union[str, Iterable[str]]
    exports_sources: Union[str, Iterable[str]]
    generators: Union[str, Iterable[str]]
    revision_mode: Literal["hash", "scm"]
    should_configure: bool
    should_build: bool
    should_install: bool
    should_test: bool
    in_local_cache: bool # A boolean attribute useful for conditional logic to apply in user folders local commands. It will return True if the conanfile resides in the local cache ( we are installing the package) and False if we are running the conanfile in a user folder (local Conan commands).
    develop: bool
    default_channel: str
    default_user: str
    options: Dict[str, Iterable[Union[bool, str, Literal["ANY"]]]]
    default_options: Dict[str, Union[bool, str]]
    provides: Optional[Iterable[str]]
    deprecated: bool # This attribute declares that the recipe is deprecated, causing a user-friendly warning message to be emitted whenever it is used
    folders: Folders # for layout
    patterns: Incomplete
    win_bash: bool
    win_bash_run: bool
    tested_reference_str: str
    output: ScopedOutput # logger
    display_name: str
    compatible_packages: List[str] # TODO: ?
    buildenv_info: Environment
    runenv_info: Environment
    conf_info: Conf
    env_scripts: Dict[str, Any] # TODO: ?
    cpp: Infos
    layouts: Layouts
    deps_cpp_info: DepsCppInfo
    env_info: EnvInfo
    deps_env_info: DepsEnvInfo
    user_info: UserInfo
    deps_user_info: DepsUserInfo
    virtualbuildenv: bool
    virtualrunenv: bool
    python_requires_extend: Union[str, Iterable[str]]
    info: ConanInfo # package_id info object

    conan_data = Dict[str, Any] # dict read from conandata.yml
    test_type: Literal["requires", "build_requires", "explicit"]
    no_copy_source: bool  # Tells the recipe that the source code will not be copied from the source folder to the build folder.
    scm: ConanSCM  # Used to clone/checkout a repository
    # methods
    python_requires: Union[str, Iterable[str]]
    # @property
    # def python_requires(self) -> Dict[str, PyRequire]: ...
    # @python_requires.setter
    # def python_requires(self, value: Union[str, Iterable[str]]): ...
    @property
    def build_policy_missing(self): """ Sets build_policy to missing" """
    @property
    def build_policy_always(self): """ Sets build_policy to always" """
    @property
    def context(self) -> Literal["host", "build"]: """ Only in install. """
    @property
    def dependencies(self) -> ConanFileDependencies: """ Conan recipes provide access to their dependencies via 
        the self.dependencies attribute. This attribute is extensively used by generators like CMakeDeps or MSBuildDeps 
        to generate the necessary files for the build."""
    @property
    def ref(self) -> ConanFileReference: """ Own reference """
    @property
    def pref(self): """ TODO: don't know when this can be used"""
    @property
    def buildenv(self) -> Environment: """ Package-specific Environment """
    @property
    def runenv(self) -> Environment: """ Package-specific runtime Environment """
    @property
    def new_cpp_info(self) -> NewCppInfo: """TODO: ? """
    @property
    def recipe_folder(self)-> str: ...
    @property
    def recipe_path(self) -> Path:  """ Recipe folder as path """
    @property
    def source_folder(self): """ The folder in which the source code lives. """
    @property
    def source_path(self) -> Path: """ Source folder as path """
    @property
    def export_sources_folder(self): ...
    @property
    def export_sources_path(self) -> Path: """ Export sources folder as path """
    @property
    def export_folder(self)-> str: ...
    @property
    def export_path(self) -> Path:  """ Export folder as path """
    @property
    def build_folder(self)-> str: ...
    @property
    def build_path(self) -> Path:  """ Build folder as path """
    @property
    def package_folder(self)-> str: ...
    @property
    def package_path(self) -> Path: """ Package folder as path """
    @property
    def install_folder(self) -> str: """ The folder in which the installation of packages outputs the generator files with the information of dependencies """
    @property
    def generators_folder(self)-> str: ...
    @property
    def generators_path(self) -> Path:  """ Generators folder as path """
    @property
    def imports_folder(self)-> str:  """ Imports folder ... """
    @property
    def env(self):  """Apply the self.deps_env_info into a copy of self._conan_env_values (will prioritize the
        self._conan_env_values, user specified from profiles or -e first, then inherited) """
    @property
    def channel(self): ...
    @property
    def user(self): ...
    @property
    def cpp_info(self) -> CppInfo:  """ Is responsible for storing all the information needed by consumers of a package: include directories, library names, library paths… There are some default values that will be applied automatically if not indicated otherwise.
        This object should be filled in package_info() method."""
    
    # TODO: this is correct, but not yet supported 
    # @property
    # def settings(self) -> Settings: ...
    # @settings.setter
    # def settings(self, value: Optional[Iterable[str]]) -> None: ...
    settings: Optional[Union[Literal["os", "compiler", "build_type", "arch"], 
                             Iterable[Literal["os", "compiler", "build_type", "arch"]]],
                            #  Dict[str, Settings]
                             ]


    # combined requirements methods and attributes
    # @property
    # @overload
    # def requires(self) -> Optional[Iterable[str]]: ...
    # @overload
    def requires(self, requirement: str, override: bool=False, private: bool=False) -> None: ...
    # @requires.setter
    # def requires(self, value: Optional[Iterable[str]]): ...

    #@property
    # @overload
    # def build_requires(self) -> None: ...
    # @overload
    def build_requires(self, requirement: str, force_host_context: bool=False) -> None: ...
    # @build_requires.setter
    # def build_requires(self, value: Optional[Iterable[str]]): ...

    # @property
    # @overload
    # def tool_requires(self) -> None: ...
    # @overload
    def tool_requires(self, requirement: str, force_host_context: bool=False) -> None: ...
    # @tool_requires.setter
    # def tool_requires(self, value: Optional[Iterable[str]]): ...

    def __init__(self, output, runner, display_name: str = ..., user: Incomplete |
                 None = ..., channel: Incomplete | None = ...) -> None: ...
    
    def init(self) -> None: """ This is an optional method for initializing conanfile values, designed for inheritance from python requires """
    def export(self): """ Equivalent to the exports attribute, but in method form. It supports the self.copy() to do pattern based copy of files from the local user folder (the folder containing the conanfile.py) to the cache export_folder"""
    def export_sources(self): """Equivalent to the exports_sources attribute, but in method form. It supports the self.copy() to do pattern based copy of files from the local user folder (the folder containing the conanfile.py) to the cache export_sources_folder"""
    def generate(self): """ The purpose of generate() is to prepare the build, generating the necessary files. """
    def source(self) -> None: """ Method used to retrieve the source code from any other external
        origin like github using $ git clone or just a regular download."""
    def system_requirements(self) -> None: """ this method can be overwritten to implement logic for system package
        managers, as apt-get

        You can define self.global_system_requirements = True, if you want the installation
        to be for all packages (not depending on settings/options/requirements)
        """
    def config_options(self) -> None: """ modify options, probably conditioned to some settings. This call is executed
        before config_settings. E.g.
        if self.settings.os == "Windows":
            del self.options.shared  # shared/static not supported in win
        """
    def configure(self) -> None: """ modify settings, probably conditioned to some options. This call is executed
        after config_options. E.g.
        if self.options.header_only:
            self.settings.clear()
        This is also the place for conditional requirements
        """
    def build_requirements(self) -> None: """ The requires specified in this method are only installed and 
    used when the package is built from sources. If there is an existing pre-compiled binary, then the tool requirements for this package will not be retrieved. """

    def requirements(self)-> None: """ Besides the requires field, more advanced requirement logic 
    can be defined in the requirements() optional method. """

    def imports(self): """ Importing files copies files from the local store to your project. This feature is handy for copying shared libraries (dylib in Mac, dll in Win) 
        to the directory of your executable, so that you don’t have to mess with your PATH to run them. """
    def build(self) -> None: """ build your project calling the desired build tools as done in the command line.
        E.g. self.run("cmake --build .") Or use the provided build helpers. E.g. cmake.build()
        """

    def layout(self)-> None: """ In the layout() method you can adjust self.folders and self.cpp. """
    def package(self) -> None: """ package the needed files from source and build folders.
        E.g. self.copy("*.h", src="src/includes", dst="includes")
        """
    def package_info(self) -> None: """ define cpp_build_info, flags, etc """

    def package_id(self) -> None: """ modify the binary info, typically to narrow values
        e.g.: self.info.settings.compiler = "Any" => All compilers will generate same ID
        """
    def test(self) -> None:  """ test the generated executable.
        E.g.  self.run("./example")
        """

    # Helper functions 
    def run(self, command, output: bool = ..., cwd: Incomplete | None = ..., 
        win_bash: bool = ..., subsystem: Incomplete | None = ...,
        msys_mingw: bool = ..., ignore_errors: bool = ..., run_environment: bool = ...,
          with_login: bool = ..., env: str = ..., scope: str = ...): """
          For running commmands. Can be called in any function. """
        
    def copy(self, pattern: str, dst="", src="", root_package=None, folder=False, 
             ignore_case=True, excludes=None, keep_path=True): 
       """  Copy files with the pattern from src to dst. 
       Can be called in imports, package and deploy methods. """
    def copy_deps(self, pattern: str, dst="", src="", root_package=None, folder=False, 
                  ignore_case=True, excludes=None, keep_path=True): 
       """  Alias for copy in deploy, the same as method inside imports() method. """
    def collect_libs(self, folder: str | None = ...): " Deprecated... "