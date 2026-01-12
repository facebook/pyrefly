from _typeshed import Incomplete
from conans.client import tools as tools
from conans.client.build.visual_environment import VisualStudioBuildEnvironment as VisualStudioBuildEnvironment, vs_build_type_flags as vs_build_type_flags, vs_std_cpp as vs_std_cpp
from conans.client.tools.env import environment_append as environment_append, no_op as no_op
from conans.client.tools.intel import intel_compilervars as intel_compilervars
from conans.client.tools.oss import cpu_count as cpu_count
from conans.client.tools.win import vcvars_command as vcvars_command
from conans.errors import ConanException as ConanException
from conans.model.conan_file import ConanFile as ConanFile
from conans.model.version import Version as Version
from conans.util.env_reader import get_env as get_env
from conans.util.files import decode_text as decode_text, save as save
from conans.util.runners import version_runner as version_runner

class MSBuild:
    _conanfile: Incomplete
    _settings: Incomplete
    _output: Incomplete
    build_env: Incomplete
    def __init__(self, conanfile) -> None: ...
    def build(self, project_file, targets: Incomplete | None = ..., upgrade_project: bool = ..., build_type: Incomplete | None = ..., arch: Incomplete | None = ..., parallel: bool = ..., force_vcvars: bool = ..., toolset: Incomplete | None = ..., platforms: Incomplete | None = ..., use_env: bool = ..., vcvars_ver: Incomplete | None = ..., winsdk_version: Incomplete | None = ..., properties: Incomplete | None = ..., output_binary_log: Incomplete | None = ..., property_file_name: Incomplete | None = ..., verbosity: Incomplete | None = ..., definitions: Incomplete | None = ..., user_property_file_name: Incomplete | None = ...): ...
    def get_command(self, project_file, props_file_path: Incomplete | None = ..., targets: Incomplete | None = ..., upgrade_project: bool = ..., build_type: Incomplete | None = ..., arch: Incomplete | None = ..., parallel: bool = ..., toolset: Incomplete | None = ..., platforms: Incomplete | None = ..., use_env: bool = ..., properties: Incomplete | None = ..., output_binary_log: Incomplete | None = ..., verbosity: Incomplete | None = ..., user_property_file_name: Incomplete | None = ...): ...
    def _get_props_file_contents(self, definitions: Incomplete | None = ...): ...
    @staticmethod
    def get_version(settings): ...
