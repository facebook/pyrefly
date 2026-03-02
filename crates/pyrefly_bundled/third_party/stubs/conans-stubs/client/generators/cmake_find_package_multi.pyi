from _typeshed import Incomplete
from conans.client.generators import CMakeFindPackageGenerator as CMakeFindPackageGenerator
from conans.client.generators.cmake import DepsCppCmake as DepsCppCmake
from conans.client.generators.cmake_find_package_common import CMakeFindPackageCommonMacros as CMakeFindPackageCommonMacros, find_transitive_dependencies as find_transitive_dependencies, target_template as target_template
from conans.client.generators.cmake_multi import extend as extend
from conans.util.files import save as save

class CMakeFindPackageMultiGenerator(CMakeFindPackageGenerator):
    name: str
    config_template: Incomplete
    targets_template: Incomplete
    target_properties: Incomplete
    build_modules: Incomplete
    config_version_template: Incomplete
    components_target_build_type_tpl: Incomplete
    components_targets_tpl: Incomplete
    components_config_tpl: Incomplete
    configuration: Incomplete
    configurations: Incomplete
    output_path: Incomplete
    def __init__(self, conanfile) -> None: ...
    def generate(self) -> None: ...
    @property
    def filename(self) -> None: ...
    @property
    def content(self): ...
    def _config_filename(self, pkg_filename): ...
    def _config_version_filename(self, pkg_filename): ...
    def _config(self, filename, name, namespace, version, public_deps_names): ...
