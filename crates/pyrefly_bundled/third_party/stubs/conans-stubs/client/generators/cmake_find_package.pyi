from _typeshed import Incomplete
from conans.client.generators.cmake import DepsCppCmake as DepsCppCmake
from conans.client.generators.cmake_find_package_common import CMakeFindPackageCommonMacros as CMakeFindPackageCommonMacros, find_transitive_dependencies as find_transitive_dependencies, target_template as target_template
from conans.client.generators.cmake_multi import extend as extend
from conans.model import Generator as Generator
from conans.model.conan_generator import GeneratorComponentsMixin as GeneratorComponentsMixin

class CMakeFindPackageGenerator(GeneratorComponentsMixin, Generator):
    name: str
    find_template: Incomplete
    find_components_tpl: Incomplete
    @property
    def filename(self) -> None: ...
    @classmethod
    def _get_filename(cls, obj): ...
    @property
    def content(self): ...
    def _get_components(self, pkg_name, cpp_info): ...
    def _find_for_dep(self, pkg_name, pkg_findname, pkg_filename, pkg_namespace, cpp_info): ...
