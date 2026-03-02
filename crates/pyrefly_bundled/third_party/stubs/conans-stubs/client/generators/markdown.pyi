from _typeshed import Incomplete
from conans.model import Generator as Generator

requirement_tpl: Incomplete
macros: Incomplete
buildsystem_cmake_tpl: Incomplete
buildsystem_vs_tpl: Incomplete
buildsystem_autotools_tpl: Incomplete
buildsystem_other_tpl: Incomplete
components_tpl: Incomplete
headers_tpl: Incomplete

class MarkdownGenerator(Generator):
    @staticmethod
    def _list_headers(requirement) -> Generator[Incomplete, None, None]: ...
    @staticmethod
    def _list_requires(requirement): ...
    @property
    def filename(self) -> None: ...
    @property
    def content(self): ...
