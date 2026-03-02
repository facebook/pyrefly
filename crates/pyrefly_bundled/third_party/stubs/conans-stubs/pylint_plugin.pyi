from _typeshed import Incomplete
from pylint.checkers import BaseChecker
from pylint.interfaces import IRawChecker

def register(linter) -> None: ...
def transform_conanfile(node) -> None: ...
def _python_requires_member(): ...

class ConanDeprecatedImportsChecker(BaseChecker):
    __implements__ = IRawChecker
    deprecated_imports_pattern: Incomplete
    name: str
    msgs: Incomplete
    options: Incomplete
    def process_module(self, node) -> None: ...
