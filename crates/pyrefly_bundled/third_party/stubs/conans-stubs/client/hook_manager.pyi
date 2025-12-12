from _typeshed import Incomplete
from conans.client.output import ScopedOutput as ScopedOutput
from conans.client.tools.files import chdir as chdir
from conans.errors import ConanException as ConanException, NotFoundException as NotFoundException
from conans.util.files import save as save

attribute_checker_hook: str
valid_hook_methods: Incomplete

class HookManager:
    _hooks_folder: Incomplete
    _hook_names: Incomplete
    hooks: Incomplete
    output: Incomplete
    _attribute_checker_path: Incomplete
    _mutex: Incomplete
    def __init__(self, hooks_folder, hook_names, output) -> None: ...
    def execute(self, method_name, **kwargs) -> None: ...
    def load_hooks(self) -> None: ...
    def _load_hook(self, hook_name) -> None: ...
    @staticmethod
    def _load_module_from_file(hook_path): ...
