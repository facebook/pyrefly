from _typeshed import Incomplete
from conans.errors import ConanException as ConanException
from conans.model.ref import ConanFileReference as ConanFileReference, check_valid_ref as check_valid_ref
from conans.util.files import load as load
from conans.util.templates import render_layout_file as render_layout_file

DEFAULT_LAYOUT_FILE: str
LAYOUTS_FOLDER: str

def get_editable_abs_path(path, cwd, cache_folder): ...

class EditableLayout:
    BUILD_FOLDER: str
    SOURCE_FOLDER: str
    cpp_info_dirs: Incomplete
    folders: Incomplete
    _filepath: Incomplete
    def __init__(self, filepath) -> None: ...
    def folder(self, ref, name, settings, options): ...
    @staticmethod
    def _work_on_item(value): ...
    def _parse_layout_file(self, ref, settings, options): ...
    def _load_data(self, ref, settings, options): ...
    def apply_to(self, ref, cpp_info, settings: Incomplete | None = ..., options: Incomplete | None = ...) -> None: ...
