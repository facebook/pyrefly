from _typeshed import Incomplete
from conans.client.graph.graph import RECIPE_EDITABLE as RECIPE_EDITABLE
from conans.errors import ConanException as ConanException
from conans.model.editable_layout import EditableLayout as EditableLayout, get_editable_abs_path as get_editable_abs_path
from conans.model.ref import ConanFileReference as ConanFileReference
from conans.paths import CONANFILE as CONANFILE
from conans.util.files import load as load, save as save

class LocalPackage:
    _base_folder: Incomplete
    _conanfile_folder: Incomplete
    layout: Incomplete
    generators: Incomplete
    def __init__(self, base_folder, data, cache, ws_layout, ws_generators, ref) -> None: ...
    @property
    def root_folder(self): ...

class Workspace:
    default_filename: str
    _cache: Incomplete
    _ws_generator: Incomplete
    _workspace_packages: Incomplete
    _base_folder: Incomplete
    def __init__(self, path, cache) -> None: ...
    def generate(self, install_folder, graph, output) -> None: ...
    def get_editable_dict(self): ...
    def __getitem__(self, ref): ...
    @property
    def root(self): ...
    _root: Incomplete
    def _loads(self, text) -> None: ...
