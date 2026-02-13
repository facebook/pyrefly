from _typeshed import Incomplete
from conans.errors import ConanException as ConanException
from conans.model.options import OptionsValues as OptionsValues
from conans.model.ref import ConanFileReference as ConanFileReference
from conans.util.files import load as load, save as save

GRAPH_INFO_FILE: str

class GraphInfo:
    options: Incomplete
    root: Incomplete
    profile_host: Incomplete
    profile_build: Incomplete
    graph_lock: Incomplete
    def __init__(self, profile_host: Incomplete | None = ..., profile_build: Incomplete | None = ..., options: Incomplete | None = ..., root_ref: Incomplete | None = ...) -> None: ...
    @staticmethod
    def load(path): ...
    @staticmethod
    def loads(text): ...
    def save(self, folder, filename: Incomplete | None = ...) -> None: ...
    def _dumps(self): ...
