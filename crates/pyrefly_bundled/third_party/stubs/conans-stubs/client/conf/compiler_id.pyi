from _typeshed import Incomplete
from conans.client.runner import ConanRunner as ConanRunner
from conans.model.version import Version as Version
from conans.util.files import rmdir as rmdir

GCC: str
LLVM_GCC: str
CLANG: str
APPLE_CLANG: str
SUNCC: str
VISUAL_STUDIO: str
INTEL: str
QCC: str
MCST_LCC: str
MSVC: str

class CompilerId:
    _name: Incomplete
    _major: Incomplete
    _minor: Incomplete
    _patch: Incomplete
    _version: Incomplete
    def __init__(self, name, major, minor, patch) -> None: ...
    @property
    def name(self): ...
    @property
    def major(self): ...
    @property
    def minor(self): ...
    @property
    def patch(self): ...
    @property
    def version(self): ...
    @property
    def major_minor(self): ...
    def __str__(self): ...
    def __repr__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...

UNKNOWN_COMPILER: Incomplete
MSVC_TO_VS_VERSION: Incomplete

def _parse_compiler_version(defines): ...
def detect_compiler_id(executable, runner: Incomplete | None = ...): ...
