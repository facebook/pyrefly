from _typeshed import Incomplete
from conans.errors import ConanException as ConanException

class BuildMode:
    _out: Incomplete
    outdated: bool
    missing: bool
    never: bool
    cascade: bool
    patterns: Incomplete
    _unused_patterns: Incomplete
    _excluded_patterns: Incomplete
    all: bool
    def __init__(self, params, output) -> None: ...
    def forced(self, conan_file, ref, with_deps_to_build: bool = ...): ...
    def allowed(self, conan_file): ...
    def report_matches(self) -> None: ...
