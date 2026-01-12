from _typeshed import Incomplete
from collections.abc import Generator
from conans.model.ref import PackageReference as PackageReference
from conans.util.files import save as save

class RowResult:
    remote: Incomplete
    reference: Incomplete
    _data: Incomplete
    def __init__(self, remote, reference, data) -> None: ...
    @property
    def recipe(self): ...
    @property
    def package_id(self): ...
    @property
    def outdated(self): ...
    def row(self, headers) -> Generator[Incomplete, None, None]: ...

class Headers:
    _preferred_ordering: Incomplete
    keys: Incomplete
    options: Incomplete
    requires: Incomplete
    settings: Incomplete
    def __init__(self, settings, options, requires, keys) -> None: ...
    def row(self, n_rows: int = ...): ...
    @staticmethod
    def _group_settings(settings): ...

class Results:
    _results: Incomplete
    requires: bool
    settings: Incomplete
    options: Incomplete
    remotes: Incomplete
    def __init__(self, results) -> None: ...
    def get_headers(self, keys=...): ...
    def packages(self) -> Generator[Incomplete, None, None]: ...

def html_binary_graph(search_info, reference, table_filename, template) -> None: ...
