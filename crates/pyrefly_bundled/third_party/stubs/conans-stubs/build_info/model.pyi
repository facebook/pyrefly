from _typeshed import Incomplete
from typing import NamedTuple

class BuildInfo:
    modules: Incomplete
    def __init__(self) -> None: ...
    def serialize(self): ...

class BuildInfoModule:
    id: str
    artifacts: Incomplete
    dependencies: Incomplete
    def __init__(self) -> None: ...
    def serialize(self): ...

class BuildInfoModuleArtifact(NamedTuple):
    type: Incomplete
    sha1: Incomplete
    md5: Incomplete
    name: Incomplete

class BuildInfoModuleDependency(NamedTuple):
    id: Incomplete
    type: Incomplete
    sha1: Incomplete
    md5: Incomplete
