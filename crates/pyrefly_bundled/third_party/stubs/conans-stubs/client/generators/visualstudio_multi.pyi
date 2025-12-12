from _typeshed import Incomplete
from conans.client.generators import VisualStudioGenerator as VisualStudioGenerator
from conans.client.tools import msvs_toolset as msvs_toolset
from conans.errors import ConanException as ConanException
from conans.model import Generator as Generator
from conans.util.files import load as load

class _VSSettings:
    _props: Incomplete
    def __init__(self, settings) -> None: ...
    @property
    def filename(self): ...
    @property
    def condition(self): ...

class VisualStudioMultiGenerator(Generator):
    multi_content_template: str
    @property
    def filename(self) -> None: ...
    @property
    def content(self): ...
