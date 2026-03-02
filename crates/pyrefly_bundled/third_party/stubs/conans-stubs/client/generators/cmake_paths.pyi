from conans.client.generators.cmake import DepsCppCmake as DepsCppCmake
from conans.model import Generator as Generator

class CMakePathsGenerator(Generator):
    name: str
    @property
    def filename(self): ...
    @property
    def content(self): ...
