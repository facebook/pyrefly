from conans.model import Generator as Generator

class SConsGenerator(Generator):
    @property
    def filename(self): ...
    @property
    def content(self): ...
