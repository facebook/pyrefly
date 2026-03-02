from conans.model import Generator as Generator

class VisualStudioLegacyGenerator(Generator):
    template: str
    @property
    def filename(self): ...
    @property
    def content(self): ...
