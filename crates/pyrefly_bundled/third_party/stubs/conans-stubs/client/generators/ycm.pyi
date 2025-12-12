from conans.model import Generator as Generator

class YouCompleteMeGenerator(Generator):
    template: str
    @property
    def filename(self) -> None: ...
    @property
    def content(self): ...
