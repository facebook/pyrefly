from conans.client.tools.files import VALID_LIB_EXTENSIONS as VALID_LIB_EXTENSIONS
from conans.model import Generator as Generator
from conans.paths import BUILD_INFO_VISUAL_STUDIO as BUILD_INFO_VISUAL_STUDIO

class VisualStudioGenerator(Generator):
    template: str
    properties_template: str
    item_template: str
    def _format_items(self): ...
    @property
    def filename(self): ...
    def _format_properties(self, build_info, condition): ...
    @property
    def content(self): ...
