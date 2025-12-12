from matplotlib.transforms import Bbox

from ._backend_tk import _BackendTk, FigureCanvasTk
from .backend_agg import FigureCanvasAgg

class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def draw(self) -> None: ...
    def blit(self, bbox: Bbox = ...) -> None: ...

class _BackendTkAgg(_BackendTk):
    FigureCanvas: type[FigureCanvasAgg] = ...
