from .backend_agg import FigureCanvasAgg
from .backend_qt import _BackendQT, FigureCanvasQT

class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):
    def __init__(self, figure=...) -> None: ...
    def paintEvent(self, event) -> None: ...
    def print_figure(self, *args, **kwargs) -> None: ...

class _BackendQTAgg(_BackendQT):
    FigureCanvas = FigureCanvasQTAgg
