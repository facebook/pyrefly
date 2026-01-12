from sympy.plotting.pygletplot.plot import PygletPlot as Plot
from sympy.utilities.decorator import doctest_depends_on

@doctest_depends_on(modules=("pyglet",))
def PygletPlot(*args, **kwargs) -> Plot: ...
