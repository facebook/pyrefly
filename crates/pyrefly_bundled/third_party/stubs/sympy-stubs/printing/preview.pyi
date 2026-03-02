from sympy.utilities.decorator import doctest_depends_on

__doctest_requires__ = ...

def system_default_viewer(fname, fmt) -> None: ...
def pyglet_viewer(fname, fmt) -> None: ...
@doctest_depends_on(exe=("latex", "dvipng"), modules=("pyglet",), disable_viewers=("evince", "gimp", "superior-dvi-viewer"))
def preview(
    expr,
    output=...,
    viewer=...,
    euler=...,
    packages=...,
    filename=...,
    outputbuffer=...,
    preamble=...,
    dvioptions=...,
    outputTexFile=...,
    extra_preamble=...,
    fontsize=...,
    **latex_settings,
) -> None: ...
