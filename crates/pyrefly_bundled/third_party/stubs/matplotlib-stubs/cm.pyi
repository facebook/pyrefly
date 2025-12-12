from collections.abc import Mapping
from typing import TypedDict

from matplotlib._stubs_utils._typing import ArrayLike
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, MultivarColormap, Normalize, SegmentedBivarColormap

class __getattr__:
    LUTSIZE = ...

class ColormapRegistry(Mapping):
    def __init__(self, cmaps) -> None: ...
    def __getitem__(self, item: str): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __call__(self): ...
    def register(
        self,
        cmap: Colormap,
        *,
        name: str = ...,
        force: bool = ...,
    ) -> None: ...
    def unregister(self, name: str) -> None: ...

def register_cmap(
    name: str = ...,
    cmap: Colormap = ...,
    *,
    override_builtin: bool = ...,
) -> None: ...
def get_cmap(name: Colormap | str | None = ..., lut: int | None = ...) -> Colormap: ...
def unregister_cmap(name: str) -> Colormap | None: ...

class ScalarMappable:
    def __init__(
        self,
        norm: Normalize | None = ...,
        cmap: str | Colormap = ...,
    ) -> None: ...
    callbacksSM = ...
    def to_rgba(self, x, alpha=..., bytes=..., norm=...) -> tuple[float, float, float, float]: ...
    def set_array(self, A: ArrayLike | None) -> None: ...
    def get_array(self): ...
    def get_cmap(self) -> Colormap: ...
    def get_clim(self) -> tuple: ...
    def set_clim(self, vmin: float = ..., vmax: float = ...) -> None: ...
    def get_alpha(self) -> float: ...
    def set_cmap(self, cmap: Colormap | str | None) -> None: ...
    @property
    def norm(self) -> Normalize: ...
    @norm.setter
    def norm(self, norm: Normalize) -> None: ...
    def set_norm(self, norm: Normalize | None): ...
    def autoscale(self) -> None: ...
    def autoscale_None(self) -> None: ...
    def changed(self) -> None: ...

Accent: ListedColormap
Accent_r: ListedColormap
Blues: LinearSegmentedColormap
Blues_r: LinearSegmentedColormap
BrBG: LinearSegmentedColormap
BrBG_r: LinearSegmentedColormap
BuGn: LinearSegmentedColormap
BuGn_r: LinearSegmentedColormap
BuPu: LinearSegmentedColormap
BuPu_r: LinearSegmentedColormap
CMRmap: LinearSegmentedColormap
CMRmap_r: LinearSegmentedColormap
Dark2: ListedColormap
Dark2_r: ListedColormap
GnBu: LinearSegmentedColormap
GnBu_r: LinearSegmentedColormap
Grays: LinearSegmentedColormap
Grays_r: LinearSegmentedColormap
Greens: LinearSegmentedColormap
Greens_r: LinearSegmentedColormap
Greys: LinearSegmentedColormap
Greys_r: LinearSegmentedColormap
OrRd: LinearSegmentedColormap
OrRd_r: LinearSegmentedColormap
Oranges: LinearSegmentedColormap
Oranges_r: LinearSegmentedColormap
PRGn: LinearSegmentedColormap
PRGn_r: LinearSegmentedColormap
Paired: ListedColormap
Paired_r: ListedColormap
Pastel1: ListedColormap
Pastel1_r: ListedColormap
Pastel2: ListedColormap
Pastel2_r: ListedColormap
PiYG: LinearSegmentedColormap
PiYG_r: LinearSegmentedColormap
PuBu: LinearSegmentedColormap
PuBuGn: LinearSegmentedColormap
PuBuGn_r: LinearSegmentedColormap
PuBu_r: LinearSegmentedColormap
PuOr: LinearSegmentedColormap
PuOr_r: LinearSegmentedColormap
PuRd: LinearSegmentedColormap
PuRd_r: LinearSegmentedColormap
Purples: LinearSegmentedColormap
Purples_r: LinearSegmentedColormap
RdBu: LinearSegmentedColormap
RdBu_r: LinearSegmentedColormap
RdGy: LinearSegmentedColormap
RdGy_r: LinearSegmentedColormap
RdPu: LinearSegmentedColormap
RdPu_r: LinearSegmentedColormap
RdYlBu: LinearSegmentedColormap
RdYlBu_r: LinearSegmentedColormap
RdYlGn: LinearSegmentedColormap
RdYlGn_r: LinearSegmentedColormap
Reds: LinearSegmentedColormap
Reds_r: LinearSegmentedColormap
Set1: ListedColormap
Set1_r: ListedColormap
Set2: ListedColormap
Set2_r: ListedColormap
Set3: ListedColormap
Set3_r: ListedColormap
Spectral: LinearSegmentedColormap
Spectral_r: LinearSegmentedColormap
Wistia: LinearSegmentedColormap
Wistia_r: LinearSegmentedColormap
YlGn: LinearSegmentedColormap
YlGnBu: LinearSegmentedColormap
YlGnBu_r: LinearSegmentedColormap
YlGn_r: LinearSegmentedColormap
YlOrBr: LinearSegmentedColormap
YlOrBr_r: LinearSegmentedColormap
YlOrRd: LinearSegmentedColormap
YlOrRd_r: LinearSegmentedColormap
afmhot: LinearSegmentedColormap
afmhot_r: LinearSegmentedColormap
autumn: LinearSegmentedColormap
autumn_r: LinearSegmentedColormap
berlin: ListedColormap
berlin_r: ListedColormap
binary: LinearSegmentedColormap
binary_r: LinearSegmentedColormap

class _BivarCmaps(TypedDict):
    BiPeak: SegmentedBivarColormap
    BiOrangeBlue: SegmentedBivarColormap
    BiCone: SegmentedBivarColormap

bivar_cmaps: _BivarCmaps

bone: LinearSegmentedColormap
bone_r: LinearSegmentedColormap
brg: LinearSegmentedColormap
brg_r: LinearSegmentedColormap
bwr: LinearSegmentedColormap
bwr_r: LinearSegmentedColormap
cividis: ListedColormap
cividis_r: ListedColormap

class _CmapsListed(TypedDict):
    magma: ListedColormap
    inferno: ListedColormap
    plasma: ListedColormap
    viridis: ListedColormap
    cividis: ListedColormap
    twilight: ListedColormap
    twilight_shifted: ListedColormap
    turbo: ListedColormap
    berlin: ListedColormap
    managua: ListedColormap
    vanimo: ListedColormap

cmaps_listed: _CmapsListed

class _Datad(TypedDict):
    Blues: tuple
    BrBG: tuple
    BuGn: tuple
    BuPu: tuple
    CMRmap: dict
    GnBu: tuple
    Greens: tuple
    Greys: tuple
    OrRd: tuple
    Oranges: tuple
    PRGn: tuple
    PiYG: tuple
    PuBu: tuple
    PuBuGn: tuple
    PuOr: tuple
    PuRd: tuple
    Purples: tuple
    RdBu: tuple
    RdGy: tuple
    RdPu: tuple
    RdYlBu: tuple
    RdYlGn: tuple
    Reds: tuple
    Spectral: tuple
    Wistia: dict
    YlGn: tuple
    YlGnBu: tuple
    YlOrBr: tuple
    YlOrRd: tuple
    afmhot: dict
    autumn: dict
    binary: dict
    bone: dict
    brg: tuple
    bwr: tuple
    cool: dict
    coolwarm: dict
    copper: dict
    cubehelix: dict
    flag: dict
    gist_earth: dict
    gist_gray: dict
    gist_heat: dict
    gist_ncar: dict
    gist_rainbow: tuple
    gist_stern: dict
    gist_yarg: dict
    gnuplot: dict
    gnuplot2: dict
    gray: dict
    hot: dict
    hsv: dict
    jet: dict
    nipy_spectral: dict
    ocean: dict
    pink: dict
    prism: dict
    rainbow: dict
    seismic: tuple
    spring: dict
    summer: dict
    terrain: tuple
    winter: dict
    Accent: dict
    Dark2: dict
    Paired: dict
    Pastel1: dict
    Pastel2: dict
    Set1: dict
    Set2: dict
    Set3: dict
    tab10: dict
    tab20: dict
    tab20b: dict
    tab20c: dict

datad: _Datad

flag: LinearSegmentedColormap
flag_r: LinearSegmentedColormap
gist_earth: LinearSegmentedColormap
gist_earth_r: LinearSegmentedColormap
gist_gray: LinearSegmentedColormap
gist_gray_r: LinearSegmentedColormap
gist_grey: LinearSegmentedColormap
gist_grey_r: LinearSegmentedColormap
gist_heat: LinearSegmentedColormap
gist_heat_r: LinearSegmentedColormap
gist_ncar: LinearSegmentedColormap
gist_ncar_r: LinearSegmentedColormap
gist_rainbow: LinearSegmentedColormap
gist_rainbow_r: LinearSegmentedColormap
gist_stern: LinearSegmentedColormap
gist_stern_r: LinearSegmentedColormap
gist_yarg: LinearSegmentedColormap
gist_yarg_r: LinearSegmentedColormap
gist_yerg: LinearSegmentedColormap
gist_yerg_r: LinearSegmentedColormap
gnuplot: LinearSegmentedColormap
gnuplot2: LinearSegmentedColormap
gnuplot2_r: LinearSegmentedColormap
gnuplot_r: LinearSegmentedColormap
gray: LinearSegmentedColormap
gray_r: LinearSegmentedColormap
grey: LinearSegmentedColormap
grey_r: LinearSegmentedColormap
hot: LinearSegmentedColormap
hot_r: LinearSegmentedColormap
hsv: LinearSegmentedColormap
hsv_r: LinearSegmentedColormap
inferno: ListedColormap
inferno_r: ListedColormap
jet: LinearSegmentedColormap
jet_r: LinearSegmentedColormap
magma: ListedColormap
magma_r: ListedColormap
managua: ListedColormap
managua_r: ListedColormap

_MultivarCmaps = TypedDict("_MultivarCmaps", {"2VarAddA": MultivarColormap, "2VarSubA": MultivarColormap, "3VarAddA": MultivarColormap})
multivar_cmaps: _MultivarCmaps

nipy_spectral: LinearSegmentedColormap
nipy_spectral_r: LinearSegmentedColormap
ocean: LinearSegmentedColormap
ocean_r: LinearSegmentedColormap
pink: LinearSegmentedColormap
pink_r: LinearSegmentedColormap
plasma: ListedColormap
plasma_r: ListedColormap
prism: LinearSegmentedColormap
prism_r: LinearSegmentedColormap
rainbow: LinearSegmentedColormap
rainbow_r: LinearSegmentedColormap
seismic: LinearSegmentedColormap
seismic_r: LinearSegmentedColormap
spring: LinearSegmentedColormap
spring_r: LinearSegmentedColormap
summer: LinearSegmentedColormap
summer_r: LinearSegmentedColormap
tab10: ListedColormap
tab10_r: ListedColormap
tab20: ListedColormap
tab20_r: ListedColormap
tab20b: ListedColormap
tab20b_r: ListedColormap
tab20c: ListedColormap
tab20c_r: ListedColormap
terrain: LinearSegmentedColormap
terrain_r: LinearSegmentedColormap
turbo: ListedColormap
turbo_r: ListedColormap
twilight: ListedColormap
twilight_r: ListedColormap
twilight_shifted: ListedColormap
twilight_shifted_r: ListedColormap
vanimo: ListedColormap
vanimo_r: ListedColormap
viridis: ListedColormap
viridis_r: ListedColormap
winter: LinearSegmentedColormap
winter_r: LinearSegmentedColormap
