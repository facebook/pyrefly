from collections.abc import (
    Callable,
    MutableMapping,
    Sequence,
)
from typing import (
    Any,
    Literal,
    Protocol,
    overload,
)

from matplotlib.colors import Colormap
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._typing import (
    Axis,
    ExcelWriterMergeCells,
    FilePath,
    HashableT,
    HashableT1,
    HashableT2,
    IndexLabel,
    IntervalClosedType,
    Level,
    QuantileInterpolation,
    Scalar,
    StorageOptions,
    T,
    WriteBuffer,
    WriteExcelBuffer,
    np_ndarray,
    np_ndarray_str,
)

from pandas.io.excel import ExcelWriter
from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    ExtFormatter,
    StyleExportDict,
    StylerRenderer,
    Subset,
)

class _SeriesFunc(Protocol):
    def __call__(
        self, series: Series, /, *args: Any, **kwargs: Any
    ) -> list | Series: ...

class _DataFrameFunc(Protocol):
    def __call__(
        self, series: DataFrame, /, *args: Any, **kwargs: Any
    ) -> np_ndarray | DataFrame: ...

class _MapCallable(Protocol):
    def __call__(
        self, first_arg: Scalar, /, *args: Any, **kwargs: Any
    ) -> str | None: ...

class Styler(StylerRenderer):
    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = ...,
        table_styles: CSSStyles | None = ...,
        uuid: str | None = ...,
        caption: str | tuple[str, str] | None = ...,
        table_attributes: str | None = ...,
        cell_ids: bool = ...,
        na_rep: str | None = ...,
        uuid_len: int = ...,
        decimal: str | None = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        formatter: ExtFormatter | None = ...,
    ) -> None: ...
    def concat(self, other: Styler) -> Styler: ...
    @overload
    def map(
        self,
        func: Callable[[Scalar], str | None],
        subset: Subset | None = ...,
    ) -> Styler: ...
    @overload
    def map(
        self,
        func: _MapCallable,
        subset: Subset | None = ...,
        **kwargs: Any,
    ) -> Styler: ...
    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = ...,
        css_class: str | None = ...,
        as_title_attribute: bool = ...,
    ) -> Styler: ...
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: list[HashableT1] | None = None,
        header: list[HashableT2] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Literal["openpyxl", "xlsxwriter"] | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: Literal["centering", "raggedleft", "raggedright"] | None = ...,
        hrules: bool | None = ...,
        clines: (
            Literal["all;data", "all;index", "skip-last;data", "skip-last;index"] | None
        ) = ...,
        label: str | None = ...,
        caption: str | tuple[str, str] | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: Literal["c", "t", "b", "naive"] | None = ...,
        multicol_align: Literal["r", "c", "l", "naive-l", "naive-r"] | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        buf: None = None,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: Literal["centering", "raggedleft", "raggedright"] | None = ...,
        hrules: bool | None = ...,
        clines: (
            Literal["all;data", "all;index", "skip-last;data", "skip-last;index"] | None
        ) = ...,
        label: str | None = ...,
        caption: str | tuple[str, str] | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: Literal["c", "t", "b", "naive"] | None = ...,
        multicol_align: Literal["r", "c", "l", "naive-l", "naive-r"] | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> str: ...
    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def to_html(
        self,
        buf: None = None,
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs: Any,
    ) -> str: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = None,
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> str: ...
    def set_td_classes(self, classes: DataFrame) -> Styler: ...
    def __copy__(self) -> Styler: ...
    def __deepcopy__(self, memo: MutableMapping[int, Any] | None) -> Styler: ...
    def clear(self) -> None: ...
    @overload
    def apply(
        self,
        func: _SeriesFunc | Callable[[Series], list | Series],
        axis: Axis = ...,
        subset: Subset | None = ...,
        **kwargs: Any,
    ) -> Styler: ...
    @overload
    def apply(
        self,
        func: _DataFrameFunc | Callable[[DataFrame], np_ndarray | DataFrame],
        axis: None,
        subset: Subset | None = ...,
        **kwargs: Any,
    ) -> Styler: ...
    def apply_index(
        self,
        func: Callable[[Series], list[str] | np_ndarray_str | Series[str]],
        axis: Axis = ...,
        level: Level | list[Level] | None = ...,
        **kwargs: Any,
    ) -> Styler: ...
    def map_index(
        self,
        func: Callable[[Scalar], str | None],
        axis: Axis = ...,
        level: Level | list[Level] | None = ...,
        **kwargs: Any,
    ) -> Styler: ...
    def set_table_attributes(self, attributes: str) -> Styler: ...
    def export(self) -> StyleExportDict: ...
    def use(self, styles: StyleExportDict) -> Styler: ...
    def set_uuid(self, uuid: str) -> Styler: ...
    def set_caption(self, caption: str | tuple[str, str]) -> Styler: ...
    def set_sticky(
        self,
        axis: Axis = 0,
        pixel_size: int | None = None,
        levels: Level | list[Level] | None = None,
    ) -> Styler: ...
    def set_table_styles(
        self,
        table_styles: dict[HashableT, CSSStyles] | CSSStyles | None = None,
        axis: Axis = 0,
        overwrite: bool = True,
        css_class_names: dict[str, str] | None = None,
    ) -> Styler: ...
    def hide(
        self,
        subset: Subset | None = ...,
        axis: Axis = ...,
        level: Level | list[Level] | None = ...,
        names: bool = ...,
    ) -> Styler: ...
    def background_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: (
            Sequence[float]
            | Sequence[Sequence[float]]
            | np_ndarray
            | DataFrame
            | Series
            | None
        ) = None,
    ) -> Styler: ...
    def text_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: (
            Sequence[float]
            | Sequence[Sequence[float]]
            | np_ndarray
            | DataFrame
            | Series
            | None
        ) = None,
    ) -> Styler: ...
    def set_properties(
        self, subset: Subset | None = ..., **kwargs: str | int
    ) -> Styler: ...
    def bar(
        self,
        subset: Subset | None = None,
        axis: Axis | None = 0,
        *,
        color: str | list[str] | tuple[str, str] | None = None,
        cmap: str | Colormap | None = None,
        width: float = 100,
        height: float = 100,
        align: (
            Literal["left", "right", "zero", "mid", "mean"]
            | float
            | Callable[[Series | np_ndarray | DataFrame], float]
        ) = "mid",
        vmin: float | None = None,
        vmax: float | None = None,
        props: str = "width: 10em;",
    ) -> Styler: ...
    def highlight_null(
        self,
        color: str | None = "red",
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler: ...
    def highlight_max(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler: ...
    def highlight_min(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler: ...
    def highlight_between(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        left: Scalar | list[Scalar] | None = None,
        right: Scalar | list[Scalar] | None = None,
        inclusive: IntervalClosedType = "both",
        props: str | None = None,
    ) -> Styler: ...
    def highlight_quantile(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        q_left: float = 0,
        q_right: float = 1,
        interpolation: QuantileInterpolation = "linear",
        inclusive: IntervalClosedType = "both",
        props: str | None = None,
    ) -> Styler: ...
    @classmethod
    def from_custom_template(
        cls,
        searchpath: str | list[str],
        html_table: str | None = ...,
        html_style: str | None = ...,
    ) -> type[Styler]: ...
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
