# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from types import EllipsisType
from typing import Any, Literal, overload

import shape_extensions
from numpy.__config__ import (
    show as show_config,
)
from numpy._array_api_info import (
    __array_namespace_info__ as __array_namespace_info__,
)
from numpy._core._asarray import (
    require as require,
)
from numpy._core._type_aliases import (
    sctypeDict as sctypeDict,
)
from numpy._core._ufunc_config import (
    errstate as errstate,
    getbufsize as getbufsize,
    geterr as geterr,
    geterrcall as geterrcall,
    setbufsize as setbufsize,
    seterr as seterr,
    seterrcall as seterrcall,
)
from numpy._core.arrayprint import (
    array2string as array2string,
    array_repr as array_repr,
    array_str as array_str,
    format_float_positional as format_float_positional,
    format_float_scientific as format_float_scientific,
    get_printoptions as get_printoptions,
    printoptions as printoptions,
    set_printoptions as set_printoptions,
)
from numpy._core.einsumfunc import (
    einsum as einsum,
    einsum_path as einsum_path,
)
from numpy._core.fromnumeric import (
    all as all,
    amax as amax,
    amin as amin,
    any as any,
    argmax as argmax,
    argpartition as argpartition,
    argsort as argsort,
    around as around,
    choose as choose,
    compress as compress,
    cumprod as cumprod,
    cumsum as cumsum,
    cumulative_prod as cumulative_prod,
    cumulative_sum as cumulative_sum,
    diagonal as diagonal,
    matrix_transpose as matrix_transpose,
    ndim as ndim,
    nonzero as nonzero,
    partition as partition,
    prod as prod,
    ptp as ptp,
    put as put,
    ravel as ravel,
    repeat as repeat,
    reshape as reshape,
    resize as resize,
    searchsorted as searchsorted,
    shape as shape,
    size as size,
    sort as sort,
    squeeze as squeeze,
    std as std,
    swapaxes as swapaxes,
    take as take,
    trace as trace,
    transpose as transpose,
    var as var,
)
from numpy._core.function_base import (
    geomspace as geomspace,
    linspace as linspace,
    logspace as logspace,
)
from numpy._core.getlimits import (
    finfo as finfo,
    iinfo as iinfo,
)
from numpy._core.memmap import (
    memmap as memmap,
)
from numpy._core.multiarray import (
    array as array,
    asanyarray as asanyarray,
    asarray as asarray,
    ascontiguousarray as ascontiguousarray,
    asfortranarray as asfortranarray,
    bincount as bincount,
    busday_count as busday_count,
    busday_offset as busday_offset,
    busdaycalendar as busdaycalendar,
    can_cast as can_cast,
    concatenate as concatenate,
    copyto as copyto,
    datetime_as_string as datetime_as_string,
    datetime_data as datetime_data,
    dot as dot,
    empty_like as empty_like,
    flatiter as flatiter,
    frombuffer as frombuffer,
    fromfile as fromfile,
    fromiter as fromiter,
    frompyfunc as frompyfunc,
    fromstring as fromstring,
    inner as inner,
    is_busday as is_busday,
    lexsort as lexsort,
    may_share_memory as may_share_memory,
    min_scalar_type as min_scalar_type,
    nditer as nditer,
    nested_iters as nested_iters,
    packbits as packbits,
    promote_types as promote_types,
    putmask as putmask,
    result_type as result_type,
    shares_memory as shares_memory,
    unpackbits as unpackbits,
    vdot as vdot,
    where as where,
)
from numpy._core.numeric import (
    allclose as allclose,
    argwhere as argwhere,
    array_equal as array_equal,
    array_equiv as array_equiv,
    astype as astype,
    base_repr as base_repr,
    binary_repr as binary_repr,
    convolve as convolve,
    correlate as correlate,
    count_nonzero as count_nonzero,
    cross as cross,
    flatnonzero as flatnonzero,
    fromfunction as fromfunction,
    full_like as full_like,
    indices as indices,
    isclose as isclose,
    isfortran as isfortran,
    isscalar as isscalar,
    moveaxis as moveaxis,
    ones_like as ones_like,
    outer as outer,
    roll as roll,
    rollaxis as rollaxis,
    tensordot as tensordot,
    zeros_like as zeros_like,
)
from numpy._core.numerictypes import (
    isdtype as isdtype,
    issubdtype as issubdtype,
    ScalarType as ScalarType,
    typecodes as typecodes,
)
from numpy._core.records import (
    recarray as recarray,
    record as record,
)
from numpy._core.shape_base import (
    atleast_1d as atleast_1d,
    atleast_2d as atleast_2d,
    atleast_3d as atleast_3d,
    block as block,
    hstack as hstack,
    stack as stack,
    unstack as unstack,
    vstack as vstack,
)
from numpy._shapes import (
    diag_extent,
    matmul_shape,
    matvec_shape,
    reduce_shape,
    vecdot_shape,
    vecmat_shape,
)
from numpy._typing._extended_precision import (
    complex256 as complex256,
    float128 as float128,
)
from numpy.lib import (
    scimath as emath,
)
from numpy.lib._arraypad_impl import (
    pad as pad,
)
from numpy.lib._arraysetops_impl import (
    ediff1d as ediff1d,
    intersect1d as intersect1d,
    isin as isin,
    setdiff1d as setdiff1d,
    setxor1d as setxor1d,
    union1d as union1d,
    unique as unique,
    unique_all as unique_all,
    unique_counts as unique_counts,
    unique_inverse as unique_inverse,
    unique_values as unique_values,
)
from numpy.lib._function_base_impl import (
    angle as angle,
    append as append,
    asarray_chkfinite as asarray_chkfinite,
    average as average,
    bartlett as bartlett,
    blackman as blackman,
    copy as copy,
    corrcoef as corrcoef,
    cov as cov,
    delete as delete,
    diff as diff,
    digitize as digitize,
    extract as extract,
    flip as flip,
    gradient as gradient,
    hamming as hamming,
    hanning as hanning,
    i0 as i0,
    insert as insert,
    interp as interp,
    iterable as iterable,
    kaiser as kaiser,
    median as median,
    meshgrid as meshgrid,
    percentile as percentile,
    piecewise as piecewise,
    place as place,
    quantile as quantile,
    rot90 as rot90,
    select as select,
    sinc as sinc,
    sort_complex as sort_complex,
    trapezoid as trapezoid,
    trim_zeros as trim_zeros,
    unwrap as unwrap,
    vectorize as vectorize,
)
from numpy.lib._histograms_impl import (
    histogram as histogram,
    histogram_bin_edges as histogram_bin_edges,
    histogramdd as histogramdd,
)
from numpy.lib._index_tricks_impl import (
    c_ as c_,
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
    index_exp as index_exp,
    ix_ as ix_,
    mgrid as mgrid,
    ndenumerate as ndenumerate,
    ndindex as ndindex,
    ogrid as ogrid,
    r_ as r_,
    ravel_multi_index as ravel_multi_index,
    s_ as s_,
    unravel_index as unravel_index,
)
from numpy.lib._nanfunctions_impl import (
    nanargmax as nanargmax,
    nanargmin as nanargmin,
    nancumprod as nancumprod,
    nancumsum as nancumsum,
    nanmax as nanmax,
    nanmean as nanmean,
    nanmedian as nanmedian,
    nanmin as nanmin,
    nanpercentile as nanpercentile,
    nanprod as nanprod,
    nanquantile as nanquantile,
    nanstd as nanstd,
    nansum as nansum,
    nanvar as nanvar,
)
from numpy.lib._npyio_impl import (
    fromregex as fromregex,
    genfromtxt as genfromtxt,
    load as load,
    loadtxt as loadtxt,
    save as save,
    savetxt as savetxt,
    savez as savez,
    savez_compressed as savez_compressed,
)
from numpy.lib._polynomial_impl import (
    poly as poly,
    poly1d as poly1d,
    polyadd as polyadd,
    polyder as polyder,
    polydiv as polydiv,
    polyfit as polyfit,
    polyint as polyint,
    polymul as polymul,
    polysub as polysub,
    polyval as polyval,
    roots as roots,
)
from numpy.lib._shape_base_impl import (
    apply_along_axis as apply_along_axis,
    apply_over_axes as apply_over_axes,
    array_split as array_split,
    column_stack as column_stack,
    dsplit as dsplit,
    dstack as dstack,
    hsplit as hsplit,
    kron as kron,
    put_along_axis as put_along_axis,
    split as split,
    take_along_axis as take_along_axis,
    tile as tile,
    vsplit as vsplit,
)
from numpy.lib._stride_tricks_impl import (
    broadcast_arrays as broadcast_arrays,
    broadcast_shapes as broadcast_shapes,
    broadcast_to as broadcast_to,
)
from numpy.lib._twodim_base_impl import (
    diagflat as diagflat,
    fliplr as fliplr,
    flipud as flipud,
    histogram2d as histogram2d,
    mask_indices as mask_indices,
    tri as tri,
    tril as tril,
    tril_indices as tril_indices,
    tril_indices_from as tril_indices_from,
    triu as triu,
    triu_indices as triu_indices,
    triu_indices_from as triu_indices_from,
    vander as vander,
)
from numpy.lib._type_check_impl import (
    common_type as common_type,
    imag as imag,
    iscomplex as iscomplex,
    iscomplexobj as iscomplexobj,
    isreal as isreal,
    isrealobj as isrealobj,
    mintypecode as mintypecode,
    nan_to_num as nan_to_num,
    real as real,
    real_if_close as real_if_close,
    typename as typename,
)
from numpy.lib._ufunclike_impl import (
    fix as fix,
    isneginf as isneginf,
    isposinf as isposinf,
)
from numpy.lib._utils_impl import (
    get_include as get_include,
    info as info,
    show_runtime as show_runtime,
)
from numpy.matrixlib import (
    asmatrix as asmatrix,
    bmat as bmat,
    matrix as matrix,
)
from shape_extensions import broadcast, Flag, Index, index_shape, Int, IntTuple, IntVar

# Preserve NumPy's canonical re-exports before local shape-aware declarations.
from . import (
    char as char,
    core as core,
    ctypeslib as ctypeslib,
    dtypes as dtypes,
    exceptions as exceptions,
    f2py as f2py,
    fft as fft,
    lib as lib,
    linalg as linalg,
    ma as ma,
    polynomial as polynomial,
    random as random,
    rec as rec,
    strings as strings,
    testing as testing,
    typing as typing,
)

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None
type _BasicIndex = int | slice | list[int] | None | EllipsisType

class generic: ...
class bool_(generic): ...
class float32(generic): ...
class float64(generic): ...
class int32(generic): ...
class int64(generic): ...
class intp(generic): ...

type _IndexScalar = int | bool_ | int32 | int64 | intp
type _IndexSequence = Sequence[_IndexScalar] | Sequence[Sequence[_IndexScalar]]

class dtype[Scalar = Any]:
    @overload
    def __new__[ScalarT: generic](cls, dtype: type[ScalarT]) -> dtype[ScalarT]: ...
    @overload
    def __new__(cls, dtype: Any = ...) -> dtype: ...
    def __init__(self, dtype: Any = ...) -> None: ...

# `ndarray` declares a `dtype` attribute, which shadows the class above throughout
# its body. Annotations inside the class reach the class through this alias.
_dtype = dtype

class ndarray[Shape: _Shape = _Shape, DType = Any]:
    shape: Shape
    dtype: DType
    @overload
    def __len__[N: IntVar](self: ndarray[[N]]) -> Int[N]: ...
    @overload
    def __len__[N: IntVar, M: IntVar](self: ndarray[[N, M]]) -> Int[N]: ...
    @overload
    def __getitem__[
        N: IntVar,
        M: IntVar,
        I: IntVar,
        RowIndexScalar: (int32, int64, intp),
        ColumnIndexScalar: (int32, int64, intp),
    ](
        self: ndarray[[N, M], DType],
        key: tuple[
            ndarray[[I], _dtype[RowIndexScalar]],
            ndarray[[I], _dtype[ColumnIndexScalar]],
        ],
    ) -> ndarray[[I], DType]: ...
    @overload
    def __getitem__[I: Index](
        self: ndarray[Shape, DType], key: I
    ) -> ndarray[index_shape(Shape, I), DType]: ...
    # TODO(stroxler): Model general array-valued indices precisely enough to
    # reject non-integer dtypes and incompatible advanced-index shapes.
    @overload
    def __getitem__(
        self: ndarray[Shape, DType],
        key: _BasicIndex
        | _IndexScalar
        | _IndexSequence
        | ndarray
        | tuple[_BasicIndex | _IndexScalar | _IndexSequence | ndarray, ...],
    ) -> ndarray[IntTuple, DType]: ...
    # Only 2-D transpose is modeled for the NumPy shape-stub MVP.
    @property
    def T[N: IntVar, P: IntVar](
        self: ndarray[[N, P], DType],
    ) -> ndarray[[P, N], DType]: ...
    @overload
    def __add__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __add__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __radd__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __radd__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __sub__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __sub__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __rsub__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __rsub__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __truediv__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __truediv__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __rtruediv__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __rtruediv__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __mul__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __mul__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __rmul__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __rmul__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __floordiv__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __floordiv__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __rfloordiv__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __rfloordiv__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __mod__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __mod__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __rmod__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __rmod__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    def __neg__(self) -> ndarray[Shape, DType]: ...
    def __pos__(self) -> ndarray[Shape, DType]: ...
    @overload
    def __pow__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __pow__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    @overload
    def __rpow__(self, other: int | float) -> ndarray[Shape, DType]: ...
    @overload
    def __rpow__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[broadcast(Shape, OtherShape), DType]: ...
    def __matmul__[OtherShape: _Shape](
        self, other: ndarray[OtherShape]
    ) -> ndarray[matmul_shape(Shape, OtherShape), DType]: ...
    def mean[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        *,
        keepdims: KeepDims = False,
    ) -> ndarray[reduce_shape(Shape, Axis, KeepDims), DType]: ...
    def sum[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        *,
        keepdims: KeepDims = False,
    ) -> ndarray[reduce_shape(Shape, Axis, KeepDims), DType]: ...
    def min[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        *,
        keepdims: KeepDims = False,
    ) -> ndarray[reduce_shape(Shape, Axis, KeepDims), DType]: ...
    def max[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        *,
        keepdims: KeepDims = False,
    ) -> ndarray[reduce_shape(Shape, Axis, KeepDims), DType]: ...

class ufunc:
    __name__: str
    nin: int
    nout: int
    nargs: int
    ntypes: int
    types: list[str]
    identity: Any
    signature: str | None
    def __call__(self, /, *args: Any, **kwargs: Any) -> Any: ...
    def accumulate(self, array: Any, /, *args: Any, **kwargs: Any) -> Any: ...
    def reduce(self, array: Any, /, *args: Any, **kwargs: Any) -> Any: ...
    def reduceat(
        self, array: Any, indices: Any, /, *args: Any, **kwargs: Any
    ) -> Any: ...
    def outer(self, a: Any, b: Any, /, **kwargs: Any) -> Any: ...
    def at(self, a: ndarray, indices: Any, b: Any = None, /) -> None: ...

class _UnaryUFunc(ufunc):
    @overload
    def __call__[Shape: _Shape](
        self, x: ndarray[Shape], /, out: Any = None, **kwargs: Any
    ) -> ndarray[Shape]: ...
    @overload
    def __call__(self, x: Any, /, out: Any = None, **kwargs: Any) -> Any: ...

class _BinaryUFunc(ufunc):
    @overload
    def __call__[Shape1: _Shape, Shape2: _Shape](
        self,
        x1: ndarray[Shape1],
        x2: ndarray[Shape2],
        /,
        out: Any = None,
        **kwargs: Any,
    ) -> ndarray[broadcast(Shape1, Shape2)]: ...
    @overload
    def __call__[Shape: _Shape](
        self, x1: ndarray[Shape], x2: Any, /, out: Any = None, **kwargs: Any
    ) -> ndarray[Shape]: ...
    @overload
    def __call__[Shape: _Shape](
        self, x1: Any, x2: ndarray[Shape], /, out: Any = None, **kwargs: Any
    ) -> ndarray[Shape]: ...
    @overload
    def __call__(self, x1: Any, x2: Any, /, out: Any = None, **kwargs: Any) -> Any: ...

# The result dtype stays gradual because dtype promotion is not modeled, while
# `ndarray.__matmul__` carries the left operand's dtype.
class _MatmulUFunc(ufunc):
    def __call__[LeftShape: _Shape, RightShape: _Shape](
        self,
        a: ndarray[LeftShape],
        b: ndarray[RightShape],
        /,
        out: Any = None,
        **kwargs: Any,
    ) -> ndarray[matmul_shape(LeftShape, RightShape), Any]: ...

class _UnaryTwoOutputUFunc(ufunc):
    @overload
    def __call__[Shape: _Shape](
        self, x: ndarray[Shape], /, out: Any = None, *outputs: Any, **kwargs: Any
    ) -> tuple[ndarray[Shape], ndarray[Shape]]: ...
    @overload
    def __call__(
        self, x: Any, /, out: Any = None, *outputs: Any, **kwargs: Any
    ) -> tuple[Any, Any]: ...

class _BinaryTwoOutputUFunc(ufunc):
    @overload
    def __call__[Shape1: _Shape, Shape2: _Shape](
        self,
        x1: ndarray[Shape1],
        x2: ndarray[Shape2],
        /,
        out: Any = None,
        *outputs: Any,
        **kwargs: Any,
    ) -> tuple[
        ndarray[broadcast(Shape1, Shape2)],
        ndarray[broadcast(Shape1, Shape2)],
    ]: ...
    @overload
    def __call__[Shape: _Shape](
        self,
        x1: ndarray[Shape],
        x2: Any,
        /,
        out: Any = None,
        *outputs: Any,
        **kwargs: Any,
    ) -> tuple[ndarray[Shape], ndarray[Shape]]: ...
    @overload
    def __call__[Shape: _Shape](
        self,
        x1: Any,
        x2: ndarray[Shape],
        /,
        out: Any = None,
        *outputs: Any,
        **kwargs: Any,
    ) -> tuple[ndarray[Shape], ndarray[Shape]]: ...
    @overload
    def __call__(
        self,
        x1: Any,
        x2: Any,
        /,
        out: Any = None,
        *outputs: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]: ...

class _MatvecUFunc(ufunc):
    @overload
    def __call__[Shape1: _Shape, Shape2: _Shape](
        self,
        x1: ndarray[Shape1],
        x2: ndarray[Shape2],
        /,
        out: Any = None,
        **kwargs: Any,
    ) -> ndarray[matvec_shape(Shape1, Shape2)]: ...
    @overload
    def __call__(self, x1: Any, x2: Any, /, out: Any = None, **kwargs: Any) -> Any: ...

class _VecdotUFunc(ufunc):
    @overload
    def __call__[Shape1: _Shape, Shape2: _Shape](
        self,
        x1: ndarray[Shape1],
        x2: ndarray[Shape2],
        /,
        out: Any = None,
        **kwargs: Any,
    ) -> ndarray[vecdot_shape(Shape1, Shape2)]: ...
    @overload
    def __call__(self, x1: Any, x2: Any, /, out: Any = None, **kwargs: Any) -> Any: ...

class _VecmatUFunc(ufunc):
    @overload
    def __call__[Shape1: _Shape, Shape2: _Shape](
        self,
        x1: ndarray[Shape1],
        x2: ndarray[Shape2],
        /,
        out: Any = None,
        **kwargs: Any,
    ) -> ndarray[vecmat_shape(Shape1, Shape2)]: ...
    @overload
    def __call__(self, x1: Any, x2: Any, /, out: Any = None, **kwargs: Any) -> Any: ...

abs: _UnaryUFunc
absolute: _UnaryUFunc
acos: _UnaryUFunc
acosh: _UnaryUFunc
arccos: _UnaryUFunc
arccosh: _UnaryUFunc
arcsin: _UnaryUFunc
arcsinh: _UnaryUFunc
arctan: _UnaryUFunc
arctanh: _UnaryUFunc
asin: _UnaryUFunc
asinh: _UnaryUFunc
atan: _UnaryUFunc
atanh: _UnaryUFunc
bitwise_count: _UnaryUFunc
bitwise_invert: _UnaryUFunc
bitwise_not: _UnaryUFunc
cbrt: _UnaryUFunc
ceil: _UnaryUFunc
conj: _UnaryUFunc
conjugate: _UnaryUFunc
cos: _UnaryUFunc
cosh: _UnaryUFunc
deg2rad: _UnaryUFunc
degrees: _UnaryUFunc
exp: _UnaryUFunc
exp2: _UnaryUFunc
expm1: _UnaryUFunc
fabs: _UnaryUFunc
floor: _UnaryUFunc
invert: _UnaryUFunc
isfinite: _UnaryUFunc
isinf: _UnaryUFunc
isnan: _UnaryUFunc
isnat: _UnaryUFunc
log: _UnaryUFunc
log10: _UnaryUFunc
log1p: _UnaryUFunc
log2: _UnaryUFunc
logical_not: _UnaryUFunc
negative: _UnaryUFunc
positive: _UnaryUFunc
rad2deg: _UnaryUFunc
radians: _UnaryUFunc
reciprocal: _UnaryUFunc
rint: _UnaryUFunc
sign: _UnaryUFunc
signbit: _UnaryUFunc
sin: _UnaryUFunc
sinh: _UnaryUFunc
spacing: _UnaryUFunc
sqrt: _UnaryUFunc
square: _UnaryUFunc
tan: _UnaryUFunc
tanh: _UnaryUFunc
trunc: _UnaryUFunc

add: _BinaryUFunc
arctan2: _BinaryUFunc
atan2: _BinaryUFunc
bitwise_and: _BinaryUFunc
bitwise_left_shift: _BinaryUFunc
bitwise_or: _BinaryUFunc
bitwise_right_shift: _BinaryUFunc
bitwise_xor: _BinaryUFunc
copysign: _BinaryUFunc
divide: _BinaryUFunc
equal: _BinaryUFunc
float_power: _BinaryUFunc
floor_divide: _BinaryUFunc
fmax: _BinaryUFunc
fmin: _BinaryUFunc
fmod: _BinaryUFunc
gcd: _BinaryUFunc
greater: _BinaryUFunc
greater_equal: _BinaryUFunc
heaviside: _BinaryUFunc
hypot: _BinaryUFunc
lcm: _BinaryUFunc
ldexp: _BinaryUFunc
left_shift: _BinaryUFunc
less: _BinaryUFunc
less_equal: _BinaryUFunc
logaddexp: _BinaryUFunc
logaddexp2: _BinaryUFunc
logical_and: _BinaryUFunc
logical_or: _BinaryUFunc
logical_xor: _BinaryUFunc
matmul: _MatmulUFunc
maximum: _BinaryUFunc
minimum: _BinaryUFunc
mod: _BinaryUFunc
multiply: _BinaryUFunc
nextafter: _BinaryUFunc
not_equal: _BinaryUFunc
pow: _BinaryUFunc
power: _BinaryUFunc
remainder: _BinaryUFunc
right_shift: _BinaryUFunc
subtract: _BinaryUFunc
true_divide: _BinaryUFunc

frexp: _UnaryTwoOutputUFunc
modf: _UnaryTwoOutputUFunc
divmod: _BinaryTwoOutputUFunc
matvec: _MatvecUFunc
vecdot: _VecdotUFunc
vecmat: _VecmatUFunc

def round[Shape: _Shape](x: ndarray[Shape]) -> ndarray[Shape]: ...
def clip[Shape: _Shape](
    a: ndarray[Shape], a_min: int | float, a_max: int | float
) -> ndarray[Shape]: ...
def fill_diagonal[N: IntVar, DType](
    a: ndarray[[N, N], DType],
    val: Any,
    wrap: bool = False,
) -> None: ...
@overload
def diag[N: IntVar, DType, K: Flag[int] = 0](
    v: ndarray[[N], DType], k: K = 0
) -> ndarray[[diag_extent(Int[N], K), diag_extent(Int[N], K)], DType]: ...

# TODO(stroxler): Model the shape arithmetic here; we can do better than `int`.
@overload
def diag[M: IntVar, N: IntVar, DType](
    v: ndarray[[M, N], DType], k: int = 0
) -> ndarray[[int], DType]: ...

# Trailing fallback for ranks the precise overloads do not model, so their dtype survives
# instead of degrading to `Any`. The parameter shape is a type variable rather than
# `IntTuple`: a gradual parameter shape would also match known-rank arguments whose dtype is
# gradual, and that ambiguity collapses their precise result to a gradual shape.
@overload
def diag[S: _Shape, DType](
    v: ndarray[S, DType], k: int = 0
) -> ndarray[IntTuple, DType]: ...
def arange[N: IntVar](stop: Int[N], /) -> ndarray[[N], dtype[intp]]: ...
@overload
def expand_dims[N: IntVar, M: IntVar, DType](
    a: ndarray[[N, M], DType],
    axis: Literal[0, -3],
) -> ndarray[[1, N, M], DType]: ...
@overload
def expand_dims[N: IntVar, M: IntVar, DType](
    a: ndarray[[N, M], DType],
    axis: Literal[1, -2],
) -> ndarray[[N, 1, M], DType]: ...
@overload
def expand_dims[N: IntVar, M: IntVar, DType](
    a: ndarray[[N, M], DType],
    axis: Literal[2, -1],
) -> ndarray[[N, M, 1], DType]: ...

# These stubs track reduction shapes but leave reduction dtype gradual.
def sum[
    Shape: _Shape,
    DType,
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
](
    a: ndarray[Shape, DType], axis: Axis = None, *, keepdims: KeepDims = False
) -> ndarray[reduce_shape(Shape, Axis, KeepDims), Any]: ...
def mean[
    Shape: _Shape,
    DType,
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
](
    a: ndarray[Shape, DType], axis: Axis = None, *, keepdims: KeepDims = False
) -> ndarray[reduce_shape(Shape, Axis, KeepDims), Any]: ...
def min[
    Shape: _Shape,
    DType,
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
](
    a: ndarray[Shape, DType], axis: Axis = None, *, keepdims: KeepDims = False
) -> ndarray[reduce_shape(Shape, Axis, KeepDims), Any]: ...
def max[
    Shape: _Shape,
    DType,
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
](
    a: ndarray[Shape, DType], axis: Axis = None, *, keepdims: KeepDims = False
) -> ndarray[reduce_shape(Shape, Axis, KeepDims), Any]: ...
@overload
def argmin[N: IntVar, M: IntVar](
    a: ndarray[[N, M]],
    axis: Literal[0, -2],
    *,
    keepdims: Literal[False] = False,
) -> ndarray[[M], dtype[intp]]: ...
@overload
def argmin[N: IntVar, M: IntVar](
    a: ndarray[[N, M]],
    axis: Literal[1, -1],
    *,
    keepdims: Literal[False] = False,
) -> ndarray[[N], dtype[intp]]: ...

# TODO(stroxler): Replace these finite tuple-shape constructor overloads with a
# generic `Shape: tuple[int, ...]` overload once whole-shape parameters flow
# through downstream array operations without degrading to unknown.
@overload
def zeros[N: IntVar, ScalarT: generic](
    shape: Int[N], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def zeros[N: IntVar, ScalarT: generic](
    shape: IntTuple[N], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def zeros[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def zeros[N: IntVar, ScalarT: generic](
    shape: Int[N], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def zeros[N: IntVar, ScalarT: generic](
    shape: IntTuple[N], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def zeros[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def zeros[N: IntVar](
    shape: Int[N], dtype: None = ..., order: str = ...
) -> ndarray[[N], dtype[float64]]: ...
@overload
def zeros[N: IntVar](
    shape: IntTuple[N], dtype: None = ..., order: str = ...
) -> ndarray[[N], dtype[float64]]: ...
@overload
def zeros[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: None = ..., order: str = ...
) -> ndarray[[N, M], dtype[float64]]: ...
@overload
def zeros[N: IntVar](shape: Int[N], dtype: Any, order: str = ...) -> ndarray[[N]]: ...
@overload
def zeros[N: IntVar](
    shape: IntTuple[N], dtype: Any, order: str = ...
) -> ndarray[[N]]: ...
@overload
def zeros[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: Any, order: str = ...
) -> ndarray[[N, M]]: ...
@overload
def ones[N: IntVar, ScalarT: generic](
    shape: Int[N], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def ones[N: IntVar, ScalarT: generic](
    shape: IntTuple[N], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def ones[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def ones[N: IntVar, ScalarT: generic](
    shape: Int[N], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def ones[N: IntVar, ScalarT: generic](
    shape: IntTuple[N], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def ones[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def ones[N: IntVar](
    shape: Int[N], dtype: None = ..., order: str = ...
) -> ndarray[[N], dtype[float64]]: ...
@overload
def ones[N: IntVar](
    shape: IntTuple[N], dtype: None = ..., order: str = ...
) -> ndarray[[N], dtype[float64]]: ...
@overload
def ones[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: None = ..., order: str = ...
) -> ndarray[[N, M], dtype[float64]]: ...
@overload
def ones[N: IntVar](shape: Int[N], dtype: Any, order: str = ...) -> ndarray[[N]]: ...
@overload
def ones[N: IntVar](
    shape: IntTuple[N], dtype: Any, order: str = ...
) -> ndarray[[N]]: ...
@overload
def ones[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: Any, order: str = ...
) -> ndarray[[N, M]]: ...
@overload
def full[N: IntVar, ScalarT: generic](
    shape: Int[N],
    fill_value: Any,
    dtype: type[ScalarT],
    order: str = ...,
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def full[N: IntVar, ScalarT: generic](
    shape: IntTuple[N],
    fill_value: Any,
    dtype: type[ScalarT],
    order: str = ...,
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def full[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M],
    fill_value: Any,
    dtype: type[ScalarT],
    order: str = ...,
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def full[N: IntVar, ScalarT: generic](
    shape: Int[N],
    fill_value: Any,
    dtype: dtype[ScalarT],
    order: str = ...,
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def full[N: IntVar, ScalarT: generic](
    shape: IntTuple[N],
    fill_value: Any,
    dtype: dtype[ScalarT],
    order: str = ...,
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def full[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M],
    fill_value: Any,
    dtype: dtype[ScalarT],
    order: str = ...,
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def full[N: IntVar](
    shape: Int[N], fill_value: Any, dtype: Any = ..., order: str = ...
) -> ndarray[[N]]: ...
@overload
def full[N: IntVar](
    shape: IntTuple[N], fill_value: Any, dtype: Any = ..., order: str = ...
) -> ndarray[[N]]: ...
@overload
def full[N: IntVar, M: IntVar](
    shape: IntTuple[N, M],
    fill_value: Any,
    dtype: Any = ...,
    order: str = ...,
) -> ndarray[[N, M]]: ...
@overload
def empty[N: IntVar, ScalarT: generic](
    shape: Int[N], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def empty[N: IntVar, ScalarT: generic](
    shape: IntTuple[N], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def empty[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M], dtype: type[ScalarT], order: str = ...
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def empty[N: IntVar, ScalarT: generic](
    shape: Int[N], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def empty[N: IntVar, ScalarT: generic](
    shape: IntTuple[N], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N], dtype[ScalarT]]: ...
@overload
def empty[N: IntVar, M: IntVar, ScalarT: generic](
    shape: IntTuple[N, M], dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N, M], dtype[ScalarT]]: ...
@overload
def empty[N: IntVar](
    shape: Int[N], dtype: None = ..., order: str = ...
) -> ndarray[[N], dtype[float64]]: ...
@overload
def empty[N: IntVar](
    shape: IntTuple[N], dtype: None = ..., order: str = ...
) -> ndarray[[N], dtype[float64]]: ...
@overload
def empty[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: None = ..., order: str = ...
) -> ndarray[[N, M], dtype[float64]]: ...
@overload
def empty[N: IntVar](shape: Int[N], dtype: Any, order: str = ...) -> ndarray[[N]]: ...
@overload
def empty[N: IntVar](
    shape: IntTuple[N], dtype: Any, order: str = ...
) -> ndarray[[N]]: ...
@overload
def empty[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: Any, order: str = ...
) -> ndarray[[N, M]]: ...
@overload
def eye[N: IntVar, ScalarT: generic](
    N: Int[N], M: None = ..., k: int = ..., *, dtype: type[ScalarT], order: str = ...
) -> ndarray[[N, N], dtype[ScalarT]]: ...
@overload
def eye[N: IntVar, ScalarT: generic](
    N: Int[N], M: None, k: int, dtype: type[ScalarT], order: str = ...
) -> ndarray[[N, N], dtype[ScalarT]]: ...
@overload
def eye[N: IntVar, ScalarT: generic](
    N: Int[N],
    M: None = ...,
    k: int = ...,
    *,
    dtype: dtype[ScalarT],
    order: str = ...,
) -> ndarray[[N, N], dtype[ScalarT]]: ...
@overload
def eye[N: IntVar, ScalarT: generic](
    N: Int[N], M: None, k: int, dtype: dtype[ScalarT], order: str = ...
) -> ndarray[[N, N], dtype[ScalarT]]: ...
@overload
def eye[N: IntVar](
    N: Int[N], M: None = ..., k: int = ..., dtype: None = ..., order: str = ...
) -> ndarray[[N, N], dtype[float64]]: ...
@overload
def eye[N: IntVar](
    N: Int[N], M: None = ..., k: int = ..., dtype: Any = ..., order: str = ...
) -> ndarray[[N, N]]: ...
@overload
def identity[N: IntVar, ScalarT: generic](
    n: Int[N], dtype: type[ScalarT], *, like: Any = ...
) -> ndarray[[N, N], dtype[ScalarT]]: ...
@overload
def identity[N: IntVar, ScalarT: generic](
    n: Int[N], dtype: dtype[ScalarT], *, like: Any = ...
) -> ndarray[[N, N], dtype[ScalarT]]: ...
@overload
def identity[N: IntVar](
    n: Int[N], dtype: None = ..., *, like: Any = ...
) -> ndarray[[N, N], dtype[float64]]: ...
@overload
def identity[N: IntVar](
    n: Int[N], dtype: Any, *, like: Any = ...
) -> ndarray[[N, N]]: ...
