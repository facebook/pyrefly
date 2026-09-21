# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive type stubs for PyTorch with shape inference.

Shape inference is expressed through type-level functions in annotations. Library-specific
functions are defined in `torch/_shapes.pyi`.
"""

import builtins
from collections.abc import Iterator, Sequence
from math import e as e, nan as nan
from types import EllipsisType
from typing import Any, Callable, Literal, overload, Self, TYPE_CHECKING, Unpack

from shape_extensions import (
    broadcast,
    Elements,
    Flag,
    Index,
    index_shape,
    IntTuple,
    IntTuples,
    IntVar,
    MapIntTuples,
    RegularNestedList,
)
from torch import return_types

# `Generator` is not defined anywhere in this package, and resolving it relies
# on how a partial stub package is looked up. The `py.typed` file here contains
# the word `partial`, which tells the type checker this package covers only some
# of `torch`: for a submodule the package does not define, the checker falls
# back to the real torch stubs. `torch._C` is one of those, so `Generator` comes
# from torch itself.
#
# That fallback is per-module rather than per-name. Defining this module shadows
# torch's version of it outright, so every supported top-level name must be
# declared here.
from torch._C import (
    AcceleratorError as AcceleratorError,
    AggregationType as AggregationType,
    AliasDb as AliasDb,
    AnyType as AnyType,
    Argument as Argument,
    autocast_decrement_nesting as autocast_decrement_nesting,
    autocast_increment_nesting as autocast_increment_nesting,
    AwaitType as AwaitType,
    BenchmarkConfig as BenchmarkConfig,
    BenchmarkExecutionStats as BenchmarkExecutionStats,
    Block as Block,
    BoolType as BoolType,
    BufferDict as BufferDict,
    CallStack as CallStack,
    ClassType as ClassType,
    clear_autocast_cache as clear_autocast_cache,
    CompilationUnit as CompilationUnit,
    ComplexType as ComplexType,
    ConcreteModuleType as ConcreteModuleType,
    ConcreteModuleTypeBuilder as ConcreteModuleTypeBuilder,
    default_generator as default_generator,
    DeserializationStorageContext as DeserializationStorageContext,
    DeviceObjType as DeviceObjType,
    DictType as DictType,
    DisableTorchFunction as DisableTorchFunction,
    DisableTorchFunctionSubclass as DisableTorchFunctionSubclass,
    DispatchKey as DispatchKey,
    DispatchKeySet as DispatchKeySet,
    EnumType as EnumType,
    ErrorReport as ErrorReport,
    Event as Event,
    FileCheck as FileCheck,
    finfo as finfo,
    FloatType as FloatType,
    fork as fork,
    FunctionSchema as FunctionSchema,
    Future as Future,
    FutureType as FutureType,
    Generator as Generator,
    get_autocast_cpu_dtype as get_autocast_cpu_dtype,
    get_autocast_dtype as get_autocast_dtype,
    get_autocast_gpu_dtype as get_autocast_gpu_dtype,
    get_default_dtype as get_default_dtype,
    get_num_interop_threads as get_num_interop_threads,
    Graph as Graph,
    GraphExecutorState as GraphExecutorState,
    has_lapack as has_lapack,
    has_mkl as has_mkl,
    has_openmp as has_openmp,
    has_spectral as has_spectral,
    iinfo as iinfo,
    import_ir_module as import_ir_module,
    import_ir_module_from_buffer as import_ir_module_from_buffer,
    InferredType as InferredType,
    InterfaceType as InterfaceType,
    IntType as IntType,
    IODescriptor as IODescriptor,
    is_anomaly_check_nan_enabled as is_anomaly_check_nan_enabled,
    is_anomaly_enabled as is_anomaly_enabled,
    is_autocast_cache_enabled as is_autocast_cache_enabled,
    is_autocast_cpu_enabled as is_autocast_cpu_enabled,
    is_autocast_enabled as is_autocast_enabled,
    is_grad_enabled as is_grad_enabled,
    is_inference_mode_enabled as is_inference_mode_enabled,
    JITException as JITException,
    layout as layout,
    ListType as ListType,
    LiteScriptModule as LiteScriptModule,
    LockingLogger as LockingLogger,
    memory_format as memory_format,
    merge_type_from_type_comment as merge_type_from_type_comment,
    ModuleDict as ModuleDict,
    Node as Node,
    NoneType as NoneType,
    NoopLogger as NoopLogger,
    NumberType as NumberType,
    OptionalType as OptionalType,
    ParameterDict as ParameterDict,
    parse_ir as parse_ir,
    parse_schema as parse_schema,
    parse_type_comment as parse_type_comment,
    PyTorchFileReader as PyTorchFileReader,
    PyTorchFileWriter as PyTorchFileWriter,
    qscheme as qscheme,
    RRefType as RRefType,
    ScriptDict as ScriptDict,
    ScriptFunction as ScriptFunction,
    ScriptList as ScriptList,
    ScriptMethod as ScriptMethod,
    ScriptModule as ScriptModule,
    ScriptModuleSerializer as ScriptModuleSerializer,
    ScriptObject as ScriptObject,
    SerializationStorageContext as SerializationStorageContext,
    set_anomaly_enabled as set_anomaly_enabled,
    set_autocast_cache_enabled as set_autocast_cache_enabled,
    set_autocast_cpu_dtype as set_autocast_cpu_dtype,
    set_autocast_cpu_enabled as set_autocast_cpu_enabled,
    set_autocast_dtype as set_autocast_dtype,
    set_autocast_enabled as set_autocast_enabled,
    set_autocast_gpu_dtype as set_autocast_gpu_dtype,
    set_flush_denormal as set_flush_denormal,
    set_num_interop_threads as set_num_interop_threads,
    set_num_threads as set_num_threads,
    Size as Size,
    Stream as Stream,
    StreamObjType as StreamObjType,
    StringType as StringType,
    SymBoolType as SymBoolType,
    SymIntType as SymIntType,
    Tag as Tag,
    TensorBase as _TensorBase,
    TensorType as TensorType,
    ThroughputBenchmark as ThroughputBenchmark,
    TracingState as TracingState,
    TupleType as TupleType,
    Type as Type,
    unify_type_list as unify_type_list,
    UnionType as UnionType,
    Use as Use,
    Value as Value,
    wait as wait,
)
from torch._C._VariableFunctions import (
    abs_ as abs_,
    absolute as absolute,
    acos as acos,
    acos_ as acos_,
    acosh_ as acosh_,
    adaptive_avg_pool1d as adaptive_avg_pool1d,
    adaptive_max_pool1d as adaptive_max_pool1d,
    addbmm as addbmm,
    addcdiv as addcdiv,
    addcmul as addcmul,
    addmv as addmv,
    addmv_ as addmv_,
    addr as addr,
    adjoint as adjoint,
    affine_grid_generator as affine_grid_generator,
    alias_copy as alias_copy,
    alpha_dropout as alpha_dropout,
    alpha_dropout_ as alpha_dropout_,
    amax as amax,
    amin as amin,
    angle as angle,
    arccos as arccos,
    arccos_ as arccos_,
    arccosh as arccosh,
    arccosh_ as arccosh_,
    arcsin_ as arcsin_,
    arcsinh as arcsinh,
    arcsinh_ as arcsinh_,
    arctan as arctan,
    arctan2 as arctan2,
    arctan_ as arctan_,
    arctanh as arctanh,
    arctanh_ as arctanh_,
    argwhere as argwhere,
    as_strided as as_strided,
    as_strided_ as as_strided_,
    as_strided_copy as as_strided_copy,
    as_strided_scatter as as_strided_scatter,
    asarray as asarray,
    asin_ as asin_,
    asinh_ as asinh_,
    atan_ as atan_,
    atanh_ as atanh_,
    avg_pool1d as avg_pool1d,
    baddbmm as baddbmm,
    bartlett_window as bartlett_window,
    batch_norm as batch_norm,
    batch_norm_backward_elemt as batch_norm_backward_elemt,
    batch_norm_backward_reduce as batch_norm_backward_reduce,
    batch_norm_elemt as batch_norm_elemt,
    batch_norm_gather_stats as batch_norm_gather_stats,
    batch_norm_gather_stats_with_counts as batch_norm_gather_stats_with_counts,
    batch_norm_stats as batch_norm_stats,
    batch_norm_update_stats as batch_norm_update_stats,
    bilinear as bilinear,
    binary_cross_entropy_with_logits as binary_cross_entropy_with_logits,
    bincount as bincount,
    binomial as binomial,
    blackman_window as blackman_window,
    bucketize as bucketize,
    can_cast as can_cast,
    ccol_indices_copy as ccol_indices_copy,
    ceil_ as ceil_,
    celu as celu,
    celu_ as celu_,
    channel_shuffle as channel_shuffle,
    cholesky_inverse as cholesky_inverse,
    choose_qparams_optimized as choose_qparams_optimized,
    clamp_ as clamp_,
    clamp_max as clamp_max,
    clamp_max_ as clamp_max_,
    clamp_min as clamp_min,
    clamp_min_ as clamp_min_,
    clip_ as clip_,
    clone as clone,
    col_indices_copy as col_indices_copy,
    column_stack as column_stack,
    combinations as combinations,
    complex as complex,
    conj as conj,
    conj_physical as conj_physical,
    conj_physical_ as conj_physical_,
    constant_pad_nd as constant_pad_nd,
    conv1d as conv1d,
    conv2d as conv2d,
    conv3d as conv3d,
    conv_tbc as conv_tbc,
    conv_transpose1d as conv_transpose1d,
    conv_transpose2d as conv_transpose2d,
    conv_transpose3d as conv_transpose3d,
    convolution as convolution,
    corrcoef as corrcoef,
    cos_ as cos_,
    cosh as cosh,
    cosh_ as cosh_,
    cosine_embedding_loss as cosine_embedding_loss,
    cosine_similarity as cosine_similarity,
    cov as cov,
    crow_indices_copy as crow_indices_copy,
    ctc_loss as ctc_loss,
    cudnn_affine_grid_generator as cudnn_affine_grid_generator,
    cudnn_batch_norm as cudnn_batch_norm,
    cudnn_convolution as cudnn_convolution,
    cudnn_convolution_add_relu as cudnn_convolution_add_relu,
    cudnn_convolution_relu as cudnn_convolution_relu,
    cudnn_convolution_transpose as cudnn_convolution_transpose,
    cudnn_grid_sampler as cudnn_grid_sampler,
    cudnn_is_acceptable as cudnn_is_acceptable,
    cumulative_trapezoid as cumulative_trapezoid,
    deg2rad_ as deg2rad_,
    dequantize as dequantize,
    detach as detach,
    detach_ as detach_,
    detach_copy as detach_copy,
    diag as diag,
    diagflat as diagflat,
    diagonal_copy as diagonal_copy,
    diagonal_scatter as diagonal_scatter,
    diff as diff,
    divide as divide,
    dropout as dropout,
    dropout_ as dropout_,
    dsmm as dsmm,
    dsplit as dsplit,
    dstack as dstack,
    embedding as embedding,
    embedding_bag as embedding_bag,
    embedding_renorm_ as embedding_renorm_,
    empty_permuted as empty_permuted,
    empty_quantized as empty_quantized,
    empty_strided as empty_strided,
    erf_ as erf_,
    erfc_ as erfc_,
    exp2 as exp2,
    exp2_ as exp2_,
    exp_ as exp_,
    expand_copy as expand_copy,
    expm1_ as expm1_,
    fake_quantize_per_channel_affine as fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine as fake_quantize_per_tensor_affine,
    fbgemm_linear_fp16_weight as fbgemm_linear_fp16_weight,
    fbgemm_linear_fp16_weight_fp32_activation as fbgemm_linear_fp16_weight_fp32_activation,
    fbgemm_linear_int8_weight as fbgemm_linear_int8_weight,
    fbgemm_linear_int8_weight_fp32_activation as fbgemm_linear_int8_weight_fp32_activation,
    fbgemm_linear_quantize_weight as fbgemm_linear_quantize_weight,
    fbgemm_pack_gemm_matrix_fp16 as fbgemm_pack_gemm_matrix_fp16,
    fbgemm_pack_quantized_matrix as fbgemm_pack_quantized_matrix,
    feature_alpha_dropout as feature_alpha_dropout,
    feature_alpha_dropout_ as feature_alpha_dropout_,
    feature_dropout as feature_dropout,
    feature_dropout_ as feature_dropout_,
    fill as fill,
    fill_ as fill_,
    fix as fix,
    fix_ as fix_,
    fliplr as fliplr,
    flipud as flipud,
    float_power as float_power,
    floor_ as floor_,
    floor_divide as floor_divide,
    frac as frac,
    frac_ as frac_,
    frexp as frexp,
    frobenius_norm as frobenius_norm,
    from_file as from_file,
    frombuffer as frombuffer,
    fused_moving_avg_obs_fake_quant as fused_moving_avg_obs_fake_quant,
    gcd as gcd,
    gcd_ as gcd_,
    geqrf as geqrf,
    ger as ger,
    gradient as gradient,
    greater as greater,
    greater_equal as greater_equal,
    grid_sampler as grid_sampler,
    grid_sampler_2d as grid_sampler_2d,
    grid_sampler_3d as grid_sampler_3d,
    group_norm as group_norm,
    gru as gru,
    gru_cell as gru_cell,
    hamming_window as hamming_window,
    hardshrink as hardshrink,
    hash_tensor as hash_tensor,
    heaviside as heaviside,
    hinge_embedding_loss as hinge_embedding_loss,
    histc as histc,
    histogram as histogram,
    histogramdd as histogramdd,
    hsmm as hsmm,
    hsplit as hsplit,
    hspmm as hspmm,
    hstack as hstack,
    i0 as i0,
    i0_ as i0_,
    igamma as igamma,
    igammac as igammac,
    imag as imag,
    index_put_ as index_put_,
    index_reduce as index_reduce,
    indices_copy as indices_copy,
    inner as inner,
    instance_norm as instance_norm,
    int_repr as int_repr,
    is_complex as is_complex,
    is_conj as is_conj,
    is_distributed as is_distributed,
    is_floating_point as is_floating_point,
    is_inference as is_inference,
    is_neg as is_neg,
    is_nonzero as is_nonzero,
    is_same_size as is_same_size,
    is_signed as is_signed,
    is_vulkan_available as is_vulkan_available,
    isin as isin,
    isinf as isinf,
    istft as istft,
    kaiser_window as kaiser_window,
    kl_div as kl_div,
    kron as kron,
    layer_norm as layer_norm,
    lcm as lcm,
    lcm_ as lcm_,
    ldexp as ldexp,
    ldexp_ as ldexp_,
    less as less,
    less_equal as less_equal,
    log1p as log1p,
    log1p_ as log1p_,
    log2 as log2,
    log2_ as log2_,
    log10_ as log10_,
    log_ as log_,
    log_softmax as log_softmax,
    logaddexp as logaddexp,
    logaddexp2 as logaddexp2,
    logcumsumexp as logcumsumexp,
    logical_xor as logical_xor,
    logit as logit,
    logit_ as logit_,
    logspace as logspace,
    lstm as lstm,
    lstm_cell as lstm_cell,
    lu_unpack as lu_unpack,
    margin_ranking_loss as margin_ranking_loss,
    max_pool1d as max_pool1d,
    max_pool1d_with_indices as max_pool1d_with_indices,
    max_pool2d as max_pool2d,
    max_pool3d as max_pool3d,
    miopen_batch_norm as miopen_batch_norm,
    miopen_convolution as miopen_convolution,
    miopen_convolution_add_relu as miopen_convolution_add_relu,
    miopen_convolution_relu as miopen_convolution_relu,
    miopen_convolution_transpose as miopen_convolution_transpose,
    miopen_ctc_loss as miopen_ctc_loss,
    miopen_depthwise_convolution as miopen_depthwise_convolution,
    miopen_rnn as miopen_rnn,
    mkldnn_adaptive_avg_pool2d as mkldnn_adaptive_avg_pool2d,
    mkldnn_convolution as mkldnn_convolution,
    mkldnn_linear_backward_weights as mkldnn_linear_backward_weights,
    mkldnn_max_pool2d as mkldnn_max_pool2d,
    mkldnn_max_pool3d as mkldnn_max_pool3d,
    mkldnn_rnn_layer as mkldnn_rnn_layer,
    msort as msort,
    multiply as multiply,
    mvlgamma as mvlgamma,
    nan_to_num as nan_to_num,
    nan_to_num_ as nan_to_num_,
    nanmean as nanmean,
    nanmedian as nanmedian,
    nanquantile as nanquantile,
    nansum as nansum,
    narrow_copy as narrow_copy,
    native_batch_norm as native_batch_norm,
    native_channel_shuffle as native_channel_shuffle,
    native_dropout as native_dropout,
    native_group_norm as native_group_norm,
    native_layer_norm as native_layer_norm,
    native_norm as native_norm,
    neg_ as neg_,
    negative as negative,
    negative_ as negative_,
    nonzero as nonzero,
    nonzero_static as nonzero_static,
    norm_except_dim as norm_except_dim,
    not_equal as not_equal,
    nuclear_norm as nuclear_norm,
    orgqr as orgqr,
    ormqr as ormqr,
    pairwise_distance as pairwise_distance,
    pdist as pdist,
    permute_copy as permute_copy,
    pinverse as pinverse,
    pixel_shuffle as pixel_shuffle,
    pixel_unshuffle as pixel_unshuffle,
    poisson_nll_loss as poisson_nll_loss,
    positive as positive,
    prelu as prelu,
    promote_types as promote_types,
    q_per_channel_axis as q_per_channel_axis,
    q_per_channel_scales as q_per_channel_scales,
    q_per_channel_zero_points as q_per_channel_zero_points,
    q_scale as q_scale,
    q_zero_point as q_zero_point,
    qr as qr,
    quantize_per_channel as quantize_per_channel,
    quantize_per_tensor as quantize_per_tensor,
    quantize_per_tensor_dynamic as quantize_per_tensor_dynamic,
    quantized_batch_norm as quantized_batch_norm,
    quantized_gru_cell as quantized_gru_cell,
    quantized_lstm_cell as quantized_lstm_cell,
    quantized_max_pool1d as quantized_max_pool1d,
    quantized_max_pool2d as quantized_max_pool2d,
    quantized_max_pool3d as quantized_max_pool3d,
    quantized_rnn_relu_cell as quantized_rnn_relu_cell,
    quantized_rnn_tanh_cell as quantized_rnn_tanh_cell,
    rad2deg_ as rad2deg_,
    randint_like as randint_like,
    range as range,
    ravel as ravel,
    real as real,
    reciprocal as reciprocal,
    reciprocal_ as reciprocal_,
    relu_ as relu_,
    renorm as renorm,
    resize_as_ as resize_as_,
    resize_as_sparse_ as resize_as_sparse_,
    resolve_conj as resolve_conj,
    resolve_neg as resolve_neg,
    result_type as result_type,
    rms_norm as rms_norm,
    rnn_relu as rnn_relu,
    rnn_relu_cell as rnn_relu_cell,
    rnn_tanh as rnn_tanh,
    rnn_tanh_cell as rnn_tanh_cell,
    roll as roll,
    rot90 as rot90,
    round_ as round_,
    row_indices_copy as row_indices_copy,
    row_stack as row_stack,
    rrelu as rrelu,
    rrelu_ as rrelu_,
    rsqrt_ as rsqrt_,
    rsub as rsub,
    saddmm as saddmm,
    scalar_tensor as scalar_tensor,
    scatter_add as scatter_add,
    scatter_reduce as scatter_reduce,
    searchsorted as searchsorted,
    select_copy as select_copy,
    select_scatter as select_scatter,
    selu as selu,
    selu_ as selu_,
    sgn as sgn,
    sigmoid_ as sigmoid_,
    signbit as signbit,
    sin_ as sin_,
    sinc as sinc,
    sinc_ as sinc_,
    sinh as sinh,
    sinh_ as sinh_,
    slice_copy as slice_copy,
    slice_inverse as slice_inverse,
    slice_scatter as slice_scatter,
    smm as smm,
    softmax as softmax,
    sparse_bsc_tensor as sparse_bsc_tensor,
    sparse_bsr_tensor as sparse_bsr_tensor,
    sparse_compressed_tensor as sparse_compressed_tensor,
    sparse_coo_tensor as sparse_coo_tensor,
    sparse_csc_tensor as sparse_csc_tensor,
    sparse_csr_tensor as sparse_csr_tensor,
    split_copy as split_copy,
    split_with_sizes as split_with_sizes,
    split_with_sizes_copy as split_with_sizes_copy,
    spmm as spmm,
    sqrt_ as sqrt_,
    square as square,
    square_ as square_,
    squeeze_copy as squeeze_copy,
    sspaddmm as sspaddmm,
    subtract as subtract,
    svd as svd,
    swapaxes as swapaxes,
    swapdims as swapdims,
    sym_constrain_range as sym_constrain_range,
    sym_constrain_range_for_size as sym_constrain_range_for_size,
    t as t,
    t_copy as t_copy,
    tan_ as tan_,
    tanh_ as tanh_,
    tensor_split as tensor_split,
    threshold as threshold,
    threshold_ as threshold_,
    transpose_copy as transpose_copy,
    trapezoid as trapezoid,
    trapz as trapz,
    triplet_margin_loss as triplet_margin_loss,
    true_divide as true_divide,
    trunc as trunc,
    trunc_ as trunc_,
    unbind_copy as unbind_copy,
    unflatten as unflatten,
    unfold_copy as unfold_copy,
    unsafe_chunk as unsafe_chunk,
    unsafe_split as unsafe_split,
    unsafe_split_with_sizes as unsafe_split_with_sizes,
    unsqueeze_copy as unsqueeze_copy,
    values_copy as values_copy,
    vander as vander,
    vdot as vdot,
    view_as_complex_copy as view_as_complex_copy,
    view_as_real_copy as view_as_real_copy,
    view_copy as view_copy,
    vsplit as vsplit,
    vstack as vstack,
    xlogy as xlogy,
    xlogy_ as xlogy_,
    zero_ as zero_,
)
from torch._higher_order_ops import cond as cond
from torch._lobpcg import lobpcg as lobpcg
from torch._shapes import (
    arange_extent,
    arange_step_extent,
    cat_shape,
    chunk_shapes,
    diag_embed_shape,
    diagonal_shape,
    dim_shape,
    eig_shape,
    einsum_shape,
    expand_shape,
    flatten_shape,
    gather_shape,
    index_fill_shape,
    index_select_shape,
    indexed_source_shape,
    matmul_shape,
    movedim_scalar_shape,
    movedim_tuple_shape,
    multinomial_shape,
    narrow_shape,
    nonnegative_extent,
    numel_shape,
    permute_shape,
    put_shape,
    reduce_shape,
    reduce_shape_no_keep,
    repeat_interleave_checked_shape,
    repeat_interleave_output_shape,
    repeat_interleave_shape,
    repeat_shape,
    replace_axis_extent,
    reshape_shape,
    scatter_shape,
    select_shape,
    size_dim_shape,
    slogdet_shape,
    split_sections_shapes,
    split_size_shapes,
    squeeze_shape,
    stack_shape,
    take_along_dim_shape,
    take_shape,
    tensordot_shape,
    tile_shape,
    topk_shape,
    transpose_shape,
    unbind_shape,
    unfold_shape,
    unsqueeze_shape,
)
from torch._tensor_str import set_printoptions as set_printoptions
from torch.amp import autocast as autocast, GradScaler as GradScaler
from torch.autograd import enable_grad as enable_grad
from torch.func import vmap as vmap
from torch.functional import (
    align_tensors as align_tensors,
    atleast_1d as atleast_1d,
    atleast_2d as atleast_2d,
    atleast_3d as atleast_3d,
    block_diag as block_diag,
    broadcast_tensors as broadcast_tensors,
    cartesian_prod as cartesian_prod,
    cdist as cdist,
    chain_matmul as chain_matmul,
    unique_consecutive as unique_consecutive,
    unravel_index as unravel_index,
)
from torch.random import (
    get_rng_state as get_rng_state,
    initial_seed as initial_seed,
    seed as seed,
    set_rng_state as set_rng_state,
    thread_safe_generator as thread_safe_generator,
)
from torch.storage import TypedStorage as TypedStorage, UntypedStorage as UntypedStorage

if TYPE_CHECKING:
    from shape_extensions import Int as _Int

__all__ = ["Tensor"]

type _Shape = IntTuple
type _Scalar = builtins.bool | builtins.int | builtins.float | builtins.complex
type _TensorLike[Shape: _Shape] = Tensor[Shape] | _Scalar
type _RealScalar = builtins.bool | builtins.int | builtins.float
type _RealTensorLike[Shape: _Shape] = Tensor[Shape] | _RealScalar
type _IntegerScalar = builtins.bool | builtins.int
type _IntegerTensorLike[Shape: _Shape] = Tensor[Shape] | _IntegerScalar
type _BasicIndex = builtins.int | slice | list[builtins.int] | None | EllipsisType
type _TensorScalar = builtins.bool | builtins.int | builtins.float | builtins.complex
type _LegacyTensorScalar = builtins.bool | builtins.int | builtins.float

# ============================================================================
# Device Type
# ============================================================================

class device:
    """Represents the device on which a Tensor is or will be allocated."""

    type: str
    index: builtins.int | None
    def __init__(self, type: str | device, index: int = 0) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

# Dtype constants
qint8: Any
quint8: Any
float16: Any
float32: Any
float64: Any
int8: Any
int16: Any
int32: Any
int64: Any
int: Any
bool: Any
ops: Any

# ============================================================================
# Tensor Class
# ============================================================================

class Tensor[Shape: _Shape = _Shape](_TensorBase):
    """
    PyTorch Tensor with shape type parameter.

    The shape is tracked at the type level, allowing static verification
    of tensor operations.

    Most shape transformations are handled by meta-shape functions registered
    in the type checker, not by explicit type signatures here.
    """

    @overload
    def __new__(cls, *, device: Any = None) -> Tensor[[0]]: ...
    # Unsupported scalar forms stay gradual; `Never` would incorrectly make the caller's
    # remaining control flow unreachable.
    @overload
    def __new__(
        cls, data: builtins.bool, *, device: Any = None
    ) -> Tensor[IntTuple]: ...
    @overload
    def __new__[Size: IntTuple](
        cls, *size: *Size, device: Any = None
    ) -> Tensor[Size]: ...
    @overload
    def __new__(cls, *, data: builtins.int, device: Any = None) -> Tensor[IntTuple]: ...
    @overload
    def __new__(
        cls,
        data: builtins.float | builtins.complex,
        *,
        device: Any = None,
    ) -> Tensor[IntTuple]: ...
    @overload
    def __new__[DataShape: IntTuple](
        cls, data: Tensor[DataShape], *, device: Any = None
    ) -> Tensor[DataShape]: ...
    @overload
    def __new__[DataShape: IntTuple](
        cls,
        data: RegularNestedList[DataShape, _LegacyTensorScalar],
        *,
        device: Any = None,
    ) -> Tensor[DataShape]: ...
    @overload
    def __new__(cls, data: Any, *, device: Any = None) -> Tensor[IntTuple]: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    # ==== Tensor Properties ====
    shape: Shape  # Tensor shape as a tuple
    requires_grad: builtins.bool  # Whether gradient tracking is enabled
    grad: Any  # Gradient storage is intentionally gradual in this partial overlay.
    device: Any  # Device where tensor is stored (cpu, cuda, etc.)
    dtype: Any  # Data type of tensor elements (float32, int64, etc.)
    ndim: builtins.int  # Number of dimensions
    T: Self  # Transpose property (for 2D tensors). Use .t() method for shape inference.
    real: Self  # Real part of complex tensor (shape-preserving)
    imag: Self  # Imaginary part of complex tensor (shape-preserving)

    # The overlay intentionally models only shape-relevant members. Keep the
    # rest of Tensor's large API gradual until it receives a precise signature.
    def __getattr__(self, name: str) -> Any: ...
    def __iter__(self) -> Iterator[Tensor]: ...
    def __array__(self, dtype: Any = None) -> Any: ...
    def zero_(self) -> Self: ...
    def add_(self, other: Tensor | builtins.int | builtins.float) -> Self: ...
    def pin_memory(self, device: Any = None) -> Self: ...
    def byte(self, memory_format: Any = None) -> Self: ...
    def reshape_as[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[OtherShape]: ...
    def unique(self, *args: Any, **kwargs: Any) -> Any: ...
    def diagonal[
        Offset: Flag[builtins.int] = 0,
        Dim1: Flag[builtins.int] = 0,
        Dim2: Flag[builtins.int] = 1,
    ](
        self: Tensor[Shape], offset: Offset = 0, dim1: Dim1 = 0, dim2: Dim2 = 1
    ) -> Tensor[diagonal_shape(Shape, Offset, Dim1, Dim2)]: ...
    def data_ptr(self) -> builtins.int: ...
    def is_contiguous(self, memory_format: Any = None) -> builtins.bool: ...

    # Note: Use .dim() method for rank (ndim removed in favor of dim())
    # ==== Indexing ====
    @overload
    def __getitem__[I: Index](
        self: Tensor[Shape], index: I
    ) -> Tensor[index_shape(Shape, I)]: ...
    @overload
    def __getitem__(
        self: Tensor,
        index: _BasicIndex
        | Sequence[builtins.int]
        | Sequence[Sequence[builtins.int]]
        | Tensor
        | tuple[
            _BasicIndex
            | Sequence[builtins.int]
            | Sequence[Sequence[builtins.int]]
            | Tensor,
            ...,
        ],
    ) -> Tensor[IntTuple]: ...
    def __setitem__(
        self: Tensor,
        index: _BasicIndex
        | Sequence[builtins.int]
        | Sequence[Sequence[builtins.int]]
        | Tensor
        | tuple[
            _BasicIndex
            | Sequence[builtins.int]
            | Sequence[Sequence[builtins.int]]
            | Tensor,
            ...,
        ],
        value: Tensor | int | float,
    ) -> None:
        """Set values in tensor via indexing. Mutates tensor in-place."""
        ...

    # ==== Matrix Multiplication ====
    # Uses meta-shape for shape inference

    def __matmul__[Left: IntTuple, Right: IntTuple](
        self: Tensor[Left], other: Tensor[Right]
    ) -> Tensor[matmul_shape(Left, Right)]:
        """Matrix multiplication (@). Shape inference via meta-shape: torch.Tensor.matmul"""
        ...

    # ==== Arithmetic Operations ====

    def __add__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    def __sub__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    def __mul__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    def __mod__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    def __truediv__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    def __floordiv__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...

    # Reverse operations for scalars
    def __radd__(self, other: float | int) -> Self: ...
    def __rsub__(self, other: float | int) -> Self: ...
    def __rmul__(self, other: float | int) -> Self: ...
    def __rtruediv__(self, other: float | int) -> Self: ...
    def __rpow__(self, other: float | int) -> Self: ...

    # Power operations
    def __pow__[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...

    # Unary operations
    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __int__(self) -> builtins.int: ...
    def __index__(self) -> builtins.int: ...
    def __len__(self) -> builtins.int: ...

    # ==== Comparison Operations ====

    @overload
    def __eq__[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...  # type: ignore[override]
    @overload
    def __eq__(self, other: float | int) -> Self: ...  # type: ignore[override]
    @overload
    def __ne__[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...  # type: ignore[override]
    @overload
    def __ne__(self, other: float | int) -> Self: ...  # type: ignore[override]
    # Ordering results are gradual because they are tensors elementwise, but
    # scalar tensors are also valid in truth-valued comparison protocols.
    def __lt__(self, other: Tensor | float | int) -> Any: ...
    def __le__(self, other: Tensor | float | int) -> Any: ...
    def __gt__(self, other: Tensor | float | int) -> Any: ...
    def __ge__(self, other: Tensor | float | int) -> Any: ...

    # ==== Bitwise Operations ====
    # Elementwise on integer and boolean tensors, broadcasting exactly as the
    # comparison operators do. Combining masks with `&` and `|` is the common
    # case, and those masks come from the comparisons above.

    @overload
    def __and__[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    @overload
    def __and__(self, other: bool | int) -> Self: ...
    @overload
    def __or__[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    @overload
    def __or__(self, other: bool | int) -> Self: ...
    @overload
    def __xor__[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
    @overload
    def __xor__(self, other: bool | int) -> Self: ...
    def __rand__(self, other: bool | int) -> Self: ...
    def __ror__(self, other: bool | int) -> Self: ...
    def __rxor__(self, other: bool | int) -> Self: ...
    def __invert__(self) -> Self: ...

    # ==== Shape Manipulation Operations ====
    # Handled by meta-shape functions - simplified signatures

    @overload
    def reshape[Shape: IntTuple, NewShape: IntTuple](
        self: Tensor[Shape], *shape: *NewShape
    ) -> Tensor[reshape_shape(Shape, NewShape)]:
        """Reshape tensor. Shape inference via type-level DSL."""
        ...

    @overload
    def reshape[Shape: IntTuple, NewShape: IntTuple](
        self: Tensor[Shape], shape: NewShape
    ) -> Tensor[reshape_shape(Shape, NewShape)]:
        """Reshape tensor. Shape inference via type-level DSL."""
        ...

    @overload
    def reshape(self, shape: Sequence[builtins.int]) -> Tensor: ...
    @overload
    def view[Shape: IntTuple, NewShape: IntTuple](
        self: Tensor[Shape], *shape: *NewShape
    ) -> Tensor[reshape_shape(Shape, NewShape)]:
        """View tensor. Shape inference via type-level DSL."""
        ...

    @overload
    def view[Shape: IntTuple, NewShape: IntTuple](
        self: Tensor[Shape], shape: NewShape
    ) -> Tensor[reshape_shape(Shape, NewShape)]:
        """View tensor. Shape inference via type-level DSL."""
        ...

    @overload
    def view(self, shape: Sequence[builtins.int]) -> Tensor: ...
    def flatten[
        Shape: IntTuple,
        StartDim: Flag[builtins.int],
        EndDim: Flag[builtins.int],
    ](
        self: Tensor[Shape], start_dim: StartDim = 0, end_dim: EndDim = -1
    ) -> Tensor[flatten_shape(Shape, StartDim, EndDim)]:
        """Flatten dimensions. Shape inference via the type-level DSL."""
        ...

    def transpose[
        Shape: IntTuple,
        Dim0: Flag[builtins.int],
        Dim1: Flag[builtins.int],
    ](
        self: Tensor[Shape], dim0: Dim0, dim1: Dim1
    ) -> Tensor[transpose_shape(Shape, Dim0, Dim1)]:
        """Transpose two dimensions. Shape inference via meta-shape: torch.transpose"""
        ...

    @overload
    def permute[Shape: IntTuple, Dims: Flag[tuple[builtins.int, ...]]](
        self: Tensor[Shape], *dims: *Dims
    ) -> Tensor[permute_shape(Shape, Dims)]:
        """Permute dimensions. Shape inference via type-level DSL."""
        ...

    @overload
    def permute[Shape: IntTuple, Dims: Flag[tuple[builtins.int, ...]]](
        self: Tensor[Shape], dims: Dims
    ) -> Tensor[permute_shape(Shape, Dims)]:
        """Permute dimensions. Shape inference via type-level DSL."""
        ...

    @overload
    def permute(self, *dims: builtins.int) -> Tensor: ...
    @overload
    def permute(self, dims: tuple[builtins.int, ...]) -> Tensor: ...
    def squeeze[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    ](self: Tensor[Shape], dim: Dim = None) -> Tensor[squeeze_shape(Shape, Dim)]:
        """Remove dimensions of size 1. Shape inference via meta-shape: torch.squeeze"""
        ...

    def unsqueeze[Shape: IntTuple, Dim: Flag[builtins.int]](
        self: Tensor[Shape], dim: Dim
    ) -> Tensor[unsqueeze_shape(Shape, Dim)]:
        """Add dimension of size 1. Shape inference via meta-shape: torch.unsqueeze"""
        ...

    @overload
    def repeat[Shape: IntTuple, Sizes: IntTuple](
        self: Tensor[Shape], *sizes: *Sizes
    ) -> Tensor[repeat_shape(Shape, Sizes)]:
        """Repeat tensor. Shape inference via type-level DSL."""
        ...

    @overload
    def repeat[Shape: IntTuple, Sizes: IntTuple](
        self: Tensor[Shape], sizes: Sizes
    ) -> Tensor[repeat_shape(Shape, Sizes)]:
        """Repeat tensor. Shape inference via type-level DSL."""
        ...

    def t[M: IntVar, N: IntVar](self: Tensor[[M, N]]) -> Tensor[[N, M]]:
        """Transpose 2D tensor. Swaps dimensions."""
        ...

    @overload
    def expand[Shape: IntTuple, Sizes: IntTuple](
        self: Tensor[Shape], *sizes: *Sizes
    ) -> Tensor[expand_shape(Shape, Sizes)]:
        """Expand tensor. Shape inference via type-level DSL."""
        ...

    @overload
    def expand[Shape: IntTuple, Sizes: IntTuple](
        self: Tensor[Shape], sizes: Sizes
    ) -> Tensor[expand_shape(Shape, Sizes)]:
        """Expand tensor. Shape inference via type-level DSL."""
        ...

    def expand_as[S: IntTuple](self: Tensor, other: Tensor[S]) -> Tensor[S]:
        """Expand tensor to match the shape of `other`."""
        ...

    @overload
    def repeat_interleave[
        Shape: IntTuple,
        Repeats: _Int,
        OutputSize: _Int,
        Dim: Flag[builtins.int | None],
    ](
        self: Tensor[Shape],
        repeats: Repeats,
        dim: Dim = None,
        *,
        output_size: OutputSize,
    ) -> Tensor[repeat_interleave_checked_shape(Shape, Repeats, OutputSize, Dim)]: ...
    @overload
    def repeat_interleave[
        Shape: IntTuple,
        OutputSize: _Int,
        Dim: Flag[builtins.int | None],
    ](
        self: Tensor[Shape],
        repeats: Tensor,
        dim: Dim = None,
        *,
        output_size: OutputSize,
    ) -> Tensor[repeat_interleave_output_shape(Shape, OutputSize, Dim)]: ...
    @overload
    def repeat_interleave[
        Shape: IntTuple,
        Repeats: _Int,
        Dim: Flag[builtins.int | None],
    ](
        self: Tensor[Shape],
        repeats: Repeats,
        dim: Dim = None,
        *,
        output_size: None = None,
    ) -> Tensor[repeat_interleave_shape(Shape, Repeats, Dim)]: ...
    @overload
    def repeat_interleave(
        self: Tensor,
        repeats: builtins.int | Tensor,
        dim: builtins.int | None = None,
        *,
        output_size: builtins.int | None = None,
    ) -> Tensor:
        """Repeat elements along a dimension."""
        ...

    def contiguous(self) -> Self:
        """Returns a contiguous tensor. Shape inference via generic fixture signature."""
        ...

    def clone(self) -> Self:
        """Returns a copy. Shape inference via generic fixture signature."""
        ...

    def detach(self) -> Self:
        """Returns detached tensor. Shape inference via generic fixture signature."""
        ...

    # ==== Tensor Creation Methods ====
    # These create new tensors; shape depends on size args, not self's shape.

    @overload
    def new_zeros(
        self,
        size: tuple[builtins.int, ...],
        *,
        dtype: Any = None,
        layout: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
        pin_memory: builtins.bool = False,
    ) -> Tensor:
        """Create zero-filled tensor with same dtype/device."""
        ...

    @overload
    def new_zeros(
        self,
        size: builtins.int,
        *sizes: builtins.int,
        dtype: Any = None,
        layout: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
        pin_memory: builtins.bool = False,
    ) -> Tensor:
        """Create zero-filled tensor with same dtype/device."""
        ...

    @overload
    def new_ones(
        self,
        size: tuple[builtins.int, ...],
        *,
        dtype: Any = None,
        layout: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
        pin_memory: builtins.bool = False,
    ) -> Tensor: ...
    @overload
    def new_ones(
        self,
        size: builtins.int,
        *sizes: builtins.int,
        dtype: Any = None,
        layout: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
        pin_memory: builtins.bool = False,
    ) -> Tensor:
        """Create one-filled tensor with same dtype/device."""
        ...

    def new_tensor(
        self,
        data: Any,
        dtype: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
    ) -> Tensor:
        """Create tensor from data with same dtype/device context."""
        ...

    def new_empty(
        self,
        *size: builtins.int,
        dtype: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
    ) -> Tensor:
        """Create uninitialized tensor with same dtype/device."""
        ...

    def new_full(
        self,
        size: tuple[builtins.int, ...],
        fill_value: builtins.float | builtins.int,
        dtype: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
    ) -> Tensor:
        """Create tensor filled with fill_value, same dtype/device."""
        ...

    # ==== Dtype Conversion Methods ====
    # Note: These method names shadow Python builtins, so type annotations
    # after this point should use builtins.int, builtins.bool, builtins.float

    def float(self) -> Self:
        """Convert tensor to float32 dtype. Shape-preserving operation."""
        ...

    def half(self) -> Self:
        """Convert tensor to float16 dtype. Shape-preserving operation."""
        ...

    def double(self) -> Self:
        """Convert tensor to float64 dtype. Shape-preserving operation."""
        ...

    def int(self) -> Self:
        """Convert tensor to int32 dtype. Shape-preserving operation."""
        ...

    def long(self) -> Self:
        """Convert tensor to int64 dtype. Shape-preserving operation."""
        ...

    def bool(self) -> Self:
        """Convert tensor to bool dtype. Shape-preserving operation."""
        ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Convert tensor dtype/device. Shape-preserving operation."""
        ...

    def type_as(self, other: Tensor) -> Self:
        """Convert tensor to same dtype as other tensor. Shape-preserving operation."""
        ...

    def cuda(self, device: Any = None) -> Self:
        """Move tensor to CUDA device. Shape-preserving operation."""
        ...

    def cpu(self) -> Self:
        """Move tensor to CPU. Shape-preserving operation."""
        ...

    data: Self  # Raw data tensor (same shape)

    def copy_(self, src: Tensor, non_blocking: builtins.bool = False) -> Self:
        """Copy elements from src into self in-place. Shape-preserving."""
        ...

    def fill_(self, value: Any) -> Self:
        """Fill tensor in-place. Shape-preserving."""
        ...

    def backward(
        self, gradient: Tensor | None = None, retain_graph: builtins.bool | None = None
    ) -> None:
        """Compute gradient of current tensor w.r.t. graph leaves."""
        ...

    def requires_grad_(self, requires_grad: builtins.bool = True) -> Self:
        """Enable/disable gradient tracking in-place. Shape-preserving."""
        ...

    def item(self: Tensor[[]]) -> builtins.float | builtins.int:
        """Return a Python scalar from a rank-zero tensor."""
        ...

    def tolist(self: Tensor) -> Any:
        """Returns tensor as a nested Python list."""
        ...

    def numpy(self: Tensor) -> Any:
        """Returns tensor as a NumPy array."""
        ...

    # TODO: Restrict this to statically single-element tensors once the type
    # system can express that predicate without rejecting gradual shapes.
    def __float__(self: Tensor) -> builtins.float:
        """Return the value of a single-element tensor as a Python float."""
        ...

    def tile[Shape: IntTuple, Repeats: IntTuple](
        self: Tensor[Shape], dims: Repeats
    ) -> Tensor[tile_shape(Shape, Repeats)]:
        """Tile tensor. Shape inference via type-level DSL."""
        ...

    def select[Shape: IntTuple, Dim: Flag[builtins.int], Index: _Int](
        self: Tensor[Shape], dim: Dim, index: Index
    ) -> Tensor[select_shape(Shape, Dim, Index)]:
        """Select along dimension. Shape inference via meta-shape: torch.Tensor.select"""
        ...

    def narrow[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        Start: _Int,
        Length: _Int,
    ](
        self: Tensor[Shape], dim: Dim, start: Start, length: Length
    ) -> Tensor[narrow_shape(Shape, Dim, Start, Length)]:
        """Narrow tensor along dimension. Shape inference via meta-shape: torch.Tensor.narrow"""
        ...

    @overload
    def split[
        Shape: IntTuple,
        SplitSize: _Int,
        Dim: Flag[builtins.int],
    ](
        self: Tensor[Shape], split_size_or_sections: SplitSize, dim: Dim = 0
    ) -> MapIntTuples[lambda S: Tensor[S], split_size_shapes(Shape, SplitSize, Dim)]:
        """Split tensor into chunks. Shape inference via the type-level DSL."""
        ...

    @overload
    def split[
        Shape: IntTuple,
        Sections: IntTuple,
        Dim: Flag[builtins.int],
    ](
        self: Tensor[Shape], split_size_or_sections: Sections, dim: Dim = 0
    ) -> MapIntTuples[lambda S: Tensor[S], split_sections_shapes(Shape, Sections, Dim)]:
        """Split tensor into variable-sized chunks. Shape inference via the type-level DSL."""
        ...

    # A list is mutable, so V2 cannot preserve its element values; this trailing
    # overload keeps the documented list spelling checking without shape inference.
    # TODO(stroxler): Route lists through `split_sections_shapes` once list values
    # survive to the type level.
    @overload
    def split(
        self: Tensor, split_size_or_sections: list[int], dim: int = 0
    ) -> tuple[Tensor, ...]:
        """Split tensor into variable-sized chunks. Shape inference unavailable for lists."""
        ...

    def chunk[
        Shape: IntTuple,
        Chunks: _Int,
        Dim: Flag[builtins.int],
    ](
        self: Tensor[Shape], chunks: Chunks, dim: Dim = 0
    ) -> MapIntTuples[lambda S: Tensor[S], chunk_shapes(Shape, Chunks, Dim)]:
        """Split tensor into chunks. Shape inference via the type-level DSL."""
        ...

    def index_select[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        IndexShape: IntTuple,
    ](
        self: Tensor[Shape], dim: Dim, index: Tensor[IndexShape]
    ) -> Tensor[index_select_shape(Shape, Dim, IndexShape)]:
        """Select elements along dimension. Shape inference via meta-shape: torch.Tensor.index_select"""
        ...

    def gather[Shape: IntTuple, Dim: Flag[builtins.int], IndexShape: IntTuple](
        self: Tensor[Shape], dim: Dim, index: Tensor[IndexShape]
    ) -> Tensor[gather_shape(Shape, Dim, IndexShape)]:
        """Gather elements along dimension. Output shape matches index shape."""
        ...

    def scatter[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        IndexShape: IntTuple,
        SourceShape: IntTuple,
    ](
        self: Tensor[Shape],
        dim: Dim,
        index: Tensor[IndexShape],
        src: Tensor[SourceShape],
    ) -> Tensor[scatter_shape(Shape, Dim, IndexShape, SourceShape)]:
        """Scatter elements along dimension. Shape-preserving operation."""
        ...

    def masked_select(self: Tensor, mask: Tensor) -> Tensor[[Any]]:
        """Select elements with mask. Returns 1D tensor with data-dependent size."""
        ...

    # ==== Phase 1.1: Missing Shape Operations (Methods) ====

    def unbind[Shape: IntTuple, Dim: Flag[builtins.int]](
        self: Tensor[Shape], dim: Dim = 0
    ) -> tuple[Tensor[unbind_shape(Shape, Dim)], ...]:
        """Remove dimension by slicing along it. Shape inference via meta-shape: torch.Tensor.unbind"""
        ...

    @overload
    def movedim[
        Shape: IntTuple,
        Source: Flag[builtins.int],
        Destination: Flag[builtins.int],
    ](
        self: Tensor[Shape], source: Source, destination: Destination
    ) -> Tensor[movedim_scalar_shape(Shape, Source, Destination)]:
        """Move a single dimension to a new position."""
        ...

    @overload
    def movedim[Shape: IntTuple, Source: IntTuple, Destination: IntTuple](
        self: Tensor[Shape], source: Source, destination: Destination
    ) -> Tensor[movedim_tuple_shape(Shape, Source, Destination)]:
        """Move multiple dimensions to new positions. Shape inference via meta-shape: torch.Tensor.movedim"""
        ...

    @overload
    def moveaxis[
        Shape: IntTuple,
        Source: Flag[builtins.int],
        Destination: Flag[builtins.int],
    ](
        self: Tensor[Shape], source: Source, destination: Destination
    ) -> Tensor[movedim_scalar_shape(Shape, Source, Destination)]:
        """Alias for movedim with a single source and destination."""
        ...

    @overload
    def moveaxis[Shape: IntTuple, Source: IntTuple, Destination: IntTuple](
        self: Tensor[Shape], source: Source, destination: Destination
    ) -> Tensor[movedim_tuple_shape(Shape, Source, Destination)]:
        """Alias for movedim. Shape inference via meta-shape: torch.Tensor.moveaxis"""
        ...

    def unfold[
        Shape: IntTuple,
        Dimension: Flag[builtins.int],
        Size: Flag[builtins.int],
        Step: Flag[builtins.int],
    ](
        self: Tensor[Shape], dimension: Dimension, size: Size, step: Step
    ) -> Tensor[unfold_shape(Shape, Dimension, Size, Step)]:
        """Returns sliding window view. Shape inference via meta-shape: torch.Tensor.unfold"""
        ...

    # TODO(stroxler): Preserve the V1 `tuple[int, ...]` fallback for a bare `Tensor` if that
    # distinction remains useful after the V2 migration is complete.
    @overload
    def size[Shape: IntTuple](self: Tensor[Shape]) -> Shape: ...
    @overload
    def size[Shape: IntTuple, Dim: Flag[builtins.int]](
        self: Tensor[Shape], dim: Dim
    ) -> _Int[size_dim_shape(Shape, Dim)]: ...

    # ==== Reduction Operations ====
    # Handled by meta-shape functions - simplified signatures

    def sum[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Sum along dimension(s). Shape inference via meta-shape: torch.Tensor.sum"""
        ...

    def mean[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Mean along dimension(s). Shape inference via meta-shape: torch.mean"""
        ...

    def all[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Check if all elements are True. Shape inference via meta-shape: torch.Tensor.all"""
        ...

    def any[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Check if any element is True. Shape inference via meta-shape: torch.Tensor.any"""
        ...

    @overload
    def max[Shape: IntTuple](self: Tensor[Shape]) -> Tensor[[]]:
        """Max of all elements (scalar). Shape inference via meta-shape: torch.Tensor.max"""
        ...

    @overload
    def max[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
        self: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
    ) -> return_types.max[reduce_shape(Shape, Dim, Keepdim)]:
        """Max along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.max"""
        ...

    @overload
    def min[Shape: IntTuple](self: Tensor[Shape]) -> Tensor[[]]:
        """Min of all elements (scalar). Shape inference via meta-shape: torch.Tensor.min"""
        ...

    @overload
    def min[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
        self: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
    ) -> return_types.min[reduce_shape(Shape, Dim, Keepdim)]:
        """Min along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.min"""
        ...

    def prod[
        Shape: IntTuple,
        Dim: Flag[builtins.int | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Product along dimension(s). Shape inference via meta-shape: torch.prod"""
        ...

    def std[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Standard deviation along dimension(s). Shape inference via meta-shape: torch.std"""
        ...

    def var[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Variance along dimension(s). Shape inference via meta-shape: torch.var"""
        ...

    def argmax[
        Shape: IntTuple,
        Dim: Flag[builtins.int | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Argmax along dimension(s). Shape inference via meta-shape: torch.argmax"""
        ...

    def argmin[
        Shape: IntTuple,
        Dim: Flag[builtins.int | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Argmin along dimension(s). Shape inference via meta-shape: torch.argmin"""
        ...

    # ==== Phase 1.2: Missing Reduction Operations (Methods) ====

    @overload
    def median[Shape: IntTuple](self: Tensor[Shape]) -> Tensor[[]]:
        """Median of all elements (scalar). Shape inference via meta-shape: torch.Tensor.median"""
        ...

    @overload
    def median[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
        self: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
    ) -> return_types.median[reduce_shape(Shape, Dim, Keepdim)]:
        """Median along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.median"""
        ...

    def logsumexp[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...]],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Log-sum-exp along dimension(s). Shape inference via meta-shape: torch.Tensor.logsumexp"""
        ...

    def count_nonzero[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    ](self: Tensor[Shape], dim: Dim = None) -> Tensor[reduce_shape_no_keep(Shape, Dim)]:
        """Count non-zero elements. Shape inference via meta-shape: torch.Tensor.count_nonzero"""
        ...

    def aminmax[
        Shape: IntTuple,
        Dim: Flag[builtins.int | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], *, dim: Dim = None, keepdim: Keepdim = False
    ) -> return_types.aminmax[reduce_shape(Shape, Dim, Keepdim)]:
        """Min and max along dimension(s). Shape inference via meta-shape: torch.Tensor.aminmax"""
        ...

    def norm[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape],
        p: int | float = 2,
        dim: Dim = None,
        keepdim: Keepdim = False,
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
        """Compute norm. Shape inference via meta-shape: torch.Tensor.norm"""
        ...

    def dist(self: Tensor, other: Tensor, p: int | float = 2) -> Tensor[[]]:
        """Compute distance to another tensor. Returns scalar tensor."""
        ...

    def cumsum[Shape: IntTuple](self: Tensor[Shape], dim: int) -> Tensor[Shape]:
        """Cumulative sum along dimension. Shape-preserving operation."""
        ...

    def cumprod[Shape: IntTuple](self: Tensor[Shape], dim: int) -> Tensor[Shape]:
        """Cumulative product along dimension. Shape-preserving operation."""
        ...

    def cummax[Shape: IntTuple](
        self: Tensor[Shape], dim: int
    ) -> return_types.cummax[Shape]:
        """Cumulative maximum along dimension. Returns (values, indices). Shape-preserving operation."""
        ...

    def cummin[Shape: IntTuple](
        self: Tensor[Shape], dim: int
    ) -> return_types.cummin[Shape]:
        """Cumulative minimum along dimension. Returns (values, indices). Shape-preserving operation."""
        ...

    # ==== Tier 2: Additional Reduction Methods ====

    def mode[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
        self: Tensor[Shape], dim: Dim = -1, keepdim: Keepdim = False
    ) -> return_types.mode[reduce_shape(Shape, Dim, Keepdim)]:
        """Mode along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.mode"""
        ...

    def topk[Shape: IntTuple, K: _Int, Dim: Flag[builtins.int]](
        self: Tensor[Shape],
        k: K,
        dim: Dim = -1,
        largest: bool = True,
        sorted: bool = True,
    ) -> return_types.topk[topk_shape(Shape, Dim, K)]:
        """Top k elements. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.topk"""
        ...

    def sort[Shape: IntTuple](
        self: Tensor[Shape],
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> return_types.sort[Shape]:
        """Sort tensor. Returns (values, indices). Shape-preserving operation."""
        ...

    def kthvalue[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], k: int, dim: Dim = -1, keepdim: Keepdim = False
    ) -> return_types.kthvalue[reduce_shape(Shape, Dim, Keepdim)]:
        """Kth smallest value. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.kthvalue"""
        ...

    # ==== Phase 1.3: Tensor Creation Operations (Methods) ====

    def diag_embed[
        Shape: IntTuple,
        Offset: Flag[builtins.int],
        Dim1: Flag[builtins.int],
        Dim2: Flag[builtins.int],
    ](
        self: Tensor[Shape], offset: Offset = 0, dim1: Dim1 = -2, dim2: Dim2 = -1
    ) -> Tensor[diag_embed_shape(Shape, Offset, Dim1, Dim2)]:
        """Create diagonal tensor. Shape inference via meta-shape: torch.Tensor.diag_embed"""
        ...

    def tril(self, diagonal: int = 0) -> Self:
        """Lower triangular part. Shape inference via generic fixture signature."""
        ...

    def triu(self, diagonal: int = 0) -> Self:
        """Upper triangular part. Shape inference via generic fixture signature."""
        ...

    # ==== Phase 1.4: Basic Linear Algebra Operations (Methods) ====

    def matmul[Left: IntTuple, Right: IntTuple](
        self: Tensor[Left], other: Tensor[Right]
    ) -> Tensor[matmul_shape(Left, Right)]:
        """Matrix multiplication. Shape inference via meta-shape: torch.Tensor.matmul"""
        ...

    def mm[N: IntVar, K: IntVar, M: IntVar](
        self: Tensor[[N, K]], mat2: Tensor[[K, M]]
    ) -> Tensor[[N, M]]:
        """Matrix multiplication (2D @ 2D). Output: [N, M]."""
        ...

    def bmm[B: IntVar, N: IntVar, K: IntVar, M: IntVar](
        self: Tensor[[B, N, K]], mat2: Tensor[[B, K, M]]
    ) -> Tensor[[B, N, M]]:
        """Batch matrix multiplication (3D @ 3D). Output: [B, N, M]."""
        ...

    def mv[M: IntVar, K: IntVar](self: Tensor[[M, K]], vec: Tensor[[K]]) -> Tensor[[M]]:
        """Matrix-vector multiplication (2D @ 1D). Output: [M]."""
        ...

    def dot[N: IntVar](self: Tensor[[N]], other: Tensor[[N]]) -> Tensor[[]]:
        """Dot product. Returns scalar tensor."""
        ...

    # ==== Phase 2: Arithmetic & Basic Operations (Methods) ====

    # Arithmetic methods
    def add[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise addition. Shape inference via generic fixture signature."""
        ...

    def sub[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise subtraction. Shape inference via generic fixture signature."""
        ...

    def mul[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise multiplication. Shape inference via generic fixture signature."""
        ...

    def div[OtherShape: _Shape = []](
        self,
        other: _TensorLike[OtherShape],
        *,
        rounding_mode: str | None = None,
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise division. Shape inference via generic fixture signature."""
        ...

    def pow[OtherShape: _Shape = []](
        self, exponent: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise power. Shape inference via generic fixture signature."""
        ...

    def neg(self) -> Self:
        """Element-wise negation. Shape inference via generic fixture signature."""
        ...

    def abs(self) -> Self:
        """Element-wise absolute value. Shape inference via generic fixture signature."""
        ...

    def floor(self) -> Self:
        """Element-wise floor. Shape inference via generic fixture signature."""
        ...

    def ceil(self) -> Self:
        """Element-wise ceiling. Shape inference via generic fixture signature."""
        ...

    def round(self) -> Self:
        """Element-wise rounding. Shape inference via generic fixture signature."""
        ...

    # Point-wise math methods
    def sin(self) -> Self:
        """Element-wise sine. Shape inference via generic fixture signature."""
        ...

    def cos(self) -> Self:
        """Element-wise cosine. Shape inference via generic fixture signature."""
        ...

    def tan(self) -> Self:
        """Element-wise tangent. Shape inference via generic fixture signature."""
        ...

    def exp(self) -> Self:
        """Element-wise exponential. Shape inference via generic fixture signature."""
        ...

    def log(self) -> Self:
        """Element-wise natural logarithm. Shape inference via generic fixture signature."""
        ...

    def sqrt(self) -> Self:
        """Element-wise square root. Shape inference via generic fixture signature."""
        ...

    def tanh(self) -> Self:
        """Element-wise hyperbolic tangent. Shape inference via generic fixture signature."""
        ...

    def asin(self) -> Self:
        """Element-wise arcsine. Shape inference via generic fixture signature."""
        ...

    def acos(self) -> Self:
        """Element-wise arccosine. Shape inference via generic fixture signature."""
        ...

    def atan(self) -> Self:
        """Element-wise arctangent. Shape inference via generic fixture signature."""
        ...

    def sinh(self) -> Self:
        """Element-wise hyperbolic sine. Shape inference via generic fixture signature."""
        ...

    def cosh(self) -> Self:
        """Element-wise hyperbolic cosine. Shape inference via generic fixture signature."""
        ...

    def exp2(self) -> Self:
        """Element-wise base-2 exponential. Shape inference via generic fixture signature."""
        ...

    def expm1(self) -> Self:
        """Element-wise exp(x)-1. Shape inference via generic fixture signature."""
        ...

    def log2(self) -> Self:
        """Element-wise base-2 logarithm. Shape inference via generic fixture signature."""
        ...

    def log10(self) -> Self:
        """Element-wise base-10 logarithm. Shape inference via generic fixture signature."""
        ...

    def log1p(self) -> Self:
        """Element-wise log(1+x). Shape inference via generic fixture signature."""
        ...

    def rsqrt(self) -> Self:
        """Element-wise reciprocal square root. Shape inference via generic fixture signature."""
        ...

    def square(self) -> Self:
        """Element-wise square. Shape inference via generic fixture signature."""
        ...

    def reciprocal(self) -> Self:
        """Element-wise reciprocal. Shape inference via generic fixture signature."""
        ...

    def sign(self) -> Self:
        """Element-wise sign. Shape inference via generic fixture signature."""
        ...

    def sigmoid(self) -> Self:
        """Element-wise sigmoid. Shape inference via generic fixture signature."""
        ...

    def trunc(self) -> Self:
        """Element-wise truncation. Shape inference via generic fixture signature."""
        ...

    def frac(self) -> Self:
        """Element-wise fractional part. Shape inference via generic fixture signature."""
        ...

    # Comparison methods
    def eq[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise equality. Shape inference via generic fixture signature."""
        ...

    def ne[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise inequality. Shape inference via generic fixture signature."""
        ...

    def lt[OtherShape: _Shape = []](
        self, other: _RealTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise less than. Shape inference via generic fixture signature."""
        ...

    def le[OtherShape: _Shape = []](
        self, other: _RealTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise less than or equal. Shape inference via generic fixture signature."""
        ...

    def gt[OtherShape: _Shape = []](
        self, other: _RealTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise greater than. Shape inference via generic fixture signature."""
        ...

    def ge[OtherShape: _Shape = []](
        self, other: _RealTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise greater than or equal. Shape inference via generic fixture signature."""
        ...

    # Logical methods
    def logical_and[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise logical AND. Shape inference via generic fixture signature."""
        ...

    def logical_or[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise logical OR. Shape inference via generic fixture signature."""
        ...

    def logical_not(self) -> Self:
        """Element-wise logical NOT. Shape inference via generic fixture signature."""
        ...

    # Activation methods
    def relu(self) -> Self:
        """ReLU activation. Shape inference via generic fixture signature."""
        ...

    # Clamping methods
    def clamp(
        self,
        min: Tensor | builtins.float | builtins.int | None = None,
        max: Tensor | builtins.float | builtins.int | None = None,
    ) -> Self:
        """Clamp tensor values. Shape inference via generic fixture signature."""
        ...

    def clip(
        self,
        min: Tensor | builtins.float | builtins.int | None = None,
        max: Tensor | builtins.float | builtins.int | None = None,
    ) -> Self:
        """Alias for clamp. Shape inference via generic fixture signature."""
        ...

    def clamp_min(self, min: Tensor | builtins.float | builtins.int) -> Self:
        """Clamp tensor values from below. Shape inference via generic fixture signature."""
        ...

    def clamp_max(self, max: Tensor | builtins.float | builtins.int) -> Self:
        """Clamp tensor values from above. Shape inference via generic fixture signature."""
        ...

    # Additional mathematical methods
    def atan2[OtherShape: _Shape](
        self, other: Tensor[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise arctangent. Shape inference via generic fixture signature."""
        ...

    def hypot(self, other: Tensor) -> Self:
        """Element-wise hypotenuse. Shape inference via generic fixture signature."""
        ...

    def lerp(self, end: Tensor, weight: float) -> Self:
        """Linear interpolation. Shape inference via generic fixture signature."""
        ...

    def fmod(self, other: Tensor) -> Self:
        """Element-wise modulo. Shape inference via generic fixture signature."""
        ...

    def remainder(self, other: Tensor | int | float) -> Self:
        """Element-wise remainder. Shape inference via generic fixture signature."""
        ...

    def copysign(self, other: Tensor) -> Self:
        """Copy sign. Shape inference via generic fixture signature."""
        ...

    def nextafter(self, other: Tensor) -> Self:
        """Next floating-point value. Shape inference via generic fixture signature."""
        ...

    def erf(self) -> Self:
        """Error function. Shape inference via generic fixture signature."""
        ...

    def erfc(self) -> Self:
        """Complementary error function. Shape inference via generic fixture signature."""
        ...

    def erfinv(self) -> Self:
        """Inverse error function. Shape inference via generic fixture signature."""
        ...

    def lgamma(self) -> Self:
        """Log gamma function. Shape inference via generic fixture signature."""
        ...

    def digamma(self) -> Self:
        """Digamma function. Shape inference via generic fixture signature."""
        ...

    def polygamma(self, n: int) -> Self:
        """Polygamma function. Shape inference via generic fixture signature."""
        ...

    def asinh(self) -> Self:
        """Inverse hyperbolic sine. Shape inference via generic fixture signature."""
        ...

    def acosh(self) -> Self:
        """Inverse hyperbolic cosine. Shape inference via generic fixture signature."""
        ...

    def atanh(self) -> Self:
        """Inverse hyperbolic tangent. Shape inference via generic fixture signature."""
        ...

    def deg2rad(self) -> Self:
        """Convert degrees to radians. Shape inference via generic fixture signature."""
        ...

    def deg2rad_(self) -> Self:
        """Convert degrees to radians in-place. Shape inference via generic fixture signature."""
        ...

    def rad2deg(self) -> Self:
        """Convert radians to degrees. Shape inference via generic fixture signature."""
        ...

    # Bitwise methods
    def bitwise_and[OtherShape: _Shape = []](
        self, other: _IntegerTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Bitwise AND. Shape inference via generic fixture signature."""
        ...

    def bitwise_or[OtherShape: _Shape = []](
        self, other: _IntegerTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Bitwise OR. Shape inference via generic fixture signature."""
        ...

    def bitwise_xor[OtherShape: _Shape = []](
        self, other: _IntegerTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Bitwise XOR. Shape inference via generic fixture signature."""
        ...

    def bitwise_not(self) -> Self:
        """Bitwise NOT. Shape inference via generic fixture signature."""
        ...

    def bitwise_left_shift[OtherShape: _Shape = []](
        self, other: _IntegerTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Bitwise left shift. Shape inference via generic fixture signature."""
        ...

    def bitwise_right_shift[OtherShape: _Shape = []](
        self, other: _IntegerTensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Bitwise right shift. Shape inference via generic fixture signature."""
        ...

    # Additional comparison/validation methods
    def isclose(self, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> Self:
        """Check if tensors are close. Shape inference via generic fixture signature."""
        ...

    def isreal(self) -> Self:
        """Check if elements are real. Shape inference via generic fixture signature."""
        ...

    def isposinf(self) -> Self:
        """Check if positive infinity. Shape inference via generic fixture signature."""
        ...

    def isneginf(self) -> Self:
        """Check if negative infinity. Shape inference via generic fixture signature."""
        ...

    def isnan(self) -> Self:
        """Check if elements are NaN. Shape inference via generic fixture signature."""
        ...

    def isinf(self) -> Self:
        """Check if elements are infinity. Shape inference via generic fixture signature."""
        ...

    def isfinite(self) -> Self:
        """Check if elements are finite. Shape inference via generic fixture signature."""
        ...

    def is_floating_point(self) -> builtins.bool:
        """Check if tensor has floating point dtype."""
        ...

    def maximum[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise maximum. Shape inference via generic fixture signature."""
        ...

    def minimum[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise minimum. Shape inference via generic fixture signature."""
        ...

    def fmax[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise maximum (NaN handling). Shape inference via generic fixture signature."""
        ...

    def fmin[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]:
        """Element-wise minimum (NaN handling). Shape inference via generic fixture signature."""
        ...

    # ==== Phase 4: Advanced Linear Algebra Methods ====

    def cholesky(self, upper: bool = False) -> Self:
        """Cholesky decomposition. Shape inference via generic fixture signature."""
        ...

    def inverse(self) -> Self:
        """Matrix inverse. Shape inference via generic fixture signature."""
        ...

    def det[Batch: IntTuple, M: IntVar, N: IntVar](
        self: Tensor[[*Elements[Batch], M, N]],
    ) -> Tensor[Batch]:
        """Determinant. Returns batch dimensions only (drops last 2 dims)."""
        ...

    def logdet[Batch: IntTuple, M: IntVar, N: IntVar](
        self: Tensor[[*Elements[Batch], M, N]],
    ) -> Tensor[Batch]:
        """Log determinant. Returns batch dimensions only (drops last 2 dims)."""
        ...

    @overload
    def slogdet[Batch: IntTuple, M: IntVar, N: IntVar](
        self: Tensor[[*Elements[Batch], M, N]],
    ) -> return_types.slogdet[Batch]: ...
    @overload
    def slogdet[Shape: IntTuple](
        self: Tensor[Shape],
    ) -> return_types.slogdet[slogdet_shape(Shape)]: ...
    def matrix_power(self, n: int) -> Self:
        """Matrix power. Shape inference via generic fixture signature."""
        ...

    def trace[Batch: IntTuple, M: IntVar, N: IntVar](
        self: Tensor[[*Elements[Batch], M, N]],
    ) -> Tensor[Batch]:
        """Matrix trace. Returns batch dimensions only (drops last 2 dims)."""
        ...

    # ==== Phase 5: Advanced Indexing & Conditional Methods ====

    def masked_fill[InputShape: IntTuple, MaskShape: IntTuple](
        self: Tensor[InputShape], mask: Tensor[MaskShape], value: float
    ) -> Tensor[broadcast(InputShape, MaskShape)]:
        """Fill masked elements. Shape inference via generic signature"""
        ...

    def masked_fill_(self, mask: Tensor, value: float) -> Self:
        """Fill masked elements in-place. Shape inference via generic signature"""
        ...

    def masked_scatter[InputShape: IntTuple, MaskShape: IntTuple](
        self: Tensor[InputShape], mask: Tensor[MaskShape], source: Tensor
    ) -> Tensor[broadcast(InputShape, MaskShape)]:
        """Scatter into masked positions. Shape inference via generic fixture signature."""
        ...

    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Self:
        """Scatter into masked positions in-place. Shape inference via generic fixture signature."""
        ...

    def index_add[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        IndexShape: IntTuple,
        SourceShape: IntTuple,
    ](
        self: Tensor[Shape],
        dim: Dim,
        index: Tensor[IndexShape],
        source: Tensor[SourceShape],
        alpha: float = 1,
    ) -> Tensor[indexed_source_shape(Shape, Dim, IndexShape, SourceShape)]:
        """Add values at indices. Shape inference via generic fixture signature."""
        ...

    def index_add_[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        IndexShape: IntTuple,
        SourceShape: IntTuple,
    ](
        self: Tensor[Shape],
        dim: Dim,
        index: Tensor[IndexShape],
        source: Tensor[SourceShape],
        alpha: float = 1,
    ) -> Tensor[indexed_source_shape(Shape, Dim, IndexShape, SourceShape)]:
        """Add values at indices in-place. Shape inference via generic fixture signature."""
        ...

    def index_copy[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        IndexShape: IntTuple,
        SourceShape: IntTuple,
    ](
        self: Tensor[Shape],
        dim: Dim,
        index: Tensor[IndexShape],
        source: Tensor[SourceShape],
    ) -> Tensor[indexed_source_shape(Shape, Dim, IndexShape, SourceShape)]:
        """Copy values to indices. Shape inference via generic fixture signature."""
        ...

    def index_copy_[
        Shape: IntTuple,
        Dim: Flag[builtins.int],
        IndexShape: IntTuple,
        SourceShape: IntTuple,
    ](
        self: Tensor[Shape],
        dim: Dim,
        index: Tensor[IndexShape],
        source: Tensor[SourceShape],
    ) -> Tensor[indexed_source_shape(Shape, Dim, IndexShape, SourceShape)]:
        """Copy values to indices in-place. Shape inference via generic fixture signature."""
        ...

    def index_put(
        self,
        indices: tuple[Tensor, ...],
        values: Tensor,
        accumulate: bool = False,
    ) -> Self:
        """Put values at indices. Shape inference via generic fixture signature."""
        ...

    def index_put_(
        self,
        indices: tuple[Tensor, ...],
        values: Tensor,
        accumulate: bool = False,
    ) -> Self:
        """Put values at indices in-place. Shape inference via generic fixture signature."""
        ...

    def index_fill[Shape: IntTuple, Dim: Flag[builtins.int], IndexShape: IntTuple](
        self: Tensor[Shape], dim: Dim, index: Tensor[IndexShape], value: float
    ) -> Tensor[index_fill_shape(Shape, Dim, IndexShape)]:
        """Fill indices with value. Shape inference via generic fixture signature."""
        ...

    def index_fill_[Shape: IntTuple, Dim: Flag[builtins.int], IndexShape: IntTuple](
        self: Tensor[Shape], dim: Dim, index: Tensor[IndexShape], value: float
    ) -> Tensor[index_fill_shape(Shape, Dim, IndexShape)]:
        """Fill indices with value in-place. Shape inference via generic fixture signature."""
        ...

    def take[Shape: IntTuple, IndexShape: IntTuple](
        self: Tensor[Shape], index: Tensor[IndexShape]
    ) -> Tensor[take_shape(Shape, IndexShape)]:
        """Take elements at indices. Output shape matches index shape."""
        ...

    def take_along_dim[
        Shape: IntTuple,
        IndexShape: IntTuple,
        Dim: Flag[builtins.int | None],
    ](
        self: Tensor[Shape], indices: Tensor[IndexShape], dim: Dim = None
    ) -> Tensor[take_along_dim_shape(Shape, IndexShape, Dim)]: ...
    def put[Shape: IntTuple, IndexShape: IntTuple, SourceShape: IntTuple](
        self: Tensor[Shape],
        index: Tensor[IndexShape],
        source: Tensor[SourceShape],
        accumulate: bool = False,
    ) -> Tensor[put_shape(Shape, IndexShape, SourceShape)]:
        """Put values at indices. Shape inference via generic fixture signature."""
        ...

    def put_[Shape: IntTuple, IndexShape: IntTuple, SourceShape: IntTuple](
        self: Tensor[Shape],
        index: Tensor[IndexShape],
        source: Tensor[SourceShape],
        accumulate: bool = False,
    ) -> Tensor[put_shape(Shape, IndexShape, SourceShape)]:
        """Put values at indices in-place. Shape inference via generic fixture signature."""
        ...

    # ==== Phase 6: Specialized Operations (Methods) ====

    def bernoulli(self, p: float = 0.5) -> Self:
        """Sample from Bernoulli distribution. Shape inference via generic fixture signature."""
        ...

    def bernoulli_(self, p: float = 0.5) -> Self:
        """Sample from Bernoulli distribution in-place. Shape inference via generic fixture signature."""
        ...

    @overload
    def multinomial[Shape: IntTuple, NumSamples: _Int](
        self: Tensor[Shape],
        num_samples: NumSamples,
        replacement: Literal[False] = False,
        *,
        generator: Generator | None = None,
    ) -> Tensor[multinomial_shape(Shape, NumSamples, False)]: ...
    @overload
    def multinomial[Shape: IntTuple, NumSamples: _Int](
        self: Tensor[Shape],
        num_samples: NumSamples,
        replacement: Literal[True],
        *,
        generator: Generator | None = None,
    ) -> Tensor[multinomial_shape(Shape, NumSamples, True)]: ...
    @overload
    def multinomial[Shape: IntTuple, NumSamples: _Int](
        self: Tensor[Shape],
        num_samples: NumSamples,
        replacement: builtins.bool,
        *,
        generator: Generator | None = None,
    ) -> Tensor[multinomial_shape(Shape, NumSamples, True)]:
        # A non-literal flag might permit replacement, so the return type must
        # not apply the without-replacement upper bound.
        """Sample from multinomial distribution. Shape inference via meta-shape: torch.Tensor.multinomial"""
        ...

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> Self:
        """Fill with normal distribution in-place. Shape inference via generic fixture signature."""
        ...

    def random_(
        self,
        low: int = 0,
        high: int | None = None,
        *,
        generator: Generator | None = None,
    ) -> Self:
        """Fill with random integers in-place. Shape inference via generic fixture signature."""
        ...

    def uniform_(
        self,
        low: float = 0.0,
        high: float = 1.0,
        *,
        generator: Generator | None = None,
    ) -> Self:
        """Fill with uniform distribution in-place. Shape inference via generic fixture signature."""
        ...

    def numel(self: Tensor[Shape]) -> _Int[numel_shape(Shape)]:
        """Return the number of elements."""
        ...

    def dim(self: Tensor[Shape]) -> _Int[dim_shape(Shape)]:
        """Number of dimensions. Shape inference via type-level DSL."""
        ...

    def nelement(self: Tensor[Shape]) -> _Int[numel_shape(Shape)]:
        """Return the number of elements."""
        ...

# ============================================================================
# Module-level Functions
# ============================================================================

def matmul[Left: IntTuple, Right: IntTuple](
    self: Tensor[Left], other: Tensor[Right]
) -> Tensor[matmul_shape(Left, Right)]:
    """Matrix multiplication function. Shape inference via meta-shape: torch.matmul"""
    ...

def cat[Shapes: IntTuples, Dim: Flag[builtins.int]](
    tensors: MapIntTuples[lambda S: Tensor[S], Shapes], dim: Dim = 0
) -> Tensor[cat_shape(Shapes, Dim)]:
    """Concatenate tensors. Shape inference via meta-shape: torch.cat"""
    ...

def concat[Shapes: IntTuples, Dim: Flag[builtins.int]](
    tensors: MapIntTuples[lambda S: Tensor[S], Shapes], dim: Dim = 0
) -> Tensor[cat_shape(Shapes, Dim)]:
    """Alias for concatenate/cat. Shape inference via meta-shape: torch.cat"""
    ...

@overload
def concatenate[Shapes: IntTuples, Dim: Flag[builtins.int]](
    tensors: MapIntTuples[lambda S: Tensor[S], Shapes], dim: Dim = 0
) -> Tensor[cat_shape(Shapes, Dim)]:
    """Alias for concat/cat. Shape inference via meta-shape: torch.cat"""
    ...

@overload
def concatenate[Shapes: IntTuples, Axis: Flag[builtins.int]](
    tensors: MapIntTuples[lambda S: Tensor[S], Shapes], *, axis: Axis
) -> Tensor[cat_shape(Shapes, Axis)]: ...
@overload
def stack[Shapes: IntTuples, Dim: Flag[builtins.int]](
    tensors: MapIntTuples[lambda S: Tensor[S], Shapes], dim: Dim = 0
) -> Tensor[stack_shape(Shapes, Dim)]:
    """Stack tensors (adds new dimension)."""
    ...

@overload
def stack(tensors: Sequence[Any], dim: builtins.int = 0) -> Tensor: ...
def transpose[
    Shape: IntTuple,
    Dim0: Flag[builtins.int],
    Dim1: Flag[builtins.int],
](
    self: Tensor[Shape], dim0: Dim0, dim1: Dim1
) -> Tensor[transpose_shape(Shape, Dim0, Dim1)]:
    """Transpose two dimensions. Shape inference via meta-shape: torch.transpose"""
    ...

def flip[Shape: IntTuple](
    input: Tensor[Shape], dims: int | tuple[int, ...] | list[int]
) -> Tensor[Shape]:
    """Reverse tensor elements along dimensions. Shape-preserving."""
    ...

@overload
def reshape[Shape: IntTuple, NewShape: IntTuple](
    self: Tensor[Shape], shape: NewShape
) -> Tensor[reshape_shape(Shape, NewShape)]:
    """Reshape tensor. Shape inference via type-level DSL."""
    ...

@overload
def reshape(self: Tensor, shape: Sequence[builtins.int]) -> Tensor: ...
def squeeze[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
](self: Tensor[Shape], dim: Dim = None) -> Tensor[squeeze_shape(Shape, Dim)]:
    """Remove dimensions of size 1. Shape inference via meta-shape: torch.squeeze"""
    ...

def unsqueeze[Shape: IntTuple, Dim: Flag[builtins.int]](
    self: Tensor[Shape], dim: Dim
) -> Tensor[unsqueeze_shape(Shape, Dim)]:
    """Add dimension of size 1. Shape inference via meta-shape: torch.unsqueeze"""
    ...

@overload
def repeat_interleave[
    Shape: IntTuple,
    Repeats: _Int,
    OutputSize: _Int,
    Dim: Flag[builtins.int | None],
](
    input: Tensor[Shape],
    repeats: Repeats,
    dim: Dim = None,
    *,
    output_size: OutputSize,
) -> Tensor[repeat_interleave_checked_shape(Shape, Repeats, OutputSize, Dim)]: ...
@overload
def repeat_interleave[
    Shape: IntTuple,
    OutputSize: _Int,
    Dim: Flag[builtins.int | None],
](
    input: Tensor[Shape],
    repeats: Tensor,
    dim: Dim = None,
    *,
    output_size: OutputSize,
) -> Tensor[repeat_interleave_output_shape(Shape, OutputSize, Dim)]: ...
@overload
def repeat_interleave[Shape: IntTuple, Repeats: _Int, Dim: Flag[builtins.int | None]](
    input: Tensor[Shape],
    repeats: Repeats,
    dim: Dim = None,
    *,
    output_size: None = None,
) -> Tensor[repeat_interleave_shape(Shape, Repeats, Dim)]: ...
@overload
def repeat_interleave(
    input: Tensor,
    repeats: builtins.int | Tensor,
    dim: builtins.int | None = None,
    *,
    output_size: builtins.int | None = None,
) -> Tensor:
    """Repeat tensor elements."""
    ...

def segment_reduce(
    data: Tensor,
    reduce: str,
    *,
    lengths: Tensor | None = None,
    indices: Tensor | None = None,
    offsets: Tensor | None = None,
    axis: int = 0,
    unsafe: bool = False,
    initial: int | float | None = None,
) -> Tensor:
    """Reduce values by segment. Data-dependent shape."""
    ...

@overload
def permute[Shape: IntTuple, Dims: Flag[tuple[builtins.int, ...]]](
    self: Tensor[Shape], dims: Dims
) -> Tensor[permute_shape(Shape, Dims)]:
    """Permute dimensions. Shape inference via type-level DSL."""
    ...

@overload
def permute(self: Tensor, dims: tuple[builtins.int, ...]) -> Tensor: ...
def sum[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Sum along dimension(s). Shape inference via meta-shape: torch.sum"""
    ...

def mean[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Mean along dimension(s). Shape inference via meta-shape: torch.mean"""
    ...

@overload
def max[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[[]]:
    """Max of all elements (scalar). Shape inference via meta-shape: torch.max"""
    ...

@overload
def max[Shape: IntTuple, OtherShape: IntTuple](
    input: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise maximum of two tensors."""
    ...

@overload
def max[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
    input: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
) -> return_types.max[reduce_shape(Shape, Dim, Keepdim)]:
    """Max along dimension. Returns (values, indices). Shape inference via meta-shape: torch.max"""
    ...

@overload
def min[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[[]]:
    """Min of all elements (scalar). Shape inference via meta-shape: torch.min"""
    ...

@overload
def min[Shape: IntTuple, OtherShape: IntTuple](
    input: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise minimum of two tensors."""
    ...

@overload
def min[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
    input: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
) -> return_types.min[reduce_shape(Shape, Dim, Keepdim)]:
    """Min along dimension. Returns (values, indices). Shape inference via meta-shape: torch.min"""
    ...

def prod[Shape: IntTuple, Dim: Flag[builtins.int | None], Keepdim: Flag[builtins.bool]](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Product along dimension(s). Shape inference via meta-shape: torch.prod"""
    ...

def std[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Standard deviation. Shape inference via meta-shape: torch.std"""
    ...

def var[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Variance. Shape inference via meta-shape: torch.var"""
    ...

def argmax[
    Shape: IntTuple,
    Dim: Flag[builtins.int | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Argmax. Shape inference via meta-shape: torch.argmax"""
    ...

def argmin[
    Shape: IntTuple,
    Dim: Flag[builtins.int | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Argmin. Shape inference via meta-shape: torch.argmin"""
    ...

def flatten[
    Shape: IntTuple,
    StartDim: Flag[builtins.int],
    EndDim: Flag[builtins.int],
](
    self: Tensor[Shape], start_dim: StartDim = 0, end_dim: EndDim = -1
) -> Tensor[flatten_shape(Shape, StartDim, EndDim)]:
    """Flatten dimensions. Shape inference via the type-level DSL."""
    ...

# ==== Tensor Creation Functions ====

@overload
def randn[Shape: IntTuple](
    *size: *Shape,
    dtype: Any = None,
    device: Any = None,
    generator: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor with random values. Shape is inferred from `size`."""
    ...

@overload
def randn[Shape: IntTuple](
    size: Shape,
    *,
    dtype: Any = None,
    device: Any = None,
    generator: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor with random values. Shape is inferred from `size`."""
    ...

@overload
def randn(
    size: Sequence[builtins.int],
    *,
    dtype: Any = None,
    device: Any = None,
    generator: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor: ...
@overload
def rand[Shape: IntTuple](
    *size: *Shape,
    dtype: Any = None,
    device: Any = None,
    generator: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor with random values [0, 1). Shape is inferred from `size`."""
    ...

@overload
def rand[Shape: IntTuple](
    size: Shape,
    *,
    dtype: Any = None,
    device: Any = None,
    generator: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor with random values [0, 1). Shape is inferred from `size`."""
    ...

@overload
def rand(
    size: Sequence[builtins.int],
    *,
    dtype: Any = None,
    device: Any = None,
    generator: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor: ...
@overload
def zeros[Shape: IntTuple](
    *size: *Shape,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor filled with zeros. Shape is inferred from `size`."""
    ...

@overload
def zeros[Shape: IntTuple](
    size: Shape,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor filled with zeros. Shape is inferred from `size`."""
    ...

@overload
def zeros(
    size: Sequence[builtins.int],
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor: ...
@overload
def ones[Shape: IntTuple](
    *size: *Shape,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor filled with ones. Shape is inferred from `size`."""
    ...

@overload
def ones[Shape: IntTuple](
    size: Shape,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor filled with ones. Shape is inferred from `size`."""
    ...

@overload
def ones(
    size: Sequence[builtins.int],
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor: ...
@overload
def empty[Shape: IntTuple](
    *size: *Shape, dtype: Any = None, device: Any = None
) -> Tensor[Shape]:
    """Create uninitialized tensor. Shape is inferred from `size`."""
    ...

@overload
def empty[Shape: IntTuple](
    size: Shape, dtype: Any = None, device: Any = None
) -> Tensor[Shape]:
    """Create uninitialized tensor. Shape is inferred from `size`."""
    ...

@overload
def full[Shape: IntTuple](
    size: Shape,
    fill_value: builtins.float,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor[Shape]:
    """Create tensor filled with a value. Shape is inferred from `size`."""
    ...

@overload
def full(
    size: Sequence[builtins.int],
    fill_value: builtins.float,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
) -> Tensor: ...
@overload
def arange[End: IntVar](
    end: _Int[End], *, dtype: int | None = None, device: Any = None
) -> Tensor[[arange_extent(_Int[End])]]:
    """Create 1D tensor with range [0, end). Shape inference via meta-shape: torch.arange"""
    ...

@overload
def arange[Start: IntVar, End: IntVar, Step: Flag[builtins.int]](
    start: _Int[Start],
    end: _Int[End],
    step: Step = 1,
    *,
    dtype: int | None = None,
    device: Any = None,
) -> Tensor[[arange_step_extent(_Int[Start], _Int[End], Step)]]:
    """Create 1D tensor with range [start, end) with step. Shape inference via meta-shape: torch.arange"""
    ...

@overload
def arange(end: int, *, dtype: int | None = None, device: Any = None) -> Tensor[[int]]:
    """Create 1D tensor with a gradual bound."""
    ...

@overload
def arange(
    start: int,
    end: int,
    step: int = 1,
    *,
    dtype: int | None = None,
    device: Any = None,
) -> Tensor[[int]]:
    """Create 1D tensor with gradual bounds or step."""
    ...

def linspace[Steps: IntVar](
    start: float,
    end: float,
    steps: _Int[Steps],
    *,
    dtype: Any = None,
    device: Any = None,
) -> Tensor[[nonnegative_extent(_Int[Steps])]]:
    """Create a 1D tensor with one linearly spaced value per step."""
    ...

def logspace[Steps: IntVar](
    start: float,
    end: float,
    steps: _Int[Steps],
    base: float = 10.0,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Tensor[[nonnegative_extent(_Int[Steps])]]:
    """Create a 1D tensor with one logarithmically spaced value per step."""
    ...

@overload
def eye[N: IntVar](
    n: _Int[N], *, dtype: Any = None, device: Any = None
) -> Tensor[[nonnegative_extent(_Int[N]), nonnegative_extent(_Int[N])]]:
    """Create a square 2D identity matrix."""
    ...

@overload
def eye[N: IntVar, M: IntVar](
    n: _Int[N], m: _Int[M], *, dtype: Any = None, device: Any = None
) -> Tensor[[nonnegative_extent(_Int[N]), nonnegative_extent(_Int[M])]]:
    """Create a rectangular 2D identity matrix."""
    ...

# ==== Shape Manipulation Functions ====

def broadcast_to[InputShape: IntTuple, TargetShape: IntTuple](
    input: Tensor[InputShape], shape: TargetShape
) -> Tensor[expand_shape(InputShape, TargetShape)]:
    """Broadcast a tensor to `shape`."""
    ...

def tile[Shape: IntTuple, Repeats: IntTuple](
    input: Tensor[Shape], dims: Repeats
) -> Tensor[tile_shape(Shape, Repeats)]:
    """Tile tensor by repeating. Shape inference via type-level DSL."""
    ...

def select[Shape: IntTuple, Dim: Flag[builtins.int], Index: _Int](
    self: Tensor[Shape], dim: Dim, index: Index
) -> Tensor[select_shape(Shape, Dim, Index)]:
    """Select along dimension. Shape inference via meta-shape: torch.select"""
    ...

def narrow[Shape: IntTuple, Dim: Flag[builtins.int], Start: _Int, Length: _Int](
    self: Tensor[Shape], dim: Dim, start: Start, length: Length
) -> Tensor[narrow_shape(Shape, Dim, Start, Length)]:
    """Narrow tensor along dimension. Shape inference via meta-shape: torch.narrow"""
    ...

@overload
def split[
    Shape: IntTuple,
    SplitSize: _Int,
    Dim: Flag[builtins.int],
](
    self: Tensor[Shape], split_size_or_sections: SplitSize, dim: Dim = 0
) -> MapIntTuples[lambda S: Tensor[S], split_size_shapes(Shape, SplitSize, Dim)]:
    """Split tensor into chunks. Shape inference via the type-level DSL."""
    ...

@overload
def split[
    Shape: IntTuple,
    Sections: IntTuple,
    Dim: Flag[builtins.int],
](
    self: Tensor[Shape], split_size_or_sections: Sections, dim: Dim = 0
) -> MapIntTuples[lambda S: Tensor[S], split_sections_shapes(Shape, Sections, Dim)]:
    """Split tensor into variable-sized chunks. Shape inference via the type-level DSL."""
    ...

# A list is mutable, so V2 cannot preserve its element values; this trailing
# overload keeps the documented list spelling checking without shape inference.
# TODO(stroxler): Route lists through `split_sections_shapes` once list values
# survive to the type level.
@overload
def split(
    self: Tensor, split_size_or_sections: list[int], dim: int = 0
) -> tuple[Tensor, ...]:
    """Split tensor into variable-sized chunks. Shape inference unavailable for lists."""
    ...

def chunk[
    Shape: IntTuple,
    Chunks: _Int,
    Dim: Flag[builtins.int],
](
    self: Tensor[Shape], chunks: Chunks, dim: Dim = 0
) -> MapIntTuples[lambda S: Tensor[S], chunk_shapes(Shape, Chunks, Dim)]:
    """Split tensor into chunks. Shape inference via the type-level DSL."""
    ...

def index_select[
    Shape: IntTuple,
    Dim: Flag[builtins.int],
    IndexShape: IntTuple,
](
    self: Tensor[Shape], dim: Dim, index: Tensor[IndexShape]
) -> Tensor[index_select_shape(Shape, Dim, IndexShape)]:
    """Select elements along dimension. Shape inference via meta-shape: torch.index_select"""
    ...

def gather[Shape: IntTuple, Dim: Flag[builtins.int], IndexShape: IntTuple](
    input: Tensor[Shape], dim: Dim, index: Tensor[IndexShape]
) -> Tensor[gather_shape(Shape, Dim, IndexShape)]:
    """Gather elements along dimension. Output shape matches index shape."""
    ...

def scatter[
    Shape: IntTuple,
    Dim: Flag[builtins.int],
    IndexShape: IntTuple,
    SourceShape: IntTuple,
](
    input: Tensor[Shape],
    dim: Dim,
    index: Tensor[IndexShape],
    src: Tensor[SourceShape],
) -> Tensor[scatter_shape(Shape, Dim, IndexShape, SourceShape)]:
    """Scatter elements along dimension. Shape-preserving operation."""
    ...

def masked_select(self: Tensor, mask: Tensor) -> Tensor[[Any]]:
    """Select elements with mask. Returns 1D tensor with data-dependent size."""
    ...

# ==== Phase 1.1: Missing Shape Operations ====

def unbind[Shape: IntTuple, Dim: Flag[builtins.int]](
    self: Tensor[Shape], dim: Dim = 0
) -> tuple[Tensor[unbind_shape(Shape, Dim)], ...]:
    """Remove dimension by slicing along it. Shape inference via meta-shape: torch.unbind"""
    ...

@overload
def movedim[
    Shape: IntTuple,
    Source: Flag[builtins.int],
    Destination: Flag[builtins.int],
](
    input: Tensor[Shape], source: Source, destination: Destination
) -> Tensor[movedim_scalar_shape(Shape, Source, Destination)]:
    """Move a single dimension to a new position."""
    ...

@overload
def movedim[Shape: IntTuple, Source: IntTuple, Destination: IntTuple](
    input: Tensor[Shape], source: Source, destination: Destination
) -> Tensor[movedim_tuple_shape(Shape, Source, Destination)]:
    """Move multiple dimensions to new positions. Shape inference via meta-shape: torch.movedim"""
    ...

@overload
def moveaxis[
    Shape: IntTuple,
    Source: Flag[builtins.int],
    Destination: Flag[builtins.int],
](
    input: Tensor[Shape], source: Source, destination: Destination
) -> Tensor[movedim_scalar_shape(Shape, Source, Destination)]:
    """Alias for movedim with a single source and destination."""
    ...

@overload
def moveaxis[Shape: IntTuple, Source: IntTuple, Destination: IntTuple](
    input: Tensor[Shape], source: Source, destination: Destination
) -> Tensor[movedim_tuple_shape(Shape, Source, Destination)]:
    """Alias for movedim. Shape inference via meta-shape: torch.moveaxis"""
    ...

# ==== Additional Reduction Functions ====

def all[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Check if all elements are True. Shape inference via meta-shape: torch.all"""
    ...

def any[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim = None, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Check if any element is True. Shape inference via meta-shape: torch.any"""
    ...

# ==== Phase 1.2: Missing Reduction Operations ====

@overload
def median[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[[]]:
    """Median of all elements (scalar). Shape inference via meta-shape: torch.median"""
    ...

@overload
def median[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
    input: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
) -> return_types.median[reduce_shape(Shape, Dim, Keepdim)]:
    """Median along dimension. Returns (values, indices). Shape inference via meta-shape: torch.median"""
    ...

def logsumexp[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...]],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Log-sum-exp along dimension(s). Shape inference via meta-shape: torch.logsumexp"""
    ...

def count_nonzero[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
](input: Tensor[Shape], dim: Dim = None) -> Tensor[reduce_shape_no_keep(Shape, Dim)]:
    """Count non-zero elements. Shape inference via meta-shape: torch.count_nonzero"""
    ...

def aminmax[
    Shape: IntTuple,
    Dim: Flag[builtins.int | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape], *, dim: Dim = None, keepdim: Keepdim = False
) -> return_types.aminmax[reduce_shape(Shape, Dim, Keepdim)]:
    """Min and max along dimension(s). Shape inference via meta-shape: torch.aminmax"""
    ...

def norm[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape],
    p: int | float = 2,
    dim: Dim = None,
    keepdim: Keepdim = False,
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]:
    """Compute norm. Shape inference via meta-shape: torch.norm"""
    ...

def dist(input: Tensor, other: Tensor, p: int | float = 2) -> Tensor[[]]:
    """Compute distance between tensors. Returns scalar tensor."""
    ...

def cumsum[Shape: IntTuple](input: Tensor[Shape], dim: int) -> Tensor[Shape]:
    """Cumulative sum along dimension. Shape-preserving operation."""
    ...

def cumprod[Shape: IntTuple](input: Tensor[Shape], dim: int) -> Tensor[Shape]:
    """Cumulative product along dimension. Shape-preserving operation."""
    ...

def cummax[Shape: IntTuple](
    input: Tensor[Shape], dim: int
) -> return_types.cummax[Shape]:
    """Cumulative maximum along dimension. Returns (values, indices). Shape-preserving operation."""
    ...

def cummin[Shape: IntTuple](
    input: Tensor[Shape], dim: int
) -> return_types.cummin[Shape]:
    """Cumulative minimum along dimension. Returns (values, indices). Shape-preserving operation."""
    ...

# Tier 2: Additional reduction operations (always return tuples)
def mode[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
    input: Tensor[Shape], dim: Dim = -1, keepdim: Keepdim = False
) -> return_types.mode[reduce_shape(Shape, Dim, Keepdim)]:
    """Mode along dimension. Returns (values, indices). Shape inference via meta-shape: torch.mode"""
    ...

def topk[Shape: IntTuple, K: _Int, Dim: Flag[builtins.int]](
    self: Tensor[Shape],
    k: K,
    dim: Dim = -1,
    largest: bool = True,
    sorted: bool = True,
) -> return_types.topk[topk_shape(Shape, Dim, K)]:
    """Top k elements. Returns (values, indices). Shape inference via meta-shape: torch.topk"""
    ...

def sort[Shape: IntTuple](
    input: Tensor[Shape], dim: int = -1, descending: bool = False, stable: bool = False
) -> return_types.sort[Shape]:
    """Sort tensor. Returns (values, indices). Shape-preserving operation."""
    ...

def kthvalue[Shape: IntTuple, Dim: Flag[builtins.int], Keepdim: Flag[builtins.bool]](
    input: Tensor[Shape], k: int, dim: Dim = -1, keepdim: Keepdim = False
) -> return_types.kthvalue[reduce_shape(Shape, Dim, Keepdim)]:
    """Kth smallest value. Returns (values, indices). Shape inference via meta-shape: torch.kthvalue"""
    ...

# Tier 3: Statistical operations returning tuples
@overload
def var_mean[Shape: IntTuple](
    input: Tensor[Shape], unbiased: builtins.bool = True
) -> tuple[Tensor[[]], Tensor[[]]]:
    """Variance and mean over all dimensions."""
    ...

@overload
def var_mean[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape],
    dim: Dim,
    unbiased: builtins.bool = True,
    keepdim: Keepdim = False,
) -> tuple[
    Tensor[reduce_shape(Shape, Dim, Keepdim)], Tensor[reduce_shape(Shape, Dim, Keepdim)]
]:
    """Variance and mean. Returns (var, mean). Shape inference via meta-shape: torch.var_mean"""
    ...

@overload
def std_mean[Shape: IntTuple](
    input: Tensor[Shape], unbiased: builtins.bool = True
) -> tuple[Tensor[[]], Tensor[[]]]:
    """Standard deviation and mean over all dimensions."""
    ...

@overload
def std_mean[
    Shape: IntTuple,
    Dim: Flag[builtins.int | tuple[builtins.int, ...] | None],
    Keepdim: Flag[builtins.bool],
](
    input: Tensor[Shape],
    dim: Dim,
    unbiased: builtins.bool = True,
    keepdim: Keepdim = False,
) -> tuple[
    Tensor[reduce_shape(Shape, Dim, Keepdim)], Tensor[reduce_shape(Shape, Dim, Keepdim)]
]:
    """Standard deviation and mean. Returns (std, mean). Shape inference via meta-shape: torch.std_mean"""
    ...

# ==== Phase 1.3: Tensor Creation Operations ====

def zeros_like[Shape: IntTuple](
    input: Tensor[Shape],
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
    memory_format: Any = None,
) -> Tensor[Shape]:
    """Create zeros with same shape. Shape inference via generic fixture signature."""
    ...

def ones_like[Shape: IntTuple](
    input: Tensor[Shape],
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
    memory_format: Any = None,
) -> Tensor[Shape]:
    """Create ones with same shape. Shape inference via generic fixture signature."""
    ...

def full_like[Shape: IntTuple](
    input: Tensor[Shape], fill_value: float
) -> Tensor[Shape]:
    """Create tensor with same shape filled with value. Shape inference via generic fixture signature."""
    ...

def empty_like[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Create uninitialized tensor with same shape. Shape inference via generic fixture signature."""
    ...

def rand_like[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Create random tensor [0,1) with same shape. Shape inference via generic fixture signature."""
    ...

def randn_like[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Create random normal tensor with same shape. Shape inference via generic fixture signature."""
    ...

def diag_embed[
    Shape: IntTuple,
    Offset: Flag[builtins.int],
    Dim1: Flag[builtins.int],
    Dim2: Flag[builtins.int],
](
    self: Tensor[Shape], offset: Offset = 0, dim1: Dim1 = -2, dim2: Dim2 = -1
) -> Tensor[diag_embed_shape(Shape, Offset, Dim1, Dim2)]:
    """Create diagonal tensor. Shape inference via meta-shape: torch.diag_embed"""
    ...

def tril[Shape: IntTuple](input: Tensor[Shape], diagonal: int = 0) -> Tensor[Shape]:
    """Lower triangular part. Shape inference via generic fixture signature."""
    ...

def triu[Shape: IntTuple](input: Tensor[Shape], diagonal: int = 0) -> Tensor[Shape]:
    """Upper triangular part. Shape inference via generic fixture signature."""
    ...

def tril_indices(
    row: builtins.int, col: builtins.int, offset: builtins.int = 0
) -> Tensor[[2, Any]]:
    """Indices of the lower triangular part. The count depends on the argument values."""
    ...

def triu_indices(
    row: builtins.int, col: builtins.int, offset: builtins.int = 0
) -> Tensor[[2, Any]]:
    """Indices of the upper triangular part. The count depends on the argument values."""
    ...

# ==== Phase 1.4: Basic Linear Algebra Operations ====

# Note: matmul is already defined above with static typing at line 341
# We keep it there for backward compatibility, but meta-shape handles general cases

def mm[N: IntVar, K: IntVar, M: IntVar](
    input: Tensor[[N, K]], mat2: Tensor[[K, M]]
) -> Tensor[[N, M]]:
    """Matrix multiplication (2D @ 2D). Output: [N, M]."""
    ...

def bmm[B: IntVar, N: IntVar, K: IntVar, M: IntVar](
    input: Tensor[[B, N, K]], mat2: Tensor[[B, K, M]]
) -> Tensor[[B, N, M]]:
    """Batch matrix multiplication (3D @ 3D). Output: [B, N, M]."""
    ...

def mv[M: IntVar, K: IntVar](input: Tensor[[M, K]], vec: Tensor[[K]]) -> Tensor[[M]]:
    """Matrix-vector multiplication (2D @ 1D). Output: [M]."""
    ...

def dot[N: IntVar](input: Tensor[[N]], other: Tensor[[N]]) -> Tensor[[]]:
    """Dot product (1D @ 1D → scalar). Returns scalar tensor."""
    ...

# ==== Phase 2: Arithmetic & Basic Operations ====
# All operations preserve shape (use IdentityMetaShape)

# Arithmetic operations (element-wise)
def add[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise addition. Shape inference via generic fixture signature."""
    ...

def sub[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise subtraction. Shape inference via generic fixture signature."""
    ...

def mul[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise multiplication. Shape inference via generic fixture signature."""
    ...

def div[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape],
    other: _TensorLike[OtherShape],
    *,
    rounding_mode: str | None = None,
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise division. Shape inference via generic fixture signature."""
    ...

@overload
def pow[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], exponent: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise power. Shape inference via generic fixture signature."""
    ...

@overload
def pow[Shape: IntTuple](
    input: builtins.float | builtins.int, exponent: Tensor[Shape]
) -> Tensor[Shape]: ...
def neg[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise negation. Shape inference via generic fixture signature."""
    ...

def abs[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise absolute value. Shape inference via generic fixture signature."""
    ...

def floor[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise floor. Shape inference via generic fixture signature."""
    ...

def ceil[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise ceiling. Shape inference via generic fixture signature."""
    ...

def round[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise rounding. Shape inference via generic fixture signature."""
    ...

# Point-wise mathematical operations
def sin[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise sine. Shape inference via generic fixture signature."""
    ...

def cos[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise cosine. Shape inference via generic fixture signature."""
    ...

def tan[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise tangent. Shape inference via generic fixture signature."""
    ...

def exp[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise exponential. Shape inference via generic fixture signature."""
    ...

def log[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise natural logarithm. Shape inference via generic fixture signature."""
    ...

def sqrt[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise square root. Shape inference via generic fixture signature."""
    ...

def tanh[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise hyperbolic tangent. Shape inference via generic fixture signature."""
    ...

def sigmoid[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise sigmoid. Shape inference via generic fixture signature."""
    ...

# Comparison operations
def eq[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise equality. Shape inference via generic fixture signature."""
    ...

def ne[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise inequality. Shape inference via generic fixture signature."""
    ...

def lt[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _RealTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise less than. Shape inference via generic fixture signature."""
    ...

def le[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _RealTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise less than or equal. Shape inference via generic fixture signature."""
    ...

def gt[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _RealTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise greater than. Shape inference via generic fixture signature."""
    ...

def ge[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _RealTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise greater than or equal. Shape inference via generic fixture signature."""
    ...

# Logical operations
def logical_and[Shape: IntTuple, OtherShape: IntTuple](
    input: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise logical AND. Shape inference via generic fixture signature."""
    ...

def logical_or[Shape: IntTuple, OtherShape: IntTuple](
    input: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise logical OR. Shape inference via generic fixture signature."""
    ...

def logical_not[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise logical NOT. Shape inference via generic fixture signature."""
    ...

# Clamping
def clamp[Shape: IntTuple](
    input: Tensor[Shape],
    min: Tensor | builtins.float | builtins.int | None = None,
    max: Tensor | builtins.float | builtins.int | None = None,
) -> Tensor[Shape]:
    """Clamp tensor values. Shape inference via generic fixture signature."""
    ...

def clip[Shape: IntTuple](
    input: Tensor[Shape],
    min: Tensor | builtins.float | builtins.int | None = None,
    max: Tensor | builtins.float | builtins.int | None = None,
) -> Tensor[Shape]:
    """Alias for clamp. Shape inference via generic fixture signature."""
    ...

# Activation functions (relu is most common, others in torch.nn.functional)
def relu[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """ReLU activation. Shape inference via generic fixture signature."""
    ...

# Additional mathematical operations
def atan2[Shape: IntTuple, OtherShape: IntTuple](
    input: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise arctangent of input/other. Shape inference via generic fixture signature."""
    ...

def hypot[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise hypotenuse. Shape inference via generic fixture signature."""
    ...

def asin[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise arcsine. Shape inference via generic fixture signature."""
    ...

def arcsin[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise arcsine (alias of asin). Shape inference via generic fixture signature."""
    ...

def atan[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise arctangent. Shape inference via generic fixture signature."""
    ...

def lerp[Shape: IntTuple](
    input: Tensor[Shape], end: Tensor, weight: float
) -> Tensor[Shape]:
    """Linear interpolation. Shape inference via generic fixture signature."""
    ...

def fmod[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise modulo. Shape inference via generic fixture signature."""
    ...

def remainder[Shape: IntTuple](
    input: Tensor[Shape], other: Tensor | int | float
) -> Tensor[Shape]:
    """Element-wise remainder. Shape inference via generic fixture signature."""
    ...

def copysign[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Copy sign. Shape inference via generic fixture signature."""
    ...

def nextafter[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Next floating-point value. Shape inference via generic fixture signature."""
    ...

def erf[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Error function. Shape inference via generic fixture signature."""
    ...

def erfc[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Complementary error function. Shape inference via generic fixture signature."""
    ...

def erfinv[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Inverse error function. Shape inference via generic fixture signature."""
    ...

def lgamma[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Log gamma function. Shape inference via generic fixture signature."""
    ...

def digamma[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Digamma function. Shape inference via generic fixture signature."""
    ...

def polygamma[Shape: IntTuple](n: int, input: Tensor[Shape]) -> Tensor[Shape]:
    """Polygamma function. Shape inference via generic fixture signature."""
    ...

def asinh[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Inverse hyperbolic sine. Shape inference via generic fixture signature."""
    ...

def acosh[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Inverse hyperbolic cosine. Shape inference via generic fixture signature."""
    ...

def atanh[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Inverse hyperbolic tangent. Shape inference via generic fixture signature."""
    ...

def deg2rad[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Convert degrees to radians. Shape inference via generic fixture signature."""
    ...

def rad2deg[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Convert radians to degrees. Shape inference via generic fixture signature."""
    ...

# Bitwise operations
def bitwise_and[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _IntegerTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Bitwise AND. Shape inference via generic fixture signature."""
    ...

def equal(input: Tensor, other: Tensor) -> builtins.bool:
    """Return whether two tensors have the same size and elements."""
    ...

def bitwise_or[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _IntegerTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Bitwise OR. Shape inference via generic fixture signature."""
    ...

def bitwise_xor[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _IntegerTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Bitwise XOR. Shape inference via generic fixture signature."""
    ...

def bitwise_not[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Bitwise NOT. Shape inference via generic fixture signature."""
    ...

def bitwise_left_shift[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _IntegerTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Bitwise left shift. Shape inference via generic fixture signature."""
    ...

def bitwise_right_shift[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _IntegerTensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Bitwise right shift. Shape inference via generic fixture signature."""
    ...

# Additional comparison/validation operations
def isclose[Shape: IntTuple](
    input: Tensor[Shape], other: Tensor, rtol: float = 1e-05, atol: float = 1e-08
) -> Tensor[Shape]:
    """Check if tensors are close. Shape inference via generic fixture signature."""
    ...

def isreal[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Check if elements are real. Shape inference via generic fixture signature."""
    ...

def isposinf[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Check if elements are positive infinity. Shape inference via generic fixture signature."""
    ...

def isneginf[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Check if elements are negative infinity. Shape inference via generic fixture signature."""
    ...

def isfinite[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Check if elements are finite. Shape inference via generic fixture signature."""
    ...

def isnan[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Check if elements are NaN."""
    ...

def argsort[Shape: IntTuple](
    input: Tensor[Shape],
    dim: builtins.int = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = False,
) -> Tensor[Shape]: ...
def diagonal[
    Shape: IntTuple,
    Offset: Flag[builtins.int] = 0,
    Dim1: Flag[builtins.int] = 0,
    Dim2: Flag[builtins.int] = 1,
](
    input: Tensor[Shape], offset: Offset = 0, dim1: Dim1 = 0, dim2: Dim2 = 1
) -> Tensor[diagonal_shape(Shape, Offset, Dim1, Dim2)]: ...
def quantile(
    input: Tensor,
    q: builtins.float | Tensor,
    dim: builtins.int | None = None,
    keepdim: builtins.bool = False,
    *,
    interpolation: str = "linear",
    out: Tensor | None = None,
) -> Tensor: ...
def allclose(
    input: Tensor,
    other: Tensor,
    rtol: builtins.float = 1e-05,
    atol: builtins.float = 1e-08,
    equal_nan: builtins.bool = False,
) -> builtins.bool: ...
def maximum[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise maximum. Shape inference via generic fixture signature."""
    ...

def minimum[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise minimum. Shape inference via generic fixture signature."""
    ...

def expm1[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]: ...
def log10[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]: ...
def sign[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]: ...
def fmax[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise maximum (NaN handling). Shape inference via generic fixture signature."""
    ...

def fmin[Shape: IntTuple, OtherShape: IntTuple = []](
    input: Tensor[Shape], other: _TensorLike[OtherShape]
) -> Tensor[broadcast(Shape, OtherShape)]:
    """Element-wise minimum (NaN handling). Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 4: Advanced Linear Algebra Operations
# ==============================================================================

# Advanced matmul operations
@overload
def tensordot[Left: IntTuple, Right: IntTuple, Dims: Flag[builtins.int]](
    self: Tensor[Left], other: Tensor[Right], dims: Dims = 2
) -> Tensor[tensordot_shape(Left, Right, Dims)]:
    """Tensor contraction over specified dimensions. Shape inference via meta-shape: torch.tensordot"""
    ...

@overload
def tensordot(self: Tensor, other: Tensor, dims: tuple[list[int], list[int]]) -> Tensor:
    """Tensor contraction over explicit axis lists.

    TODO(stroxler): Preserve the result shape once the V2 DSL accepts structured axis lists.
    """
    ...

def einsum[Spec: Flag[str], Shapes: IntTuples](
    spec: Spec, *operands: Unpack[MapIntTuples[lambda S: Tensor[S], Shapes]]
) -> Tensor[einsum_shape(Spec, Shapes)]:
    """Einstein summation convention. Shape inference via type-level DSL."""
    ...

# Eigenvalue decomposition
@overload
def eig[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]], eigenvectors: bool = False
) -> tuple[Tensor[[*Elements[Batch], M]], Tensor[[*Elements[Batch], M, N]]]: ...
@overload
def eig[Shape: IntTuple](
    self: Tensor[Shape], eigenvectors: bool = False
) -> tuple[Tensor[eig_shape(Shape)], Tensor[Shape]]: ...
@overload
def eigh[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]], UPLO: str = "L"
) -> tuple[Tensor[[*Elements[Batch], M]], Tensor[[*Elements[Batch], M, N]]]: ...
@overload
def eigh[Shape: IntTuple](
    self: Tensor[Shape], UPLO: str = "L"
) -> tuple[Tensor[eig_shape(Shape)], Tensor[Shape]]: ...

# Cholesky decomposition
def cholesky[Shape: IntTuple](
    input: Tensor[Shape], upper: bool = False
) -> Tensor[Shape]:
    """Cholesky decomposition. Shape inference via generic fixture signature."""
    ...

# Linear system solvers
def solve[Shape: IntTuple, OtherShape: IntTuple](
    self: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[Shape]: ...
def triangular_solve[Shape: IntTuple, OtherShape: IntTuple](
    self: Tensor[Shape], other: Tensor[OtherShape], upper: bool = True
) -> Tensor[Shape]: ...
def cholesky_solve[Shape: IntTuple, OtherShape: IntTuple](
    self: Tensor[Shape], other: Tensor[OtherShape], upper: bool = False
) -> Tensor[Shape]: ...
def lu_solve[Shape: IntTuple, OtherShape: IntTuple, PivotShape: IntTuple](
    self: Tensor[Shape],
    other: Tensor[OtherShape],
    LU_pivots: Tensor[PivotShape],
) -> Tensor[Shape]: ...

# Matrix inverse
def inverse[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Matrix inverse. Shape inference via generic fixture signature."""
    ...

# Determinant
def det[Batch: IntTuple, M: IntVar, N: IntVar](
    input: Tensor[[*Elements[Batch], M, N]],
) -> Tensor[Batch]:
    """Determinant. Returns batch dimensions only (drops last 2 dims)."""
    ...

def logdet[Batch: IntTuple, M: IntVar, N: IntVar](
    input: Tensor[[*Elements[Batch], M, N]],
) -> Tensor[Batch]:
    """Log determinant. Returns batch dimensions only (drops last 2 dims)."""
    ...

@overload
def slogdet[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]],
) -> return_types.slogdet[Batch]: ...
@overload
def slogdet[Shape: IntTuple](
    self: Tensor[Shape],
) -> return_types.slogdet[slogdet_shape(Shape)]: ...

# Matrix power and exponential
def matrix_power[Shape: IntTuple](input: Tensor[Shape], n: int) -> Tensor[Shape]:
    """Matrix power. Shape inference via generic fixture signature."""
    ...

def matrix_exp[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Matrix exponential. Shape inference via generic fixture signature."""
    ...

# Trace
def trace[Batch: IntTuple, M: IntVar, N: IntVar](
    input: Tensor[[*Elements[Batch], M, N]],
) -> Tensor[Batch]:
    """Matrix trace. Returns batch dimensions only (drops last 2 dims)."""
    ...

# Matrix rank
def matrix_rank[Batch: IntTuple, M: IntVar, N: IntVar](
    input: Tensor[[*Elements[Batch], M, N]], tol: float = None, symmetric: bool = False
) -> Tensor[Batch]:
    """Matrix rank. Returns batch dimensions only (drops last 2 dims)."""
    ...

# ==============================================================================
# Phase 5: Advanced Indexing & Conditional Operations
# ==============================================================================

# Conditional operations
# TODO(stroxler): Infer the broadcast result shape.
@overload
def where(condition: Tensor[Any]) -> tuple[Tensor, ...]: ...
@overload
def where[
    ConditionShape: IntTuple,
    InputShape: IntTuple,
    OtherShape: IntTuple,
](
    condition: Tensor[ConditionShape],
    input: Tensor[InputShape],
    other: Tensor[OtherShape],
    *,
    out: Tensor | None = None,
) -> Tensor[broadcast(ConditionShape, broadcast(InputShape, OtherShape))]: ...
@overload
def where[ConditionShape: IntTuple, InputShape: IntTuple](
    condition: Tensor[ConditionShape],
    input: Tensor[InputShape],
    other: builtins.bool | builtins.int | builtins.float | builtins.complex,
) -> Tensor[broadcast(ConditionShape, InputShape)]: ...

# PyTorch names a scalar value parameter `self`, not `input`.
@overload
def where[ConditionShape: IntTuple, OtherShape: IntTuple](
    condition: Tensor[ConditionShape],
    self: builtins.bool | builtins.int | builtins.float | builtins.complex,
    other: Tensor[OtherShape],
) -> Tensor[broadcast(ConditionShape, OtherShape)]: ...
@overload
def where[ConditionShape: IntTuple](
    condition: Tensor[ConditionShape],
    self: builtins.bool | builtins.int | builtins.float | builtins.complex,
    other: builtins.bool | builtins.int | builtins.float | builtins.complex,
) -> Tensor[ConditionShape]: ...
def masked_fill[Shape: IntTuple, MaskShape: IntTuple](
    input: Tensor[Shape], mask: Tensor[MaskShape], value: float
) -> Tensor[broadcast(Shape, MaskShape)]:
    """Fill masked elements. Shape inference via generic fixture signature."""
    ...

def masked_scatter[Shape: IntTuple, MaskShape: IntTuple](
    input: Tensor[Shape], mask: Tensor[MaskShape], source: Tensor
) -> Tensor[broadcast(Shape, MaskShape)]:
    """Scatter into masked positions. Shape inference via generic fixture signature."""
    ...

# Advanced indexing operations
def index_add[
    Shape: IntTuple,
    Dim: Flag[builtins.int],
    IndexShape: IntTuple,
    SourceShape: IntTuple,
](
    input: Tensor[Shape],
    dim: Dim,
    index: Tensor[IndexShape],
    source: Tensor[SourceShape],
    alpha: float = 1,
) -> Tensor[indexed_source_shape(Shape, Dim, IndexShape, SourceShape)]:
    """Add values at indices. Shape inference via generic fixture signature."""
    ...

def index_copy[
    Shape: IntTuple,
    Dim: Flag[builtins.int],
    IndexShape: IntTuple,
    SourceShape: IntTuple,
](
    input: Tensor[Shape],
    dim: Dim,
    index: Tensor[IndexShape],
    source: Tensor[SourceShape],
) -> Tensor[indexed_source_shape(Shape, Dim, IndexShape, SourceShape)]:
    """Copy values to indices. Shape inference via generic fixture signature."""
    ...

def index_put[Shape: IntTuple](
    input: Tensor[Shape],
    indices: tuple[Tensor, ...],
    values: Tensor,
    accumulate: bool = False,
) -> Tensor[Shape]:
    """Put values at indices. Shape inference via generic fixture signature."""
    ...

def index_fill[Shape: IntTuple, Dim: Flag[builtins.int], IndexShape: IntTuple](
    input: Tensor[Shape], dim: Dim, index: Tensor[IndexShape], value: float
) -> Tensor[index_fill_shape(Shape, Dim, IndexShape)]:
    """Fill indices with value. Shape inference via generic fixture signature."""
    ...

# Take/put operations
def take[Shape: IntTuple, IndexShape: IntTuple](
    input: Tensor[Shape], index: Tensor[IndexShape]
) -> Tensor[take_shape(Shape, IndexShape)]:
    """Take elements at indices. Output shape matches index shape."""
    ...

def take_along_dim[
    Shape: IntTuple,
    IndexShape: IntTuple,
    Dim: Flag[builtins.int | None],
](
    self: Tensor[Shape], indices: Tensor[IndexShape], dim: Dim = None
) -> Tensor[take_along_dim_shape(Shape, IndexShape, Dim)]: ...
def put[Shape: IntTuple, IndexShape: IntTuple, SourceShape: IntTuple](
    input: Tensor[Shape],
    index: Tensor[IndexShape],
    source: Tensor[SourceShape],
    accumulate: bool = False,
) -> Tensor[put_shape(Shape, IndexShape, SourceShape)]:
    """Put values at indices. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 6: Specialized Operations
# ==============================================================================

# Random sampling operations
def bernoulli[Shape: IntTuple](input: Tensor[Shape], p: float = 0.5) -> Tensor[Shape]:
    """Sample from Bernoulli distribution. Shape inference via generic fixture signature."""
    ...

@overload
def multinomial[Shape: IntTuple, NumSamples: _Int](
    input: Tensor[Shape],
    num_samples: NumSamples,
    replacement: Literal[False] = False,
    *,
    generator: Generator | None = None,
) -> Tensor[multinomial_shape(Shape, NumSamples, False)]: ...
@overload
def multinomial[Shape: IntTuple, NumSamples: _Int](
    input: Tensor[Shape],
    num_samples: NumSamples,
    replacement: Literal[True],
    *,
    generator: Generator | None = None,
) -> Tensor[multinomial_shape(Shape, NumSamples, True)]: ...
@overload
def multinomial[Shape: IntTuple, NumSamples: _Int](
    input: Tensor[Shape],
    num_samples: NumSamples,
    replacement: builtins.bool,
    *,
    generator: Generator | None = None,
) -> Tensor[multinomial_shape(Shape, NumSamples, True)]:
    # A non-literal flag might permit replacement, so the return type must not
    # apply the without-replacement upper bound.
    """Sample from multinomial distribution. Shape inference via meta-shape: torch.multinomial"""
    ...

@overload
def normal[MeanShape: IntTuple, StdShape: IntTuple](
    mean: Tensor[MeanShape], std: Tensor[StdShape]
) -> Tensor[broadcast(MeanShape, StdShape)]:
    """Sample from a normal distribution. Tensor parameters are broadcast."""
    ...

@overload
def normal[Shape: IntTuple](mean: Tensor[Shape], std: float) -> Tensor[Shape]:
    """Sample from a normal distribution. The output has the mean tensor's shape."""
    ...

@overload
def normal[Shape: IntTuple](mean: float, std: Tensor[Shape]) -> Tensor[Shape]:
    """Sample from a normal distribution. The output has the standard-deviation tensor's shape."""
    ...

@overload
def normal[Shape: IntTuple](mean: float, std: float, size: Shape) -> Tensor[Shape]:
    """Sample from a normal distribution. Shape is inferred from `size`."""
    ...

def poisson[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Sample from Poisson distribution. Shape inference via generic fixture signature."""
    ...

# Tensor property functions
def numel[Shape: IntTuple](input: Tensor[Shape]) -> _Int[numel_shape(Shape)]:
    """Return the number of elements."""
    ...

# ==============================================================================
# Data Types and Context Managers
# ==============================================================================

# Data type constants
__version__: str
long: Any = ...  # torch.long dtype constant
float32: Any = ...  # torch.float32 dtype constant
float64: Any = ...  # torch.float64 dtype constant
bfloat16: Any = ...  # torch.bfloat16 dtype constant
int32: Any = ...  # torch.int32 dtype constant
int64: Any = ...  # torch.int64 dtype constant

pi: float = ...  # torch.pi value constant
inf: float = ...  # torch.inf value constant

# dtype type (for type annotations)
class dtype:
    """PyTorch data type."""

    ...

# ==============================================================================
# Tensor Creation with dtype support
# ==============================================================================

@overload
def tensor(
    data: _TensorScalar,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
    pin_memory: builtins.bool = False,
) -> Tensor[[]]: ...
@overload
def tensor[Shape: IntTuple = []](
    data: Tensor[Shape] | RegularNestedList[Shape, _TensorScalar],
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
    pin_memory: builtins.bool = False,
) -> Tensor[Shape]: ...
@overload
def tensor(
    data: Any,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: builtins.bool = False,
    pin_memory: builtins.bool = False,
) -> Tensor[IntTuple]:
    """Create a tensor from data, preserving or inferring its shape when possible."""
    ...

def as_tensor(
    data: Any,
    dtype: Any = None,
    device: Any = None,
) -> Tensor:
    """Create tensor from data, sharing memory with the input when possible."""
    ...

def from_numpy(ndarray: Any) -> Tensor:
    """Create a CPU tensor that shares memory with a numpy array."""
    ...

def randint[Shape: IntTuple](
    low: int,
    high: int,
    size: Shape,
    *,
    generator: Any = None,
    dtype: Any = None,
    device: Any = None,
    requires_grad: bool = False,
) -> Tensor[Shape]:
    """Create a tensor of random integers. Shape is inferred from `size`."""
    ...

def randperm(
    n: int,
    *,
    generator: Any = None,
    dtype: Any = None,
    device: Any = None,
) -> Tensor:
    """Create a random permutation of integers. Returns shapeless tensor (shape depends on n)."""
    ...

# ==============================================================================
# Additional Math Operations
# ==============================================================================

def rsqrt[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Reciprocal square root (1/sqrt(x)). Shape-preserving element-wise operation."""
    ...

def outer[M: IntVar, N: IntVar](
    input: Tensor[[M]], vec2: Tensor[[N]]
) -> Tensor[[M, N]]:
    """Outer product of two 1D tensors. Output: [M, N]."""
    ...

def polar[Shape: IntTuple](abs: Tensor[Shape], angle: Tensor[Shape]) -> Tensor[Shape]:
    """Construct complex tensor from polar coordinates. Shape-preserving operation."""
    ...

def view_as_complex[S: IntTuple](input: Tensor[[*Elements[S], 2]]) -> Tensor[S]:
    """View a real tensor as complex. Last dim of size 2 is consumed."""
    ...

def view_as_real[S: IntTuple](input: Tensor[S]) -> Tensor[[*Elements[S], 2]]:
    """View a complex tensor as real. Appends trailing dim of size 2."""
    ...

def hann_window[N: IntVar](
    window_length: _Int[N],
    periodic: bool = True,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Tensor[[N]]:
    """Create a Hann window tensor of size (window_length,)."""
    ...

def stft[Batch: IntTuple, F: IntVar](
    input: Tensor[Batch],
    n_fft: _Int[F],
    hop_length: int | None = None,
    win_length: int | None = None,
    window: Tensor | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool | None = None,
    return_complex: bool | None = None,
) -> Tensor[[*Elements[Batch], F // 2 + 1, int]]:
    """Short-time Fourier transform.

    Input: (*Batch, L) — signal (1D or batched).
    Output: (*Batch, n_fft // 2 + 1, n_frames).
    Frequency bins = n_fft // 2 + 1 (deterministic from n_fft).
    Time frames depends on input length, hop_length, center — not tracked.
    """
    ...

def addmm[N: IntVar, K: IntVar, M: IntVar](
    input: Tensor[[N, M]],
    mat1: Tensor[[N, K]],
    mat2: Tensor[[K, M]],
    *,
    beta: float = 1,
    alpha: float = 1,
) -> Tensor[[N, M]]:
    """Matrix multiply with add: beta * input + alpha * (mat1 @ mat2)."""
    ...

def cross[B: IntTuple](
    input: Tensor[[*Elements[B], 3]],
    other: Tensor[[*Elements[B], 3]],
    dim: int = -1,
) -> Tensor[[*Elements[B], 3]]:
    """Cross product of two tensors along a dimension of size 3."""
    ...

# Context managers
class no_grad:
    """Context manager and decorator that disables gradient tracking.

    Usage:
        # As context manager:
        with torch.no_grad():
            output = model(input)

        # As decorator, with or without parentheses:
        @torch.no_grad()
        def inference(x):
            return model(x)
    """

    # Mirrors `_NoParamDecoratorContextManager.__new__`: a bare `@torch.no_grad`
    # passes the function to the constructor and gets it back.
    @overload
    def __new__[F: Callable[..., Any]](cls, orig_func: F) -> F: ...
    @overload
    def __new__(cls, orig_func: None = None) -> Self: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]: ...

class inference_mode:
    def __init__(self, mode: builtins.bool = True) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]: ...

class OutOfMemoryError(RuntimeError): ...

def meshgrid(*tensors: Tensor, indexing: str = "ij") -> tuple[Tensor, ...]:
    """Create coordinate grids from 1D input tensors.

    For N input tensors, returns N tensors each with N dimensions.
    Shape inference depends on input tensor shapes; returns shapeless tuple.
    """
    ...

# The functions below carry no shape information. They are declared because this
# module shadows torch's own `__init__`, as described at the `torch._C` import
# above, so omitting them removes them from `torch` for every dependent target.
def manual_seed(seed: int) -> Generator:
    """Set the seed for generating random numbers on all devices."""
    ...

def save(
    obj: object,
    f: Any,
    pickle_module: Any = ...,
    pickle_protocol: int = ...,
    _use_new_zipfile_serialization: bool = True,
) -> None:
    """Save an object to a file."""
    ...

def load(
    f: Any,
    map_location: Any = None,
    pickle_module: Any = None,
    *,
    weights_only: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Load an object saved with `torch.save`."""
    ...

# Re-export the public submodules `torch/__init__.py` imports, so `torch.cuda.x`
# resolves without an explicit `import torch.cuda`.
from torch import (
    accelerator as accelerator,
    amp as amp,
    autograd as autograd,
    backends as backends,
    cpu as cpu,
    cuda as cuda,
    distributed as distributed,
    distributions as distributions,
    export as export,
    fft as fft,
    func as func,
    futures as futures,
    hub as hub,
    jit as jit,
    library as library,
    linalg as linalg,
    mps as mps,
    mtia as mtia,
    multiprocessing as multiprocessing,
    nested as nested,
    nn as nn,
    optim as optim,
    overrides as overrides,
    profiler as profiler,
    random as random,
    return_types as return_types,
    serialization as serialization,
    sparse as sparse,
    special as special,
    testing as testing,
    types as types,
    utils as utils,
    version as version,
    xpu as xpu,
)
