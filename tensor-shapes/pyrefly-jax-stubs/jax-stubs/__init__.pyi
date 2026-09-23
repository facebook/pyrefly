# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Any, Callable

# Override jax.Array with shape-aware version.
from jax._array import Array as Array

# Override jit and vmap because jax annotations fail with shaped arrays.
def jit[F: Callable[..., Any]](fun: F, *args: Any, **kwargs: Any) -> F: ...
def vmap(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]: ...

# Remaining APIs mirror imports in jax/__init__.py,
# generated here based on jax v0.11.2
from jax import (
    api_util as api_util,
    custom_batching as custom_batching,
    custom_derivatives as custom_derivatives,
    debug as debug,
    distributed as distributed,
    dlpack as dlpack,
    dtypes as dtypes,
    errors as errors,
    experimental as experimental,
    export as export,
    ffi as ffi,
    image as image,
    interpreters as interpreters,
    lib as lib,
    monitoring as monitoring,
    ops as ops,
    profiler as profiler,
    random as random,
    scipy as scipy,
    sharding as sharding,
    stages as stages,
    tree as tree,
    tree_util as tree_util,
    typing as typing,
)
from jax._src.ad_checkpoint import (
    checkpoint as checkpoint,
    checkpoint_policies as checkpoint_policies,
    custom_remat as custom_remat,
    remat as remat,
)
from jax._src.api import (
    block_until_ready as block_until_ready,
    clear_caches as clear_caches,
    copy_to_host_async as copy_to_host_async,
    device_get as device_get,
    device_put as device_put,
    device_put_replicated as device_put_replicated,
    device_put_sharded as device_put_sharded,
    disable_jit as disable_jit,
    effects_barrier as effects_barrier,
    eval_shape as eval_shape,
    fwd_and_bwd as fwd_and_bwd,
    grad as grad,
    hessian as hessian,
    Inline as Inline,
    jacfwd as jacfwd,
    jacobian as jacobian,
    jacrev as jacrev,
    jvp as jvp,
    linear_transpose as linear_transpose,
    linearize as linearize,
    live_arrays as live_arrays,
    make_jaxpr as make_jaxpr,
    named_call as named_call,
    named_scope as named_scope,
    value_and_grad as value_and_grad,
    vjp as vjp,
)
from jax._src.array import (
    make_array_from_callback as make_array_from_callback,
    make_array_from_process_local_data as make_array_from_process_local_data,
    make_array_from_single_device_arrays as make_array_from_single_device_arrays,
    Shard as Shard,
)
from jax._src.callback import pure_callback as pure_callback
from jax._src.compiler import CompilerEffortLevel as CompilerEffortLevel
from jax._src.config import (
    allow_f16_reductions as allow_f16_reductions,
    array_garbage_collection_guard as array_garbage_collection_guard,
    auto_pcast as auto_pcast,
    check_tracer_leaks as check_tracer_leaks,
    checking_leaks as checking_leaks,
    config as config,
    debug_infs as debug_infs,
    debug_key_reuse as debug_key_reuse,
    debug_nans as debug_nans,
    default_device as default_device,
    default_matmul_precision as default_matmul_precision,
    default_prng_impl as default_prng_impl,
    enable_checks as enable_checks,
    enable_custom_prng as enable_custom_prng,
    enable_x64 as enable_x64,
    explain_cache_misses as explain_cache_misses,
    jax2tf_associative_scan_reductions as jax2tf_associative_scan_reductions,
    legacy_prng_key as legacy_prng_key,
    log_compiles as log_compiles,
    make_user_context as make_user_context,
    no_execution as no_execution,
    no_tracing as no_tracing,
    numpy_dtype_promotion as numpy_dtype_promotion,
    numpy_rank_promotion as numpy_rank_promotion,
    remove_size_one_mesh_axis_from_type as remove_size_one_mesh_axis_from_type,
    softmax_custom_jvp as softmax_custom_jvp,
    thread_guard as thread_guard,
    threefry_partitionable as threefry_partitionable,
    transfer_guard as transfer_guard,
    transfer_guard_device_to_device as transfer_guard_device_to_device,
    transfer_guard_device_to_host as transfer_guard_device_to_host,
    transfer_guard_host_to_device as transfer_guard_host_to_device,
)
from jax._src.core import (
    ensure_compile_time_eval as ensure_compile_time_eval,
    ShapeDtypeStruct as ShapeDtypeStruct,
    typeof as typeof,
)
from jax._src.custom_derivatives import (
    closure_convert as closure_convert,
    custom_gradient as custom_gradient,
    custom_jvp as custom_jvp,
    custom_vjp as custom_vjp,
)
from jax._src.dtypes import float0 as float0
from jax._src.environment_info import print_environment_info as print_environment_info
from jax._src.indexing import ds as ds
from jax._src.mesh import (
    get_abstract_mesh as get_abstract_mesh,
    use_abstract_mesh as use_abstract_mesh,
)
from jax._src.partition_spec import P as P
from jax._src.pjit import reshard as reshard
from jax._src.pmap import pmap as pmap
from jax._src.shard_map import (
    shard_map as shard_map,
    smap as smap,
)
from jax._src.sharding_impls import (
    get_mesh as get_mesh,
    make_mesh as make_mesh,
    NamedSharding as NamedSharding,
    set_mesh as set_mesh,
)
from jax._src.xla_bridge import (
    default_backend as default_backend,
    device_count as device_count,
    devices as devices,
    host_count as host_count,
    host_id as host_id,
    host_ids as host_ids,
    local_device_count as local_device_count,
    local_devices as local_devices,
    process_count as process_count,
    process_index as process_index,
    process_indices as process_indices,
)
from jax.ref import (
    empty_ref as empty_ref,
    free_ref as free_ref,
    freeze as freeze,
    new_ref as new_ref,
    Ref as Ref,
)
from jax.version import (
    __version__ as __version__,
    __version_info__ as __version_info__,
)
from jaxlib._jax import Device as Device

# Import local typestubs for submodules defined in pyrefly-jax-stubs
from . import (
    lax as lax,
    nn as nn,
    numpy as numpy,
)

ad: SimpleNamespace
memory: SimpleNamespace
