# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from jax._array import Array
from jax._src.core import ManualAxisType as ManualAxisType
from jax._src.mesh import (
    AbstractDevice as AbstractDevice,
    AbstractMesh as AbstractMesh,
    AxisType as AxisType,
    get_abstract_mesh as get_abstract_mesh,
    Mesh as Mesh,
    use_abstract_mesh as use_abstract_mesh,
)
from jax._src.partition_spec import (
    PartitionSpec as PartitionSpec,
    UnreducedKind as UnreducedKind,
)
from jax._src.pjit import (
    auto_axes as auto_axes,
    explicit_axes as explicit_axes,
)
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    get_mesh as get_mesh,
    make_single_device_sharding as make_single_device_sharding,
    NamedSharding as NamedSharding,
    set_mesh as set_mesh,
    SingleDeviceSharding as SingleDeviceSharding,
)
from shape_extensions import IntTuple

def reshard[Shape: IntTuple](
    xs: Array[Shape], out_shardings: NamedSharding | PartitionSpec
) -> Array[Shape]: ...
