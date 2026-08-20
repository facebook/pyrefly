# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from shape_extensions.dsl import (
    Error,
    parse_einsum_equation,
    prod,
    shape_dsl_function,
    ShapedArray,
    sum,
    symint,
    Unknown,
)

@type_shape_dsl_function
def eig_shape(shape: IntTuple) -> IntTuple:
    if len(shape) < 2:
        if len(shape) == 0:
            return dsl.Invalid("eig requires at least 2D input, got 0D tensor")
        return dsl.Invalid("eig requires at least 2D input, got 1D tensor")
    return dsl.IntTuple((shape[i] for i in range(len(shape) - 1)))

@type_shape_dsl_function
def eigvals_shape(shape: IntTuple) -> IntTuple:
    if len(shape) < 2:
        if len(shape) == 0:
            return dsl.Invalid("eigvals requires at least 2D input, got 0D tensor")
        return dsl.Invalid("eigvals requires at least 2D input, got 1D tensor")
    return dsl.IntTuple((shape[i] for i in range(len(shape) - 1)))

@type_shape_dsl_function
def slogdet_shape(shape: IntTuple) -> IntTuple:
    if len(shape) < 2:
        if len(shape) == 0:
            return dsl.Invalid("slogdet requires at least 2D input, got 0D tensor")
        return dsl.Invalid("slogdet requires at least 2D input, got 1D tensor")
    return dsl.IntTuple((shape[i] for i in range(len(shape) - 2)))

@shape_dsl_function
def normalize_dim(rank: int, dim: int) -> int:
    if dim < 0:
        return dim + rank
    return dim

@shape_dsl_function
def int_max(a: int, b: int) -> int:
    if a > b:
        return a
    return b

@shape_dsl_function
def replace_dim(
    dims: list[int | symint], i: int, value: int | symint
) -> list[int | symint]:
    return dims[:i] + [value] + dims[i + 1 :]

@shape_dsl_function
def insert_dim(
    dims: list[int | symint], i: int, value: int | symint
) -> list[int | symint]:
    return dims[:i] + [value] + dims[i:]

@shape_dsl_function
def broadcast(a: list[int | symint], b: list[int | symint]) -> list[int | symint]:
    max_len = int_max(len(a), len(b))
    padded_a = [1 for _ in range(max_len - len(a))] + a
    padded_b = [1 for _ in range(max_len - len(b))] + b
    return [bd if ad == 1 else ad for ad, bd in zip(padded_a, padded_b)]

@shape_dsl_function
def broadcast_int(
    expr: int | symint | list[int | symint], n: int
) -> list[int | symint]:
    if isinstance(expr, list):
        return expr
    return [expr for _ in range(n)]

@type_shape_dsl_function
def reduce_shape(
    shape: IntTuple,
    dim: int | tuple[int, ...] | None,
    keepdim: bool,
) -> IntTuple:
    if dim is None:
        dims = range(len(shape))
    elif dsl.is_int_value(dim):
        if dim == -1:
            if len(shape) == 0:
                return shape
            if keepdim:
                return dsl.concat(shape[:-1], dsl.IntTuple((1,)))
            return shape[:-1]
        dims = (dim,)
    elif len(dim) == 0:
        dims = range(len(shape))
    else:
        dims = dim
    if len(shape) == 0:
        # PyTorch lets either 0 or -1 name the scalar reduction axis. After
        # normalization, using both is therefore a duplicate dimension.
        if any(item != 0 and item != -1 for item in dims):
            return dsl.Invalid("dimension out of range")
    elif any(item < 0 - len(shape) or item >= len(shape) for item in dims):
        return dsl.Invalid("dimension out of range")
    normalized = tuple(
        (
            0 if len(shape) == 0 else (item + len(shape) if item < 0 else item)
            for item in dims
        )
    )
    if any(normalized.count(item) > 1 for item in normalized):
        return dsl.Invalid("duplicate dimension")
    if keepdim:
        return dsl.IntTuple(
            (1 if index in normalized else shape[index] for index in range(len(shape)))
        )
    return dsl.IntTuple(
        (shape[index] for index in range(len(shape)) if index not in normalized)
    )

@type_shape_dsl_function
def reduce_shape_no_keep(
    shape: IntTuple, dim: int | tuple[int, ...] | None
) -> IntTuple:
    keepdim = False
    return reduce_shape(shape, dim, keepdim)

@type_shape_dsl_function
def cosine_similarity_shape(shape: IntTuple, dim: int) -> IntTuple:
    if dim == -1:
        if len(shape) == 0:
            return shape
        return shape[:-1]
    if len(shape) == 0:
        if dim == 0:
            return shape
        return dsl.Invalid("cosine_similarity dimension out of range")
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("cosine_similarity dimension out of range")
    return dsl.IntTuple(
        (
            shape[index]
            for index in range(len(shape))
            if index != (dim + len(shape) if dim < 0 else dim)
        )
    )

@shape_dsl_function
def contains(lst: list[int], val: int) -> bool:
    return len([x for x in lst if x == val]) > 0

@shape_dsl_function
def scatter(size: int, indices: list[int], values: list[int], fill: int) -> list[int]:
    matches = [[k for k in range(len(indices)) if indices[k] == i] for i in range(size)]
    return [values[m[0]] if len(m) > 0 else fill for m in matches]

@shape_dsl_function
def move_dims(
    dims: list[int | symint], source: int | list[int], dest: int | list[int], rank: int
) -> list[int | symint]:
    src = broadcast_int(source, 1)
    dst = broadcast_int(dest, 1)
    src_norm = [normalize_dim(rank, s) for s in src]
    dst_norm = [normalize_dim(rank, d) for d in dst]
    non_dst = [i for i in range(rank) if not contains(dst_norm, i)]
    remaining = [i for i in range(rank) if not contains(src_norm, i)]
    perm = scatter(rank, dst_norm + non_dst, src_norm + remaining, 0)
    return [dims[p] for p in perm]

@shape_dsl_function
def conv_spatial_out(
    input_dim: int | symint,
    kernel: int | symint,
    stride: int | symint,
    padding: int | symint,
    dilation: int | symint,
) -> int | symint:
    return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

@shape_dsl_function
def reshape_ir(self: ShapedArray, shape: list[int | symint]) -> ShapedArray:
    minus_one_count = len([d for d in shape if d == -1])
    if minus_one_count > 1:
        raise Error("can only specify one unknown dimension as -1")
    has_bad_neg = len([d for d in shape if isinstance(d, int) and d < -1]) > 0
    if has_bad_neg:
        raise Error("invalid negative dimension value (only -1 is allowed)")
    has_zero = len([d for d in shape if isinstance(d, int) and d == 0]) > 0
    if has_zero:
        raise Error("reshape dimensions cannot contain 0")
    if minus_one_count > 0:
        known = prod([d for d in shape if d != -1])
        total = prod(self.shape)
        if isinstance(total, int) and isinstance(known, int) and total % known != 0:
            raise Error(
                "could not infer size for dimension -1: expected "
                + str(total)
                + " to be divisible by "
                + str(known)
            )
        return ShapedArray(shape=[total // known if d == -1 else d for d in shape])
    return ShapedArray(shape=shape)

@type_shape_dsl_function
def squeeze_shape(shape: IntTuple, dim: int | None) -> IntTuple:
    if dim is None:
        return dsl.IntTuple(
            (shape[index] for index in range(len(shape)) if shape[index] != 1)
        )
    if dsl.is_int_value(dim):
        if len(shape) == 0:
            if dim == 0 or dim == -1:
                return shape
            return dsl.Invalid("squeeze dimension out of range")
        if dim < 0 - len(shape) or dim >= len(shape):
            return dsl.Invalid("squeeze dimension out of range")
        return dsl.IntTuple(
            (
                shape[index]
                for index in range(len(shape))
                if index != (dim + len(shape) if dim < 0 else dim) or shape[index] != 1
            )
        )
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def unsqueeze_shape(shape: IntTuple, dim: int) -> IntTuple:
    if dim == -1:
        return dsl.concat(shape, dsl.IntTuple((1,)))
    if dim < 0 - len(shape) - 1 or dim > len(shape):
        return dsl.Invalid("unsqueeze dimension out of range")
    return dsl.IntTuple(
        (
            1
            if index == (dim + len(shape) + 1 if dim < 0 else dim)
            else shape[
                index - 1
                if index > (dim + len(shape) + 1 if dim < 0 else dim)
                else index
            ]
            for index in range(len(shape) + 1)
        )
    )

@type_shape_dsl_function
def transpose_shape(shape: IntTuple, dim0: int, dim1: int) -> IntTuple:
    if dim0 == dim1 and (dim0 == 0 or dim0 == -1):
        return shape
    if len(shape) == 0:
        if (dim0 == 0 or dim0 == -1) and (dim1 == 0 or dim1 == -1):
            return shape
        return dsl.Invalid("transpose dimension out of range")
    if (
        dim0 < 0 - len(shape)
        or dim0 >= len(shape)
        or dim1 < 0 - len(shape)
        or dim1 >= len(shape)
    ):
        return dsl.Invalid("transpose dimension out of range")
    if dim0 == dim1:
        return shape
    return dsl.IntTuple(
        (
            shape[dim1 + len(shape) if dim1 < 0 else dim1]
            if index == (dim0 + len(shape) if dim0 < 0 else dim0)
            else shape[dim0 + len(shape) if dim0 < 0 else dim0]
            if index == (dim1 + len(shape) if dim1 < 0 else dim1)
            else shape[index]
            for index in range(len(shape))
        )
    )

@shape_dsl_function
def permute_ir(self: ShapedArray, dims: list[int]) -> ShapedArray:
    rank = len(self.shape)
    if len(dims) != rank:
        raise Error("permute: expected " + str(rank) + " dims, got " + str(len(dims)))
    return ShapedArray(shape=[self.shape[normalize_dim(rank, d)] for d in dims])

@shape_dsl_function
def flatten_ir(self: ShapedArray, start_dim: int = 0, end_dim: int = -1) -> ShapedArray:
    rank = len(self.shape)
    s = normalize_dim(rank, start_dim)
    e = normalize_dim(rank, end_dim)
    return ShapedArray(
        shape=self.shape[:s] + [prod(self.shape[s : e + 1])] + self.shape[e + 1 :]
    )

@shape_dsl_function
def expand_ir(self: ShapedArray, sizes: list[int | symint]) -> ShapedArray:
    return ShapedArray(shape=[d if t == -1 else t for d, t in zip(self.shape, sizes)])

@shape_dsl_function
def repeat_ir(self: ShapedArray, sizes: list[int | symint]) -> ShapedArray:
    return ShapedArray(shape=[d * r for d, r in zip(self.shape, sizes)])

@shape_dsl_function
def movedim_ir(
    self: ShapedArray, source: int | list[int], destination: int | list[int]
) -> ShapedArray:
    return ShapedArray(
        shape=move_dims(self.shape, source, destination, len(self.shape))
    )

@type_shape_dsl_function
def unfold_checked_shape(
    shape: IntTuple,
    dimension_size: Int,
    normalized: int,
    window_size: Int,
    step: int,
) -> IntTuple:
    # A symbolic extent cannot prove this ordering invalid, so preserve its formula.
    if dsl.is_concrete_int(dimension_size):
        if dsl.is_concrete_int(window_size):
            if dimension_size < window_size:
                return dsl.Invalid("unfold size must not exceed the selected dimension")
    window_count = (dimension_size - window_size) // step + 1
    replaced = dsl.IntTuple(
        (
            window_count if index == normalized else shape[index]
            for index in range(len(shape))
        )
    )
    return dsl.concat(replaced, dsl.IntTuple((window_size,)))

@type_shape_dsl_function
def unfold_shape(shape: IntTuple, dimension: int, size: int, step: int) -> IntTuple:
    # Binding the rank lets both branches assign the normalized Flag value consistently.
    rank = len(shape)
    if rank == 0:
        if dimension != 0 and dimension != -1:
            return dsl.Invalid("unfold dimension out of range")
        if size < 0:
            return dsl.Invalid("unfold size must be non-negative")
        if size > 1:
            return dsl.Invalid("unfold size must not exceed the selected dimension")
        if step < 1:
            return dsl.Invalid("unfold step must be greater than zero")
        return dsl.IntTuple((size + 0,))
    if dimension < 0:
        normalized = dimension + rank
    else:
        normalized = dimension + 0
    if normalized < 0 or normalized >= rank:
        return dsl.Invalid("unfold dimension out of range")
    if size < 0:
        return dsl.Invalid("unfold size must be non-negative")
    if step < 1:
        return dsl.Invalid("unfold step must be greater than zero")
    window_size = size + 0
    dimension_size = shape[normalized]
    return unfold_checked_shape(shape, dimension_size, normalized, window_size, step)

@shape_dsl_function
def cat_ir(tensors: list[ShapedArray], dim: int = 0) -> ShapedArray:
    first = tensors[0]
    d = normalize_dim(len(first.shape), dim)
    return ShapedArray(
        shape=[
            sum([t.shape[i] for t in tensors]) if i == d else dim_val
            for i, dim_val in enumerate(first.shape)
        ]
    )

@shape_dsl_function
def stack_ir(tensors: list[ShapedArray], dim: int = 0) -> ShapedArray:
    first = tensors[0]
    d = normalize_dim(len(first.shape) + 1, dim)
    return ShapedArray(shape=insert_dim(first.shape, d, len(tensors)))

@shape_dsl_function
def tile_ir(self: ShapedArray, dims: list[int]) -> ShapedArray:
    rank = len(self.shape)
    if len(dims) > rank:
        extra = len(dims) - rank
        return ShapedArray(
            shape=[r for r in dims[:extra]]
            + [d * r for d, r in zip(self.shape, dims[extra:])]
        )
    return ShapedArray(shape=[d * r for d, r in zip(self.shape, dims)])

@type_shape_dsl_function
def select_shape(shape: IntTuple, dim: int) -> IntTuple:
    if dim == -1:
        if len(shape) == 0:
            return dsl.Invalid("select dimension out of range")
        return shape[:-1]
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("select dimension out of range")
    return dsl.IntTuple(
        (
            shape[index]
            for index in range(len(shape))
            if index != (dim + len(shape) if dim < 0 else dim)
        )
    )

@type_shape_dsl_function
def unbind_shape(shape: IntTuple, dim: int) -> IntTuple:
    if dim == -1:
        if len(shape) == 0:
            return dsl.Invalid("unbind dimension out of range")
        return shape[:-1]
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("unbind dimension out of range")
    return dsl.IntTuple(
        (
            shape[index]
            for index in range(len(shape))
            if index != (dim + len(shape) if dim < 0 else dim)
        )
    )

@type_shape_dsl_function
def replace_axis_extent(shape: IntTuple, dim: int, extent: Int) -> IntTuple:
    if dim == -1:
        if len(shape) == 0:
            return dsl.Invalid("dimension out of range")
        return dsl.concat(shape[:-1], dsl.IntTuple((extent,)))
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("dimension out of range")
    return dsl.IntTuple(
        (
            extent if index == (dim + len(shape) if dim < 0 else dim) else shape[index]
            for index in range(len(shape))
        )
    )

@type_shape_dsl_function
def topk_shape(shape: IntTuple, dim: int, extent: Int) -> IntTuple:
    if len(shape) == 0:
        if dim == 0 or dim == -1:
            return shape
        return dsl.Invalid("topk dimension out of range")
    return replace_axis_extent(shape, dim, extent)

@type_shape_dsl_function
def multinomial_shape(shape: IntTuple, num_samples: Int) -> IntTuple:
    if len(shape) == 1:
        return dsl.IntTuple((num_samples,))
    if len(shape) == 2:
        return dsl.IntTuple((shape[0], num_samples))
    return dsl.Invalid("multinomial expects 1D or 2D input")

@shape_dsl_function
def split_ir(
    self: ShapedArray,
    split_size_or_sections: int | symint | list[int | symint] | None = None,
    dim: int = 0,
) -> list[ShapedArray]:
    d = normalize_dim(len(self.shape), dim)
    if isinstance(split_size_or_sections, list):
        return [
            ShapedArray(shape=replace_dim(self.shape, d, section))
            for section in split_size_or_sections
        ]
    if isinstance(split_size_or_sections, int):
        dim_val = self.shape[d]
        if isinstance(dim_val, int):
            count = (dim_val + split_size_or_sections - 1) // split_size_or_sections
            return [
                ShapedArray(
                    shape=replace_dim(
                        self.shape,
                        d,
                        split_size_or_sections
                        if i < count - 1
                        else dim_val - (count - 1) * split_size_or_sections,
                    )
                )
                for i in range(count)
            ]
        return [
            ShapedArray(shape=replace_dim(self.shape, d, split_size_or_sections)),
            ...,
        ]
    if split_size_or_sections != None:
        quotient = self.shape[d] // split_size_or_sections
        if isinstance(quotient, int):
            return [
                ShapedArray(shape=replace_dim(self.shape, d, split_size_or_sections))
                for _ in range(quotient)
            ]
        return [
            ShapedArray(shape=replace_dim(self.shape, d, split_size_or_sections)),
            ...,
        ]
    return Unknown

@shape_dsl_function
def chunk_ir(self: ShapedArray, chunks: int, dim: int = 0) -> list[ShapedArray]:
    d = normalize_dim(len(self.shape), dim)
    dim_val = self.shape[d]
    if isinstance(dim_val, int):
        chunk_size = (dim_val + chunks - 1) // chunks
        return [
            ShapedArray(
                shape=replace_dim(
                    self.shape,
                    d,
                    chunk_size
                    if i < chunks - 1
                    else dim_val - (chunks - 1) * chunk_size,
                )
            )
            for i in range(chunks)
        ]
    return [
        ShapedArray(shape=replace_dim(self.shape, d, dim_val // chunks))
        for i in range(chunks)
    ]

@type_shape_dsl_function
def index_select_shape(shape: IntTuple, dim: int, index_shape: IntTuple) -> IntTuple:
    if len(index_shape) == 0:
        if dim == -1:
            if len(shape) == 0:
                return dsl.Invalid("index_select dimension out of range")
            return dsl.concat(shape[:-1], dsl.IntTuple((1,)))
        if dim < 0 - len(shape) or dim >= len(shape):
            return dsl.Invalid("index_select dimension out of range")
        return dsl.IntTuple(
            (
                1 if index == (dim + len(shape) if dim < 0 else dim) else shape[index]
                for index in range(len(shape))
            )
        )
    if len(index_shape) != 1:
        return dsl.Invalid("index_select index must be 0D or 1D")
    index_extent = index_shape[0]
    if dim == -1:
        if len(shape) == 0:
            return dsl.Invalid("index_select dimension out of range")
        return dsl.concat(shape[:-1], dsl.IntTuple((index_extent,)))
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("index_select dimension out of range")
    return dsl.IntTuple(
        (
            index_extent
            if index == (dim + len(shape) if dim < 0 else dim)
            else shape[index]
            for index in range(len(shape))
        )
    )

@shape_dsl_function
def repeat_interleave_ir(
    self: ShapedArray,
    repeats: int | symint | ShapedArray,
    dim: int | None = None,
    output_size: int | symint | None = None,
) -> ShapedArray:
    if output_size != None:
        if dim == None:
            return ShapedArray(shape=[output_size])
        d = normalize_dim(len(self.shape), dim)
        return ShapedArray(shape=replace_dim(self.shape, d, output_size))
    if isinstance(repeats, ShapedArray):
        return Unknown
    if dim == None:
        return ShapedArray(shape=[prod(self.shape) * repeats])
    d = normalize_dim(len(self.shape), dim)
    return ShapedArray(shape=replace_dim(self.shape, d, self.shape[d] * repeats))

@shape_dsl_function
def repeat_interleave_input_ir(
    input: ShapedArray,
    repeats: int | symint | ShapedArray,
    dim: int | None = None,
    output_size: int | symint | None = None,
) -> ShapedArray:
    return repeat_interleave_ir(input, repeats, dim, output_size)

@type_shape_dsl_function
def arange_extent(end: Int) -> Int:
    # Construct zero in the `Int` domain so it can be passed as the starting dimension.
    origin = end - end
    unit_step = 1
    return arange_step_extent(origin, end, unit_step)

@type_shape_dsl_function
def arange_step_extent(start: Int, end: Int, step: int) -> Int:
    # `step` is a Flag value because its sign determines the rounding direction. Symbolic bounds
    # use the truncating expression, which is exact when the step divides the range.
    if step == 0:
        return dsl.Invalid("arange step must be nonzero")
    difference = end - start
    if dsl.is_concrete_int(start):
        if dsl.is_concrete_int(end):
            if step > 0:
                if end < start:
                    return dsl.Invalid("arange bounds are inconsistent with step")
                return (difference + step - 1) // step
            if start < end:
                return dsl.Invalid("arange bounds are inconsistent with step")
            negative_step = 0 - step
            return ((0 - difference) + negative_step - 1) // negative_step
    return difference // step

@type_shape_dsl_function
def diag_embed_shape(shape: IntTuple, offset: int, dim1: int, dim2: int) -> IntTuple:
    if len(shape) == 0:
        return dsl.Invalid("diag_embed input must have at least one dimension")
    output_rank = len(shape) + 1
    if dim1 < 0:
        normalized_dim1 = dim1 + output_rank
    else:
        normalized_dim1 = dim1 + 0
    if dim2 < 0:
        normalized_dim2 = dim2 + output_rank
    else:
        normalized_dim2 = dim2 + 0
    if (
        normalized_dim1 < 0
        or normalized_dim1 >= output_rank
        or normalized_dim2 < 0
        or normalized_dim2 >= output_rank
    ):
        return dsl.Invalid("diag_embed dimension out of range")
    if normalized_dim1 == normalized_dim2:
        return dsl.Invalid("diag_embed dimensions must be different")
    if offset < 0:
        extent = shape[-1] - offset
    else:
        extent = shape[-1] + offset
    return dsl.IntTuple(
        (
            extent
            if index == normalized_dim1 or index == normalized_dim2
            else shape[
                index
                - (1 if normalized_dim1 < index else 0)
                - (1 if normalized_dim2 < index else 0)
            ]
            for index in range(output_rank)
        )
    )

@type_shape_dsl_function
def matmul_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    r1 = len(left)
    r2 = len(right)
    if r1 == 1 and r2 == 1:
        return dsl.IntTuple(())
    if r1 == 1 and r2 >= 2:
        return dsl.concat(right[:-2], right[-1:])
    if r1 >= 2 and r2 == 1:
        return left[:-1]
    if r1 == 2 and r2 == 2:
        return dsl.IntTuple((left[0], right[1]))
    if r1 == 2 and r2 >= 3:
        return dsl.concat(right[:-2], dsl.IntTuple((left[0], right[-1])))
    if r1 >= 3 and r2 == 2:
        return dsl.concat(left[:-2], dsl.IntTuple((left[-2], right[1])))
    if r1 >= 3 and r2 >= 3:
        # Batch dimensions prefer a non-unit dimension and otherwise the left operand.
        if r1 < r2:
            extra = r2 - r1
            batch = dsl.IntTuple(
                (
                    right[i]
                    if i < extra
                    else left[i - extra]
                    if left[i - extra] == right[i]
                    else right[i]
                    if left[i - extra] == 1
                    else left[i - extra]
                    for i in range(r2 - 2)
                )
            )
        else:
            extra = r1 - r2
            batch = dsl.IntTuple(
                (
                    left[i]
                    if i < extra or left[i] == right[i - extra]
                    else right[i - extra]
                    if left[i] == 1
                    else left[i]
                    for i in range(r1 - 2)
                )
            )
        return dsl.concat(batch, dsl.IntTuple((left[-2], right[-1])))
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def tensordot_shape(left: IntTuple, right: IntTuple, dims: int) -> IntTuple:
    if dims < 0:
        return dsl.Invalid("tensordot dims must be non-negative")
    if dims > len(left) or dims > len(right):
        return dsl.Invalid("tensordot dims exceeds input rank")
    # TODO(stroxler): Validate contracted dimensions pairwise. This rule currently validates only
    # ranks.
    return dsl.concat(left[: len(left) - dims], right[dims:])

@shape_dsl_function
def apply_einsum(
    output_map: list[list[int]], check_pairs: list[list[int]], inputs: list[ShapedArray]
) -> ShapedArray:
    bad_dims = [
        1
        for i0, d0, i1, d1 in check_pairs
        if isinstance(inputs[i0].shape[d0], int)
        and isinstance(inputs[i1].shape[d1], int)
        and inputs[i0].shape[d0] != inputs[i1].shape[d1]
    ]
    if len(bad_dims) > 0:
        raise Error("einsum: inconsistent dimensions for repeated index")
    return ShapedArray(shape=[inputs[inp].shape[dim] for inp, dim in output_map])

@shape_dsl_function
def einsum_ir(spec: str, operands: list[ShapedArray] | None = None) -> ShapedArray:
    if operands != None:
        output_map, check_pairs = parse_einsum_equation(spec)
        return apply_einsum(output_map, check_pairs, operands)
    return Unknown

@type_shape_dsl_function
def conv_shape(
    input_shape: IntTuple,
    weight_shape: IntTuple,
    stride: int | tuple[int, ...] | None,
    padding: int | tuple[int, ...] | None,
    dilation: int | tuple[int, ...] | None,
) -> IntTuple:
    # `zip` stops at the shortest input, so unequal ranks would silently drop
    # trailing spatial dimensions instead of reporting the mismatch.
    if len(input_shape) != len(weight_shape):
        return dsl.Invalid("convolution input and weight must have the same rank")
    spatial_rank = len(input_shape) - 2
    if stride is None:
        return dsl.Invalid("convolution stride cannot be None")
    elif dsl.is_int_value(stride):
        strides = tuple(stride for _ in range(spatial_rank))
    else:
        strides = stride
    if padding is None:
        return dsl.Invalid("convolution padding cannot be None")
    elif dsl.is_int_value(padding):
        paddings = tuple(padding for _ in range(spatial_rank))
    else:
        paddings = padding
    if dilation is None:
        return dsl.Invalid("convolution dilation cannot be None")
    elif dsl.is_int_value(dilation):
        dilations = tuple(dilation for _ in range(spatial_rank))
    else:
        dilations = dilation
    input_spatial = input_shape[2:]
    weight_spatial = weight_shape[2:]
    spatial = dsl.IntTuple(
        (
            (s + 2 * p - dil * (k - 1) - 1) // st + 1
            for s, k, st, p, dil in zip(
                input_spatial,
                weight_spatial,
                strides,
                paddings,
                dilations,
            )
        )
    )
    return dsl.concat(dsl.IntTuple((input_shape[0], weight_shape[0])), spatial)

@type_shape_dsl_function
def conv_transpose_shape(
    input_shape: IntTuple,
    weight_shape: IntTuple,
    stride: int | tuple[int, ...] | None,
    padding: int | tuple[int, ...] | None,
    output_padding: int | tuple[int, ...] | None,
    dilation: int | tuple[int, ...] | None,
    groups: int,
) -> IntTuple:
    spatial_rank = len(input_shape) - 2
    if stride is None:
        return dsl.Invalid("convolution stride cannot be None")
    elif dsl.is_int_value(stride):
        strides = tuple(stride for _ in range(spatial_rank))
    else:
        strides = stride
    if padding is None:
        return dsl.Invalid("convolution padding cannot be None")
    elif dsl.is_int_value(padding):
        paddings = tuple(padding for _ in range(spatial_rank))
    else:
        paddings = padding
    if output_padding is None:
        return dsl.Invalid("convolution output_padding cannot be None")
    elif dsl.is_int_value(output_padding):
        output_paddings = tuple(output_padding for _ in range(spatial_rank))
    else:
        output_paddings = output_padding
    if dilation is None:
        return dsl.Invalid("convolution dilation cannot be None")
    elif dsl.is_int_value(dilation):
        dilations = tuple(dilation for _ in range(spatial_rank))
    else:
        dilations = dilation
    input_spatial = input_shape[2:]
    weight_spatial = weight_shape[2:]
    spatial = dsl.IntTuple(
        (
            (s - 1) * st - 2 * p + dil * (k - 1) + op + 1
            for s, k, st, p, op, dil in zip(
                input_spatial,
                weight_spatial,
                strides,
                paddings,
                output_paddings,
                dilations,
            )
        )
    )
    # Transposed convolution stores per-group output channels in `weight_shape[1]`.
    return dsl.concat(dsl.IntTuple((input_shape[0], weight_shape[1] * groups)), spatial)

@shape_dsl_function
def pool_ir(
    self: ShapedArray,
    kernel_size: int | list[int],
    stride: int | list[int] | None = None,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
    return_indices: bool = False,
) -> ShapedArray:
    spatial_dims = len(self.shape) - 2
    ks_list = broadcast_int(kernel_size, spatial_dims)
    stride_list = ks_list if stride == None else broadcast_int(stride, spatial_dims)
    padding_list = broadcast_int(padding, spatial_dims)
    dilation_list = broadcast_int(dilation, spatial_dims)
    out = [self.shape[0], self.shape[1]] + [
        conv_spatial_out(s, k, st, p, dil)
        for s, k, st, p, dil in zip(
            self.shape[2:], ks_list, stride_list, padding_list, dilation_list
        )
    ]
    if return_indices:
        return [ShapedArray(shape=out), ShapedArray(shape=out)]
    return ShapedArray(shape=out)

@shape_dsl_function
def adaptive_pool_ir(
    self: ShapedArray, output_size: int | symint | list[int | symint]
) -> ShapedArray:
    out_sizes = broadcast_int(output_size, len(self.shape) - 2)
    return ShapedArray(shape=[self.shape[0], self.shape[1]] + out_sizes)

@shape_dsl_function
def interpolate_ir(
    self: ShapedArray,
    size: int | symint | list[int | symint] | None = None,
    scale_factor: int | symint | None = None,
) -> ShapedArray:
    if size != None:
        return ShapedArray(
            shape=[self.shape[0], self.shape[1]]
            + broadcast_int(size, len(self.shape) - 2)
        )
    if scale_factor != None:
        return ShapedArray(
            shape=[self.shape[0], self.shape[1]]
            + [d * scale_factor for d in self.shape[2:]]
        )
    raise Error("interpolate requires either 'size' or 'scale_factor' argument")

@shape_dsl_function
def loss_ir(self: ShapedArray, reduction: str = "mean") -> ShapedArray:
    if reduction == "none":
        return ShapedArray(shape=self.shape)
    return ShapedArray(shape=[])

@shape_dsl_function
def pad_ir(self: ShapedArray, pad: list[int]) -> ShapedArray:
    rank = len(self.shape)
    num_pad_dims = len(pad) // 2
    offsets = [
        pad[(rank - 1 - i) * 2] + pad[(rank - 1 - i) * 2 + 1]
        if i >= rank - num_pad_dims
        else 0
        for i in range(rank)
    ]
    return ShapedArray(shape=[d + offsets[i] for i, d in enumerate(self.shape)])

# The `+ 0` branches below keep normalized axes in one deferred integer domain.
@type_shape_dsl_function
def rfft_shape(shape: IntTuple, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    extent = shape[axis] // 2 + 1
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def rfft_literal_shape(shape: IntTuple, n: int, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    extent = n // 2 + 1
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def rfft_n_shape(shape: IntTuple, n: Int, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((n // 2 + 1,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def irfft_shape(shape: IntTuple, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    extent = 2 * (shape[axis] - 1)
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def irfft_literal_shape(shape: IntTuple, n: int, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    # Arithmetic converts the `Flag[int]` length to an `Int` dimension.
    extent = n + 0
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def irfft_n_shape(shape: IntTuple, n: Int, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    return dsl.concat(dsl.concat(shape[:axis], dsl.IntTuple((n,))), shape[axis + 1 :])

@type_shape_dsl_function
def size_dim_shape(shape: IntTuple, dim: int) -> Int:
    if len(shape) == 0:
        return dsl.Invalid("size dimension out of range")
    # A symbolic-rank shape has no known `len`, so the range check below gives up on it even
    # when its last dimension is known. Answer `-1` from the known suffix first.
    if dim == -1:
        last = shape[-1]
        return last
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("size dimension out of range")
    result = shape[dim]
    return result

@type_shape_dsl_function
def numel_shape(shape: IntTuple) -> Int:
    return dsl.prod(shape)

@shape_dsl_function
def dim_ir(self: ShapedArray) -> int:
    return len(self.shape)

@shape_dsl_function
def item_ir(self: ShapedArray) -> ShapedArray:
    if len(self.shape) != 0:
        raise Error(
            "item() only works on 0-dimensional tensors, got "
            + str(len(self.shape))
            + "D tensor"
        )
    return Unknown

@shape_dsl_function
def nn_flatten_forward_ir(
    input: ShapedArray, start_dim: symint = 1, end_dim: symint = -1
) -> ShapedArray:
    return flatten_ir(input, start_dim, end_dim)

@shape_dsl_function
def nn_maxpool_forward_ir(
    input: ShapedArray,
    kernel_size: symint = 1,
    stride: symint | None = None,
    padding: symint = 0,
    dilation: symint = 1,
) -> ShapedArray:
    return pool_ir(input, kernel_size, stride, padding, dilation)

@shape_dsl_function
def nn_avgpool_forward_ir(
    input: ShapedArray,
    kernel_size: symint = 1,
    stride: symint | None = None,
    padding: symint = 0,
) -> ShapedArray:
    return pool_ir(input, kernel_size, stride, padding, 1)

@shape_dsl_function
def nn_upsample_forward_ir(
    input: ShapedArray, size: symint | None = None, scale_factor: symint | None = None
) -> ShapedArray:
    return interpolate_ir(input, size, scale_factor)

@shape_dsl_function
def nn_pixel_shuffle_forward_ir(
    input: ShapedArray, upscale_factor: symint
) -> ShapedArray:
    r = upscale_factor
    return ShapedArray(
        shape=[input.shape[0], input.shape[1] // (r * r)]
        + [d * r for d in input.shape[2:]]
    )

@shape_dsl_function
def nn_glu_forward_ir(input: ShapedArray, dim: symint = 1) -> ShapedArray:
    rank = len(input.shape)
    d = normalize_dim(rank, dim)
    return ShapedArray(shape=replace_dim(input.shape, d, input.shape[d] // 2))

@shape_dsl_function
def nn_lstm_forward_ir(
    input: ShapedArray,
    input_size: symint,
    hidden_size: symint,
    num_layers: symint = 1,
    bidirectional: bool = False,
) -> [ShapedArray, ShapedArray, ShapedArray]:
    nd = 2 if bidirectional else 1
    output = ShapedArray(shape=[input.shape[0], input.shape[1], hidden_size * nd])
    h_n = ShapedArray(shape=[num_layers * nd, input.shape[0], hidden_size])
    c_n = ShapedArray(shape=[num_layers * nd, input.shape[0], hidden_size])
    return [output, h_n, c_n]

@shape_dsl_function
def nn_gru_forward_ir(
    input: ShapedArray,
    input_size: symint,
    hidden_size: symint,
    num_layers: symint = 1,
    bidirectional: bool = False,
) -> [ShapedArray, ShapedArray]:
    nd = 2 if bidirectional else 1
    output = ShapedArray(shape=[input.shape[0], input.shape[1], hidden_size * nd])
    h_n = ShapedArray(shape=[num_layers * nd, input.shape[0], hidden_size])
    return [output, h_n]

@shape_dsl_function
def nn_lstmcell_forward_ir(
    input: ShapedArray, input_size: symint, hidden_size: symint
) -> [ShapedArray, ShapedArray]:
    h = ShapedArray(shape=[input.shape[0], hidden_size])
    c = ShapedArray(shape=[input.shape[0], hidden_size])
    return [h, c]

@shape_dsl_function
def nn_reflectionpad2d_forward_ir(input: ShapedArray, padding: symint) -> ShapedArray:
    return ShapedArray(
        shape=[
            input.shape[0],
            input.shape[1],
            input.shape[2] + 2 * padding,
            input.shape[3] + 2 * padding,
        ]
    )
