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

# TODO(stroxler): Use `IntTuple` slicing here once it preserves the symbolic-rank cases covered by
# these generators, then share the common rank validation among the three helpers.
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

@type_shape_dsl_function
def reshape_shape(shape: IntTuple, target: IntTuple) -> IntTuple:
    inferred = tuple(
        (
            1
            for dimension in target
            if dsl.is_concrete_int(dimension) and dimension == -1
        )
    )
    if len(inferred) > 1:
        return dsl.Invalid("can only specify one unknown dimension as -1")
    if any(dsl.is_concrete_int(dimension) and dimension < -1 for dimension in target):
        return dsl.Invalid("invalid negative dimension value (only -1 is allowed)")
    # Element counts are the only facts both branches need, so each product is evaluated
    # once here and shared: a `-1` target divides `total` by `known`, a fully specified
    # target compares them. Both diagnostics below require concrete products, so a
    # symbolic input keeps its dimensions and a fully open one recovers gradually.
    known_shape = dsl.IntTuple(
        (
            dimension
            for dimension in target
            if not (dsl.is_concrete_int(dimension) and dimension == -1)
        )
    )
    known = dsl.prod(known_shape)
    total = dsl.prod(shape)
    if len(inferred) == 0:
        if dsl.is_concrete_int(total) and dsl.is_concrete_int(known) and total != known:
            return dsl.Invalid("reshape target element count does not match the input")
        return target
    if dsl.is_concrete_int(known):
        if known == 0:
            return dsl.Invalid("could not infer size for dimension -1")
        if dsl.is_concrete_int(total) and total % known != 0:
            return dsl.Invalid("could not infer size for dimension -1")
    return dsl.IntTuple(
        (
            total // known
            if dsl.is_concrete_int(dimension) and dimension == -1
            else dimension
            for dimension in target
        )
    )

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

@type_shape_dsl_function
def permute_shape(shape: IntTuple, dims: int | tuple[int, ...] | None) -> IntTuple:
    if dims is None or dsl.is_int_value(dims):
        return dsl.Invalid("permute dimensions must be a sequence")
    if len(dims) != len(shape):
        return dsl.Invalid("permute dimensions must match the input rank")
    if any(dim < 0 - len(shape) or dim >= len(shape) for dim in dims):
        return dsl.Invalid("permute dimension out of range")
    normalized_dims = tuple((dim + len(shape) if dim < 0 else dim for dim in dims))
    duplicate_offsets = tuple(
        (
            normalized_dims.index(dim) - position
            for position, dim in zip(range(len(normalized_dims)), normalized_dims)
        )
    )
    if any(offset != 0 for offset in duplicate_offsets):
        return dsl.Invalid("permute dimensions must be unique")
    return dsl.IntTuple((shape[dim] for dim in normalized_dims))

@type_shape_dsl_function
def flatten_shape(shape: IntTuple, start_dim: int, end_dim: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        if (start_dim == 0 or start_dim == -1) and (end_dim == 0 or end_dim == -1):
            return dsl.IntTuple((1,))
        return dsl.Invalid("flatten dimension out of range for scalar input")
    if start_dim < 0 - rank or start_dim >= rank:
        return dsl.Invalid("flatten start_dim out of range")
    if end_dim < 0 - rank or end_dim >= rank:
        return dsl.Invalid("flatten end_dim out of range")
    if start_dim < 0:
        start = start_dim + rank
    else:
        start = start_dim + 0
    if end_dim < 0:
        end = end_dim + rank
    else:
        end = end_dim + 0
    if start > end:
        return dsl.Invalid("flatten start_dim cannot come after end_dim")
    return dsl.concat(
        dsl.concat(shape[:start], dsl.IntTuple((dsl.prod(shape[start : end + 1]),))),
        shape[end + 1 :],
    )

@type_shape_dsl_function
def expand_shape(shape: IntTuple, sizes: IntTuple) -> IntTuple:
    if len(sizes) < len(shape):
        return dsl.Invalid("expand target rank cannot be smaller than input rank")
    extra = len(sizes) - len(shape)
    leading = sizes[:extra]
    if any(dsl.is_concrete_int(size) and size == -1 for size in leading):
        return dsl.Invalid("expand cannot use -1 for a new leading dimension")
    if any(dsl.is_concrete_int(size) and size < -1 for size in sizes):
        return dsl.Invalid("expand target dimension cannot be less than -1")
    # A target only contradicts its source when both are concrete: a singleton source
    # broadcasts to any target and a -1 target copies the source, so every other pairing
    # (in particular any symbolic dimension) stays gradual. `any` cannot bind a `zip`,
    # so the paired verdicts are materialized as flags first.
    aligned = sizes[extra:]
    conflicts = dsl.IntTuple(
        (
            1
            if dsl.is_concrete_int(source)
            and dsl.is_concrete_int(target)
            and source != 1
            and target != -1
            and source != target
            else 0
            for source, target in zip(shape, aligned)
        )
    )
    if any(conflict == 1 for conflict in conflicts):
        return dsl.Invalid("expand cannot resize a non-singleton dimension")
    expanded = dsl.IntTuple(
        (
            source
            if (dsl.is_concrete_int(source) and source != 1)
            or (dsl.is_concrete_int(target) and target == -1)
            else target
            for source, target in zip(shape, aligned)
        )
    )
    return dsl.concat(leading, expanded)

@type_shape_dsl_function
def repeat_shape(shape: IntTuple, repeats: IntTuple) -> IntTuple:
    if len(repeats) < len(shape):
        return dsl.Invalid(
            "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
        )
    extra = len(repeats) - len(shape)
    return dsl.IntTuple(
        (
            repeats[index] if index < extra else shape[index - extra] * repeats[index]
            for index in range(len(repeats))
        )
    )

@type_shape_dsl_function
def movedim_scalar_shape(shape: IntTuple, source: int, destination: int) -> IntTuple:
    if not dsl.is_int_value(source) or not dsl.is_int_value(destination):
        return dsl.IntTuple.gradual()
    rank = len(shape)
    if rank == 0:
        # A rank-0 tensor still admits one implicit axis, so both spellings of it
        # (0 and -1) are legal and the move is a no-op. Each argument is checked
        # independently so the reported axis matches the offending argument.
        if source != 0 and source != -1:
            return dsl.Invalid("movedim source dimension out of range")
        if destination != 0 and destination != -1:
            return dsl.Invalid("movedim destination dimension out of range")
        return shape
    if source < 0 - rank or source >= rank:
        return dsl.Invalid("movedim source dimension out of range")
    if destination < 0 - rank or destination >= rank:
        return dsl.Invalid("movedim destination dimension out of range")
    normalized_source = (source + rank) % rank
    normalized_destination = (destination + rank) % rank
    return dsl.IntTuple(
        (
            shape[normalized_source]
            if index == normalized_destination
            else shape[index + 1]
            if normalized_source < normalized_destination
            and index >= normalized_source
            and index < normalized_destination
            else shape[index - 1]
            if normalized_source > normalized_destination
            and index > normalized_destination
            and index <= normalized_source
            else shape[index]
            for index in range(rank)
        )
    )

@type_shape_dsl_function
def movedim_tuple_shape(
    shape: IntTuple, source: IntTuple, destination: IntTuple
) -> IntTuple:
    if len(source) != len(destination):
        return dsl.Invalid("movedim source and destination must have equal length")
    rank = len(shape)
    if rank == 0:
        # A rank-0 tensor admits one implicit axis that only 0 and -1 name, so the
        # permutation arithmetic below (which divides by `rank`) must stay
        # unreached. Each sequence is checked independently so the reported axis
        # matches the offending argument.
        if any(
            dsl.is_concrete_int(axis) and axis != 0 and axis != -1 for axis in source
        ):
            return dsl.Invalid("movedim source dimension out of range")
        if any(
            dsl.is_concrete_int(axis) and axis != 0 and axis != -1
            for axis in destination
        ):
            return dsl.Invalid("movedim destination dimension out of range")
        concrete_source = tuple((axis for axis in source if dsl.is_concrete_int(axis)))
        concrete_destination = tuple(
            (axis for axis in destination if dsl.is_concrete_int(axis))
        )
        if len(concrete_source) > 1:
            return dsl.Invalid("movedim source dimensions must be unique")
        if len(concrete_destination) > 1:
            return dsl.Invalid("movedim destination dimensions must be unique")
        if any(not dsl.is_concrete_int(axis) for axis in source) or any(
            not dsl.is_concrete_int(axis) for axis in destination
        ):
            return dsl.IntTuple.gradual()
        return shape
    if any(
        dsl.is_concrete_int(axis) and (axis < 0 - rank or axis >= rank)
        for axis in source
    ):
        return dsl.Invalid("movedim source dimension out of range")
    if any(
        dsl.is_concrete_int(axis) and (axis < 0 - rank or axis >= rank)
        for axis in destination
    ):
        return dsl.Invalid("movedim destination dimension out of range")
    concrete_source = tuple(
        (
            axis + rank if axis < 0 else axis
            for axis in source
            if dsl.is_concrete_int(axis)
        )
    )
    concrete_destination = tuple(
        (
            axis + rank if axis < 0 else axis
            for axis in destination
            if dsl.is_concrete_int(axis)
        )
    )
    if any(concrete_source.count(axis) > 1 for axis in concrete_source):
        return dsl.Invalid("movedim source dimensions must be unique")
    if any(concrete_destination.count(axis) > 1 for axis in concrete_destination):
        return dsl.Invalid("movedim destination dimensions must be unique")
    if len(concrete_source) != len(source) or len(concrete_destination) != len(
        destination
    ):
        return dsl.IntTuple.gradual()
    normalized_source = concrete_source
    normalized_destination = concrete_destination
    non_destination = tuple(
        (axis for axis in range(rank) if axis not in normalized_destination)
    )
    remaining = tuple((axis for axis in range(rank) if axis not in normalized_source))
    moved_keys = tuple(
        (
            dst * rank + src
            for src, dst in zip(normalized_source, normalized_destination)
        )
    )
    # Membership guards must short-circuit before calling .index on either tuple.
    permutation = tuple(
        (
            pair % rank
            for pair in range(rank * rank)
            if pair in moved_keys
            or (
                pair // rank in non_destination
                and pair % rank in remaining
                and non_destination.index(pair // rank) == remaining.index(pair % rank)
            )
        )
    )
    return dsl.IntTuple((shape[axis] for axis in permutation))

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
    # TODO(stroxler): Preserve symbolic configuration values instead of returning a gradual shape
    # when the DSL can represent their arithmetic and range constraints.
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

@type_shape_dsl_function
def tile_shape(shape: IntTuple, repeats: IntTuple) -> IntTuple:
    if len(repeats) >= len(shape):
        return repeat_shape(shape, repeats)
    extra = len(shape) - len(repeats)
    return dsl.IntTuple(
        (
            shape[index] if index < extra else shape[index] * repeats[index - extra]
            for index in range(len(shape))
        )
    )

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

@type_shape_dsl_function
def repeat_interleave_shape(shape: IntTuple, repeats: Int, dim: int | None) -> IntTuple:
    # A concrete negative count has no valid extent, so it is rejected ahead of every
    # multiplication below; a symbolic count has no decidable sign and stays exact. An
    # Int dimension can only be compared against a literal as a tuple element, so the
    # count is wrapped in a singleton before the sign test.
    if any(
        dsl.is_concrete_int(count) and count < 0 for count in dsl.IntTuple((repeats,))
    ):
        return dsl.Invalid("repeat_interleave repeats must be non-negative")
    if dim is None:
        return dsl.IntTuple((dsl.prod(shape) * repeats,))
    if dsl.is_int_value(dim):
        if len(shape) == 0:
            # A rank-0 input still produces the rank-1 flattened result, and only 0 and
            # -1 name that synthesized axis.
            if dim == 0 or dim == -1:
                return dsl.IntTuple((repeats,))
            return dsl.Invalid("repeat_interleave dimension out of range")
        if dim < 0 - len(shape) or dim >= len(shape):
            return dsl.Invalid("repeat_interleave dimension out of range")
        return dsl.IntTuple(
            (
                shape[index] * repeats
                if index == (dim + len(shape) if dim < 0 else dim)
                else shape[index]
                for index in range(len(shape))
            )
        )
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def repeat_interleave_checked_shape(
    shape: IntTuple, repeats: Int, output_size: Int, dim: int | None
) -> IntTuple:
    if any(
        dsl.is_concrete_int(count) and count < 0 for count in dsl.IntTuple((repeats,))
    ):
        return dsl.Invalid("repeat_interleave repeats must be non-negative")
    if any(
        dsl.is_concrete_int(size) and size < 0 for size in dsl.IntTuple((output_size,))
    ):
        return dsl.Invalid("repeat_interleave output_size must be non-negative")
    if dim is None:
        extent = dsl.prod(shape) * repeats
        if (
            dsl.is_concrete_int(extent)
            and dsl.is_concrete_int(output_size)
            and extent != output_size
        ):
            return dsl.Invalid(
                "repeat_interleave output_size does not match the result"
            )
        return repeat_interleave_output_shape(shape, output_size, dim)
    if dsl.is_int_value(dim):
        if len(shape) == 0:
            if dim == 0 or dim == -1:
                if (
                    dsl.is_concrete_int(repeats)
                    and dsl.is_concrete_int(output_size)
                    and repeats != output_size
                ):
                    return dsl.Invalid(
                        "repeat_interleave output_size does not match the result"
                    )
            return repeat_interleave_output_shape(shape, output_size, dim)
        if dim < 0 - len(shape) or dim >= len(shape):
            return repeat_interleave_output_shape(shape, output_size, dim)
        extent = shape[dim + len(shape) if dim < 0 else dim] * repeats
        if (
            dsl.is_concrete_int(extent)
            and dsl.is_concrete_int(output_size)
            and extent != output_size
        ):
            return dsl.Invalid(
                "repeat_interleave output_size does not match the result"
            )
    return repeat_interleave_output_shape(shape, output_size, dim)

@type_shape_dsl_function
def repeat_interleave_output_shape(
    shape: IntTuple, output_size: Int, dim: int | None
) -> IntTuple:
    if any(
        dsl.is_concrete_int(size) and size < 0 for size in dsl.IntTuple((output_size,))
    ):
        return dsl.Invalid("repeat_interleave output_size must be non-negative")
    if dim is None:
        return dsl.IntTuple((output_size,))
    if dsl.is_int_value(dim):
        if len(shape) == 0:
            if dim == 0 or dim == -1:
                return dsl.IntTuple((output_size,))
            return dsl.Invalid("repeat_interleave dimension out of range")
        if dim < 0 - len(shape) or dim >= len(shape):
            return dsl.Invalid("repeat_interleave dimension out of range")
        return dsl.IntTuple(
            (
                output_size
                if index == (dim + len(shape) if dim < 0 else dim)
                else shape[index]
                for index in range(len(shape))
            )
        )
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def arange_extent(end: Int) -> Int:
    # Construct zero in the `Int` domain so it can be passed as the starting dimension.
    origin = end - end
    unit_step = 1
    return arange_step_extent(origin, end, unit_step)

@type_shape_dsl_function
def arange_step_extent(start: Int, end: Int, step: int) -> Int:
    # TODO(stroxler): Implement symbolic ceiling division. The truncating fallback is exact only
    # when `step` divides the range.
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
    # TODO(stroxler): Preserve symbolic offset and dimension values instead of returning gradual
    # when the DSL can represent their ordering constraints.
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

@type_shape_dsl_function
def pool_shape(
    input: IntTuple,
    spatial_dims: int,
    kernel_size: int | tuple[int, ...] | None,
    stride: int | tuple[int, ...] | None,
    padding: int | tuple[int, ...] | None,
    dilation: int | tuple[int, ...] | None,
    ceil_mode: bool,
) -> IntTuple:
    rank = len(input)
    if rank != spatial_dims + 1 and rank != spatial_dims + 2:
        return dsl.Invalid("pooling requires spatial rank + 1 or + 2 input")
    # A scalar argument applies to every axis, so it normalizes to a fixed tuple; an
    # omitted stride pools with adjacent windows of the normalized kernel. Only the
    # stride is optional: the DSL narrows an argument to its sequence shape solely by
    # ruling `None` out first, so every argument spells `None` and the ones that have
    # no omitted meaning reject it.
    if kernel_size is None:
        return dsl.Invalid("pooling kernel cannot be None")
    elif dsl.is_int_value(kernel_size):
        kernels = tuple((kernel_size for _ in range(spatial_dims)))
    else:
        kernels = kernel_size
    if stride is None:
        strides = kernels
    elif dsl.is_int_value(stride):
        strides = tuple((stride for _ in range(spatial_dims)))
    else:
        strides = stride
    if padding is None:
        return dsl.Invalid("pooling padding cannot be None")
    elif dsl.is_int_value(padding):
        paddings = tuple((padding for _ in range(spatial_dims)))
    else:
        paddings = padding
    if dilation is None:
        return dsl.Invalid("pooling dilation cannot be None")
    elif dsl.is_int_value(dilation):
        dilations = tuple((dilation for _ in range(spatial_dims)))
    else:
        dilations = dilation
    if len(kernels) != spatial_dims:
        return dsl.Invalid("pooling kernel must match the spatial rank")
    if len(strides) != spatial_dims:
        return dsl.Invalid("pooling stride must match the spatial rank")
    if len(paddings) != spatial_dims:
        return dsl.Invalid("pooling padding must match the spatial rank")
    if len(dilations) != spatial_dims:
        return dsl.Invalid("pooling dilation must match the spatial rank")
    # Every check below is a direct predicate rather than one gated on concreteness:
    # a value the checker cannot decide makes the whole call recover gradually. The
    # DSL is not re-evaluated after a type parameter is specialized, so a deferred
    # expression would be arithmetic that no validation ever revisits.
    if any(size < 1 for size in kernels):
        return dsl.Invalid("pooling kernel must be positive")
    if any(step < 1 for step in strides):
        return dsl.Invalid("pooling stride must be positive")
    if any(pad < 0 for pad in paddings):
        return dsl.Invalid("pooling padding must be nonnegative")
    if any(rate < 1 for rate in dilations):
        return dsl.Invalid("pooling dilation must be positive")
    # ATen caps padding at half of the raw kernel. That bound is what keeps the ceil
    # correction below dividing by 1 or 2 rather than by zero. `any` cannot iterate a
    # `zip`, so the per-axis slack is materialized first.
    slack = dsl.IntTuple((size - 2 * pad for size, pad in zip(kernels, paddings)))
    if any(value < 0 for value in slack):
        return dsl.Invalid("pooling padding must be at most half the kernel size")
    input_spatial = input[rank - spatial_dims :]
    if ceil_mode and any(not dsl.is_concrete_int(extent) for extent in input_spatial):
        # The ceil correction expands rapidly when composed, so keep its rank while
        # making only the spatial dimensions gradual.
        return dsl.concat(
            input[: rank - spatial_dims],
            dsl.IntTuple((dsl.Int.gradual() for _index in range(spatial_dims))),
        )
    # `ceil_mode` rounds the window count up, but ATen drops a final window that
    # starts inside the padding; valid padding makes the naive ceil result exceed
    # that last-window limit by at most one, so the correction is 1 // (2 - excess).
    spatial = dsl.IntTuple(
        (
            (
                (extent + 2 * pad - (rate * (size - 1) + 1) + step - 1) // step
                + 1
                - 1
                // (
                    2
                    - (
                        (extent + 2 * pad - (rate * (size - 1) + 1) + step - 1) // step
                        + 1
                        - ((extent + pad - 1) // step + 1)
                    )
                )
            )
            if ceil_mode
            else (extent + 2 * pad - (rate * (size - 1) + 1)) // step + 1
            for extent, size, step, pad, rate in zip(
                input_spatial, kernels, strides, paddings, dilations
            )
        )
    )
    # TODO(stroxler): Validate output positivity once the DSL can prove symbolic inequalities
    # without discarding the computed shape formula.
    if rank == spatial_dims + 1:
        return dsl.concat(input[:1], spatial)
    else:
        return dsl.concat(input[:2], spatial)

@type_shape_dsl_function
def adaptive_pool1d_shape(input_shape: IntTuple, output: Int) -> IntTuple:
    if len(input_shape) != 2 and len(input_shape) != 3:
        return dsl.Invalid("adaptive_pool1d requires 2D or 3D input")
    return dsl.concat(input_shape[:-1], dsl.IntTuple((output,)))

@type_shape_dsl_function
def adaptive_pool2d_shape(input_shape: IntTuple, height: Int, width: Int) -> IntTuple:
    if len(input_shape) != 3 and len(input_shape) != 4:
        return dsl.Invalid("adaptive_pool2d requires 3D or 4D input")
    return dsl.concat(input_shape[:-2], dsl.IntTuple((height, width)))

@type_shape_dsl_function
def adaptive_pool3d_shape(
    input_shape: IntTuple, depth: Int, height: Int, width: Int
) -> IntTuple:
    if len(input_shape) != 4 and len(input_shape) != 5:
        return dsl.Invalid("adaptive_pool3d requires 4D or 5D input")
    return dsl.concat(input_shape[:-3], dsl.IntTuple((depth, height, width)))

@type_shape_dsl_function
def adaptive_pool_gradual_shape(
    input_shape: IntTuple, spatial_dimensions: int
) -> IntTuple:
    if spatial_dimensions == 1:
        if len(input_shape) != 2 and len(input_shape) != 3:
            return dsl.Invalid("adaptive_pool1d requires 2D or 3D input")
    elif spatial_dimensions == 2:
        if len(input_shape) != 3 and len(input_shape) != 4:
            return dsl.Invalid("adaptive_pool2d requires 3D or 4D input")
    elif spatial_dimensions == 3:
        if len(input_shape) != 4 and len(input_shape) != 5:
            return dsl.Invalid("adaptive_pool3d requires 4D or 5D input")
    else:
        return dsl.Invalid("adaptive pooling supports one to three spatial dimensions")
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def interpolate_scalar_shape(
    input: IntTuple, size: Int | None, scale_factor: Int | None
) -> IntTuple:
    # Positivity is deliberately unchecked here, unlike in the tuple helpers
    # below: a direct predicate would make every symbolic scalar argument
    # undecidable and therefore gradual, which is the precision this arm exists
    # to preserve. Torch validates the value at runtime.
    rank = len(input)
    if rank < 3 or rank > 5:
        return dsl.Invalid("interpolate requires rank 3, 4, or 5")
    if size is not None and scale_factor is not None:
        return dsl.Invalid("interpolate accepts only one of size or scale_factor")
    elif size is not None:
        output = dsl.IntTuple((size for _ in range(rank - 2)))
    elif scale_factor is not None:
        output = dsl.IntTuple((input[i + 2] * scale_factor for i in range(rank - 2)))
    else:
        return dsl.Invalid("interpolate requires size or scale_factor")
    return dsl.concat(input[:2], output)

@type_shape_dsl_function
def interpolate_size_shape(input: IntTuple, size: IntTuple) -> IntTuple:
    rank = len(input)
    if rank < 3 or rank > 5:
        return dsl.Invalid("interpolate requires rank 3, 4, or 5")
    if len(size) != rank - 2:
        return dsl.Invalid("interpolate size must match the spatial rank")
    # A direct predicate, not one gated on concreteness: an entry the checker
    # cannot decide makes the whole call recover gradually. The DSL is not
    # re-evaluated once a type parameter is specialized, so gating would leave
    # a shape that no validation ever revisits.
    if any(dim < 1 for dim in size):
        return dsl.Invalid("interpolate size must be positive")
    return dsl.concat(input[:2], size)

@type_shape_dsl_function
def interpolate_scale_shape(input: IntTuple, scale_factor: IntTuple) -> IntTuple:
    rank = len(input)
    if rank < 3 or rank > 5:
        return dsl.Invalid("interpolate requires rank 3, 4, or 5")
    if len(scale_factor) != rank - 2:
        return dsl.Invalid("interpolate scale_factor must match the spatial rank")
    # Direct predicate for the same reason as in `interpolate_size_shape`: an
    # undecidable factor must not be multiplied into a shape that is never
    # re-checked once the factor is known.
    if any(factor < 1 for factor in scale_factor):
        return dsl.Invalid("interpolate scale_factor must be positive")
    spatial = dsl.IntTuple((input[i] for i in range(2, rank)))
    output = dsl.IntTuple((dim * factor for dim, factor in zip(spatial, scale_factor)))
    return dsl.concat(input[:2], output)

# Reduction precedence shared by every `torch.nn.functional` loss: the legacy
# `reduce`/`size_average` flags override `reduction`, in that order. `unreduced_shape`
# is the loss family's result before reduction; it is not necessarily the input shape.
@type_shape_dsl_function
def loss_shape(
    unreduced_shape: IntTuple,
    reduction: str,
    size_average: bool | None,
    reduce: bool | None,
) -> IntTuple:
    if reduce is None:
        if size_average is None:
            if reduction == "none":
                return unreduced_shape
            if reduction == "mean" or reduction == "sum":
                return dsl.IntTuple(())
            return dsl.Invalid("loss reduction must be 'none', 'mean', or 'sum'")
        return dsl.IntTuple(())
    if not reduce:
        return unreduced_shape
    return dsl.IntTuple(())

# NLL and cross-entropy score one class dimension away: `(N, C, *D)` becomes `(N, *D)`,
# and an unbatched `(C,)` input becomes a scalar.
@type_shape_dsl_function
def classification_loss_shape(
    input_shape: IntTuple,
    reduction: str,
    size_average: bool | None,
    reduce: bool | None,
) -> IntTuple:
    if len(input_shape) == 0:
        return dsl.Invalid("classification loss requires a class dimension")
    if len(input_shape) == 1:
        scalar = dsl.IntTuple(())
        return loss_shape(scalar, reduction, size_average, reduce)
    scored = dsl.concat(input_shape[:1], input_shape[2:])
    return loss_shape(scored, reduction, size_average, reduce)

# Pairwise distance broadcasts its operands, then removes the trailing feature dimension.
@type_shape_dsl_function
def pairwise_distance_shape(
    left_shape: IntTuple, right_shape: IntTuple, broadcast_shape: IntTuple
) -> IntTuple:
    if len(left_shape) == 0:
        return dsl.Invalid("triplet_margin_loss requires at least 1D input")
    if len(left_shape) != len(right_shape):
        return dsl.Invalid("triplet_margin_loss inputs must have the same rank")
    return broadcast_shape[:-1]

# Cosine-embedding loss accepts either two vectors with a scalar target or two
# matrices with a one-dimensional target.
@type_shape_dsl_function
def cosine_embedding_score_shape(
    input1_shape: IntTuple,
    input2_shape: IntTuple,
    broadcast_shape: IntTuple,
    target_shape: IntTuple,
) -> IntTuple:
    if len(input1_shape) != len(input2_shape):
        return dsl.Invalid("cosine_embedding_loss inputs must have the same rank")
    if len(input1_shape) == 1:
        if len(target_shape) != 0:
            return dsl.Invalid(
                "cosine_embedding_loss requires a scalar target for 1D inputs"
            )
    elif len(input1_shape) == 2:
        if len(target_shape) != 1:
            return dsl.Invalid(
                "cosine_embedding_loss requires a 1D target for 2D inputs"
            )
    else:
        return dsl.Invalid("cosine_embedding_loss requires 1D or 2D inputs")
    return broadcast_shape[:-1]

# KL divergence adds `batchmean`, which is always a scalar.
@type_shape_dsl_function
def kl_div_loss_shape(
    input_shape: IntTuple,
    reduction: str,
    size_average: bool | None,
    reduce: bool | None,
) -> IntTuple:
    if reduce is None and size_average is None and reduction == "batchmean":
        return dsl.IntTuple(())
    return loss_shape(input_shape, reduction, size_average, reduce)

# `padding` holds `(before, after)` amounts for trailing dimensions, innermost pair
# first, so dimension `i` picks up the pair at offset `(rank - 1 - i) * 2`.
@type_shape_dsl_function
def _pad_shape(shape: IntTuple, padding: IntTuple) -> IntTuple:
    rank = len(shape)
    if len(padding) == 0:
        return shape
    num_pad_dims = len(padding) // 2
    if num_pad_dims * 2 != len(padding):
        return dsl.Invalid("pad must have an even number of entries")
    if rank == 0:
        return dsl.Invalid("pad does not support scalar input")
    if num_pad_dims > rank:
        return dsl.Invalid("pad has more padding pairs than input dimensions")
    return dsl.IntTuple(
        (
            shape[i] + padding[(rank - 1 - i) * 2] + padding[(rank - 1 - i) * 2 + 1]
            if i >= rank - num_pad_dims
            else shape[i]
            for i in range(rank)
        )
    )

# `len` and indexing need an `IntTuple` parameter, so the Flag tuple value is
# rebuilt as one before `_pad_shape` can inspect it.
@type_shape_dsl_function
def pad_shape(shape: IntTuple, pad: tuple[int, ...]) -> IntTuple:
    padding = dsl.IntTuple((item for item in pad))
    return _pad_shape(shape, padding)

@type_shape_dsl_function
def symmetric_pad2d_shape(input: IntTuple, padding: int) -> IntTuple:
    if len(input) != 3 and len(input) != 4:
        return dsl.Invalid("2D padding requires 3D or 4D input")
    return dsl.IntTuple(
        (
            input[index] + 2 * padding if index >= len(input) - 2 else input[index]
            for index in range(len(input))
        )
    )

@type_shape_dsl_function
def pixel_shuffle_shape(input: IntTuple, upscale_factor: Int) -> IntTuple:
    if len(input) < 3:
        return dsl.Invalid("PixelShuffle requires at least 3D input")
    if any(
        dsl.is_concrete_int(factor) and factor <= 0
        for factor in dsl.IntTuple((upscale_factor,))
    ):
        return dsl.Invalid("PixelShuffle upscale_factor must be positive")
    channels = input[-3]
    if (
        dsl.is_concrete_int(channels)
        and channels % (upscale_factor * upscale_factor) != 0
    ):
        return dsl.Invalid(
            "PixelShuffle input channels must be divisible by upscale_factor squared"
        )
    return dsl.concat(
        input[:-3],
        dsl.IntTuple(
            (
                channels // (upscale_factor * upscale_factor),
                input[-2] * upscale_factor,
                input[-1] * upscale_factor,
            )
        ),
    )

@type_shape_dsl_function
def glu_shape(input: IntTuple, dim: int) -> IntTuple:
    if dim < 0 - len(input) or dim >= len(input):
        return dsl.Invalid("GLU dimension out of range")
    extent = input[dim]
    if dsl.is_concrete_int(extent) and extent % 2 != 0:
        return dsl.Invalid("GLU input dimension must be even")
    halved = extent // 2
    return replace_axis_extent(input, dim, halved)

# `n` defaults to the existing extent of the transformed axis, so `None` and an
# explicit length differ only in which value feeds the halved output extent.
@type_shape_dsl_function
def rfft_shape(shape: IntTuple, n: Int | None, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    if n is None:
        transformed = dsl.IntTuple((shape[axis] // 2 + 1,))
    else:
        transformed = dsl.IntTuple((n // 2 + 1,))
    return dsl.concat(dsl.concat(shape[:axis], transformed), shape[axis + 1 :])

# The inverse transform undoes the halving: without `n` it reconstructs the even
# signal length, and with `n` the requested length becomes the axis extent.
@type_shape_dsl_function
def irfft_shape(shape: IntTuple, n: Int | None, dim: int) -> IntTuple:
    rank = len(shape)
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT dimension out of range")
    if n is None:
        transformed = dsl.IntTuple((2 * (shape[axis] - 1),))
    else:
        transformed = dsl.IntTuple((n,))
    return dsl.concat(dsl.concat(shape[:axis], transformed), shape[axis + 1 :])

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
    # TODO(stroxler): Preserve products of symbolic-rank shapes and derived symbolic dimensions
    # instead of returning a gradual `Int` when the dimension representation can express them.
    return dsl.prod(shape)

@type_shape_dsl_function
def dim_shape(shape: IntTuple) -> Int:
    return len(shape)

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
