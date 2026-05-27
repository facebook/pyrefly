# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from shape_extensions.dsl import shape_dsl_function

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
def remove_dim(dims: list[int | symint], i: int) -> list[int | symint]:
    return dims[:i] + dims[i + 1 :]

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

@shape_dsl_function
def reduce_shape(
    dims: list[int | symint], dim: int | list[int] | None, keepdim: bool
) -> list[int | symint]:
    if dim == None:
        if keepdim:
            return [1 for _ in range(len(dims))]
        return []
    dim_list = dim if isinstance(dim, list) else [dim]
    norm = [normalize_dim(len(dims), d) for d in dim_list]
    return [
        1 if i in norm else elem
        for i, elem in enumerate(dims)
        if not (i in norm) or keepdim
    ]

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
def reshape_ir(self: Tensor, shape: list[int | symint]) -> Tensor:
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
        known = shape_extensions.dsl.prod([d for d in shape if d != -1])
        total = shape_extensions.dsl.prod(self.shape)
        if isinstance(total, int) and isinstance(known, int) and total % known != 0:
            raise Error(
                "could not infer size for dimension -1: expected "
                + str(total)
                + " to be divisible by "
                + str(known)
            )
        return Tensor(shape=[total // known if d == -1 else d for d in shape])
    return Tensor(shape=shape)

@shape_dsl_function
def squeeze_ir(self: Tensor, dim: int | None = None) -> Tensor:
    if dim == None:
        return Tensor(shape=[d for d in self.shape if d != 1])
    idx = normalize_dim(len(self.shape), dim)
    return Tensor(
        shape=[d for i, d in enumerate(self.shape) if not (i == idx and d == 1)]
    )

@shape_dsl_function
def unsqueeze_ir(self: Tensor, dim: int) -> Tensor:
    d = normalize_dim(len(self.shape) + 1, dim)
    return Tensor(shape=insert_dim(self.shape, d, 1))

@shape_dsl_function
def transpose_ir(self: Tensor, dim0: int, dim1: int) -> Tensor:
    rank = len(self.shape)
    d0 = normalize_dim(rank, dim0)
    d1 = normalize_dim(rank, dim1)
    return Tensor(
        shape=[
            self.shape[d1] if i == d0 else self.shape[d0] if i == d1 else d
            for i, d in enumerate(self.shape)
        ]
    )

@shape_dsl_function
def permute_ir(self: Tensor, dims: list[int]) -> Tensor:
    rank = len(self.shape)
    if len(dims) != rank:
        raise Error("permute: expected " + str(rank) + " dims, got " + str(len(dims)))
    return Tensor(shape=[self.shape[normalize_dim(rank, d)] for d in dims])

@shape_dsl_function
def flatten_ir(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    rank = len(self.shape)
    s = normalize_dim(rank, start_dim)
    e = normalize_dim(rank, end_dim)
    return Tensor(
        shape=self.shape[:s]
        + [shape_extensions.dsl.prod(self.shape[s : e + 1])]
        + self.shape[e + 1 :]
    )

@shape_dsl_function
def expand_ir(self: Tensor, sizes: list[int | symint]) -> Tensor:
    return Tensor(shape=[d if t == -1 else t for d, t in zip(self.shape, sizes)])

@shape_dsl_function
def repeat_ir(self: Tensor, sizes: list[int | symint]) -> Tensor:
    return Tensor(shape=[d * r for d, r in zip(self.shape, sizes)])

@shape_dsl_function
def unbind_ir(self: Tensor, dim: int = 0) -> list[Tensor]:
    d = normalize_dim(len(self.shape), dim)
    return [Tensor(shape=remove_dim(self.shape, d)), ...]

@shape_dsl_function
def movedim_ir(
    self: Tensor, source: int | list[int], destination: int | list[int]
) -> Tensor:
    return Tensor(shape=move_dims(self.shape, source, destination, len(self.shape)))

@shape_dsl_function
def unfold_ir(
    self: Tensor, dimension: int, size: int | symint, step: int = 1
) -> Tensor:
    d = normalize_dim(len(self.shape), dimension)
    new_dim = (self.shape[d] - size) // step + 1
    return Tensor(shape=replace_dim(self.shape, d, new_dim) + [size])

@shape_dsl_function
def cat_ir(tensors: list[Tensor], dim: int = 0) -> Tensor:
    first = tensors[0]
    d = normalize_dim(len(first.shape), dim)
    return Tensor(
        shape=[
            shape_extensions.dsl.sum([t.shape[i] for t in tensors])
            if i == d
            else dim_val
            for i, dim_val in enumerate(first.shape)
        ]
    )

@shape_dsl_function
def stack_ir(tensors: list[Tensor], dim: int = 0) -> Tensor:
    first = tensors[0]
    d = normalize_dim(len(first.shape) + 1, dim)
    return Tensor(shape=insert_dim(first.shape, d, len(tensors)))

@shape_dsl_function
def broadcast_to_ir(self: Tensor, shape: list[int | symint]) -> Tensor:
    return Tensor(shape=shape)

@shape_dsl_function
def tile_ir(self: Tensor, dims: list[int]) -> Tensor:
    rank = len(self.shape)
    if len(dims) > rank:
        extra = len(dims) - rank
        return Tensor(
            shape=[r for r in dims[:extra]]
            + [d * r for d, r in zip(self.shape, dims[extra:])]
        )
    return Tensor(shape=[d * r for d, r in zip(self.shape, dims)])

@shape_dsl_function
def select_ir(self: Tensor, dim: int) -> Tensor:
    d = normalize_dim(len(self.shape), dim)
    return Tensor(shape=remove_dim(self.shape, d))

@shape_dsl_function
def narrow_ir(self: Tensor, dim: int, length: int | symint) -> Tensor:
    return Tensor(
        shape=replace_dim(self.shape, normalize_dim(len(self.shape), dim), length)
    )

@shape_dsl_function
def split_ir(
    self: Tensor,
    split_size_or_sections: int | symint | list[int | symint] | None = None,
    dim: int = 0,
) -> list[Tensor]:
    d = normalize_dim(len(self.shape), dim)
    if isinstance(split_size_or_sections, list):
        return [
            Tensor(shape=replace_dim(self.shape, d, section))
            for section in split_size_or_sections
        ]
    if isinstance(split_size_or_sections, int):
        dim_val = self.shape[d]
        if isinstance(dim_val, int):
            count = (dim_val + split_size_or_sections - 1) // split_size_or_sections
            return [
                Tensor(
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
        return [Tensor(shape=replace_dim(self.shape, d, split_size_or_sections)), ...]
    if split_size_or_sections != None:
        quotient = self.shape[d] // split_size_or_sections
        if isinstance(quotient, int):
            return [
                Tensor(shape=replace_dim(self.shape, d, split_size_or_sections))
                for _ in range(quotient)
            ]
        return [Tensor(shape=replace_dim(self.shape, d, split_size_or_sections)), ...]
    return Unknown

@shape_dsl_function
def chunk_ir(self: Tensor, chunks: int, dim: int = 0) -> list[Tensor]:
    d = normalize_dim(len(self.shape), dim)
    dim_val = self.shape[d]
    if isinstance(dim_val, int):
        chunk_size = (dim_val + chunks - 1) // chunks
        return [
            Tensor(
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
        Tensor(shape=replace_dim(self.shape, d, dim_val // chunks))
        for i in range(chunks)
    ]

@shape_dsl_function
def index_select_ir(self: Tensor, dim: int, index: Tensor) -> Tensor:
    return Tensor(
        shape=replace_dim(
            self.shape, normalize_dim(len(self.shape), dim), index.shape[0]
        )
    )

@shape_dsl_function
def reduce_ir(
    self: Tensor, dim: int | list[int] | None = None, keepdim: bool = False
) -> Tensor:
    if dim == None:
        return Tensor(shape=reduce_shape(self.shape, dim, keepdim))
    if isinstance(dim, list):
        return Tensor(shape=reduce_shape(self.shape, dim, keepdim))
    return Tensor(shape=reduce_single(self.shape, dim, keepdim))

@shape_dsl_function
def reduce_single(
    dims: list[int | symint], dim: int, keepdim: bool
) -> list[int | symint]:
    before = dims[:dim]
    if dim == -1:
        if keepdim:
            return before + [1]
        return before
    after = dims[dim + 1 :]
    if keepdim:
        return before + [1] + after
    return before + after

@shape_dsl_function
def min_max_median_ir(
    self: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor:
    if dim == None:
        return Tensor(shape=[])
    s = reduce_shape(self.shape, dim, keepdim)
    return [Tensor(shape=s), Tensor(shape=s)]

@shape_dsl_function
def aminmax_ir(
    self: Tensor, dim: int | list[int] | None = None, keepdim: bool = False
) -> [Tensor, Tensor]:
    s = reduce_shape(self.shape, dim, keepdim)
    return [Tensor(shape=s), Tensor(shape=s)]

@shape_dsl_function
def tuple_reduce_ir(
    self: Tensor, dim: int = -1, keepdim: bool = False
) -> [Tensor, Tensor]:
    s = reduce_shape(self.shape, dim, keepdim)
    return [Tensor(shape=s), Tensor(shape=s)]

@shape_dsl_function
def topk_ir(self: Tensor, k: int | symint, dim: int = -1) -> [Tensor, Tensor]:
    s = replace_dim(self.shape, normalize_dim(len(self.shape), dim), k)
    return [Tensor(shape=s), Tensor(shape=s)]

@shape_dsl_function
def repeat_interleave_ir(
    self: Tensor, repeats: int | symint, dim: int | None = None
) -> Tensor:
    if dim == None:
        return Tensor(shape=[shape_extensions.dsl.prod(self.shape) * repeats])
    d = normalize_dim(len(self.shape), dim)
    return Tensor(shape=replace_dim(self.shape, d, self.shape[d] * repeats))

@shape_dsl_function
def cosine_similarity_ir(x1: Tensor, x2: Tensor, dim: int = 1) -> Tensor:
    s = broadcast(x1.shape, x2.shape)
    return Tensor(shape=reduce_single(s, normalize_dim(len(s), dim), False))

@shape_dsl_function
def randn_ir(size: list[int | symint]) -> Tensor:
    return Tensor(shape=size)

@shape_dsl_function
def randint_ir(low: int, high: int, size: list[int | symint]) -> Tensor:
    return Tensor(shape=size)

@shape_dsl_function
def linspace_ir(steps: int | symint) -> Tensor:
    return Tensor(shape=[steps])

@shape_dsl_function
def eye_ir(n: int | symint, m: int | symint | None = None) -> Tensor:
    if m == None:
        return Tensor(shape=[n, n])
    return Tensor(shape=[n, m])

@shape_dsl_function
def arange_ir(
    start: int | symint | None = None,
    end: int | symint | None = None,
    step: int | symint | None = None,
) -> Tensor:
    if start != None and end != None and step != None:
        return Tensor(shape=[(end - start) // step])
    if start != None and end != None:
        return Tensor(shape=[end - start])
    if end != None:
        return Tensor(shape=[end])
    if start != None:
        return Tensor(shape=[start])
    return Unknown

@shape_dsl_function
def normal_ir(
    mean: Tensor | None = None, std: Tensor | None = None, size: list[int] | None = None
) -> Tensor:
    if size != None:
        return Tensor(shape=[s for s in size])
    if mean != None:
        return Tensor(shape=mean.shape)
    if std != None:
        return Tensor(shape=std.shape)
    return Unknown

@shape_dsl_function
def diag_embed_ir(self: Tensor, offset: int = 0) -> Tensor:
    new_dim = self.shape[-1] + (offset if offset >= 0 else -offset)
    return Tensor(shape=self.shape[:-1] + [new_dim, new_dim])

@shape_dsl_function
def tri_indices_ir(row: int | symint, col: int | symint, offset: int = 0) -> Tensor:
    return Tensor(shape=[2, 0])

@shape_dsl_function
def matmul_ir(self: Tensor, other: Tensor) -> Tensor:
    r1 = len(self.shape)
    r2 = len(other.shape)
    if r1 == 1 and r2 == 1:
        return Tensor(shape=[])
    if r1 == 1 and r2 == 2:
        return Tensor(shape=[other.shape[1]])
    if r1 == 2 and r2 == 1:
        return Tensor(shape=[self.shape[0]])
    if r1 == 2 and r2 == 2:
        return Tensor(shape=[self.shape[0], other.shape[1]])
    if r1 == 2 and r2 >= 3:
        return Tensor(shape=other.shape[:-2] + [self.shape[0]] + [other.shape[-1]])
    if r1 >= 3 and r2 == 2:
        return Tensor(shape=self.shape[:-2] + [self.shape[-2]] + [other.shape[1]])
    if r1 >= 3 and r2 >= 3:
        return Tensor(
            shape=broadcast(self.shape[:-2], other.shape[:-2])
            + [self.shape[-2]]
            + [other.shape[-1]]
        )
    return Unknown

@shape_dsl_function
def mv_ir(self: Tensor, vec: Tensor) -> Tensor:
    if len(self.shape) != 2:
        raise Error("mv expects 2D matrix, got " + str(len(self.shape)) + "D tensor")
    if len(vec.shape) != 1:
        raise Error("mv expects 1D vector, got " + str(len(vec.shape)) + "D tensor")
    return Tensor(shape=[self.shape[0]])

@shape_dsl_function
def outer_ir(self: Tensor, vec2: Tensor) -> Tensor:
    if len(self.shape) != 1 or len(vec2.shape) != 1:
        raise Error(
            "outer expects 1D tensors, got "
            + str(len(self.shape))
            + "D and "
            + str(len(vec2.shape))
            + "D"
        )
    return Tensor(shape=[self.shape[0], vec2.shape[0]])

@shape_dsl_function
def tensordot_ir(self: Tensor, other: Tensor, dims: int) -> Tensor:
    return Tensor(shape=self.shape[: len(self.shape) - dims] + other.shape[dims:])

@shape_dsl_function
def apply_einsum(
    output_map: list[list[int]], check_pairs: list[list[int]], inputs: list[Tensor]
) -> Tensor:
    bad_dims = [
        1
        for i0, d0, i1, d1 in check_pairs
        if isinstance(inputs[i0].shape[d0], int)
        and isinstance(inputs[i1].shape[d1], int)
        and inputs[i0].shape[d0] != inputs[i1].shape[d1]
    ]
    if len(bad_dims) > 0:
        raise Error("einsum: inconsistent dimensions for repeated index")
    return Tensor(shape=[inputs[inp].shape[dim] for inp, dim in output_map])

@shape_dsl_function
def einsum_ir(spec: str, operands: list[Tensor] | None = None) -> Tensor:
    if operands != None:
        output_map, check_pairs = shape_extensions.dsl.parse_einsum_equation(spec)
        return apply_einsum(output_map, check_pairs, operands)
    return Unknown

@shape_dsl_function
def eigvals_ir(self: Tensor) -> Tensor:
    if len(self.shape) < 2:
        raise Error(
            "eigvals requires at least 2D input, got "
            + str(len(self.shape))
            + "D tensor"
        )
    return Tensor(shape=self.shape[:-2] + [self.shape[-2]])

@shape_dsl_function
def eig_ir(self: Tensor) -> [Tensor, Tensor]:
    if len(self.shape) < 2:
        raise Error(
            "eig requires at least 2D input, got " + str(len(self.shape)) + "D tensor"
        )
    batch = self.shape[:-2]
    return [
        Tensor(shape=batch + [self.shape[-2]]),
        Tensor(shape=batch + self.shape[-2:]),
    ]

@shape_dsl_function
def slogdet_ir(self: Tensor) -> [Tensor, Tensor]:
    if len(self.shape) < 2:
        raise Error(
            "slogdet requires at least 2D input, got "
            + str(len(self.shape))
            + "D tensor"
        )
    return [Tensor(shape=self.shape[:-2]), Tensor(shape=self.shape[:-2])]

@shape_dsl_function
def solve_ir(self: Tensor, other: Tensor) -> Tensor:
    return Tensor(shape=other.shape)

@shape_dsl_function
def solve_reversed_ir(self: Tensor, other: Tensor) -> Tensor:
    return Tensor(shape=self.shape)

@shape_dsl_function
def conv_ir(
    self: Tensor,
    weight: Tensor,
    stride: int | list[int] = 1,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
) -> Tensor:
    spatial_dims = len(self.shape) - 2
    stride_list = broadcast_int(stride, spatial_dims)
    padding_list = broadcast_int(padding, spatial_dims)
    dilation_list = broadcast_int(dilation, spatial_dims)
    return Tensor(
        shape=[self.shape[0], weight.shape[0]]
        + [
            conv_spatial_out(s, k, st, p, dil)
            for s, k, st, p, dil in zip(
                self.shape[2:],
                weight.shape[2:],
                stride_list,
                padding_list,
                dilation_list,
            )
        ]
    )

@shape_dsl_function
def conv_transpose_ir(
    self: Tensor,
    weight: Tensor,
    stride: int | list[int] = 1,
    padding: int | list[int] = 0,
    output_padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
) -> Tensor:
    spatial_dims = len(self.shape) - 2
    stride_list = broadcast_int(stride, spatial_dims)
    padding_list = broadcast_int(padding, spatial_dims)
    outpad_list = broadcast_int(output_padding, spatial_dims)
    dilation_list = broadcast_int(dilation, spatial_dims)
    return Tensor(
        shape=[self.shape[0], weight.shape[1]]
        + [
            (s - 1) * st - 2 * p + dil * (k - 1) + op + 1
            for s, k, st, p, op, dil in zip(
                self.shape[2:],
                weight.shape[2:],
                stride_list,
                padding_list,
                outpad_list,
                dilation_list,
            )
        ]
    )

@shape_dsl_function
def pool_ir(
    self: Tensor,
    kernel_size: int | list[int],
    stride: int | list[int] | None = None,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
    return_indices: bool = False,
) -> Tensor:
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
        return [Tensor(shape=out), Tensor(shape=out)]
    return Tensor(shape=out)

@shape_dsl_function
def adaptive_pool_ir(
    self: Tensor, output_size: int | symint | list[int | symint]
) -> Tensor:
    out_sizes = broadcast_int(output_size, len(self.shape) - 2)
    return Tensor(shape=[self.shape[0], self.shape[1]] + out_sizes)

@shape_dsl_function
def interpolate_ir(
    self: Tensor,
    size: int | symint | list[int | symint] | None = None,
    scale_factor: int | symint | None = None,
) -> Tensor:
    if size != None:
        return Tensor(
            shape=[self.shape[0], self.shape[1]]
            + broadcast_int(size, len(self.shape) - 2)
        )
    if scale_factor != None:
        return Tensor(
            shape=[self.shape[0], self.shape[1]]
            + [d * scale_factor for d in self.shape[2:]]
        )
    raise Error("interpolate requires either 'size' or 'scale_factor' argument")

@shape_dsl_function
def loss_ir(self: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "none":
        return Tensor(shape=self.shape)
    return Tensor(shape=[])

@shape_dsl_function
def pad_ir(self: Tensor, pad: list[int]) -> Tensor:
    rank = len(self.shape)
    num_pad_dims = len(pad) // 2
    offsets = [
        pad[(rank - 1 - i) * 2] + pad[(rank - 1 - i) * 2 + 1]
        if i >= rank - num_pad_dims
        else 0
        for i in range(rank)
    ]
    return Tensor(shape=[d + offsets[i] for i, d in enumerate(self.shape)])

@shape_dsl_function
def rfft_ir(self: Tensor, n: int | symint | None = None, dim: int = -1) -> Tensor:
    d = normalize_dim(len(self.shape), dim)
    if n != None:
        return Tensor(shape=replace_dim(self.shape, d, n // 2 + 1))
    return Tensor(shape=replace_dim(self.shape, d, self.shape[d] // 2 + 1))

@shape_dsl_function
def irfft_ir(self: Tensor, n: int | symint | None = None, dim: int = -1) -> Tensor:
    d = normalize_dim(len(self.shape), dim)
    if n != None:
        return Tensor(shape=replace_dim(self.shape, d, n))
    return Tensor(shape=replace_dim(self.shape, d, 2 * (self.shape[d] - 1)))

@shape_dsl_function
def size_ir(self: Tensor, dim: int | None = None) -> int | symint:
    if dim != None:
        return self.shape[normalize_dim(len(self.shape), dim)]
    return [d for d in self.shape]

@shape_dsl_function
def numel_ir(self: Tensor) -> int | symint:
    return shape_extensions.dsl.prod(self.shape)

@shape_dsl_function
def dim_ir(self: Tensor) -> int:
    return len(self.shape)

@shape_dsl_function
def item_ir(self: Tensor) -> Tensor:
    if len(self.shape) != 0:
        raise Error(
            "item() only works on 0-dimensional tensors, got "
            + str(len(self.shape))
            + "D tensor"
        )
    return Unknown

@shape_dsl_function
def tolist_ir(self: Tensor) -> Tensor:
    return Unknown

@shape_dsl_function
def multinomial_ir(self: Tensor, num_samples: int | symint) -> Tensor:
    return Tensor(shape=self.shape[:-1] + [num_samples])

@shape_dsl_function
def where_ir(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return Tensor(shape=x.shape)

@shape_dsl_function
def take_along_dim_ir(self: Tensor, indices: Tensor) -> Tensor:
    return Tensor(shape=indices.shape)

@shape_dsl_function
def nn_flatten_forward_ir(
    input: Tensor, start_dim: symint = 1, end_dim: symint = -1
) -> Tensor:
    return flatten_ir(input, start_dim, end_dim)

@shape_dsl_function
def nn_maxpool_forward_ir(
    input: Tensor,
    kernel_size: symint = 1,
    stride: symint | None = None,
    padding: symint = 0,
    dilation: symint = 1,
) -> Tensor:
    return pool_ir(input, kernel_size, stride, padding, dilation)

@shape_dsl_function
def nn_avgpool_forward_ir(
    input: Tensor,
    kernel_size: symint = 1,
    stride: symint | None = None,
    padding: symint = 0,
) -> Tensor:
    return pool_ir(input, kernel_size, stride, padding, 1)

@shape_dsl_function
def nn_upsample_forward_ir(
    input: Tensor, size: symint | None = None, scale_factor: symint | None = None
) -> Tensor:
    return interpolate_ir(input, size, scale_factor)

@shape_dsl_function
def nn_pixel_shuffle_forward_ir(input: Tensor, upscale_factor: symint) -> Tensor:
    r = upscale_factor
    return Tensor(
        shape=[input.shape[0], input.shape[1] // (r * r)]
        + [d * r for d in input.shape[2:]]
    )

@shape_dsl_function
def nn_glu_forward_ir(input: Tensor, dim: symint = 1) -> Tensor:
    rank = len(input.shape)
    d = normalize_dim(rank, dim)
    return Tensor(shape=replace_dim(input.shape, d, input.shape[d] // 2))

@shape_dsl_function
def nn_lstm_forward_ir(
    input: Tensor,
    input_size: symint,
    hidden_size: symint,
    num_layers: symint = 1,
    bidirectional: bool = False,
) -> [Tensor, Tensor, Tensor]:
    nd = 2 if bidirectional else 1
    output = Tensor(shape=[input.shape[0], input.shape[1], hidden_size * nd])
    h_n = Tensor(shape=[num_layers * nd, input.shape[0], hidden_size])
    c_n = Tensor(shape=[num_layers * nd, input.shape[0], hidden_size])
    return [output, h_n, c_n]

@shape_dsl_function
def nn_gru_forward_ir(
    input: Tensor,
    input_size: symint,
    hidden_size: symint,
    num_layers: symint = 1,
    bidirectional: bool = False,
) -> [Tensor, Tensor]:
    nd = 2 if bidirectional else 1
    output = Tensor(shape=[input.shape[0], input.shape[1], hidden_size * nd])
    h_n = Tensor(shape=[num_layers * nd, input.shape[0], hidden_size])
    return [output, h_n]

@shape_dsl_function
def nn_lstmcell_forward_ir(
    input: Tensor, input_size: symint, hidden_size: symint
) -> [Tensor, Tensor]:
    h = Tensor(shape=[input.shape[0], hidden_size])
    c = Tensor(shape=[input.shape[0], hidden_size])
    return [h, c]

@shape_dsl_function
def nn_reflectionpad2d_forward_ir(input: Tensor, padding: symint) -> Tensor:
    return Tensor(
        shape=[
            input.shape[0],
            input.shape[1],
            input.shape[2] + 2 * padding,
            input.shape[3] + 2 * padding,
        ]
    )
