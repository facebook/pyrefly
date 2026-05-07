# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for symbolic dimension support
# Week 2-3: Systematic tests with TypeVar-based dimension variables
# Using modern Python 3.12+ generic syntax: def f[N]

from typing import assert_type, reveal_type

import torch
import torch.fft
import torch.linalg
import torch.nn.functional as F
from torch import Tensor

# ==== Week 2: Symbolic Dimension Tests ====


def accepts_symbolic_returns_symbolic[N](x: Tensor[N, 3]) -> Tensor[N, 3]:
    """Identity function with symbolic dimension - preserves shape"""
    return x


def test_symbolic_identity():
    """Verify symbolic dimensions are accepted and preserved through function calls"""
    # Create concrete tensor
    x_concrete: Tensor[2, 3] = torch.randn(2, 3)

    # Call function with symbolic signature: Tensor[N, 3] -> Tensor[N, 3]
    # At call site, N binds to 2
    result = accepts_symbolic_returns_symbolic(x_concrete)

    # Result type: Tensor[2, 3] (N=2 substituted)
    assert_type(result, Tensor[2, 3])


def concat_symbolic[N, M](x: Tensor[N, 3], y: Tensor[M, 3]) -> Tensor[N + M, 3]:
    """Concat with symbolic dimension addition: N + M"""
    return torch.cat([x, y], dim=0)


def test_concat_adds_dimensions():
    """ConcatMetaShape should produce N + M expression in output type"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    y: Tensor[5, 3] = torch.randn(5, 3)

    # Call symbolic function: N=2, M=5
    # Return type: Tensor[N + M, 3] with N=2, M=5 → Tensor[7, 3]
    z = concat_symbolic(x, y)

    assert_type(z, Tensor[7, 3])


def flatten_symbolic[B, N, M](x: Tensor[B, N, M]) -> Tensor[B * N * M]:
    """Flatten with symbolic dimension multiplication"""
    return x.flatten()


def test_flatten_multiplies_dimensions():
    """FlattenMetaShape should produce B * N * M expression"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)

    # Call symbolic function: B=2, N=3, M=4
    # Return type: Tensor[B * N * M] with substitution → Tensor[24]
    y = flatten_symbolic(x)

    assert_type(y, Tensor[24])


def tile_symbolic[N](x: Tensor[N, 3]) -> Tensor[N * 2, 3]:
    """Tile multiplies dimensions"""
    return x.tile((2, 1))


def test_tile_multiplies_dimension():
    """TileMetaShape should compute N * 2"""
    x: Tensor[5, 3] = torch.randn(5, 3)

    # Call: N=5, return Tensor[N * 2] = Tensor[10, 3]
    y = tile_symbolic(x)

    assert_type(y, Tensor[10, 3])


def repeat_symbolic[N, M](x: Tensor[N, M]) -> Tensor[N * 2, M * 3]:
    """Repeat multiplies each dimension"""
    return x.repeat(2, 3)


def test_repeat_multiplies_dimensions():
    """RepeatMetaShape should compute N * 2, M * 3"""
    x: Tensor[4, 5] = torch.randn(4, 5)

    # Call: N=4, M=5, return Tensor[8, 15]
    y = repeat_symbolic(x)

    assert_type(y, Tensor[8, 15])


def process_batch[B, D](x: Tensor[B, D]) -> Tensor[B, D]:
    """Identity operation preserves symbolic dimensions"""
    return torch.relu(x)


def test_identity_preserves_symbolic():
    """Identity operations should preserve symbolic dimensions"""
    x: Tensor[32, 512] = torch.randn(32, 512)

    # Call: B=32, D=512, return Tensor[32, 512]
    y = process_batch(x)

    assert_type(y, Tensor[32, 512])


# ==== More Symbolic Tests ====


def transpose_symbolic[N, M](x: Tensor[N, M]) -> Tensor[M, N]:
    """Transpose swaps symbolic dimensions"""
    return x.transpose(0, 1)


def test_transpose_swaps_symbolic():
    """Transpose should swap dimension positions"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    y = transpose_symbolic(x)
    assert_type(y, Tensor[5, 3])


def permute_symbolic[N, M, K](x: Tensor[N, M, K]) -> Tensor[K, N, M]:
    """Permute with symbolic dimensions"""
    # Permute (2, 0, 1) reorders [N, M, K] → [K, N, M]
    return x.permute(2, 0, 1)


def test_permute_reorders_symbolic():
    """Permute correctly reorders symbolic dimensions"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    y = permute_symbolic(x)
    # Should return Tensor[4, 2, 3] (reordered from [2, 3, 4])
    assert_type(y, Tensor[4, 2, 3])


def reduce_symbolic[N, M](x: Tensor[N, M]) -> Tensor[N]:
    """Reduce along dimension removes it"""
    result = torch.sum(x, dim=1)
    # Returns Tensor[N] (removed M dimension)
    return result


def test_reduce_removes_dimension():
    """Reduction should remove the reduced dimension"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    y = reduce_symbolic(x)
    assert_type(y, Tensor[3])


def matmul_symbolic[B, N, M, K](
    a: Tensor[B, N, M], b: Tensor[B, M, K]
) -> Tensor[B, N, K]:
    """MatMul with symbolic batch dimension"""
    return torch.matmul(a, b)


def test_matmul_symbolic_batch():
    """MatMul preserves batch, combines inner dimensions"""
    a: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    b: Tensor[2, 4, 5] = torch.randn(2, 4, 5)
    c = matmul_symbolic(a, b)
    assert_type(c, Tensor[2, 3, 5])


def mixed_literal_symbolic[N](x: Tensor[N, 28, 28]) -> Tensor[N * 784]:
    """Mix symbolic batch with literal spatial dims"""
    # flatten() flattens all dims: N * 28 * 28 = N * 784
    # Note: Might produce (N * 28) * 28 which needs better simplification
    return x.flatten()


def test_mixed_literal_symbolic():
    """Mixing symbolic and literal dimensions"""
    x: Tensor[4, 28, 28] = torch.randn(4, 28, 28)
    y = mixed_literal_symbolic(x)
    # N=4, 28*28=784, so N*784 = 4*784 = 3136
    assert_type(y, Tensor[3136])


# ==== Complex Expressions ====


def nested_operations[N, M](x: Tensor[N, 3], y: Tensor[M, 3]) -> Tensor[(N + M) * 3]:
    """Nested expressions: concat then flatten"""
    concatenated = torch.cat([x, y], dim=0)  # Tensor[N + M, 3]
    flattened = concatenated.flatten()  # Tensor[(N + M) * 3]
    return flattened


def test_nested_expressions():
    """Complex expressions with concat and flatten"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    y: Tensor[5, 3] = torch.randn(5, 3)
    z = nested_operations(x, y)
    # (N + M) * 3 with N=2, M=5 → (2 + 5) * 3 = 21
    assert_type(z, Tensor[21])


# ==== Future: Size/Numel with Symbolic (requires Literal subscript parsing) ====

# These would demonstrate Dim[N] and Literal[N*M] return types
# Commented out until parser supports TypeVar inside Literal subscript

# def get_size_symbolic[N, M](x: Tensor[N, M]) -> tuple[Dim[N], Dim[M]]:
#     """Size returns tuple of symbolic dimensions as Literal types"""
#     return x.size()

# def get_numel_symbolic[N, M](x: Tensor[N, M]) -> Literal[N * M]:
#     """Numel returns symbolic product as Literal type"""
#     return x.numel()

# def test_size_returns_symbolic():
#     x: Tensor[3, 4] = torch.randn(3, 4)
#     size = get_size_symbolic(x)
#     expected: tuple[Literal[3], Literal[4]] = size

# def test_numel_returns_symbolic_product():
#     x: Tensor[3, 4] = torch.randn(3, 4)
#     n = get_numel_symbolic(x)
#     expected: Literal[12] = n


# ==== Shape Manipulation Tests ====


def squeeze_symbolic[N](x: Tensor[N, 1, 3]) -> Tensor[N, 3]:
    """Squeeze removes size-1 dimensions"""
    return x.squeeze(1)


def test_squeeze_symbolic():
    """Squeeze should remove dimension"""
    x: Tensor[5, 1, 3] = torch.randn(5, 1, 3)
    y = squeeze_symbolic(x)
    assert_type(y, Tensor[5, 3])


def unsqueeze_symbolic[N, M](x: Tensor[N, M]) -> Tensor[N, 1, M]:
    """Unsqueeze adds size-1 dimension"""
    return x.unsqueeze(1)


def test_unsqueeze_symbolic():
    """Unsqueeze should add dimension"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    y = unsqueeze_symbolic(x)
    assert_type(y, Tensor[3, 1, 4])


def select_symbolic[N, M](x: Tensor[N, M, 3]) -> Tensor[N, 3]:
    """Select with symbolic dimensions"""
    # Select along dim=1 removes that dimension
    return x.select(1, 0)


def test_select_symbolic():
    """Select correctly removes selected dimension with symbolic dims"""
    x: Tensor[2, 5, 3] = torch.randn(2, 5, 3)
    y = select_symbolic(x)
    # Should return Tensor[2, 3] (removed middle dimension)
    assert_type(y, Tensor[2, 3])


# ==== Reduction with keepdim ====


def reduce_keepdim[N, M](x: Tensor[N, M]) -> Tensor[N, 1]:
    """Reduction with keepdim preserves rank"""
    return torch.sum(x, dim=1, keepdim=True)


def test_reduce_keepdim_symbolic():
    """Reduction with keepdim replaces dimension with 1"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    y = reduce_keepdim(x)
    assert_type(y, Tensor[3, 1])


def min_with_dim[N, M](x: Tensor[N, M]) -> tuple[Tensor[N], Tensor[N]]:
    """Min with dim returns tuple (values, indices)"""
    return torch.min(x, dim=1)


def test_min_tuple_symbolic():
    """Min with dim should return tuple"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    values, indices = min_with_dim(x)
    assert_type(values, Tensor[3])
    assert_type(indices, Tensor[3])


# ==== Linear Algebra with Symbolic Batch ====


def mm_symbolic[N, M, K](a: Tensor[N, M], b: Tensor[M, K]) -> Tensor[N, K]:
    """Matrix multiply without batch"""
    return torch.mm(a, b)


def test_mm_symbolic():
    """mm combines dimensions"""
    a: Tensor[3, 4] = torch.randn(3, 4)
    b: Tensor[4, 5] = torch.randn(4, 5)
    c = mm_symbolic(a, b)
    assert_type(c, Tensor[3, 5])


def bmm_symbolic[B, N, M, K](a: Tensor[B, N, M], b: Tensor[B, M, K]) -> Tensor[B, N, K]:
    """Batched mm preserves batch dimension"""
    return torch.bmm(a, b)


def test_bmm_symbolic():
    """bmm with symbolic batch"""
    a: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    b: Tensor[2, 4, 5] = torch.randn(2, 4, 5)
    c = bmm_symbolic(a, b)
    assert_type(c, Tensor[2, 3, 5])


# ==== Broadcasting ====


def broadcast_add[N, M](x: Tensor[N, M], y: Tensor[N, M]) -> Tensor[N, M]:
    """Element-wise addition preserves shape"""
    return x + y


def test_broadcast_add_symbolic():
    """Broadcasting operations preserve symbolic dimensions"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    y: Tensor[3, 4] = torch.randn(3, 4)
    z = broadcast_add(x, y)
    assert_type(z, Tensor[3, 4])


# ==== Stack (adds dimension) ====


def stack_symbolic[N](x: Tensor[N, 3], y: Tensor[N, 3]) -> Tensor[2, N, 3]:
    """Stack adds new dimension"""
    return torch.stack([x, y], dim=0)


def test_stack_symbolic():
    """Stack should add dimension"""
    x: Tensor[5, 3] = torch.randn(5, 3)
    y: Tensor[5, 3] = torch.randn(5, 3)
    z = stack_symbolic(x, y)
    assert_type(z, Tensor[2, 5, 3])


# ==== Indexing ====


def index_select_symbolic[N](x: Tensor[N, 10], indices: Tensor[5]) -> Tensor[N, 5]:
    """Index select replaces dimension with index count"""
    return torch.index_select(x, dim=1, index=indices)


def test_index_select_symbolic():
    """Index select with symbolic dimensions"""
    x: Tensor[3, 10] = torch.randn(3, 10)
    # Create indices tensor (note: torch.tensor might not be in stubs)
    indices: Tensor[5] = torch.randn(5)  # Using randn instead
    y = index_select_symbolic(x, indices)
    assert_type(y, Tensor[3, 5])


# ==== Reshape with Symbolic (Future Work) ====

# Note: Reshape with symbolic dimensions requires connecting the type-level expressions
# to runtime values extracted via size(). This requires additional flow analysis.
# For now, reshape works when called with explicit literal arguments.

# Future enhancement:
# def reshape_symbolic[N, M](x: Tensor[N, M]) -> Tensor[N * M]:
#     n, m = x.size()  # Extract runtime values
#     return x.reshape(n * m)  # Type system should know n*m = N*M
#
# This requires: Type system to track that size() extracts match type-level dimensions

# ==== Convolution with Symbolic Spatial Dimensions (Week 3) ====


def conv2d_symbolic[H, W](
    x: Tensor[1, 3, H, W], weight: Tensor[64, 3, 3, 3]
) -> Tensor[1, 64, H, W]:
    """Conv2d with symbolic spatial dimensions and padding=1 preserves size"""
    import torch.nn.functional as F

    # With kernel=3, stride=1, padding=1: output size = input size
    return F.conv2d(x, weight, stride=1, padding=1)


def test_conv2d_preserves_spatial_symbolic():
    """Conv2d with padding=1, kernel=3, stride=1 preserves spatial dims"""
    x: Tensor[1, 3, 224, 224] = torch.randn(1, 3, 224, 224)
    weight: Tensor[64, 3, 3, 3] = torch.randn(64, 3, 3, 3)
    y = conv2d_symbolic(x, weight)
    # H=224, W=224 preserved
    assert_type(y, Tensor[1, 64, 224, 224])


def conv2d_stride2[H, W](x: Tensor[1, 3, H, W], weight: Tensor[64, 3, 3, 3]) -> Tensor:
    """Conv2d with stride=2 (formula needs symbolic computation)"""
    import torch.nn.functional as F

    # With kernel=3, stride=2, padding=1:
    # output = floor((H + 2*1 - 1*(3-1) - 1) / 2) + 1
    #        = floor((H + 2 - 2 - 1) / 2) + 1
    #        = floor((H - 1) / 2) + 1
    # This requires symbolic division - returns shapeless for now
    return F.conv2d(x, weight, stride=2, padding=1)


def test_conv2d_stride2_symbolic():
    """Conv2d with stride=2 and symbolic (currently returns shapeless)"""
    x: Tensor[1, 3, 224, 224] = torch.randn(1, 3, 224, 224)
    weight: Tensor[64, 3, 3, 3] = torch.randn(64, 3, 3, 3)
    y = conv2d_stride2(x, weight)
    # Should compute: (224 + 2 - 2 - 1) / 2 + 1 = 111.5 → 112
    # For now, verifying it type-checks with shapeless
    assert_type(y, Tensor)


# ==== P0: Testing Updated Meta-Shapes (Week 3 Verification) ====


def pool_symbolic[H, W](x: Tensor[1, 3, H, W]) -> Tensor[1, 3, H, W]:
    """Max pool with padding=1, kernel=3, stride=1 preserves size"""
    import torch.nn.functional as F

    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def test_pool_symbolic():
    """Verify PoolMetaShape works with symbolic"""
    x: Tensor[1, 3, 56, 56] = torch.randn(1, 3, 56, 56)
    y = pool_symbolic(x)
    assert_type(y, Tensor[1, 3, 56, 56])


def pad_symbolic[H, W](x: Tensor[1, 3, H, W]) -> Tensor[1, 3, H + 4, W + 6]:
    """Pad with symbolic dimensions"""
    import torch.nn.functional as F

    # Pad (left=3, right=3, top=2, bottom=2) adds 6 to width, 4 to height
    return F.pad(x, (3, 3, 2, 2))


def test_pad_symbolic():
    """Verify PadMetaShape correctly adds padding to symbolic dimensions"""
    x: Tensor[1, 3, 28, 28] = torch.randn(1, 3, 28, 28)
    y = pad_symbolic(x)
    # Should return Tensor[1, 3, 32, 34] (added 4 to height, 6 to width)
    assert_type(y, Tensor[1, 3, 32, 34])


# ==== P1: Spot-Check Tests for Identity Operations ====
# Verifying that ~100 identity operations work with symbolic dimensions


def test_identity_gelu[N, M](x: Tensor[N, M]):
    """GELU activation preserves symbolic shape"""
    y = F.gelu(x)
    assert_type(y, Tensor[N, M])


def test_identity_sin[N, M](x: Tensor[N, M]):
    """Sin preserves symbolic shape"""
    y = torch.sin(x)
    assert_type(y, Tensor[N, M])


def test_identity_exp[N, M](x: Tensor[N, M]):
    """Exp preserves symbolic shape"""
    y = torch.exp(x)
    assert_type(y, Tensor[N, M])


def test_identity_log[N, M](x: Tensor[N, M]):
    """Log preserves symbolic shape"""
    y = torch.log(x)
    assert_type(y, Tensor[N, M])


def test_identity_sqrt[N, M](x: Tensor[N, M]):
    """Sqrt preserves symbolic shape"""
    y = torch.sqrt(x.abs())  # abs to avoid sqrt of negative
    assert_type(y, Tensor[N, M])


def test_identity_abs[N, M](x: Tensor[N, M]):
    """Abs preserves symbolic shape"""
    y = torch.abs(x)
    assert_type(y, Tensor[N, M])


def test_identity_neg[N, M](x: Tensor[N, M]):
    """Negation preserves symbolic shape"""
    y = torch.neg(x)
    assert_type(y, Tensor[N, M])


def test_identity_tanh[N, M](x: Tensor[N, M]):
    """Tanh preserves symbolic shape"""
    y = torch.tanh(x)
    assert_type(y, Tensor[N, M])


def test_identity_floor[N, M](x: Tensor[N, M]):
    """Floor preserves symbolic shape"""
    y = torch.floor(x)
    assert_type(y, Tensor[N, M])


def test_identity_clamp[N, M](x: Tensor[N, M]):
    """Clamp preserves symbolic shape"""
    y = torch.clamp(x, min=-1.0, max=1.0)
    assert_type(y, Tensor[N, M])


# Test all identity operations with concrete tensor
_test_tensor = torch.randn(3, 4)
test_identity_gelu(_test_tensor)
test_identity_sin(_test_tensor)
test_identity_exp(_test_tensor)
test_identity_log(_test_tensor.abs())  # abs for log
test_identity_sqrt(_test_tensor)
test_identity_abs(_test_tensor)
test_identity_neg(_test_tensor)
test_identity_tanh(_test_tensor)
test_identity_floor(_test_tensor)
test_identity_clamp(_test_tensor)

# ==== P1: Spot-Check Tests for Binary Operations ====


def test_binary_sub[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Subtraction preserves symbolic shape"""
    z = torch.sub(x, y)
    assert_type(z, Tensor[N, M])


def test_binary_div[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Division preserves symbolic shape"""
    z = torch.div(x, y)
    assert_type(z, Tensor[N, M])


def test_binary_pow[N, M](x: Tensor[N, M]):
    """Power preserves symbolic shape"""
    z = torch.pow(x, 2.0)
    assert_type(z, Tensor[N, M])


def test_binary_eq[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Equality comparison preserves symbolic shape"""
    z = torch.eq(x, y)
    assert_type(z, Tensor[N, M])


def test_binary_lt[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Less than preserves symbolic shape"""
    z = torch.lt(x, y)
    assert_type(z, Tensor[N, M])


def test_binary_logical_and[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Logical AND preserves symbolic shape"""
    # Create bool tensors first
    x_bool: Tensor[N, M] = x.abs()
    y_bool: Tensor[N, M] = y.abs()
    z = torch.logical_and(x_bool, y_bool)
    assert_type(z, Tensor[N, M])


def test_binary_maximum[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Element-wise maximum preserves symbolic shape"""
    z = torch.maximum(x, y)
    assert_type(z, Tensor[N, M])


def test_binary_mul_operator[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Multiplication operator (* ) preserves symbolic shape"""
    z = x * y
    assert_type(z, Tensor[N, M])


def test_binary_hypot[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Hypot preserves symbolic shape"""
    z = torch.hypot(x, y)
    assert_type(z, Tensor[N, M])


def test_binary_fmax[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Fmax preserves symbolic shape"""
    z = torch.fmax(x, y)
    assert_type(z, Tensor[N, M])


# Test all binary operations with concrete tensors
_t1 = torch.randn(3, 4)
_t2 = torch.randn(3, 4)
test_binary_sub(_t1, _t2)
test_binary_div(_t1, _t2.abs())
test_binary_pow(_t1.abs())
test_binary_eq(_t1, _t2)
test_binary_lt(_t1, _t2)
test_binary_logical_and(_t1, _t2)
test_binary_maximum(_t1, _t2)
test_binary_mul_operator(_t1, _t2)
test_binary_hypot(_t1, _t2)
test_binary_fmax(_t1, _t2)

# ==== P1: Spot-Check Tests for Reductions ====


def test_reduce_prod[N, M](x: Tensor[N, M]):
    """Product reduction removes dimension"""
    y = torch.prod(x, dim=1)
    assert_type(y, Tensor[N])


def test_reduce_var[N, M](x: Tensor[N, M]):
    """Variance reduction removes dimension"""
    y = torch.var(x, dim=1)
    assert_type(y, Tensor[N])


def test_reduce_argmax[N, M](x: Tensor[N, M]):
    """Argmax reduction removes dimension"""
    y = torch.argmax(x, dim=1)
    assert_type(y, Tensor[N])


def test_reduce_argmin[N, M](x: Tensor[N, M]):
    """Argmin reduction removes dimension"""
    y = torch.argmin(x, dim=0)
    assert_type(y, Tensor[M])


def test_reduce_logsumexp[N, M](x: Tensor[N, M]):
    """Logsumexp reduction removes dimension"""
    y = torch.logsumexp(x, dim=1)
    assert_type(y, Tensor[N])


# Test all reduction operations with concrete tensor
_test_tensor_34 = torch.randn(3, 4)
test_reduce_prod(_test_tensor_34)
test_reduce_var(_test_tensor_34)
test_reduce_argmax(_test_tensor_34)
test_reduce_argmin(_test_tensor_34)
test_reduce_logsumexp(_test_tensor_34)

# ==== P1 Priority 1: Test Untested Variants from Tested Meta-Shapes ====


def test_max_with_dim[N, M](x: Tensor[N, M]):
    """Max with dim returns tuple (like min)"""
    values, indices = torch.max(x, dim=1)
    assert_type(values, Tensor[N])
    assert_type(indices, Tensor[N])


def test_median_with_dim[N, M](x: Tensor[N, M]):
    """Median with dim returns tuple (like min/max)"""
    values, indices = torch.median(x, dim=1)
    assert_type(values, Tensor[N])
    assert_type(indices, Tensor[N])


def test_cumsum[N, M](x: Tensor[N, M]):
    """Cumsum preserves shape"""
    y = torch.cumsum(x, dim=1)
    assert_type(y, Tensor[N, M])


def test_cumprod[N, M](x: Tensor[N, M]):
    """Cumprod preserves shape"""
    y = torch.cumprod(x, dim=0)
    assert_type(y, Tensor[N, M])


def test_cummax[N, M](x: Tensor[N, M]):
    """Cummax returns tuple and preserves shape"""
    values, indices = torch.cummax(x, dim=1)
    assert_type(values, Tensor[N, M])
    assert_type(indices, Tensor[N, M])


def test_cummin[N, M](x: Tensor[N, M]):
    """Cummin returns tuple and preserves shape"""
    values, indices = torch.cummin(x, dim=0)
    assert_type(values, Tensor[N, M])
    assert_type(indices, Tensor[N, M])


def test_mode[N, M](x: Tensor[N, M]):
    """Mode reduces dimension and returns tuple"""
    values, indices = torch.mode(x, dim=1)
    assert_type(values, Tensor[N])
    assert_type(indices, Tensor[N])


def test_topk[N, M](x: Tensor[N, M]):
    """Topk changes dimension size and returns tuple"""
    values, indices = torch.topk(x, k=3, dim=1)
    # Replaces dim 1 with k=3: [N, M] → [N, 3]
    assert_type(values, Tensor[N, 3])
    assert_type(indices, Tensor[N, 3])


def test_sort[N, M](x: Tensor[N, M]):
    """Sort preserves shape and returns tuple"""
    values, indices = torch.sort(x, dim=1)
    assert_type(values, Tensor[N, M])
    assert_type(indices, Tensor[N, M])


def test_kthvalue[N, M](x: Tensor[N, M]):
    """Kthvalue reduces dimension and returns tuple"""
    values, indices = torch.kthvalue(x, k=3, dim=1)
    # Removes dim 1: [N, M] → [N]
    assert_type(values, Tensor[N])
    assert_type(indices, Tensor[N])


def test_var_mean[N, M](x: Tensor[N, M]):
    """Var_mean reduces dimension and returns tuple"""
    var, mean = torch.var_mean(x, dim=1)
    assert_type(var, Tensor[N])
    assert_type(mean, Tensor[N])


def test_std_mean[N, M](x: Tensor[N, M]):
    """Std_mean reduces dimension and returns tuple"""
    std_val, mean = torch.std_mean(x, dim=1)
    assert_type(std_val, Tensor[N])
    assert_type(mean, Tensor[N])


def test_mv[N, M](mat: Tensor[N, M], vec: Tensor[M]):
    """Matrix-vector multiply"""
    result = torch.mv(mat, vec)
    # [N, M] @ [M] → [N]
    assert_type(result, Tensor[N])


def test_dot[N](x: Tensor[N], y: Tensor[N]):
    """Dot product returns scalar"""
    result = torch.dot(x, y)
    # Scalar output
    assert_type(result, Tensor[()])


def test_all[N, M](x: Tensor[N, M]):
    """All reduces dimension"""
    result = torch.all(x, dim=1)
    assert_type(result, Tensor[N])


def test_any[N, M](x: Tensor[N, M]):
    """Any reduces dimension"""
    result = torch.any(x, dim=0)
    assert_type(result, Tensor[M])


# Test all priority 1 operations with concrete tensors
_t34 = torch.randn(3, 4)
_t310 = torch.randn(3, 10)
_vec4 = torch.randn(4)
_vec5 = torch.randn(5)
test_max_with_dim(_t34)
test_median_with_dim(_t34)
test_cumsum(_t34)
test_cumprod(_t34)
test_cummax(_t34)
test_cummin(_t34)
test_mode(_t34)
test_topk(_t310)
test_sort(_t34)
test_kthvalue(_t310)
test_var_mean(_t34)
test_std_mean(_t34)
test_mv(_t34, _vec4)
test_dot(_vec5, _vec5)
test_all(_t34.abs())
test_any(_t34.abs())

# ==== P1 Priority 3: Test Likely-Working Operations ====


def test_eig[M](A: Tensor[M, M]):
    """Eigenvalue decomposition with symbolic"""
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    assert_type(eigenvalues, Tensor[M])
    assert_type(eigenvectors, Tensor[M, M])


def test_eigh[M](A: Tensor[M, M]):
    """Hermitian eigenvalue decomposition"""
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    assert_type(eigenvalues, Tensor[M])
    assert_type(eigenvectors, Tensor[M, M])


def test_eigvals[M](A: Tensor[M, M]):
    """Eigenvalues only"""
    eigenvalues = torch.linalg.eigvals(A)
    assert_type(eigenvalues, Tensor[M])


def test_cholesky[M](A: Tensor[M, M]):
    """Cholesky preserves shape"""
    L = torch.linalg.cholesky(A)
    assert_type(L, Tensor[M, M])


def test_det[B, M](A: Tensor[B, M, M]):
    """Determinant removes matrix dimensions"""
    d = torch.linalg.det(A)
    # Removes last 2 dims: [B, M, M] → [B]
    assert_type(d, Tensor[B])


def test_trace[B, M](A: Tensor[B, M, M]):
    """Trace removes matrix dimensions"""
    t = torch.trace(A)
    # Removes last 2 dims: [B, M, M] → [B]
    assert_type(t, Tensor[B])


def test_narrow[N, M](x: Tensor[N, M]):
    """Narrow replaces dimension size"""
    y = torch.narrow(x, dim=1, start=5, length=10)
    # Replaces dim 1 with length=10: [N, M] → [N, 10]
    assert_type(y, Tensor[N, 10])


def test_gather[N, M](x: Tensor[N, M], index: Tensor[N, 5]):
    """Gather returns index shape"""
    y = torch.gather(x, dim=1, index=index)
    # Returns index shape: [N, 5]
    assert_type(y, Tensor[N, 5])


def test_where[N, M](condition: Tensor[N, M], x: Tensor[N, M], y: Tensor[N, M]):
    """Where preserves shape"""
    result = torch.where(condition, x, y)
    assert_type(result, Tensor[N, M])


def test_zeros_like[N, M](x: Tensor[N, M]):
    """Zeros_like copies shape"""
    y = torch.zeros_like(x)
    assert_type(y, Tensor[N, M])


# Test all priority 3 operations with concrete tensors
_mat55 = torch.randn(5, 5)
_mat255 = torch.randn(2, 5, 5)
_t1020 = torch.randn(10, 20)
_t310_b = torch.randn(3, 10)
_idx35 = torch.randn(3, 5)  # Using randn for index (simplified)
_cond34 = torch.randn(3, 4)
test_eig(_mat55)
test_eigh(_mat55)
test_eigvals(_mat55)
test_cholesky(_mat55)
test_det(_mat255)
test_trace(_mat255)
test_narrow(_t1020)
test_gather(_t310_b, _idx35)
test_where(_cond34, _t34, _t34)
test_zeros_like(_t34)

# ==== P1 Priority 4: Remaining Operations - Creation ====


def test_zeros_like_verified[N, M](x: Tensor[N, M]):
    """Already tested above, adding comment for completeness"""
    y = torch.zeros_like(x)
    assert_type(y, Tensor[N, M])


test_zeros_like_verified(_t34)


def test_ones_like[N, M](x: Tensor[N, M]):
    """Ones_like copies shape"""
    y = torch.ones_like(x)
    assert_type(y, Tensor[N, M])


def test_randn_like[N, M](x: Tensor[N, M]):
    """Randn_like copies shape"""
    y = torch.randn_like(x)
    assert_type(y, Tensor[N, M])


def test_empty_like[N, M](x: Tensor[N, M]):
    """Empty_like copies shape"""
    y = torch.empty_like(x)
    assert_type(y, Tensor[N, M])


# Test creation operations
test_ones_like(_t34)
test_randn_like(_t34)
test_empty_like(_t34)

# Creation ops with explicit sizes - these create from literals, not symbolic
# These are less relevant for symbolic dimension testing since they take literal sizes

# ==== Remaining Indexing Operations ====


def test_scatter[N, M](x: Tensor[N, M], index: Tensor[N, 5], src: Tensor[N, 5]):
    """Scatter preserves input shape"""
    y = torch.scatter(x, dim=1, index=index, src=src)
    assert_type(y, Tensor[N, M])


def test_masked_fill[N, M](x: Tensor[N, M], mask: Tensor[N, M]):
    """Masked_fill preserves shape"""
    y = torch.masked_fill(x, mask, 0.0)
    assert_type(y, Tensor[N, M])


def test_take[N, M](x: Tensor[N, M], index: Tensor[5]):
    """Take returns index shape"""
    y = torch.take(x, index)
    # Returns shape of index: [5]
    assert_type(y, Tensor[5])


def test_take_along_dim[N, M](x: Tensor[N, M], indices: Tensor[N, 5]):
    """Take_along_dim returns index shape"""
    y = torch.take_along_dim(x, indices, dim=1)
    # Returns indices shape: [N, 5]
    assert_type(y, Tensor[N, 5])


def test_index_add[N, M](x: Tensor[N, M], index: Tensor[5], source: Tensor[N, 5]):
    """Index_add preserves shape"""
    y = torch.index_add(x, dim=1, index=index, source=source)
    assert_type(y, Tensor[N, M])


# Test indexing operations
_idx5 = torch.randn(5)
_src35 = torch.randn(3, 5)
test_scatter(_t310_b, _idx35, _src35)
test_masked_fill(_t34, _t34)
test_take(_t34, _idx5)
test_take_along_dim(_t310_b, _idx35)
test_index_add(_t310_b, _idx5, _src35)

# ==== Remaining Dimension Operations ====


def test_movedim[N, M, K](x: Tensor[N, M, K]):
    """Movedim reorders dimensions"""
    y = torch.movedim(x, source=0, destination=2)
    # Moves dim 0 to position 2: [N, M, K] → [M, K, N]
    assert_type(y, Tensor[M, K, N])


def test_expand[N](x: Tensor[N, 1]):
    """Expand to target size"""
    # expand() with runtime values - checking what it actually returns
    n = x.size(0)
    reveal_type(n)
    y = x.expand(n, 5)
    # Expands [N, 1] → [N, 5] (keeps dim 0, broadcasts dim 1)
    assert_type(y, Tensor[N, 5])


def test_t_2d[N, M](x: Tensor[N, M]):
    """2D transpose (t method)"""
    y = x.t()
    assert_type(y, Tensor[M, N])


# Test dimension operations
_t234 = torch.randn(2, 3, 4)
_t31 = torch.randn(3, 1)
test_movedim(_t234)
test_expand(_t31)
test_t_2d(_t34)

# ==== Specialized Operations ====


def test_fft[N](x: Tensor[N]):
    """FFT preserves shape"""
    y = torch.fft.fft(x)
    assert_type(y, Tensor[N])


def test_ifft[N](x: Tensor[N]):
    """Inverse FFT preserves shape"""
    y = torch.fft.ifft(x)
    assert_type(y, Tensor[N])


def test_rfft[N](x: Tensor[N]):
    """Real FFT changes dimension"""
    y = torch.fft.rfft(x)
    assert_type(y, Tensor[N // 2 + 1])


def test_mse_loss[N, M](input: Tensor[N, M], target: Tensor[N, M]):
    """MSE loss reduces to scalar"""
    loss = F.mse_loss(input, target)
    # Default reduction='mean' → scalar
    assert_type(loss, Tensor[()])


def test_adaptive_avg_pool2d[B](x: Tensor[B, 64, 56, 56]):
    """Adaptive pool outputs target size with symbolic batch dimension"""
    y = F.adaptive_avg_pool2d(x, (7, 7))
    # Adaptive pool preserves batch dimension B and outputs literal spatial dims
    assert_type(y, Tensor[B, 64, 7, 7])


def test_diag_embed[B, N](x: Tensor[B, N]):
    """Diag_embed creates matrix from vector"""
    y = torch.diag_embed(x)
    reveal_type(y)
    # Creates diagonal matrix: [B, N] → [B, N, N]
    assert_type(y, Tensor[B, N, N])


def test_norm_symbolic[N, M](x: Tensor[N, M]):
    """Norm with symbolic dimensions"""
    # Frobenius norm reduces to scalar
    n = torch.norm(x)
    assert_type(n, Tensor[()])


def test_dist[N, M](x: Tensor[N, M], y: Tensor[N, M]):
    """Distance returns scalar"""
    d = torch.dist(x, y)
    assert_type(d, Tensor[()])


# Test specialized operations
_t8 = torch.randn(8)
_t10 = torch.randn(10)
_t2645656 = torch.randn(2, 64, 56, 56)
_t25 = torch.randn(2, 5)
test_fft(_t8)
test_ifft(_t8)
test_rfft(_t10)
test_mse_loss(_t34, _t34)
test_adaptive_avg_pool2d(_t2645656)
test_diag_embed(_t25)
test_norm_symbolic(_t34)
test_dist(_t34, _t34)
