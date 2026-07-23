# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive type stubs for PyTorch with shape inference.

Shape inference is declared via @uses_shape_dsl(ir_fn) decorators. The IR
functions are defined in torch/_shapes.pyi and evaluated by the DSL interpreter
in crates/pyrefly_types/src/meta_shape_dsl.rs.
"""

import builtins
from typing import Any, overload, Self, TYPE_CHECKING

import shape_extensions
from shape_extensions import Elements, IntTuple, IntVar, uses_shape_dsl
from torch._shapes import (
    aminmax_ir,
    arange_ir,
    binary_broadcast_ir,
    broadcast_to_ir,
    cat_ir,
    chunk_ir,
    diag_embed_ir,
    dim_ir,
    eig_ir,
    einsum_ir,
    expand_ir,
    eye_ir,
    flatten_ir,
    index_select_ir,
    item_ir,
    linspace_ir,
    matmul_ir,
    min_max_median_ir,
    movedim_ir,
    multinomial_ir,
    mv_ir,
    narrow_ir,
    normal_ir,
    numel_ir,
    outer_ir,
    permute_ir,
    randint_ir,
    randn_ir,
    reduce_ir,
    repeat_interleave_input_ir,
    repeat_interleave_ir,
    repeat_ir,
    reshape_ir,
    select_ir,
    size_ir,
    slogdet_ir,
    solve_ir,
    solve_reversed_ir,
    split_ir,
    squeeze_ir,
    stack_ir,
    take_along_dim_ir,
    tensordot_ir,
    tile_ir,
    tolist_ir,
    topk_ir,
    transpose_ir,
    tri_indices_ir,
    tuple_reduce_ir,
    unbind_ir,
    unfold_ir,
    unsqueeze_ir,
    where_ir,
)

if TYPE_CHECKING:
    from shape_extensions import Int as _Int

__all__ = ["Tensor"]

type _Shape = IntTuple
type _AnyShape = tuple[Any, ...]

# ============================================================================
# Device Type
# ============================================================================

class device:
    """Represents the device on which a Tensor is or will be allocated."""
    def __init__(self, type: str, index: int = 0) -> None: ...

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

@shape_extensions.shaped_array(shape="Shape")
class Tensor[Shape: _Shape = _AnyShape]:
    """
    PyTorch Tensor with shape type parameter.

    The shape is tracked at the type level, allowing static verification
    of tensor operations.

    Most shape transformations are handled by meta-shape functions registered
    in the type checker, not by explicit type signatures here.
    """

    # ==== Tensor Properties ====
    shape: Shape  # Tensor shape as a tuple
    requires_grad: bool  # Whether gradient tracking is enabled
    device: Any  # Device where tensor is stored (cpu, cuda, etc.)
    dtype: Any  # Data type of tensor elements (float32, int64, etc.)
    ndim: builtins.int  # Number of dimensions
    T: Self  # Transpose property (for 2D tensors). Use .t() method for shape inference.
    real: Self  # Real part of complex tensor (shape-preserving)
    imag: Self  # Imaginary part of complex tensor (shape-preserving)
    # Note: Use .dim() method for rank (ndim removed in favor of dim())
    # ==== Indexing ====
    def __getitem__(
        self: Tensor,
        index: int
        | slice
        | tuple[int | slice | Tensor | list[int] | None, ...]
        | Tensor
        | list[int],
    ) -> Tensor:
        """Index into tensor. Shape inference via meta-shape: torch.Tensor.__getitem__"""
        ...

    def __setitem__(
        self: Tensor,
        index: int
        | slice
        | tuple[int | slice | Tensor | list[int] | None, ...]
        | Tensor
        | list[int],
        value: Tensor | int | float,
    ) -> None:
        """Set values in tensor via indexing. Mutates tensor in-place."""
        ...

    # ==== Matrix Multiplication ====
    # Uses meta-shape for shape inference

    @uses_shape_dsl(matmul_ir)
    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        """Matrix multiplication (@). Shape inference via meta-shape: torch.Tensor.matmul"""
        ...

    # ==== Arithmetic Operations ====

    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __add__(self, other: Tensor) -> Self: ...
    @overload
    def __add__(self, other: float | int) -> Self: ...
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __sub__(self, other: Tensor) -> Self: ...
    @overload
    def __sub__(self, other: float | int) -> Self: ...
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __mul__(self, other: Tensor) -> Self: ...
    @overload
    def __mul__(self, other: float | int) -> Self: ...
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __truediv__(self, other: Tensor) -> Self: ...
    @overload
    def __truediv__(self, other: float | int) -> Self: ...

    # Reverse operations for scalars
    def __radd__(self, other: float | int) -> Self: ...
    def __rsub__(self, other: float | int) -> Self: ...
    def __rmul__(self, other: float | int) -> Self: ...
    def __rtruediv__(self, other: float | int) -> Self: ...
    def __rpow__(self, other: float | int) -> Self: ...

    # Power operations
    def __pow__(self, other: Tensor | float | int) -> Self: ...

    # Unary operations
    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __int__(self) -> builtins.int: ...
    def __len__(self) -> builtins.int: ...

    # ==== Comparison Operations ====

    def __eq__(self, other: Tensor | float | int) -> Self: ...  # type: ignore[override]
    def __ne__(self, other: Tensor | float | int) -> Self: ...  # type: ignore[override]
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __lt__(self, other: Tensor) -> Tensor: ...
    @overload
    def __lt__(self, other: float | int) -> Self: ...
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __le__(self, other: Tensor) -> Tensor: ...
    @overload
    def __le__(self, other: float | int) -> Self: ...
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __gt__(self, other: Tensor) -> Tensor: ...
    @overload
    def __gt__(self, other: float | int) -> Self: ...
    @uses_shape_dsl(binary_broadcast_ir)
    @overload
    def __ge__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ge__(self, other: float | int) -> Self: ...

    # ==== Shape Manipulation Operations ====
    # Handled by meta-shape functions - simplified signatures

    @uses_shape_dsl(reshape_ir)
    @overload
    def reshape(self: Tensor, *shape: int) -> Tensor:
        """Reshape tensor. Shape inference via meta-shape: torch.Tensor.reshape"""
        ...

    @uses_shape_dsl(reshape_ir)
    @overload
    def reshape(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Reshape tensor. Shape inference via meta-shape: torch.Tensor.reshape"""
        ...

    @uses_shape_dsl(reshape_ir)
    @overload
    def view(self: Tensor, *shape: int) -> Tensor:
        """View (alias for reshape). Shape inference via meta-shape: torch.Tensor.view"""
        ...

    @uses_shape_dsl(reshape_ir)
    @overload
    def view(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        """View (alias for reshape). Shape inference via meta-shape: torch.Tensor.view"""
        ...

    @uses_shape_dsl(flatten_ir)
    def flatten(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        """Flatten dimensions. Shape inference via meta-shape: torch.flatten"""
        ...

    @uses_shape_dsl(transpose_ir)
    def transpose(self: Tensor, dim0: int, dim1: int) -> Tensor:
        """Transpose two dimensions. Shape inference via meta-shape: torch.transpose"""
        ...

    @uses_shape_dsl(permute_ir)
    @overload
    def permute(self: Tensor, *dims: int) -> Tensor:
        """Permute dimensions. Shape inference via meta-shape: torch.Tensor.permute"""
        ...

    @overload
    def permute(self: Tensor, dims: tuple[int, ...]) -> Tensor:
        """Permute dimensions. Shape inference via meta-shape: torch.Tensor.permute"""
        ...

    @uses_shape_dsl(squeeze_ir)
    def squeeze(self: Tensor, dim: int | None = None) -> Tensor:
        """Remove dimensions of size 1. Shape inference via meta-shape: torch.squeeze"""
        ...

    @uses_shape_dsl(unsqueeze_ir)
    def unsqueeze(self: Tensor, dim: int) -> Tensor:
        """Add dimension of size 1. Shape inference via meta-shape: torch.unsqueeze"""
        ...

    @uses_shape_dsl(repeat_ir)
    @overload
    def repeat(self: Tensor, *sizes: int) -> Tensor:
        """Repeat tensor. Shape inference via meta-shape: torch.Tensor.repeat"""
        ...

    @overload
    def repeat(self: Tensor, sizes: tuple[int, ...]) -> Tensor:
        """Repeat tensor. Shape inference via meta-shape: torch.Tensor.repeat"""
        ...

    def t[M: IntVar, N: IntVar](self: Tensor[[M, N]]) -> Tensor[[N, M]]:
        """Transpose 2D tensor. Swaps dimensions."""
        ...

    @uses_shape_dsl(expand_ir)
    def expand(self: Tensor, *sizes: int) -> Tensor:
        """Expand tensor. Shape inference via meta-shape: torch.Tensor.expand"""
        ...

    def expand_as[S: IntTuple](self: Tensor, other: Tensor[S]) -> Tensor[S]:
        """Expand tensor to match the shape of `other`."""
        ...

    @uses_shape_dsl(repeat_interleave_ir)
    def repeat_interleave(
        self: Tensor,
        repeats: int | Tensor,
        dim: int | None = None,
        *,
        output_size: int | None = None,
    ) -> Tensor:
        """Repeat elements along a dimension.

        Shape inference via DSL (repeat_interleave_ir):
        - dim=None: 1D output of size numel * repeats.
        - dim=D, repeats=int: shape[D] *= repeats, others preserved.
        - repeats=Tensor with output_size: shape[D] = output_size.
        - repeats=Tensor without output_size: falls back to unrefined.
        """
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

    def new_zeros(
        self,
        *size: builtins.int,
        dtype: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
    ) -> Tensor:
        """Create zero-filled tensor with same dtype/device."""
        ...

    def new_ones(
        self,
        *size: builtins.int,
        dtype: Any = None,
        device: Any = None,
        requires_grad: builtins.bool = False,
    ) -> Tensor:
        """Create one-filled tensor with same dtype/device."""
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

    def to(
        self, dtype: Any = None, device: Any = None, non_blocking: builtins.bool = False
    ) -> Self:
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

    @uses_shape_dsl(item_ir)
    def item(self: Tensor) -> float | int:
        """Returns Python scalar from 0-dimensional tensor. Shape inference via meta-shape: torch.Tensor.item"""
        ...

    @uses_shape_dsl(tolist_ir)
    def tolist(self: Tensor) -> Any:
        """Returns tensor as nested Python list. Shape inference via meta-shape: torch.Tensor.tolist"""
        ...

    @uses_shape_dsl(tile_ir)
    def tile(self: Tensor, dims: tuple[int, ...]) -> Tensor:
        """Tile tensor. Shape inference via meta-shape: torch.Tensor.tile"""
        ...

    @uses_shape_dsl(select_ir)
    def select(self: Tensor, dim: int, index: int) -> Tensor:
        """Select along dimension. Shape inference via meta-shape: torch.Tensor.select"""
        ...

    @uses_shape_dsl(narrow_ir)
    def narrow(self: Tensor, dim: int, start: int, length: int) -> Tensor:
        """Narrow tensor along dimension. Shape inference via meta-shape: torch.Tensor.narrow"""
        ...

    @uses_shape_dsl(split_ir)
    @overload
    def split(
        self: Tensor, split_size_or_sections: int, dim: int = 0
    ) -> tuple[Tensor, ...]:
        """Split tensor into chunks. Shape inference via meta-shape: torch.Tensor.split"""
        ...

    @overload
    def split(
        self: Tensor, split_size_or_sections: list[int], dim: int = 0
    ) -> tuple[Tensor, ...]:
        """Split tensor into variable-sized chunks. Shape inference via meta-shape: torch.Tensor.split"""
        ...

    @uses_shape_dsl(chunk_ir)
    def chunk(self: Tensor, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
        """Split tensor into chunks. Shape inference via meta-shape: torch.Tensor.chunk"""
        ...

    @uses_shape_dsl(index_select_ir)
    def index_select(self: Tensor, dim: int, index: Tensor) -> Tensor:
        """Select elements along dimension. Shape inference via meta-shape: torch.Tensor.index_select"""
        ...

    def gather[IndexShape: IntTuple](
        self: Tensor, dim: int, index: Tensor[IndexShape]
    ) -> Tensor[IndexShape]:
        """Gather elements along dimension. Output shape matches index shape."""
        ...

    def scatter[Shape: IntTuple](
        self: Tensor[Shape], dim: int, index: Tensor, src: Tensor
    ) -> Tensor[Shape]:
        """Scatter elements along dimension. Shape-preserving operation."""
        ...

    def masked_select(self: Tensor, mask: Tensor) -> Tensor[[Any]]:
        """Select elements with mask. Returns 1D tensor with data-dependent size."""
        ...

    # ==== Phase 1.1: Missing Shape Operations (Methods) ====

    @uses_shape_dsl(unbind_ir)
    def unbind(self: Tensor, dim: int = 0) -> tuple[Tensor, ...]:
        """Remove dimension by slicing along it. Shape inference via meta-shape: torch.Tensor.unbind"""
        ...

    @uses_shape_dsl(movedim_ir)
    @overload
    def movedim(self: Tensor, source: int, destination: int) -> Tensor:
        """Move single dimension to new position. Shape inference via meta-shape: torch.Tensor.movedim"""
        ...

    @overload
    def movedim(
        self: Tensor, source: tuple[int, ...], destination: tuple[int, ...]
    ) -> Tensor:
        """Move multiple dimensions to new positions. Shape inference via meta-shape: torch.Tensor.movedim"""
        ...

    @uses_shape_dsl(movedim_ir)
    @overload
    def moveaxis(self: Tensor, source: int, destination: int) -> Tensor:
        """Alias for movedim. Shape inference via meta-shape: torch.Tensor.moveaxis"""
        ...

    @overload
    def moveaxis(
        self: Tensor, source: tuple[int, ...], destination: tuple[int, ...]
    ) -> Tensor:
        """Alias for movedim. Shape inference via meta-shape: torch.Tensor.moveaxis"""
        ...

    @uses_shape_dsl(unfold_ir)
    def unfold(self: Tensor, dimension: int, size: int, step: int) -> Tensor:
        """Returns sliding window view. Shape inference via meta-shape: torch.Tensor.unfold"""
        ...

    @uses_shape_dsl(size_ir)
    @overload
    def size(self: Tensor) -> tuple[builtins.int, ...]:
        """Returns the size of the tensor as a tuple. Shape inference via meta-shape: torch.Tensor.size"""
        ...

    @overload
    def size(self: Tensor, dim: builtins.int) -> builtins.int:
        """Returns the size of a specific dimension. Shape inference via meta-shape: torch.Tensor.size"""
        ...

    # ==== Reduction Operations ====
    # Handled by meta-shape functions - simplified signatures

    @uses_shape_dsl(reduce_ir)
    @overload
    def sum(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Sum along dimension(s). Shape inference via meta-shape: torch.Tensor.sum"""
        ...

    @overload
    def sum(self: Tensor, dim: tuple[int, ...], keepdim: bool = False) -> Tensor:
        """Sum along multiple dimensions. Shape inference via meta-shape: torch.Tensor.sum"""
        ...

    @uses_shape_dsl(reduce_ir)
    def mean(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Mean along dimension(s). Shape inference via meta-shape: torch.mean"""
        ...

    @uses_shape_dsl(min_max_median_ir)
    @overload
    def max(self: Tensor) -> Tensor:
        """Max of all elements (scalar). Shape inference via meta-shape: torch.Tensor.max"""
        ...

    @overload
    def max(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
        """Max along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.max"""
        ...

    @uses_shape_dsl(min_max_median_ir)
    @overload
    def min(self: Tensor) -> Tensor:
        """Min of all elements (scalar). Shape inference via meta-shape: torch.Tensor.min"""
        ...

    @overload
    def min(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
        """Min along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.min"""
        ...

    @uses_shape_dsl(reduce_ir)
    def prod(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Product along dimension(s). Shape inference via meta-shape: torch.prod"""
        ...

    @uses_shape_dsl(reduce_ir)
    def std(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Standard deviation along dimension(s). Shape inference via meta-shape: torch.std"""
        ...

    @uses_shape_dsl(reduce_ir)
    def var(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Variance along dimension(s). Shape inference via meta-shape: torch.var"""
        ...

    @uses_shape_dsl(reduce_ir)
    def argmax(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Argmax along dimension(s). Shape inference via meta-shape: torch.argmax"""
        ...

    @uses_shape_dsl(reduce_ir)
    def argmin(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Argmin along dimension(s). Shape inference via meta-shape: torch.argmin"""
        ...

    # ==== Phase 1.2: Missing Reduction Operations (Methods) ====

    @uses_shape_dsl(min_max_median_ir)
    @overload
    def median(self: Tensor) -> Tensor:
        """Median of all elements (scalar). Shape inference via meta-shape: torch.Tensor.median"""
        ...

    @overload
    def median(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
        """Median along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.median"""
        ...

    @uses_shape_dsl(reduce_ir)
    def logsumexp(
        self: Tensor, dim: int | None = None, keepdim: bool = False
    ) -> Tensor:
        """Log-sum-exp along dimension(s). Shape inference via meta-shape: torch.Tensor.logsumexp"""
        ...

    @uses_shape_dsl(reduce_ir)
    def count_nonzero(self: Tensor, dim: int | None = None) -> Tensor:
        """Count non-zero elements. Shape inference via meta-shape: torch.Tensor.count_nonzero"""
        ...

    @uses_shape_dsl(aminmax_ir)
    def aminmax(
        self: Tensor, dim: int | None = None, keepdim: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Min and max along dimension(s). Shape inference via meta-shape: torch.Tensor.aminmax"""
        ...

    @uses_shape_dsl(reduce_ir)
    def norm(
        self: Tensor,
        p: int | float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
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
    ) -> tuple[Tensor[Shape], Tensor[Shape]]:
        """Cumulative maximum along dimension. Returns (values, indices). Shape-preserving operation."""
        ...

    def cummin[Shape: IntTuple](
        self: Tensor[Shape], dim: int
    ) -> tuple[Tensor[Shape], Tensor[Shape]]:
        """Cumulative minimum along dimension. Returns (values, indices). Shape-preserving operation."""
        ...

    # ==== Tier 2: Additional Reduction Methods ====

    @uses_shape_dsl(tuple_reduce_ir)
    def mode(
        self: Tensor, dim: int = -1, keepdim: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Mode along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.mode"""
        ...

    @uses_shape_dsl(topk_ir)
    def topk(
        self: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Top k elements. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.topk"""
        ...

    def sort[Shape: IntTuple](
        self: Tensor[Shape],
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> tuple[Tensor[Shape], Tensor[Shape]]:
        """Sort tensor. Returns (values, indices). Shape-preserving operation."""
        ...

    @uses_shape_dsl(tuple_reduce_ir)
    def kthvalue(
        self: Tensor, k: int, dim: int = -1, keepdim: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Kth smallest value. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.kthvalue"""
        ...

    # ==== Phase 1.3: Tensor Creation Operations (Methods) ====

    @uses_shape_dsl(diag_embed_ir)
    def diag_embed(
        self: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1
    ) -> Tensor:
        """Create diagonal tensor. Shape inference via meta-shape: torch.Tensor.diag_embed"""
        ...

    def tril(self, diagonal: int = 0) -> Self:
        """Lower triangular part. Shape inference via generic fixture signature."""
        ...

    def triu(self, diagonal: int = 0) -> Self:
        """Upper triangular part. Shape inference via generic fixture signature."""
        ...

    # ==== Phase 1.4: Basic Linear Algebra Operations (Methods) ====

    @uses_shape_dsl(matmul_ir)
    def matmul(self: Tensor, other: Tensor) -> Tensor:
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

    @uses_shape_dsl(mv_ir)
    def mv(self: Tensor, vec: Tensor) -> Tensor:
        """Matrix-vector multiplication. Shape inference via meta-shape: torch.Tensor.mv"""
        ...

    def dot(self: Tensor, other: Tensor) -> Tensor[[]]:
        """Dot product. Returns scalar tensor."""
        ...

    # ==== Phase 2: Arithmetic & Basic Operations (Methods) ====

    # Arithmetic methods
    def add(self, other: Tensor) -> Self:
        """Element-wise addition. Shape inference via generic fixture signature."""
        ...

    def sub(self, other: Tensor) -> Self:
        """Element-wise subtraction. Shape inference via generic fixture signature."""
        ...

    def mul(self, other: Tensor) -> Self:
        """Element-wise multiplication. Shape inference via generic fixture signature."""
        ...

    def div(
        self, other: Tensor | int | float, *, rounding_mode: str | None = None
    ) -> Self:
        """Element-wise division. Shape inference via generic fixture signature."""
        ...

    def pow(self, exponent: float | Tensor) -> Self:
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
    def eq(self, other: Tensor) -> Self:
        """Element-wise equality. Shape inference via generic fixture signature."""
        ...

    def ne(self, other: Tensor) -> Self:
        """Element-wise inequality. Shape inference via generic fixture signature."""
        ...

    def lt(self, other: Tensor) -> Self:
        """Element-wise less than. Shape inference via generic fixture signature."""
        ...

    def le(self, other: Tensor) -> Self:
        """Element-wise less than or equal. Shape inference via generic fixture signature."""
        ...

    def gt(self, other: Tensor) -> Self:
        """Element-wise greater than. Shape inference via generic fixture signature."""
        ...

    def ge(self, other: Tensor) -> Self:
        """Element-wise greater than or equal. Shape inference via generic fixture signature."""
        ...

    # Logical methods
    def logical_and(self, other: Tensor) -> Self:
        """Element-wise logical AND. Shape inference via generic fixture signature."""
        ...

    def logical_or(self, other: Tensor) -> Self:
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
    def clamp(self, min: float | None = None, max: float | None = None) -> Self:
        """Clamp tensor values. Shape inference via generic fixture signature."""
        ...

    def clip(self, min: float | None = None, max: float | None = None) -> Self:
        """Alias for clamp. Shape inference via generic fixture signature."""
        ...

    # Additional mathematical methods
    def atan2(self, other: Tensor) -> Self:
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

    def rad2deg(self) -> Self:
        """Convert radians to degrees. Shape inference via generic fixture signature."""
        ...

    # Bitwise methods
    def bitwise_and(self, other: Tensor) -> Self:
        """Bitwise AND. Shape inference via generic fixture signature."""
        ...

    def bitwise_or(self, other: Tensor) -> Self:
        """Bitwise OR. Shape inference via generic fixture signature."""
        ...

    def bitwise_xor(self, other: Tensor) -> Self:
        """Bitwise XOR. Shape inference via generic fixture signature."""
        ...

    def bitwise_not(self) -> Self:
        """Bitwise NOT. Shape inference via generic fixture signature."""
        ...

    def bitwise_left_shift(self, other: Tensor) -> Self:
        """Bitwise left shift. Shape inference via generic fixture signature."""
        ...

    def bitwise_right_shift(self, other: Tensor) -> Self:
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

    def maximum(self, other: Tensor) -> Self:
        """Element-wise maximum. Shape inference via generic fixture signature."""
        ...

    def minimum(self, other: Tensor) -> Self:
        """Element-wise minimum. Shape inference via generic fixture signature."""
        ...

    def fmax(self, other: Tensor) -> Self:
        """Element-wise maximum (NaN handling). Shape inference via generic fixture signature."""
        ...

    def fmin(self, other: Tensor) -> Self:
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

    @uses_shape_dsl(slogdet_ir)
    def slogdet(self: Tensor) -> tuple[Tensor, Tensor]:
        """Sign and log determinant. Shape inference via meta-shape: torch.Tensor.slogdet"""
        ...

    def matrix_power(self, n: int) -> Self:
        """Matrix power. Shape inference via generic fixture signature."""
        ...

    def trace[Batch: IntTuple, M: IntVar, N: IntVar](
        self: Tensor[[*Elements[Batch], M, N]],
    ) -> Tensor[Batch]:
        """Matrix trace. Returns batch dimensions only (drops last 2 dims)."""
        ...

    # ==== Phase 5: Advanced Indexing & Conditional Methods ====

    def masked_fill(self, mask: Tensor, value: float) -> Self:
        """Fill masked elements. Shape inference via generic signature"""
        ...

    def masked_fill_(self, mask: Tensor, value: float) -> Self:
        """Fill masked elements in-place. Shape inference via generic signature"""
        ...

    def masked_scatter(self, mask: Tensor, source: Tensor) -> Self:
        """Scatter into masked positions. Shape inference via generic fixture signature."""
        ...

    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Self:
        """Scatter into masked positions in-place. Shape inference via generic fixture signature."""
        ...

    def index_add(
        self, dim: int, index: Tensor, source: Tensor, alpha: float = 1
    ) -> Self:
        """Add values at indices. Shape inference via generic fixture signature."""
        ...

    def index_add_(
        self, dim: int, index: Tensor, source: Tensor, alpha: float = 1
    ) -> Self:
        """Add values at indices in-place. Shape inference via generic fixture signature."""
        ...

    def index_copy(self, dim: int, index: Tensor, source: Tensor) -> Self:
        """Copy values to indices. Shape inference via generic fixture signature."""
        ...

    def index_copy_(self, dim: int, index: Tensor, source: Tensor) -> Self:
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

    def index_fill(self, dim: int, index: Tensor, value: float) -> Self:
        """Fill indices with value. Shape inference via generic fixture signature."""
        ...

    def index_fill_(self, dim: int, index: Tensor, value: float) -> Self:
        """Fill indices with value in-place. Shape inference via generic fixture signature."""
        ...

    def take[IndexShape: IntTuple](
        self: Tensor, index: Tensor[IndexShape]
    ) -> Tensor[IndexShape]:
        """Take elements at indices. Output shape matches index shape."""
        ...

    @uses_shape_dsl(take_along_dim_ir)
    def take_along_dim(self: Tensor, indices: Tensor, dim: int) -> Tensor:
        """Take along dimension. Shape inference via meta-shape: torch.Tensor.take_along_dim"""
        ...

    def put(self, index: Tensor, source: Tensor, accumulate: bool = False) -> Self:
        """Put values at indices. Shape inference via generic fixture signature."""
        ...

    def put_(self, index: Tensor, source: Tensor, accumulate: bool = False) -> Self:
        """Put values at indices in-place. Shape inference via generic fixture signature."""
        ...

    # ==== Phase 6: Specialized Operations (Methods) ====

    def bernoulli(self, p: float = 0.5) -> Self:
        """Sample from Bernoulli distribution. Shape inference via generic fixture signature."""
        ...

    def bernoulli_(self, p: float = 0.5) -> Self:
        """Sample from Bernoulli distribution in-place. Shape inference via generic fixture signature."""
        ...

    @uses_shape_dsl(multinomial_ir)
    def multinomial(
        self: Tensor, num_samples: int, replacement: bool = False
    ) -> Tensor:
        """Sample from multinomial distribution. Shape inference via meta-shape: torch.Tensor.multinomial"""
        ...

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> Self:
        """Fill with normal distribution in-place. Shape inference via generic fixture signature."""
        ...

    def random_(self, low: int = 0, high: int | None = None) -> Self:
        """Fill with random integers in-place. Shape inference via generic fixture signature."""
        ...

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> Self:
        """Fill with uniform distribution in-place. Shape inference via generic fixture signature."""
        ...

    @uses_shape_dsl(numel_ir)
    def numel(self: Tensor) -> int:
        """Number of elements. Shape inference via meta-shape: torch.Tensor.numel"""
        ...

    @uses_shape_dsl(dim_ir)
    def dim(self: Tensor) -> int:
        """Number of dimensions. Shape inference via meta-shape: torch.Tensor.dim"""
        ...

    @uses_shape_dsl(numel_ir)
    def nelement(self: Tensor) -> int:
        """Number of elements. Shape inference via meta-shape: torch.Tensor.nelement"""
        ...

# ============================================================================
# Module-level Functions
# ============================================================================

@uses_shape_dsl(matmul_ir)
def matmul(self: Tensor, other: Tensor) -> Tensor:
    """Matrix multiplication function. Shape inference via meta-shape: torch.matmul"""
    ...

@uses_shape_dsl(cat_ir)
def cat(tensors: list[Tensor] | tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Concatenate tensors. Shape inference via meta-shape: torch.cat"""
    ...

@uses_shape_dsl(cat_ir)
def concat(tensors: list[Tensor] | tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Alias for concatenate/cat. Shape inference via meta-shape: torch.cat"""
    ...

@uses_shape_dsl(stack_ir)
def stack(tensors: list[Tensor] | tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Stack tensors (adds new dimension)."""
    ...

@uses_shape_dsl(transpose_ir)
def transpose(self: Tensor, dim0: int, dim1: int) -> Tensor:
    """Transpose two dimensions. Shape inference via meta-shape: torch.transpose"""
    ...

def flip[Shape: IntTuple](
    input: Tensor[Shape], dims: int | tuple[int, ...] | list[int]
) -> Tensor[Shape]:
    """Reverse tensor elements along dimensions. Shape-preserving."""
    ...

@uses_shape_dsl(reshape_ir)
def reshape(self: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape tensor. Shape inference via meta-shape: torch.reshape"""
    ...

@uses_shape_dsl(squeeze_ir)
def squeeze(self: Tensor, dim: int | None = None) -> Tensor:
    """Remove dimensions of size 1. Shape inference via meta-shape: torch.squeeze"""
    ...

@uses_shape_dsl(unsqueeze_ir)
def unsqueeze(self: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1. Shape inference via meta-shape: torch.unsqueeze"""
    ...

@uses_shape_dsl(repeat_interleave_input_ir)
def repeat_interleave(
    input: Tensor,
    repeats: int | Tensor,
    dim: int | None = None,
    *,
    output_size: int | None = None,
) -> Tensor:
    """Repeat tensor elements. Shape inference via meta-shape: torch.repeat_interleave"""
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

@uses_shape_dsl(permute_ir)
def permute(self: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Permute dimensions. Shape inference via meta-shape: torch.permute"""
    ...

@uses_shape_dsl(reduce_ir)
def sum(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Sum along dimension(s). Shape inference via meta-shape: torch.sum"""
    ...

@uses_shape_dsl(reduce_ir)
def mean(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Mean along dimension(s). Shape inference via meta-shape: torch.mean"""
    ...

@uses_shape_dsl(min_max_median_ir)
@overload
def max(self: Tensor) -> Tensor:
    """Max of all elements (scalar). Shape inference via meta-shape: torch.max"""
    ...

@overload
def max(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Max along dimension. Returns (values, indices). Shape inference via meta-shape: torch.max"""
    ...

@uses_shape_dsl(min_max_median_ir)
@overload
def min(self: Tensor) -> Tensor:
    """Min of all elements (scalar). Shape inference via meta-shape: torch.min"""
    ...

@overload
def min[S: IntTuple](input: Tensor[S], other: Tensor) -> Tensor[S]:
    """Element-wise minimum of two tensors."""
    ...

@overload
def min(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Min along dimension. Returns (values, indices). Shape inference via meta-shape: torch.min"""
    ...

@uses_shape_dsl(reduce_ir)
def prod(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Product along dimension(s). Shape inference via meta-shape: torch.prod"""
    ...

@uses_shape_dsl(reduce_ir)
def std(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Standard deviation. Shape inference via meta-shape: torch.std"""
    ...

@uses_shape_dsl(reduce_ir)
def var(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Variance. Shape inference via meta-shape: torch.var"""
    ...

@uses_shape_dsl(reduce_ir)
def argmax(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Argmax. Shape inference via meta-shape: torch.argmax"""
    ...

@uses_shape_dsl(reduce_ir)
def argmin(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Argmin. Shape inference via meta-shape: torch.argmin"""
    ...

@uses_shape_dsl(flatten_ir)
def flatten(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten dimensions. Shape inference via meta-shape: torch.flatten"""
    ...

# ==== Tensor Creation Functions ====

@uses_shape_dsl(randn_ir)
@overload
def randn(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values. Shape inference via meta-shape: torch.randn"""
    ...

@overload
def randn(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values (tuple size). Shape inference via meta-shape: torch.randn"""
    ...

@uses_shape_dsl(randn_ir)
@overload
def rand(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values [0, 1). Shape inference via meta-shape: torch.rand"""
    ...

@overload
def rand(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values (tuple size). Shape inference via meta-shape: torch.rand"""
    ...

@uses_shape_dsl(randn_ir)
@overload
def zeros(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with zeros. Shape inference via meta-shape: torch.zeros"""
    ...

@overload
def zeros(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with zeros (tuple size). Shape inference via meta-shape: torch.zeros"""
    ...

@uses_shape_dsl(randn_ir)
@overload
def ones(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with ones. Shape inference via meta-shape: torch.ones"""
    ...

@overload
def ones(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with ones (tuple size). Shape inference via meta-shape: torch.ones"""
    ...

@uses_shape_dsl(randn_ir)
@overload
def empty(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create uninitialized tensor. Shape inference via meta-shape: torch.empty"""
    ...

@overload
def empty(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create uninitialized tensor (tuple size). Shape inference via meta-shape: torch.empty"""
    ...

@uses_shape_dsl(randn_ir)
def full(size: tuple[int, ...], fill_value: float) -> Tensor:
    """Create tensor filled with value. Shape inference via meta-shape: torch.full"""
    ...

# arange overloads - Int is compatible with int, so meta-shape handles both
@uses_shape_dsl(arange_ir)
@overload
def arange(end: int) -> Tensor:
    """Create 1D tensor with range [0, end). Shape inference via meta-shape: torch.arange"""
    ...

@overload
def arange(end: int, *, dtype: int | None = None, device: Any = None) -> Tensor:
    """Create 1D tensor with range [0, end). Shape inference via meta-shape: torch.arange"""
    ...

@overload
def arange(start: int, end: int, step: int = 1) -> Tensor:
    """Create 1D tensor with range [start, end) with step. Shape inference via meta-shape: torch.arange"""
    ...

@overload
def arange(
    start: int,
    end: int,
    step: int = 1,
    *,
    dtype: int | None = None,
    device: Any = None,
) -> Tensor:
    """Create 1D tensor with range [start, end) with step. Shape inference via meta-shape: torch.arange"""
    ...

@uses_shape_dsl(linspace_ir)
def linspace(
    start: float, end: float, steps: int, *, dtype: Any = None, device: Any = None
) -> Tensor:
    """Create 1D tensor with linearly spaced values. Shape inference via meta-shape: torch.linspace"""
    ...

@uses_shape_dsl(eye_ir)
def eye(n: int) -> Tensor:
    """Create 2D identity matrix. Shape inference via meta-shape: torch.eye"""
    ...

# ==== Shape Manipulation Functions ====

@uses_shape_dsl(broadcast_to_ir)
def broadcast_to(self: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Broadcast tensor to shape. Shape inference via meta-shape: torch.broadcast_to"""
    ...

@uses_shape_dsl(tile_ir)
def tile(self: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Tile tensor by repeating. Shape inference via meta-shape: torch.tile"""
    ...

@uses_shape_dsl(select_ir)
def select(self: Tensor, dim: int, index: int) -> Tensor:
    """Select along dimension. Shape inference via meta-shape: torch.select"""
    ...

@uses_shape_dsl(narrow_ir)
def narrow(self: Tensor, dim: int, start: int, length: int) -> Tensor:
    """Narrow tensor along dimension. Shape inference via meta-shape: torch.narrow"""
    ...

@uses_shape_dsl(split_ir)
def split(
    self: Tensor, split_size_or_sections: int, dim: int = 0
) -> tuple[Tensor, ...]:
    """Split tensor into chunks. Shape inference via meta-shape: torch.split"""
    ...

@uses_shape_dsl(chunk_ir)
def chunk(self: Tensor, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
    """Split tensor into chunks. Shape inference via meta-shape: torch.chunk"""
    ...

@uses_shape_dsl(index_select_ir)
def index_select(self: Tensor, dim: int, index: Tensor) -> Tensor:
    """Select elements along dimension. Shape inference via meta-shape: torch.index_select"""
    ...

def gather[IndexShape: IntTuple](
    input: Tensor, dim: int, index: Tensor[IndexShape]
) -> Tensor[IndexShape]:
    """Gather elements along dimension. Output shape matches index shape."""
    ...

def scatter[Shape: IntTuple](
    input: Tensor[Shape], dim: int, index: Tensor, src: Tensor
) -> Tensor[Shape]:
    """Scatter elements along dimension. Shape-preserving operation."""
    ...

def masked_select(self: Tensor, mask: Tensor) -> Tensor[[Any]]:
    """Select elements with mask. Returns 1D tensor with data-dependent size."""
    ...

# ==== Phase 1.1: Missing Shape Operations ====

@uses_shape_dsl(unbind_ir)
def unbind(self: Tensor, dim: int = 0) -> tuple[Tensor, ...]:
    """Remove dimension by slicing along it. Shape inference via meta-shape: torch.unbind"""
    ...

@uses_shape_dsl(movedim_ir)
@overload
def movedim(self: Tensor, source: int, destination: int) -> Tensor:
    """Move single dimension to new position. Shape inference via meta-shape: torch.movedim"""
    ...

@overload
def movedim(
    self: Tensor, source: tuple[int, ...], destination: tuple[int, ...]
) -> Tensor:
    """Move multiple dimensions to new positions. Shape inference via meta-shape: torch.movedim"""
    ...

@uses_shape_dsl(movedim_ir)
@overload
def moveaxis(self: Tensor, source: int, destination: int) -> Tensor:
    """Alias for movedim. Shape inference via meta-shape: torch.moveaxis"""
    ...

@overload
def moveaxis(
    self: Tensor, source: tuple[int, ...], destination: tuple[int, ...]
) -> Tensor:
    """Alias for movedim. Shape inference via meta-shape: torch.moveaxis"""
    ...

@uses_shape_dsl(unfold_ir)
def unfold(self: Tensor, dimension: int, size: int, step: int) -> Tensor:
    """Returns sliding window view. Shape inference via meta-shape: torch.unfold"""
    ...

# ==== Additional Reduction Functions ====

@uses_shape_dsl(reduce_ir)
def all(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Check if all elements are True. Shape inference via meta-shape: torch.all"""
    ...

@uses_shape_dsl(reduce_ir)
def any(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Check if any element is True. Shape inference via meta-shape: torch.any"""
    ...

# ==== Phase 1.2: Missing Reduction Operations ====

@uses_shape_dsl(min_max_median_ir)
@overload
def median(self: Tensor) -> Tensor:
    """Median of all elements (scalar). Shape inference via meta-shape: torch.median"""
    ...

@overload
def median(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Median along dimension. Returns (values, indices). Shape inference via meta-shape: torch.median"""
    ...

@uses_shape_dsl(reduce_ir)
def logsumexp(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Log-sum-exp along dimension(s). Shape inference via meta-shape: torch.logsumexp"""
    ...

@uses_shape_dsl(reduce_ir)
def count_nonzero(self: Tensor, dim: int | None = None) -> Tensor:
    """Count non-zero elements. Shape inference via meta-shape: torch.count_nonzero"""
    ...

@uses_shape_dsl(aminmax_ir)
def aminmax(
    self: Tensor, dim: int | None = None, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """Min and max along dimension(s). Shape inference via meta-shape: torch.aminmax"""
    ...

@uses_shape_dsl(reduce_ir)
def norm(
    self: Tensor,
    p: int | float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
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
) -> tuple[Tensor[Shape], Tensor[Shape]]:
    """Cumulative maximum along dimension. Returns (values, indices). Shape-preserving operation."""
    ...

def cummin[Shape: IntTuple](
    input: Tensor[Shape], dim: int
) -> tuple[Tensor[Shape], Tensor[Shape]]:
    """Cumulative minimum along dimension. Returns (values, indices). Shape-preserving operation."""
    ...

# Tier 2: Additional reduction operations (always return tuples)
@uses_shape_dsl(tuple_reduce_ir)
def mode(self: Tensor, dim: int = -1, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Mode along dimension. Returns (values, indices). Shape inference via meta-shape: torch.mode"""
    ...

@uses_shape_dsl(topk_ir)
def topk(
    self: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[Tensor, Tensor]:
    """Top k elements. Returns (values, indices). Shape inference via meta-shape: torch.topk"""
    ...

def sort[Shape: IntTuple](
    input: Tensor[Shape], dim: int = -1, descending: bool = False, stable: bool = False
) -> tuple[Tensor[Shape], Tensor[Shape]]:
    """Sort tensor. Returns (values, indices). Shape-preserving operation."""
    ...

@uses_shape_dsl(tuple_reduce_ir)
def kthvalue(
    self: Tensor, k: int, dim: int = -1, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """Kth smallest value. Returns (values, indices). Shape inference via meta-shape: torch.kthvalue"""
    ...

# Tier 3: Statistical operations returning tuples
@uses_shape_dsl(aminmax_ir)
def var_mean(
    self: Tensor,
    dim: int | tuple[int, ...] | None = None,
    unbiased: bool = True,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Variance and mean. Returns (var, mean). Shape inference via meta-shape: torch.var_mean"""
    ...

@uses_shape_dsl(aminmax_ir)
def std_mean(
    self: Tensor,
    dim: int | tuple[int, ...] | None = None,
    unbiased: bool = True,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Standard deviation and mean. Returns (std, mean). Shape inference via meta-shape: torch.std_mean"""
    ...

# ==== Phase 1.3: Tensor Creation Operations ====

def zeros_like[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
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

@uses_shape_dsl(diag_embed_ir)
def diag_embed(self: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    """Create diagonal tensor. Shape inference via meta-shape: torch.diag_embed"""
    ...

def tril[Shape: IntTuple](input: Tensor[Shape], diagonal: int = 0) -> Tensor[Shape]:
    """Lower triangular part. Shape inference via generic fixture signature."""
    ...

def triu[Shape: IntTuple](input: Tensor[Shape], diagonal: int = 0) -> Tensor[Shape]:
    """Upper triangular part. Shape inference via generic fixture signature."""
    ...

@uses_shape_dsl(tri_indices_ir)
def tril_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """Indices of lower triangular part. Shape inference via meta-shape: torch.tril_indices"""
    ...

@uses_shape_dsl(tri_indices_ir)
def triu_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """Indices of upper triangular part. Shape inference via meta-shape: torch.triu_indices"""
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

@uses_shape_dsl(mv_ir)
def mv(self: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector multiplication (2D @ 1D). Shape inference via meta-shape: torch.mv"""
    ...

def dot(input: Tensor, other: Tensor) -> Tensor[[]]:
    """Dot product (1D @ 1D → scalar). Returns scalar tensor."""
    ...

# ==== Phase 2: Arithmetic & Basic Operations ====
# All operations preserve shape (use IdentityMetaShape)

# Arithmetic operations (element-wise)
def add[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise addition. Shape inference via generic fixture signature."""
    ...

def sub[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise subtraction. Shape inference via generic fixture signature."""
    ...

def mul[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise multiplication. Shape inference via generic fixture signature."""
    ...

def div[Shape: IntTuple](
    input: Tensor[Shape],
    other: Tensor | int | float,
    *,
    rounding_mode: str | None = None,
) -> Tensor[Shape]:
    """Element-wise division. Shape inference via generic fixture signature."""
    ...

def pow[Shape: IntTuple](
    input: Tensor[Shape], exponent: float | Tensor
) -> Tensor[Shape]:
    """Element-wise power. Shape inference via generic fixture signature."""
    ...

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
def eq[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise equality. Shape inference via generic fixture signature."""
    ...

def ne[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise inequality. Shape inference via generic fixture signature."""
    ...

def lt[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise less than. Shape inference via generic fixture signature."""
    ...

def le[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise less than or equal. Shape inference via generic fixture signature."""
    ...

def gt[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise greater than. Shape inference via generic fixture signature."""
    ...

def ge[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise greater than or equal. Shape inference via generic fixture signature."""
    ...

# Logical operations
def logical_and[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise logical AND. Shape inference via generic fixture signature."""
    ...

def logical_or[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise logical OR. Shape inference via generic fixture signature."""
    ...

def logical_not[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Element-wise logical NOT. Shape inference via generic fixture signature."""
    ...

# Clamping
def clamp[Shape: IntTuple](
    input: Tensor[Shape], min: float | None = None, max: float | None = None
) -> Tensor[Shape]:
    """Clamp tensor values. Shape inference via generic fixture signature."""
    ...

def clip[Shape: IntTuple](
    input: Tensor[Shape], min: float | None = None, max: float | None = None
) -> Tensor[Shape]:
    """Alias for clamp. Shape inference via generic fixture signature."""
    ...

# Activation functions (relu is most common, others in torch.nn.functional)
def relu[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """ReLU activation. Shape inference via generic fixture signature."""
    ...

# Additional mathematical operations
def atan2[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise arctangent of input/other. Shape inference via generic fixture signature."""
    ...

def hypot[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise hypotenuse. Shape inference via generic fixture signature."""
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
def bitwise_and[Shape: IntTuple](
    input: Tensor[Shape], other: Tensor | int | bool
) -> Tensor[Shape]:
    """Bitwise AND. Shape inference via generic fixture signature."""
    ...

def equal(input: Tensor, other: Tensor) -> builtins.bool:
    """Return whether two tensors have the same size and elements."""
    ...

def bitwise_or[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Bitwise OR. Shape inference via generic fixture signature."""
    ...

def bitwise_xor[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Bitwise XOR. Shape inference via generic fixture signature."""
    ...

def bitwise_not[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Bitwise NOT. Shape inference via generic fixture signature."""
    ...

def bitwise_left_shift[Shape: IntTuple](
    input: Tensor[Shape], other: Tensor
) -> Tensor[Shape]:
    """Bitwise left shift. Shape inference via generic fixture signature."""
    ...

def bitwise_right_shift[Shape: IntTuple](
    input: Tensor[Shape], other: Tensor
) -> Tensor[Shape]:
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

def maximum[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise maximum. Shape inference via generic fixture signature."""
    ...

def minimum[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise minimum. Shape inference via generic fixture signature."""
    ...

def fmax[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise maximum (NaN handling). Shape inference via generic fixture signature."""
    ...

def fmin[Shape: IntTuple](input: Tensor[Shape], other: Tensor) -> Tensor[Shape]:
    """Element-wise minimum (NaN handling). Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 4: Advanced Linear Algebra Operations
# ==============================================================================

# Advanced matmul operations
@uses_shape_dsl(tensordot_ir)
def tensordot(
    self: Tensor, other: Tensor, dims: int | tuple[list[int], list[int]] = 2
) -> Tensor:
    """Tensor contraction over specified dimensions. Shape inference via meta-shape: torch.tensordot"""
    ...

@uses_shape_dsl(einsum_ir)
def einsum(spec: str, *operands: Tensor) -> Tensor:
    """Einstein summation convention. Shape inference via meta-shape: torch.einsum"""
    ...

# Eigenvalue decomposition
@uses_shape_dsl(eig_ir)
def eig(self: Tensor, eigenvectors: bool = False) -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition. Shape inference via meta-shape: torch.eig"""
    ...

@uses_shape_dsl(eig_ir)
def eigh(self: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Hermitian eigenvalue decomposition. Shape inference via meta-shape: torch.eigh"""
    ...

# Cholesky decomposition
def cholesky[Shape: IntTuple](
    input: Tensor[Shape], upper: bool = False
) -> Tensor[Shape]:
    """Cholesky decomposition. Shape inference via generic fixture signature."""
    ...

# Linear system solvers
@uses_shape_dsl(solve_ir)
def solve(self: Tensor, other: Tensor) -> Tensor:
    """Solve linear system. Shape inference via meta-shape: torch.solve"""
    ...

@uses_shape_dsl(solve_reversed_ir)
def triangular_solve(self: Tensor, other: Tensor, upper: bool = True) -> Tensor:
    """Solve triangular system. Shape inference via meta-shape: torch.triangular_solve"""
    ...

@uses_shape_dsl(solve_reversed_ir)
def cholesky_solve(self: Tensor, other: Tensor, upper: bool = False) -> Tensor:
    """Solve using Cholesky. Shape inference via meta-shape: torch.cholesky_solve"""
    ...

@uses_shape_dsl(solve_ir)
def lu_solve(self: Tensor, other: Tensor, LU_pivots: Tensor) -> Tensor:
    """Solve using LU decomposition. Shape inference via meta-shape: torch.lu_solve"""
    ...

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

@uses_shape_dsl(slogdet_ir)
def slogdet(self: Tensor) -> tuple[Tensor, Tensor]:
    """Sign and log determinant. Shape inference via meta-shape: torch.slogdet"""
    ...

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
@uses_shape_dsl(where_ir)
def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Conditional element-wise selection. Shape inference via meta-shape: torch.where"""
    ...

def masked_fill[Shape: IntTuple](
    input: Tensor[Shape], mask: Tensor, value: float
) -> Tensor[Shape]:
    """Fill masked elements. Shape inference via generic fixture signature."""
    ...

def masked_scatter[Shape: IntTuple](
    input: Tensor[Shape], mask: Tensor, source: Tensor
) -> Tensor[Shape]:
    """Scatter into masked positions. Shape inference via generic fixture signature."""
    ...

# Advanced indexing operations
def index_add[Shape: IntTuple](
    input: Tensor[Shape], dim: int, index: Tensor, source: Tensor, alpha: float = 1
) -> Tensor[Shape]:
    """Add values at indices. Shape inference via generic fixture signature."""
    ...

def index_copy[Shape: IntTuple](
    input: Tensor[Shape], dim: int, index: Tensor, source: Tensor
) -> Tensor[Shape]:
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

def index_fill[Shape: IntTuple](
    input: Tensor[Shape], dim: int, index: Tensor, value: float
) -> Tensor[Shape]:
    """Fill indices with value. Shape inference via generic fixture signature."""
    ...

# Take/put operations
def take[IndexShape: IntTuple](
    input: Tensor, index: Tensor[IndexShape]
) -> Tensor[IndexShape]:
    """Take elements at indices. Output shape matches index shape."""
    ...

@uses_shape_dsl(take_along_dim_ir)
def take_along_dim(self: Tensor, indices: Tensor, dim: int) -> Tensor:
    """Take along dimension. Shape inference via meta-shape: torch.take_along_dim"""
    ...

def put[Shape: IntTuple](
    input: Tensor[Shape], index: Tensor, source: Tensor, accumulate: bool = False
) -> Tensor[Shape]:
    """Put values at indices. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 6: Specialized Operations
# ==============================================================================

# Random sampling operations
def bernoulli[Shape: IntTuple](input: Tensor[Shape], p: float = 0.5) -> Tensor[Shape]:
    """Sample from Bernoulli distribution. Shape inference via generic fixture signature."""
    ...

@uses_shape_dsl(multinomial_ir)
def multinomial(self: Tensor, num_samples: int, replacement: bool = False) -> Tensor:
    """Sample from multinomial distribution. Shape inference via meta-shape: torch.multinomial"""
    ...

@uses_shape_dsl(normal_ir)
@overload
def normal(mean: Tensor, std: Tensor) -> Tensor:
    """Sample from normal distribution (tensor mean, tensor std). Shape inference via meta-shape: torch.normal"""
    ...

@overload
def normal(mean: Tensor, std: float) -> Tensor:
    """Sample from normal distribution (tensor mean, scalar std). Shape inference via meta-shape: torch.normal"""
    ...

@overload
def normal(mean: float, std: Tensor) -> Tensor:
    """Sample from normal distribution (scalar mean, tensor std). Shape inference via meta-shape: torch.normal"""
    ...

@overload
def normal(mean: float, std: float, size: tuple[int, ...]) -> Tensor:
    """Sample from normal distribution (scalar mean/std, explicit size). Shape inference via meta-shape: torch.normal"""
    ...

def poisson[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Sample from Poisson distribution. Shape inference via generic fixture signature."""
    ...

# Tensor property functions
@uses_shape_dsl(numel_ir)
def numel[Dims: IntTuple](self: Tensor[Dims]) -> int:
    """Number of elements. Shape inference via meta-shape: torch.numel"""
    ...

# ==============================================================================
# Data Types and Context Managers
# ==============================================================================

# Data type constants
long: Any = ...  # torch.long dtype constant
float32: Any = ...  # torch.float32 dtype constant
float64: Any = ...  # torch.float64 dtype constant
bfloat16: Any = ...  # torch.bfloat16 dtype constant
int32: Any = ...  # torch.int32 dtype constant
int64: Any = ...  # torch.int64 dtype constant

# dtype type (for type annotations)
class dtype:
    """PyTorch data type."""

    ...

# ==============================================================================
# Tensor Creation with dtype support
# ==============================================================================

def tensor(
    data: Any,
    dtype: Any = None,
    device: Any = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create tensor from data. Returns shapeless tensor (shape depends on input data)."""
    ...

@uses_shape_dsl(randint_ir)
def randint(
    low: int,
    high: int,
    size: tuple[int, ...],
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create tensor of random integers. Returns shapeless tensor (shape depends on size arg)."""
    ...

# ==============================================================================
# Additional Math Operations
# ==============================================================================

def rsqrt[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Reciprocal square root (1/sqrt(x)). Shape-preserving element-wise operation."""
    ...

@uses_shape_dsl(outer_ir)
def outer(self: Tensor, vec2: Tensor) -> Tensor:
    """Outer product of two 1D tensors. Shape inference via meta-shape: torch.outer"""
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

        # As decorator:
        @torch.no_grad()
        def inference(x):
            return model(x)
    """
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def __call__(self, func) -> Any: ...  # For decorator usage

def meshgrid(*tensors: Tensor, indexing: str = "ij") -> tuple[Tensor, ...]:
    """Create coordinate grids from 1D input tensors.

    For N input tensors, returns N tensors each with N dimensions.
    Shape inference depends on input tensor shapes; returns shapeless tuple.
    """
    ...
