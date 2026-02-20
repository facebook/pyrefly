# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive type stubs for PyTorch with shape inference.

Shape inference is handled by meta-shape functions registered in the type checker.
This stub file only defines the basic tensor class and operations that don't use
meta-shapes (like broadcasting arithmetic and matmul).

For operations handled by meta-shapes, see pyrefly_types/src/meta_shape.rs:
- torch.reshape, torch.cat, torch.broadcast_to
- torch.squeeze, torch.unsqueeze, torch.transpose, torch.permute
- torch.sum, torch.mean, torch.prod, torch.min, torch.max, torch.all, torch.any
"""

import builtins
from typing import Any, overload, Self

__all__ = ["Tensor"]

# ============================================================================
# Tensor Class
# ============================================================================

class Tensor[*Shape]:
    """
    PyTorch Tensor with shape type parameter.

    The shape is tracked at the type level, allowing static verification
    of tensor operations.

    Most shape transformations are handled by meta-shape functions registered
    in the type checker, not by explicit type signatures here.
    """

    # ==== Tensor Properties ====
    shape: tuple[int, ...]  # Tensor shape as a tuple
    requires_grad: bool  # Whether gradient tracking is enabled
    device: Any  # Device where tensor is stored (cpu, cuda, etc.)
    dtype: Any  # Data type of tensor elements (float32, int64, etc.)
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

    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        """Matrix multiplication (@). Shape inference via meta-shape: torch.Tensor.matmul"""
        ...

    # ==== Arithmetic Operations ====
    # These use broadcasting, which is handled separately from meta-shapes

    def __add__(self, other: Tensor | float | int) -> Self: ...
    def __sub__(self, other: Tensor | float | int) -> Self: ...
    def __mul__(self, other: Tensor | float | int) -> Self: ...
    def __truediv__(self, other: Tensor | float | int) -> Self: ...

    # Reverse operations for scalars
    def __radd__(self, other: float | int) -> Self: ...
    def __rsub__(self, other: float | int) -> Self: ...
    def __rmul__(self, other: float | int) -> Self: ...
    def __rtruediv__(self, other: float | int) -> Self: ...
    def __rpow__(self, other: float | int) -> Self: ...

    # Power operations
    def __pow__(self, other: Tensor | float | int) -> Self: ...

    # ==== Comparison Operations ====
    # Return boolean tensors with the same shape

    def __eq__(self, other: Tensor | float | int) -> Self: ...  # type: ignore[override]
    def __ne__(self, other: Tensor | float | int) -> Self: ...  # type: ignore[override]
    def __lt__(self, other: Tensor | float | int) -> Self: ...
    def __le__(self, other: Tensor | float | int) -> Self: ...
    def __gt__(self, other: Tensor | float | int) -> Self: ...
    def __ge__(self, other: Tensor | float | int) -> Self: ...

    # ==== Shape Manipulation Operations ====
    # Handled by meta-shape functions - simplified signatures

    @overload
    def reshape(self: Tensor, *shape: int) -> Tensor:
        """Reshape tensor. Shape inference via meta-shape: torch.Tensor.reshape"""
        ...

    @overload
    def reshape(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Reshape tensor. Shape inference via meta-shape: torch.Tensor.reshape"""
        ...

    @overload
    def view(self: Tensor, *shape: int) -> Tensor:
        """View (alias for reshape). Shape inference via meta-shape: torch.Tensor.view"""
        ...

    @overload
    def view(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        """View (alias for reshape). Shape inference via meta-shape: torch.Tensor.view"""
        ...

    def flatten(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        """Flatten dimensions. Shape inference via meta-shape: torch.flatten"""
        ...

    def transpose(self: Tensor, dim0: int, dim1: int) -> Tensor:
        """Transpose two dimensions. Shape inference via meta-shape: torch.transpose"""
        ...

    @overload
    def permute(self: Tensor, *dims: int) -> Tensor:
        """Permute dimensions. Shape inference via meta-shape: torch.Tensor.permute"""
        ...

    @overload
    def permute(self: Tensor, dims: tuple[int, ...]) -> Tensor:
        """Permute dimensions. Shape inference via meta-shape: torch.Tensor.permute"""
        ...

    def squeeze(self: Tensor, dim: int | None = None) -> Tensor:
        """Remove dimensions of size 1. Shape inference via meta-shape: torch.squeeze"""
        ...

    def unsqueeze(self: Tensor, dim: int) -> Tensor:
        """Add dimension of size 1. Shape inference via meta-shape: torch.unsqueeze"""
        ...

    @overload
    def repeat(self: Tensor, *sizes: int) -> Tensor:
        """Repeat tensor. Shape inference via meta-shape: torch.Tensor.repeat"""
        ...

    @overload
    def repeat(self: Tensor, sizes: tuple[int, ...]) -> Tensor:
        """Repeat tensor. Shape inference via meta-shape: torch.Tensor.repeat"""
        ...

    def t[M, N](self: Tensor[M, N]) -> Tensor[N, M]:
        """Transpose 2D tensor. Swaps dimensions."""
        ...

    def expand(self: Tensor, *sizes: int) -> Tensor:
        """Expand tensor. Shape inference via meta-shape: torch.Tensor.expand"""
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

    def item(self: Tensor) -> float | int:
        """Returns Python scalar from 0-dimensional tensor. Shape inference via meta-shape: torch.Tensor.item"""
        ...

    def tolist(self: Tensor) -> Any:
        """Returns tensor as nested Python list. Shape inference via meta-shape: torch.Tensor.tolist"""
        ...

    def tile(self: Tensor, dims: tuple[int, ...]) -> Tensor:
        """Tile tensor. Shape inference via meta-shape: torch.Tensor.tile"""
        ...

    def select(self: Tensor, dim: int, index: int) -> Tensor:
        """Select along dimension. Shape inference via meta-shape: torch.Tensor.select"""
        ...

    def narrow(self: Tensor, dim: int, start: int, length: int) -> Tensor:
        """Narrow tensor along dimension. Shape inference via meta-shape: torch.Tensor.narrow"""
        ...

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

    def chunk(self: Tensor, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
        """Split tensor into chunks. Shape inference via meta-shape: torch.Tensor.chunk"""
        ...

    def index_select(self: Tensor, dim: int, index: Tensor) -> Tensor:
        """Select elements along dimension. Shape inference via meta-shape: torch.Tensor.index_select"""
        ...

    def gather[*IndexShape](
        self: Tensor, dim: int, index: Tensor[*IndexShape]
    ) -> Tensor[*IndexShape]:
        """Gather elements along dimension. Output shape matches index shape."""
        ...

    def scatter[*Shape](
        self: Tensor[*Shape], dim: int, index: Tensor, src: Tensor
    ) -> Tensor[*Shape]:
        """Scatter elements along dimension. Shape-preserving operation."""
        ...

    def masked_select(self: Tensor, mask: Tensor) -> Tensor:
        """Select elements with mask. Shape inference via meta-shape: torch.Tensor.masked_select"""
        ...

    # ==== Phase 1.1: Missing Shape Operations (Methods) ====

    def unbind(self: Tensor, dim: int = 0) -> tuple[Tensor, ...]:
        """Remove dimension by slicing along it. Shape inference via meta-shape: torch.Tensor.unbind"""
        ...

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

    def unfold(self: Tensor, dimension: int, size: int, step: int) -> Tensor:
        """Returns sliding window view. Shape inference via meta-shape: torch.Tensor.unfold"""
        ...

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

    @overload
    def sum(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Sum along dimension(s). Shape inference via meta-shape: torch.Tensor.sum"""
        ...

    @overload
    def sum(self: Tensor, dim: tuple[int, ...], keepdim: bool = False) -> Tensor:
        """Sum along multiple dimensions. Shape inference via meta-shape: torch.Tensor.sum"""
        ...

    def mean(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Mean along dimension(s). Shape inference via meta-shape: torch.mean"""
        ...

    @overload
    def max(self: Tensor) -> Tensor:
        """Max of all elements (scalar). Shape inference via meta-shape: torch.Tensor.max"""
        ...

    @overload
    def max(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
        """Max along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.max"""
        ...

    @overload
    def min(self: Tensor) -> Tensor:
        """Min of all elements (scalar). Shape inference via meta-shape: torch.Tensor.min"""
        ...

    @overload
    def min(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
        """Min along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.min"""
        ...

    def prod(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Product along dimension(s). Shape inference via meta-shape: torch.prod"""
        ...

    def std(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Standard deviation along dimension(s). Shape inference via meta-shape: torch.std"""
        ...

    def var(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Variance along dimension(s). Shape inference via meta-shape: torch.var"""
        ...

    def argmax(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Argmax along dimension(s). Shape inference via meta-shape: torch.argmax"""
        ...

    def argmin(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Argmin along dimension(s). Shape inference via meta-shape: torch.argmin"""
        ...

    # ==== Phase 1.2: Missing Reduction Operations (Methods) ====

    @overload
    def median(self: Tensor) -> Tensor:
        """Median of all elements (scalar). Shape inference via meta-shape: torch.Tensor.median"""
        ...

    @overload
    def median(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
        """Median along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.median"""
        ...

    def logsumexp(
        self: Tensor, dim: int | None = None, keepdim: bool = False
    ) -> Tensor:
        """Log-sum-exp along dimension(s). Shape inference via meta-shape: torch.Tensor.logsumexp"""
        ...

    def count_nonzero(self: Tensor, dim: int | None = None) -> Tensor:
        """Count non-zero elements. Shape inference via meta-shape: torch.Tensor.count_nonzero"""
        ...

    def aminmax(
        self: Tensor, dim: int | None = None, keepdim: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Min and max along dimension(s). Shape inference via meta-shape: torch.Tensor.aminmax"""
        ...

    def norm(
        self: Tensor,
        p: int | float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        """Compute norm. Shape inference via meta-shape: torch.Tensor.norm"""
        ...

    def dist(self: Tensor, other: Tensor, p: int | float = 2) -> Tensor[()]:
        """Compute distance to another tensor. Returns scalar tensor."""
        ...

    def cumsum[*Shape](self: Tensor[*Shape], dim: int) -> Tensor[*Shape]:
        """Cumulative sum along dimension. Shape-preserving operation."""
        ...

    def cumprod[*Shape](self: Tensor[*Shape], dim: int) -> Tensor[*Shape]:
        """Cumulative product along dimension. Shape-preserving operation."""
        ...

    def cummax[*Shape](
        self: Tensor[*Shape], dim: int
    ) -> tuple[Tensor[*Shape], Tensor[*Shape]]:
        """Cumulative maximum along dimension. Returns (values, indices). Shape-preserving operation."""
        ...

    def cummin[*Shape](
        self: Tensor[*Shape], dim: int
    ) -> tuple[Tensor[*Shape], Tensor[*Shape]]:
        """Cumulative minimum along dimension. Returns (values, indices). Shape-preserving operation."""
        ...

    # ==== Tier 2: Additional Reduction Methods ====

    def mode(
        self: Tensor, dim: int = -1, keepdim: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Mode along dimension. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.mode"""
        ...

    def topk(
        self: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Top k elements. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.topk"""
        ...

    def sort[*Shape](
        self: Tensor[*Shape],
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> tuple[Tensor[*Shape], Tensor[*Shape]]:
        """Sort tensor. Returns (values, indices). Shape-preserving operation."""
        ...

    def kthvalue(
        self: Tensor, k: int, dim: int = -1, keepdim: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Kth smallest value. Returns (values, indices). Shape inference via meta-shape: torch.Tensor.kthvalue"""
        ...

    # ==== Phase 1.3: Tensor Creation Operations (Methods) ====

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

    def matmul(self: Tensor, other: Tensor) -> Tensor:
        """Matrix multiplication. Shape inference via meta-shape: torch.Tensor.matmul"""
        ...

    def mm[N, K, M](self: Tensor[N, K], mat2: Tensor[K, M]) -> Tensor[N, M]:
        """Matrix multiplication (2D @ 2D). Output: [N, M]."""
        ...

    def bmm[B, N, K, M](
        self: Tensor[B, N, K], mat2: Tensor[B, K, M]
    ) -> Tensor[B, N, M]:
        """Batch matrix multiplication (3D @ 3D). Output: [B, N, M]."""
        ...

    def mv(self: Tensor, vec: Tensor) -> Tensor:
        """Matrix-vector multiplication. Shape inference via meta-shape: torch.Tensor.mv"""
        ...

    def dot(self: Tensor, other: Tensor) -> Tensor[()]:
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

    def div(self, other: Tensor) -> Self:
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

    def remainder(self, other: Tensor) -> Self:
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

    def det[*Batch, M, N](self: Tensor[*Batch, M, N]) -> Tensor[*Batch]:
        """Determinant. Returns batch dimensions only (drops last 2 dims)."""
        ...

    def logdet[*Batch, M, N](self: Tensor[*Batch, M, N]) -> Tensor[*Batch]:
        """Log determinant. Returns batch dimensions only (drops last 2 dims)."""
        ...

    def slogdet(self: Tensor) -> tuple[Tensor, Tensor]:
        """Sign and log determinant. Shape inference via meta-shape: torch.Tensor.slogdet"""
        ...

    def matrix_power(self, n: int) -> Self:
        """Matrix power. Shape inference via generic fixture signature."""
        ...

    def trace[*Batch, M, N](self: Tensor[*Batch, M, N]) -> Tensor[*Batch]:
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

    def take[*IndexShape](
        self: Tensor, index: Tensor[*IndexShape]
    ) -> Tensor[*IndexShape]:
        """Take elements at indices. Output shape matches index shape."""
        ...

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

    def multinomial(
        self: Tensor, num_samples: int, replacement: bool = False
    ) -> Tensor:
        """Sample from multinomial distribution. Shape inference via meta-shape: torch.Tensor.multinomial"""
        ...

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> Self:
        """Fill with normal distribution in-place. Shape inference via generic fixture signature."""
        ...

    def random_(self, low: int = 0, high: int = None) -> Self:
        """Fill with random integers in-place. Shape inference via generic fixture signature."""
        ...

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> Self:
        """Fill with uniform distribution in-place. Shape inference via generic fixture signature."""
        ...

    def numel(self: Tensor) -> int:
        """Number of elements. Shape inference via meta-shape: torch.Tensor.numel"""
        ...

    def dim(self: Tensor) -> int:
        """Number of dimensions. Shape inference via meta-shape: torch.Tensor.dim"""
        ...

    def nelement(self: Tensor) -> int:
        """Number of elements. Shape inference via meta-shape: torch.Tensor.nelement"""
        ...

# ============================================================================
# Module-level Functions
# ============================================================================

def matmul(self: Tensor, other: Tensor) -> Tensor:
    """Matrix multiplication function. Shape inference via meta-shape: torch.matmul"""
    ...

def cat(tensors: list[Tensor] | tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Concatenate tensors. Shape inference via meta-shape: torch.cat"""
    ...

def stack(tensors: list[Tensor] | tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Stack tensors (adds new dimension)."""
    ...

def transpose(self: Tensor, dim0: int, dim1: int) -> Tensor:
    """Transpose two dimensions. Shape inference via meta-shape: torch.transpose"""
    ...

def reshape(self: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape tensor. Shape inference via meta-shape: torch.reshape"""
    ...

def squeeze(self: Tensor, dim: int | None = None) -> Tensor:
    """Remove dimensions of size 1. Shape inference via meta-shape: torch.squeeze"""
    ...

def unsqueeze(self: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1. Shape inference via meta-shape: torch.unsqueeze"""
    ...

def permute(self: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Permute dimensions. Shape inference via meta-shape: torch.permute"""
    ...

def sum(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Sum along dimension(s). Shape inference via meta-shape: torch.sum"""
    ...

def mean(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Mean along dimension(s). Shape inference via meta-shape: torch.mean"""
    ...

@overload
def max(self: Tensor) -> Tensor:
    """Max of all elements (scalar). Shape inference via meta-shape: torch.max"""
    ...

@overload
def max(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Max along dimension. Returns (values, indices). Shape inference via meta-shape: torch.max"""
    ...

@overload
def min(self: Tensor) -> Tensor:
    """Min of all elements (scalar). Shape inference via meta-shape: torch.min"""
    ...

@overload
def min(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Min along dimension. Returns (values, indices). Shape inference via meta-shape: torch.min"""
    ...

def prod(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Product along dimension(s). Shape inference via meta-shape: torch.prod"""
    ...

def std(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Standard deviation. Shape inference via meta-shape: torch.std"""
    ...

def var(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Variance. Shape inference via meta-shape: torch.var"""
    ...

def argmax(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Argmax. Shape inference via meta-shape: torch.argmax"""
    ...

def argmin(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Argmin. Shape inference via meta-shape: torch.argmin"""
    ...

def flatten(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten dimensions. Shape inference via meta-shape: torch.flatten"""
    ...

def stack(tensors: list[Tensor] | tuple[Tensor, ...], dim: int = 0) -> Tensor:
    """Stack tensors. Shape inference via meta-shape: torch.stack"""
    ...

# ==== Tensor Creation Functions ====

@overload
def randn(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values. Shape inference via meta-shape: torch.randn"""
    ...

@overload
def randn(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values (tuple size). Shape inference via meta-shape: torch.randn"""
    ...

@overload
def rand(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values [0, 1). Shape inference via meta-shape: torch.rand"""
    ...

@overload
def rand(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor with random values (tuple size). Shape inference via meta-shape: torch.rand"""
    ...

@overload
def zeros(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with zeros. Shape inference via meta-shape: torch.zeros"""
    ...

@overload
def zeros(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with zeros (tuple size). Shape inference via meta-shape: torch.zeros"""
    ...

@overload
def ones(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with ones. Shape inference via meta-shape: torch.ones"""
    ...

@overload
def ones(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create tensor filled with ones (tuple size). Shape inference via meta-shape: torch.ones"""
    ...

@overload
def empty(*size: int, dtype: Any = None, device: Any = None) -> Tensor:
    """Create uninitialized tensor. Shape inference via meta-shape: torch.empty"""
    ...

@overload
def empty(size: tuple[int, ...], dtype: Any = None, device: Any = None) -> Tensor:
    """Create uninitialized tensor (tuple size). Shape inference via meta-shape: torch.empty"""
    ...

def full(size: tuple[int, ...], fill_value: float) -> Tensor:
    """Create tensor filled with value. Shape inference via meta-shape: torch.full"""
    ...

# arange overloads - Dim is compatible with int, so meta-shape handles both
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

def linspace(start: float, end: float, steps: int) -> Tensor:
    """Create 1D tensor with linearly spaced values. Shape inference via meta-shape: torch.linspace"""
    ...

def eye(n: int) -> Tensor:
    """Create 2D identity matrix. Shape inference via meta-shape: torch.eye"""
    ...

# ==== Shape Manipulation Functions ====

def broadcast_to(self: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Broadcast tensor to shape. Shape inference via meta-shape: torch.broadcast_to"""
    ...

def tile(self: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Tile tensor by repeating. Shape inference via meta-shape: torch.tile"""
    ...

def select(self: Tensor, dim: int, index: int) -> Tensor:
    """Select along dimension. Shape inference via meta-shape: torch.select"""
    ...

def narrow(self: Tensor, dim: int, start: int, length: int) -> Tensor:
    """Narrow tensor along dimension. Shape inference via meta-shape: torch.narrow"""
    ...

def split(
    self: Tensor, split_size_or_sections: int, dim: int = 0
) -> tuple[Tensor, ...]:
    """Split tensor into chunks. Shape inference via meta-shape: torch.split"""
    ...

def chunk(self: Tensor, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
    """Split tensor into chunks. Shape inference via meta-shape: torch.chunk"""
    ...

def index_select(self: Tensor, dim: int, index: Tensor) -> Tensor:
    """Select elements along dimension. Shape inference via meta-shape: torch.index_select"""
    ...

def gather[*IndexShape](
    input: Tensor, dim: int, index: Tensor[*IndexShape]
) -> Tensor[*IndexShape]:
    """Gather elements along dimension. Output shape matches index shape."""
    ...

def scatter[*Shape](
    input: Tensor[*Shape], dim: int, index: Tensor, src: Tensor
) -> Tensor[*Shape]:
    """Scatter elements along dimension. Shape-preserving operation."""
    ...

def masked_select(self: Tensor, mask: Tensor) -> Tensor:
    """Select elements with mask. Shape inference via meta-shape: torch.masked_select"""
    ...

# ==== Phase 1.1: Missing Shape Operations ====

def unbind(self: Tensor, dim: int = 0) -> tuple[Tensor, ...]:
    """Remove dimension by slicing along it. Shape inference via meta-shape: torch.unbind"""
    ...

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

def unfold(self: Tensor, dimension: int, size: int, step: int) -> Tensor:
    """Returns sliding window view. Shape inference via meta-shape: torch.unfold"""
    ...

# ==== Additional Reduction Functions ====

def all(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Check if all elements are True. Shape inference via meta-shape: torch.all"""
    ...

def any(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Check if any element is True. Shape inference via meta-shape: torch.any"""
    ...

# ==== Phase 1.2: Missing Reduction Operations ====

@overload
def median(self: Tensor) -> Tensor:
    """Median of all elements (scalar). Shape inference via meta-shape: torch.median"""
    ...

@overload
def median(self: Tensor, dim: int, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Median along dimension. Returns (values, indices). Shape inference via meta-shape: torch.median"""
    ...

def logsumexp(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Log-sum-exp along dimension(s). Shape inference via meta-shape: torch.logsumexp"""
    ...

def count_nonzero(self: Tensor, dim: int | None = None) -> Tensor:
    """Count non-zero elements. Shape inference via meta-shape: torch.count_nonzero"""
    ...

def aminmax(
    self: Tensor, dim: int | None = None, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """Min and max along dimension(s). Shape inference via meta-shape: torch.aminmax"""
    ...

def norm(
    self: Tensor,
    p: int | float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Compute norm. Shape inference via meta-shape: torch.norm"""
    ...

def dist(input: Tensor, other: Tensor, p: int | float = 2) -> Tensor[()]:
    """Compute distance between tensors. Returns scalar tensor."""
    ...

def cumsum[*Shape](input: Tensor[*Shape], dim: int) -> Tensor[*Shape]:
    """Cumulative sum along dimension. Shape-preserving operation."""
    ...

def cumprod[*Shape](input: Tensor[*Shape], dim: int) -> Tensor[*Shape]:
    """Cumulative product along dimension. Shape-preserving operation."""
    ...

def cummax[*Shape](
    input: Tensor[*Shape], dim: int
) -> tuple[Tensor[*Shape], Tensor[*Shape]]:
    """Cumulative maximum along dimension. Returns (values, indices). Shape-preserving operation."""
    ...

def cummin[*Shape](
    input: Tensor[*Shape], dim: int
) -> tuple[Tensor[*Shape], Tensor[*Shape]]:
    """Cumulative minimum along dimension. Returns (values, indices). Shape-preserving operation."""
    ...

# Tier 2: Additional reduction operations (always return tuples)
def mode(self: Tensor, dim: int = -1, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Mode along dimension. Returns (values, indices). Shape inference via meta-shape: torch.mode"""
    ...

def topk(
    self: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[Tensor, Tensor]:
    """Top k elements. Returns (values, indices). Shape inference via meta-shape: torch.topk"""
    ...

def sort[*Shape](
    input: Tensor[*Shape], dim: int = -1, descending: bool = False, stable: bool = False
) -> tuple[Tensor[*Shape], Tensor[*Shape]]:
    """Sort tensor. Returns (values, indices). Shape-preserving operation."""
    ...

def kthvalue(
    self: Tensor, k: int, dim: int = -1, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """Kth smallest value. Returns (values, indices). Shape inference via meta-shape: torch.kthvalue"""
    ...

# Tier 3: Statistical operations returning tuples
def var_mean(
    self: Tensor,
    dim: int | tuple[int, ...] | None = None,
    unbiased: bool = True,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Variance and mean. Returns (var, mean). Shape inference via meta-shape: torch.var_mean"""
    ...

def std_mean(
    self: Tensor,
    dim: int | tuple[int, ...] | None = None,
    unbiased: bool = True,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Standard deviation and mean. Returns (std, mean). Shape inference via meta-shape: torch.std_mean"""
    ...

# ==== Phase 1.3: Tensor Creation Operations ====

def zeros_like[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Create zeros with same shape. Shape inference via generic fixture signature."""
    ...

def ones_like[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Create ones with same shape. Shape inference via generic fixture signature."""
    ...

def full_like[*Shape](input: Tensor[*Shape], fill_value: float) -> Tensor[*Shape]:
    """Create tensor with same shape filled with value. Shape inference via generic fixture signature."""
    ...

def empty_like[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Create uninitialized tensor with same shape. Shape inference via generic fixture signature."""
    ...

def rand_like[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Create random tensor [0,1) with same shape. Shape inference via generic fixture signature."""
    ...

def randn_like[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Create random normal tensor with same shape. Shape inference via generic fixture signature."""
    ...

def diag_embed(self: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    """Create diagonal tensor. Shape inference via meta-shape: torch.diag_embed"""
    ...

def tril[*Shape](input: Tensor[*Shape], diagonal: int = 0) -> Tensor[*Shape]:
    """Lower triangular part. Shape inference via generic fixture signature."""
    ...

def triu[*Shape](input: Tensor[*Shape], diagonal: int = 0) -> Tensor[*Shape]:
    """Upper triangular part. Shape inference via generic fixture signature."""
    ...

def tril_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """Indices of lower triangular part. Shape inference via meta-shape: torch.tril_indices"""
    ...

def triu_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """Indices of upper triangular part. Shape inference via meta-shape: torch.triu_indices"""
    ...

# ==== Phase 1.4: Basic Linear Algebra Operations ====

# Note: matmul is already defined above with static typing at line 341
# We keep it there for backward compatibility, but meta-shape handles general cases

def mm[N, K, M](input: Tensor[N, K], mat2: Tensor[K, M]) -> Tensor[N, M]:
    """Matrix multiplication (2D @ 2D). Output: [N, M]."""
    ...

def bmm[B, N, K, M](input: Tensor[B, N, K], mat2: Tensor[B, K, M]) -> Tensor[B, N, M]:
    """Batch matrix multiplication (3D @ 3D). Output: [B, N, M]."""
    ...

def mv(self: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector multiplication (2D @ 1D). Shape inference via meta-shape: torch.mv"""
    ...

def dot(input: Tensor, other: Tensor) -> Tensor[()]:
    """Dot product (1D @ 1D → scalar). Returns scalar tensor."""
    ...

# ==== Phase 2: Arithmetic & Basic Operations ====
# All operations preserve shape (use IdentityMetaShape)

# Arithmetic operations (element-wise)
def add[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise addition. Shape inference via generic fixture signature."""
    ...

def sub[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise subtraction. Shape inference via generic fixture signature."""
    ...

def mul[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise multiplication. Shape inference via generic fixture signature."""
    ...

def div[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise division. Shape inference via generic fixture signature."""
    ...

def pow[*Shape](input: Tensor[*Shape], exponent: float | Tensor) -> Tensor[*Shape]:
    """Element-wise power. Shape inference via generic fixture signature."""
    ...

def neg[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise negation. Shape inference via generic fixture signature."""
    ...

def abs[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise absolute value. Shape inference via generic fixture signature."""
    ...

def floor[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise floor. Shape inference via generic fixture signature."""
    ...

def ceil[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise ceiling. Shape inference via generic fixture signature."""
    ...

def round[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise rounding. Shape inference via generic fixture signature."""
    ...

# Point-wise mathematical operations
def sin[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise sine. Shape inference via generic fixture signature."""
    ...

def cos[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise cosine. Shape inference via generic fixture signature."""
    ...

def tan[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise tangent. Shape inference via generic fixture signature."""
    ...

def exp[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise exponential. Shape inference via generic fixture signature."""
    ...

def log[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise natural logarithm. Shape inference via generic fixture signature."""
    ...

def sqrt[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise square root. Shape inference via generic fixture signature."""
    ...

def tanh[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise hyperbolic tangent. Shape inference via generic fixture signature."""
    ...

# Comparison operations
def eq[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise equality. Shape inference via generic fixture signature."""
    ...

def ne[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise inequality. Shape inference via generic fixture signature."""
    ...

def lt[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise less than. Shape inference via generic fixture signature."""
    ...

def le[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise less than or equal. Shape inference via generic fixture signature."""
    ...

def gt[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise greater than. Shape inference via generic fixture signature."""
    ...

def ge[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise greater than or equal. Shape inference via generic fixture signature."""
    ...

# Logical operations
def logical_and[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise logical AND. Shape inference via generic fixture signature."""
    ...

def logical_or[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise logical OR. Shape inference via generic fixture signature."""
    ...

def logical_not[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Element-wise logical NOT. Shape inference via generic fixture signature."""
    ...

# Clamping
def clamp[*Shape](
    input: Tensor[*Shape], min: float | None = None, max: float | None = None
) -> Tensor[*Shape]:
    """Clamp tensor values. Shape inference via generic fixture signature."""
    ...

def clip[*Shape](
    input: Tensor[*Shape], min: float | None = None, max: float | None = None
) -> Tensor[*Shape]:
    """Alias for clamp. Shape inference via generic fixture signature."""
    ...

# Activation functions (relu is most common, others in torch.nn.functional)
def relu[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """ReLU activation. Shape inference via generic fixture signature."""
    ...

# Additional mathematical operations
def atan2[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise arctangent of input/other. Shape inference via generic fixture signature."""
    ...

def hypot[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise hypotenuse. Shape inference via generic fixture signature."""
    ...

def lerp[*Shape](input: Tensor[*Shape], end: Tensor, weight: float) -> Tensor[*Shape]:
    """Linear interpolation. Shape inference via generic fixture signature."""
    ...

def fmod[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise modulo. Shape inference via generic fixture signature."""
    ...

def remainder[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise remainder. Shape inference via generic fixture signature."""
    ...

def copysign[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Copy sign. Shape inference via generic fixture signature."""
    ...

def nextafter[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Next floating-point value. Shape inference via generic fixture signature."""
    ...

def erf[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Error function. Shape inference via generic fixture signature."""
    ...

def erfc[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Complementary error function. Shape inference via generic fixture signature."""
    ...

def erfinv[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Inverse error function. Shape inference via generic fixture signature."""
    ...

def lgamma[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Log gamma function. Shape inference via generic fixture signature."""
    ...

def digamma[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Digamma function. Shape inference via generic fixture signature."""
    ...

def polygamma[*Shape](n: int, input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Polygamma function. Shape inference via generic fixture signature."""
    ...

def asinh[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Inverse hyperbolic sine. Shape inference via generic fixture signature."""
    ...

def acosh[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Inverse hyperbolic cosine. Shape inference via generic fixture signature."""
    ...

def atanh[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Inverse hyperbolic tangent. Shape inference via generic fixture signature."""
    ...

def deg2rad[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Convert degrees to radians. Shape inference via generic fixture signature."""
    ...

def rad2deg[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Convert radians to degrees. Shape inference via generic fixture signature."""
    ...

# Bitwise operations
def bitwise_and[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Bitwise AND. Shape inference via generic fixture signature."""
    ...

def bitwise_or[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Bitwise OR. Shape inference via generic fixture signature."""
    ...

def bitwise_xor[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Bitwise XOR. Shape inference via generic fixture signature."""
    ...

def bitwise_not[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Bitwise NOT. Shape inference via generic fixture signature."""
    ...

def bitwise_left_shift[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Bitwise left shift. Shape inference via generic fixture signature."""
    ...

def bitwise_right_shift[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Bitwise right shift. Shape inference via generic fixture signature."""
    ...

# Additional comparison/validation operations
def isclose[*Shape](
    input: Tensor[*Shape], other: Tensor, rtol: float = 1e-05, atol: float = 1e-08
) -> Tensor[*Shape]:
    """Check if tensors are close. Shape inference via generic fixture signature."""
    ...

def isreal[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Check if elements are real. Shape inference via generic fixture signature."""
    ...

def isposinf[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Check if elements are positive infinity. Shape inference via generic fixture signature."""
    ...

def isneginf[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Check if elements are negative infinity. Shape inference via generic fixture signature."""
    ...

def maximum[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise maximum. Shape inference via generic fixture signature."""
    ...

def minimum[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise minimum. Shape inference via generic fixture signature."""
    ...

def fmax[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise maximum (NaN handling). Shape inference via generic fixture signature."""
    ...

def fmin[*Shape](input: Tensor[*Shape], other: Tensor) -> Tensor[*Shape]:
    """Element-wise minimum (NaN handling). Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 4: Advanced Linear Algebra Operations
# ==============================================================================

# Advanced matmul operations
def tensordot(
    self: Tensor, other: Tensor, dims: int | tuple[list[int], list[int]] = 2
) -> Tensor:
    """Tensor contraction over specified dimensions. Shape inference via meta-shape: torch.tensordot"""
    ...

def einsum(spec: str, *operands: Tensor) -> Tensor:
    """Einstein summation convention. Shape inference via meta-shape: torch.einsum"""
    ...

# Eigenvalue decomposition
def eig(self: Tensor, eigenvectors: bool = False) -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition. Shape inference via meta-shape: torch.eig"""
    ...

def eigh(self: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Hermitian eigenvalue decomposition. Shape inference via meta-shape: torch.eigh"""
    ...

# Cholesky decomposition
def cholesky[*Shape](input: Tensor[*Shape], upper: bool = False) -> Tensor[*Shape]:
    """Cholesky decomposition. Shape inference via generic fixture signature."""
    ...

# Linear system solvers
def solve(self: Tensor, other: Tensor) -> Tensor:
    """Solve linear system. Shape inference via meta-shape: torch.solve"""
    ...

def triangular_solve(self: Tensor, other: Tensor, upper: bool = True) -> Tensor:
    """Solve triangular system. Shape inference via meta-shape: torch.triangular_solve"""
    ...

def cholesky_solve(self: Tensor, other: Tensor, upper: bool = False) -> Tensor:
    """Solve using Cholesky. Shape inference via meta-shape: torch.cholesky_solve"""
    ...

def lu_solve(self: Tensor, other: Tensor, LU_pivots: Tensor) -> Tensor:
    """Solve using LU decomposition. Shape inference via meta-shape: torch.lu_solve"""
    ...

# Matrix inverse
def inverse[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Matrix inverse. Shape inference via generic fixture signature."""
    ...

# Determinant
def det[*Batch, M, N](input: Tensor[*Batch, M, N]) -> Tensor[*Batch]:
    """Determinant. Returns batch dimensions only (drops last 2 dims)."""
    ...

def logdet[*Batch, M, N](input: Tensor[*Batch, M, N]) -> Tensor[*Batch]:
    """Log determinant. Returns batch dimensions only (drops last 2 dims)."""
    ...

def slogdet(self: Tensor) -> tuple[Tensor, Tensor]:
    """Sign and log determinant. Shape inference via meta-shape: torch.slogdet"""
    ...

# Matrix power and exponential
def matrix_power[*Shape](input: Tensor[*Shape], n: int) -> Tensor[*Shape]:
    """Matrix power. Shape inference via generic fixture signature."""
    ...

def matrix_exp[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Matrix exponential. Shape inference via generic fixture signature."""
    ...

# Trace
def trace[*Batch, M, N](input: Tensor[*Batch, M, N]) -> Tensor[*Batch]:
    """Matrix trace. Returns batch dimensions only (drops last 2 dims)."""
    ...

# Matrix rank
def matrix_rank[*Batch, M, N](
    input: Tensor[*Batch, M, N], tol: float = None, symmetric: bool = False
) -> Tensor[*Batch]:
    """Matrix rank. Returns batch dimensions only (drops last 2 dims)."""
    ...

# ==============================================================================
# Phase 5: Advanced Indexing & Conditional Operations
# ==============================================================================

# Conditional operations
def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Conditional element-wise selection. Shape inference via meta-shape: torch.where"""
    ...

def masked_fill[*Shape](
    input: Tensor[*Shape], mask: Tensor, value: float
) -> Tensor[*Shape]:
    """Fill masked elements. Shape inference via generic fixture signature."""
    ...

def masked_scatter[*Shape](
    input: Tensor[*Shape], mask: Tensor, source: Tensor
) -> Tensor[*Shape]:
    """Scatter into masked positions. Shape inference via generic fixture signature."""
    ...

# Advanced indexing operations
def index_add[*Shape](
    input: Tensor[*Shape], dim: int, index: Tensor, source: Tensor, alpha: float = 1
) -> Tensor[*Shape]:
    """Add values at indices. Shape inference via generic fixture signature."""
    ...

def index_copy[*Shape](
    input: Tensor[*Shape], dim: int, index: Tensor, source: Tensor
) -> Tensor[*Shape]:
    """Copy values to indices. Shape inference via generic fixture signature."""
    ...

def index_put[*Shape](
    input: Tensor[*Shape],
    indices: tuple[Tensor, ...],
    values: Tensor,
    accumulate: bool = False,
) -> Tensor[*Shape]:
    """Put values at indices. Shape inference via generic fixture signature."""
    ...

def index_fill[*Shape](
    input: Tensor[*Shape], dim: int, index: Tensor, value: float
) -> Tensor[*Shape]:
    """Fill indices with value. Shape inference via generic fixture signature."""
    ...

# Take/put operations
def take[*IndexShape](input: Tensor, index: Tensor[*IndexShape]) -> Tensor[*IndexShape]:
    """Take elements at indices. Output shape matches index shape."""
    ...

def take_along_dim(self: Tensor, indices: Tensor, dim: int) -> Tensor:
    """Take along dimension. Shape inference via meta-shape: torch.take_along_dim"""
    ...

def put[*Shape](
    input: Tensor[*Shape], index: Tensor, source: Tensor, accumulate: bool = False
) -> Tensor[*Shape]:
    """Put values at indices. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 6: Specialized Operations
# ==============================================================================

# Random sampling operations
def bernoulli[*Shape](input: Tensor[*Shape], p: float = 0.5) -> Tensor[*Shape]:
    """Sample from Bernoulli distribution. Shape inference via generic fixture signature."""
    ...

def multinomial(self: Tensor, num_samples: int, replacement: bool = False) -> Tensor:
    """Sample from multinomial distribution. Shape inference via meta-shape: torch.multinomial"""
    ...

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

def poisson[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Sample from Poisson distribution. Shape inference via generic fixture signature."""
    ...

# Tensor property functions
def numel[*Dims](self: Tensor[*Dims]) -> int:
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

# ==============================================================================
# Additional Math Operations
# ==============================================================================

def rsqrt[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Reciprocal square root (1/sqrt(x)). Shape-preserving element-wise operation."""
    ...

def outer(self: Tensor, vec2: Tensor) -> Tensor:
    """Outer product of two 1D tensors. Shape inference via meta-shape: torch.outer"""
    ...

def polar[*Shape](abs: Tensor[*Shape], angle: Tensor[*Shape]) -> Tensor[*Shape]:
    """Construct complex tensor from polar coordinates. Shape-preserving operation."""
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
