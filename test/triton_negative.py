import triton
import triton.language as tl

@triton.jit(strict=True)
def test_invalid_store(ptr):
    x = tl.load(ptr)
    # Test 1: Shape mismatch, widens to AnyShape
    tl.store(ptr, x + 1)  # type: ignore

@triton.jit(strict=True)
def test_shape_mismatch(ptr1, ptr2):
    x = tl.load(ptr1)
    y = tl.load(ptr2)
    # Test 2: Dimension mismatch
    z = x + y  # type: ignore

@triton.jit(strict=True)
def test_unsupported_api(ptr):
    # Test 3: Unknown API, falls back to Any
    x = tl.unknown_api(ptr)  # type: ignore

@triton.jit(strict=True)
def test_bad_constexpr(SIZE):
    # Test 4: missing tl.constexpr annotation
    if SIZE == 128:  # type: ignore
        pass

@triton.jit(strict=True)
def test_rank_inconsistency(ptr):
    # Test 5: Rank inconsistency broadcast
    offsets = tl.arange(0, 128)
    bad_broadcast = tl.broadcast(offsets, (128, 128))  # type: ignore
