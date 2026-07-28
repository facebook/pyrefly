import triton
import triton.language as tl

@triton.jit(strict=True)
def test_load_store(ptr, val):
    # Test 1: Load from pointer
    x = tl.load(ptr)
    
    # Test 2: Store matching dtype
    tl.store(ptr, x)

@triton.jit(strict=True)
def test_binary_op(ptr):
    x = tl.load(ptr)
    # Test 3: Binary op promotion
    y = x + 1.0
    tl.store(ptr, y)

@triton.jit(strict=True)
def test_arange():
    # Test 4: Arange
    offsets = tl.arange(0, 128)

@triton.jit(strict=True)
def test_constexpr(BLOCK_SIZE: tl.constexpr):
    # Test 5: constexpr folding
    if BLOCK_SIZE == 128:
        pass
