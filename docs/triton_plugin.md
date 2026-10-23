# Triton support

Pyrefly recognizes `tl.constexpr` parameter annotations on functions decorated with
`@triton.jit` or `@triton.jit(...)`. Imported aliases and re-exports also work.

Triton uses `tl.constexpr` to mark a compile-time argument. The argument can contain
an integer, a Boolean, a string, a tuple, or another compile-time value.
The annotation does not require a Python `constexpr` instance. See the
[Triton constexpr implementation](https://github.com/triton-lang/triton/blob/main/python/triton/language/core.py).

Pyrefly currently represents these parameter values as `Any`. This permits scalar
operations and ordinary Python defaults without errors about the `constexpr`
class. Pyrefly continues to check other annotations and statements in the function.

```python
import triton
import triton.language as tl


@triton.jit
def next_block(block: tl.constexpr = 128) -> int:
    return block + 1
```

Functions without a recognized Triton decorator retain normal Python annotation
rules. A different decorator named `jit` or a different class named `constexpr`
does not enable this behavior. Nested functions require their own Triton decorator.

This support does not infer constants from kernel calls. It does not yet check
Triton pointer dtypes, tensor shapes, broadcasting, or kernel launch arguments.
It does not introduce a `strict=True` argument or a separate plugin interface.
These features remain outside this first step for
[issue #3884](https://github.com/facebook/pyrefly/issues/3884).

Run the regression tests without Triton or a GPU:

```bash
cargo test -p pyrefly --lib test_triton_constexpr
```
