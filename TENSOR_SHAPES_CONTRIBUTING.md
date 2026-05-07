# Contributing to Tensor Shape Support

Pyrefly's tensor shape tracking is designed so that coverage of the PyTorch
library can be extended without understanding Pyrefly's internals. This page
explains the three mechanisms for specifying shape transforms and how to add
new ones.

## Architecture overview

Shape tracking uses three complementary mechanisms:

1. **Fixture stubs** — `.pyi` files with shape-generic type signatures.
   Covers modules like `nn.Linear`, `nn.Conv2d`, and functions like
   `torch.mm`.
2. **DSL functions** — shape transform specifications written in a tiny
   Python subset, registered in `tensor_ops_registry.rs`. Covers operations
   with complex shape logic like `reshape`, `cat`, `transpose`, and
   `F.interpolate`.
3. **Special handlers** — built into Pyrefly for constructs that need
   deeper type system integration, like `nn.Sequential` chaining, `.shape`
   attribute access, and `.size()`.

Most contributions involve fixture stubs or DSL functions. Special handlers
require changes to Pyrefly's Rust source.

## Fixture stubs

### Where they live

```
test/tensor_shapes/fixtures/torch/
├── __init__.pyi
├── nn/
│   ├── __init__.pyi      # nn.Linear, nn.Conv2d, nn.LSTM, etc.
│   └── functional.pyi    # F.relu, F.softmax, F.conv2d, etc.
├── distributions/
│   └── ...               # torch.distributions
└── ...
```

The `search_path` config option tells Pyrefly to look here for type
information, overriding the real `torch` stubs.

### How stubs work

A fixture stub provides a shape-generic type signature. For example,
`nn.Linear`:

```python
class Linear[N, M](Module):
    def __init__(self, in_features: Dim[N], out_features: Dim[M],
                 bias: bool = True) -> None: ...

    def forward[*Xs](self, input: Tensor[*Xs, N]) -> Tensor[*Xs, M]: ...
```

The constructor captures the input and output dimensions as type parameters.
The `forward` method uses those parameters plus a variadic `*Xs` for batch
dimensions.

### Writing a new stub

1. **Identify the shape signature.** What are the input dimensions, output
   dimensions, and how do they relate?
2. **Make constructor parameters `Dim[X]`** for parameters that determine
   tensor dimensions. Non-shape parameters (`bias`, `dropout`) stay as
   their original types.
3. **Write the `forward` signature** expressing the shape transform. Use
   `*Xs` or `*Bs` for batch dimensions that pass through unchanged.
4. **Add the stub** to the appropriate `.pyi` file in the fixtures
   directory.
5. **Test it** by writing a small model that uses the op and running the
   checker.

### Example: adding a new module

Suppose you want to add `nn.GroupNorm`, which preserves spatial dimensions:

```python
class GroupNorm[NumGroups, NumChannels](Module):
    def __init__(
        self,
        num_groups: Dim[NumGroups],
        num_channels: Dim[NumChannels],
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None: ...

    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...
```

Since `GroupNorm` doesn't change the shape, the forward signature is simply
`Tensor[*S] -> Tensor[*S]`.

## DSL functions

### Where they live

DSL functions are registered in:

```
tensor_ops_registry.rs
```

Each entry maps a qualified PyTorch function name to a shape transform
specification written in a tiny Python subset.

### The DSL subset

The DSL supports:

- Lists and list comprehensions
- Arithmetic (`+`, `-`, `*`, `//`)
- `zip`, `len`, indexing
- `Tensor(shape=[...])` to construct result shapes
- `self.shape` to access input shapes
- Conditionals (`if`/`else`)

### Example: `torch.repeat`

```python
def repeat_ir(self: Tensor, sizes: list[int | symint]) -> Tensor:
    return Tensor(shape=[d * r for d, r in zip(self.shape, sizes)])
```

This says: the output shape is the element-wise product of the input shape
and the `sizes` argument.

### Example: `torch.cat`

```python
def cat_ir(tensors: list[Tensor], dim: int = 0) -> Tensor:
    shapes = [t.shape for t in tensors]
    result = list(shapes[0])
    for s in shapes[1:]:
        result[dim] = result[dim] + s[dim]
    return Tensor(shape=result)
```

This sums the shapes along the concatenation dimension and preserves all
others.

### Adding a new DSL function

1. **Write the shape transform** in the DSL subset. Focus on the
   relationship between input and output shapes.
2. **Register it** in `tensor_ops_registry.rs` with the qualified PyTorch
   name (e.g., `"torch.nn.functional.adaptive_avg_pool2d"`).
3. **Test it** by using the op in a model and checking that `reveal_type`
   produces the expected shape.

## Ported models

### Where they live

```
test/tensor_shapes/models/
```

Each file is a fully annotated port of a real-world PyTorch model with
`assert_type` checkpoints and smoke tests.

### Adding a new model

1. Choose a model from [TorchBench](https://github.com/pytorch/benchmark)
   or another source.
2. Port it using the [tutorials](https://pyrefly.org/en/docs/tensor-shapes-tutorial-basics/)
   or the [agent skill](https://pyrefly.org/en/docs/tensor-shapes-ai-porting/).
3. Add `assert_type` after every shape-changing operation.
4. Add smoke tests at the bottom of the file.
5. Run `verify_port.sh` to check for issues.

### `verify_port.sh`

This script checks a ported model for common issues:

```bash
.claude/skills/port-model/verify_port.sh test/tensor_shapes/models/<model>.py
```

It reports:

| Metric | Description |
|--------|------------|
| `ig` | `type: ignore` count |
| `bs` | Bare `Tensor` in signatures |
| `bv` | Bare `Tensor` in variable annotations |
| `sh` | Shaped `assert_type` count |
| `ba` | Bare `assert_type` count |
| `sm` | Smoke test count |

## Testing

After adding stubs, DSL functions, or ported models, run the test suite:

```bash
# Run a specific test
buck test pyrefly:pyrefly_library -- tensor_shape

# Run all tests
buck test pyrefly:pyrefly_library
```

For external contributors using cargo:

```bash
cargo test tensor_shape
```
