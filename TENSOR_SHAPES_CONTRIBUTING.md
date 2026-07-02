# Contributing to Tensor Shape Support

Pyrefly's tensor shape tracking is designed so most PyTorch coverage can be
extended by editing stubs and tests, without changing Pyrefly's Rust internals.
This page explains the main mechanisms and how to validate changes.

Most external contributions should be stub-only or example/test-only changes.
Kernel changes are possible, but they are a narrower workflow for changes to
Pyrefly's shape machinery or the `shape_extensions` runtime package.

## Architecture Overview

Shape tracking uses three complementary mechanisms:

1. **Fixture stubs**: `.pyi` files with shape-generic type signatures. These
   cover modules like `nn.Linear`, `nn.Conv2d`, and functions like `torch.mm`.
2. **Shape DSL functions**: shape transforms written in a small Python subset in
   `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`, decorated with
   `@shape_dsl_function`, and attached to stubs with `@uses_shape_dsl(...)`.
   These cover operations with computed shape logic like `reshape`, `cat`,
   pooling, convolution, and interpolation.
3. **Special handlers**: Pyrefly implementation logic for constructs that need
   deeper type system integration, like `nn.Sequential` chaining, `.shape`,
   `.size()`, `assert_shape`, and decorator interpretation.

The first two mechanisms live in `tensor-shapes/` and are the normal way to add
or improve shape coverage. Special handlers require Pyrefly implementation
changes and should be treated as kernel work.

## Fixture Stubs

### Where They Live

```text
tensor-shapes/pyrefly-torch-stubs/torch-stubs/
|-- __init__.pyi
|-- _shapes.pyi
|-- nn/
|   |-- __init__.pyi      # nn.Linear, nn.Conv2d, nn.LSTM, etc.
|   `-- functional.pyi    # F.relu, F.softmax, F.conv2d, etc.
|-- distributions/
|   `-- ...               # torch.distributions
`-- ...
```

The tensor-shape test runner passes `tensor-shapes/` as a Pyrefly search path,
so these stubs override the normal `torch` stubs during validation.

### How Stubs Work

A fixture stub provides a shape-generic type signature. For example,
`nn.Linear`:

```python
class Linear[N, M](Module):
    def __init__(
        self,
        in_features: Dim[N],
        out_features: Dim[M],
        bias: bool = True,
    ) -> None: ...

    def forward[*Xs](self, input: Tensor[*Xs, N]) -> Tensor[*Xs, M]: ...
```

The constructor captures input and output dimensions as type parameters. The
`forward` method uses those parameters plus a variadic `*Xs` for batch
dimensions.

### Writing a New Stub

1. Identify the shape signature: input dimensions, output dimensions, and how
   they relate.
2. Use `Dim[X]` for parameters that determine tensor dimensions. Non-shape
   parameters like `bias` and `dropout` stay as their original types.
3. Write the method or function signature expressing the shape transform. Use
   `*Xs` or `*Bs` for batch dimensions that pass through unchanged.
4. Add the stub to the appropriate `.pyi` file in `tensor-shapes/pyrefly-torch-stubs/torch-stubs`.
5. Add or update focused tests under `tensor-shapes/pyrefly-torch-stubs/test/`.

### Example: Adding a New Module

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

Since `GroupNorm` does not change shape, the forward signature is simply
`Tensor[*S] -> Tensor[*S]`.

## Shape DSL Functions

Use the DSL when a plain signature cannot express the output shape.

### Where They Live

DSL functions live in:

```text
tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi
```

Stubs attach a DSL function with `@uses_shape_dsl(...)`. For example, a stub may
declare a broad return type like `Tensor` and let the DSL refine the result shape
at call sites:

```python
from shape_extensions import uses_shape_dsl
from torch._shapes import reshape_ir

@uses_shape_dsl(reshape_ir)
def reshape(self: Tensor, shape: tuple[int, ...]) -> Tensor: ...
```

### The DSL Subset

The DSL is intentionally small. It supports common shape computation patterns,
including:

- `ShapedArray(shape=[...])` to construct result shapes
- `self.shape` and other shaped-array argument shapes
- Lists, slices, indexing, and comprehensions
- Arithmetic such as `+`, `-`, `*`, `//`, `%`, and `**`
- `if` / `else`
- Helper calls to other `@shape_dsl_function` functions
- DSL helpers from `shape_extensions.dsl`, such as `prod`, `sum`, `Unknown`,
  and `Error`

Keep DSL functions simple and algebraic. They are analyzed by Pyrefly; they are
not normal runtime implementations of PyTorch operations.

### Example: `torch.cat`

```python
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
```

This sums shapes along the concatenation dimension and preserves all others.

### Adding a New DSL Function

1. Write the shape transform in `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`.
2. Decorate it with `@shape_dsl_function`.
3. Attach it to the public stub with `@uses_shape_dsl(...)`.
4. Add positive tests that assert the computed shape.
5. Add negative tests with `# E:` expectations if the DSL should reject invalid
   shapes or report shape errors.

## Ported Models

### Where They Live

```text
tensor-shapes/pyrefly-torch-stubs/examples/
```

Each file is a fully annotated port of a real-world PyTorch model with
`assert_type` checkpoints and smoke tests.

### Adding a New Model

1. Choose a model from [TorchBench](https://github.com/pytorch/benchmark) or
   another source.
2. Port it using the
   [tutorials](https://pyrefly.org/en/docs/tensor-shapes-tutorial-basics/) or
   the [agent skill](https://pyrefly.org/en/docs/tensor-shapes-ai-porting/).
3. Add `assert_type` or `assert_shape` checkpoints after shape-changing
   operations.
4. Add smoke tests at the bottom of the file when runtime execution is useful.
5. Run `verify_port.sh` to check for common quality issues.

### `verify_port.sh`

This script checks a ported model for common issues:

```bash
tensor-shapes/skills/add-shape-types-to-torch-model/verify_port.sh tensor-shapes/pyrefly-torch-stubs/examples/<model>.py
```

It reports:

| Metric | Description |
|--------|-------------|
| `ig` | `type: ignore` count |
| `bs` | Bare `Tensor` in signatures |
| `bv` | Bare `Tensor` in variable annotations |
| `sh` | Shaped `assert_type` count |
| `ba` | Bare `assert_type` count |
| `sm` | Smoke test count |

## Testing Stub and Example Changes

For most contributions, the important validation is the tensor-shape Pyrefly
runner. It checks the focused tests, negative expectations, jaxtyping examples,
and the example corpus using the shape-aware stubs.

Build Pyrefly first, then run:

```bash
cargo build
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py
```

If your build uses a custom target directory, `run_pyrefly.py` respects
`CARGO_TARGET_DIR`. You can also pass the binary explicitly:

```bash
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py --pyrefly /path/to/pyrefly
```

Run a single suite while iterating:

```bash
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py --suite torch-positive
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py --suite torch-negative
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py --suite torch-examples
```

Use `--nocapture` when you want the full Pyrefly output on success. By default,
the runner prints a compact `PASS ...` line and only dumps checker output on
failure.

In an internal Buck checkout, the equivalent static validation targets are:

```bash
buck test tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_all_test
buck test tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_error_test
buck test tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_jaxtyping_test
buck test tensor-shapes/pyrefly-torch-stubs/test:tensor_shapes_jaxtyping_error_test
buck test tensor-shapes/pyrefly-torch-stubs/examples:torch_examples_test
```

The project-level `test.py` runner keeps tensor-shape validation separate from
the default Pyrefly test loop. To run just these validations through `test.py`:

```bash
python3 test.py --no-fmt --no-lint --no-test --tensor-shapes --no-conformance --no-jsonschema
```

## Runtime Tests

Runtime tests validate that the annotation helpers and runnable example models
behave correctly in Python, not just in Pyrefly's static checker.

The tests live in:

```text
tensor-shapes/pyrefly-torch-stubs/test/runtime_tests/
```

Run them from a Python 3.12+ virtualenv with `torch` installed:

```bash
python3.12 -m venv .tensor-shapes-venv
. .tensor-shapes-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py
```

Run one suite while iterating:

```bash
python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py --suite annotation
python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py --suite model
```

The runtime runner sets up import paths for `shape_extensions` and the runnable
example modules. In an internal Buck checkout, the existing runtime targets are:

```bash
buck test tensor-shapes/pyrefly-torch-stubs/test:annotation_runtime_test
buck test tensor-shapes/pyrefly-torch-stubs/test:model_runtime_test
```

## Kernel Tests

Most contributors should not need this section. Use these tests when you change
Pyrefly's tensor-shape kernel rather than only stubs or examples. Kernel changes
include:

- `shape_extensions` primitives or decorators
- `assert_shape` type-checker behavior
- `@shape_dsl_function` parsing, validation, or evaluation
- `@uses_shape_dsl` handling
- special handlers in Pyrefly's Rust source

The focused Pyrefly unit tests live in:

```text
pyrefly/lib/test/shape_dsl.rs
```

Run them with Cargo:

```bash
cargo test shape_dsl
```

In an internal Buck checkout:

```bash
buck test pyrefly:pyrefly_library -- shape_dsl
```

Kernel tests are intentionally much smaller than the stub/example suites. They
cover the core primitives and invariants; the tensor-shape stub tests stress
the DSL through realistic PyTorch signatures.

## Pre-Commit Checks

Before handing off changes, run formatting and linting:

```bash
./test.py --no-test --no-tensor-shapes --no-conformance --no-jsonschema
```

Also run the relevant tensor-shape checks for the files you touched:

- Stub/test/example changes: `python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py`
- Runtime helper or runnable model changes:
  `python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py`
- Kernel changes: `cargo test shape_dsl` or the Buck equivalent above
