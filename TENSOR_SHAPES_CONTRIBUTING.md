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
2. **Type-level shape DSL functions**: shape transforms written in a small
   Python subset in `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`,
   decorated with `@type_shape_dsl_function`, and called directly from public
   return annotations. These cover operations with computed shape logic like
   reductions, padding, pooling, and convolution.
3. **Special handlers**: Pyrefly implementation logic for constructs that need
   deeper type system integration, like `nn.Sequential` chaining, `.shape`,
   `.size()`, `assert_shape`, and decorator interpretation.

The first two mechanisms live in `tensor-shapes/` and are the normal way to add
or improve shape coverage. Some stubs still use the older
`@shape_dsl_function` and `@uses_shape_dsl(...)` mechanism while they are being
migrated. Do not add new V1 rules. Special handlers require Pyrefly
implementation changes and should be treated as kernel work.

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
        in_features: SymInt[N],
        out_features: SymInt[M],
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
2. Use `SymInt[X]` for parameters that determine tensor dimensions. Non-shape
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
        num_groups: SymInt[NumGroups],
        num_channels: SymInt[NumChannels],
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

Public stubs call a type-level DSL function directly in their return annotation.
For example:

```python
from shape_extensions import IntTuple, type_shape_dsl_function
import shape_extensions.dsl as dsl

@type_shape_dsl_function
def repeat_shape(shape: IntTuple, repeats: IntTuple) -> IntTuple:
    if len(repeats) < len(shape):
        return dsl.Invalid("repeat dimensions cannot be shorter than the input rank")
    extra = len(repeats) - len(shape)
    return dsl.IntTuple(
        repeats[i] if i < extra else shape[i - extra] * repeats[i]
        for i in range(len(repeats))
    )

def repeat[Shape: IntTuple, Repeats: IntTuple](
    self: Tensor[Shape], *sizes: *Repeats
) -> Tensor[repeat_shape(Shape, Repeats)]: ...
```

### The DSL Subset

The DSL is intentionally small. Its main value domains are `Int` for one shape
dimension and `IntTuple` for a complete shape. Runtime configuration values are
connected through `Flag[...]` type parameters on public signatures. The body
language supports common shape computations, including:

- `dsl.IntTuple(...)` to construct result shapes
- `len`, indexing, slicing, and bounded generator expressions
- Arithmetic such as `+`, `-`, `*`, `//`, and `%`
- `if` / `else`
- Single-assignment local variables
- Direct returns from other `@type_shape_dsl_function` helpers
- DSL operations such as `dsl.concat`, `dsl.prod`, and `dsl.Invalid`
- `dsl.Int.gradual()` for a gradual dimension expression, and a direct
  `return dsl.IntTuple.gradual()` for a gradual shape

Keep DSL functions simple and algebraic. They are analyzed by Pyrefly; they are
not normal runtime implementations of PyTorch operations.

### Integer Helper Arguments

Declare a helper parameter as `Int` when it consumes one dimension, or as
`Int | None` when `None` has a distinct meaning. For a public runtime integer
that passes through to a helper, prefer a type parameter bound exactly by shape
`Int` and annotate the runtime parameter with that type parameter:

```python
@type_shape_dsl_function
def resize_shape(size: Int) -> IntTuple:
    return dsl.IntTuple((size,))

def resize[N: Int](size: N) -> Tensor[resize_shape(N)]: ...
```

This form requires the bound to be exactly `Int`. Use `IntVar` instead when a
type parameter names a symbolic dimension in direct `IntTuple` or list shape
syntax. If that symbolic dimension is also passed to a DSL helper, wrap it with
`Int[...]` at the call boundary:

```python
def zeros[N: IntVar](n: Int[N]) -> Tensor[[N]]: ...
def resize_symbolic[N: IntVar](size: Int[N]) -> Tensor[resize_shape(Int[N])]: ...
```

Raw `IntVar` arguments and arithmetic such as `N + 1` are rejected in helper
calls; write `Int[N]` and `Int[N] + 1` instead. `Int[N] | None` written directly
as an argument is type-union syntax, not a runtime DSL value. Pass `Int[N]`,
`None`, or a type parameter whose bound is exactly `Int | None`, as appropriate.
An argument that still resolves to `Int | None` is accepted as gradual until
control flow narrows it; the non-`None` branch can then use it as an `Int`.

A broad runtime `int` used as a dimension becomes a gradual dimension, which
preserves known rank and other dimensions. `Any` remains unknown rather than
being treated as a gradual integer. `dsl.Int.gradual()` is itself an `Int`
expression, so it can participate in arithmetic and `dsl.IntTuple(...)`
construction. `dsl.IntTuple.gradual()` currently represents a whole gradual
shape only as a direct DSL-function return; it cannot be assigned to a local or
embedded in a larger expression.

`D[...]` and `D(...)` remain compatibility wrappers for annotations that Python
would otherwise evaluate eagerly. They do not replace `Int[...]`: `D[N]` still
contains a raw `IntVar` and is rejected, while `D[Int[N] + 1]` is valid.

Use `dsl.is_concrete_int(value)` with an `Int` or `Int | None` value when a
branch requires an integer literal known during shape evaluation; it is false
for `None`, symbolic dimensions, and gradual `Int` values. Use
`dsl.is_int_value(value)` to narrow the integer member of a compatible
`Flag[int | tuple[int, ...] | None]` value. That predicate does not prove the
integer is concrete.

The type-level DSL used by the NumPy and JAX stubs is a separate, smaller
subset, and it is still being built out. Two things about it are worth knowing
before writing one, because neither is guessable:

- A DSL function that uses an unsupported construct evaluates to `Unknown` at
  every call site, and the call site itself reports nothing. Type check the stub
  files to see the real diagnostic; the runner does this for you as the `stubs`
  suite.
- A parameter typed `int | tuple[int, ...]` cannot be iterated after narrowing
  with `is_int_value` alone. Leading with an `is None` check makes the narrowing
  work, so such parameters are declared `int | tuple[int, ...] | None` with a
  body that rejects `None`. Both `conv_shape` in the Torch stubs and
  `reshape_shape` in the JAX stubs do this.

### Example: reduction

```python
@type_shape_dsl_function
def reduce_shape(shape: IntTuple, dim: int, keepdim: bool) -> IntTuple:
    axis = dim % len(shape)
    return dsl.IntTuple(
        1 if keepdim and i == axis else shape[i]
        for i in range(len(shape))
        if keepdim or i != axis
    )
```

The public stub binds its input shape and runtime options to type parameters,
then calls `reduce_shape(...)` in the return annotation.

### Adding a New DSL Function

1. Write the shape transform in `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`.
2. Decorate it with `@type_shape_dsl_function`.
3. Bind the relevant public arguments with `Int`, `IntVar`, `IntTuple`, or
   `Flag[...]` type parameters and call the DSL function from the return
   annotation.
4. Add positive tests that use `assert_type` to check the computed shape.
5. Add negative tests with `# E:` expectations if the DSL should reject invalid
   shapes or report shape errors.

The older decorator-based DSL remains only for rules that have not yet been
migrated. Avoid combining V1 and V2 logic in new rules; if V2 cannot yet express
the operation, document the gap rather than adding new V1 surface area.

### Current Limitations

The type-level DSL uses a small, composable language and does not model every
shape behavior precisely. Current known limitations include symbolic
`arange` rounding, symbolic configuration values for `unfold` and `diag_embed`,
structured `tensordot` axis lists, products of symbolic-rank shapes or derived
symbolic dimensions, and list-based padding. Keep these cases gradual where
necessary, add focused tests, and leave a `TODO(stroxler)` at the affected rule
so the loss of precision remains visible.

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

It also type checks the stub files themselves, reported as a `stubs` suite.
This matters more than it sounds: Pyrefly reports errors only for the files it
is asked to check, so a stub reached through `--search-path` is silent. A stub
that fails to compile does not announce itself, it just stops contributing
types, and every call site quietly infers `Unknown` -- which looks like a
missing rule rather than a broken one. Checking the stubs directly turns that
into an error with a line number.

The Torch package opts out for now, via `check_stubs=False` in its
`run_pyrefly.py`. Most of its errors are in `torch-stubs/_shapes.pyi`, whose V1
`@shape_dsl_function` bodies are not valid Python. Type-level DSL files do check
cleanly, so migrating those rules is what removes the opt-out.

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

There are no Buck test targets for the stubs. An internal checkout runs the same
runner and only sources Pyrefly differently, via `--buck`:

```bash
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py --buck
```

To run every library at once, static and runtime, exactly as both CI systems do:

```bash
python3 tensor-shapes/run_tests.py           # add --buck in an internal checkout
python3 tensor-shapes/run_tests.py --static-only   # no virtualenv needed
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

Runtime tests need the shared virtualenv, which serves torch, numpy and jax
together. Bootstrapping is the only step that downloads anything, so it is also
the only step that needs network access -- on a Meta machine, via fwdproxy:

```bash
python3 tensor-shapes/bootstrap_venv.py            # add --fwdproxy internally
python3 tensor-shapes/run_tests.py --runtime-only
```

The virtualenv defaults to `~/.tensor-shapes-venv`; set `$TENSOR_SHAPES_VENV` to
put it elsewhere. The runners never create it, and never reach the network: if it
is missing they say so and print the bootstrap command. Type checking does not
need it at all.

Run one suite while iterating:

```bash
python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py --suite annotation
python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py --suite model
```

The runtime runner sets up import paths for `shape_extensions` and the runnable
example modules. Runtime tests are the same in an internal checkout: they run
against the virtualenv, never through Buck, so that no workflow ever rebuilds
torch, numpy or jax.

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

Python files in the tensor-shape packages use Ruff's formatter, rather than
Black. Format them from the repository root with the same Ruff version as CI:

```bash
uv tool run --from ruff==0.16.5 ruff format \
  tensor-shapes
```

The `skills` directory is documentation rather than corpus source and is
excluded because Ruff also formats Python snippets embedded in Markdown.

Before handing off changes, also run the repository formatting and linting:

```bash
./test.py --no-test --no-tensor-shapes --no-conformance --no-jsonschema
```

Also run the relevant tensor-shape checks for the files you touched:

- Stub/test/example changes: `python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py`
- Runtime helper or runnable model changes:
  `python tensor-shapes/pyrefly-torch-stubs/run_runtime_tests.py`
- Kernel changes: `cargo test shape_dsl` or the Buck equivalent above
