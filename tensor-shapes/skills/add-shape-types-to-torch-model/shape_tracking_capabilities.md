# How Shape Tracking Works

## Core concepts

**`Tensor[[B, C, H, W]]`** — a tensor with typed dimensions. `Tensor` takes a
single shape parameter (`class Tensor[Shape: _Shape = _AnyShape]`), so a
multi-dim shape goes in DOUBLE brackets. Each dimension can be a literal
(`3`, `64`), a type variable (`B`, `C`), or an arithmetic expression
(`D // NHead`, `2 * H - 1`, `H * W`). Single-bracket multi-dim
(`Tensor[B, C, H, W]`) is obsolete and does not type-check. Single brackets are
for a whole-shape carrier: `Tensor[S]` where `S: IntTuple`.

**`Int[X]`** — bridges a runtime integer to a type-level symbol. When a
function takes `dim: Int[D]` and receives `64`, the checker binds `D = 64`.
All arithmetic on Int values produces Int results: `dim // 2` is `Int[D // 2]`,
`dim * 3` is `Int[D * 3]`, etc. These expressions propagate through constructor
args, method params, and tensor shapes.

**Type variables model symbolic integers.** A method `forward[B, T]` has two
symbolic integers bound at each call site. Class-level params
(`class Encoder[D, NHead]`) are bound at construction and fixed for the
instance. Only independent degrees of freedom get type params — derived dims
use expressions (`D // NHead`, not a separate `HeadDim` param).

## The three shape-tracking mechanisms

Paths below are shown relative to the **stub root** — the directory Pyrefly
resolves the `torch` stubs from. It is `tensor-shapes/pyrefly-torch-stubs/torch-stubs/` in an fbsource
checkout; in other environments (a clone, or stubs installed into a virtualenv)
it lives elsewhere. `pyrefly dump-config` reports the resolved location.

### 1. Shape-aware stubs

**Location:** the stub root and its subdirectories (`nn/`,
`distributions/`, `optim/`, `quantization/`).

`.pyi` files with type signatures for PyTorch classes and functions. Common
patterns:
- `Self` return — preserves exact shape (e.g., `.float()`, `.contiguous()`)
- `Tensor[S] → Tensor[S]` with `S: IntTuple` — shape-preserving whole-shape
  carrier (e.g., `F.relu`, `nn.LayerNorm`). For a *trailing* dim after any batch
  shape, use `Tensor[[*Elements[Bs], D]]` with `Bs: IntTuple`.
- Generic params — capture constructor args, compute output shape in `forward`
  (e.g., `nn.Linear[In, Out]`, `nn.Conv2d[InC, OutC, K, S, P, D]`)
- `Int[N]` capture — binds a runtime int arg to a type-level dim

**How to check if an op is supported:** Open the `.pyi` file and search for the
class or function. If the return type is bare `Tensor`, shapes aren't tracked. If
it uses `Self`, a whole-shape `Tensor[S]` (`S: IntTuple`), generics, or a call to
a shape function (`Tensor[reshape_shape(Shape, NewShape)]`), it's tracked.

**How to recover a missing shape (only if the user opted into stub changes):**
Change the stub's return type. Use `Self` for identity ops, `Tensor[S]`
(`S: IntTuple`) for shape-preserving ops, generic params for transforms, or a
shape function call for argument-dependent computation. If stubs are
off-limits, leave the op untracked — it degrades to a bare `Tensor`, which you
record as a gap rather than fixing.

### 2. Type-level shape functions

**Location:** stub declarations call them directly in their return annotations,
in `tensor-shapes/pyrefly-torch-stubs/torch-stubs/**/*.pyi`; the functions live in
`tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi` and are imported from stubs as
`torch._shapes` because `torch-stubs` provides the `torch` package for type
checking.

Python-like shape functions evaluated at type-check time. Two parts:

- **Call site** (in the relevant stub file): the return annotation applies the
  function to the signature's own type parameters, so shape computation is part
  of the declared type rather than an attachment to it:

  ```python
  def reshape[Shape: IntTuple, NewShape: IntTuple](
      self: Tensor[Shape], *shape: *NewShape
  ) -> Tensor[reshape_shape(Shape, NewShape)]: ...
  ```

  An argument the function needs as a literal is captured with `Flag[int]`
  (`dim: Dim` with `Dim: Flag[builtins.int]`); a collection of input shapes
  arrives as an `IntTuples` value through
  `MapIntTuples[lambda S: Tensor[S], Shapes]` (see `torch.cat`).

- **Definitions** (`_shapes.pyi`): functions decorated with
  `@type_shape_dsl_function` that compute an output `IntTuple` (or `Int`) from
  input shapes and arguments. For example, `reshape_shape` handles `-1`
  inference and `cat_shape` sums along the concat dim.

**How to check if an op is supported:** Open the relevant stub declaration and
read its return annotation. If it calls a shape function, confirm that function
exists in `_shapes.pyi`.

**How to add support:** Write the function in `_shapes.pyi`, decorate it with
`@type_shape_dsl_function`, and call it from the stub's return annotation. These
functions are Python-like — look at existing ones for patterns. They support
conditionals (`x if cond else y`), comprehensions, calls to other
`@type_shape_dsl_function`s, and the `shape_extensions.dsl` helpers
(`dsl.IntTuple`, `dsl.concat`, `dsl.prod`, `dsl.is_concrete_int`, and
`dsl.Invalid("...")` to report an ill-formed call).

### 3. Special handlers

**Location:** `pyrefly/lib/alt/` (various `.rs` files)

Hard-coded Rust logic for patterns that don't fit stubs or shape functions:
- `nn.Sequential` chaining (`nn_module_specials.rs`)
- `.shape` attribute returning typed tuple (`attr.rs`)
- Tensor indexing — integer, slice, tensor, multi-axis (`expr.rs`)
- Tuple slicing, star unpacking (`expr.rs`)

**How to check:** These are less discoverable — search the Rust source or ask.

## When shapes are lost — trace upstream

When a result appears unrefined, the op that APPEARS to lose shapes is usually
not the problem. Trace back:

1. **Is the INPUT already bare?** No op can recover shapes from bare `Tensor`.
   Find where shapes were actually lost — that's the real fix.
2. **`int` where `Int` needed?** Shapes enter as unrefined when a function
   takes `int` instead of `Int[X]`. Fix: change the param type.
3. **`list` where `tuple` needed?** `torch.cat([a, b])` homogenizes element
   types. Fix: `torch.cat((a, b))`.
4. **Branch join widening?** Two branches produce different types → widening.
   Fix: compute output in each branch independently, or use Optional narrowing.
5. **Inlined expressions?** `f(g(x))` sometimes loses shapes that
   `y = g(x); f(y)` preserves. Fix: break into separate assignments.
6. **Stub returning bare?** Check whether its return annotation computes a
   shape. If not, fix the `.pyi` signature or add a shape function.
7. **Shape function missing?** Add it in `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`,
   decorate it with `@type_shape_dsl_function`, and call it from the stub's
   return annotation.

## What IS genuinely shapeless

Very few patterns truly can't be tracked:
- **Data-dependent result counts**: `torch.nonzero`, `t[bool_mask]` (output
  length depends on mask content, not shape)
- **Data-dependent accumulation**: conditional `torch.cat` where element count
  depends on runtime control flow
- **A1 algebraic gap**: `N * (X // N) = X` — unsound for floor division.
  Note: `(a * b) // b → a` IS simplified (sound).

Everything else should be trackable. If you think something is shapeless, check
the three mechanisms first — stubs, shape functions, special handlers.

## Current API surface

The `shape_extensions` package is what your port imports. Its public exports:

- **`Int`** — binds a runtime integer to a type-level symbol (`dim: Int[D]`).
- **`IntVar`** — the bound for a *scalar* dimension type param
  (`class Net[D: IntVar]`, `def forward[B: IntVar]`). Bare PEP 695 params
  (`forward[B]`) are obsolete — always give the bound.
- **`IntTuple`** — the bound for a *variadic / whole-shape* type param
  (`Bs: IntTuple`, `Shape: IntTuple`). A whole-shape tensor is `Tensor[S]`
  with `S: IntTuple`.
- **`Elements`** — unpacks a variadic batch inside a shape:
  `Tensor[[*Elements[Bs], D]]` with `Bs: IntTuple`.
- **`assert_shape`** — runtime shape assertion (companion to compile-time
  `assert_type`).
- **`shape_extensions.torchscript`** — import this module instead of
  `shape_extensions` to make shape annotations survive TorchScript compilation.
  It re-exports everything `shape_extensions` does, and importing it enables
  compatibility mode. It must be an import rather than a call because
  TorchScript reads class attribute annotations out of `__annotations__`, so
  the mode has to be on before an annotated class body is evaluated.
- **`shaped_array`** — `@shaped_array(shape=...)` class decorator for non-torch
  array types (numpy-style).
- **`IntTuples`**, **`MapIntTuples`**, **`Flag`**, **`ProxyMethod`** —
  stub-authoring primitives; you rarely write these in a port.
  `MapIntTuples[lambda S: Tensor[S], Shapes]` is how a stub accepts a
  collection of tensors and keeps each element's shape.

There is NO exported `TypeVar`; use `IntVar`.

**Variadic batch idiom** (any number of leading batch dims):

```python
def forward[Bs: IntTuple](
    self, x: Tensor[[*Elements[Bs], D]]
) -> Tensor[[*Elements[Bs], D]]: ...
```

(see `examples/tacotron2.py`, `examples/nanogpt.py`). The old `*Bs` / `Tensor[*S]`
/ `Tensor[*Bs, D]` PEP-646 style is obsolete.

**Shape-function internals** live in `shape_extensions.dsl` and appear only
inside `_shapes.pyi`: the constructors `IntTuple` and `IntTuples`, the computations
`concat`, `prod`, `sum` and `einsum`, the tests `is_concrete_int` and `is_int_value`,
and `Invalid("...")` to reject an
ill-formed call. When an argument is too open to determine an answer, return
`Int.gradual()`, `IntTuple.gradual()` or `IntTuples.gradual()` — a gradual
result degrades to an unrefined shape, whereas `Invalid` reports an error to the
user. You only touch these when authoring a shape rule, not when porting a model.
