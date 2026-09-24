# How Shape Tracking Works

## Core concepts

**`Tensor[[B, C, H, W]]`** — a tensor with typed dimensions. `Tensor` is an
ordinary generic class with one shape parameter
(`class Tensor[Shape: IntTuple = IntTuple]`), so a
multi-dim shape goes in DOUBLE brackets. Each dimension can be a literal
(`3`, `64`), a type variable (`B`, `C`), or an arithmetic expression
(`D // NHead`, `2 * H - 1`, `H * W`). Single-bracket multi-dim
(`Tensor[B, C, H, W]`) is obsolete and does not type-check. Single brackets are
for a whole shape: `Tensor[S]` where `S: IntTuple`. NumPy and JAX arrays use the
same ordinary-class pattern; the legacy `@shaped_array` decorator is not used by
the current stubs.

**`Int[X]`** — bridges a runtime integer to a type-level symbol. When a
function takes `dim: Int[D]` and receives `64`, the checker binds `D = 64`.
All arithmetic on Int values produces Int results: `dim // 2` is `Int[D // 2]`,
`dim * 3` is `Int[D * 3]`, etc. These expressions propagate through constructor
args, method params, and tensor shapes.

**Type variables model symbolic integers.** A method
`forward[B: IntVar, T: IntVar]` has two symbolic integers bound at each call
site. Class-level params (`class Encoder[D: IntVar, NHead: IntVar]`) are bound
at construction and fixed for the instance. Only independent degrees of freedom
get type params — derived dims use expressions (`D // NHead`, not a separate
`HeadDim` param).

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
- `Tensor[S] → Tensor[S]` with `S: IntTuple` — preserves the whole shape
  (e.g., `F.relu`, `nn.LayerNorm`). For a *trailing* dim after any batch
  shape, use `Tensor[[*Bs, D]]` with `Bs: IntTuple`; use
  `*Elements[Bs]` only when annotations evaluate eagerly at runtime.
- Generic params — capture constructor args, compute output shape in `forward`
  (e.g., `nn.Linear[In, Out]`, `nn.Conv2d[InC, OutC, K, S, P, D]`)
- `Int[N]` capture — binds a runtime int arg to a type-level dim

**How to check if an op is supported:** Open the `.pyi` file and search for the
class or function. A bare `Tensor` is shorthand for the gradual
`Tensor[IntTuple]`, so its rank and dimensions are not tracked. If
it uses `Self`, a whole-shape `Tensor[S]` (`S: IntTuple`), generics, or a call to
a shape function (`Tensor[reshape_shape(Shape, NewShape)]`), it's tracked.

**How to recover a missing shape (only if stub changes are in scope):** first
classify the result. A declared bare return is gradual; an omitted public symbol
in a partial overlay may be unavailable; a third-party call may resolve from its
real package. For a true stub gap, use `Self` for identity ops, `Tensor[S]`
(`S: IntTuple`) for shape-preserving ops, generic params for transforms, or a
shape function call for argument-dependent computation. Otherwise preserve the
known component contract with a cast or typed interface and record the boundary.

**Third-party overlays:** `tensor-shapes/pyrefly-einops-stubs` provides
shape-aware `rearrange`, `reduce`, `repeat`, and `einsum`. Add that root whenever
the model imports einops. Without it, the real package can make a transform
incorrectly appear to return the input shape unchanged. Dynamic `axes_lengths`
may still need a precise local cast.

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
- Legacy `@shaped_array` attribute and indexing behavior for pinned older stubs
- Tuple slicing, star unpacking (`expr.rs`)

Current array stubs declare the `.shape` attribute normally. Their `__getitem__`
overloads call the `index_shape` DSL operation for integer, slice, tensor, and
multi-axis indexing; they do not use the legacy hard-coded indexing behavior.

**How to check:** These are less discoverable — search the Rust source or ask.

## When shapes are lost — trace upstream

When a result appears unrefined, the op that APPEARS to lose shapes is usually
not the problem. Trace back:

1. **Is the input already bare?** Most operations cannot infer relationships to
   lost dimensions, although an operation with a fully specified target (such as
   `reshape` to literal dimensions) can establish a new output shape. Find the
   first loss and preserve only relationships justified downstream.
2. **`int` where `Int` is needed?** A new shape-bearing parameter should use
   `Int[X]`. In an existing public API that is a static narrowing, so preserve
   the signature unless the user agrees; regain a justified shape at the next
   component boundary instead.
3. **Collection element types lost?** Fixed list literals can preserve distinct
   shapes, but dynamically built or broadly annotated lists homogenize members.
   Probe the collection. A fixed or typed tuple is an option only if it preserves
   runtime behavior; otherwise contain the gradual boundary.
4. **Branch join widening?** Two branches produce different types. Preserve the
   original control flow and use a typed boundary; suggest a branch rewrite only
   as separately approved work.
5. **Inlined expressions?** `f(g(x))` can lose shapes that named intermediates
   preserve. Splitting the expression is optional source refactoring; otherwise
   use a narrow cast at the known boundary.
6. **Stub returning bare?** Check whether its return annotation computes a
   shape. If stub work is in scope, refine the signature or add a shape function;
   otherwise record the gap.
7. **Shape function missing?** When shape-logic work is in scope, add it in
   `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`, decorate it with
   `@type_shape_dsl_function`, and call it from the stub's return annotation.

## What remains gradual

Some boundaries are inherently data-dependent; others are valid dynamic Python
that the static shape type system cannot represent without redesigning the
program:
- **Mixed-shape tensor containers and heterogeneous module containers** whose
  runtime position determines a different shape or transform.
- **Dynamic module construction** through factories, YAML, `getattr`, or runtime
  lists whose member types are erased.
- **Data-dependent result counts**: `torch.nonzero`, `t[bool_mask]` (output
  length depends on mask content, not shape).
- **Data-dependent accumulation**: conditional `torch.cat` where element count
  depends on runtime control flow.
- **A1 algebraic gap**: `N * (X // N) = X` — unsound for floor division.
  Note: `(a * b) // b → a` IS simplified (sound).

For these, preserve the known contract with a narrow cast or typed interface and
report the boundary. Do not rewrite runtime structure unless the user asks.

## Commonly used model-port symbols

The model-port API commonly uses:

- **`Int`** — binds a runtime integer to a type-level symbol (`dim: Int[D]`).
- **`IntVar`** — the bound for a *scalar* dimension type param
  (`class Net[D: IntVar]`, `def forward[B: IntVar]`). Bare PEP 695 params
  (`forward[B]`) are obsolete — always give the bound.
- **`IntTuple`** — the bound for a *variadic / whole-shape* type param
  (`Bs: IntTuple`, `Shape: IntTuple`). A whole-shape tensor is `Tensor[S]`
  with `S: IntTuple`; a trailing known dimension is `Tensor[[*Bs, D]]`.
- **`Elements`** — unpacks a variadic batch inside a shape:
  `Tensor[[*Elements[Bs], D]]` with `Bs: IntTuple`. A bare `*Bs` splat
  checks identically and is the preferred spelling; `Elements` is only
  required when the annotation itself evaluates at runtime, since
  unpacking a bare `TypeVar` raises `TypeError`. That means: use bare
  `*Bs` in check-only files (including stubs and anything under
  `from __future__ import annotations`), and `*Elements[Bs]` in files
  that execute their annotations eagerly.
- **`assert_shape`** — runtime shape assertion (companion to compile-time
  `assert_type`).
- **`shape_extensions.torchscript`** — import this module instead of
  `shape_extensions` to make shape annotations survive TorchScript compilation.
  It re-exports everything `shape_extensions` does, and importing it enables
  compatibility mode. It must be an import rather than a call because
  TorchScript reads class attribute annotations out of `__annotations__`, so
  the mode has to be on before an annotated class body is evaluated.
- **`static_jaxtyping`** — declares the symbolic names used by existing
  jaxtyping annotations so Pyrefly can check them statically. For a production
  codebase whose goal is shape checking rather than native-syntax migration,
  this can be a lower-churn alternative to translating every annotation.
- **`IntTuples`**, **`MapIntTuples`**, **`Flag`**, **`ProxyMethod`**,
  **`broadcast`**, **`gufunc_broadcast`**, and **`index_shape`** —
  stub-authoring primitives; you rarely write these in a model port.
  `MapIntTuples[lambda S: Tensor[S], Shapes]` is how a stub accepts a
  collection of tensors and keeps each element's shape.

There is NO exported `TypeVar`; use `IntVar`.

**Variadic batch idiom** (any number of leading batch dims):

```python
def forward[Bs: IntTuple](
    self, x: Tensor[[*Bs, D]]
) -> Tensor[[*Bs, D]]: ...
```

(see `examples/tacotron2.py`, `examples/nanogpt.py`). In files whose
annotations evaluate at runtime, spell the splat `*Elements[Bs]` instead;
a bare `*Bs` raises `TypeError` when evaluated. Splats outside a shape
list (`Tensor[*S]`, `Tensor[*Bs, D]`) remain invalid.

**Shape-function internals** live in `shape_extensions.dsl` and appear only
inside `_shapes.pyi`: the constructors `IntTuple` and `IntTuples`, the computations
`concat`, `prod`, `sum` and `einsum`, the tests `is_concrete_int` and `is_int_value`,
and `Invalid("...")` to reject an
ill-formed call. When an argument is too open to determine an answer, return
`Int.gradual()`, `IntTuple.gradual()` or `IntTuples.gradual()` — a gradual
result degrades to an unrefined shape, whereas `Invalid` reports an error to the
user. You only touch these when authoring a shape rule, not when porting a model.
