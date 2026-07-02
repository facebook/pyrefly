# How Shape Tracking Works

## Core concepts

**`Tensor[B, C, H, W]`** — a tensor with typed dimensions. Each dimension can
be a literal (`3`, `64`), a type variable (`B`, `C`), or an arithmetic
expression (`D // NHead`, `2 * H - 1`, `H * W`).

**`Dim[X]`** — bridges a runtime integer to a type-level symbol. When a
function takes `dim: Dim[D]` and receives `64`, the checker binds `D = 64`.
All arithmetic on Dim values produces Dim results: `dim // 2` is `Dim[D // 2]`,
`dim * 3` is `Dim[D * 3]`, etc. These expressions propagate through constructor
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
- `Tensor[*S] → Tensor[*S]` — shape-preserving (e.g., `F.relu`, `nn.LayerNorm`)
- Generic params — capture constructor args, compute output shape in `forward`
  (e.g., `nn.Linear[In, Out]`, `nn.Conv2d[InC, OutC, K, S, P, D]`)
- `_Dim[N]` capture — captures runtime int args as type-level dims

**How to check if an op is supported:** Open the `.pyi` file and search for the
class or function. If the return type is bare `Tensor`, shapes aren't tracked —
unless the declaration has `@uses_shape_dsl(...)`. If it uses `Self`, `[*S]`,
generics, or a `@uses_shape_dsl(...)` decorator, it's tracked.

**How to recover a missing shape (only if the user opted into stub changes):**
Change the stub's return type. Use `Self` for identity ops, `Tensor[*S]` for
shape-preserving ops, generic params for transforms, or `@uses_shape_dsl(...)`
for shape functions that need argument-dependent computation. If stubs are
off-limits, leave the op untracked — it degrades to a bare `Tensor`, which you
record as a gap rather than fixing.

### 2. DSL functions

**Location:** declarations use `@uses_shape_dsl(ir_fn)` in
`tensor-shapes/pyrefly-torch-stubs/torch-stubs/**/*.pyi`; IR functions live in
`tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi` and are imported from stubs as
`torch._shapes` because `torch-stubs` provides the `torch` package for type
checking.

Python-like shape functions interpreted at type-check time. Two parts:

- **Declaration** (in the relevant stub file): imports an IR function and
  attaches it to a function or method with `@uses_shape_dsl(ir_fn)`. For
  `nn.Module` classes, the decorator can capture constructor arguments with
  `capture_init=[...]` and connect them to `forward`.

- **DSL definitions** (`_shapes.pyi`): Python-like functions that compute
  output shapes from input shapes and arguments. For example, `reshape_ir`
  handles `-1` inference, `cat_ir` sums along the concat dim.

**How to check if an op is supported:** Open the relevant stub declaration and
look for `@uses_shape_dsl(...)`. If it has a decorator, confirm the named IR
function exists in `_shapes.pyi`.

**How to add support:** Write a DSL function in `_shapes.pyi` that computes
the output shape, decorate it with `@shape_dsl_function`, then attach it from
the stub declaration with `@uses_shape_dsl(...)`. DSL functions are
Python-like — look at existing ones for patterns. The DSL supports conditionals
(`x if cond else y`), list comprehensions, and calls to helper functions like
`normalize_dim`.

### 3. Special handlers

**Location:** `pyrefly/lib/alt/` (various `.rs` files)

Hard-coded Rust logic for patterns that don't fit stubs or DSL:
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
2. **`int` where `Dim` needed?** Shapes enter as unrefined when a function
   takes `int` instead of `Dim[X]`. Fix: change the param type.
3. **`list` where `tuple` needed?** `torch.cat([a, b])` homogenizes element
   types. Fix: `torch.cat((a, b))`.
4. **Branch join widening?** Two branches produce different types → widening.
   Fix: compute output in each branch independently, or use Optional narrowing.
5. **Inlined expressions?** `f(g(x))` sometimes loses shapes that
   `y = g(x); f(y)` preserves. Fix: break into separate assignments.
6. **Stub returning bare?** Check whether it has `@uses_shape_dsl(...)`. If
   not, fix the `.pyi` signature or add DSL support.
7. **DSL missing?** Add the IR function in `tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`,
   decorate it with `@shape_dsl_function`, and attach it with
   `@uses_shape_dsl(...)`.

## What IS genuinely shapeless

Very few patterns truly can't be tracked:
- **Data-dependent result counts**: `torch.nonzero`, `t[bool_mask]` (output
  length depends on mask content, not shape)
- **Data-dependent accumulation**: conditional `torch.cat` where element count
  depends on runtime control flow
- **A1 algebraic gap**: `N * (X // N) = X` — unsound for floor division.
  Note: `(a * b) // b → a` IS simplified (sound).

Everything else should be trackable. If you think something is shapeless, check
the three mechanisms first — stubs, DSL, special handlers.
