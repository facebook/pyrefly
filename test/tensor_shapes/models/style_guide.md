# Tensor Shape Annotation Style Guide

A practical guide to adding tensor shape annotations to PyTorch models using
pyrefly's type system. Patterns and methodology drawn from
[21 ported TorchBenchmark models](models/).

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [How to Approach Typing a Model](#2-how-to-approach-typing-a-model)
3. [What Should Work (and what to do when it doesn't)](#3-what-should-work)
4. [What Won't Work (Yet)](#4-what-wont-work-yet)
5. [Generics: Batch and Sequence Dimensions](#5-generics-batch-and-sequence-dimensions)
6. [Class-Level Type Parameters](#6-class-level-type-parameters)
7. [Linear Pipeline](#7-linear-pipeline)
8. [Homogeneous Layer Stacking](#8-homogeneous-layer-stacking-transformers)
9. [Encoder-Decoder with Skip Connections](#9-encoder-decoder-with-skip-connections)
10. [Recursive Chains with Exponential Shapes](#10-recursive-chains-with-exponential-shapes)
11. [Config Classes](#11-config-classes)
12. [Techniques Reference](#12-techniques-reference)
13. [Smoke Tests](#13-smoke-tests)

---

## 1. Getting Started

### Setup

Tensor shape checking requires two things in your `pyrefly.toml`:

```toml
tensor-shapes = true

search_path = [
    "fixtures",
]
```

**`tensor-shapes = true`** enables shape inference for `Tensor`, including
subscript syntax (`Tensor[B, C, H, W]`), algebraic dimension arithmetic, and
shape-aware dispatch for operations like `conv2d`, `view`, and `cat`.

**`search_path`** points to a directory of *fixture stubs* — `.pyi` files that
provide shape-generic type signatures for PyTorch modules and functions. The
real `torch` library's type stubs don't carry shape information, so the
fixtures replace them with shape-aware versions (e.g., `nn.Conv2d.__init__`
that captures kernel size, stride, and padding as type-level values, and a
`forward` that computes the output spatial dimensions).

The fixtures also provide the `torch_shapes` package, which exports `Dim` — the
bridge between runtime integer values and type-level symbols. The package also
includes some utilities to support runtime evaluation of types with shapes.

### Concrete shape annotations

The simplest annotation is a fully concrete shape — no generics at all:

```python
x: Tensor[4, 3, 64, 64] = torch.randn(4, 3, 64, 64)
```

This says `x` is a 4D tensor with batch size 4, 3 channels, height 64, width 64.

### `assert_type` as documentation + regression test

`assert_type` is a no-op at runtime. The type checker verifies that the
expression has exactly the stated type. Use it after key operations to
document what shape you expect and catch regressions:

```python
h = F.relu(self.fc1(x))
assert_type(h, Tensor[B, 512])  # checked by pyrefly, zero runtime cost
```

In practice, you won't need `assert_type` for every intermediate — pyrefly
shows inferred shapes as inlay type hints in your editor. The hints appear
automatically, so you can verify shapes visually without writing any
assertions. Use `assert_type` only at key checkpoints where you want a
permanent regression guard.

---

## 2. How to Approach Typing a Model

### Step 1: Identify the degrees of freedom

Read the model and list the **independent** dimensions — batch size, sequence
length, channel counts, spatial sizes, number of heads, etc. Each independent
dimension gets exactly one type parameter. Two rules:

- **If two dims are always equal, use one param.** For example, if the image
  encoder's output channels always equal the mask decoder's transformer dim,
  use a single `D` — not separate `OutC` and `PromptD`.
- **If one dim is derived from another, express the relation.** Head dimension
  is `D // NHead`, not an independent `HeadDim`. Mask spatial is `4 * ES`, not
  an independent `MH`. Window count is `IS // WS`, not `NWindows`.

```python
# Bad: too many independent params, hides the relation
class Attention[D, NHead, HeadDim](nn.Module): ...

# Good: HeadDim is derived
class Attention[D, NHead](nn.Module):
    def __init__(self, dim: Dim[D], num_heads: Dim[NHead]) -> None:
        self.head_dim = dim // num_heads  # Dim[D // NHead]
```

### Step 2: Type the constructor

Constructor params that set dimensions become `Dim[...]`. This binds class
type parameters. Sub-modules constructed with these Dims automatically get
typed — Conv2d, Linear, LSTM, etc. stubs capture channel/feature dims.

```python
class PromptEncoder[D, ES, MIC](nn.Module):
    def __init__(self, embed_dim: Dim[D], emb_size: Dim[ES],
                 mask_in_chans: Dim[MIC]) -> None:
        self.mask_conv1 = nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2)
        # Conv2d captures MIC//4 as output channels
```

### Step 3: Type the forward signature

Use class params (fixed at construction) for architecture dims and method
params (vary per call) for batch/sequence/spatial dims:

```python
def forward[B, N, M](
    self,
    points: tuple[Tensor[B, N, 2], Tensor[B, N]] | None,
    boxes: Tensor[B, M, 4] | None,
    masks: Tensor[B, 1, 4 * ES, 4 * ES] | None,
) -> tuple[Tensor, Tensor[B, D, ES, ES]]:
```

Then verify intermediates with `assert_type` at key checkpoints — after
reshapes, matmuls, conv chains, and branch joins.

### Binding order matters

When type vars appear in derived positions (`Tensor[B, QH * QW, KH * KW]`),
put bare `Dim[X]` params BEFORE tensor params so the checker binds them first:

```python
# Good: dims-first, derived expressions in tensors
def add_decomposed_rel_pos[B, QH, QW, KH, KW, HD](
    q_h: Dim[QH], q_w: Dim[QW], k_h: Dim[KH], k_w: Dim[KW],
    attn: Tensor[B, QH * QW, KH * KW],
    q: Tensor[B, QH * QW, HD],
    rel_pos_h: Tensor[RPH, HD],
    rel_pos_w: Tensor[RPW, HD],
) -> Tensor[B, QH * QW, KH * KW]: ...

# Bad: checker can't infer QH from QH*QW in a tensor param
def add_decomposed_rel_pos[B, QH, QW, ...](
    attn: Tensor[B, QH * QW, KH * KW], ...
) -> ...: ...
```

---

## 3. What Should Work (and what to do when it doesn't)

### Shape-preserving and shape-transforming ops are tracked

The type system tracks shapes through nearly all standard PyTorch operations:

- **Identity ops** (stubs with `Self` return): `.float()`, `.contiguous()`,
  `.detach()`, `.clone()`, `.type_as()`, `.to()`
- **Shape-preserving** (stubs with `Tensor[*S] → Tensor[*S]`): `F.relu`,
  `F.gelu`, `F.dropout`, `F.softmax`, `nn.LayerNorm`, `nn.BatchNorm`,
  `torch.sigmoid`, `torch.tanh`
- **Parameterized transforms** (stubs capture constructor args): `nn.Linear`,
  `nn.Conv2d`, `nn.Embedding`, `nn.LSTM`
- **Computed shapes** (DSL functions): `reshape`/`view` (with `-1` inference),
  `flatten`, `permute`, `transpose`, `cat`, `stack`, `expand`, `repeat`,
  `matmul`, `torch.arange`, `torch.zeros`, `torch.outer`, `F.interpolate`
- **Special handlers**: `nn.Sequential` chaining, `.shape` attribute,
  `.size()`, tuple slicing, star unpacking

**If an op appears to lose shapes, it is almost certainly a bug or a missing
stub — not a fundamental limitation.** Check the DSL registry
(`tensor_ops_registry.rs`) and fixture stubs before concluding anything is
untracked.

### When shapes are lost, trace upstream

The op that appears to lose shapes is often not the problem. Trace back to
find where shape info was actually lost:

- **`int` where `Dim` is needed.** If a function takes `size: int` but the
  caller passes a runtime value, shapes enter as unrefined. Fix: change to
  `size: Dim[S]`. Example: `start_pos: int` → `start_pos: Dim[SP] | None`.

- **`list[...]` where `tuple[...]` is needed.** List literals homogenize
  element types. `torch.cat([a, b])` loses per-tensor shapes;
  `torch.cat((a, b))` preserves them.

- **Branch join widening.** Two branches produce different tensor types →
  the checker widens at the join. Fix: restructure to compute independently
  in each branch, or use Optional narrowing.

  ```python
  # Bad: branch join widens keys/values
  if cached:
      keys = cache[:b, :sp+t]   # Tensor[B, SP+T, ...]
  else:
      keys = xk                  # Tensor[B, T, ...]
  # keys is now Tensor | Tensor[...] — widened

  # Good: compute output in each branch independently
  if cached:
      keys = cache[:b, :sp+t]
      output = matmul(softmax(xq @ keys.T), values)
  else:
      output = matmul(softmax(xq @ xk.T), xv)
  # output is Tensor[B, NHead, T, HeadDim] in both branches
  ```

- **Inlined expressions lose shapes.** `f(g(x))` sometimes loses shapes
  that `y = g(x); f(y)` preserves. If you see unexpected bare `Tensor`
  from a composed call, break it into separate assignments.

- **`super()` on generic base classes.** `super().method()` may not
  propagate type params. If the base class is typed, the inherited method
  should already have the right return type — avoid unnecessary overrides.

### Genuinely unknowable shapes are rare

Most things that look data-dependent aren't:

- Boolean masks (`x != 0`) **preserve** shape — the mask has the same shape
  as the input. Operations on it (`.float()`, `*`, `torch.sum`) are tracked.
- Autoregressive loops have typed **elements** (`Tensor[B, 80]`) even if the
  list length is unknown. `torch.stack(mel_outputs, dim=2)` tracks.
- Window partition counts (`B * (H // WS) * (W // WS)`) are **computable**
  from the spatial dims and window size.
- `Embedding.weight` **is typed** (`Tensor[V, D]`). If shapes are lost
  downstream, the blocker is elsewhere.

Genuinely unknowable shapes (bare `Tensor` with comment):
- Data-dependent token counts: conditional `torch.cat` accumulation where the
  number of tokens depends on which prompts are provided.
- Stop-token-controlled sequence length where the LENGTH itself is unknown
  (but individual elements at each step are typed).

### Annotation fallback vs `type: ignore`

- **Annotation fallback**: the checker can't produce the type, but the RHS is
  compatible (e.g., unrefined → typed). No error, no `type: ignore` needed.
  ```python
  dense: Tensor[B, D, ES, ES] = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, h, w)
  ```

- **`type: ignore`**: the checker produces a WRONG type (algebraic gap). Last
  resort. Always include a comment explaining the specific gap.
  ```python
  return out  # type: ignore[bad-return]  # A1: 4*(S//4) ≠ S
  ```

- **Never use bare `Tensor` when you know the shape.** If you know it's
  `Tensor[B, D, H, W]`, annotate it. Bare `Tensor` is only for genuinely
  unknowable shapes.

---

## 4. What Won't Work (Yet)

### A1: `N * (X // N) = X`

Floor division loses the remainder, so `N * (X // N)` only equals `X` when
`X` is divisible by `N`. The checker can't assume this. Affects:
- BiLSTM output: `2 * (D // 2)` — use `type: ignore`
- Multi-head reassembly: `NHead * (D // NHead)` — use `type: ignore`
- Encoder-decoder round-trip: `4 * (S // 4)` — use `type: ignore`

Note: `(a * b) // b → a` IS simplified (sound for all positive integers).
Only the reverse direction is unsound.

### Default values for Dim params

`Literal[0]` is not assignable to `Dim[SP]`. Use `Optional` instead:

```python
# Won't work:
def forward[SP](self, start_pos: Dim[SP] = 0): ...

# Works:
def forward[SP](self, start_pos: Dim[SP] | None = None): ...
```

### ModuleList erases type params

When blocks with different type param values are stored in a list, the list
type must use `Any` for varying params. After iterating, re-annotate:

```python
layers: list[ViTBlock[D, Any, Any, Any, Any]] = [...]
for blk in self.blocks:
    h = blk(h)
# Re-annotate after loop — ModuleList iteration erases type params
h_out: Tensor[B, PS, PS, D] = h  # type: ignore[bad-assignment]
```

### `setattr`/`getattr` with dynamic strings

The checker can't resolve attribute names computed at runtime:
```python
model: nn.Sequential = getattr(self, "layer" + str(i))  # type: ignore[assignment]
```

---

## 5. Generics: Batch and Sequence Dimensions

### Method-level type parameters

Most forward methods have at least one dynamic dimension — a dimension that can
be different for different calls at runtime, such as batch size. Declare it as a
type parameter on the method:

```python
def forward[B](self, x: Tensor[B, 3, 64, 64]) -> Tensor[B, 10]:
    ...
```

`B` is bound when the method is called. If the caller passes a `Tensor[32, 3, 64, 64]`,
then `B = 32` and the return type is `Tensor[32, 10]`.

### Multiple dynamic dimensions

When more than one dimension varies across calls, add more type parameters:

```python
def forward[B, T](self, x: Tensor[B, T, 512]) -> Tensor[B, T, 512]:
    ...
```

---

## 6. Class-Level Type Parameters

### Connecting constructor args to forward signatures

When a module's channel count is chosen at construction time and used in the
forward signature, make it a class-level type parameter and accept the
corresponding `Dim[...]` in `__init__`:

```python
class DoubleConv[InC, OutC](nn.Module):
    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        ...

    def forward[B, H, W](self, x: Tensor[B, InC, H, W]) -> Tensor[B, OutC, H, W]:
        ...
```

`Dim[InC]` is the key construct from `torch_shapes`: it connects a runtime
integer value to a type-level symbol. When someone writes `DoubleConv(3, 64)`,
the type checker sees `c_in: Dim[InC]` receiving `3`, binds `InC = 3`, and
infers the module type as `DoubleConv[3, 64]`. From there, the forward
signature resolves to `Tensor[B, 3, H, W] -> Tensor[B, 64, H, W]`.

---

## Algebraic Expressions and Shape Transforms

### Arithmetic in annotations

Annotations can contain arithmetic on type parameters:

```python
Tensor[B, 2 * C, H // 2, W // 2]     # downsample
Tensor[B, C // 2, H * 2, W * 2]       # upsample
Tensor[B, InC + GR, H, W]             # dense connection (concat)
Tensor[B, NHead * DK]                 # multi-head reshape
Tensor[B, T, 3 * NEmbedding]          # Q/K/V projection
Tensor[B, C * 2 ** I, H // 2 ** I]    # exponential scaling
```

The type checker automatically simplifies expressions: `2 * C // 2` cancels
to `C`, `(H - 1) * 2 + 2` simplifies to `H * 2`, and floor-div literal
extraction turns `(X + 4*K) // 4` into `K + X // 4`.

### Shape transforms from operations

Many PyTorch operations produce output shapes computed from their inputs
and parameters. The type checker tracks these automatically:

- **Spatial convolutions** (Conv1d/2d/3d, ConvTranspose): output spatial dims
  are computed from the standard formulas using kernel size, stride, padding,
  and dilation. For example, `Conv2d(kernel_size=3, padding=1, stride=1)`
  preserves spatial dimensions, while `Conv2d(kernel_size=3, stride=2)` halves
  them (approximately).
- **Pooling** (MaxPool, AvgPool): same formula as Conv. `MaxPool2d(2)` gives
  `H // 2` for even `H`.
- **Linear**: `nn.Linear(in, out)` maps last dim from `in` to `out`.
- **Reshape/view**: `x.view(B, C, -1)` infers the `-1` dim from total element
  count. Multiple `-1`s are rejected.
- **Concatenation**: `torch.cat([a, b], dim=1)` produces a sum dim (`C1 + C2`).
- **Slicing**: `x[:, 3:7]` computes `stop - start = 4`. Negative indices and
  symbolic bounds are supported.
- **PixelShuffle**: `nn.PixelShuffle(r)` maps `(B, C*r², H, W)` to
  `(B, C, H*r, W*r)`.
- **Flatten**: `nn.Flatten(1)` collapses dims 1..end into a product.
- **LSTM**: output shape depends on `hidden_size` and `bidirectional`.
  The DSL computes `nd = 2 if bidirectional else 1` and tracks output
  as `Tensor[B, T, hidden_size * nd]`.
- **Distributions**: `Normal`, `TransformedDistribution` are generic over
  `*EventShape`. `rsample()` and `log_prob()` preserve event shape.

When an operation's output shape can't be statically determined, the result
is an unrefined `Tensor`. But this is rare — see
[Section 3](#3-what-should-work) for how to diagnose and fix.

---

## 7. Linear Pipeline

Modules called in sequence with known shapes. The simplest architecture pattern.

### Example: MLP actor (soft_actor_critic.py)

```python
class BaselineActor[S, A](nn.Module):
    def __init__(self, state_size: Dim[S], action_size: Dim[A]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.out = nn.Linear(400, action_size)

    def forward[B](self, state: Tensor[B, S]) -> Tensor[B, A]:
        h1 = F.relu(self.fc1(state))
        assert_type(h1, Tensor[B, 400])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, 400])
        act = torch.tanh(self.out(h2))
        assert_type(act, Tensor[B, A])
        return act
```

### Example: FCN renderer (learning_to_paint.py)

A deeper pipeline with reshape and `PixelShuffle`:

```python
class FCN(nn.Module):
    def forward[B](self, x: Tensor[B, 10]) -> Tensor[B, 128, 128]:
        # MLP: 10 -> 512 -> 1024 -> 2048 -> 4096
        h1 = F.relu(self.fc1(x))
        assert_type(h1, Tensor[B, 512])
        ...
        h4 = F.relu(self.fc4(h3))
        assert_type(h4, Tensor[B, 4096])

        # Reshape to spatial
        spatial = h4.view(x.size(0), 16, 16, 16)
        assert_type(spatial, Tensor[B, 16, 16, 16])

        # Conv + PixelShuffle(2) stages
        s1 = self.pixel_shuffle(self.conv2(F.relu(self.conv1(spatial))))
        assert_type(s1, Tensor[B, 8, 32, 32])
        ...
```

**Key insight**: `assert_type` after key steps serves as both documentation
and a regression guard. If you refactor a layer and shapes change, the type
checker catches it immediately.

---

## 8. Homogeneous Layer Stacking (Transformers)

When every layer has the same type, use `nn.ModuleList[LayerType]` and iterate:

```python
class Encoder[NHead, DK, DInner](nn.Module):
    def __init__(self, n_head: Dim[NHead], d_k: Dim[DK],
                 d_inner: Dim[DInner], n_layers: int = 6) -> None:
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(n_head, d_k, d_inner) for _ in range(n_layers)]
        )

    def forward[B, T](
        self, src_seq: Tensor[B, T, NHead * DK]
    ) -> Tensor[B, T, NHead * DK]:
        enc_output = src_seq
        for layer in self.layer_stack:
            enc_output, _attn = layer(enc_output)
            assert_type(enc_output, Tensor[B, T, NHead * DK])
        return enc_output
```

Because each `EncoderLayer` is `Tensor[B, T, D] -> Tensor[B, T, D]`, the
loop preserves the shape invariant and the type checker is satisfied.

### Shape-preserving activations

Many architectures accept an activation function as a parameter (ReLU, GELU,
etc.). Since each activation's forward is `Tensor[*S] -> Tensor[*S]`, define a
type alias:

```python
ShapePreservingActivation = (
    type[nn.ReLU] | type[nn.GELU] | type[nn.SiLU] | type[nn.Tanh]
)

class ResNetBlock[C](nn.Module):
    def __init__(self, c: Dim[C], act_fn: ShapePreservingActivation) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            act_fn(),  # instantiate the activation class
            ...
        )
```

This works because `nn.ReLU`, `nn.GELU`, etc. each have a shape-preserving
forward signature, and `nn.Sequential` chains them correctly.

---

## 9. Encoder-Decoder with Skip Connections

Encoder-decoder architectures (UNet, Demucs, Super SloMo) encode the input to
a bottleneck and then decode back, with skip connections between corresponding
encoder and decoder layers.

### Shape-preserving recursion

The key insight is that each encode-recurse-decode cycle preserves the shape:
encode changes `(B, C, H, W)` to `(B, 2C, H', W')`, recursion preserves that
shape, and decode restores `(B, C, H, W)` via the skip connection. This gives
a recursive signature:

```python
class UNet[NChannels, NClasses](nn.Module):
    def _encode[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: int
    ) -> Tensor[B, 2 * C, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        idx = len(self.downs) - depth
        down: Down[C, 2 * C] = self.downs[idx]
        return down(x)

    def _decode[B, C, H, W](
        self,
        skip: Tensor[B, C, H, W],
        deep: Tensor[B, 2 * C, (H - 2) // 2 + 1, (W - 2) // 2 + 1],
        depth: int,
    ) -> Tensor[B, C, H, W]:
        idx = len(self.ups) - depth
        up: Up[2 * C, C] = self.ups[idx]
        return up(deep, skip)

    def recurse[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C, H, W]:
        if depth == 0:
            return x
        skip = x
        encoded = self._encode(x, depth)
        middle = self.recurse(encoded, depth - 1)
        decoded = self._decode(skip, middle, depth)
        return decoded
```

### `list[Stage[Any]]` + narrowing annotation

Python has no way to express "element `i` of this list has type
`Stage[C * 2**i]`". The workaround is:

1. Declare the list with `Any`: `list[Down[Any, Any]]`
2. Narrow at the access site:

```python
down: Down[C, 2 * C] = self.downs[idx]
```

The `Any` erases element-level type info, and the annotation re-introduces it
for each use. This is the standard pattern for heterogeneous `ModuleList`s.

### Known algebraic gap

Some algebraic equivalences can't be automatically proven. For example,
`((H - 2) // 2 + 1) * 2` does not simplify back to `H`. When you hit this,
use `type: ignore` with a comment explaining the gap:

```python
return up(deep, skip)  # type: ignore[bad-argument-type]  # ((H-2)//2+1)*2 = H
```

Keep these to an absolute minimum and document each one.

---

## 10. Recursive Chains with Exponential Shapes

When each stage doubles or halves a dimension, the result after `I` stages
involves `2**I`. Use `@overload` to separate the base case from the recursive
case:

```python
class Generator(nn.Module):
    def _apply_stage[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: int
    ) -> Tensor[B, C // 2, (H - 1) * 2 + 2, (W - 1) * 2 + 2]:
        idx = len(self.up_stages) - depth
        stage: GenUpStage[C] = self.up_stages[idx]
        return stage(x)

    @overload
    def _chain[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[1]
    ) -> Tensor[B, C // 2, H * 2, W * 2]: ...

    @overload
    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C // 2 ** I, H * 2 ** I, W * 2 ** I]: ...

    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> (Tensor[B, C // 2, H * 2, W * 2]
         | Tensor[B, C // 2 ** I, H * 2 ** I, W * 2 ** I]):
        y = self._apply_stage(x, depth)
        if depth == 1:
            return y
        return self._chain(y, depth - 1)
```

The base-case overload (`depth: Dim[1]`) handles the single-stage case where
the formula simplifies concretely. The recursive overload uses `2**I` to
express the exponential relationship.

### `_apply_stage` + `_chain` pattern

This two-method pattern appears in DCGAN (generator + discriminator), ResNet,
and DenseNet:

- **`_apply_stage`**: applies a single stage from the `ModuleList`, using the
  narrowing annotation to type the list element.
- **`_chain`**: recursively applies `_apply_stage` with overloaded return types.

The caller invokes `_chain` with a concrete depth:

```python
def forward[B](self, input: Tensor[B, 100, 1, 1]) -> Tensor[B, 3, 64, 64]:
    h0 = F.relu(self.project_bn(self.project(input)))
    assert_type(h0, Tensor[B, 512, 4, 4])
    h1 = self._chain(h0, 3)  # 512->64, 4->32
    assert_type(h1, Tensor[B, 64, 32, 32])
    return torch.tanh(self.output(h1))
```

---

## 11. Config Classes

### `@dataclass` with type parameters

For models with many hyperparameters, use a generic `@dataclass`:

```python
@dataclass
class GPTConfig[VocabSize, BlockSize, NEmbedding, NHead, NLayer]:
    block_size: Dim[BlockSize]
    vocab_size: Dim[VocabSize]
    n_layer: Dim[NLayer]
    n_head: Dim[NHead]
    n_embd: Dim[NEmbedding]
    dropout: float = 0.0
    bias: bool = True
```

Modules extract shape parameters from the config:

```python
class MLP[NEmbedding](nn.Module):
    def __init__(self, config: GPTConfig[Any, Any, NEmbedding, Any, Any]):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
```

Use `Any` for config type parameters that the module doesn't care about.

### `Final` class attributes for constants

When a model has fixed hyperparameters, `Final` class attributes let `Literal`
arithmetic work at the type level:

```python
class DCGAN:
    nc: Final = 3
    nz: Final = 100
    ngf: Final = 64
    ndf: Final = 64
```

The type checker resolves `DCGAN.ngf * 8` to `Literal[512]`, so you can write:

```python
self.project = nn.ConvTranspose2d(DCGAN.nz, DCGAN.ngf * 8, 4, 1, 0)
# Type checker infers: ConvTranspose2d[100, 512, ...]
```

---

## 12. Techniques Reference

### Variable renaming for shape changes

When a variable's shape changes, use a new name instead of reassigning:

```python
# Good: different names for different shapes
h1 = F.relu(self.fc1(x))          # Tensor[B, 512]
h2 = F.relu(self.fc2(h1))         # Tensor[B, 1024]

# Good: numbered stages
x1 = self.inc(x)                  # Tensor[B, 64, 256, 256]
x2 = self.down1(x1)               # Tensor[B, 128, 128, 128]

# Avoid: reassignment that changes shape
x = F.relu(self.fc1(x))           # was Tensor[B, 10], now Tensor[B, 512]
x = F.relu(self.fc2(x))           # was Tensor[B, 512], now Tensor[B, 1024]
```

Reassignment is fine when the shape doesn't change (e.g., residual connections,
dropout, layer norm):

```python
x = x + self.attn(self.ln_1(x))   # same shape: Tensor[B, T, D]
x = x + self.mlp(self.ln_2(x))    # same shape: Tensor[B, T, D]
```

### Concatenation

`torch.cat` along a dimension produces a sum type:

```python
sa = torch.cat((state, action), dim=1)  # Tensor[B, S + A]
skip_cat = torch.cat([x2, x1_up], dim=1)  # Tensor[B, C1 + C2, H, W]
```

### `@overload` for type narrowing

Use `@overload` when the return type depends on a parameter value, such as
recursive depth. The `_chain` methods in dcgan and resnet demonstrate this:

```python
@overload
def _chain[B, C, H, W](
    self, x: Tensor[B, C, H, W], depth: Dim[1]
) -> Tensor[B, C // 2, H * 2, W * 2]: ...

@overload
def _chain[I, B, C, H, W](
    self, x: Tensor[B, C, H, W], depth: Dim[I]
) -> Tensor[B, C // 2 ** I, H * 2 ** I, W * 2 ** I]: ...
```

### `type: ignore` with comment

Use sparingly, only for algebraic equivalences the checker can't prove.
Always add a comment explaining what equivalence is assumed:

```python
# Known gap: (HeadDim // 2) * 2 = HeadDim
rotated = apply_rotary(q)  # type: ignore[bad-argument-type]
```

### No indexed list types

Python's type system has no way to say "element `i` of this list has type
`Stage[C * 2**i]`". This is the fundamental reason heterogeneous `ModuleList`s
need the narrowing pattern:

```python
# Declare with Any to erase element-level types
stages: list[GenUpStage[Any]] = [
    GenUpStage(512),   # GenUpStage[512]
    GenUpStage(256),   # GenUpStage[256]
    GenUpStage(128),   # GenUpStage[128]
]
self.up_stages = nn.ModuleList(stages)

# Narrow at each access site
stage: GenUpStage[C] = self.up_stages[idx]
```

### Tracing shape loss

When a result appears unrefined, don't annotate it as bare `Tensor` and move
on. Instead, trace upstream to find where shapes were actually lost. See
[Section 3](#3-what-should-work) for the full diagnostic approach. Common
fixes:

- Change `int` to `Dim[X]` so shapes enter the function typed
- Use `tuple(...)` instead of `list[...]` for `torch.cat` arguments
- Break inlined expressions into separate assignments
- Fix the stub if an op returns bare `Tensor` when it shouldn't

---

## 13. Smoke Tests

Every model file ends with `test_*` functions that exercise the model at
concrete dimensions:

```python
def test_baseline_actor():
    """Test simple MLP actor: state(24) -> action(4)."""
    actor = BaselineActor(24, 4)
    state: Tensor[8, 24] = torch.randn(8, 24)
    act = actor(state)
    assert_type(act, Tensor[8, 4])
```

### Guidelines

- **Concrete dims in tests**: use `Tensor[4, 64, 32, 32]`, not generic dims.
  This lets the type checker verify the full shape calculation.
- **Test building blocks individually**: verify each module's shape transform
  before testing the full model.
- **Test end-to-end pipelines**: verify the composed model produces the
  expected output shape.
- **Multiple configurations**: if the model supports different settings
  (e.g., bilinear vs non-bilinear UNet), test each one.

```python
def test_unet():
    """End-to-end: non-bilinear UNet for 2-class segmentation."""
    model = UNet(3, 2)
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert_type(out, Tensor[1, 2, 256, 256])

def test_gan_pipeline():
    """End-to-end: generate fake images, then discriminate them."""
    netG = Generator()
    netD = Discriminator()
    noise: Tensor[16, 100, 1, 1] = torch.randn(16, 100, 1, 1)
    fake = netG(noise)
    assert_type(fake, Tensor[16, 3, 64, 64])
    verdict = netD(fake)
    assert_type(verdict, Tensor[16, 1, 1, 1])
```

---

## Model Index

| Pattern | Models | Key concept |
|---------|--------|-------------|
| Linear Pipeline | [learning_to_paint](models/learning_to_paint.py), [soft_actor_critic](models/soft_actor_critic.py), [deeprecommender](models/deeprecommender.py) | Sequential layers, `assert_type` checkpoints |
| Homogeneous Stacking | [nanogpt](models/nanogpt.py), [gptfast](models/gptfast.py), [speech_transformer](models/speech_transformer.py), [llama](models/llama.py) | `ModuleList` iteration, shape-preserving loops |
| Encoder-Decoder Skip | [unet](models/unet.py), [super_slomo](models/super_slomo.py), [demucs](models/demucs.py), [stargan](models/stargan.py) | Recursive `encode`-`decode`, generic spatial dim `S` |
| Recursive Exponential | [dcgan](models/dcgan.py), [resnet](models/resnet.py), [densenet](models/densenet.py) | `@overload` base/recursive, `2**I` expressions |
| Config Classes | [nanogpt](models/nanogpt.py), [gptfast](models/gptfast.py), [dcgan](models/dcgan.py), [llama](models/llama.py) | `@dataclass` type params, `Final` constants |
| ShapePreservingActivation | [resnet](models/resnet.py) | Union of activation types as callable |
| Multi-Head Attention | [llama](models/llama.py), [sam](models/sam.py) | Reshape+transpose multi-head, `D // NHead`, RoPE |
| KV Cache | [llama](models/llama.py) | Optional `start_pos`, typed cache, branch-per-path |
| Windowed Attention | [sam](models/sam.py) | Window partition/unpartition with `Dim[WS]`, generic `H, W` on attention |
| Typed Distributions | [drq](models/drq.py) | `Distribution[*EventShape]`, `SquashedNormal` |
| Variadic Batch | [tacotron2](models/tacotron2.py) | `forward[*Bs]` for any-batch-shape support |
| Autoregressive Loop | [tacotron2](models/tacotron2.py) | `list[Tensor[B, 80]]` + `torch.stack`, typed elements |
| Dims-First Params | [sam](models/sam.py) | Bind bare `Dim[X]` before derived `Tensor[..., X*Y, ...]` |
| Conv Chain Formulas | [sam](models/sam.py), [background_matting](models/background_matting.py), [stargan](models/stargan.py) | `4*ES → 2*ES → ES`, `(S-16)//16+1` through Conv2d/ConvTranspose2d |
