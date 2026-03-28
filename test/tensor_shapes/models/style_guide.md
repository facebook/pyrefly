# Tensor Shape Annotation Style Guide

A practical guide to adding tensor shape annotations to PyTorch models using
pyrefly's type system. All patterns are drawn from
[12 ported TorchBenchmark models](models/).

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Generics: Batch and Sequence Dimensions](#2-generics-batch-and-sequence-dimensions)
3. [Class-Level Type Parameters](#3-class-level-type-parameters)
4. [Algebraic Expressions and Shape Transforms](#4-algebraic-expressions-and-shape-transforms)
5. [Linear Pipeline](#5-linear-pipeline)
6. [Homogeneous Layer Stacking](#6-homogeneous-layer-stacking-transformers)
7. [Encoder-Decoder with Skip Connections](#7-encoder-decoder-with-skip-connections)
8. [Recursive Chains with Exponential Shapes](#8-recursive-chains-with-exponential-shapes)
9. [Config Classes](#9-config-classes)
10. [Techniques Reference](#10-techniques-reference)
11. [Smoke Tests](#11-smoke-tests)

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

## 2. Generics: Batch and Sequence Dimensions

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

## 3. Class-Level Type Parameters

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

## 4. Algebraic Expressions and Shape Transforms

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
- **LSTM**: output shape depends on `hidden_size` and `num_directions`.

When an operation's output shape can't be statically determined (e.g.,
tensor-as-index, complex broadcasting), the result is an unrefined `Tensor`
without shape parameters.

---

## 5. Linear Pipeline

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

## 6. Homogeneous Layer Stacking (Transformers)

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

## 7. Encoder-Decoder with Skip Connections

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

## 8. Recursive Chains with Exponential Shapes

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

## 9. Config Classes

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

## 10. Techniques Reference

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

### Shapeless fallback

Some operations return unrefined `Tensor` (without shape parameters). Known
cases:

- **Symbolic slice on concrete buffer**: `self.pe[:, :length]` where `pe` is a
  precomputed buffer with concrete shape and `length` is symbolic
  (speech_transformer `PositionalEncoding`)
- **Tensor-as-index**: `t[ind]` where `ind` is a tensor used as an index
  (super_slomo coefficient interpolation)
- **Complex broadcasting**: operations combining tensors with shapes that
  require runtime broadcasting resolution (demucs `upsample`)

When you encounter a shapeless result, annotate the return type as `Tensor`
and add a comment:

```python
def forward[B, T](self, input: Tensor[B, T, DModel]) -> Tensor:
    """Returns shapeless Tensor: symbolic slice on concrete buffer."""
    length = input.size(1)
    return self.pe[:, :length]
```

---

## 11. Smoke Tests

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
| Linear Pipeline | [learning_to_paint](models/learning_to_paint.py), [soft_actor_critic](models/soft_actor_critic.py) | Sequential layers, `assert_type` checkpoints |
| Homogeneous Stacking | [nanogpt](models/nanogpt.py), [gptfast](models/gptfast.py), [speech_transformer](models/speech_transformer.py) | `ModuleList` iteration, shape-preserving loops |
| Encoder-Decoder Skip | [unet](models/unet.py), [super_slomo](models/super_slomo.py), [demucs](models/demucs.py) | Recursive `encode`-`decode`, `list[Stage[Any]]` narrowing |
| Recursive Exponential | [dcgan](models/dcgan.py), [resnet](models/resnet.py), [densenet](models/densenet.py) | `@overload` base/recursive, `2**I` expressions |
| Config Classes | [nanogpt](models/nanogpt.py), [gptfast](models/gptfast.py), [dcgan](models/dcgan.py) | `@dataclass` type params, `Final` constants |
| ShapePreservingActivation | [resnet](models/resnet.py) | Union of activation types as callable |
