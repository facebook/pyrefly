---
name: port-model
description: >
  Port a PyTorch model to use pyrefly's tensor shape type system (Tensor[B, C, H, W],
  Dim[T]). Use this skill whenever the user wants to add shape annotations
  to a PyTorch model, type a model with tensor dimensions, port a model to use shape
  tracking, or annotate model forward methods with tensor shapes. Also use when the
  user mentions tensor shape ports, Dim types for PyTorch, or pyrefly shape checking
  on a model file. Invoke BEFORE starting any model port — the skill's gated workflow
  prevents common failure modes.
---

You are porting a PyTorch model to use pyrefly's tensor shape type system.

**This skill requires detailed step-by-step output.** Each step produces
a mandatory output artifact (a table, a checklist, a count). These
artifacts are the deliverables — do not compress, summarize, or skip them.
The next step's input depends on the previous step's output. If you don't
have the artifact, you cannot proceed.

# Pre-flight

**Create tasks** for each gate and the module loop/verification phases
that follow. Update them as you complete each stage — this gives
visibility during long-running ports.

Complete these gates before writing any code.

## Gate 0: Understand the system

Read `shape_tracking_capabilities.md` (this skill dir). It explains the
three shape-tracking mechanisms (fixture stubs, DSL functions, special
handlers) and how to check each one. You need this context to make the
Gate 1 audit meaningful — knowing whether an op exists in a stub is not
the same as knowing whether its shapes are tracked.

**Do NOT read `style_guide.md` yet.** It is comparison material for the
verification phase. Reading it now biases you toward known patterns
before you have empirically probed this model's shapes.

## Gate 1: Audit ops

List every `nn.Module` subclass and `torch`/`F.` function called in the
model. Check each against the fixture stubs (`.pyi` files in
`test/tensor_shapes/fixtures/torch/`) and the DSL registry
(`tensor_ops_registry.rs` — search for the qualified name). Add any missing
stubs or DSL functions BEFORE starting the port. This step should take
minutes — you are scanning two known locations.

**Do NOT use code search agents or web search for this step.** The two
locations above are exhaustive. For each op in your list, check whether it
appears in the stubs/registry — a targeted grep or scan, not a full read
of every file. You need to confirm presence and spot missing attributes
(e.g., `bias` on `Conv2d`), not memorize every signature.

**Paste your audit table** in your response before proceeding to Gate 2:

```
## Gate 1: Ops audit
| Op | Stub location | DSL fn (or "not registered") | Status |
|----|---------------|------------------------------|--------|
| nn.Conv2d | nn/__init__.pyi:491 — generic [InC,OutC,K,S,P,D] | not registered (stub generic) | ✓ tracked |
| F.adaptive_avg_pool2d | functional.pyi:315 — bare return | adaptive_avg_pool2d_ir | ✓ tracked (DSL) |
| ...
```

Filling the "DSL fn" column requires searching `tensor_ops_registry.rs`
for the qualified name (e.g., `"torch.nn.functional.adaptive_avg_pool2d"`).
Write "not registered" only after confirming the name is absent — do not
leave this column blank or write "check registry".

If any op is missing or returns bare, fix stubs/DSL BEFORE proceeding.

## Gate 2: Inventory the original

Write the inventory as a comment block at the top of the port file, using
this exact format:

```python
# ## Inventory
# - [ ] ClassName.__init__ — Dims: param1, param2; int: param3
# - [ ] ClassName.forward
# - [ ] function_name — utility, no tensors
# ...
```

**Every class, function, and method in the original file must appear, and
every item must be ported.** Do not skip or exclude anything. If a class
depends on a library without fixture stubs, add the missing stubs or DSL
in Gate 1 rather than omitting the class. If adding full stubs is
impractical (e.g., the library is large and only one op is needed), add
a minimal stub for the specific ops used.

For each class, list constructor parameters and whether each is Dim or
int — this feeds Step 1 of the module loop.

**Check off items as you port them** (`[ ]` → `[x]`). Do not proceed to
verification with unchecked items.

# Transition to module loop

You have completed pre-flight. You have NOT written any model code yet.

**Your analysis so far is a hypothesis.** The module loop tests it
empirically. If you catch yourself planning multiple modules' forwards
in your head, STOP — you are substituting reasoning for testing, and
reasoning is less reliable.

Write the file with imports, the inventory comment, and utility functions
(no tensor shapes). Then start Step 1 for the FIRST module only.

Read `porting_principles.md` (this skill dir) for the mindset: why we port,
priority order, and fixture philosophy.

# Module loop

**Repeat the following for each module in dependency order.** Each module's
typing may inform the next — e.g., discovering that a submodule tracks
shapes internally changes how the parent handles its loop.

**ONE MODULE AT A TIME.** Complete Steps 1–6 for module A, paste the
Step 6 checklist, THEN start Step 1 for module B. If you find yourself
typing two modules' constructors before running the checker, you have
already entered the primary failure mode: writing the entire file and
validating at the end leads to over-use of typed interfaces and under-use
of `assert_type`.

## Step 1: Inventory parameters

List every constructor parameter. For each, decide:
- **Dim**: the value determines a tensor dimension (flows to `nn.Linear`,
  `nn.Conv2d`, tensor creation, or any typed function that uses the value
  as a shape dimension).
- **int**: iteration count (`n_layers`, `n_res_block`) or boolean-like flag.

If in doubt, make it `Dim`. The cost is one more type param; the cost of
`int` is permanent shape loss in everything downstream.

**Critical rules:**
- Every `int` that flows to a sub-module constructor (`nn.Linear(dim, ...)`)
  MUST be `Dim`. No exceptions.
- Never cast Dim to int (`int(dim)`, `self.x = int(dim)`) — `Dim` is a
  subtype of `int`, so the cast only kills tracking. Exception:
  `bool`/`float` conversion is necessary, but `int * Dim` produces
  Unknown — check whether it reaches tensor shapes before fixing. Don't
  replace with if/else branching (union of concrete expressions is worse
  than one Unknown).
- Derived dims use expressions (`D // NHead`, `4 * ES`), not independent
  type params. Only independent degrees of freedom get type params.
- **Dimensions from `list[int]`.** `list[int]` element access (e.g.,
  `hidden_units[-1]`) erases the concrete value to `int`. Add an explicit
  `Dim` field to the config or constructor for that value.
- Use `nn.Buffer` and `nn.Parameter`, not `register_buffer`/
  `register_parameter`.
- **Bridge dims.** When part of the model is untracked (e.g., features
  built via `nn.Sequential(*list)`), look for dimensions that connect
  the untracked section to tracked downstream modules. For example, if
  features output feeds a Linear classifier, the Linear's `in_features`
  is a bridge dim — making it a class type param enables annotation
  fallback to recover a shaped type (e.g., `Tensor[B, LastC]`) that
  then flows naturally through downstream ops. Without it, annotation
  fallback can only recover bare `Tensor` or batch-only shapes.

  ```python
  class Model[NC, LC](nn.Module):
      def __init__(self, num_classes: Dim[NC] = 1000,
                   last_channel: Dim[LC] = 1280):
          ...
          self.classifier = nn.Linear(last_channel, num_classes)
  ```

  Here `LC` bridges the untracked feature extractor to the typed
  classifier, recovering `Tensor[B, NC]` at the output.
- **`Dim[X] | None` for optional dimensions.** When a parameter is
  `Optional[int]` but flows to tensor shapes when present, type it as
  `Dim[X] | None`, not `Optional[int]`. Example:
  `rank_k: Optional[int]` → `rank_k: Dim[RK] | None`. In the forward
  method, narrow with `if rank_k is not None:` — the checker then
  treats `rank_k` as `Dim[RK]` inside the branch. Leaving it as
  `Optional[int]` permanently loses tracking in every downstream op.
- **Parameterized config dataclasses.** When multiple modules consume
  dimensions from the same `@dataclass` config, note it — Step 2
  shows how to parameterize the config so dims propagate across
  module boundaries.
- **Lazy-initialized buffer attributes.** An attribute's type is fixed
  at the declaration, not at later assignments. Declaring
  `self.x: Tensor | None = None` and assigning a real tensor in a
  setup hook (e.g., `setup_caches`) loses the shape forever — every
  read site sees `Tensor | None`, and the best you get after a None
  check is bare `Tensor`. If the field is always assigned before
  first use, initialize it eagerly in `__init__` with the real shape:

  ```python
  # Bad: causal_mask is Tensor | None everywhere; shape is lost
  class Attention(nn.Module):
      def __init__(self, ...):
          self.causal_mask: Tensor | None = None
      def setup_caches(self, max_seq_len: int):
          self.causal_mask = torch.tril(
              torch.ones(max_seq_len, max_seq_len)
          )

  # Good: causal_mask is Tensor[MS, MS] from declaration
  class Attention[MS](nn.Module):
      def __init__(self, max_seq_len: Dim[MS], ...):
          self.causal_mask: Tensor[MS, MS] = torch.zeros(
              max_seq_len, max_seq_len
          )
  ```

  Reserve `Tensor | None` for fields that may *genuinely* never be
  set. If late init is unavoidable because the shape depends on a
  runtime decision, accept that bare `Tensor` post-narrow is the
  right answer and document it as a Step 4 receipt.

## Step 2: Type the constructor

Write `__init__` with the `Dim` params from Step 1. Construct sub-modules
using those Dims — they get typed automatically.

**Default values for Dim params:** `Literal[0]` is not assignable to
`Dim[X]` as a default. Use PEP 696 type-parameter defaults instead:

```python
# Won't work — Literal[1000] not assignable to Dim[NC]:
def __init__(self, num_classes: Dim[NC] = 1000): ...

# Works — NC defaults to 1000 at the type level:
class Model[NC = 1000](nn.Module):
    def __init__(self, num_classes: Dim[NC] = 1000): ...
```

**Constructor patterns that break shape tracking:**
- **`nn.Sequential(*list_var)`** erases module types — the Sequential
  returns bare `Tensor`. Only `nn.Sequential(M1(), M2(), M3())` with
  direct arguments is tracked. Extract shape-changing modules (Linear,
  Conv2d) as individual attributes and chain in `forward`.
- **Factory functions returning `nn.Sequential`** erase all type
  parameters at the function boundary. Use a class with a typed
  `forward` method instead.
- **`getattr(nn, name)()`** returns `Any`. Replace with a union of
  typed `nn.Module` subclass types.
- **Method-level type params on class fields.** If a method creates
  shaped tensors assigned to `self.field`, the field can't carry the
  method's type params — it reverts to bare `Tensor`. Move creation
  to `__init__` so type params become class-level.

**Parameterized config dataclasses.** When a `@dataclass` holds
dimension hyperparameters consumed by multiple modules, make it
generic so dims propagate through constructors:

```python
@dataclass
class Config[D, NHead, VocabSize]:
    dim: Dim[D]
    n_head: Dim[NHead]
    vocab_size: Dim[VocabSize]
    dropout: float = 0.0
```

Modules extract only the params they need using `Any` for the rest:

```python
class MLP[D](nn.Module):
    def __init__(self, config: Config[D, Any, Any]):
        super().__init__()
        self.fc = nn.Linear(config.dim, 4 * config.dim)
```

Without this, each module must independently accept and thread
every dim through its constructor — error-prone and verbose.

If the original config had default values (e.g., `dim: int = 768`),
combine the two patterns above — give the dataclass type params PEP
696 defaults so callers can omit dims:

```python
@dataclass
class Config[D = 768, NHead = 12, VocabSize = 50257]:
    dim: Dim[D] = 768
    n_head: Dim[NHead] = 12
    vocab_size: Dim[VocabSize] = 50257
    dropout: float = 0.0
```

Now `Config()` produces `Config[768, 12, 50257]` and
`Config(dim=1024)` produces `Config[1024, 12, 50257]` — dims
propagate even when callers don't pass every parameter.

**DO NOT write the forward method yet.** The forward signature and
`assert_type` expressions depend on what the checker infers, which you
don't know until Step 3.

Run the checker to verify the constructor compiles. **Paste the checker
output** (0 errors, or the errors you need to fix) before proceeding.

## Step 3: Probe the forward

First, **count the local variables** in the forward method — every
assignment to a name (e.g., `x = ...`, `out = ...`, `result = ...`)
is a local. Write them down.

Then add `reveal_type` on EVERY local variable. Run the checker.

**Paste the results in your response** using this exact format. Step 4
takes this table as input — if you don't have it, you cannot proceed.

```
# reveal_type results for ClassName.forward:
# Locals: N (list them: var1, var2, var3)
# var1 (line N): Tensor[B, C, H, W]  → SHAPED
# var2 (line M): Tensor               → BARE — investigate in Step 4
# var3 (line P): Tensor[B, D]         → SHAPED
```

Verify: does the number of reveal_type entries match the local count?
If not, you missed some — go back and add them.

**If a reveal_type result contradicts your understanding of the op**
(e.g., spatial dims unchanged after a strided conv, or a shaped op
returning bare), write a small isolating test, run the checker, and
confirm the behavior before proceeding. Either your understanding is
wrong (update your mental model) or the checker has a simplification
you should document.

This table is your Step 4 input. Do not write `assert_type` until Step 4
is complete for every BARE entry. The results tell you:
- Shaped type → the checker tracks this op. Write `assert_type` in Step 5.
- Bare `Tensor` → shape lost. Investigate in Step 4 before deciding.

## Step 4: Restructure for tracking

Many patterns that LOOK dynamic have trackable substructure.
Conditional branching over matmul/bmm chains, for example, is fully
trackable if the dimension values are `Dim`-typed.

For EACH bare `Tensor` from Step 3, attempt ALL applicable restructurings
before falling back to typed interface:

□ **`int()` or `round()` wrapping a Dim value?** Remove it. If the
  argument is already int-compatible (e.g., `round()` on an integer
  `expand_ratio`), the wrapper is a no-op that kills tracking.

□ **`nn.Sequential(*list_var)`?** Extract shape-changing modules
  (Linear, Conv2d, etc.) as individual attributes and chain them in
  `forward`. Shape-preserving modules (activations, norms, dropout)
  can remain grouped since their output shape equals their input.
  Note: this applies to `nn.Sequential(*list_variable)` where the
  modules come from a list. `nn.Sequential(M1(), M2(), M3())` with
  direct arguments IS tracked — don't restructure it.

□ **`nn.Sequential` subclass?** The special handler tracks shapes when
  a Sequential is CALLED as an attribute (`self.net(x)`), but NOT when
  forward is inherited from a Sequential base class. Convert subclasses
  to composition: replace `class Foo(nn.Sequential)` /
  `super().__init__(m1, m2, m3)` with `class Foo(nn.Module)` /
  `self.net = nn.Sequential(m1, m2, m3)` and delegate forward to
  `self.net(x)`. This is the minimal change for full shape tracking.

□ **`list[...]` where `tuple[...]` is needed?** `torch.cat([a, b])`
  homogenizes element types. Use `torch.cat((a, b))`. Same for
  `.split([d, k, k])` → `.split((d, k, k))`.

□ **Branch join widening?** If the first iteration changes shape but
  subsequent iterations preserve it, separate the first iteration:
  `x = layers[0](input)` then loop over `layers[1:]`. The dual works
  too: separate the last iteration if only the final output matters.

□ **Loop over `ModuleList` widens tensor type?** Same fix as branch
  join: separate the shape-changing iteration from shape-preserving ones.

□ **Tensor accumulation for `stack`/`cat`?** Type the list with the
  element shape, then annotate the stack/cat result with the full shape
  including the new dimension (the DSL can't infer collection size from
  a dynamic loop).

□ **Inlined expressions?** `f(g(x))` sometimes loses shapes that
  `y = g(x); f(y)` preserves. Break into separate assignments.

□ **Missing stub or DSL registration?** Check the fixture stubs and
  `tensor_ops_registry.rs`. Fix if missing — stubs are your code.

□ **About to claim an op is untracked?** Check the DSL registry,
  fixture stubs, and special handlers first. The system tracks reshape,
  flatten, permute, transpose, cat, stack, matmul, arange, zeros,
  outer, interpolate, einsum, and many more.

After EACH restructuring, re-run `reveal_type` and update your records.

**STOP before using typed interface.** For each bare variable where you
want typed interface, paste this filled-out receipt in your response:

```
## Typed interface receipt [<Module>.<var>]: <variable> in <ClassName.forward>
- int()/round() cast: [removed / not applicable — reason]
- Sequential(*list): [restructured / not applicable — reason]
- list→tuple: [not applicable — reason]
- Branch join: [not applicable — reason]
- Inlined expressions: [split / not applicable — reason]
- Missing stub/DSL: [checked stubs + registry — reason]
- Dim | None reclassification: [reclassified param X / not applicable — reason]
- Bridge dim: [promoted X to class Dim / not applicable — reason]
- Config parameterization: [parameterized Config[...] / not applicable — reason]
Result: still bare after all checks. Using typed interface because ___.
```

If you cannot fill this out, you have not completed Step 4. Go back.

"Restructure" usually means a 2–3 line change: separating an iteration,
removing an `int()` cast, or adding a `Dim` type param. It does NOT mean
rewriting the algorithm. If you find yourself writing significantly
different logic, you've gone too far. Even partial dim tracking (e.g.,
output dim only) is far more useful than none.

**`type: ignore` categories.** Before writing `type: ignore`, identify
which category applies:
- **A1 algebraic gap** (`N * (X // N) ≠ X`): no fix, use `type: ignore`.
- **Conditional equality** (e.g., `Inp == Oup` at runtime but separate
  type params): no fix, use `type: ignore`.
- **Stub or DSL gap**: fix the stub or DSL instead.
- **`return-value` mismatch from untracked sub-section**: don't
  `type: ignore`. The fix is upstream — find the bridge dim
  connecting the untracked section (e.g., `nn.Sequential(*list)`
  features) to the tracked downstream input (e.g., a classifier
  Linear), promote it to a class type param per Step 1's bridge-dim
  rule, then use annotation fallback to recover the shaped return.
- **Branch join**: try restructuring first.

**Bare `Tensor` where you know the shape?** Use `assert_type` to verify
inference, not annotation fallback. Annotation fallback silently accepts
bare `Tensor` — it doesn't prove tracking works. If the checker can't
infer the shape, trace upstream to find where shapes were actually lost.

## Step 5: Write forward and assert_type

**Annotation hierarchy** (most to least desirable):
1. **`assert_type`** — verifies the checker's inference. Proves the system
   works, not just that you annotated correctly.
2. **Annotation fallback** — `x: Tensor[B, C, H, W] = unrefined_op(...)`.
   Use when the op returns unrefined but you know the shape. Document WHY.
3. **`type: ignore`** — the checker produces a WRONG type (algebraic gap
   or conditional equality). Last resort. Always include a comment
   explaining the specific gap.
4. **Bare `Tensor`** — shape genuinely unknowable. Data-dependent token
   counts, conditional accumulation. Document the specific reason.

Type the forward signature:
- Class params for fixed dims (set at construction), method params for
  per-call dims (batch size, sequence length, spatial dims).
- Put parameters whose type vars appear in bare (directly bindable)
  positions BEFORE parameters where they appear inside arithmetic
  expressions. The checker needs to bind the bare params first.
- **Don't hide known class dims inside variadic params.** If the module
  has a class-level Dim `D`, use `Tensor[*Bs, D]` not `Tensor[*S]`.

Replace every `reveal_type` with `assert_type` using the recorded types:
- Shaped `reveal_type` → `assert_type(x, Tensor[...])` with that shape.
- Bare `reveal_type` → `assert_type(x, Tensor)` to document the tracking
  gap, plus a comment noting the root cause (e.g., `# Sequential(*list)`).

**Every local variable in every forward method gets an `assert_type`.**
No exceptions — even inside typed-interface modules. Typed interface means
the *boundary* is typed — it does NOT mean internals are exempt from
`assert_type`. If you think a shape expression is "too complex to write,"
you are guessing — look at what `reveal_type` showed you. The checker
simplifies aggressively.

Run the checker. Fix any `assert_type` failures.

**VERIFY before leaving Step 5.** Paste this in your response:

```
# assert_type count for ClassName.forward:
# Locals: var1, var2, var3 (N total)
# assert_type calls: N
# Match: yes
```

If the counts don't match, you missed some. Go back and add them.

**Step 4 receipt check.** Every bare `assert_type(x, Tensor)` and
every annotation fallback (`x: Tensor[B, C] = untracked_op(...)`)
must cite the Step 4 receipt that justifies it. If no receipt exists,
go back to Step 4 — the restructuring attempt was skipped.

```
# Bare/fallback assert_types and their Step 4 receipts:
# - var2 (bare): receipt MLP.var2 — Sequential(*list), not restructurable
# - var3 (fallback): receipt MLP.var3 — stub returns unrefined, shape known from context
# - var5 (bare): receipt MLP.var5 — input is bare (upstream contagion)
```

**Smoke tests at the bottom of the file** must use `assert_type` on
the typed output, not `assert out.shape == (...)`. Runtime shape
asserts don't exercise pyrefly — they only prove the model runs.
Example:

```python
model = MyModel(num_classes=10)
x = torch.randn(2, 3, 32, 32)  # inferred as Tensor[2, 3, 32, 32]
out = model(x)
assert_type(out, Tensor[2, 10])  # not: assert out.shape == (2, 10)
```

## Step 6: Post-module checklist

Copy this template into your response and fill **every line** before
proceeding to the next module.

```
### Post-module: <ClassName>
- type: ignore count: ___
  For each: [line] [category: A1 / conditional / stub-gap] [fix attempted]
- Step 4 receipts: [list receipt IDs, or "none — all locals shaped"]
- int params: [list each int param and why it's not Dim, or "none"]
- int() casts: [list each, or "none"]
- Sequential(*list): [list each instance and what you did, or "none"]
- bare Tensor in sigs: [list each with reason, or "none"]
- assert_type: ___ checkpoints covering ___ locals in ___ forward methods
- missing stubs: [list each, or "none"]
```

Do not proceed to the next module with unfilled lines.

# Verification (draft review)

Everything above produced a DRAFT. This phase reviews it.

## Run verify_port.sh

Run:

```bash
.claude/skills/port-model/verify_port.sh test/tensor_shapes/models/<model>.py
```

**Paste the FULL output** in your response. Do not summarize or
paraphrase — the raw output is the artifact.

## Investigate each warning

For EACH warning in the verify_port.sh output, write one of:
- **Fixed**: what you changed and why.
- **Accepted**: why this warning is not actionable (cite the specific
  category — A1 algebraic, conditional equality, stub gap not worth
  fixing, etc.).

Do not write "all warnings audited" — list them individually.

## Audit bare assert_types

The port's quality metric is: what fraction of `assert_type` calls verify
a shaped type vs. document a bare `Tensor` gap? Every bare
`assert_type(x, Tensor)` is a tracking gap. Minimizing these is the goal.

For each `assert_type(x, Tensor)` in the port (bare, no shape params):
1. It MUST have a comment explaining the root cause (e.g.,
   `# Sequential(*list)`, `# input is bare`).
2. The root cause MUST have a typed interface receipt from Step 4
   (or trace to one — e.g., "input is bare" because the caller's
   Sequential(*list) was documented in the parent module's receipt).

If any bare `assert_type` lacks a comment or receipt trail, go back
and either fix the tracking gap or document it properly.

**Paste the bare audit in your response:**

```
## Bare assert_type audit
Total assert_type in forward bodies: ___
Shaped (assert_type(x, Tensor[...])): ___
Bare (assert_type(x, Tensor)): ___
Bare fraction: ___

Each bare:
- line N: var — root cause (receipt: <module>.Step4)
- line M: var — root cause (receipt: <module>.Step4)
```

## Compare against known patterns

**Read `style_guide.md` NOW — not earlier.** It is comparison material
for your draft, not preparation material. Reading it during pre-flight
biases you toward patterns you haven't empirically verified.

For each module in your port, find the closest matching pattern in the
style guide. **Paste a comparison:**

```
## Style guide comparison
| Module | My approach | Closest style guide pattern | Could I improve? |
|--------|------------|---------------------------|-----------------|
| ... | ... | ... | yes/no — reason |
```

If any row says "yes", go back and try the improvement before proceeding.
If it doesn't work, document why in the row.

## Re-run verify_port.sh

If you made any changes during this phase, re-run the script and paste
the new output. If no changes were made, write "No changes — output
unchanged."

**Re-check callers.** If you changed a module's forward signature or
return type during this phase, re-run `reveal_type` in every module
that calls it and update their `assert_type` expectations. A fix to
module X can change the inferred types in module Y's forward body.

## Completion report

Before reporting the port as done, copy and fill this template in your
response. **Do not report completion with unfilled blanks.**

```
## Port complete: <model name>
Gate 1 ops audited: ___. Stubs added/fixed: ___.
Gate 2 inventory items: ___. All checked off: yes/no.
Modules ported (dependency order): ___
Step 6 checklists filled for each: yes/no
type: ignore total: ___
  ___ A1 algebraic, ___ conditional equality, ___ stub gap, ___ other
assert_type total: ___ (___ shaped, ___ bare)
Bare fraction: ___%
Each bare assert_type has comment + receipt trail: yes/no
smoke tests: ___ — all use `assert_type` on typed output (not `.shape ==`): yes/no

Verification phase:
- verify_port.sh warnings: ___
  Fixed: ___. Accepted: ___ (each justified above).
- Style guide comparison rows: ___
  Improvements attempted: ___. Improvements that worked: ___.
- verify_port.sh re-run (if changes made): 0 actionable: yes/no
```

# Import convention

The `shape_extensions` package bridges pyrefly's type system and Python runtime.
Importing it patches `torch.Tensor`, `nn.Conv2d`, and other torch classes to
accept subscript syntax (e.g., `Tensor[B, C, H, W]`) at runtime without
crashing. It also provides `TypeVar` with arithmetic support (`N + 1`, `N // 2`
return `self` instead of `TypeError`) and `Dim` for binding runtime ints to
type-level symbols.

**Type-checking only (recommended for ports):** guard imports so annotations
are invisible at runtime:

```python
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from shape_extensions import Dim
```

**Runtime-compatible annotations:** if you need annotations to evaluate at
runtime (e.g., for runtime shape validation), import `shape_extensions` directly
(not under `TYPE_CHECKING`). Use old-style `shape_extensions.TypeVar` instead of
PEP 695 syntax, since `class Foo[T]` doesn't support arithmetic on `T` at
runtime. Alternatively, `from __future__ import annotations` defers evaluation
so annotations never execute, but then `assert_type` becomes a no-op at
runtime.
