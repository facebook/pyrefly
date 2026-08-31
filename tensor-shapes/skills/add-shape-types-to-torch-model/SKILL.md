---
name: add-shape-types-to-torch-model
description: >
  Port a PyTorch model to use pyrefly's tensor shape type system (Tensor[[B, C, H, W]],
  Int[T]). Use this skill whenever the user wants to add shape annotations
  to a PyTorch model, type a model with tensor dimensions, port a model to use shape
  tracking, or annotate model forward methods with tensor shapes. Also use when the
  user mentions tensor shape ports, Int types for PyTorch, or pyrefly shape checking
  on a model file. Invoke BEFORE starting any model port — the skill's gated workflow
  prevents common failure modes.
---

You are porting a PyTorch model to use pyrefly's tensor shape type system.

**The methodology below is mandatory; how much you write down depends on the
job.** Each step has an artifact — an audit table, a per-local `reveal_type`
dump, a typed-interface receipt, an `assert_type` count. Working through them in
order is not optional: the next step's input is the previous step's output, and
skipping a step means substituting reasoning for testing, which is the primary
failure mode.

What varies is the *deliverable*:

- **Annotating someone's own model** (the common case): do every step as internal
  reasoning, but what you hand back is the annotated model plus a short report of
  what couldn't be tracked. You don't need to paste every table — the "paste this"
  instructions below are for the contribution case.
- **Contributing a reference example** (e.g. into a maintained example corpus):
  the tables, receipts, and counts ARE part of the deliverable — paste them in
  full, because others read these ports to learn the patterns. A skill that
  invokes this one for corpus work will say so; absent that, assume the lighter
  deliverable.

# Before you start: two questions

Resolve these with the user before Gate 0. In the common case these are the only
two times you interrupt them:

1. **Confirm how you'll check, and where the stubs are.** State it operationally —
   e.g. "I'll type-check with `<command>` against the shape stubs at `<path>`,
   correct?" — and let the user confirm or correct. Don't make them reason about
   internal concepts; just confirm a command and a location. Default the command
   to `pyrefly check` (assume `pyrefly` is on `PATH`, however it was installed) and
   discover where the stubs live from `pyrefly dump-config`, which prints the
   resolved search path. If a skill invoked this one, it may supply both instead
   (for example, an in-repo build-and-check command).
2. **Ask whether they're open to changing the stubs.** Adding or refining a stub
   signature can recover shapes an op would otherwise lose.
   - **No** (or unsure): don't touch the stubs. Port the model as well as the
     existing stubs allow, collect what couldn't be tracked, and report it at the
     end (see Completion report), suggesting an upstream issue for the gaps worth
     closing. This path is fully supported — a less-complete port is the expected
     outcome here, not a failure.
   - **Yes**: you have standing permission to make straightforward stub-signature
     improvements as you go. Don't re-ask per op. Still port first and confirm the
     gap with `reveal_type` before editing a stub.

Don't ask about deeper changes (teaching Pyrefly new shape *logic*) up front —
that comes up only reactively, and rarely (see "When an op's shape is wrong").

# Pre-flight

**Create tasks** for each gate and the module loop/verification phases
that follow. Update them as you complete each stage — this gives
visibility during long-running ports.

Complete these gates before writing any code.

## Gate 0: Understand the system

Read `shape_tracking_capabilities.md` (this skill dir). It explains the
three shape-tracking mechanisms (shape-aware stubs, DSL functions, special
handlers) and how to check each one, plus the current `shape_extensions` API
surface (`Int` / `IntVar` / `IntTuple` / `Elements` / `assert_shape` / runtime
compat) and the double-bracket `Tensor[[...]]` convention. You need this context
to make the Gate 1 audit meaningful — knowing whether an op exists in a stub is
not the same as knowing whether its shapes are tracked.

**Do NOT read `style_guide.md` yet.** It is comparison material for the
verification phase. Reading it now biases you toward known patterns
before you have empirically probed this model's shapes.

## Gate 1: Audit ops

List every `nn.Module` subclass and `torch`/`F.` function called in the
model. Check each against the shape-aware torch stubs (the `.pyi` files under the
stub root you confirmed up front — `pyrefly dump-config` reports it) and any shape
DSL declarations in those stubs. The stubs attach DSL shape functions with
`@uses_shape_dsl(ir_fn)`, and the IR functions live in `_shapes.pyi` next to the
stubs, imported from stub files as `torch._shapes` because the stub package
provides the `torch` package for type checking. This is a **diagnostic** pass — you are recording
which ops are tracked and which aren't, not fixing anything. A gap here never
blocks the port. This step should take minutes — you are scanning the stub file
for the op and, only when a decorator points there, the corresponding IR
function.

**Do NOT delegate this audit to code search agents or use web search for this
step.** The torch-stubs package and its `_shapes.pyi` DSL
file are exhaustive for torch shape support. For each op in your list, check
whether it appears in the relevant stub file and whether it has a precise
generic signature, `Self`/`Tensor[S]` (whole-shape `S: IntTuple`) return, or `@uses_shape_dsl(...)`
decorator that refines a bare declared return. Use targeted file reads or
repo-approved search scoped to known files/directories, not broad recursive
shell search. You need to confirm presence and spot missing attributes
(e.g., `bias` on `Conv2d`), not memorize every signature.

**Paste your audit table** in your response before proceeding to Gate 2:

```
## Gate 1: Ops audit
| Op | Stub location | Shape DSL decorator / IR fn (or "no decorator") | Status |
|----|---------------|------------------------------|--------|
| nn.Conv2d | tensor-shapes/pyrefly-torch-stubs/torch-stubs/nn/__init__.pyi — generic [InC,OutC,K,S,P,D] | no decorator (stub generic) | tracked-stub |
| F.adaptive_avg_pool2d | tensor-shapes/pyrefly-torch-stubs/torch-stubs/nn/functional.pyi — bare declared return | `@uses_shape_dsl(adaptive_pool_ir)`, defined in `_shapes.pyi` | tracked-DSL |
| ...
```

Status is one of `tracked-stub`, `tracked-DSL`, `tracked-handler` (a special
handler tracks it), or `GAP` (no shape support found).

Filling the "Shape DSL decorator / IR fn" column requires checking whether
the stub declaration has a `@uses_shape_dsl(...)` decorator. If it does,
confirm the named IR function exists in `_shapes.pyi` next to the stubs.
Write "no decorator" only after confirming the stub declaration has no
decorator — do not leave this column blank or write "check DSL".

A `GAP` is information, not a blocker — note it and move on. You decide what to do
about gaps later: most degrade gracefully to a bare `Tensor`, and only if the user
opted into stub changes is a `GAP` worth a stub fix.

## Gate 2: Inventory the original

Write the inventory as a comment block at the top of the port file, using
this exact format:

```python
# ## Inventory
# - [ ] ClassName.__init__ — Int: param1, param2; int: param3
# - [ ] ClassName.forward
# - [ ] function_name — utility, no tensors
# ...
```

**Every class, function, and method in the original file must appear, and
every item must be ported.** Do not skip or exclude anything. If a class
depends on a library without shape-aware stubs, port it anyway — its tensors
fall back to bare `Tensor`, which is acceptable; record the gap for the report.
Only if the user opted into stub changes, and a minimal stub for the specific
ops used would recover real shapes, is adding one worthwhile.

For each class, list constructor parameters and whether each is Int or
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
priority order, and stub philosophy.

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
- **Int**: the value determines a tensor dimension (flows to `nn.Linear`,
  `nn.Conv2d`, tensor creation, or any typed function that uses the value
  as a shape dimension).
- **int**: iteration count (`n_layers`, `n_res_block`) or boolean-like flag.

If in doubt, make it `Int`. The cost is one more type param; the cost of
`int` is permanent shape loss in everything downstream.

**Critical rules:**
- Every `int` that flows to a sub-module constructor (`nn.Linear(dim, ...)`)
  MUST be `Int`. No exceptions.
- Never cast Int to int (`int(dim)`, `self.x = int(dim)`) — `Int` is a
  subtype of `int`, so the cast only kills tracking. Exception:
  `bool`/`float` conversion is necessary, but `int * Int` produces
  Unknown — check whether it reaches tensor shapes before fixing. Don't
  replace with if/else branching (union of concrete expressions is worse
  than one Unknown).
- Derived dims use expressions (`D // NHead`, `4 * ES`), not independent
  type params. Only independent degrees of freedom get type params.
- **Dimensions from `list[int]`.** `list[int]` element access (e.g.,
  `hidden_units[-1]`) erases the concrete value to `int`. Add an explicit
  `Int` field to the config or constructor for that value.
- Use `nn.Buffer` and `nn.Parameter`, not `register_buffer`/
  `register_parameter`.
- **Bridge dims.** When part of the model is untracked (e.g., features
  built via `nn.Sequential(*list)`), look for dimensions that connect
  the untracked section to tracked downstream modules. For example, if
  features output feeds a Linear classifier, the Linear's `in_features`
  is a bridge dim — making it a class type param enables annotation
  fallback to recover a shaped type (e.g., `Tensor[[B, LastC]]`) that
  then flows naturally through downstream ops. Without it, annotation
  fallback can only recover bare `Tensor` or batch-only shapes.

  ```python
  class Model[NC: IntVar, LC: IntVar](nn.Module):
      def __init__(self, num_classes: Int[NC] = 1000,
                   last_channel: Int[LC] = 1280):
          ...
          self.classifier = nn.Linear(last_channel, num_classes)
  ```

  Here `LC` bridges the untracked feature extractor to the typed
  classifier, recovering `Tensor[[B, NC]]` at the output.
- **`Int[X] | None` for optional dimensions.** When a parameter is
  `Optional[int]` but flows to tensor shapes when present, type it as
  `Int[X] | None`, not `Optional[int]`. Example:
  `rank_k: Optional[int]` → `rank_k: Int[RK] | None`. In the forward
  method, narrow with `if rank_k is not None:` — the checker then
  treats `rank_k` as `Int[RK]` inside the branch. Leaving it as
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

  # Good: causal_mask is Tensor[[MS, MS]] from declaration
  class Attention[MS: IntVar](nn.Module):
      def __init__(self, max_seq_len: Int[MS], ...):
          self.causal_mask: Tensor[[MS, MS]] = torch.zeros(
              max_seq_len, max_seq_len
          )
  ```

  Reserve `Tensor | None` for fields that may *genuinely* never be
  set. If late init is unavoidable because the shape depends on a
  runtime decision, accept that bare `Tensor` post-narrow is the
  right answer and document it as a Step 4 receipt.

## Step 2: Type the constructor

Write `__init__` with the `Int` params from Step 1. Construct sub-modules
using those Int params — they get typed automatically.

**Default values for Int params:** `Literal[0]` is not assignable to
`Int[X]` as a default. Use PEP 696 type-parameter defaults instead:

```python
# Won't work — Literal[1000] not assignable to Int[NC]:
def __init__(self, num_classes: Int[NC] = 1000): ...

# Works — NC defaults to 1000 at the type level (bound + PEP 696 default):
class Model[NC: IntVar = 1000](nn.Module):
    def __init__(self, num_classes: Int[NC] = 1000): ...
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
class Config[D: IntVar, NHead: IntVar, VocabSize: IntVar]:
    dim: Int[D]
    n_head: Int[NHead]
    vocab_size: Int[VocabSize]
    dropout: float = 0.0
```

Modules extract only the params they need using `Any` for the rest:

```python
class MLP[D: IntVar](nn.Module):
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
class Config[D: IntVar = 768, NHead: IntVar = 12, VocabSize: IntVar = 50257]:
    dim: Int[D] = 768  # type: ignore[bad-assignment]
    n_head: Int[NHead] = 12  # type: ignore[bad-assignment]
    vocab_size: Int[VocabSize] = 50257  # type: ignore[bad-assignment]
    dropout: float = 0.0
```

Two different defaults are at play here, and only one is clean:
- The **PEP 696 defaults on the type params** (`[D: IntVar = 768, ...]`) are
  what let callers omit dims — those need no ignore.
- The **dataclass field literal defaults** (`dim: Int[D] = 768`) still need
  `# type: ignore[bad-assignment]`, because a plain `int` literal is not
  assignable to `Int[D]`. This is the accepted corpus pattern (see
  `examples/finalmlp.py`). Note that *constructor*-parameter defaults
  (`def __init__(self, num_classes: Int[NC] = 1000)`) do **not** need the
  ignore — only dataclass field defaults do.

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
# var1 (line N): Tensor[[B, C, H, W]]  → SHAPED
# var2 (line M): Tensor                 → BARE — investigate in Step 4
# var3 (line P): Tensor[[B, D]]         → SHAPED
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
trackable if the dimension values are `Int`-typed.

For EACH bare `Tensor` from Step 3, attempt ALL applicable restructurings
before falling back to typed interface:

□ **`int()` or `round()` wrapping a Int value?** Remove it. If the
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

□ **Op genuinely missing from the stubs?** Confirm it's absent (check the stubs
  and any `@uses_shape_dsl(...)` IR function they reference). A missing shape is
  not a blocker — it degrades to a bare `Tensor` that you document below. If the
  user opted into stub changes and a refined signature would recover the shape,
  that's a fair fix. If instead Pyrefly computes a *wrong* shape, see "When an
  op's shape is wrong".

□ **About to claim an op is untracked?** Check the shape-aware stubs, their
  `@uses_shape_dsl(...)` decorators and IR functions, and special handlers
  first. The system tracks reshape, flatten, permute, transpose, cat, stack,
  matmul, arange, zeros, outer, interpolate, einsum, and many more.

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
- Missing stub/DSL: [checked stubs + `@uses_shape_dsl` IR — reason]
- Int | None reclassification: [reclassified param X / not applicable — reason]
- Bridge dim: [promoted X to class Int / not applicable — reason]
- Config parameterization: [parameterized Config[...] / not applicable — reason]
Result: still bare after all checks. Using typed interface because ___.
```

If you cannot fill this out, you have not completed Step 4. Go back.

"Restructure" usually means a 2–3 line change: separating an iteration,
removing an `int()` cast, or adding an `Int` type param. It does NOT mean
rewriting the algorithm. If you find yourself writing significantly
different logic, you've gone too far. Even partial dim tracking (e.g.,
output dim only) is far more useful than none.

**`type: ignore` categories.** Before writing `type: ignore`, identify
which category applies:
- **A1 algebraic gap** (`N * (X // N) ≠ X`): no fix, use `type: ignore`.
- **Conditional equality** (e.g., `Inp == Oup` at runtime but separate
  type params): no fix, use `type: ignore`.
- **Stub gap** (op missing, or its signature too loose to track): if the user
  opted into stub changes, refining the stub signature is the fix; otherwise
  document the bare result and move on. A *wrong* computed shape is a different
  case — see "When an op's shape is wrong".
- **`return-value` mismatch from untracked sub-section**: don't
  `type: ignore`. The fix is upstream — find the bridge dim
  connecting the untracked section (e.g., `nn.Sequential(*list)`
  features) to the tracked downstream input (e.g., a classifier
  Linear), promote it to a class type param per Step 1's bridge-dim
  rule, then use annotation fallback to recover the shaped return.
- **Branch join**: try restructuring first.

Once you've settled the category, use the specific error code (a bare
`# type: ignore` is rejected). The codes seen across the corpus:
- `bad-assignment` — a dataclass field literal default (`dim: Int[D] = 768`) or
  a typed fallback assignment (`x: Tensor[[B, N]] = untracked_result`).
- `arg-type` — passing a bare/looser value into a shaped parameter (common in
  init/setup helpers).
- `assert-type` / `bad-return` — an A1 algebraic gap where the computed shape
  differs from the `assert_type`/declared-return shape.
- `bad-argument-type`, `return-value` — the argument/return variants of the
  above; `return-value` from an untracked sub-section should be fixed upstream
  (see the bridge-dim bullet), not ignored.

**When an op's shape is wrong.** Everything above handles a *missing* shape (the
op falls back to bare `Tensor`) — that always degrades gracefully and never
blocks. The rare hard case is a *wrong* shape: Pyrefly computes a concrete shape
that is incorrect (e.g. integer floor-division where the real op rounds up). You
cannot annotate around this — the checker actively disagrees with reality.
When it happens, tell the user; fixing it means teaching Pyrefly new shape logic,
not editing a stub signature. If a shape-DSL skill (e.g. `modify-shaped-array-dsl`)
is available, hand off to it; otherwise file an upstream issue describing the op
and the correct rule, and document the spot with `type: ignore` for now. Don't
reach for this on ordinary bare-`Tensor` gaps — only when a computed shape is
provably wrong.

**Bare `Tensor` where you know the shape?** Use `assert_type` to verify
inference, not annotation fallback. Annotation fallback silently accepts
bare `Tensor` — it doesn't prove tracking works. If the checker can't
infer the shape, trace upstream to find where shapes were actually lost.

## Step 5: Write forward and assert_type

**Annotation hierarchy** (most to least desirable):
1. **`assert_type`** — verifies the checker's inference. Proves the system
   works, not just that you annotated correctly.
2. **Annotation fallback** — `x: Tensor[[B, C, H, W]] = unrefined_op(...)`.
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
  has a class-level Int `D`, spell the trailing dim out with the variadic
  batch idiom: `Tensor[[*Elements[Bs], D]]` (with `Bs: IntTuple`), not a
  whole-shape `Tensor[S]` that swallows `D`. See `examples/tacotron2.py`.

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
every annotation fallback (`x: Tensor[[B, C]] = untracked_op(...)`)
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
x = torch.randn(2, 3, 32, 32)  # inferred as Tensor[[2, 3, 32, 32]]
out = model(x)
assert_type(out, Tensor[[2, 10]])  # not: assert out.shape == (2, 10)
```

## Step 6: Post-module checklist

Copy this template into your response and fill **every line** before
proceeding to the next module.

```
### Post-module: <ClassName>
- type: ignore count: ___
  For each: [line] [category: A1 / conditional / stub-gap] [fix attempted]
- Step 4 receipts: [list receipt IDs, or "none — all locals shaped"]
- int params: [list each int param and why it's not Int, or "none"]
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

Run `verify_port.sh` (in this skill dir) on your port file:

```bash
tensor-shapes/skills/add-shape-types-to-torch-model/verify_port.sh <path/to/your/port.py>
```

**Paste the FULL output** in your response. Do not summarize or
paraphrase — the raw output is the artifact.

## Run the actual Pyrefly check

`verify_port.sh` is a heuristic quality gate; it does not type check the port.
You must also run Pyrefly itself against your port file. There is no
`--tensor-shapes` flag — shape tracking is on whenever the shape stubs and
`shape_extensions` are on the search path. The single-file check mirrors
`tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py`:

```bash
pyrefly check --config /dev/null --python-version 3.13 \
    --search-path <root containing torch-stubs> \
    --search-path <root containing shape_extensions> \
    path/to/your/port.py
```

The two search roots are separate: `tensor-shapes/pyrefly-torch-stubs` (the
`torch-stubs` package) and `tensor-shapes/pyrefly-shape-extensions` (the
`shape_extensions` package). In a Buck checkout you can instead pass the combined
filegroup `fbcode//pyrefly/tensor-shapes:torch-stubs-search-path` as a
`--search-path`. If a skill invoked this one, it may supply its own
build-and-check command; use that instead. `pyrefly dump-config` reports the
resolved search path when the stubs are already installed in your environment.

**Python version:** the PEP 695/696 generics syntax (`class Net[D: IntVar]`,
type-param defaults) requires `--python-version 3.12` or later; the corpus runs
`3.13`.

To type-check the whole corpus (or a stack of edits) at once:

```bash
python3 tensor-shapes/run_all_shape_tests.py --mode cargo|buck [--include-runtime-tests]
```

Paste the Pyrefly output. The result must be `0 errors`; `reveal_type` info is
acceptable only while probing and must not remain in the finished port.

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
- Pyrefly check: 0 errors: yes/no
- Style guide comparison rows: ___
  Improvements attempted: ___. Improvements that worked: ___.
- verify_port.sh re-run (if changes made): 0 actionable: yes/no

Gaps & proposed improvements (for the user):
- Ops that stayed untracked, and why: ___ (or "none")
- Stub-signature improvements that would recover shapes (if stubs weren't
  changed): ___ (or "none")
- Wrong computed shapes found (needing a shape-DSL change): ___ (or "none")
- Suggest filing an upstream issue for: ___ (or "none")
```

The "Gaps & proposed improvements" block is the user-facing payload of the
lighter deliverable — when you didn't paste the full tables, this is where the
shortcomings of the port surface. Fill it from the `GAP`s you recorded in Gate 1
and the Step 4 receipts.

# Import convention

The `shape_extensions` package bridges pyrefly's type system and Python runtime.
Importing it patches `torch.Tensor`, `nn.Conv2d`, and other torch classes to
accept subscript syntax (e.g., `Tensor[[B, C, H, W]]`) at runtime without
crashing. It also provides `IntVar` with arithmetic support (`N + 1`, `N // 2`
return `self` instead of `TypeError`) and `Int` for binding runtime ints to
type-level symbols.

`shape_extensions` is installed alongside the shape-aware torch stubs (wherever
those live in your environment — `pyrefly dump-config` reports the location). In an
fbsource Buck checkout specifically, the runtime package is
`fbcode//pyrefly/tensor-shapes/pyrefly-shape-extensions:shape_extensions`, the importable stub package is
`fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs:torch-stubs`, and the
filegroup to pass as a Pyrefly `--search-path` is
`fbcode//pyrefly/tensor-shapes:torch-stubs-search-path`.

Pick an import mode based on whether the port file will be **executed**, not
just type-checked:

*Static-check-only (common case — you only run the checker on the file):* guard
the shape imports. The file is never executed, so the annotations never evaluate
and the guarded names need not resolve at runtime.

```python
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from shape_extensions import Elements, Int, IntTuple, IntVar
```

*Runnable (the file is imported/executed):* a guarded import alone will crash,
because annotations — and PEP 695 type-param bounds — still evaluate at runtime.
Either:
- add `from __future__ import annotations` to postpone annotation evaluation,
  and import the symbols that appear in *runtime-evaluated* positions
  (`Elements`, plus the bounds `IntVar`/`IntTuple` used in `[Bs: IntTuple]`) at
  module top — see `examples/runtime/nanogpt_future_annotations_runnable.py`; or
- import all shape symbols at module top with no guard — see
  `examples/runtime/gptfast_sym_int_var_runnable.py`.

`Int` binds runtime ints; `IntVar` is the bound for scalar dim params
(`[D: IntVar]`); `IntTuple` is the bound for variadic/whole-shape params
(`[Bs: IntTuple]`); `Elements` unpacks a variadic batch
(`Tensor[[*Elements[Bs], D]]`). Import only the ones a given file uses.

**Runtime-compatible annotations:** if you need annotations to evaluate at
runtime (e.g., for runtime shape validation), import `shape_extensions` directly
(not under `TYPE_CHECKING`). Use old-style `shape_extensions.IntVar` instead of
PEP 695 syntax, since `class Foo[T]` doesn't support arithmetic on `T` at
runtime. Alternatively, `from __future__ import annotations` defers evaluation
so annotations never execute, but then `assert_type` becomes a no-op at
runtime.
