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

**This usually works.** The example corpus contains dozens of substantial,
explicitly scoped ports of real open-source research models, including
transformers, convolutional encoder-decoders, autoregressive architectures, and
dynamic module stacks. A first checker pass with errors or gradual tensors is a
diagnosis, not evidence that the model is unportable. Work module by module and
expect a few short probe–fix–check iterations before accepting a boundary.

When subagents are available, use them for bounded, independent read-only work:
map the source's stated shapes, audit the model's ops against the stubs, or
cluster checker errors by root cause. The main agent still owns the port and
integrates the evidence; do not have several agents edit the same file.

The gates below prevent the common failure mode of writing the whole port from
reasoning and checking only at the end. Apply their rigor according to the
deliverable:

- **Annotating someone's own model** (the common case): use the source evidence,
  op audit, probes, and verification below, but keep working notes private and
  checkpoint shape-changing locals and module boundaries rather than every
  trivial assignment. Hand back the annotated model plus a short report of what
  could not be tracked.
- **Contributing a reference example** (e.g. into a maintained example corpus):
  this is the exhaustive case. Record the audit table, probe every tensor local,
  keep per-local `assert_type` coverage, and provide the receipts and counts
  because other people use the port as reference material. A skill that invokes
  this one for corpus work will say so.
- **Migrating an existing production codebase:** preserve its structure and
  tests. Prioritize precise component inputs and outputs; internal inference is
  valuable evidence for those contracts but is secondary when the body is highly
  dynamic. Keep gradual regions small and recover precise shapes at their exit.
  Inventory existing shape annotations and public tensor boundaries, but do not
  add inventory comments, an `assert_type` after every local, or model-file smoke
  tests. Add focused static tests only for reusable stub behavior. Use temporary
  `reveal_type` probes as needed and remove them before handoff. Existing test
  and lint conventions take precedence.

## Existing jaxtyping annotations

If the goal is native-syntax conversion, inventory every jaxtyping annotation
and its intended shape, including local annotations. If the goal is shape
checking with minimal production churn, consider declaring dimension names with
`@static_jaxtyping("...")` and keeping the existing annotations instead. Choose
from the request and repository context; do not convert syntax merely because a
conversion is possible.

After a native conversion, compare the inventory against the new annotations:
every removed annotation must have a native counterpart with equal or better
precision, or a documented reason why that precision is not representable.

Treat each jaxtyping annotation as an information-preservation obligation, not
merely syntax to remove. Preserve its rank, literal dimensions, named-dimension
equalities, arithmetic relationships, and variadic-prefix semantics whenever
native shape types can express them. A clean type check is not sufficient if a
precise jaxtyping contract became gradual.

- Never delete a local annotation merely because current inference appears
  sufficient. Preserve it with native syntax and reuse any enclosing `IntVar`
  or `IntTuple` parameters so it continues to document and check the intended
  relationship.
- Convert a variadic prefix such as `"*B D"` using `Bs: IntTuple` and
  `Tensor[[*Bs, D]]`; do not collapse it to a bare `Tensor`. Use
  `*Elements[Bs]` only when annotations evaluate eagerly at runtime.
- When a dimension is known only from a runtime value and cannot be related to
  an input type parameter, preserve its rank and documentation with a named
  output-only `IntVar`, for example `Tensor[[B, NewH, NewW]]`. Pyrefly
  instantiates such unconstrained variables as gradual dimensions at call
  sites. Use `int` only when a useful name is unavailable, and bare `Tensor`
  only when even the rank is genuinely unknown. Record every loss of precision
  in the before/after audit.
- Do not weaken a public signature because an implementation detail is
  untracked. First try a more precise stub or a narrow `cast` immediately around
  the untracked operation. Keep all dimensions bound by parameters on both sides
  of that boundary. A runtime rewrite is separate work and requires explicit
  user direction.
- Do not assume an empty jaxtyping shape string accurately describes a scalar.
  Check how the value is used. If it accepts arbitrary ranks, bind its complete
  shape with `Shape: IntTuple` and `Tensor[[*Shape]]`; if the code requires
  trailing dimensions, spell those out after a variadic prefix.
- Treat code that reads annotations at runtime—including schema validators—as
  executable behavior. Adapt and test that consumer separately rather than
  assuming a syntactically equivalent annotation preserves validation.

# Before you start: resolve two setup choices

Resolve these from the repository and invoking skill whenever possible. Do not
interrupt the user when the checkout, config, or request already answers them;
ask only when a real ambiguity remains.

1. **Choose the check command and validate the environment.** Default to
   `pyrefly check` and inspect the resolved paths with `pyrefly dump-config`. The
   Torch stubs are partial overlays, so Pyrefly must also see a real `torch`
   installation through the selected interpreter or `--site-package-path`.
   When the model uses einops, add the `pyrefly-einops-stubs` root too; without
   that overlay, an einops transform can incorrectly appear to preserve its
   input shape. Before porting, check one known-good nearby example and require
   `0 errors`. If it fails, repair the environment rather than weakening the
   model annotations. In a repository, prefer its documented check command; an
   invoking skill may provide all paths, as the corpus skill does.
2. **Decide whether stub changes are in scope.** For someone's model or a
   production migration, default to leaving shared stubs unchanged and report
   gaps. For a maintained reference-corpus contribution, stub improvements are
   in scope by default. If the user explicitly asks for or rules out stub work,
   follow that. Ask only when neither context nor request establishes the scope.

A no-stub-change port is a supported outcome, not a failure. If stub changes are
in scope, port first and confirm each gap with `reveal_type` before editing a
stub. Do not decide about deeper Pyrefly shape-logic changes up front; react only
if a computed shape is provably wrong.

# Pre-flight

**Create tasks** for each gate and the module loop/verification phases
that follow. Update them as you complete each stage — this gives
visibility during long-running ports.

Complete these gates before writing any code.

## Gate 0: Understand the system

Read `shape_tracking_capabilities.md` and `porting_principles.md` (this skill
dir). They explain the tracking mechanisms, current API, priority order, and
stub philosophy.

Then skim the model index at the end of `style_guide.md` and open one to three
closest examples under `pyrefly-torch-stubs/examples/`. Use them as evidence
that substantial real models can be ported and as templates for
architecture-level patterns such as attention, dynamic module stacks,
encoder-decoders, and variadic batches. Existing prose can be stale: only a
current `assert_type`, focused probe, or passing check proves the present
behavior. Do not copy annotations blindly; the target source and checker remain
authoritative. Save the full style-guide comparison for verification.

## Gate 1: Audit ops

Inventory every tensor-producing or shape-changing expression in the selected
model boundary: module construction and calls, Tensor methods, operators and
indexing, `torch`/`F.` functions, module-instance methods, third-party APIs, and
dynamic dispatch. For Torch APIs, inspect the relevant partial overlay module
and read the return annotation; if it calls a type-level shape function, confirm
that function in `_shapes.pyi`. Special handlers live separately in Pyrefly's
Rust implementation, so confirm them with source or a minimal probe.

Classify each expression as `tracked-stub`, `tracked-DSL`, `tracked-handler`,
gradual `Tensor`, `Any`, `unavailable-overlay-symbol`, or `third-party-boundary`.
The distinction matters: these are partial PEP 561 overlays, and a public name
omitted from an overlaid module is unavailable rather than guaranteed to fall
back gracefully. A declared loose return may produce a gradual `Tensor`, while
an undeclared Tensor member may flow through `__getattr__` as `Any`.

This is a diagnostic pass, not a blocker. A subagent may collect the source
operation inventory, but the primary agent must validate the final
stub/DSL/handler classification against the local checkout. Do not use web
search as authority for current support.

For a reference-corpus contribution, include the audit table in the work log
before Gate 2. For ordinary and production ports, keep it as concise working
notes unless the user asks for the full audit:

```
## Gate 1: Ops audit
| Op | Stub location | Shape function in the return annotation (or "none") | Status |
|----|---------------|------------------------------|--------|
| nn.Conv2d | tensor-shapes/pyrefly-torch-stubs/torch-stubs/nn/__init__.pyi — generic [InC,OutC,K,S,P,D] | none (generic signature) | tracked-stub |
| F.adaptive_avg_pool2d | tensor-shapes/pyrefly-torch-stubs/torch-stubs/nn/functional.pyi — `Tensor[adaptive_pool2d_shape(Shape, OH, OW)]` | `adaptive_pool2d_shape`, defined in `_shapes.pyi` | tracked-DSL |
| ...
```

Status is one of `tracked-stub`, `tracked-DSL`, `tracked-handler`, `gradual`
(a declared bare return), `Any`, `unavailable-overlay-symbol`, or
`third-party-boundary`.

For a stub or DSL result, read the declaration's return annotation; if it calls
a shape function, confirm that function exists in `_shapes.pyi`. Write "none"
only after confirming that the signature's own generics compute the shape.

A non-precise result is information, not a blocker, but its exact category
determines the response. Preserve known rank and dimensions in the public
contract, isolate a dynamic or third-party boundary with a precise cast when the
shape is justified, and use bare `Tensor` only when rank itself is genuinely
unknown. An unavailable overlay symbol may require a stub declaration just to
type-check; it does not necessarily degrade to `Tensor`. If stub work is in
scope, consider a general stub improvement.

## Gate 2: Scope and inventory the original

Define the port boundary before inventorying it. For a single-file model, the
boundary is usually that file. For a repository containing many model families
(for example, a policy zoo or a framework that builds models from YAML), choose
the smallest coherent requested model plus its transitive custom tensor modules;
do not inventory unrelated training, data, CLI, or deployment code. State the
boundary and proceed. Ask only if several incompatible model families are
plausible and the request gives no way to choose. A corpus-invoking skill may
require one complete upstream file or another explicit boundary.

Mine the source for shape evidence before inventing annotations:

- existing jaxtyping or other tensor annotations;
- docstrings and inline comments such as `(B, C, H, W)` or reshape arrows;
- shape destructuring, runtime shape assertions, `view`/`reshape` arguments,
  `einsum` or einops equations, config dimensions, and test fixtures;
- layer constructor dimensions and equalities implied by residuals, matmuls,
  concatenation, and split sizes.

Turn this evidence into a short shape map and use its names consistently.
Comments are hypotheses, not proof: resolve conflicts by following the actual
operations and tests, then preserve or correct useful comments in the port.
This is especially valuable in research models, which often document shapes
more precisely than their Python annotations do.

For a reference-corpus port, write the inventory as a comment block at the top
of the port file using this format. For an ordinary port, keep the same list in
working notes; production migrations should inventory only changed public
boundaries and existing annotations.

```python
# ## Inventory
# - [ ] ClassName.__init__ — Int: param1, param2; int: param3
# - [ ] ClassName.forward
# - [ ] function_name — utility, no tensors
# ...
```

**Every class, function, and method inside the selected port boundary must
appear, and every inventory item must be ported.** Do not silently narrow the
boundary after discovering a hard dependency. A dependency outside the boundary
may remain behind a precise cast, typed interface, or gradual boundary. If stub
work is in scope, add only general improvements that help more than this model.

For each class, list constructor parameters and whether each is `Int` or `int`.
Also record configuration or mode flags that change tensor rank, layout, return
arity, or container type (training/eval, cache, export, optional backends). Use
`Literal` overloads when the existing API exposes a statically knowable mode;
otherwise preserve the honest union or gradual boundary. Runtime assertions
support preconditions but do not by themselves turn unrelated `int` values into
symbolic dimensions.

Check off corpus inventory items as you port them. Do not proceed to
verification with unchecked items inside the declared boundary.

## Phased annotation (larger codebases)

For a single-file or few-module port, one commit is usually fine. When the
boundary spans several files or many modules, consider annotating in phases
on a branch, one commit per phase. Phases order the work; they do not shrink
the boundary — every inventory item is still ported. Order phases by
dependency: leaf modules and shared contracts first, then the modules that
consume them. Run the module loop over one coherent subset per phase and
require a `0 errors` check before committing, so every commit is a working
checkpoint and later phases never debug earlier ones. Commit messages should
state the phase's coverage (which modules, public-contract coverage delta,
remaining gradual boundaries); keep `reveal_type` probes and working notes
out of the commits. For production migrations this also keeps each diff
reviewable and landable on its own.

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

**ONE MODULE AT A TIME.** Complete Steps 1–6 for module A before starting
module B; in the corpus case, record the Step 6 checklist at that point. If you
find yourself typing two modules' constructors before running the checker, you
have already entered the primary failure mode: writing the entire file and
validating at the end leads to hidden gradual boundaries.

## Step 1: Inventory parameters

List every constructor parameter. For each, decide:
- **Int**: the value determines a tensor dimension (flows to `nn.Linear`,
  `nn.Conv2d`, tensor creation, or any typed function that uses the value
  as a shape dimension).
- **int**: iteration count (`n_layers`, `n_res_block`) or boolean-like flag.

If in doubt, make it `Int`. The cost is one more type param; the cost of
`int` is permanent shape loss in everything downstream.

**Critical rules:**
- A dimension that enters new typed code should be `Int[X]`, not `int`. In an
  existing public API, changing `int` to `Int[X]` narrows the static contract;
  get the user's agreement first. Otherwise preserve the signature and recover
  precision at the first boundary where the dimension is justified.
- `len(tensor)` returns plain `int`, while `tensor.size(dim)` can preserve a
  symbolic `Int`. In annotation-only work, keep the original expression and
  restore precision at a downstream boundary; suggest the equivalent `size`
  spelling separately if it would improve inference.
- `Int` is a subtype of `int`, so `int(dim)` erases tracking when `dim` is already
  an `int` at runtime. Do not remove conversions around floats, tensors,
  `round()`, or expressions such as `seq**0.5`; those conversions can affect
  runtime behavior and are expected gradual boundaries. Arithmetic mixing a
  plain `int` with an `Int` produces plain `int`.
- Derived dims use expressions (`D // NHead`, `4 * ES`), not independent
  type params. Only independent degrees of freedom get type params. Keep
  preconditions such as divisibility, perfect-square sequence lengths,
  nonempty scans, and mask-cardinality equalities separate from shape types;
  runtime assertions document them but do not justify simplifying
  `(H // P) * P` to `H`.
- **Structured inputs and mode-dependent outputs.** Inventory heterogeneous
  tensor mappings and flags for training/eval, cache, export, or optional
  backends. Use an existing `TypedDict`/dataclass or honest union when available.
  Introducing a narrower public container type or overload requires the user's
  agreement that it matches the real API.
- **Dimensions from `list[int]` or other untracked sources.** Element access
  erases the value to `int`. Preserve the source and restore precision with a
  justified cast or typed interface when the value reaches a known component
  boundary. Adding an explicit `Int` config field or constructor parameter
  changes the API; suggest it as separate work unless the user requested it.
- Keep existing `register_buffer`, `register_parameter`, dynamic factories, and
  container construction. Replacing them with `nn.Buffer`, explicit attributes,
  or typed classes is a runtime refactor, not annotation work.
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
- **`Int[X] | None` for optional dimensions.** In new code, a genuinely
  optional shape-bearing parameter can use `Int[X] | None` and narrow with
  `if value is not None:`. Replacing an existing public `Optional[int]` with
  `Int[X] | None` narrows its static contract; do so only with user agreement.
  Otherwise preserve the public type and recover a justified shape downstream.
- **Parameterized config dataclasses.** When multiple modules consume
  dimensions from the same `@dataclass` config, note it — Step 2
  shows how to parameterize the config so dims propagate across
  module boundaries.
- **Lazy-initialized buffer attributes.** An attribute's type is fixed at its
  declaration. If `self.x: Tensor | None = None` is assigned a shaped tensor in
  a later setup hook, reads after narrowing retain only bare `Tensor`. Preserve
  the lifecycle and record this as a dynamic boundary. Eager initialization can
  recover the shape, but it changes runtime structure and requires separate user
  direction.

## Step 2: Type the constructor

Write `__init__` with the `Int` params from Step 1. Construct sub-modules
using those Int params — they get typed automatically.

**Default values for Int params:** a literal constructor default can be used
with `Int[X]` directly. A PEP 696 default is optional: it makes the unspecialized
class name carry that default at the type level.

```python
# Works; callers bind NC from the argument or literal default:
def __init__(self, num_classes: Int[NC] = 1000): ...

# Also works; bare Model means Model[1000]:
class Model[NC: IntVar = 1000](nn.Module):
    def __init__(self, num_classes: Int[NC] = 1000): ...
```

Dataclass field defaults are different: `dim: Int[D] = 768` currently needs a
specific `# type: ignore[pyrefly:bad-assignment]`.

**Constructor patterns that create expected shape-typing boundaries:**
- **`nn.Sequential(*list_var)`** and factory functions returning
  `nn.Sequential` erase the member module types at the dynamic boundary.
- **`getattr(nn, name)()`** and YAML/config-selected module factories return
  `Any` or a broad module type.
- **Heterogeneous module containers** cannot express a different shape transform
  for each runtime index.
- **Method-level type params on class fields.** A field cannot retain a type
  parameter scoped only to the method that assigned it.

Preserve known dimensions in typed public interfaces around these boundaries.
Do not extract modules, replace factories, or redesign containers solely for
shape tracking unless the user separately requests that rewrite.

**Parameterized config dataclasses.** For a new/corpus model, or when the user
agrees to the narrower static API, a `@dataclass` holding shared dimension
hyperparameters can be generic so dimensions propagate through constructors.
For an existing public production config, preserve its annotations by default
and recover shapes at component boundaries instead:

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
    dim: Int[D] = 768  # type: ignore[pyrefly:bad-assignment]
    n_head: Int[NHead] = 12  # type: ignore[pyrefly:bad-assignment]
    vocab_size: Int[VocabSize] = 50257  # type: ignore[pyrefly:bad-assignment]
    dropout: float = 0.0
```

Two different defaults are at play here, and only one is clean:
- The **PEP 696 defaults on the type params** (`[D: IntVar = 768, ...]`) are
  what let callers omit dims — those need no ignore.
- The **dataclass field literal defaults** (`dim: Int[D] = 768`) still need
  `# type: ignore[pyrefly:bad-assignment]`, because a plain `int` literal is not
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

Run the checker to verify the constructor compiles. In the corpus case, record
that output before proceeding; otherwise keep iterating without dumping routine
success output.

## Step 3: Probe the forward

For a corpus contribution, count and probe every tensor local in the forward
method. For an ordinary model, probe each shape-changing local and every point
where the shape could be lost; include enough checkpoints to cover each
operator chain and module boundary. For production code, use targeted temporary
probes around changed boundaries and checker failures.

Add `reveal_type` at those points and run the checker. In the corpus case,
record the results in this format; elsewhere keep concise working notes:

```
# reveal_type results for ClassName.forward:
# Locals: N (list them: var1, var2, var3)
# var1 (line N): Tensor[[B, C, H, W]]  → SHAPED
# var2 (line M): Tensor                 → BARE — investigate in Step 4
# var3 (line P): Tensor[[B, D]]         → SHAPED
```

For a corpus contribution, verify that the reveal count matches the tensor-local
count. In every mode, do not continue while a shape-changing path is unprobed.

**If a reveal_type result contradicts your understanding of the op**
(e.g., spatial dims unchanged after a strided conv, or a shaped op
returning bare), write a small isolating test, run the checker, and
confirm the behavior before proceeding. Either your understanding is
wrong (update your mental model) or the checker has a simplification
you should document. If a second pass still has unexplained plain or unchanged
Tensor results, re-run the environment sanity check and confirm the einops search
path before declaring the code untrackable.

This table is your Step 4 input. Do not write `assert_type` until Step 4
is complete for every BARE entry. The results tell you:
- Shaped type → the checker tracks this op. Write `assert_type` in Step 5.
- Bare `Tensor` → shape lost. Investigate in Step 4 before deciding.

## Step 4: Diagnose and contain tracking boundaries

Not all statically typed tensor code can be shape-typed. Python's type system
cannot represent a general container whose element at each runtime position has
a different shape type, nor can it recover module types chosen by an arbitrary
factory, YAML parser, `getattr`, or data-dependent control flow. Recognizing
such a boundary is a successful diagnosis, not a reason to keep forcing the
checker.

For each bare result:

1. **Trace upstream.** Confirm whether the input was already bare and identify
   the first operation that lost precision.
2. **Apply annotation-only repairs.** Add precise annotations where they do not
   narrow an existing public contract (or where the user approved that
   narrowing), preserve bridge dimensions in component signatures, and use a
   justified cast to regain precision after a dynamic region. Keep original
   `len`, conversion, config, buffer, factory, and container behavior intact.
3. **Check the exact implementation surface.** Confirm the selected overload's
   stub return and any shape function or special handler before calling it
   unsupported. Do not extrapolate from a nearby form: for example,
   `F.interpolate(scale_factor=...)` may track when a symbolic `size=...` form
   remains gradual, and `F.softmax` may be more precise than other softmax entry
   points.
4. **Use a narrow boundary when inference is unavailable.** For an untyped
   third-party transform, cast only its result to the shape established by the
   source evidence. For a dynamic module/container boundary, allow the smallest
   practical region to remain gradual, then re-establish the strongest justified
   shape with one cast or typed interface when execution exits that region. This
   keeps graduality from spreading through downstream component contracts. Use
   bare `Tensor` only when rank itself is unknown.
5. **Record genuine limitations.** Mixed-shape tensor containers, heterogeneous
   `ModuleList` values indexed or iterated dynamically, `nn.Sequential(*items)`
   built from a runtime list, dynamic factories, and data-dependent result
   counts commonly require a gradual boundary.

Fixed list literals such as `torch.cat([a, b])` can retain distinct element
shapes on current Pyrefly. Dynamically built or broadly annotated lists usually
homogenize their members. Probe before changing collection syntax; a fixed or
explicitly typed tuple is useful only when it preserves the original behavior
and the checker demonstrates the benefit.

**Do not redesign runtime code solely to improve shape coverage.** A cast is
acceptable because it does not change runtime behavior. Narrowing a public type
contract is allowed only when the user explicitly agrees that the narrower
contract is correct. Extracting modules from a dynamic container, converting
inheritance to composition, splitting loops or branches, or replacing a model
factory changes source structure and is outside this annotation skill. Report
possible narrowing or rewrite options and let the user request them separately.

Before accepting a bare result or typed boundary, record this short receipt in
working notes; include it in the response for a corpus contribution:

```
## Boundary receipt [<Module>.<variable>]
- First precision loss: <operation or incoming bare value>
- Source evidence for expected shape: <comment/assertion/equation/test>
- Stub / shape function / handler checked: <result>
- Annotation-only repairs tried: <result>
- Boundary: <precise cast / typed interface / bare Tensor and why>
- Rewrite that could improve coverage: <none, or describe for separate approval>
```

**`type: ignore` categories.** Before writing one, identify the cause:
- **A1 algebraic gap** (`N * (X // N) ≠ X`): no annotation-only fix.
- **Conditional equality** (for example, `Inp == Oup` at runtime but separate
  type params): no annotation-only fix.
- **Stub gap** (op missing, or its signature too loose to track): if stub work is
  in scope, refine the general stub; otherwise use a precise cast or document a
  gradual boundary. A wrong computed shape is a different case below.
- **`bad-return` from an untracked subsection**: do not hide a dynamic region at
  the return. Preserve bridge dimensions in the signature and use one narrow
  cast or typed boundary where precision becomes known again.
- **Branch join or heterogeneous container**: preserve runtime structure,
  contain it at a typed boundary, and suggest any rewrite as separate work.

Use `# type: ignore[pyrefly:<code>]` with the exact error code Pyrefly prints,
for example `pyrefly:bad-assignment`, `pyrefly:bad-return`,
`pyrefly:bad-argument-type`, or `pyrefly:assert-type`. Pyrefly may accept a bare
or mismatched code, so accuracy here is documentation for reviewers rather than
a validation mechanism. Do not use mypy spellings such as `arg-type`,
`return-value`, or `assignment`.

**When an op's shape is wrong.** The cases above cover missing precision, which
may appear as a gradual `Tensor`, `Any`, a third-party boundary, or an unavailable
symbol in a partial overlay. None is a reason to discard the surrounding
contract. The rarer hard case is a *wrong concrete shape*: Pyrefly computes a
specific shape that is incorrect (for example, floor division where the runtime
op rounds up). You cannot annotate around that disagreement.
When it happens, tell the user; fixing it means teaching Pyrefly new shape logic,
not editing a stub signature. If a shape-DSL skill (e.g. `modify-shaped-array-dsl`)
is available, hand off to it; otherwise file an upstream issue describing the op
and the correct rule, and document the spot with `type: ignore` for now. Don't
reach for this on ordinary bare-`Tensor` gaps — only when a computed shape is
provably wrong.

**Bare `Tensor` where you know the shape?** A shaped annotation or
`assert_type` against a gradual value can be accepted without proving inference.
First trace the loss. If it is a genuine dynamic or third-party boundary, use a
narrow explicit cast to the source-supported shape and record the boundary;
otherwise fix the annotation or stub that lost precision.

## Step 5: Write forward and assert_type

**Annotation hierarchy** (most to least desirable):
1. **`assert_type`** — verifies the checker's inference. Proves the system
   works, not just that the declared contract is accepted.
2. **Precise `cast` at a boundary** — use when dynamic or third-party code
   cannot express the shape but source evidence establishes it. This preserves
   runtime behavior and makes the non-inferred step explicit.
3. **Annotation fallback** — `x: Tensor[[B, C, H, W]] = unrefined_op(...)`.
   Pyrefly may accept a gradual RHS without proving the shape, so mark and audit
   it exactly like a cast; prefer an explicit cast when clarity matters.
4. **`type: ignore`** — the checker produces a wrong concrete type (algebraic
   gap or conditional equality). Last resort; explain the specific gap.
5. **Bare `Tensor`** — rank genuinely unknown or a dynamic boundary with no
   defensible precise contract. Document the reason.

Type the forward signature:
- Class params for fixed dims (set at construction), method params for
  per-call dims (batch size, sequence length, spatial dims).
- Put parameters whose type vars appear in bare (directly bindable)
  positions BEFORE parameters where they appear inside arithmetic
  expressions. The checker needs to bind the bare params first.
- **Don't hide known class dims inside variadic params.** If the module
  has a class-level Int `D`, spell the trailing dim out with the variadic
  batch idiom: `Tensor[[*Bs, D]]` (with `Bs: IntTuple`), not a whole-shape
  `Tensor[S]` that swallows `D`. Use `*Elements[Bs]` only for eagerly evaluated
  runtime annotations. See `examples/tacotron2.py`.

Use the recorded probes to add `assert_type` checkpoints:
- Shaped `reveal_type` → `assert_type(x, Tensor[...])` with that shape.
- Bare `reveal_type` → `assert_type(x, Tensor)` only when documenting an
  accepted tracking gap, with a comment naming the root cause.

For a corpus contribution, every tensor local in every forward method gets an
`assert_type`; verify the count before leaving the module. For an ordinary
model, checkpoint every shape-changing operation, module boundary, and repaired
shape-loss site. Production code should keep only focused assertions that fit
its existing test conventions. In all modes, do not omit a checkpoint merely
because the shape expression is complex — use what `reveal_type` showed.

For a corpus contribution, record the count in this form:

```
# assert_type count for ClassName.forward:
# Locals: var1, var2, var3 (N total)
# assert_type calls: N
# Match: yes
```

If the counts don't match, you missed some. Go back and add them.

**Boundary receipt check.** Every bare `assert_type(x, Tensor)`, precise cast,
and annotation fallback (`x: Tensor[[B, C]] = untracked_op(...)`) must cite the
Step 4 boundary receipt that justifies it. If no receipt exists, go back and
diagnose the first precision loss.

```
# Bare/cast/fallback sites and their boundary receipts:
# - var2 (bare): receipt MLP.var2 — runtime-built mixed module list
# - var3 (cast): receipt MLP.var3 — third-party transform, shape from equation
# - var5 (bare): receipt MLP.var5 — input is bare from the parent boundary
```

For reference-corpus examples, smoke tests at the bottom of the file must use
`assert_type` on the typed output, not `assert out.shape == (...)`. Runtime shape
asserts do not exercise Pyrefly; they only prove the model runs. For ordinary or
production code, preserve the project's test layout and add an equivalent
static check only where its conventions support one.

```python
model = MyModel(num_classes=10)
x = torch.randn(2, 3, 32, 32)  # inferred as Tensor[[2, 3, 32, 32]]
out = model(x)
assert_type(out, Tensor[[2, 10]])  # not: assert out.shape == (2, 10)
```

## Step 6: Post-module check

Before proceeding, confirm the module's public contract, shaped checkpoints,
gaps, and ignores are understood. For a corpus contribution, record every line
of this checklist; for other ports, keep only the items that found a gap:

```
### Post-module: <ClassName>
- type: ignore count: ___
  For each: [line] [category: A1 / conditional / stub-gap] [fix attempted]
- Boundary receipts: [list receipt IDs, or "none — all tracked"]
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

Run `verify_port.sh` on standalone and corpus model files. For production code,
use it only when its corpus-oriented checks are relevant; the repository's own
lint, type-check, and tests are authoritative.

```bash
tensor-shapes/skills/add-shape-types-to-torch-model/verify_port.sh <path/to/your/port.py>
```

For a corpus contribution, include the full output in the work log. Otherwise,
report actionable warnings and what you did about them.

## Run the actual Pyrefly check

`verify_port.sh` is a line-oriented advisory check; it does not type-check the
port or establish per-forward coverage, and multiline constructs can confuse its
counts. Treat its output as hints. You must also run Pyrefly itself against the
port file. There is no `--tensor-shapes` flag — shape tracking is on whenever
the shape stubs and `shape_extensions` are on the search path. The single-file
check mirrors `tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py`:

```bash
pyrefly check --config /dev/null --python-version 3.13 \
    --search-path <root containing torch-stubs> \
    --search-path <root containing shape_extensions> \
    --site-package-path <site-packages containing real torch> \
    path/to/your/port.py
# If the model imports einops, also pass:
#   --search-path <root containing the shape-aware einops-stubs>
```

The core search roots are `tensor-shapes/pyrefly-torch-stubs` (the
`torch-stubs` package) and `tensor-shapes/pyrefly-shape-extensions` (the
`shape_extensions` package). The partial overlay still needs the installed
Torch package for names it does not declare. For einops models, add
`tensor-shapes/pyrefly-einops-stubs`; omitting it can make a transform appear to
return the unchanged input shape rather than report a gap.

In fbsource, build `fbcode//pyrefly/tensor-shapes:torch-stubs-search-path` and
pass the output directory reported by `buck targets --show-output`—a Buck target
label is not itself a filesystem search path. Prefer an invoking skill's
build-and-check command when it supplies the complete environment. Confirm the
resolved paths with `pyrefly dump-config` and check a known-good example before
typing the target model.

**Python version:** the PEP 695/696 generics syntax (`class Net[D: IntVar]`,
type-param defaults) requires `--python-version 3.12` or later; the corpus runs
`3.13`.

To type-check the whole corpus (or a stack of edits) at once:

```bash
python3 tensor-shapes/run_all_shape_tests.py --mode auto
# In an internal checkout, including runtime tests:
python3 tensor-shapes/run_all_shape_tests.py --mode buck --include-runtime-tests
```

Run Pyrefly and require `0 errors`; `reveal_type` output is acceptable only
while probing and must not remain in the finished port. For a corpus
contribution, include the checker output in the work log. Otherwise report the
command and result concisely.

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
2. The root cause MUST have a boundary receipt from Step 4 (or trace to one —
   e.g., "input is bare" because the caller's dynamic container was documented
   in the parent module's receipt).

If any bare `assert_type` lacks a comment or boundary-receipt trail, go back and
either recover precision or document the boundary properly.

Record the full bare audit for a corpus contribution. For ordinary and
production ports, report each remaining bare boundary and its root cause
concisely:

```
## Bare assert_type audit
Total assert_type in forward bodies: ___
Shaped (assert_type(x, Tensor[...])): ___
Bare (assert_type(x, Tensor)): ___
Bare fraction: ___

Each bare:
- line N: var — root cause (boundary receipt: <module>.Step4)
- line M: var — root cause (boundary receipt: <module>.Step4)
```

## Measure shape coverage

For an ordinary or production port, prioritize **public-contract coverage**:
precisely shaped component tensor inputs and outputs divided by in-scope
component tensor boundaries. Report that one number plus each remaining gradual
boundary; do not count every local expression merely to manufacture a metric.

For a reference-corpus contribution, report three additional measures so source
evidence, casts, and inferred shapes do not get conflated:

- **Annotation preservation:** removed jaxtyping (or equivalent) sites whose
  rank, literal dimensions, and expressible relationships were preserved,
  divided by all removed annotation sites.
- **Source-evidence resolution:** shape-bearing comments, docstrings, runtime
  assertions, equations, and tests that were either represented in the native
  contract or explicitly corrected/refuted by a probe.
- **Inference coverage:** precise native shape expressions that do not rely on
  a cast, divided by all precise native shape expressions. Report precise casts,
  `cast(Any, ...)` boundaries, and bare-`Tensor` boundaries separately.

Treat 80–90% inference coverage as a solid corpus result when genuine dynamic
or third-party boundaries remain, but investigate each one; do not narrow
contracts merely to reach a clean check. Aim above 90% when the necessary stubs
or shape rules exist. Annotation preservation and source-evidence resolution
should normally be 100%; anything lower needs a specific reason.

## Compare against known patterns

Read the full `style_guide.md` now. The earlier model-index and example skim
was orientation; this pass compares the empirical draft against all known
patterns.

For each module in a corpus port, find the closest matching pattern and record
the comparison below. For other ports, apply relevant improvements without
printing the full table unless it clarifies a remaining gap:

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

For ordinary and production ports, report only: the check command and result,
public-contract coverage, remaining bare/cast boundaries with their
justifications, and any stub or shape-logic follow-ups. Do not dump the internal
gates.

For a reference-corpus contribution, fill the exhaustive template below before
reporting completion:

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
Each bare assert_type has comment + boundary-receipt trail: yes/no
Annotation preservation: ___ / ___ removed annotation sites = ___%
Source-evidence resolution: ___ / ___ comments/assertions/equations/tests = ___%
Inference coverage: ___ / ___ precise native shape expressions = ___%
Precise shape casts: ___. cast(Any) boundaries: ___. Bare Tensor boundaries: ___.
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
lighter deliverable. Fill it from non-precise Gate 1 classifications and Step 4
boundary receipts.

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
    from shape_extensions import Int, IntTuple, IntVar
```

*Runnable (the file is imported/executed):* ordinary annotations evaluate unless
postponed; PEP 695 bounds themselves are lazy. Either:
- add `from __future__ import annotations` and keep annotation-only names under
  `TYPE_CHECKING`; or
- import the runtime shape symbols at module top, using `Elements` for variadic
  splats that evaluate eagerly — see
  `examples/runtime/gptfast_sym_int_var_runnable.py`.

`Int` binds runtime ints; `IntVar` bounds scalar dimension parameters and
`IntTuple` bounds variadic/whole-shape parameters. Bare `*Bs` is preferred in
check-only files, stubs, and deferred annotations. Use `*Elements[Bs]` when an
annotation evaluates eagerly, because a bare `TypeVar` is not iterable. Deferred
annotations can still be forced by `typing.get_type_hints`, so test any runtime
annotation consumer explicitly.

For a codebase that already uses jaxtyping, decide whether the goal is native
syntax or static checking with minimal churn. `@static_jaxtyping("...")` can
declare the dimension names so Pyrefly checks the existing jaxtyping
annotations; use that alternative when preserving source annotations matters.
If native syntax is requested, translate deliberately rather than mechanically.
An empty shape string may have been used as an escape hatch even when the value
is not scalar, and `_` dimensions only promise rank. Use bare `Tensor` for
genuinely unconstrained inputs and `Tensor[[int, ...]]` when rank or fixed axes
are the useful contract.

If the model uses einops, include `pyrefly-einops-stubs` on the search path.
Those stubs compute shapes for `rearrange`, `reduce`, `repeat`, and `einsum` from
the pattern. Without them, a transform can incorrectly appear to preserve the
input shape. Patterns that require unsupported dynamic `axes_lengths` may still
return an unrefined `Tensor`; use a local cast justified by the pattern string,
for example `cast("Tensor[[B, H * W, C]]", rearrange(...))`.

**Runtime-compatible annotations:** `assert_type` is always a runtime no-op, but
its second argument and `cast`'s first argument are ordinary expressions and are
still evaluated. Quote shape expressions that should not execute. If an eagerly
evaluated expression needs arithmetic on a PEP 695 dimension parameter, use
`IntVar[N]` (often imported as `iv[N]`) because bare `N + 1` raises `TypeError`.
Import `shape_extensions` directly when runtime shape annotations are required.
