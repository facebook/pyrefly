# Porting Principles

## Priority order

1. **Faithfulness** — include everything inside the selected port boundary and
   preserve runtime behavior. Casts are acceptable because they do not affect
   runtime. Do not narrow a public contract unless the user explicitly agrees
   it is correct. Annotation work does not redesign model construction,
   containers, control flow, or layers; report possible structural improvements
   for separate user approval.
2. **Component contracts** — prioritize precise inputs and outputs at module,
   function, and data-structure boundaries. In a real codebase these contracts
   deliver most of the value, even when a difficult implementation body remains
   partly gradual.
3. **Internal confidence** — infer and check internal shapes where reasonable,
   especially around shape-changing operations. This validates boundary types,
   but exhaustive internal precision is lower priority than sound component
   contracts outside the reference-corpus case.
4. **Contain gradual regions** — after reasonable diagnosis, allow a dynamic
   subsection to remain gradual. Re-establish the strongest justified shape with
   a cast or typed interface as soon as execution leaves that subsection, so
   graduality does not spread through downstream components.
5. **Identify blockers** — trace every remaining loss to a specific stub gap,
   dynamic Python boundary, third-party API, data-dependent shape, or checker
   limitation. These inform what to build next.

**0 errors does not mean shapes were inferred.** A gradual value may satisfy a
precise annotation. Verify important internal flows with `reveal_type` or an
`assert_type` on the directly inferred expression, before any cast or contextual
annotation. A justified cast can still make a valuable component contract; just
do not count it as inference.

## Shape values at ordinary API boundaries

Shape-aware stubs expose `.shape` as an `IntTuple`. When an ordinary API needs
a fixed-rank tuple such as `tuple[int, int]`, preserve both the runtime rank
check and the static type by destructuring first:

```python
height, width = image.shape
context = RecordContext(size=(height, width))
```

Do not silence the mismatch by passing the whole `IntTuple`, converting it with
`tuple(...)`, or selecting the first two dimensions. Those alternatives either
remain variadic to the checker or silently accept an unexpected higher-rank
value.

When an overloaded third-party API chooses its scalar return type for an array
input, normalize the result at that boundary only if the runtime contract is
known to be array-valued, for example `np.asarray(colormap(values))`. Do not
weaken downstream annotations or change scalar-versus-array behavior merely to
make the inferred type convenient.

## Preserve known rank and dimension names

Use bare `Tensor` only when the rank itself is genuinely unknown. If the rank
is known but one or more sizes cannot be related to the inputs, declare named
output-only `IntVar`s and use them in the return shape. Pyrefly accepts these
unconstrained variables and instantiates them as gradual dimensions at call
sites, while their names retain useful documentation in the declaration.

Keep every dimension that can be constrained bound to an input parameter. For
example, an operation that preserves channels but computes new spatial sizes
should return `Tensor[[C, NewH, NewW]]`, where `C` is captured from the input
and `NewH` and `NewW` appear only in the return type. Do not replace a bound
dimension with `int`, an output-only variable, or bare `Tensor` merely to make
the checker accept the implementation; use a narrow `cast` at an untyped
library boundary instead.

Lack of inference is not lack of expressibility. When a third-party transform
loses shape information, keep the precise surrounding annotations and cast only
the operation's result. Before accepting a gradual type, check whether a stub
improvement, a variadic `IntTuple`, or an output-only named dimension can retain
more information. If higher coverage requires changing runtime structure,
report that possible rewrite for separate user approval.

## Stub philosophy

Whether to change the stubs is resolved from the request and repository context
before Gate 0. If stubs are out of scope, leave them alone, record untracked ops,
and move on.

If they're open to improvements, a stub gap is often a root cause worth fixing
rather than working around: a refined signature recovers the shape for *every*
model that uses that op, not just this one. That makes contributing a fix the
higher-leverage choice when it's in scope. When you do change a stub or shape DSL
function, make the fix general — capture the truth about the op, don't
special-case it for your model.
