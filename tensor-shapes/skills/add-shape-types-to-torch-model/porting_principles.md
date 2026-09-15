# Porting Principles

## Priority order

1. **Faithfulness** — include everything from the original.
   Restructuring must preserve runtime behavior: never drop constructor
   parameters, never remove conditional branches, never add or remove
   layers. Functionally equivalent restructuring is fine (e.g., extracting
   modules from a list into individual attributes, converting a Sequential
   subclass to composition).
2. **Shape coverage** — preserve every rank, literal dimension, named equality,
   arithmetic relationship, and variadic prefix that the original annotation
   expressed. Use `assert_type` to verify inference, not just annotation
   fallback. Bare `Tensor` is reserved for values whose rank is genuinely
   unknowable, with comments explaining why.
3. **Identify blockers** — every place where shapes are lost should trace back
   to a specific gap or genuinely data-dependent shape. These inform what to
   build next.

**0 errors ≠ shapes tracked.** The checker silently accepts bare `Tensor`
where a shaped `Tensor[...]` is expected (annotation fallback). The ONLY
proof that shapes are inferred is `assert_type` inside forward methods.

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

Lack of inference is not lack of expressibility. When an operation such as a
third-party transform loses shape information, keep the precise surrounding
annotations and cast only the operation's result. Before accepting a gradual
type, check whether a stub improvement, a shape-preserving rewrite, a variadic
`IntTuple`, or an output-only named dimension can retain more information.

## Stub philosophy

Whether to change the stubs at all is the user's call (you confirmed it up
front). If they're taking the stubs as given, leave them alone — record the
untracked ops as gaps and move on.

If they're open to improvements, a stub gap is often a root cause worth fixing
rather than working around: a refined signature recovers the shape for *every*
model that uses that op, not just this one. That makes contributing a fix the
higher-leverage choice when it's in scope. When you do change a stub or shape DSL
function, make the fix general — capture the truth about the op, don't
special-case it for your model.
