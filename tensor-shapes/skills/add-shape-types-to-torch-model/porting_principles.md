# Porting Principles

## Priority order

1. **Faithfulness** — include everything from the original.
   Restructuring must preserve runtime behavior: never drop constructor
   parameters, never remove conditional branches, never add or remove
   layers. Functionally equivalent restructuring is fine (e.g., extracting
   modules from a list into individual attributes, converting a Sequential
   subclass to composition).
2. **Shape coverage** — type with shapes as much as possible. Use `assert_type` to verify
   inference, not just annotation fallback. Shapeless parts get bare
   `Tensor` with comments explaining why.
3. **Identify blockers** — every place where shapes are lost should trace back
   to a specific gap or genuinely data-dependent shape. These inform what to
   build next.

**0 errors ≠ shapes tracked.** The checker silently accepts bare `Tensor`
where a shaped `Tensor[...]` is expected (annotation fallback). The ONLY
proof that shapes are inferred is `assert_type` inside forward methods.

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
