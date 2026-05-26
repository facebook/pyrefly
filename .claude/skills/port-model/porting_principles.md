# Porting Principles

## Why we port

Ports demonstrate what happens when you write a real PyTorch model with tensor
shape types. They prove real-world utility. If we exclude features or simplify
models, we're proving nothing — the hard parts are exactly where value needs
to be demonstrated.

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

## Fixture philosophy

Fixtures serve all code for all time. They should capture the truth about each
op, not just what the current models need. When you fix a stub, make the fix
general — don't special-case it for your model.
