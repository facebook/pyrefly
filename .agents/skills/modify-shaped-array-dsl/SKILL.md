---
name: modify-shaped-array-dsl
description: Use when Pyrefly computes a wrong tensor shape (or is missing one that can't be expressed in a stub signature) and you need to add or fix a shape-DSL rule. Requires a Pyrefly checkout (fbsource or a clone); not usable from a pip/site-packages install.
---

You are modifying Pyrefly's tensor-shape DSL — the logic that computes the
output shape of a torch op from its input shapes.

**This skill points at code; it does not duplicate it. Read the files below to
learn the details.** What follows is only the map and the invariant you must
uphold (add a unit test).

## How the DSL works (the 30-second version)

A shape rule is a Python function in
`tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`, decorated with
`@type_shape_dsl_function`, that computes a type-level value using a restricted
Python subset. Public stubs call the function directly in return annotations,
for example `Tensor[reshape(Shape, Target)]`. The checker validates and
evaluates these calls; CPython treats the decorator as a runtime no-op.

There are two kinds of change. A **stub-only change** edits `_shapes.pyi` and the
public return annotation to compose existing operations. A **DSL-kernel change**
edits the Rust validator or evaluator to add a genuinely new operation; reach
for it only when the rule cannot be expressed by composing the existing DSL.

The type-level DSL implementation lives primarily in
`crates/pyrefly_types/src/type_level_dsl.rs`, with separate modules for type
system operations such as `MapIntTuples`. The symbolic dimension algebra it
uses lives in `crates/pyrefly_types/src/dimension.rs`.

### Preserve tensor types in numeric formulas

Integer/float arithmetic overloads can sometimes cause a tensor expression to
lose type information during overload selection. In tensor code, make formulas
explicitly floating-point when the result is intended to remain a tensor. For
example, multiply an exponent by `1.0`, or use a floating-point base such as
`2.0` instead of `2`. These equivalent forms steer overload selection toward
floating-point tensor arithmetic.

## You MUST unit-test the DSL logic, not just an example

An end-to-end example (`tensor-shapes/pyrefly-torch-stubs/examples`) exercises an op but does
**not** pin the algebra — off-by-one, ceiling-vs-floor, and zero/negative-dim
edge cases slip through. Add a targeted test that asserts the computed shape.

Tests live in **`pyrefly/lib/test/shape_dsl.rs`**. Read nearby type-level DSL
tests before adding one. Use `assert_type` when the expected type is expressible
and inline `# E: ...` markers for diagnostics. Tests for the retained V1 kernel
compatibility path are isolated in the `legacy` module and should not be used as
templates for new rules.

Run it:
- buck: `buck test pyrefly:pyrefly_library -- <test_name>`
- cargo: `cargo test <test_name>`

After a DSL-kernel (Rust) change you must rebuild before the checker sees it:
`buck build fbcode//pyrefly:pyrefly` (or `cargo build`). Stub-only `_shapes.pyi`
edits need no rebuild.

For any DSL-kernel or broader Pyrefly core change that modifies shape
manipulation semantics (as opposed to only editing torch/numpy stubs), the
default verification gate is:

```bash
tensor-shapes/run_all_shape_tests.py
```

This gate runs the shape-relevant Rust unit tests plus the non-runtime
tensor-shape corpus tests, and defaults to cargo with automatic buck fallback.
Use `--mode buck` or `--mode cargo` when you need to pin the backend, and add
`--include-runtime-tests` only when runtime coverage is relevant.

## Contributing the change

- **fbsource**: land as a diff.
- **clone**: open a PR against the stubs / Rust source in place.
