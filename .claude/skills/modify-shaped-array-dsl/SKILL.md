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

A shape rule has two pieces. An **IR function** is a Python function in
`tensor-shapes/pyrefly-torch-stubs/torch-stubs/_shapes.pyi`, decorated `@shape_dsl_function`, that
computes shapes using a restricted Python subset (arithmetic `+ - * // %`,
comprehensions, `if`, a few builtins, `ShapedArray`). It is *traced*, not
executed by CPython. A library stub attaches it to an op with
`@uses_shape_dsl(ir_fn)` (e.g. `tensor-shapes/pyrefly-torch-stubs/torch-stubs/linalg.pyi`); the
stub's declared return is a "fixture" (gives the base `Tensor`/tuple structure)
and the IR function fills in the actual dims.

There are two kinds of change. A **stub-only change** edits `_shapes.pyi` to add
or fix an IR function composing existing arithmetic — no rebuild needed, and it
covers the large majority of cases. A **DSL-kernel change** edits the Rust
evaluator to add a genuinely new primitive operation; reach for it only when the
arithmetic you need cannot be expressed by composing what `_shapes.pyi` already
has.

How the decorator is traced into the checker (follow this chain if you need to
touch the wiring): `uses_shape_dsl`/`shape_dsl_function` are recognized in
`pyrefly/lib/export/special.rs`; the binding step extracts the IR name in
`pyrefly/lib/binding/function.rs`; the solve step resolves it to a
`ShapeTransform` in `pyrefly/lib/alt/function.rs`; it's applied at call sites via
`alt/callable.rs` (`evaluate`). The Rust evaluator and all arithmetic primitives
live in **one file**, `crates/pyrefly_types/src/meta_shape_dsl.rs` (the binop
arithmetic is `eval_binop`); the symbolic dim algebra it calls
(`SizeExpr::add/sub/mul/floor_div`) is in `crates/pyrefly_types/src/dimension.rs`.

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

Tests live in **`pyrefly/lib/test/shape_dsl.rs`**. Read it before adding one —
`shape_dsl_env()` defines IR functions in a synthetic `my_shapes.pyi` and
consumers in `my_lib.pyi`, and `testcase!` blocks assert results with
`assert_type(fn(args), Literal[n])`. Copy an existing case
(`test_uses_shape_dsl_cross_function_call` is a good template). For pure
arithmetic, an `int -> int` IR function with `assert_type(..., Literal[n])`
tests the primitives directly without needing `ShapedArray` fixtures. Use inline
`# E: ...` markers to assert compile-time DSL diagnostics.

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
