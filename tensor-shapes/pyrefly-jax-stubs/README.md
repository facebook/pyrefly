# Pyrefly JAX shape stubs

Shape-typed fixture stubs for a subset of JAX. This is a starting point for
working with the JAX core team rather than a complete model: it covers array
creation, broadcasting arithmetic, `matmul`, `reshape`, `transpose`, the
`axis`/`keepdims` reductions, and the elementwise activations in `jax.nn`.
Dimensions are modeled and dtypes are not, so adding dtypes later means modeling
JAX's own defaults (`float32` and `int32`) rather than copying the NumPy stubs.

Shape rules use the type-level DSL, `@type_shape_dsl_function`, exclusively. The
older `@shape_dsl_function` and `@uses_shape_dsl` mechanism is being replaced by
it, so nothing here should reach for that: a rule the type-level DSL cannot
express yet returns a gradual shape instead. Where a rule is imprecise for that
reason, the stub says so at the definition, along with whether a fix is
expected.

`TENSOR_SHAPES_CONTRIBUTING.md` at the repository root covers the workflow, and
`tensor-shapes/run_tests.py` runs the tests. Pyrefly checks the stubs themselves
and every `test/test_*.py`, and those same test files then run against real
JAX, so a stub that is self-consistent but wrong still fails.

Anything not listed above is simply absent rather than modeled loosely, so it is
reported as a missing attribute rather than inferred gradually.
