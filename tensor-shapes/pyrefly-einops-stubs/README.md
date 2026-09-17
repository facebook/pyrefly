# Pyrefly einops shape stubs

This package is a PEP 561 stub-only distribution for `einops`. It provides
shape-aware annotations for `rearrange`, `reduce`, `repeat`, and `einsum` using
Pyrefly's type-level shape DSL.

The precise annotations currently target PyTorch. The runtime corpus also
exercises NumPy and JAX to validate that the shared einops pattern semantics
agree across backends.

TODO: Once Pyrefly supports a `MapShape` type operator, make the annotations
generic over array libraries while preserving the input's nominal array type.

Patterns that need named `axes_lengths` currently return a gradual shape; the
core evaluator already supports those lengths, and a future call-site bridge
will make them available to the shape rule.

Run the static tests with:

```bash
python3 tensor-shapes/pyrefly-einops-stubs/run_pyrefly.py --buck
```
