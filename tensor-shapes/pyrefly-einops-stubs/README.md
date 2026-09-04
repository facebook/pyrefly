# Pyrefly einops shape stubs

This package is a PEP 561 stub-only distribution for `einops`. It provides
shape-aware annotations for `rearrange`, `reduce`, `repeat`, and `einsum` using
Pyrefly's type-level shape DSL. The `einops.layers.torch` `Rearrange` and `Reduce`
modules preserve the constructor pattern and axis lengths across calls.

The precise annotations currently target PyTorch. The runtime corpus also
exercises NumPy and JAX to validate that the shared einops pattern semantics
agree across backends.

TODO: Once Pyrefly supports a `MapShape` type operator, make the annotations
generic over array libraries while preserving the input's nominal array type.

Literal `axes_lengths` keyword arguments are carried into the shape rules, so named
input splits and repeated output axes remain tracked. Dynamically unpacked mappings
preserve the known rank but use gradual dimensions for unknown axis lengths.

Run the static tests with:

```bash
python3 tensor-shapes/pyrefly-einops-stubs/run_pyrefly.py --buck
```
