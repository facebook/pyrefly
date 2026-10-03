# pyrefly-numpy-stubs

NumPy type stubs with array shape information for Pyrefly.

This package is a PEP 561 stub-only distribution. It installs the
`numpy-stubs` stub package so Pyrefly can discover shape-aware stubs for the
runtime `numpy` package without replacing or shadowing NumPy itself.

The package is versioned in lockstep with Pyrefly and depends on the matching
`pyrefly-shape-extensions` package.

Static checks use NumPy's installed stubs for APIs that this package re-exports:

```bash
python3 tensor-shapes/pyrefly-numpy-stubs/run_pyrefly.py
```

The runner uses `~/.tensor-shapes-venv` by default. Set
`$TENSOR_SHAPES_VENV` to select another virtualenv, or pass `--python` with the
path to a virtualenv interpreter that has NumPy installed.
