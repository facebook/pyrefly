# pyrefly-torch-stubs

PyTorch type stubs with tensor shape information for Pyrefly.

This package is a PEP 561 stub-only distribution. It installs the
`torch-stubs` stub package so Pyrefly can discover shape-aware stubs for the
runtime `torch` package without replacing or shadowing PyTorch itself.

The package is versioned in lockstep with Pyrefly and depends on the matching
`pyrefly-shape-extensions` package.
