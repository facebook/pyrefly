# pyrefly-shape-extensions

Runtime helpers for Pyrefly tensor shape annotations.

This package provides the lightweight `shape_extensions` module used by
Pyrefly's tensor shape stubs. It defines runtime no-op versions of the shape
typing primitives so annotations such as `Tensor[B, T]`, `TypeVar("B")`, and
`assert_shape(x, (2, 3))` can be evaluated by Python while Pyrefly uses the
corresponding stubs for static shape checking.

The package is versioned in lockstep with Pyrefly.
