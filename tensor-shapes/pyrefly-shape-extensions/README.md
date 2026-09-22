# pyrefly-shape-extensions

Runtime helpers for Pyrefly tensor shape annotations.

This package provides the lightweight `shape_extensions` module used by
Pyrefly's tensor shape stubs. It defines runtime no-op versions of the shape
typing primitives so annotations such as `Tensor[B, T]`, `IntVar("B")`, and
`assert_shape(x.shape, (2, 3))` can be evaluated by Python while Pyrefly uses the
corresponding stubs for static shape checking.

`RegularNestedList[Shape, Domain]` represents regular (non-jagged) nested list
literals; for example, `[[1, 2], [3, 4]]` binds `Shape` to `[2, 2]`. Unsupported
containers and irregular literals use ordinary typing and any fallback overload
supplied by the consumer.

The package is versioned in lockstep with Pyrefly.
