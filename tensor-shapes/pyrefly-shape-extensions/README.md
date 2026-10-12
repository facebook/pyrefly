# pyrefly-shape-extensions

Runtime helpers for Pyrefly tensor shape annotations.

This package provides the lightweight `shape_extensions` module used by
Pyrefly's tensor shape stubs. It defines runtime no-op versions of the shape
typing primitives so annotations such as `Tensor[B, T]`, `IntVar("B")`, and
`assert_shape(x.shape, (2, 3))` can be evaluated by Python while Pyrefly uses the
corresponding stubs for static shape checking.

`NamedInts` and `CaptureNamedInts` carry named axis lengths from `**kwargs`
into shape rules for einops-like APIs.

`RegularNestedList[Shape, Domain]` represents regular (non-jagged) nested list
literals; for example, `[[1, 2], [3, 4]]` binds `Shape` to `[2, 2]`. Unsupported
containers and irregular literals use ordinary typing and any fallback overload
supplied by the consumer.

`IntTupleOrList[Values]` is a stub-authoring parameter type for APIs that accept
integer tuples and lists. A direct, unstarred list literal such as `[2, 3]`
binds `Values` to `IntTuple[2, 3]`; an existing or starred list remains gradual,
while a direct literal containing a non-integer is rejected.

The package is versioned in lockstep with Pyrefly.

## Portable shape annotations

`Shaped[T, "..."]` is an alias for `typing.Annotated`, so other type checkers
read `T` and ignore the shape string. Inside a `@shape_vars` function or class,
Pyrefly also reads the string in annotations, casts, type aliases, and class
bases. Use `@shape_vars("")` for a literal shape with no declared dimensions.

Legacy type aliases cannot capture dimensions declared on an enclosing
`@shape_vars` function or class. For example, an `Alias: TypeAlias =
Shaped[Array, "[N]"]` inside a `@shape_vars("N")` definition reports that `N`
is not in scope for the alias. This is the same restriction that applies when
legacy type aliases capture ordinary enclosing type parameters. Use the
`Shaped` annotation directly in that scope; literal-only aliases are supported.

A defaulted `@shape_vars("N")` dimension after a `*Ts` class parameter is not
supported: Pyrefly reports the declaration and can mistake a trailing ordinary
type argument for the dimension. Pyrefly accepts
`@shape_vars("N", required=True) class Required[*Ts]` with an explicit shape,
as in `Required[int, str, 3]`; `Required[int, str]` still treats `str` as a
dimension. This explicit specialization is specific to Pyrefly, not a
portable workaround for other type checkers.

At runtime, `Shaped[Base, "..."]` can appear as a class base because
`Annotated` resolves to `Base`. Pyrefly reads its shape. In Pyright 1.1.414,
the base is rejected and inherited attributes and methods are inferred as
`Unknown` downstream. Do not rely on this base spelling when Pyright users
need inherited signatures.
