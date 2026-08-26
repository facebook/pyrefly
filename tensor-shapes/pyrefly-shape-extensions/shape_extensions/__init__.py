# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Library-agnostic shape typing primitives.

These definitions provide static shape information to Pyrefly while remaining
safe to evaluate in runtime annotations.
"""

import contextlib
import typing
from dataclasses import dataclass

__all__ = [
    "D",
    "Elements",
    "Int",
    "IntTuple",
    "IntTuples",
    "IntVar",
    "Index",
    "MapIntTuples",
    "ProxyMethod",
    "RegularNestedList",
    "SymbolicArithExpr",
    "assert_shape",
    "assert_raises",
    "broadcast",
    "defines_assert_shape",
    "gufunc_broadcast",
    "index_shape",
    "static_jaxtyping",
    "type_shape_dsl_function",
]


def _return_class(cls, params):
    return cls


def _return_int(cls, params):
    return int


def _patch_torch_if_available() -> None:
    try:
        import torch  # @manual
        import torch.nn as nn  # @manual
    except ImportError:
        return

    # Make torch types subscriptable at runtime so that annotations like
    # Tensor[B, T, N] or nn.Linear[In, Out] evaluate as no-ops instead of
    # crashing with "type is not subscriptable".
    subscriptable_classes = [
        torch.Tensor,
        nn.Embedding,
        nn.Linear,
        nn.ModuleList,
        # Convolution modules
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        # Pooling modules
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    ]
    for cls in subscriptable_classes:
        if not hasattr(cls, "__class_getitem__"):
            cls.__class_getitem__ = classmethod(_return_class)


_patch_torch_if_available()


def _patch_jax_if_available() -> None:
    try:
        import jax  # @manual
    except ImportError:
        return

    if hasattr(jax, "Array") and not hasattr(jax.Array, "__class_getitem__"):
        jax.Array.__class_getitem__ = classmethod(_return_class)


_patch_jax_if_available()


class IntTuple:
    """Tuple-valued shape annotation surface.

    In type positions, Pyrefly treats `IntTuple` as a whole shape whose runtime
    representation is `tuple[int, ...]`. At runtime, calling it coerces any
    iterable to a plain tuple.
    """

    def __new__(cls, iterable=()):
        return tuple(iterable)

    def __class_getitem__(cls, params):
        return cls


class IntTuples:
    """A tuple whose elements are `IntTuple` values."""

    def __new__(cls, iterable=()):
        return tuple(iterable)


class Elements:
    """Inverse of ``tuple[Unpack[S]]``: extracts the dimensions from an ``IntTuple``.

    In the Python typing spec, ``tuple[Unpack[Ts]]`` wraps a ``TypeVarTuple`` into a
    concrete tuple type. ``Elements[S]`` is the conceptual inverse: given an ``IntTuple``
    shape ``S``, ``*Elements[S]`` splices its dimensions into a shape position,
    e.g. ``Array[[*Elements[S], OUT], DType]``.

    This fills a gap in the current typing spec — there is no standard mechanism to
    decompose a variadic shape without a ``TypeVarTuple``. Pyrefly uses the ``.pyi``
    stub for type inference; this class exists so annotations evaluate without crashing
    at runtime.
    """

    def __init__(self, shape):
        self.shape = shape

    def __class_getitem__(cls, shape):
        return cls(shape)

    def __iter__(self):
        yield self

    def __repr__(self):
        return f"Elements[{self.shape!r}]"


_T = typing.TypeVar("_T")


class Int(typing.Generic[_T]):
    """Symbolic integer type for dimension values.

    At runtime this is a no-op generic class. The type checker uses the
    .pyi stub for shape inference.
    """

    pass


class Flag(typing.Generic[_T]):
    """Marker for a literal-preserving value that controls type-level evaluation."""

    pass


class Index:
    """Marker for an index value retained for type-level shape evaluation."""

    pass


class ProxyMethod(typing.Generic[_T]):
    """Type-checker marker for method forwarding annotations."""

    pass


# `TypeVar` defaults require Python 3.13 at runtime, so omit them on Python 3.12.
if typing.TYPE_CHECKING:
    _RegularNestedShape = typing.TypeVar(
        "_RegularNestedShape", bound=IntTuple, default=IntTuple, covariant=True
    )
    _Domain = typing.TypeVar(
        "_Domain", default=bool | int | float | complex, covariant=True
    )
else:
    _RegularNestedShape = typing.TypeVar(
        "_RegularNestedShape", bound=IntTuple, covariant=True
    )
    _Domain = typing.TypeVar("_Domain", covariant=True)


class RegularNestedList(typing.Generic[_RegularNestedShape, _Domain]):
    """A regular nested list literal whose scalar leaves belong to ``Domain``.

    Here, regular means the opposite of jagged or irregular: every sibling list
    has the same shape.

    Use this as a contextual parameter type for constructor-style APIs that
    accept nested Python lists and need to infer a shape type argument. It is a
    static marker, not the runtime type of an existing list value.

    A literal such as ``[[1, 2], [3, 4]]`` binds ``Shape`` to
    ``IntTuple[2, 2]``, while ``[[1, 2], [3]]`` is jagged and therefore not
    regular. Existing containers, starred literals, and statically jagged
    literals use ordinary typing instead of this marker.
    """

    def __class_getitem__(cls, params):
        return cls


@dataclass(frozen=True)
class SymbolicArithExpr:
    """Runtime representation of symbolic dimension arithmetic."""

    op: str
    args: tuple[typing.Any, ...]

    def __str__(self):
        if self.op == "var":
            return str(self.args[0])
        if self.op == "-" and len(self.args) == 2 and self.args[0] == 0:
            return f"-{_format_symbolic_arg(self.args[1])}"
        if len(self.args) == 2:
            return (
                f"{_format_symbolic_arg(self.args[0])} "
                f"{self.op} {_format_symbolic_arg(self.args[1])}"
            )
        return repr(self)

    def __add__(self, other):
        return SymbolicArithExpr("+", (self, other))

    def __radd__(self, other):
        return SymbolicArithExpr("+", (other, self))

    def __sub__(self, other):
        return SymbolicArithExpr("-", (self, other))

    def __rsub__(self, other):
        return SymbolicArithExpr("-", (other, self))

    def __mul__(self, other):
        return SymbolicArithExpr("*", (self, other))

    def __rmul__(self, other):
        return SymbolicArithExpr("*", (other, self))

    def __floordiv__(self, other):
        return SymbolicArithExpr("//", (self, other))

    def __rfloordiv__(self, other):
        return SymbolicArithExpr("//", (other, self))

    def __pow__(self, other):
        return SymbolicArithExpr("**", (self, other))

    def __rpow__(self, other):
        return SymbolicArithExpr("**", (other, self))

    def __neg__(self):
        return SymbolicArithExpr("-", (0, self))


def _format_symbolic_arg(value):
    if (
        isinstance(value, SymbolicArithExpr)
        and value.op != "var"
        and not (value.op == "-" and len(value.args) == 2 and value.args[0] == 0)
    ):
        return f"({value})"
    return str(value)


class D:
    """Wrap a shape type variable so Python can evaluate dimension arithmetic."""

    def __new__(cls, value):
        return SymbolicArithExpr("var", (value,))

    def __class_getitem__(cls, value):
        return cls(value)


def defines_assert_shape(fn: typing.Callable) -> typing.Callable:
    """
    Decorator that marks a function as an assert_shape helper.

    Used in order to allow custom assert_shape functions if necessary. A default
    version that works for tuple-like shapes is defined in the `assert_shape`
    function of this library.
    """
    return fn


def _check_runtime_shape(actual, shape):
    """Compare `actual` against `shape`, raising on a mismatch.

    Split out from `assert_shape` so the runtime comparison reads on its own.
    Kept private because the test harness counts calls to `assert_shape`, and a
    public helper reachable from it would be counted twice.
    """

    # Preserve legacy calls that pass an array object rather than its shape.
    if not isinstance(actual, tuple) and hasattr(actual, "shape"):
        actual_tuple = tuple(actual.shape)
    else:
        actual_tuple = tuple(actual)
    expected = tuple(shape)
    if any(isinstance(dim, SymbolicArithExpr) for dim in expected):
        if len(actual_tuple) != len(expected):
            raise AssertionError(
                f"expected rank {len(expected)} for shape {expected}, got shape {actual_tuple}"
            )
    elif actual_tuple != expected:
        raise AssertionError(f"expected shape {expected}, got {actual_tuple}")
    return actual


@contextlib.contextmanager
def assert_raises(
    expected: type[BaseException] | tuple[type[BaseException], ...],
) -> typing.Iterator[None]:
    """Assert that the body raises an exception of the expected type."""

    try:
        yield
    except expected:
        return
    raise AssertionError(f"expected {expected!r} to be raised")


@defines_assert_shape
def assert_shape(actual, shape, *, runtime=None):
    """
    At runtime, assert that a tuple-like shape has the expected value.

    Pyrefly will validate that the statically modeled shape matches, similar to
    `assert_type`.

    `shape` is the shape Pyrefly infers, and normally the library produces it too,
    so one argument pins both behaviors. `runtime` is for the cases where the two
    disagree: pass it the shape the library actually produces, and the runtime
    check uses it instead of `shape`. Two things need it:

    - An expression Pyrefly infers gradually: spell `shape` as `IntTuple` when it
      has no shape at all, or as a tuple such as `(int,)` when the rank is known
      and only a dimension is not.
    - A known bug, where Pyrefly infers a shape the library does not produce.
      Writing that wrong shape as `shape` documents it and makes the test fail
      once it is fixed, rather than leaving the discrepancy unrecorded.

    TODO(stroxler): for now, symbolic dimensions are skipped at runtime,
    so in the case of a symbolic `shape` the runtime validation is only checking
    the rank for those axes. But the static analysis will fully validate.
    """

    return _check_runtime_shape(actual, shape if runtime is None else runtime)


def index_shape(_shape: IntTuple, _index: typing.Any) -> IntTuple:
    """Runtime placeholder for Pyrefly's native shape-indexing intrinsic."""

    return IntTuple()


class MapIntTuples:
    """Map a unary type lambda over an ``IntTuples`` value.

    A forward map preserves each source shape::

        MapIntTuples[lambda S: Tensor[S], tuple[IntTuple[2], IntTuple[3, 4]]]

    A map used directly as a parameter annotation reverses that relationship, so
    passing ``(Tensor[IntTuple[2]], Tensor[IntTuple[3, 4]])`` infers the source
    as ``tuple[IntTuple[2], IntTuple[3, 4]]``. Each symbolic source may occur in
    only one parameter pattern, keeping inference unambiguous.

    At runtime this is a placeholder that does not inspect its arguments,
    because valid static sources such as ``Any`` and ``Never`` cannot be mapped
    as Python values.
    """

    def __class_getitem__(cls, params):
        return tuple


_F = typing.TypeVar("_F", bound=typing.Callable)


def type_shape_dsl_function(fn: _F) -> _F:
    """Runtime no-op for a user-defined type-level shape DSL function."""

    return fn


def static_jaxtyping(
    declaration: str,
) -> typing.Callable[[_F], _F]:
    """Declare the dimension names this function's jaxtyping annotations may use.

    ``declaration`` is a space-separated list of dimension names, where a
    leading ``*`` marks a variadic shape::

        @static_jaxtyping("batch channels *rest")
        def f(x: Float[Tensor, "batch channels"]) -> Float[Tensor, "*rest"]: ...

    Pyrefly reads the declaration to scope the dimensions and check the shape
    strings. Without it, jaxtyping annotations keep their ordinary ``Annotated``
    meaning and the array shape stays gradual. At runtime this is a no-op.
    """

    def decorate(fn: _F) -> _F:
        return fn

    return decorate


# `dsl` imports the public schema classes above, so defer this import until they exist.
from . import dsl as _dsl


@type_shape_dsl_function
def gufunc_broadcast(spec: str, shapes: IntTuples) -> IntTuple:
    """Compute the output shape described by a generalized ufunc signature."""

    return _dsl._gufunc_broadcast(spec, shapes)


@type_shape_dsl_function
def broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    spec = "(),()->()"
    shapes = _dsl.IntTuples((left, right))
    return gufunc_broadcast(spec, shapes)


class IntVar:
    """Symbolic variable with arithmetic support for tensor shape dimensions.

    Like typing.TypeVar but arithmetic operations (N + 1, N * 2, etc.)
    return self instead of raising TypeError. Setting
    __class__ = typing.TypeVar makes isinstance(x, typing.TypeVar)
    return True, so Generic[N] and TypedDict + Generic[N] both work.

    In pyrefly, shape_extensions.IntVar marks symbolic integer dimensions.
    """

    __class__ = typing.TypeVar

    def __init__(self, name: str, *, bound=None):
        self.__name__ = name
        self.name = name
        self.__bound__ = bound

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __floordiv__(self, other):
        return self

    def __typing_subst__(self, arg):
        return arg

    def has_default(self):
        return False
