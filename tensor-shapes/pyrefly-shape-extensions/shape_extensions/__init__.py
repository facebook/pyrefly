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
import keyword
import typing
from dataclasses import dataclass

__all__ = [
    "Elements",
    "Int",
    "IntListLiteral",
    "IntTuple",
    "IntTupleOrList",
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
    """Explicit inverse of ``tuple[Unpack[...]]`` for shape type variables.

    In the Python typing spec, ``tuple[Unpack[Ts]]`` wraps a ``TypeVarTuple`` into a
    concrete tuple type. Pyrefly shapes accept the inverse operation directly: given
    an ``IntTuple`` shape ``S``, a bare ``*S`` splat splices its dimensions into a
    shape position, e.g. ``Array[[*S, OUT], DType]``.

    ``Elements[S]`` is the explicit spelling of that same splat. It is optional in
    annotations, which Pyrefly checks identically either way, but it is necessary
    when the annotation is evaluated at runtime: unpacking a bare ``TypeVar`` raises
    ``TypeError``, while ``Elements`` is iterable.
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
    _IntTupleT = typing.TypeVar(
        "_IntTupleT", bound=IntTuple, default=IntTuple, covariant=True
    )
    _Domain = typing.TypeVar(
        "_Domain", default=bool | int | float | complex, covariant=True
    )
else:
    _IntTupleT = typing.TypeVar("_IntTupleT", bound=IntTuple, covariant=True)
    _Domain = typing.TypeVar("_Domain", covariant=True)


class IntListLiteral(typing.Generic[_IntTupleT]):
    """A direct integer-list literal whose values are captured as an ``IntTuple``."""

    def __class_getitem__(cls, params):
        return list[int]


IntTupleOrList: typing.TypeAlias = typing.Union[
    _IntTupleT, IntListLiteral[_IntTupleT], list[int]
]


class RegularNestedList(typing.Generic[_IntTupleT, _Domain]):
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


def static_jaxtyping(declaration: str) -> typing.Callable[[_F], _F]:
    """Declare the dimension names a function or class may use with jaxtyping.

    ``declaration`` is a space-separated list of dimension names, where a
    leading ``*`` marks a variadic shape::

        @static_jaxtyping("batch channels *rest")
        def f(x: Float[Tensor, "batch channels"]) -> Float[Tensor, "*rest"]: ...

    Pyrefly reads the declaration to scope the dimensions and check the shape
    strings. Without it, jaxtyping annotations keep their ordinary ``Annotated``
    meaning and the array shape stays gradual. At runtime functions are unchanged,
    while classes become subscriptable so their static shape arguments can appear
    in evaluated annotations.
    """

    declared_names: list[str] = []
    for token in declaration.split():
        name = token.removeprefix("*")
        if (
            name != "_"
            and name.isidentifier()
            and not keyword.iskeyword(name)
            and name not in declared_names
        ):
            declared_names.append(name)
    declared_dimensions = len(declared_names)

    def decorate(value: _F) -> _F:
        if not isinstance(value, type):
            return value

        original_descriptor = value.__dict__.get("__class_getitem__")

        def class_getitem(cls, params):
            args = params if isinstance(params, tuple) else (params,)
            ordinary = args
            if cls is value:
                for _ in range(declared_dimensions):
                    if not ordinary:
                        break
                    argument = ordinary[-1]
                    is_dimension = isinstance(argument, (int, IntVar)) or (
                        isinstance(argument, list)
                        and all(isinstance(dim, (int, IntVar)) for dim in argument)
                    )
                    if not is_dimension:
                        break
                    ordinary = ordinary[:-1]

            delegate_descriptor = original_descriptor
            if delegate_descriptor is None:
                for base in cls.__mro__:
                    candidate = base.__dict__.get("__class_getitem__")
                    if candidate is None:
                        continue
                    function = getattr(candidate, "__func__", candidate)
                    if getattr(function, "__static_jaxtyping__", False):
                        continue
                    delegate_descriptor = candidate
                    break

            if not ordinary or delegate_descriptor is None:
                return cls
            ordinary_params = ordinary[0] if len(ordinary) == 1 else ordinary
            delegate = delegate_descriptor.__get__(None, cls)
            return delegate(ordinary_params)

        class_getitem.__static_jaxtyping__ = True
        value.__class_getitem__ = classmethod(class_getitem)
        return value

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

    Subscripting with a type variable, ``IntVar[N]``, is a static no-op that wraps
    a PEP 695 ``TypeVar`` for runtime evaluation: a bare ``N + 1`` raises
    ``TypeError`` when the annotation is evaluated, while ``IntVar[N] + 1`` builds
    a symbolic expression. For readability in nontrivial formulas, alias the import
    (``from shape_extensions import IntVar as iv``) and write ``iv[N]``.
    """

    __class__ = typing.TypeVar

    def __class_getitem__(cls, value):
        return SymbolicArithExpr("var", (value,))

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
