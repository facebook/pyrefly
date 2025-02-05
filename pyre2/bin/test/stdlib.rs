/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A minimal stdlib that is meant to be approximately the same as from typeshed,
//! but only contain the bits that matter for our tests.

use crate::module::module_name::ModuleName;

static ENUM: &str = r#"
from typing import Any
_magic_enum_attr = property
class EnumMeta(type):
    def __iter__[EnumMemberT](self: type[EnumMemberT]) -> Iterator[EnumMemberT]: ...
class Enum(metaclass=EnumMeta):
    _name_: str
    _value_: Any
    @_magic_enum_attr
    def name(self) -> str: ...
    @_magic_enum_attr
    def value(self) -> Any: ...
class StrEnum(str, Enum):
    _value_: str
    @_magic_enum_attr
    def value(self) -> str: ...
class IntEnum(int, Enum):
    _value_: int
    @_magic_enum_attr
    def value(self) -> int: ...
class nonmember[_EnumMemberT]():
    value: _EnumMemberT
    def __init__(self, value: _EnumMemberT) -> None: ...
class member[_EnumMemberT]():
    value: _EnumMemberT
    def __init__(self, value: _EnumMemberT) -> None: ...
class Flag(Enum): ...
"#;

static BUILTINS: &str = r#"
from typing import Any, Iterable, Iterator, MutableMapping, MutableSet, Self
class object:
    def __init__(self) -> None: ...
    def __new__(cls) -> Self: ...
class str:
    def __iter__(self) -> Iterator[str]: ...
    def __add__(self, __x: str, /) -> str: ...
    def __iadd__(self, __value: str, /) -> str: ...
class bool(int): ...
class int:
    def __add__(self, __x: int, /) -> int: ...
    def __iadd__(self, __value: int, /) -> int: ...
    def __radd__(self, value: int, /) -> int: ...

class tuple: ...
class bytes: ...
class float: ...

class complex:
    def __add__(self, value: complex, /) -> complex: ...
    def __sub__(self, value: complex, /) -> complex: ...
    def __mul__(self, value: complex, /) -> complex: ...
    def __pow__(self, value: complex, mod: None = None, /) -> complex: ...
    def __truediv__(self, value: complex, /) -> complex: ...
    def __radd__(self, value: complex, /) -> complex: ...
    def __rsub__(self, value: complex, /) -> complex: ...
    def __rmul__(self, value: complex, /) -> complex: ...
    def __rpow__(self, value: complex, mod: None = None, /) -> complex: ...
    def __rtruediv__(self, value: complex, /) -> complex: ...


class list[T](Iterable[T]):
    def __init__(self) -> None: ...
    def append(self, object: T) -> None: ...
    def extend(self, object: list[T]) -> None: ...
    def __getitem__(self, index: int) -> T: ...
    def __setitem__(self, index: int, value: T) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __add__(self, __value: list[T], /) -> list[T]: ...
    def __iadd__(self, __value: list[T], /) -> list[T]: ...
class Ellipsis: ...
class dict[K, V](MutableMapping[K, V]):
    def __getitem__(self, key: K) -> V: ...
class set[T](MutableSet[T]): ...
class slice[_StartT, _StopT, _StepT]: ...
class BaseException: ...
class Exception(BaseException): ...
class BaseExceptionGroup[T](BaseException): ...
class ExceptionGroup[T](BaseExceptionGroup[T]): ...
# Note that type does *not* inherit from Generic in the real builtins stub.
class type:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def mro(self) -> list[type]: ...
# TODO: overload for slice, tuple should be Sequence[T]
class tuple[T](Iterable[T]):
    def __getitem__(self, index: int) -> T: ...

_ClassInfo = type | tuple[_ClassInfo, ...]
def isinstance(obj: object, class_or_tuple: _ClassInfo, /) -> bool: ...
def issubclass(cls: type, class_or_tuple: _ClassInfo, /) -> bool: ...

class staticmethod: pass
class classmethod: pass
class property: pass
"#;

static TYPING: &str = r#"
class _SpecialForm: ...
Optional: _SpecialForm
Literal: _SpecialForm
Final: _SpecialForm
class Any: ...
LiteralString: _SpecialForm
Union: _SpecialForm
Tuple: _SpecialForm
Type: _SpecialForm
TypeAlias: _SpecialForm
TypeGuard: _SpecialForm
TypeIs: _SpecialForm
Unpack: _SpecialForm
Self: _SpecialForm
Callable: _SpecialForm
Generic: _SpecialForm
Protocol: _SpecialForm
Never: _SpecialForm
NoReturn: _SpecialForm
Annotated: _SpecialForm
TypedDict: _SpecialForm
Required: _SpecialForm
NotRequired: _SpecialForm
ReadOnly: _SpecialForm
Concatenate: _SpecialForm

def assert_type(x, y) -> None: ...

def final(f: _T) -> _T: ...

# This use of `final` matches typeshed stubs, and we have a test that relies on
# it to ensure that our handling of `@final` does not lead to recursion issues
# for type variables.
@final
class TypeVar:
    def __init__(self, name: str) -> None: ...

class ParamSpec:
    def __init__(self, name: str) -> None: ...

class ParamSpecArgs:
    def __init__(self, origin: ParamSpec) -> None: ...

class ParamSpecKwargs:
    def __init__(self, origin: ParamSpec) -> None: ...

class TypeVarTuple:
    def __init__(self, name: str) -> None: ...

def reveal_type(obj, /):
    return obj

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
class Iterable(Protocol[_T_co]):
    def __iter__(self) -> Iterator[_T_co]: ...
class Iterator(Iterable[_T_co], Protocol[_T_co]):
    def __next__(self) -> _T_co: ...
    def __iter__(self) -> Iterator[_T_co]: ...

_YieldT = TypeVar('_YieldT', covariant=True)
_SendT = TypeVar('_SendT', contravariant=True, default=None)
_ReturnT = TypeVar('_ReturnT', covariant=True, default=None)
class Generator(Iterator[_YieldT], Generic[_YieldT, _SendT, _ReturnT]):
    def __next__(self) -> _YieldT: ...
    def __iter__(self) -> Generator[_YieldT, _SendT, _ReturnT]: ...
    def send(self, value: _SendT, /) -> _YieldT: ...

class AsyncIterable(Protocol[_T_co]):
    def __aiter__(self) -> AsyncIterator[_T_co]: ...

class AsyncIterator(AsyncIterable[_T_co], Protocol[_T_co]):
    def __anext__(self) -> Awaitable[_T_co]: ...
    def __aiter__(self) -> AsyncIterator[_T_co]: ...

class AsyncGenerator(AsyncIterator[_YieldT], Protocol[_YieldT, _SendT]):
    def __anext__(self) -> Coroutine[Any, Any, _YieldT]: ...
    def asend(self, value: _SendT, /) -> Coroutine[Any, Any, _YieldT]: ...

class Awaitable(Protocol[_T_co]):
    def __await__(self) -> Generator[Any, Any, _T_co]: ...

class Coroutine(Awaitable[_ReturnT], Generic[_YieldT, _SendT, _ReturnT]):
    pass

class MutableSet(Iterable[_T_co]): ...

class Mapping[K, V](Iterable[K]): ...

class MutableMapping[K, V](Mapping[K, V]): ...

class TypeAliasType: ...

class NamedTuple(tuple[Any, ...]): ...

_F = TypeVar('_F', bound=Callable[..., Any])
def overload(func: _F) -> _F: ...

class Sequence[T]: ...

class MutableSequence(Sequence[_T]): ...
"#;

static TYPES: &str = r#"
from typing import Any, Callable
class EllipsisType: ...
class NoneType: ...
class TracebackType: ...
class CodeType: ...
class FunctionType:
    __code__: CodeType
class MethodType:
    __self__: object
"#;

static DATACLASSES: &str = r#"
from typing import overload, Any, Callable, TypeVar

_T = TypeVar('_T')

@overload
def dataclass(cls: type[_T], /) -> type[_T]: ...

@overload
def dataclass(
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> Callable[[type[_T]], type[_T]]: ...

def field(*args, **kwargs) -> Any: ...

class KW_ONLY: ...
"#;

static SYS: &str = r#"
platform: str
version_info: tuple[int, int, int, str, int]
"#;

static STDLIB: &[(&str, &str)] = &[
    ("builtins", BUILTINS),
    ("typing", TYPING),
    ("types", TYPES),
    ("enum", ENUM),
    ("dataclasses", DATACLASSES),
    ("sys", SYS),
];

pub fn lookup_test_stdlib(module: ModuleName) -> Option<&'static str> {
    STDLIB.iter().find_map(|(name, source)| {
        if *name == module.as_str() {
            Some(*source)
        } else {
            None
        }
    })
}
