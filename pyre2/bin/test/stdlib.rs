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
class EnumMeta(type):
    def __iter__[EnumMemberT](self: type[EnumMemberT]) -> Iterator[EnumMemberT]: ...
class Enum(metaclass=EnumMeta): ...
class StrEnum(str, Enum): ...
class IntEnum(int, Enum): ...
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
def assert_type(x, y) -> None: ...

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

_T = TypeVar('_T', covariant=True)
class Iterable(Protocol[_T]):
    def __iter__(self) -> Iterator[_T]: ...
class Iterator(Iterable[_T], Protocol[_T]):
    def __next__(self) -> _T: ...
    def __iter__(self) -> Iterator[_T]: ...

_YieldT = TypeVar('_YieldT', covariant=True)
_SendT = TypeVar('_SendT', contravariant=True, default=None)
_ReturnT = TypeVar('_ReturnT', covariant=True, default=None)
class Generator(Iterator[_YieldT], Generic[_YieldT, _SendT, _ReturnT]):
    def __next__(self) -> _YieldT: ...
    def __iter__(self) -> Generator[_YieldT, _SendT, _ReturnT]: ...
    def send(self, value: _SendT, /) -> _YieldT: ...

class Awaitable(Protocol[_T]):
    def __await__(self) -> Generator[Any, Any, _T]: ...

class Coroutine(Awaitable[_ReturnT], Generic[_YieldT, _SendT, _ReturnT]):
    pass

class MutableSet(Iterable[_T]): ...

class Mapping[K, V](Iterable[K]): ...

class MutableMapping[K, V](Mapping[K, V]): ...

class TypeAliasType: ...

class NamedTuple(tuple[Any, ...]): ...
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

pub fn lookup_test_stdlib(module: ModuleName) -> Option<&'static str> {
    match module.as_str() {
        "builtins" => Some(BUILTINS),
        "typing" => Some(TYPING),
        "types" => Some(TYPES),
        "enum" => Some(ENUM),
        _ => None,
    }
}
