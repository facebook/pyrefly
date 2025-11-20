import enum
import sre_compile
import sre_constants
import sys
from _typeshed import MaybeNone, ReadableBuffer
from collections.abc import Callable, Iterator, Mapping
from types import GenericAlias
from typing import Any, AnyStr, Final, Generic, Literal, TypeVar, final, overload
from typing_extensions import TypeAlias, deprecated

__all__ = [
    "match",
    "fullmatch",
    "search",
    "sub",
    "subn",
    "split",
    "findall",
    "finditer",
    "compile",
    "purge",
    "escape",
    "error",
    "A",
    "I",
    "L",
    "M",
    "S",
    "X",
    "U",
    "ASCII",
    "IGNORECASE",
    "LOCALE",
    "MULTILINE",
    "DOTALL",
    "VERBOSE",
    "UNICODE",
    "Match",
    "Pattern",
]
if sys.version_info < (3, 13):
    __all__ += ["template"]

if sys.version_info >= (3, 11):
    __all__ += ["NOFLAG", "RegexFlag"]

if sys.version_info >= (3, 13):
    __all__ += ["PatternError"]

    PatternError = sre_constants.error

_T = TypeVar("_T")

# The implementation defines this in re._constants (version_info >= 3, 11) or
# sre_constants. Typeshed has it here because its __module__ attribute is set to "re".
class error(Exception):
    msg: str
    pattern: str | bytes | None
    pos: int | None
    lineno: int
    colno: int
    def __init__(self, msg: str, pattern: str | bytes | None = None, pos: int | None = None) -> None: ...

@final
class Match(Generic[AnyStr]):
    @property
    def pos(self) -> int:
        """The index into the string at which the RE engine started looking for a match."""
        ...
    @property
    def endpos(self) -> int:
        """The index into the string beyond which the RE engine will not go."""
        ...
    @property
    def lastindex(self) -> int | None:
        """The integer index of the last matched capturing group."""
        ...
    @property
    def lastgroup(self) -> str | None:
        """The name of the last matched capturing group."""
        ...
    @property
    def string(self) -> AnyStr:
        """The string passed to match() or search()."""
        ...

    # The regular expression object whose match() or search() method produced
    # this match instance.
    @property
    def re(self) -> Pattern[AnyStr]:
        """The regular expression object."""
        ...
    @overload
    def expand(self: Match[str], template: str) -> str:
        """Return the string obtained by doing backslash substitution on the string template, as done by the sub() method."""
        ...
    @overload
    def expand(self: Match[bytes], template: ReadableBuffer) -> bytes:
        """Return the string obtained by doing backslash substitution on the string template, as done by the sub() method."""
        ...
    @overload
    def expand(self, template: AnyStr) -> AnyStr:
        """Return the string obtained by doing backslash substitution on the string template, as done by the sub() method."""
        ...
    # group() returns "AnyStr" or "AnyStr | None", depending on the pattern.
    @overload
    def group(self, group: Literal[0] = 0, /) -> AnyStr:
        """
        group([group1, ...]) -> str or tuple.
        Return subgroup(s) of the match by indices or names.
        For 0 returns the entire match.
        """
        ...
    @overload
    def group(self, group: str | int, /) -> AnyStr | MaybeNone:
        """
        group([group1, ...]) -> str or tuple.
        Return subgroup(s) of the match by indices or names.
        For 0 returns the entire match.
        """
        ...
    @overload
    def group(self, group1: str | int, group2: str | int, /, *groups: str | int) -> tuple[AnyStr | MaybeNone, ...]:
        """
        group([group1, ...]) -> str or tuple.
        Return subgroup(s) of the match by indices or names.
        For 0 returns the entire match.
        """
        ...
    # Each item of groups()'s return tuple is either "AnyStr" or
    # "AnyStr | None", depending on the pattern.
    @overload
    def groups(self) -> tuple[AnyStr | MaybeNone, ...]:
        """
        Return a tuple containing all the subgroups of the match, from 1.

        default
          Is used for groups that did not participate in the match.
        """
        ...
    @overload
    def groups(self, default: _T) -> tuple[AnyStr | _T, ...]:
        """
        Return a tuple containing all the subgroups of the match, from 1.

        default
          Is used for groups that did not participate in the match.
        """
        ...
    # Each value in groupdict()'s return dict is either "AnyStr" or
    # "AnyStr | None", depending on the pattern.
    @overload
    def groupdict(self) -> dict[str, AnyStr | MaybeNone]:
        """
        Return a dictionary containing all the named subgroups of the match, keyed by the subgroup name.

        default
          Is used for groups that did not participate in the match.
        """
        ...
    @overload
    def groupdict(self, default: _T) -> dict[str, AnyStr | _T]:
        """
        Return a dictionary containing all the named subgroups of the match, keyed by the subgroup name.

        default
          Is used for groups that did not participate in the match.
        """
        ...
    def start(self, group: int | str = 0, /) -> int:
        """Return index of the start of the substring matched by group."""
        ...
    def end(self, group: int | str = 0, /) -> int:
        """Return index of the end of the substring matched by group."""
        ...
    def span(self, group: int | str = 0, /) -> tuple[int, int]:
        """For match object m, return the 2-tuple (m.start(group), m.end(group))."""
        ...
    @property
    def regs(self) -> tuple[tuple[int, int], ...]: ...  # undocumented
    # __getitem__() returns "AnyStr" or "AnyStr | None", depending on the pattern.
    @overload
    def __getitem__(self, key: Literal[0], /) -> AnyStr:
        """Return self[key]."""
        ...
    @overload
    def __getitem__(self, key: int | str, /) -> AnyStr | MaybeNone:
        """Return self[key]."""
        ...
    def __copy__(self) -> Match[AnyStr]: ...
    def __deepcopy__(self, memo: Any, /) -> Match[AnyStr]: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """See PEP 585"""
        ...

@final
class Pattern(Generic[AnyStr]):
    @property
    def flags(self) -> int:
        """The regex matching flags."""
        ...
    @property
    def groupindex(self) -> Mapping[str, int]:
        """A dictionary mapping group names to group numbers."""
        ...
    @property
    def groups(self) -> int:
        """The number of capturing groups in the pattern."""
        ...
    @property
    def pattern(self) -> AnyStr:
        """The pattern string from which the RE object was compiled."""
        ...
    @overload
    def search(self: Pattern[str], string: str, pos: int = 0, endpos: int = sys.maxsize) -> Match[str] | None:
        """
        Scan through string looking for a match, and return a corresponding match object instance.

        Return None if no position in the string matches.
        """
        ...
    @overload
    def search(self: Pattern[bytes], string: ReadableBuffer, pos: int = 0, endpos: int = sys.maxsize) -> Match[bytes] | None:
        """
        Scan through string looking for a match, and return a corresponding match object instance.

        Return None if no position in the string matches.
        """
        ...
    @overload
    def search(self, string: AnyStr, pos: int = 0, endpos: int = sys.maxsize) -> Match[AnyStr] | None:
        """
        Scan through string looking for a match, and return a corresponding match object instance.

        Return None if no position in the string matches.
        """
        ...
    @overload
    def match(self: Pattern[str], string: str, pos: int = 0, endpos: int = sys.maxsize) -> Match[str] | None:
        """Matches zero or more characters at the beginning of the string."""
        ...
    @overload
    def match(self: Pattern[bytes], string: ReadableBuffer, pos: int = 0, endpos: int = sys.maxsize) -> Match[bytes] | None:
        """Matches zero or more characters at the beginning of the string."""
        ...
    @overload
    def match(self, string: AnyStr, pos: int = 0, endpos: int = sys.maxsize) -> Match[AnyStr] | None:
        """Matches zero or more characters at the beginning of the string."""
        ...
    @overload
    def fullmatch(self: Pattern[str], string: str, pos: int = 0, endpos: int = sys.maxsize) -> Match[str] | None:
        """Matches against all of the string."""
        ...
    @overload
    def fullmatch(
        self: Pattern[bytes], string: ReadableBuffer, pos: int = 0, endpos: int = sys.maxsize
    ) -> Match[bytes] | None:
        """Matches against all of the string."""
        ...
    @overload
    def fullmatch(self, string: AnyStr, pos: int = 0, endpos: int = sys.maxsize) -> Match[AnyStr] | None:
        """Matches against all of the string."""
        ...
    @overload
    def split(self: Pattern[str], string: str, maxsplit: int = 0) -> list[str | MaybeNone]:
        """Split string by the occurrences of pattern."""
        ...
    @overload
    def split(self: Pattern[bytes], string: ReadableBuffer, maxsplit: int = 0) -> list[bytes | MaybeNone]:
        """Split string by the occurrences of pattern."""
        ...
    @overload
    def split(self, string: AnyStr, maxsplit: int = 0) -> list[AnyStr | MaybeNone]:
        """Split string by the occurrences of pattern."""
        ...
    # return type depends on the number of groups in the pattern
    @overload
    def findall(self: Pattern[str], string: str, pos: int = 0, endpos: int = sys.maxsize) -> list[Any]:
        """Return a list of all non-overlapping matches of pattern in string."""
        ...
    @overload
    def findall(self: Pattern[bytes], string: ReadableBuffer, pos: int = 0, endpos: int = sys.maxsize) -> list[Any]:
        """Return a list of all non-overlapping matches of pattern in string."""
        ...
    @overload
    def findall(self, string: AnyStr, pos: int = 0, endpos: int = sys.maxsize) -> list[AnyStr]:
        """Return a list of all non-overlapping matches of pattern in string."""
        ...
    @overload
    def finditer(self: Pattern[str], string: str, pos: int = 0, endpos: int = sys.maxsize) -> Iterator[Match[str]]:
        """
        Return an iterator over all non-overlapping matches for the RE pattern in string.

        For each match, the iterator returns a match object.
        """
        ...
    @overload
    def finditer(
        self: Pattern[bytes], string: ReadableBuffer, pos: int = 0, endpos: int = sys.maxsize
    ) -> Iterator[Match[bytes]]:
        """
        Return an iterator over all non-overlapping matches for the RE pattern in string.

        For each match, the iterator returns a match object.
        """
        ...
    @overload
    def finditer(self, string: AnyStr, pos: int = 0, endpos: int = sys.maxsize) -> Iterator[Match[AnyStr]]:
        """
        Return an iterator over all non-overlapping matches for the RE pattern in string.

        For each match, the iterator returns a match object.
        """
        ...
    @overload
    def sub(self: Pattern[str], repl: str | Callable[[Match[str]], str], string: str, count: int = 0) -> str:
        """Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl."""
        ...
    @overload
    def sub(
        self: Pattern[bytes],
        repl: ReadableBuffer | Callable[[Match[bytes]], ReadableBuffer],
        string: ReadableBuffer,
        count: int = 0,
    ) -> bytes:
        """Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl."""
        ...
    @overload
    def sub(self, repl: AnyStr | Callable[[Match[AnyStr]], AnyStr], string: AnyStr, count: int = 0) -> AnyStr:
        """Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl."""
        ...
    @overload
    def subn(self: Pattern[str], repl: str | Callable[[Match[str]], str], string: str, count: int = 0) -> tuple[str, int]:
        """Return the tuple (new_string, number_of_subs_made) found by replacing the leftmost non-overlapping occurrences of pattern with the replacement repl."""
        ...
    @overload
    def subn(
        self: Pattern[bytes],
        repl: ReadableBuffer | Callable[[Match[bytes]], ReadableBuffer],
        string: ReadableBuffer,
        count: int = 0,
    ) -> tuple[bytes, int]:
        """Return the tuple (new_string, number_of_subs_made) found by replacing the leftmost non-overlapping occurrences of pattern with the replacement repl."""
        ...
    @overload
    def subn(self, repl: AnyStr | Callable[[Match[AnyStr]], AnyStr], string: AnyStr, count: int = 0) -> tuple[AnyStr, int]:
        """Return the tuple (new_string, number_of_subs_made) found by replacing the leftmost non-overlapping occurrences of pattern with the replacement repl."""
        ...
    def __copy__(self) -> Pattern[AnyStr]: ...
    def __deepcopy__(self, memo: Any, /) -> Pattern[AnyStr]: ...
    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        ...
    def __hash__(self) -> int:
        """Return hash(self)."""
        ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """See PEP 585"""
        ...

# ----- re variables and constants -----

class RegexFlag(enum.IntFlag):
    A = sre_compile.SRE_FLAG_ASCII
    ASCII = A
    DEBUG = sre_compile.SRE_FLAG_DEBUG
    I = sre_compile.SRE_FLAG_IGNORECASE
    IGNORECASE = I
    L = sre_compile.SRE_FLAG_LOCALE
    LOCALE = L
    M = sre_compile.SRE_FLAG_MULTILINE
    MULTILINE = M
    S = sre_compile.SRE_FLAG_DOTALL
    DOTALL = S
    X = sre_compile.SRE_FLAG_VERBOSE
    VERBOSE = X
    U = sre_compile.SRE_FLAG_UNICODE
    UNICODE = U
    if sys.version_info < (3, 13):
        T = sre_compile.SRE_FLAG_TEMPLATE
        TEMPLATE = T
    if sys.version_info >= (3, 11):
        NOFLAG = 0

A: Final = RegexFlag.A
ASCII: Final = RegexFlag.ASCII
DEBUG: Final = RegexFlag.DEBUG
I: Final = RegexFlag.I
IGNORECASE: Final = RegexFlag.IGNORECASE
L: Final = RegexFlag.L
LOCALE: Final = RegexFlag.LOCALE
M: Final = RegexFlag.M
MULTILINE: Final = RegexFlag.MULTILINE
S: Final = RegexFlag.S
DOTALL: Final = RegexFlag.DOTALL
X: Final = RegexFlag.X
VERBOSE: Final = RegexFlag.VERBOSE
U: Final = RegexFlag.U
UNICODE: Final = RegexFlag.UNICODE
if sys.version_info < (3, 13):
    T: Final = RegexFlag.T
    TEMPLATE: Final = RegexFlag.TEMPLATE
if sys.version_info >= (3, 11):
    NOFLAG: Final = RegexFlag.NOFLAG
_FlagsType: TypeAlias = int | RegexFlag

# Type-wise the compile() overloads are unnecessary, they could also be modeled using
# unions in the parameter types. However mypy has a bug regarding TypeVar
# constraints (https://github.com/python/mypy/issues/11880),
# which limits us here because AnyStr is a constrained TypeVar.

# pattern arguments do *not* accept arbitrary buffers such as bytearray,
# because the pattern must be hashable.
@overload
def compile(pattern: AnyStr, flags: _FlagsType = 0) -> Pattern[AnyStr]: ...
@overload
def compile(pattern: Pattern[AnyStr], flags: _FlagsType = 0) -> Pattern[AnyStr]: ...
@overload
def search(pattern: str | Pattern[str], string: str, flags: _FlagsType = 0) -> Match[str] | None: ...
@overload
def search(pattern: bytes | Pattern[bytes], string: ReadableBuffer, flags: _FlagsType = 0) -> Match[bytes] | None: ...
@overload
def match(pattern: str | Pattern[str], string: str, flags: _FlagsType = 0) -> Match[str] | None: ...
@overload
def match(pattern: bytes | Pattern[bytes], string: ReadableBuffer, flags: _FlagsType = 0) -> Match[bytes] | None: ...
@overload
def fullmatch(pattern: str | Pattern[str], string: str, flags: _FlagsType = 0) -> Match[str] | None: ...
@overload
def fullmatch(pattern: bytes | Pattern[bytes], string: ReadableBuffer, flags: _FlagsType = 0) -> Match[bytes] | None: ...
@overload
def split(pattern: str | Pattern[str], string: str, maxsplit: int = 0, flags: _FlagsType = 0) -> list[str | MaybeNone]: ...
@overload
def split(
    pattern: bytes | Pattern[bytes], string: ReadableBuffer, maxsplit: int = 0, flags: _FlagsType = 0
) -> list[bytes | MaybeNone]: ...
@overload
def findall(pattern: str | Pattern[str], string: str, flags: _FlagsType = 0) -> list[Any]: ...
@overload
def findall(pattern: bytes | Pattern[bytes], string: ReadableBuffer, flags: _FlagsType = 0) -> list[Any]: ...
@overload
def finditer(pattern: str | Pattern[str], string: str, flags: _FlagsType = 0) -> Iterator[Match[str]]: ...
@overload
def finditer(pattern: bytes | Pattern[bytes], string: ReadableBuffer, flags: _FlagsType = 0) -> Iterator[Match[bytes]]: ...
@overload
def sub(
    pattern: str | Pattern[str], repl: str | Callable[[Match[str]], str], string: str, count: int = 0, flags: _FlagsType = 0
) -> str: ...
@overload
def sub(
    pattern: bytes | Pattern[bytes],
    repl: ReadableBuffer | Callable[[Match[bytes]], ReadableBuffer],
    string: ReadableBuffer,
    count: int = 0,
    flags: _FlagsType = 0,
) -> bytes: ...
@overload
def subn(
    pattern: str | Pattern[str], repl: str | Callable[[Match[str]], str], string: str, count: int = 0, flags: _FlagsType = 0
) -> tuple[str, int]: ...
@overload
def subn(
    pattern: bytes | Pattern[bytes],
    repl: ReadableBuffer | Callable[[Match[bytes]], ReadableBuffer],
    string: ReadableBuffer,
    count: int = 0,
    flags: _FlagsType = 0,
) -> tuple[bytes, int]: ...
def escape(pattern: AnyStr) -> AnyStr: ...
def purge() -> None: ...

if sys.version_info < (3, 13):
    if sys.version_info >= (3, 11):
        @deprecated("Deprecated since Python 3.11; removed in Python 3.13. Use `re.compile()` instead.")
        def template(pattern: AnyStr | Pattern[AnyStr], flags: _FlagsType = 0) -> Pattern[AnyStr]: ...  # undocumented
    else:
        def template(pattern: AnyStr | Pattern[AnyStr], flags: _FlagsType = 0) -> Pattern[AnyStr]: ...  # undocumented
