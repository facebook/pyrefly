#!/usr/bin/env python3
"""
Validates which type constructs cause runtime errors when used with
string literal forward references in `|` union syntax.

Usage: python3.13 union_forward_ref_runtime_validation.py

Background:
  At runtime, `"Foo" | X` calls `str.__or__(X)` which returns NotImplemented,
  then Python tries `type(X).__ror__("Foo")`. If X's type doesn't implement
  `__ror__` for strings, a TypeError is raised.

  Types whose metaclass/type supports `__ror__` with strings (OK at runtime):
    - typing._SpecialForm (Union, Optional, Callable, LiteralString, Never, etc.)
    - typing._GenericAlias (List[int], Dict[str,int], Callable[[int],str])
    - typing.TypeVar, typing.ParamSpec
    - typing.Literal[...]

  Types whose metaclass/type does NOT support `__ror__` with strings (ERROR):
    - type (plain classes: int, str, list, user classes, TypedDict, NamedTuple, Protocol, etc.)
    - types.GenericAlias (list[int], dict[str,int] -- the builtin subscript)
    - NoneType (None)
    - typing.TypeAliasType (from `type X = ...`)
    - typing._AnyMeta (Any)
    - typing.TypeVarTuple
"""
import sys


def test(code: str) -> str:
    try:
        exec(code, {})
        return "OK"
    except TypeError:
        return "ERROR"
    except Exception as e:
        return f"OTHER ({type(e).__name__}: {e})"


cases: list[tuple[str, str, str]] = [
    # (description, code, expected)

    # === Plain classes (type metaclass) - all ERROR ===
    ("int", "'str' | int", "ERROR"),
    ("str (the class)", "'str' | str", "ERROR"),
    ("float", "'str' | float", "ERROR"),
    ("list (bare)", "'str' | list", "ERROR"),
    ("dict (bare)", "'str' | dict", "ERROR"),
    ("tuple (bare)", "'str' | tuple", "ERROR"),
    ("set (bare)", "'str' | set", "ERROR"),
    ("object", "'str' | object", "ERROR"),
    ("type", "'str' | type", "ERROR"),
    (
        "user class",
        "class C: pass\n'str' | C",
        "ERROR",
    ),
    (
        "user Generic class (bare)",
        "from typing import Generic, TypeVar\nT=TypeVar('T')\nclass C(Generic[T]): pass\n'str' | C",
        "ERROR",
    ),

    # === Special class definitions (TypedDict, NamedTuple, Protocol) - all ERROR ===
    (
        "TypedDict class",
        "from typing import TypedDict\nclass TD(TypedDict):\n  x: int\n'str' | TD",
        "ERROR",
    ),
    (
        "NamedTuple class",
        "from typing import NamedTuple\nclass NT(NamedTuple):\n  x: int\n'str' | NT",
        "ERROR",
    ),
    (
        "Protocol class",
        "from typing import Protocol\nclass P(Protocol):\n  def f(self) -> None: ...\n'str' | P",
        "ERROR",
    ),

    # === None ===
    ("None", "'str' | None", "ERROR"),
    ("None | 'str'", "None | 'str'", "ERROR"),

    # === Any ===
    ("Any", "from typing import Any; 'str' | Any", "ERROR"),

    # === Builtin parameterized generics (types.GenericAlias) - all ERROR ===
    ("list[int]", "'str' | list[int]", "ERROR"),
    ("dict[str,int]", "'str' | dict[str,int]", "ERROR"),
    ("tuple[int,...]", "'str' | tuple[int,...]", "ERROR"),
    ("set[int]", "'str' | set[int]", "ERROR"),
    ("frozenset[int]", "'str' | frozenset[int]", "ERROR"),
    ("type[int]", "'str' | type[int]", "ERROR"),

    # === typing module generics (typing._GenericAlias) - all OK ===
    ("List[int]", "from typing import List; 'str' | List[int]", "OK"),
    ("Dict[str,int]", "from typing import Dict; 'str' | Dict[str,int]", "OK"),
    ("Tuple[int,...]", "from typing import Tuple; 'str' | Tuple[int,...]", "OK"),
    ("Set[int]", "from typing import Set; 'str' | Set[int]", "OK"),
    ("FrozenSet[int]", "from typing import FrozenSet; 'str' | FrozenSet[int]", "OK"),
    ("Type[Any]", "from typing import Type, Any; 'str' | Type[Any]", "OK"),
    ("List (bare)", "from typing import List; 'str' | List", "OK"),
    ("Dict (bare)", "from typing import Dict; 'str' | Dict", "OK"),

    # === typing special forms - OK ===
    ("Union[int,float]", "from typing import Union; 'str' | Union[int,float]", "OK"),
    ("Optional[int]", "from typing import Optional; 'str' | Optional[int]", "OK"),
    ("LiteralString", "from typing import LiteralString; 'str' | LiteralString", "OK"),
    ("Never", "from typing import Never; 'str' | Never", "OK"),
    ("NoReturn", "from typing import NoReturn; 'str' | NoReturn", "OK"),

    # === TypeVar / ParamSpec / TypeVarTuple ===
    ("TypeVar", "from typing import TypeVar; T=TypeVar('T'); 'str' | T", "OK"),
    ("ParamSpec", "from typing import ParamSpec; P=ParamSpec('P'); 'str' | P", "OK"),
    (
        "TypeVarTuple",
        "from typing import TypeVarTuple; Ts=TypeVarTuple('Ts'); 'str' | Ts",
        "ERROR",
    ),

    # === Literal ===
    ("Literal[1]", "from typing import Literal; 'str' | Literal[1]", "OK"),

    # === Callable ===
    (
        "Callable[[int],str]",
        "from typing import Callable; 'str' | Callable[[int],str]",
        "OK",
    ),
    ("Callable (bare)", "from typing import Callable; 'str' | Callable", "OK"),

    # === User-defined generic class (parameterized) - OK ===
    (
        "C[int] (user generic)",
        "from typing import Generic, TypeVar\nT=TypeVar('T')\nclass C(Generic[T]): pass\n'str' | C[int]",
        "OK",
    ),

    # === type alias (PEP 695) ===
    ("type Alias = int", "type Alias = int\n'str' | Alias", "ERROR"),

    # === Ellipsis ===
    ("Ellipsis", "'str' | ...", "ERROR"),
]


def main() -> None:
    print(f"Python {sys.version}")
    print()
    print(f"{'Status':<12} {'Expected':<12} {'Description'}")
    print("-" * 70)

    failures = 0
    for desc, code, expected in cases:
        result = test(code)
        match = "PASS" if result == expected else "FAIL"
        if match == "FAIL":
            failures += 1
        print(f"  {result:<10} {expected:<10}  {match}  {desc}")

    print()
    if failures:
        print(f"{failures} unexpected result(s)")
        sys.exit(1)
    else:
        print("All results match expectations.")


if __name__ == "__main__":
    main()
