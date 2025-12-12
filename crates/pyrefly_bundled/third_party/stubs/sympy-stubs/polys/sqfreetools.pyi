from typing import Any, Literal

def dup_sqf_p(f, K) -> bool: ...
def dmp_sqf_p(f, u, K) -> bool: ...
def dup_sqf_norm(
    f, K
) -> tuple[
    int,
    Any | list[Any],
    tuple[Any, list[Any]]
    | Any
    | tuple[list[list[Any]], list[Any]]
    | tuple[list[list[Any]] | Any | list[Any], list[Any]]
    | list[list[Any]]
    | list[Any],
]: ...
def dmp_sqf_norm(
    f, u, K
) -> (
    tuple[
        int,
        Any | list[Any],
        tuple[Any, list[Any]]
        | Any
        | tuple[list[list[Any]], list[Any]]
        | tuple[list[list[Any]] | Any | list[Any], list[Any]]
        | list[list[Any]]
        | list[Any],
    ]
    | tuple[
        int,
        Any | list[Any] | list[list[Any]],
        tuple[Any, list[Any]]
        | Any
        | tuple[list[list[Any]], list[Any]]
        | tuple[list[list[Any]] | Any | list[Any], list[Any]]
        | list[list[Any]]
        | list[Any],
    ]
): ...
def dmp_norm(
    f, u, K
) -> (
    tuple[Any, list[Any]]
    | tuple[list[list[Any]], list[Any]]
    | tuple[list[list[Any]] | Any | list[Any], list[Any]]
    | list[list[Any]]
    | list[Any]
): ...
def dup_gf_sqf_part(f, K): ...
def dmp_gf_sqf_part(f, u, K): ...
def dup_sqf_part(f, K) -> list[Any]: ...
def dmp_sqf_part(f, u, K) -> list[Any]: ...
def dup_gf_sqf_list(f, K, all=...) -> tuple[Any, list[Any]]: ...
def dmp_gf_sqf_list(f, u, K, all=...): ...
def dup_sqf_list(f, K, all=...) -> tuple[Any, list[Any]]: ...
def dup_sqf_list_include(f, K, all=...) -> list[tuple[list[Any], Literal[1]]] | list[tuple[Any, Literal[1]]]: ...
def dmp_sqf_list(f, u, K, all=...) -> tuple[Any, list[Any]]: ...
def dmp_sqf_list_include(
    f, u, K, all=...
) -> (
    list[tuple[list[Any], Literal[1]]] | list[tuple[Any, Literal[1]]] | list[tuple[list[list[Any]] | Any | list[Any], Literal[1]]]
): ...
def dup_gff_list(f, K) -> list[Any]: ...
def dmp_gff_list(f, u, K) -> list[Any]: ...
