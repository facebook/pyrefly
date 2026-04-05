#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Inline type table indices in a CinderX module report for debugging.

Reads a per-module CinderX JSON report (types/<module>.json) and prints
a human-readable version where every type-index reference is replaced
with the actual type structure, recursively.

This is a stand-alone script with no dependencies.

Usage:
    python3 pyrefly/lib/report/cinderx/view_types.py <path/to/types/module.json>
"""

from __future__ import annotations

import json
import sys
from typing import Any


def inline_type(type_table: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    """Recursively inline a type table entry, replacing indices with structures."""
    entry = type_table[idx]
    kind = entry["kind"]

    if kind == "class":
        result: dict[str, Any] = {
            "kind": kind,
            "qname": entry["qname"],
        }
        if entry.get("args"):
            result["args"] = [inline_type(type_table, a) for a in entry["args"]]
        if entry.get("traits"):
            result["traits"] = entry["traits"]
        return result

    if kind == "callable":
        result: dict[str, Any] = {
            "kind": kind,
            "params": [inline_type(type_table, p) for p in entry["params"]],
            "return_type": inline_type(type_table, entry["return_type"]),
        }
        if entry.get("defining_func"):
            result["defining_func"] = entry["defining_func"]
        return result

    if kind == "other_form":
        result = {
            "kind": kind,
            "qname": entry["qname"],
        }
        if entry.get("args"):
            result["args"] = [inline_type(type_table, a) for a in entry["args"]]
        return result

    if kind == "bound_method":
        result = {
            "kind": kind,
            "self_type": inline_type(type_table, entry["self_type"]),
            "func_type": inline_type(type_table, entry["func_type"]),
        }
        if entry.get("defining_class"):
            result["defining_class"] = entry["defining_class"]
        return result

    if kind == "variable":
        result = {
            "kind": kind,
            "name": entry["name"],
        }
        if entry.get("bounds"):
            result["bounds"] = [inline_type(type_table, b) for b in entry["bounds"]]
        return result

    if kind == "literal":
        return {
            "kind": kind,
            "value": entry["value"],
            "promoted_type": inline_type(type_table, entry["promoted_type"]),
        }

    # Unknown kind: return as-is minus the hash.
    return {k: v for k, v in entry.items() if k != "hash"}


def format_type(ty: dict[str, Any]) -> str:
    """Format an inlined type as a compact string."""
    kind = ty["kind"]

    if kind == "literal":
        promoted = format_type(ty["promoted_type"])
        return f"Literal[{ty['value']}] -> {promoted}"

    if kind == "class":
        s = ty["qname"]
        if ty.get("args"):
            args = ", ".join(format_type(a) for a in ty["args"])
            s += f"[{args}]"
        if ty.get("traits"):
            s += f" <{', '.join(ty['traits'])}>"
        return s

    if kind == "other_form":
        s = ty["qname"]
        if ty.get("args"):
            args = ", ".join(format_type(a) for a in ty["args"])
            s += f"[{args}]"
        return s

    if kind == "callable":
        params = ", ".join(format_type(p) for p in ty["params"])
        ret = format_type(ty["return_type"])
        df = ty.get("defining_func")
        if df:
            return f"{df}({params}) -> {ret}"
        return f"({params}) -> {ret}"

    if kind == "bound_method":
        self_ty = format_type(ty["self_type"])
        func_ty = format_type(ty["func_type"])
        dc = ty.get("defining_class", "?")
        return f"BoundMethod[{dc}]({self_ty}, {func_ty})"

    if kind == "variable":
        s = ty["name"]
        if ty.get("bounds"):
            bounds = ", ".join(format_type(b) for b in ty["bounds"])
            s += f": {bounds}"
        return s

    return json.dumps(ty)


def format_location(loc: dict[str, Any]) -> str:
    """Format a source location as 'line N, col S-E'."""
    pos = loc.get("loc", loc)
    line = pos.get("start_line", "?")
    col_start = pos.get("start_col", pos.get("col", "?"))
    col_end = pos.get("end_col")
    if col_end is not None and col_end != col_start:
        return f"{line}:{col_start}-{col_end}"
    return f"{line}:{col_start}"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/types/module.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        report = json.load(f)

    type_table = report["type_table"]
    locations = report["locations"]

    for loc in locations:
        pos = format_location(loc)
        type_idx = loc["type"]
        ty = inline_type(type_table, type_idx)
        print(f"  {pos}: [{type_idx}] {format_type(ty)}")

        if loc.get("unnarrowed_type") is not None:
            unnarrowed_idx = loc["unnarrowed_type"]
            unnarrowed = inline_type(type_table, unnarrowed_idx)
            mismatch = loc.get("is_narrowed_mismatch", False)
            print(f"    unnarrowed: [{unnarrowed_idx}] {format_type(unnarrowed)}")
            print(f"    mismatch: {mismatch}")

        if loc.get("contextual_type") is not None:
            ctx_idx = loc["contextual_type"]
            ctx = inline_type(type_table, ctx_idx)
            print(f"    contextual: [{ctx_idx}] {format_type(ctx)}")


if __name__ == "__main__":
    main()
