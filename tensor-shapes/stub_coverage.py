# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check partial stub packages against the libraries they shadow."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import io
import json
import keyword
import sys
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemberTarget:
    stub_module: str
    stub_class: str
    runtime_module: str
    runtime_class: str


@dataclass(frozen=True)
class Config:
    runtime_package: str
    stub_directory: Path
    skip_modules: frozenset[str]
    module_excludes: dict[str, frozenset[str]]
    member_excludes: dict[str, frozenset[str]]
    member_targets: tuple[MemberTarget, ...]


_CONFIG_KEYS = frozenset(
    {
        "runtime_package",
        "stub_directory",
        "skip_modules",
        "module_excludes",
        "member_excludes",
        "member_targets",
    }
)
_MEMBER_TARGET_KEYS = frozenset(
    {"stub_module", "stub_class", "runtime_module", "runtime_class"}
)


def _validate_keys(
    values: dict[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    context: str,
) -> None:
    missing = required - values.keys()
    if missing:
        raise ValueError(
            f"{context} is missing required keys: {', '.join(sorted(missing))}"
        )
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(sorted(unknown))}")


def _load_member_targets(value: Any) -> tuple[MemberTarget, ...]:
    if not isinstance(value, list):
        raise ValueError("member_targets must be an array of tables")
    result = []
    for index, target in enumerate(value):
        if not isinstance(target, dict):
            raise ValueError(f"member_targets[{index}] must be a table")
        _validate_keys(
            target,
            allowed=_MEMBER_TARGET_KEYS,
            required=_MEMBER_TARGET_KEYS,
            context=f"member_targets[{index}]",
        )
        result.append(
            MemberTarget(
                stub_module=target["stub_module"],
                stub_class=target["stub_class"],
                runtime_module=target["runtime_module"],
                runtime_class=target["runtime_class"],
            )
        )
    return tuple(result)


def load_config(path: Path) -> Config:
    """Load paths relative to a package's coverage configuration."""
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    _validate_keys(
        raw,
        allowed=_CONFIG_KEYS,
        required=frozenset({"runtime_package", "stub_directory"}),
        context=f"configuration {path}",
    )
    return Config(
        runtime_package=raw["runtime_package"],
        stub_directory=path.parent / raw["stub_directory"],
        skip_modules=frozenset(raw.get("skip_modules", [])),
        module_excludes={
            module: frozenset(names)
            for module, names in raw.get("module_excludes", {}).items()
        },
        member_excludes={
            member: frozenset(names)
            for member, names in raw.get("member_excludes", {}).items()
        },
        member_targets=_load_member_targets(raw.get("member_targets", [])),
    )


def _starts_parameterized_type_alias(
    tokens: list[tokenize.TokenInfo], index: int
) -> bool:
    """Return whether `tokens[index]` starts `type Alias[...]`."""
    following = (
        token
        for token in tokens[index + 1 :]
        if token.type not in {tokenize.COMMENT, tokenize.NL}
    )
    alias = next(following, None)
    bracket = next(following, None)
    return (
        alias is not None
        and alias.type == tokenize.NAME
        and not keyword.iskeyword(alias.string)
        and bracket is not None
        and bracket.string == "["
    )


def module_name(stub_directory: Path, path: Path, runtime_package: str) -> str:
    """Convert a path in a PEP 561 `*-stubs` tree to its runtime module."""
    relative = path.relative_to(stub_directory)
    parts = list(relative.parts)
    if parts[-1] == "__init__.pyi":
        parts.pop()
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join((runtime_package, *parts))


def _statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Flatten conditional declarations while preserving lexical scope."""
    result = []
    for statement in statements:
        result.append(statement)
        if isinstance(statement, ast.If):
            result.extend(_statements(statement.body))
            result.extend(_statements(statement.orelse))
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            result.extend(_statements(statement.body))
            result.extend(_statements(statement.orelse))
            result.extend(_statements(statement.finalbody))
            for handler in statement.handlers:
                result.extend(_statements(handler.body))
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            result.extend(_statements(statement.body))
            result.extend(_statements(statement.orelse))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            result.extend(_statements(statement.body))
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                result.extend(_statements(case.body))
    return result


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _declared_names(statements: list[ast.stmt]) -> set[str]:
    """Return names declared by statements in one lexical scope."""
    result = set()
    for statement in _statements(statements):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(statement.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                result.update(_assigned_names(target))
        elif isinstance(statement, ast.AnnAssign):
            result.update(_assigned_names(statement.target))
        elif isinstance(statement, ast.TypeAlias):
            result.update(_assigned_names(statement.name))
    return result


def _literal_strings(value: ast.expr) -> set[str] | None:
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None
    result = set()
    for item in value.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        result.add(item.value)
    return result


def _literal_all(tree: ast.Module) -> set[str] | None:  # noqa: C901
    """Return every possible literal `__all__`, or `None` if it is computed."""
    result: set[str] = set()
    found = False
    initialized = False
    top_level = set(tree.body)
    for statement in _statements(tree.body):
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "__all__"
            ):
                found = True
                if not initialized or call.keywords or len(call.args) != 1:
                    return None
                if call.func.attr == "append":
                    value = call.args[0]
                    if not isinstance(value, ast.Constant) or not isinstance(
                        value.value, str
                    ):
                        return None
                    result.add(value.value)
                elif call.func.attr == "extend":
                    values = _literal_strings(call.args[0])
                    if values is None:
                        return None
                    result.update(values)
                else:
                    return None
                continue
        if isinstance(statement, ast.AugAssign):
            if "__all__" not in _assigned_names(statement.target):
                continue
            found = True
            if not initialized or not isinstance(statement.op, ast.Add):
                return None
            values = _literal_strings(statement.value)
            if values is None:
                return None
            result.update(values)
            continue
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if not any("__all__" in _assigned_names(target) for target in targets):
            continue
        found = True
        value = statement.value
        if value is None:
            continue
        values = _literal_strings(value)
        if values is None:
            return None
        if statement in top_level:
            result = values
        else:
            result.update(values)
        initialized = True
    return result if found else None


def parse_stub(path: Path) -> ast.Module:  # noqa: C901
    """Parse a stub while running under the suite's Python 3.12 virtualenv.

    The overlays use PEP 696 type-parameter defaults, which Python 3.12 cannot
    parse. Removing those defaults is safe here because coverage only needs the
    declaration tree, not annotation semantics.
    """
    tokens = list(
        tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
    )
    output: list[tokenize.TokenInfo] = []
    declaration = False
    saw_name = False
    type_parameter_depth = 0
    skipping_default = False
    for index, token in enumerate(tokens):
        kind, value = token.type, token.string
        if type_parameter_depth:
            if value in {"(", "[", "{"}:
                type_parameter_depth += 1
            elif value in {")", "]", "}"}:
                if value == "]" and type_parameter_depth == 1:
                    type_parameter_depth = 0
                    skipping_default = False
                else:
                    type_parameter_depth -= 1
            elif value == "=" and type_parameter_depth == 1:
                skipping_default = True
                continue
            elif value == "," and type_parameter_depth == 1:
                skipping_default = False
            if skipping_default:
                continue
            output.append(token)
            continue

        if kind == tokenize.NAME and (
            value in {"class", "def"}
            or (value == "type" and _starts_parameterized_type_alias(tokens, index))
        ):
            declaration = True
            saw_name = False
        elif declaration and kind == tokenize.NAME:
            saw_name = True
        elif declaration and saw_name and value == "[":
            type_parameter_depth = 1
        elif declaration and saw_name and value in {"(", ":"}:
            declaration = False
        elif declaration and kind not in {tokenize.COMMENT, tokenize.NL}:
            declaration = False
        output.append(token)
    return ast.parse(tokenize.untokenize(output), filename=str(path))


def stub_exports(path: Path, tree: ast.Module | None = None) -> set[str]:
    """Return names that a stub exposes from one module."""
    if tree is None:
        tree = parse_stub(path)
    explicit_all = _literal_all(tree)
    # `__all__` controls star imports, not whether direct attribute access is
    # valid. Include it to recognize re-exports, but retain every declaration.
    exports: set[str] = set() if explicit_all is None else set(explicit_all)
    exports.update(_declared_names(tree.body))
    for statement in _statements(tree.body):
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                if alias.name == "*":
                    raise ValueError(f"{path} uses an unsupported star import")
                # PEP 484 only treats an import as a stub re-export when it uses
                # an explicit alias (usually `from x import Y as Y`).
                if alias.asname is not None:
                    exports.add(alias.asname)
    return exports - {"__all__"}


def stub_class_members(
    path: Path, class_name: str, tree: ast.Module | None = None
) -> set[str]:
    """Return members declared directly on a class in a stub."""
    if tree is None:
        tree = parse_stub(path)
    members: set[str] = set()
    found = False
    for statement in _statements(tree.body):
        if isinstance(statement, ast.ClassDef) and statement.name == class_name:
            found = True
            members.update(_declared_names(statement.body))
    if not found:
        raise ValueError(f"{path} does not declare class {class_name}")
    return {name for name in members if not name.startswith("_")}


def runtime_exports(module: Any) -> set[str]:
    """Return the best available approximation of a module's public API."""
    exports = {name for name in dir(module) if not name.startswith("_")}
    explicit_all = getattr(module, "__all__", None)
    if explicit_all is not None and not isinstance(explicit_all, str):
        try:
            all_names = {name for name in explicit_all if isinstance(name, str)}
        except TypeError as error:
            name = getattr(module, "__name__", type(module).__name__)
            raise ValueError(
                f"runtime module {name} has a non-iterable __all__"
            ) from error
        exports.update(all_names)
    return exports


def runtime_class_members(runtime_class: type[Any]) -> set[str]:
    """Return public members declared by a runtime class or its bases."""
    return {
        name
        for base in runtime_class.__mro__
        for name in vars(base)
        if not name.startswith("_")
    }


def _import_runtime_module(name: str) -> Any:
    """Import a configured module with an actionable configuration error."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise ValueError(
            f"cannot import runtime module {name}; check the configured module name "
            "and installed package"
        ) from error


def _validate_exclusion_targets(config: Config, stub_modules: set[str]) -> None:
    member_targets = {
        f"{target.stub_module}.{target.stub_class}" for target in config.member_targets
    }
    for field, configured, valid in (
        ("skip_modules", config.skip_modules, stub_modules),
        ("module_excludes", config.module_excludes.keys(), stub_modules),
        ("member_excludes", config.member_excludes.keys(), member_targets),
    ):
        unknown = configured - valid
        if unknown:
            raise ValueError(
                f"{field} has entries with no matching stub target: "
                f"{', '.join(sorted(unknown))}"
            )


def collect(config: Config) -> dict[str, dict[str, list[str]]]:
    """Collect the declarations missing from an overlay."""
    if not config.stub_directory.is_dir():
        raise ValueError(f"stub directory is not a directory: {config.stub_directory}")
    stub_files = sorted(config.stub_directory.rglob("*.pyi"))
    if not stub_files:
        raise ValueError(
            f"stub directory contains no .pyi files: {config.stub_directory}"
        )
    paths_by_module: dict[str, Path] = {}
    for stub_file in stub_files:
        name = module_name(config.stub_directory, stub_file, config.runtime_package)
        previous = paths_by_module.get(name)
        if previous is not None:
            raise ValueError(
                f"stub paths {previous} and {stub_file} both map to runtime module "
                f"{name}"
            )
        paths_by_module[name] = stub_file

    _validate_exclusion_targets(config, set(paths_by_module))

    modules: dict[str, list[str]] = {}
    trees_by_path: dict[Path, ast.Module] = {}
    for name, stub_file in paths_by_module.items():
        if name in config.skip_modules:
            continue
        runtime_module = _import_runtime_module(name)
        tree = parse_stub(stub_file)
        trees_by_path[stub_file] = tree
        modules[name] = sorted(
            runtime_exports(runtime_module)
            - stub_exports(stub_file, tree)
            - config.module_excludes.get(name, frozenset())
        )

    members: dict[str, list[str]] = {}
    for target in config.member_targets:
        stub_file = paths_by_module.get(target.stub_module)
        if stub_file is None:
            raise ValueError(f"no stub file found for {target.stub_module}")
        key = f"{target.stub_module}.{target.stub_class}"
        runtime_module = _import_runtime_module(target.runtime_module)
        try:
            runtime_class = getattr(runtime_module, target.runtime_class)
        except AttributeError as error:
            raise ValueError(
                f"runtime module {target.runtime_module} has no class "
                f"{target.runtime_class}; fix or remove member_targets entry {key}"
            ) from error
        tree = trees_by_path.get(stub_file)
        if tree is None:
            tree = parse_stub(stub_file)
            trees_by_path[stub_file] = tree
        stub_members = stub_class_members(stub_file, target.stub_class, tree)
        excluded_members = config.member_excludes.get(key, frozenset())
        members[key] = sorted(
            name
            for name in runtime_class_members(runtime_class)
            if name not in stub_members and name not in excluded_members
        )
    return {"modules": modules, "members": members}


def format_report(
    package: str,
    version: str,
    gaps: dict[str, dict[str, list[str]]],
    *,
    show_names: bool,
) -> str:
    """Format a compact inventory, optionally including every missing name."""
    gap_count = sum(
        len(names) for section in gaps.values() for names in section.values()
    )
    declaration = "declaration" if gap_count == 1 else "declarations"
    lines = [f"{package} {version}: {gap_count} missing {declaration}"]
    for section in ("modules", "members"):
        lines.append(f"\n{section.capitalize()}:")
        for name, missing in gaps[section].items():
            lines.append(f"  {name}: {len(missing)}")
            if show_names and missing:
                lines.append("    " + ", ".join(missing))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--show-names",
        action="store_true",
        help="include every missing declaration in text output (JSON always does)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete inventory as JSON",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config.resolve())
        gaps = collect(config)
    except (OSError, SyntaxError, ValueError, tokenize.TokenError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        version = importlib.metadata.version(config.runtime_package)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    if args.json:
        print(
            json.dumps(
                {"package": config.runtime_package, "version": version, **gaps},
                indent=2,
            )
        )
    else:
        print(
            format_report(
                config.runtime_package,
                version,
                gaps,
                show_names=args.show_names,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
