#!/usr/bin/env python3
"""Generate a synthetic enterprise-scale Python codebase for indexing benchmarks.

This generator creates a package with many modules that intentionally stress:
- deep inheritance and large class graphs
- heavy typing usage (Generic, Protocol, Annotated, callable-heavy signatures)
- dense cross-module and circular import structure
- high metadata volume via Google-style docstrings
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

BASE_MODULE_NAMES = [
    "core",
    "models",
    "services",
    "utils",
    "handlers",
    "repositories",
    "api",
    "auth",
    "events",
    "analytics",
]


def build_module_names(module_count: int, base_count: int) -> list[str]:
    """Build module names with stable first N base names.

    Args:
        module_count: Total number of modules to generate.
        base_count: Number of leading base modules.

    Returns:
        Ordered module names without file extensions.
    """
    if module_count < base_count:
        raise ValueError("module_count must be >= base_count")
    if base_count > len(BASE_MODULE_NAMES):
        raise ValueError(f"base_count cannot exceed {len(BASE_MODULE_NAMES)}")

    names = BASE_MODULE_NAMES[:base_count]
    for index in range(base_count, module_count):
        names.append(f"module_{index + 1:02d}")
    return names


def google_docstring(title: str, args: Iterable[tuple[str, str]], returns: str) -> str:
    """Create a compact Google-style docstring block.

    Args:
        title: Summary line for the docstring.
        args: Iterable of (name, description) entries.
        returns: Return section description.

    Returns:
        A formatted multi-line docstring body.
    """
    arg_lines = "\n".join(f"            {name}: {description}" for name, description in args)
    return (
        f'"""{title}\n\n'
        f"        Args:\n"
        f"{arg_lines}\n\n"
        f"        Returns:\n"
        f"            {returns}\n"
        f'        """'
    )


def render_header(name: str, names: list[str], index: int, import_names: list[str]) -> str:
    """Render module header, imports, and shared typing aliases."""
    imports: list[str] = []

    local_index = import_names.index(name)
    if local_index > 0:
        imports.append(f"from . import {import_names[local_index - 1]} as prev_mod")
    if local_index > 1:
        imports.append(f"from . import {import_names[local_index - 2]} as prev_prev_mod")
    if local_index < len(import_names) - 1:
        imports.append(f"from . import {import_names[local_index + 1]} as next_mod")
    if local_index < len(import_names) - 2:
        imports.append(f"from . import {import_names[local_index + 2]} as next_next_mod")

    distant = import_names[(local_index + 7) % len(import_names)]

    header = f'''from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

{chr(10).join(imports)}

if TYPE_CHECKING:
    from . import {distant} as distant_mod

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

ComplexCallable = Callable[[List[T], Dict[str, U]], Awaitable[V]]
MergeCallable = Callable[[Sequence[V], Mapping[str, U]], Awaitable[Dict[str, V]]]
AuditTag = Annotated[str, "benchmark", "{name}"]
Vector = Annotated[List[Annotated[T, "vector-item"]], "vector"]


class Normalizer(Protocol[T, U, V]):
    """Protocol describing heavy generic normalization behavior.

    Args:
        payload: Mapping payload to normalize.
        callback: Callable converting payload fragments.

    Returns:
        Awaitable structure with normalized result.
    """

    async def normalize(
        self,
        payload: Annotated[Mapping[str, T], "normalizer-payload"],
        callback: ComplexCallable[T, U, V],
    ) -> Annotated[Dict[str, V], "normalizer-result"]:
        """Normalize payload values using an async callback.

        Args:
            payload: Input payload with string keys.
            callback: Async callback for converting each entry.

        Returns:
            Mapping of normalized values.
        """


@dataclass
class BaseNode(Generic[T, U, V]):
    """Root base node that tracks typed state for indexing workload.

    Args:
        module_name: Name of the owning module.
        state: Mutable storage used during transformations.

    Returns:
        None.
    """

    module_name: AuditTag
    state: MutableMapping[str, T] = field(default_factory=dict)

    async def collect(
        self,
        items: Vector[T],
        callback: ComplexCallable[T, U, V],
    ) -> Annotated[Tuple[V, ...], "collect-result"]:
        """Collect converted values from typed vector input.

        Args:
            items: Vectorized input items for conversion.
            callback: Async callback converting each vector element.

        Returns:
            Tuple of converted output values.
        """
        output: List[V] = []
        for position, item in enumerate(items):
            output.append(await callback([item], {{"position": position}}))
        return tuple(output)


class Layer1(BaseNode[T, U, V]):
    """First inheritance layer in deep hierarchy.

    Args:
        module_name: Name of the owning module.

    Returns:
        None.
    """


class Layer2(Layer1[T, U, V]):
    """Second inheritance layer in deep hierarchy.

    Args:
        module_name: Name of the owning module.

    Returns:
        None.
    """


class Layer3(Layer2[T, U, V]):
    """Third inheritance layer in deep hierarchy.

    Args:
        module_name: Name of the owning module.

    Returns:
        None.
    """


class Layer4(Layer3[T, U, V]):
    """Fourth inheritance layer in deep hierarchy.

    Args:
        module_name: Name of the owning module.

    Returns:
        None.
    """


class Layer5(Layer4[T, U, V]):
    """Fifth inheritance layer in deep hierarchy.

    Args:
        module_name: Name of the owning module.

    Returns:
        None.
    """
'''
    return header


def render_entity_block(module_name: str, entity_index: int) -> str:
    """Render one heavily-typed class block with multiple methods."""
    class_name = f"{module_name.title().replace('_', '')}Entity{entity_index:02d}"

    init_doc = google_docstring(
        f"Initialize {class_name} with typed mapping state.",
        [
            ("module_name", "Module label used for audit tagging."),
            ("seed", "Starting integer used to prepopulate state."),
        ],
        "None.",
    )
    orchestrate_doc = google_docstring(
        f"Run async orchestration pipeline for {class_name}.",
        [
            ("payload", "Input payload values indexed by key."),
            ("callback", "Async converter callback with generic variance."),
            ("merge", "Async merge callback producing final mapping."),
        ],
        "Aggregated mapping of orchestrated values.",
    )
    summarize_doc = google_docstring(
        f"Summarize payload for {class_name}.",
        [
            ("payload", "Input payload used for summary projection."),
            ("fallback", "Fallback value inserted for missing keys."),
        ],
        "Annotated summary map.",
    )

    return f'''

class {class_name}(Layer5[T, U, V], Generic[T, U, V]):
    """Concrete entity in synthetic enterprise hierarchy.

    Args:
        module_name: Name of the owning module.
        state: Mutable typed state map.

    Returns:
        None.
    """

    def __init__(
        self,
        module_name: AuditTag,
        seed: Annotated[int, "seed"],
    ) -> None:
        {init_doc}
        self.module_name = module_name
        self.state = {{f"k_{{offset}}": seed + offset for offset in range(4)}}

    async def orchestrate(
        self,
        payload: Annotated[Mapping[str, T], "payload"],
        callback: ComplexCallable[T, U, V],
        merge: MergeCallable[U, V],
    ) -> Annotated[Dict[str, V], "orchestrated-map"]:
        {orchestrate_doc}
        staged: List[V] = []
        for key, value in payload.items():
            transformed = await callback([value], {{"key": key, "module": self.module_name}})
            staged.append(transformed)
        return await merge(staged, {{"module": self.module_name, "entity": "{class_name}"}})

    def summarize(
        self,
        payload: Annotated[Mapping[str, T], "summary-payload"],
        fallback: Annotated[V, "fallback"],
    ) -> Annotated[Dict[str, V], "summary-map"]:
        {summarize_doc}
        summary: Dict[str, V] = {{}}
        for key in payload:
            summary[key] = fallback
        return summary
'''


def render_tail(name: str) -> str:
    """Render module-level helper functions with dense type signatures."""
    return f'''

async def execute_{name}_pipeline(
    entities: Annotated[Sequence[Layer5[int, str, float]], "entities"],
    callback: ComplexCallable[int, str, float],
    merge: MergeCallable[str, float],
) -> Annotated[Dict[str, float], "pipeline-output"]:
    """Execute all entity pipelines for a module.

    Args:
        entities: Layered entities participating in pipeline.
        callback: Async callback for conversion stage.
        merge: Async callback for merge stage.

    Returns:
        Aggregate output map for the module.
    """
    result: Dict[str, float] = {{}}
    for index, entity in enumerate(entities):
        payload: Dict[str, int] = {{f"item_{{index}}": index, "item_extra": index + 10}}
        merged = await entity.orchestrate(payload, callback, merge)  # type: ignore[attr-defined]
        result.update({{f"{{entity.module_name}}_{{k}}": v for k, v in merged.items()}})
    return result


def build_{name}_entities() -> Annotated[List[Layer5[int, str, float]], "entity-list"]:
    """Build a typed list of entities for benchmark execution.

    Args:
        None.

    Returns:
        Constructed list of layered entities.
    """
    return [
        {name.title().replace('_', '')}Entity00("{name}", 10),
        {name.title().replace('_', '')}Entity01("{name}", 20),
        {name.title().replace('_', '')}Entity02("{name}", 30),
    ]


async def noop_callback(values: List[int], metadata: Dict[str, str]) -> float:
    """Default callback used for local smoke checks.

    Args:
        values: Values that should be transformed.
        metadata: Metadata map containing event context.

    Returns:
        Floating point summary for the given values.
    """
    await asyncio.sleep(0)
    return float(sum(values) + len(metadata))


async def noop_merge(values: Sequence[float], metadata: Mapping[str, str]) -> Dict[str, float]:
    """Default merge callback used for local smoke checks.

    Args:
        values: Sequence of transformed values.
        metadata: Metadata map carrying merge context.

    Returns:
        Mapping of merged values keyed by synthetic names.
    """
    await asyncio.sleep(0)
    return {{f"m_{{i}}_{{metadata.get('module', 'unknown')}}": value for i, value in enumerate(values)}}
'''


def render_module(
    name: str,
    names: list[str],
    index: int,
    heavy: bool,
    line_target: int,
    import_names: list[str],
) -> str:
    """Render a complete module file with configurable size and complexity."""
    header = render_header(name, names, index, import_names)

    # Base modules are intentionally much larger to seed the benchmark suite.
    entity_count = 26 if heavy else 10
    body = "".join(render_entity_block(name, i) for i in range(entity_count))
    tail = render_tail(name)
    content = header + body + tail

    while len(content.splitlines()) < line_target:
        filler_index = len(content.splitlines())
        content += f'''\n\n
def filler_{name}_{filler_index:04d}(\n    callback: ComplexCallable[int, str, float],\n) -> Annotated[Callable[[List[int], Dict[str, str]], Awaitable[float]], "filler-signature"]:\n    """Return callback unchanged to enlarge AST and annotation graph.\n\n    Args:\n        callback: Existing callback to pass through unchanged.\n\n    Returns:\n        The same callback object.\n    """\n    return callback\n'''

    return content + "\n"


def write_package_files(output_dir: Path, module_names: list[str], only_base: bool, line_target: int) -> None:
    """Write package structure and module files to output directory.

    Args:
        output_dir: Destination root for generated package.
        module_names: Ordered module names.
        only_base: Whether to generate only the base module subset.
        line_target: Minimum line count per generated module.

    Returns:
        None.
    """
    package_dir = output_dir / "mock_org"
    package_dir.mkdir(parents=True, exist_ok=True)

    generated_names = [name for name in module_names if (name in BASE_MODULE_NAMES or not only_base)]

    init_file = package_dir / "__init__.py"
    init_lines = [
        '"""Synthetic enterprise package for Pyrefly perf scaling tests."""',
        "",
        "__all__ = [",
    ]
    init_lines.extend(f'    "{name}",' for name in generated_names)
    init_lines.append("]")
    init_file.write_text("\n".join(init_lines) + "\n", encoding="utf-8")

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Mock Org Perf Suite",
                "",
                "Generated synthetic Python modules for stressing import and type indexing.",
                "",
                "Use generate_mock_org.py to regenerate this dataset.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for index, name in enumerate(module_names):
        is_base = name in BASE_MODULE_NAMES
        if only_base and not is_base:
            continue
        module_text = render_module(
            name=name,
            names=module_names,
            index=index,
            heavy=is_base,
            line_target=line_target if is_base else 220,
            import_names=generated_names,
        )
        (package_dir / f"{name}.py").write_text(module_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line options for generator execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="perf/scale_test",
        help="Directory that will contain the generated mock_org package.",
    )
    parser.add_argument(
        "--module-count",
        type=int,
        default=60,
        help="Total number of modules to include in dependency graph.",
    )
    parser.add_argument(
        "--base-count",
        type=int,
        default=10,
        help="Number of leading named base modules.",
    )
    parser.add_argument(
        "--line-target",
        type=int,
        default=520,
        help="Minimum line count for each base module.",
    )
    parser.add_argument(
        "--only-base",
        action="store_true",
        help="Generate only the named base modules while still planning full graph imports.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for mock organization generation."""
    args = parse_args()
    module_names = build_module_names(module_count=args.module_count, base_count=args.base_count)
    write_package_files(
        output_dir=Path(args.output_dir),
        module_names=module_names,
        only_base=args.only_base,
        line_target=args.line_target,
    )
    print(
        f"Generated mock org at {args.output_dir}/mock_org with "
        f"{args.base_count if args.only_base else len(module_names)} module files "
        f"(graph size: {len(module_names)})."
    )


if __name__ == "__main__":
    main()
