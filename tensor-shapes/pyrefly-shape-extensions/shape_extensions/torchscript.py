# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Shape annotations with TorchScript-compatible runtime behavior enabled.

Importing this module erases shape-only runtime annotations to types
TorchScript understands. This must happen before any annotated class body is
evaluated, because TorchScript reads class attribute annotations back out of
`__annotations__`, so a call made after import would be too late.

`Int` here is the same class object as `shape_extensions.Int`, so the change is
process-global and one-way: it also affects code that imports `shape_extensions`
directly. There is intentionally no way to undo it.

Importing this module also patches TorchScript's source loading so that shape
annotations written in source, rather than read back from `__annotations__`,
are erased too. The two halves are not independently useful: a model needs both
to be scriptable.
"""

from __future__ import annotations

import ast
import textwrap

from . import *  # noqa: F401, F403
from . import __all__ as _shape_extensions_all, _return_int, Int as _Int

__all__ = [*_shape_extensions_all, "remove_shape_types_from_torch_sources"]

_Int.__class_getitem__ = classmethod(_return_int)


_SHAPE_TYPE_REPLACEMENTS = (
    ("shape_extensions.torchscript.Int[", "int"),
    ("shape_extensions.Int[", "int"),
    ("torch.Tensor[", "torch.Tensor"),
    ("Tensor[", "torch.Tensor"),
    ("Int[", "int"),
)


def _shape_type_end(source: str, open_bracket: int) -> int | None:
    depth = 0
    for i in range(open_bracket, len(source)):
        if source[i] == "[":
            depth += 1
        elif source[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _is_identifier_part(char: str) -> bool:
    return char == "_" or char.isalnum()


def _has_name_boundary(source: str, start: int) -> bool:
    return start == 0 or (
        source[start - 1] != "." and not _is_identifier_part(source[start - 1])
    )


def _replace_shape_types(source: str) -> str:
    rewritten: list[str] = []
    i = 0
    while i < len(source):
        for prefix, replacement in _SHAPE_TYPE_REPLACEMENTS:
            if source.startswith(prefix, i) and _has_name_boundary(source, i):
                end = _shape_type_end(source, i + len(prefix) - 1)
                if end is None:
                    continue
                rewritten.append(replacement)
                i = end
                break
        else:
            rewritten.append(source[i])
            i += 1
    return "".join(rewritten)


def _annotation_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _source_line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _source_offset(
    source: str, line_offsets: list[int], lineno: int, col_offset: int
) -> int:
    line_start = line_offsets[lineno - 1]
    line_end = line_offsets[lineno] if lineno < len(line_offsets) else len(source)
    line = source[line_start:line_end]
    return line_start + len(line.encode("utf-8")[:col_offset].decode("utf-8"))


def _line_indent(source: str, start: int) -> str:
    line_start = source.rfind("\n", 0, start) + 1
    prefix = source[line_start:start]
    return prefix[: len(prefix) - len(prefix.lstrip())]


def _preserve_line_count(source: str, replacement: str, indent: str) -> str:
    missing_newlines = source.count("\n") - replacement.count("\n")
    if missing_newlines <= 0:
        return replacement

    filler = [f"{indent}# shape annotation erased"] * (missing_newlines - 1)
    return "\n".join([f"({replacement}", *filler, f"{indent})"])


def _is_type_checking_sentinel(test: ast.expr) -> bool:
    """Recognize the typing sentinel, and only it.

    `TYPE_CHECKING` and `typing.TYPE_CHECKING` are the spellings this erasure
    supports. An attribute on anything else is not assumed to be the sentinel:
    a project is free to have its own `TYPE_CHECKING` flag that is live at
    runtime, and erasing that would delete real behavior. An aliased import
    such as `import typing as t` is likewise not recognized, which errs toward
    leaving code alone.
    """

    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def _erase_type_checking_blocks(source: str) -> str:
    """Replace `if TYPE_CHECKING:` blocks with `pass`, preserving line count.

    TorchScript compiles both arms of an `if`, and cannot resolve
    `TYPE_CHECKING` itself: it fails with "python value of type 'bool' cannot
    be used as a value". Such a block is dead at runtime by construction, so
    erasing it is safe, and it is what lets a scripted model carry static
    `assert_type` checks on its shapes.

    A block with an `else` is left alone: the `else` arm is the live one, and
    rewriting it is beyond what this erasure is for.
    """

    tree = ast.parse(source)
    line_offsets = _source_line_offsets(source)
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.orelse:
            continue
        if not _is_type_checking_sentinel(node.test):
            continue
        if node.end_lineno is None or node.end_col_offset is None:
            continue
        spans.append(
            (
                _source_offset(source, line_offsets, node.lineno, node.col_offset),
                _source_offset(
                    source, line_offsets, node.end_lineno, node.end_col_offset
                ),
            )
        )

    # Only the outermost of a set of nested blocks is erased. A nested block is
    # already inside the text its enclosing block replaces, so keeping both
    # would apply two edits to overlapping spans, and the second would be
    # working from offsets the first had already invalidated.
    outermost: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if outermost and start < outermost[-1][1]:
            continue
        outermost.append((start, end))

    for start, end in reversed(outermost):
        indent = _line_indent(source, start)
        filler = [f"{indent}# type-checking block erased"] * source[start:end].count(
            "\n"
        )
        replacement = "\n".join(["pass", *filler])
        source = f"{source[:start]}{replacement}{source[end:]}"
    return source


def _replace_shape_types_in_source(source: str) -> str:
    try:
        ast.parse(source)
    except IndentationError:
        source = textwrap.dedent(source)

    # Erased first, in a separate pass, so that annotations inside these blocks
    # are gone before annotation rewriting runs and cannot produce overlapping
    # edits to the same span.
    source = _erase_type_checking_blocks(source)
    tree = ast.parse(source)

    line_offsets = _source_line_offsets(source)
    replacements: list[tuple[int, int, str]] = []
    for node in _annotation_nodes(tree):
        if (
            node.end_lineno is None
            or node.end_col_offset is None
            or node.lineno is None
            or node.col_offset is None
        ):
            continue
        start = _source_offset(source, line_offsets, node.lineno, node.col_offset)
        end = _source_offset(source, line_offsets, node.end_lineno, node.end_col_offset)
        original = source[start:end]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A wholly quoted annotation. Rewrite what the quotes contain and
            # emit the result unquoted, rather than trying to consume the
            # quotes while scanning: an annotation that continues past the
            # shape type, such as `"Tensor[[B, D]] | None"`, would otherwise
            # lose its opening quote and keep its closing one.
            #
            # Dropping the quotes is deliberate. TorchScript rejects a union
            # written as a string but accepts the same union unquoted, so
            # leaving the quotes on would turn a rewrite that parses into one
            # that still fails to script. The cost is that a genuine forward
            # reference inside a rewritten annotation is now evaluated eagerly.
            # Compare against the contents rather than the quoted form, so an
            # annotation with nothing to rewrite keeps its quotes.
            rewritten = _replace_shape_types(node.value)
            changed = rewritten != node.value
        else:
            rewritten = _replace_shape_types(original)
            changed = rewritten != original
        if changed:
            replacements.append(
                (
                    start,
                    end,
                    _preserve_line_count(
                        original, rewritten, _line_indent(source, start)
                    ),
                )
            )

    for start, end, replacement in sorted(replacements, reverse=True):
        source = f"{source[:start]}{replacement}{source[end:]}"
    return source


class _ShapeTypingRemover:
    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, obj, error_msg=None):
        source_lines, file_lineno, filename = self.func(obj, error_msg)
        return (
            _replace_shape_types_in_source("".join(source_lines)).splitlines(
                keepends=True
            ),
            file_lineno,
            filename,
        )


class _ShapeAnnotationTypeRemover:
    def __init__(self, func, torch) -> None:
        self.func = func
        self.torch = torch

    def __call__(self, ann, *args, **kwargs):
        if isinstance(ann, str):
            rewritten = _replace_shape_types(ann)
            if rewritten != ann:
                rcb = kwargs.get("rcb", args[1] if len(args) > 1 else None)
                annotations = self.torch.jit.annotations
                # Resolved outside the try so that a rename or removal of these
                # private torch APIs surfaces as an error here rather than being
                # mistaken for an unresolvable annotation.
                eval_no_call = annotations._eval_no_call
                eval_env = annotations.EvalEnv(rcb)
                try:
                    ann = eval_no_call(rewritten, {}, eval_env)
                except (AttributeError, NameError, SyntaxError):
                    # Resolving the rewrite is best effort: it relies on private
                    # torch APIs, and on the rewritten text naming something
                    # resolvable in the caller's scope. Any failure means we
                    # cannot improve on the annotation we were given, so pass the
                    # original through and let TorchScript report it against the
                    # source the user wrote. Swallowing the error here would be
                    # wrong; TorchScript still rejects an annotation it cannot
                    # resolve, just with a better message than an eval traceback.
                    pass
        return self.func(ann, *args, **kwargs)


def _is_shape_typing_remover(func) -> bool:
    while func is not None:
        if isinstance(func, _ShapeTypingRemover):
            return True
        func = getattr(func, "func", None)
    return False


def _is_shape_annotation_type_remover(func) -> bool:
    while func is not None:
        if isinstance(func, _ShapeAnnotationTypeRemover):
            return True
        func = getattr(func, "func", None)
    return False


def remove_shape_types_from_torch_sources() -> None:
    """Patch TorchScript source loading to erase Pyrefly shape annotations."""

    try:
        import torch  # @manual
    except ImportError:
        return

    sources = torch._sources if hasattr(torch, "_sources") else torch.jit.frontend
    if not _is_shape_typing_remover(sources.get_source_lines_and_file):
        sources.get_source_lines_and_file = _ShapeTypingRemover(
            sources.get_source_lines_and_file
        )

    annotations = torch.jit.annotations
    if _is_shape_annotation_type_remover(annotations.ann_to_type):
        return
    annotations.ann_to_type = _ShapeAnnotationTypeRemover(
        annotations.ann_to_type, torch
    )


remove_shape_types_from_torch_sources()
