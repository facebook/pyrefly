# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import ast
import inspect
import textwrap
import typing
import unittest
from typing import assert_type, Optional, TYPE_CHECKING

import shape_extensions.torchscript
import torch
from shape_extensions.torchscript import (
    Int,
    IntVar,
    remove_shape_types_from_torch_sources,
)
from torch import Tensor

Batch = IntVar("Batch")
D = IntVar("D")
module = object()


class _Lookalike:
    """Stands in for a project flag that is live at runtime."""

    TYPE_CHECKING = True


_lookalike = _Lookalike()


def _shape_annotated_identity(
    x: "Tensor[[Batch, D]]",
    n: "Int[Batch]",
) -> "Tensor[[Batch, D]]":
    return x + n


class _ShapeAnnotatedModule(torch.nn.Module):
    cache: "Tensor[[Batch, D]]"

    def __init__(self) -> None:
        super().__init__()
        self.cache = torch.zeros(1, 1)

    def forward(
        self,
        x: "Tensor[[Batch, D]]",
        n: "Int[Batch]",
    ) -> "Tensor[[Batch, D]]":
        return x + self.cache + n


class _ShapeAnnotatedOptionalModule(torch.nn.Module):
    cache: "Optional[Tensor[[Batch, D]]]"

    def __init__(self) -> None:
        super().__init__()
        self.cache = None

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        cache = self.cache
        if cache is None:
            return x
        return x + cache


class _ShapeAnnotatedIntModule(torch.nn.Module):
    count: Int[Batch]

    def __init__(self) -> None:
        super().__init__()
        self.count = 2

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        return x + self.count


class _ShapeAnnotatedListModule(torch.nn.Module):
    offsets: "list[Int[Batch]]"

    def __init__(self) -> None:
        super().__init__()
        self.offsets = [2]

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        return x + self.offsets[0]


class _ShapeAnnotatedUnresolvableModule(torch.nn.Module):
    # `Undefined` resolves to nothing, so the rewritten annotation cannot be
    # evaluated and TorchScript has to report the failure itself.
    cache: "Undefined[Tensor[[Batch, D]]]"  # noqa: F821

    def __init__(self) -> None:
        super().__init__()
        self.cache = None

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        return x


class _Pep695AnnotatedModule[Feat: IntVar](torch.nn.Module):
    """The idiomatic modern form: shape variables as PEP 695 type parameters.

    TorchScript parses the `def` line, so the type parameter list has to be
    tolerable to it as well as the annotations.
    """

    def forward[B: IntVar](self, x: Tensor[[B, Feat]]) -> Tensor[[B, Feat]]:
        return x + 1


class _ShapeAnnotatedUnionModule(torch.nn.Module):
    """A quoted annotation that continues past the shape type.

    The rewrite has to replace the whole quoted annotation; replacing just the
    shape type inside it leaves an unbalanced quote and unparsable source.
    """

    cache: "Tensor[[Batch, D]] | None"

    def __init__(self) -> None:
        super().__init__()
        self.cache = None

    def forward(self, x: "Tensor[[Batch, D]] | None") -> "Tensor[[Batch, D]]":
        if x is None:
            return torch.zeros(1, 2)
        return x


def _shape_annotated_union(x: "Tensor[[Batch, D]] | None") -> "Int[Batch] | None":
    return None


class _ShapeAssertedModule(torch.nn.Module):
    """Static shape assertions inside a type-checking-only guard.

    TorchScript compiles both arms of a conditional and cannot resolve the
    guard's name, so the block has to be erased for this to script at all.
    The guard is spelled out in the body rather than here, so that the
    erasure assertions can match on the statement itself.
    """

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        out = x + 1
        if TYPE_CHECKING:
            assert_type(out, Tensor[[Batch, D]])
        return out


class _AnnotatedAssignmentModule(torch.nn.Module):
    """An annotated assignment, the TorchScript-safe alternative to `cast`."""

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        out: "Tensor[[Batch, D]]" = x + 1
        return out


class _NestedTypeCheckingModule(torch.nn.Module):
    """Nested type-checking-only guards.

    The inner block sits inside the text the outer one is replaced by, so
    erasing both would apply overlapping edits and corrupt the source.
    """

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        out = x + 1
        if TYPE_CHECKING:
            if TYPE_CHECKING:
                assert_type(out, Tensor[[Batch, D]])
            assert_type(out, Tensor[[Batch, D]])
        return out


class _QualifiedTypeCheckingModule(torch.nn.Module):
    """The guard reached through the `typing` module rather than imported."""

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        out = x + 1
        if typing.TYPE_CHECKING:
            assert_type(out, Tensor[[Batch, D]])
        return out


class _LookalikeFlagModule(torch.nn.Module):
    """A live runtime flag that happens to share the sentinel's name.

    Erasing this would delete real behavior, so an attribute on anything other
    than `typing` is left alone.
    """

    def forward(self, x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
        out = x
        if _lookalike.TYPE_CHECKING:
            out = out + 1
        return out


def _shape_annotated_quoted_optional(
    x: "Optional[Tensor[[Batch, D]]]",
) -> "Optional[Tensor[[Batch, D]]]":
    return x


def _shape_annotated_qualified_identity(
    x: "torch.Tensor[[Batch, D]]",
    n: "shape_extensions.torchscript.Int[Batch]",
) -> "torch.Tensor[[Batch, D]]":
    return x + n


def _shape_annotated_string_literal(x: "Tensor[[Batch, D]]") -> str:
    # Tensor[[Batch, D]]
    note = "Tensor[[Batch, D]]"
    return note


# The line breaks inside these annotations are what this fixture is testing, so
# they must survive formatting.
# fmt: off
def _shape_annotated_multiline(
    x: Tensor[
        [Batch, D],
    ],
) -> Tensor[
    [Batch, D],
]:
    return x


def _shape_annotated_nested_multiline(
    x: Optional[
        Tensor[
            [Batch, D],
        ]
    ],
) -> Optional[
    Tensor[
        [Batch, D],
    ]
]:
    return x
# fmt: on


def _shape_annotated_non_ascii_prefix(
    calfé: "Tensor[[Batch, D]]",
) -> "Tensor[[Batch, D]]":
    return calfé


def _shape_annotated_unrelated_attribute(
    x: "module.Tensor[[Batch, D]]",
) -> "module.Int[Batch]":
    return x


def _shape_annotated_unbalanced_string(x: "Tensor[[Batch, D]]") -> "Tensor[[Batch, D]]":
    note = "Tensor["
    if note == "":
        return x
    return x


class TorchScriptAnnotationStripperTest(unittest.TestCase):
    """Importing `shape_extensions.torchscript` applies the patch; these tests
    rely on that import rather than calling the patcher themselves."""

    def _rewritten_source(self, obj) -> str:
        source_lines, _, _ = torch._sources.get_source_lines_and_file(obj)
        return "".join(source_lines)

    def test_replaces_annotations_only_in_source(self) -> None:
        scripted = torch.jit.script(_shape_annotated_string_literal)
        self.assertEqual(scripted(torch.ones(1, 2)), "Tensor[[Batch, D]]")

        rewritten = self._rewritten_source(_shape_annotated_string_literal)

        self.assertIn("x: torch.Tensor", rewritten)
        self.assertIn('note = "Tensor[[Batch, D]]"', rewritten)
        self.assertIn("# Tensor[[Batch, D]]", rewritten)

    def test_preserves_multiline_source_annotation_line_count(self) -> None:
        original_lines, _ = inspect.getsourcelines(_shape_annotated_multiline)
        rewritten = self._rewritten_source(_shape_annotated_multiline)

        self.assertEqual(rewritten.count("\n"), len(original_lines))
        self.assertIn("x: (torch.Tensor", rewritten)
        self.assertIn(") -> (torch.Tensor", rewritten)
        ast.parse(textwrap.dedent(rewritten))

        scripted = torch.jit.script(_shape_annotated_multiline)
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x), x)

    def test_preserves_nested_multiline_annotation_line_count(self) -> None:
        original_lines, _ = inspect.getsourcelines(_shape_annotated_nested_multiline)
        rewritten = self._rewritten_source(_shape_annotated_nested_multiline)

        self.assertEqual(rewritten.count("\n"), len(original_lines))
        ast.parse(textwrap.dedent(rewritten))

        scripted = torch.jit.script(_shape_annotated_nested_multiline)
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x), x)

    def test_handles_non_ascii_before_annotation(self) -> None:
        rewritten = self._rewritten_source(_shape_annotated_non_ascii_prefix)

        self.assertIn("calfé: torch.Tensor", rewritten)
        self.assertIn("-> torch.Tensor", rewritten)

    def test_leaves_unrelated_attribute_annotations_alone(self) -> None:
        rewritten = self._rewritten_source(_shape_annotated_unrelated_attribute)

        self.assertIn('"module.Tensor[[Batch, D]]"', rewritten)
        self.assertIn('"module.Int[Batch]"', rewritten)

    def test_leaves_unbalanced_shape_strings_alone(self) -> None:
        rewritten = self._rewritten_source(_shape_annotated_unbalanced_string)

        self.assertIn('note = "Tensor["', rewritten)

    def test_torchscript_scripts_shape_annotated_function(self) -> None:
        scripted = torch.jit.script(_shape_annotated_identity)
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x, 2), x + 2)

    def test_torchscript_scripts_shape_annotated_module(self) -> None:
        scripted = torch.jit.script(_ShapeAnnotatedModule())
        x = torch.ones(1, 1)
        torch.testing.assert_close(scripted(x, 2), x + 2)

    def test_torchscript_scripts_composite_class_annotation(self) -> None:
        scripted = torch.jit.script(_ShapeAnnotatedOptionalModule())
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x), x)

    def test_torchscript_scripts_container_class_annotation(self) -> None:
        scripted = torch.jit.script(_ShapeAnnotatedListModule())
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x), x + 2)

    def test_torchscript_scripts_quoted_composite_function_annotation(self) -> None:
        scripted = torch.jit.script(_shape_annotated_quoted_optional)
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x), x)
        self.assertIsNone(scripted(None))

    def test_torchscript_scripts_qualified_shape_annotations(self) -> None:
        scripted = torch.jit.script(_shape_annotated_qualified_identity)
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x, 2), x + 2)

    def test_unresolvable_rewrite_defers_to_torchscript_diagnostic(self) -> None:
        with self.assertRaises(Exception) as caught:
            torch.jit.script(_ShapeAnnotatedUnresolvableModule())

        # The point is that the failure comes from TorchScript rather than
        # escaping as a raw error out of our own evaluation of the rewrite.
        # The wording is TorchScript's and is not contractual, so only the
        # exception type is asserted.
        self.assertNotIsInstance(caught.exception, (NameError, SyntaxError))

    def test_torchscript_module_enables_runtime_compat_before_class_definition(
        self,
    ) -> None:
        scripted = torch.jit.script(_ShapeAnnotatedIntModule())
        x = torch.ones(1, 2)
        torch.testing.assert_close(scripted(x), x + 2)

    def test_torchscript_scripts_pep695_type_parameters(self) -> None:
        scripted = torch.jit.script(_Pep695AnnotatedModule())
        x = torch.ones(1, 2)

        torch.testing.assert_close(scripted(x), x + 1)

    def test_rewrites_quoted_union_annotation(self) -> None:
        rewritten = self._rewritten_source(_shape_annotated_union)

        # The whole quoted annotation is replaced, leaving no stray quote.
        self.assertIn("x: torch.Tensor | None", rewritten)
        self.assertIn("-> int | None", rewritten)
        ast.parse(textwrap.dedent(rewritten))

    def test_torchscript_scripts_quoted_union_annotation(self) -> None:
        scripted = torch.jit.script(_ShapeAnnotatedUnionModule())
        x = torch.ones(1, 2)

        torch.testing.assert_close(scripted(x), x)

    def test_erases_type_checking_blocks(self) -> None:
        rewritten = self._rewritten_source(_ShapeAssertedModule)

        # The docstring mentions TYPE_CHECKING, so match the statement itself.
        self.assertNotIn("if TYPE_CHECKING:", rewritten)
        self.assertNotIn("assert_type(", rewritten)
        self.assertIn("# type-checking block erased", rewritten)
        ast.parse(textwrap.dedent(rewritten))

    def test_preserves_line_count_when_erasing_type_checking(self) -> None:
        original_lines, _ = inspect.getsourcelines(_ShapeAssertedModule)
        rewritten = self._rewritten_source(_ShapeAssertedModule)

        self.assertEqual(rewritten.count("\n"), len(original_lines))

    def test_torchscript_scripts_module_with_shape_assertions(self) -> None:
        scripted = torch.jit.script(_ShapeAssertedModule())
        x = torch.ones(1, 2)

        torch.testing.assert_close(scripted(x), x + 1)

    def test_torchscript_scripts_annotated_assignment(self) -> None:
        scripted = torch.jit.script(_AnnotatedAssignmentModule())
        x = torch.ones(1, 2)

        torch.testing.assert_close(scripted(x), x + 1)

    def test_erases_only_the_outermost_of_nested_blocks(self) -> None:
        original_lines, _ = inspect.getsourcelines(_NestedTypeCheckingModule)
        rewritten = self._rewritten_source(_NestedTypeCheckingModule)

        # Overlapping edits used to eat surrounding source while still parsing,
        # so check the line count and the neighbouring statement, not just that
        # the result is syntactically valid.
        self.assertEqual(rewritten.count("\n"), len(original_lines))
        self.assertIn("return out", rewritten)
        self.assertNotIn("assert_type(", rewritten)
        ast.parse(textwrap.dedent(rewritten))

    def test_erases_qualified_typing_sentinel(self) -> None:
        rewritten = self._rewritten_source(_QualifiedTypeCheckingModule)

        self.assertNotIn("typing.TYPE_CHECKING", rewritten)
        self.assertIn("# type-checking block erased", rewritten)

    def test_leaves_lookalike_runtime_flag_alone(self) -> None:
        rewritten = self._rewritten_source(_LookalikeFlagModule)

        self.assertIn("_lookalike.TYPE_CHECKING", rewritten)
        self.assertNotIn("# type-checking block erased", rewritten)

    def test_lookalike_flag_still_runs_at_runtime(self) -> None:
        x = torch.ones(1, 2)

        torch.testing.assert_close(_LookalikeFlagModule()(x), x + 1)

    def test_patching_is_idempotent(self) -> None:
        sources = torch._sources
        annotations = torch.jit.annotations
        already_patched = (sources.get_source_lines_and_file, annotations.ann_to_type)

        remove_shape_types_from_torch_sources()

        self.assertIs(sources.get_source_lines_and_file, already_patched[0])
        self.assertIs(annotations.ann_to_type, already_patched[1])
