# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import stub_coverage
from stub_coverage import (
    collect,
    Config,
    format_report,
    load_config,
    main,
    MemberTarget,
    module_name,
    parse_stub,
    runtime_class_members,
    runtime_exports,
    stub_class_members,
    stub_exports,
)


class StubCoverageTest(unittest.TestCase):
    def test_config_keys_are_validated(self) -> None:
        cases = (
            (
                'runtime_package = "package"\nstub_directory = "stubs"\n'
                "unknown = true\n",
                "configuration .* has unknown keys: unknown",
            ),
            (
                'runtime_package = "package"\n',
                "configuration .* is missing required keys: stub_directory",
            ),
            (
                'runtime_package = "package"\nstub_directory = "stubs"\n'
                "[[member_targets]]\n"
                'stub_module = "package"\n'
                'stub_class = "Class"\n'
                'runtime_module = "package"\n',
                r"member_targets\[0\] is missing required keys: runtime_class",
            ),
            (
                'runtime_package = "package"\nstub_directory = "stubs"\n'
                "[[member_targets]]\n"
                'stub_module = "package"\n'
                'stub_class = "Class"\n'
                'runtime_module = "package"\n'
                'runtime_class = "Class"\n'
                "unknown = true\n",
                r"member_targets\[0\] has unknown keys: unknown",
            ),
            (
                'runtime_package = "package"\nstub_directory = "stubs"\n'
                'member_targets = ["Class"]\n',
                r"member_targets\[0\] must be a table",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stub_coverage.toml"
            for contents, message in cases:
                with self.subTest(message=message):
                    path.write_text(contents)
                    with self.assertRaisesRegex(ValueError, message):
                        load_config(path)

    def test_stub_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import Exported as Exported, Internal

__all__ = ["Exported"]

type Alias = int

class Box[T = int]:
    value: T
    def method(self) -> None: ...

def function[T: int = int](value: T) -> T: ...
"""
            )
            self.assertEqual(
                stub_exports(path),
                {"Alias", "Box", "Exported", "function"},
            )
            self.assertEqual(stub_class_members(path, "Box"), {"method", "value"})

    def test_conditional_class_members_are_unioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
if sys.version_info >= (3, 13):
    class Versioned:
        newer: int
else:
    class Versioned:
        older: int
"""
            )
            self.assertEqual(stub_class_members(path, "Versioned"), {"newer", "older"})

    def test_parameterized_alias_defaults_and_incremental_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
__all__: list[str]
__all__ = ["Alias"]
__all__ += ["loop_name", "with_name", "match_name"]

type Alias[T = int] = list[T]
for item in values:
    loop_name, *rest = item
with context:
    with_name: int
match value:
    case _:
        match_name: int
"""
            )
            self.assertEqual(
                stub_exports(path),
                {"Alias", "loop_name", "match_name", "rest", "with_name"},
            )

    def test_literal_all_append_and_extend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import Appended, Extended1, Extended2

__all__ = []
__all__.append("Appended")
__all__.extend(["Extended1", "Extended2"])
"""
            )
            self.assertEqual(
                stub_exports(path),
                {"Appended", "Extended1", "Extended2"},
            )

    def test_conditional_all_branches_are_unioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import New, Old

if sys.version_info >= (3, 13):
    __all__ = ["New"]
else:
    __all__ = ["Old"]
"""
            )
            self.assertEqual(stub_exports(path), {"New", "Old"})

    def test_top_level_all_reassignment_replaces_previous_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import Current, Previous

__all__ = ["Previous"]
__all__ = ["Current"]
"""
            )
            self.assertEqual(stub_exports(path), {"Current"})

    def test_computed_all_falls_back_to_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import Exported as Exported

__all__ = exported_names()
__all__ += ["Unknown"]

def function() -> None: ...
"""
            )
            self.assertEqual(stub_exports(path), {"Exported", "function"})

    def test_uninitialized_all_augmentation_falls_back_to_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import Exported as Exported

__all__ += ["Unknown"]
"""
            )
            self.assertEqual(stub_exports(path), {"Exported"})

    def test_computed_all_extension_falls_back_to_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
from elsewhere import Exported as Exported, Hidden

__all__ = ["Hidden"]
__all__.extend(exported_names())
"""
            )
            self.assertEqual(stub_exports(path), {"Exported"})

    def test_type_soft_keyword_uses_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                """
type: Annotated[int, lambda x=1: x]
runtime_type = type(Annotated[int, lambda x=1: x])
"""
            )
            self.assertEqual(stub_exports(path), {"runtime_type", "type"})

    def test_type_expression_does_not_trigger_parameter_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text("value = type is registry[lambda x=1: x]\n")
            tree = parse_stub(path)
            self.assertEqual(
                ast.unparse(tree),
                "value = type is registry[lambda x=1: x]",
            )

    def test_stub_parse_errors_preserve_source_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pyi"
            path.write_text(
                "class Box[\n    T = int,\n]:\n    pass\n\ndef broken() return None\n"
            )
            with self.assertRaises(SyntaxError) as context:
                parse_stub(path)
            self.assertEqual(context.exception.lineno, 6)
            self.assertEqual(context.exception.offset, 14)

    def test_module_name(self) -> None:
        root = Path("package-stubs")
        self.assertEqual(module_name(root, root / "__init__.pyi", "package"), "package")
        self.assertEqual(
            module_name(root, root / "nested" / "api.pyi", "package"),
            "package.nested.api",
        )

    def test_collect_rejects_missing_or_empty_stub_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            for stub_directory, message in (
                (root / "missing", "stub directory is not a directory"),
                (empty, "stub directory contains no .pyi files"),
            ):
                with self.subTest(stub_directory=stub_directory):
                    config = Config(
                        runtime_package="unused",
                        stub_directory=stub_directory,
                        skip_modules=frozenset(),
                        module_excludes={},
                        member_excludes={},
                        member_targets=(),
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        collect(config)

    def test_collect_rejects_duplicate_module_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stub_directory = Path(directory)
            (stub_directory / "module").mkdir()
            (stub_directory / "module.pyi").write_text("")
            (stub_directory / "module" / "__init__.pyi").write_text("")
            config = Config(
                runtime_package="package",
                stub_directory=stub_directory,
                skip_modules=frozenset(),
                module_excludes={},
                member_excludes={},
                member_targets=(),
            )
            with self.assertRaisesRegex(
                ValueError,
                "both map to runtime module package.module",
            ):
                collect(config)

    def test_collect_reports_invalid_runtime_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stub_directory = Path(directory)
            (stub_directory / "__init__.pyi").write_text("class Missing: ...\n")
            cases = (
                (
                    Config(
                        runtime_package="missing_runtime_package",
                        stub_directory=stub_directory,
                        skip_modules=frozenset(),
                        module_excludes={},
                        member_excludes={},
                        member_targets=(),
                    ),
                    "cannot import runtime module missing_runtime_package",
                ),
                (
                    Config(
                        runtime_package="sys",
                        stub_directory=stub_directory,
                        skip_modules=frozenset(),
                        module_excludes={},
                        member_excludes={},
                        member_targets=(
                            MemberTarget(
                                stub_module="sys",
                                stub_class="Missing",
                                runtime_module="sys",
                                runtime_class="Missing",
                            ),
                        ),
                    ),
                    "runtime module sys has no class Missing; "
                    "fix or remove member_targets entry sys.Missing",
                ),
            )
            for config, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        collect(config)

    def test_collect_rejects_exclusions_without_matching_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stub_directory = Path(directory)
            (stub_directory / "__init__.pyi").write_text("")
            cases = (
                (
                    {"missing"},
                    {},
                    {},
                    "skip_modules has entries with no matching stub target: missing",
                ),
                (
                    set(),
                    {"missing": frozenset()},
                    {},
                    "module_excludes has entries with no matching stub target: missing",
                ),
                (
                    set(),
                    {},
                    {"sys.Missing": frozenset()},
                    "member_excludes has entries with no matching stub target: sys.Missing",
                ),
            )
            for skip_modules, module_excludes, member_excludes, message in cases:
                with self.subTest(message=message):
                    config = Config(
                        runtime_package="sys",
                        stub_directory=stub_directory,
                        skip_modules=frozenset(skip_modules),
                        module_excludes=module_excludes,
                        member_excludes=member_excludes,
                        member_targets=(),
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        collect(config)

    def test_collect_parses_each_stub_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stub_directory = Path(directory)
            (stub_directory / "__init__.pyi").write_text("class int: ...\n")
            config = Config(
                runtime_package="builtins",
                stub_directory=stub_directory,
                skip_modules=frozenset(),
                module_excludes={},
                member_excludes={},
                member_targets=(
                    MemberTarget(
                        stub_module="builtins",
                        stub_class="int",
                        runtime_module="builtins",
                        runtime_class="int",
                    ),
                ),
            )
            with mock.patch.object(
                stub_coverage, "parse_stub", wraps=parse_stub
            ) as mocked_parse:
                collect(config)
            mocked_parse.assert_called_once()

    def test_runtime_exports_include_all_public_names(self) -> None:
        class Module:
            __all__ = ["listed", "_listed_private"]
            listed = None
            visible = None
            _listed_private = None
            _private = None

        self.assertEqual(
            runtime_exports(Module()),
            {"listed", "visible", "_listed_private"},
        )

    def test_runtime_exports_reject_non_iterable_all(self) -> None:
        class Module:
            __name__ = "package"
            __all__ = 1

        with self.assertRaisesRegex(
            ValueError, "runtime module package has a non-iterable __all__"
        ):
            runtime_exports(Module())

    def test_runtime_class_members_include_bases_but_not_metaclass(self) -> None:
        class Meta(type):
            metaclass_only = None

        class Base:
            inherited = None

        class Child(Base, metaclass=Meta):
            direct = None

        self.assertEqual(runtime_class_members(Child), {"direct", "inherited"})

    def test_report_can_include_names(self) -> None:
        gaps = {"modules": {"package": ["missing"]}, "members": {}}
        self.assertEqual(
            format_report("package", "1.0", gaps, show_names=True),
            "package 1.0: 1 missing declaration\n\n"
            "Modules:\n"
            "  package: 1\n"
            "    missing\n\n"
            "Members:",
        )

    def test_main_reports_validation_errors_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stub_coverage.toml"
            path.write_text('runtime_package = "package"\n')
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = main([str(path)])
            self.assertEqual(result, 2)
            self.assertRegex(
                stderr.getvalue(),
                r"^error: configuration .* is missing required keys: stub_directory\n$",
            )

    def test_main_reports_file_toml_and_stub_errors_without_tracebacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.toml"
            malformed.write_text("invalid =\n")
            invalid_stub_directory = root / "stubs"
            invalid_stub_directory.mkdir()
            (invalid_stub_directory / "__init__.pyi").write_text("value =\n")
            invalid_stub = root / "invalid_stub.toml"
            invalid_stub.write_text(
                'runtime_package = "sys"\nstub_directory = "stubs"\n'
            )
            for path, message in (
                (root / "missing.toml", "No such file or directory"),
                (malformed, "Invalid value"),
                (invalid_stub, "invalid syntax"),
            ):
                with self.subTest(path=path):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        result = main([str(path)])
                    self.assertEqual(result, 2)
                    self.assertIn(message, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_reports_token_errors_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub_directory = root / "stubs"
            stub_directory.mkdir()
            (stub_directory / "__init__.pyi").write_text('value = """\n')
            config = root / "stub_coverage.toml"
            config.write_text('runtime_package = "sys"\nstub_directory = "stubs"\n')
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = main([str(config)])
            self.assertEqual(result, 2)
            self.assertIn("EOF in multi-line string", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
