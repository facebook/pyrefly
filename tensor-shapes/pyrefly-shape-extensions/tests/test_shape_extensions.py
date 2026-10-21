# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import typing
import unittest
from pathlib import Path
from unittest import mock

import shape_extensions
from shape_extensions import _PostImportHook, _PostImportLoader, shape_vars


class ShapeVarsRuntimeTest(unittest.TestCase):
    def test_empty_declaration_preserves_functions(self) -> None:
        def identity(value: int) -> int:
            return value

        self.assertIs(shape_vars("")(identity), identity)

    def test_empty_declaration_preserves_generic_type_arguments(self) -> None:
        @shape_vars("")
        class Box[T]:
            pass

        specialized = Box[int]
        self.assertIs(typing.get_origin(specialized), Box)
        self.assertEqual(typing.get_args(specialized), (int,))


class LegacyTypeVarRuntimeTest(unittest.TestCase):
    def test_supports_generics_and_shape_splats(self) -> None:
        namespace: dict[str, typing.Any] = {}
        exec(
            textwrap.dedent(
                """
                from typing import Generic
                from shape_extensions import IntTuple, TypeVar

                Shape = TypeVar("Shape", bound=IntTuple)

                class Box(Generic[Shape]):
                    pass

                Specialized = Box[int]
                unpacked = [1, *Shape, 2]
                """
            ),
            namespace,
        )
        shape = namespace["Shape"]

        self.assertIsInstance(shape, typing.TypeVar)
        self.assertEqual(typing.get_args(namespace["Specialized"]), (int,))
        self.assertEqual(namespace["unpacked"], [1, shape, 2])


class PostImportHookTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        sys.path.insert(0, directory.name)
        self.addCleanup(sys.path.remove, directory.name)
        self.name = f"post_import_target_{self.id().rsplit('.', 1)[-1]}"
        self.addCleanup(sys.modules.pop, self.name, None)
        self.calls: list[types.ModuleType] = []

    def write_module(self, body: str) -> None:
        (self.directory / f"{self.name}.py").write_text(textwrap.dedent(body))
        importlib.invalidate_caches()

    def install_hook(self) -> _PostImportHook:
        def callback() -> None:
            module = sys.modules[self.name]
            vars(module)["patched"] = True
            self.calls.append(module)

        hook = _PostImportHook(self.name, callback)
        sys.meta_path.insert(0, hook)
        self.addCleanup(hook.remove)
        return hook

    def test_callback_runs_after_module_body_and_removes_hook(self) -> None:
        self.write_module(
            """
            patched = False
            body_finished = True
            """
        )
        hook = self.install_hook()

        module = importlib.import_module(self.name)

        self.assertEqual(self.calls, [module])
        self.assertTrue(module.body_finished)
        self.assertTrue(module.patched)
        self.assertNotIn(hook, sys.meta_path)
        self.assertNotIsInstance(module.__loader__, _PostImportLoader)
        spec = module.__spec__
        assert spec is not None
        self.assertIs(spec.loader, module.__loader__)

    def test_repeated_import_runs_callback_once(self) -> None:
        self.write_module("")
        self.install_hook()

        first = importlib.import_module(self.name)
        second = importlib.import_module(self.name)

        self.assertIs(first, second)
        self.assertEqual(self.calls, [first])

    def test_find_spec_alone_keeps_hook(self) -> None:
        self.write_module("")
        hook = self.install_hook()

        self.assertIsNotNone(importlib.util.find_spec(self.name))

        self.assertEqual(self.calls, [])
        self.assertIn(hook, sys.meta_path)
        self.assertNotIn(self.name, sys.modules)

    def test_legacy_loader_removes_hook(self) -> None:
        hook = self.install_hook()
        loader = mock.Mock(spec=["load_module"])
        spec = importlib.util.spec_from_loader(self.name, loader)
        assert spec is not None
        finder = mock.Mock()
        finder.find_spec.return_value = spec

        with mock.patch.object(sys, "meta_path", [hook, finder]):
            self.assertIs(hook.find_spec(self.name, None), spec)
            self.assertNotIn(hook, sys.meta_path)
            self.assertIs(spec.loader, loader)

    def test_failed_import_keeps_hook_for_retry(self) -> None:
        self.write_module(
            """
            import os

            if os.environ.get("POST_IMPORT_TARGET_FAIL"):
                raise RuntimeError("failed import")
            """
        )
        hook = self.install_hook()

        with mock.patch.dict(os.environ, {"POST_IMPORT_TARGET_FAIL": "1"}):
            with self.assertRaisesRegex(RuntimeError, "failed import"):
                importlib.import_module(self.name)
        self.assertEqual(self.calls, [])
        self.assertIn(hook, sys.meta_path)

        module = importlib.import_module(self.name)
        self.assertEqual(self.calls, [module])
        self.assertNotIn(hook, sys.meta_path)

    def test_concurrent_imports_return_after_callback(self) -> None:
        self.write_module(
            """
            import time

            patched = False
            time.sleep(0.2)
            """
        )
        self.install_hook()
        observed: list[bool] = []

        def import_target() -> None:
            observed.append(importlib.import_module(self.name).patched)

        threads = [threading.Thread(target=import_target) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(observed, [True] * 4)
        self.assertEqual(len(self.calls), 1)


# A stand-in for Torch that defines every class `shape_extensions` patches. The
# environment can make it fail once, import slowly, or import `shape_extensions`
# during its own initialization.
_FAKE_TORCH_INIT = """
import os
import sys
import time

sys.fake_torch_attempts = getattr(sys, "fake_torch_attempts", 0) + 1
if os.environ.get("FAKE_TORCH_FAIL_ONCE") and sys.fake_torch_attempts == 1:
    raise RuntimeError("fake torch failed to import")
time.sleep(float(os.environ.get("FAKE_TORCH_DELAY", "0")))
if os.environ.get("FAKE_TORCH_IMPORTS_SHAPE_EXTENSIONS"):
    import shape_extensions

from . import nn


class Tensor:
    pass


fully_imported = True
"""

_FAKE_TORCH_NN_INIT = """
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    cls = globals()[name] = type(name, (), {})
    return cls
"""


class TorchPatchTest(unittest.TestCase):
    """Runs scripts in subprocesses, importing a stand-in Torch package."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        package = Path(directory.name) / "torch"
        (package / "nn").mkdir(parents=True)
        (package / "__init__.py").write_text(_FAKE_TORCH_INIT)
        (package / "nn" / "__init__.py").write_text(_FAKE_TORCH_NN_INIT)
        shape_extensions_root = Path(shape_extensions.__file__).parent.parent
        self.python_path = os.pathsep.join([directory.name, str(shape_extensions_root)])

    def run_script(self, script: str, **env: str) -> None:
        prelude = textwrap.dedent(
            """
            import sys

            def assert_patched(torch):
                assert torch.Tensor[2, 3] is torch.Tensor
                assert torch.nn.Linear[2, 3] is torch.nn.Linear

            def hooks():
                return [
                    finder
                    for finder in sys.meta_path
                    if type(finder).__name__ == "_PostImportHook"
                ]
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", prelude + textwrap.dedent(script)],
            env={**os.environ, **env, "PYTHONPATH": self.python_path},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_importing_shape_extensions_does_not_import_torch(self) -> None:
        self.run_script(
            """
            import shape_extensions

            assert "torch" not in sys.modules
            assert len(hooks()) == 1
            """
        )

    def test_torch_imported_first_is_patched_immediately(self) -> None:
        self.run_script(
            """
            import torch
            import shape_extensions

            assert_patched(torch)
            assert hooks() == []
            """
        )

    def test_torch_imported_later_is_patched_before_import_returns(self) -> None:
        self.run_script(
            """
            import shape_extensions
            import torch

            assert torch.fully_imported
            assert_patched(torch)
            assert hooks() == []
            """
        )

    def test_patch_is_idempotent(self) -> None:
        self.run_script(
            """
            import torch
            import shape_extensions

            descriptor = torch.Tensor.__dict__["__class_getitem__"]
            shape_extensions._patch_torch()
            shape_extensions._patch_torch_when_imported()
            assert torch.Tensor.__dict__["__class_getitem__"] is descriptor
            assert hooks() == []
            """
        )

    def test_failed_torch_import_is_patched_on_retry(self) -> None:
        self.run_script(
            """
            import shape_extensions

            try:
                import torch
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected the first import to fail")
            assert "torch" not in sys.modules
            assert len(hooks()) == 1

            import torch

            assert_patched(torch)
            assert hooks() == []
            """,
            FAKE_TORCH_FAIL_ONCE="1",
        )

    def test_patch_failure_does_not_fail_torch_import(self) -> None:
        self.run_script(
            """
            import warnings
            from unittest import mock

            import shape_extensions

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with mock.patch.object(
                    shape_extensions, "_patch_torch", side_effect=AttributeError("missing class")
                ):
                    import torch

            assert torch.fully_imported
            assert sys.modules["torch"] is torch
            assert hooks() == []
            assert len(caught) == 1
            assert issubclass(caught[0].category, RuntimeWarning)
            assert "missing class" in str(caught[0].message)
            """
        )

    def test_waits_for_torch_import_in_another_thread(self) -> None:
        self.run_script(
            """
            import threading
            import time

            thread = threading.Thread(target=__import__, args=("torch",))
            thread.start()
            while "torch" not in sys.modules:
                time.sleep(0.001)
            import shape_extensions

            import torch

            assert torch.fully_imported
            assert_patched(torch)
            assert hooks() == []
            thread.join()
            """,
            FAKE_TORCH_DELAY="0.5",
        )

    def test_import_during_torch_initialization_leaves_torch_unpatched(self) -> None:
        # Torch is partially initialized while it imports `shape_extensions`, so
        # its classes may not exist yet, and its loader is already running.
        self.run_script(
            """
            import torch

            assert torch.fully_imported
            assert not hasattr(torch.Tensor, "__class_getitem__")
            assert hooks() == []
            """,
            FAKE_TORCH_IMPORTS_SHAPE_EXTENSIONS="1",
        )
