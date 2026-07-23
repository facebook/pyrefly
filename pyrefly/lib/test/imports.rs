/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::fs_anyhow;

use crate::test::util::TestEnv;
use crate::testcase;

fn env_class_x() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
class X: ...
x: X = X()
"#,
    )
}

fn env_class_x_deeper() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.pyi", "");
    t.add_with_path(
        "foo.bar",
        "foo/bar.pyi",
        r#"
class X: ...
x: X = X()
"#,
    );
    t
}

testcase!(
    test_imports_works,
    env_class_x(),
    r#"
from typing import assert_type
from foo import x, X
assert_type(x, X)
"#,
);

testcase!(
    test_imports_broken,
    env_class_x(),
    r#"
from foo import x, X
class Y: ...
b: Y = x  # E: `X` is not assignable to `Y`
"#,
);

testcase!(
    test_imports_star,
    env_class_x(),
    r#"
from typing import assert_type
from foo import *
y: X = x
assert_type(y, X)
"#,
);

testcase!(
    test_imports_star_dunder,
    TestEnv::one("foo", "def __derp__() -> int: ..."),
    r#"
from typing import assert_type
from foo import *
__derp__() # E: Could not find name `__derp__`
"#,
);

testcase!(
    test_imports_module_single,
    env_class_x(),
    r#"
from typing import assert_type
import foo
y: foo.X = foo.x
assert_type(y, foo.X)
"#,
);

testcase!(
    test_imports_module_as,
    env_class_x(),
    r#"
from typing import assert_type
import foo as bar
y: bar.X = bar.x
assert_type(y, bar.X)
"#,
);

testcase!(
    test_imports_module_nested,
    env_class_x_deeper(),
    r#"
from typing import assert_type
import foo.bar
y: foo.bar.X = foo.bar.x
assert_type(y, foo.bar.X)
"#,
);

testcase!(
    test_import_overwrite,
    env_class_x(),
    r#"
from foo import X, x
class X: ...
y: X = x  # E: `foo.X` is not assignable to `main.X`
"#,
);

fn env_imports_dot() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.pyi", "");
    t.add_with_path("foo.bar", "foo/bar/__init__.pyi", "");
    t.add_with_path("foo.bar.baz", "foo/bar/baz.pyi", "from .qux import x");
    t.add_with_path("foo.bar.qux", "foo/bar/qux.pyi", "x: int = 1");
    t
}

testcase!(
    test_imports_dot,
    env_imports_dot(),
    r#"
from typing import assert_type
from foo.bar.baz import x
assert_type(x, int)
"#,
);

testcase!(
    test_access_nonexistent_module,
    env_imports_dot(),
    r#"
import foo.bar.baz
foo.qux.wibble.wobble # E: No attribute `qux` in module `foo`
"#,
);

fn env_star_reexport() -> TestEnv {
    let mut t = TestEnv::new();
    t.add("base", "class Foo: ...");
    t.add("second", "from base import *");
    t
}

testcase!(
    test_imports_star_transitive,
    env_star_reexport(),
    r#"
from typing import assert_type
from second import *
assert_type(Foo(), Foo)
"#,
);

fn env_redefine_class() -> TestEnv {
    TestEnv::one("foo", "class Foo: ...")
}

testcase!(
    bug = "The anywhere lookup of Foo in the function body finds both the imported and locally defined classes",
    test_redefine_class,
    env_redefine_class(),
    r#"
from typing import assert_type
from foo import *
class Foo: ...
def f(x: Foo) -> Foo:
    return Foo() # E: Returned type `foo.Foo | main.Foo` is not assignable to declared return type `main.Foo`
assert_type(f(Foo()), Foo)
"#,
);

testcase!(
    test_dont_export_underscore,
    TestEnv::one("foo", "x: int = 1\n_y: int = 2"),
    r#"
from typing import assert_type, Any
from foo import *
assert_type(x, int)
assert_type(_y, Any)  # E: Could not find name `_y`
"#,
);

fn env_import_different_submodules() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.pyi", "");
    t.add_with_path("foo.bar", "foo/bar.pyi", "x: int = 1");
    t.add_with_path("foo.baz", "foo/baz.pyi", "x: str = 'a'");
    t
}

testcase!(
    test_import_different_submodules,
    env_import_different_submodules(),
    r#"
from typing import assert_type
import foo.bar
import foo.baz

assert_type(foo.bar.x, int)
assert_type(foo.baz.x, str)
"#,
);

testcase!(
    test_import_flow,
    env_import_different_submodules(),
    r#"
from typing import assert_type
import foo.bar

def test():
    assert_type(foo.bar.x, int)
    assert_type(foo.baz.x, str)

import foo.baz
"#,
);

testcase!(
    test_bad_import,
    r#"
from typing import assert_type, Any
from builtins import not_a_real_value  # E: Could not import `not_a_real_value` from `builtins`
assert_type(not_a_real_value, Any)
"#,
);

testcase!(
    test_bad_relative_import,
    r#"
from ... import does_not_exist  # E: Could not resolve relative import `...`
"#,
);

fn env_all_x() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
__all__ = ["x"]
x: int = 1
y: int = 3
    "#,
    )
}

testcase!(
    test_import_all,
    env_all_x(),
    r#"
from foo import *
z = y  # E: Could not find name `y`
"#,
);

fn env_broken_export() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.pyi", "from foo.bar import *");
    t.add_with_path(
        "foo.bar",
        "foo/bar.pyi",
        r#"
from foo import baz  # E: Could not import `baz` from `foo`
__all__ = []
"#,
    );
    t
}

testcase!(
    test_broken_export,
    env_broken_export(),
    r#"
import foo
"#,
);

fn env_main_guard() -> TestEnv {
    let mut t = TestEnv::new();
    t.add(
        "foo",
        r#"
x = 1
z = 0
if __name__ == "__main__":
    y = 2
    z = 3
"#,
    );
    t
}

testcase!(
    test_main_guard_not_exported,
    env_main_guard(),
    r#"
from foo import x
from foo import y  # E: Could not import `y` from `foo`
"#,
);

testcase!(
    test_main_guard_not_in_wildcard,
    env_main_guard(),
    r#"
from foo import *
x
y  # E: Could not find name `y`
"#,
);

// `z` is defined both at module level and inside the main guard. The
// `main_guard_only &= in_main_guard` merge must keep it importable via both
// direct and wildcard import. Regression guard against the `&=` becoming `=`.
testcase!(
    test_main_guard_merge_keeps_export,
    env_main_guard(),
    r#"
from foo import z
from foo import *
z
"#,
);

// `__all__` is defined at module level but mutated inside the main guard.
// The guard-only `append`/`extend` mutations must not leak into the wildcard
// surface, since they don't run at import time.
fn env_main_guard_dunder_all_mutation() -> TestEnv {
    let mut t = TestEnv::new();
    t.add(
        "foo",
        r#"
x = 1
y = 2
__all__ = ["x"]
if __name__ == "__main__":
    __all__.append("y")
    __all__.extend(["y"])
"#,
    );
    t
}

testcase!(
    test_main_guard_dunder_all_mutation_not_in_wildcard,
    env_main_guard_dunder_all_mutation(),
    r#"
from foo import *
x
y  # E: Could not find name `y`
"#,
);

fn env_relative_import_star() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.pyi", "from .bar import *");
    t.add_with_path("foo.bar", "foo/bar.pyi", "x: int = 5");
    t
}

testcase!(
    test_relative_import_star,
    env_relative_import_star(),
    r#"
from typing import assert_type
import foo

assert_type(foo.x, int)
"#,
);

fn env_dunder_init_with_submodule() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.py", "x: str = ''");
    t.add_with_path("foo.bar", "foo/bar.py", "x: int = 0");
    t
}

testcase!(
    test_from_package_import_module,
    env_dunder_init_with_submodule(),
    r#"
from foo import bar
from typing import assert_type
assert_type(bar.x, int)
from foo import baz  # E: Could not import `baz` from `foo`
"#,
);

testcase!(
    test_import_dunder_init_and_submodule,
    env_dunder_init_with_submodule(),
    r#"
from typing import assert_type
import foo
import foo.bar
assert_type(foo.x, str)
assert_type(foo.bar.x, int)
"#,
);

testcase!(
    test_import_dunder_init_without_submodule,
    env_dunder_init_with_submodule(),
    r#"
from typing import assert_type
import foo
assert_type(foo.x, str)
v = foo.bar.x  # E: Module `foo.bar` exists, but was not imported explicitly.
assert_type(v, int)
"#,
);

fn env_dunder_init_with_submodule2() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.py", "x: str = ''");
    t.add_with_path("foo.bar", "foo/bar/__init__.py", "x: int = 0");
    t.add_with_path("foo.bar.baz", "foo/bar/baz.py", "x: float = 4.2");
    t
}

testcase!(
    test_import_dunder_init_submodule_only,
    env_dunder_init_with_submodule2(),
    r#"
from typing import assert_type
import foo.bar.baz
assert_type(foo.x, str)
assert_type(foo.bar.x, int)
assert_type(foo.bar.baz.x, float)
"#,
);

fn env_dunder_init_overlap_submodule() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.py", "bar: str = ''");
    t.add_with_path("foo.bar", "foo/bar.py", "x: int = 0");
    t
}

testcase!(
    test_import_dunder_init_overlap_submodule_first,
    env_dunder_init_overlap_submodule(),
    r#"
from typing import assert_type
import foo.bar
import foo
assert_type(foo.bar.x, int)
"#,
);

testcase!(
    test_import_dunder_init_overlap_without_submodule,
    env_dunder_init_overlap_submodule(),
    r#"
from typing import assert_type
import foo
assert_type(foo.bar, str)
foo.bar.x # E: Object of class `str` has no attribute `x`
"#,
);

testcase!(
    test_import_dunder_init_overlap_submodule_only,
    env_dunder_init_overlap_submodule(),
    r#"
from typing import assert_type
import foo.bar
assert_type(foo.bar.x, int)
"#,
);
testcase!(
    test_from_dunder_init_import_submodule_no_extra_import,
    env_dunder_init_overlap_submodule(),
    r#"
from typing import assert_type
from foo import bar
assert_type(bar, str)
"#,
);

testcase!(
    test_from_dunder_init_import_submodule_with_extra_import,
    env_dunder_init_overlap_submodule(),
    r#"
from typing import assert_type
import foo.bar
from foo import bar
assert_type(bar, str)
"#,
);

fn env_dunder_init_reexport_submodule() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.py", "from .bar import x");
    t.add_with_path("foo.bar", "foo/bar.py", "x: int = 0");
    t
}

testcase!(
    test_import_dunder_init_reexport_submodule,
    env_dunder_init_reexport_submodule(),
    r#"
from typing import assert_type
import foo
assert_type(foo.x, int)
assert_type(foo.bar.x, int)
"#,
);

fn env_export_all_wrongly() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
__all__ = ['bad_definition']  # E: Name `bad_definition` is listed in `__all__` but is not defined in the module
__all__.extend(bad_module.__all__)  # E: Could not find name `bad_module`
"#,
    )
}

testcase!(
    test_export_all_wrongly,
    env_export_all_wrongly(),
    r#"
from foo import bad_definition
x = bad_definition
"#,
);

testcase!(
    test_export_all_wrongly_missing_other_name,
    env_export_all_wrongly(),
    r#"
from foo import missing_definition  # E: Could not import `missing_definition` from `foo`
x = missing_definition
"#,
);

testcase!(
    test_export_all_wrongly_does_not_export_implicit_builtin,
    TestEnv::one(
        "foo",
        r#"
# At runtime, listing a name in `__all__` does not create a module attribute:
# `from foo import len` and `from foo import *` both fail unless `len` is
# explicitly bound in the module, e.g. with `from builtins import len`.
__all__ = ["len"]  # E: Name `len` is listed in `__all__` but is not defined in the module
len([])
"#,
    ),
    r#"
from typing import reveal_type
from foo import len
reveal_type(len)  # E: revealed type: Unknown
"#,
);

testcase!(
    test_export_all_wrongly_star,
    env_export_all_wrongly(),
    r#"
from foo import *
"#,
);

testcase!(
    bug = "False negative",
    test_export_all_not_module,
    r#"
class not_module:
    __all__ = []

__all__ = []
__all__.extend(not_module.__all__)  # Should get an error about not_module not being imported
    # But Pyright doesn't give an error, so maybe we shouldn't either??
"#,
);

fn env_blank() -> TestEnv {
    TestEnv::one("foo", "")
}

testcase!(
    test_import_blank,
    env_blank(),
    r#"
import foo
x = foo.bar  # E: No attribute `bar` in module `foo`
"#,
);

testcase!(
    test_missing_import_named,
    r#"
from foo import bar  # E: Cannot find module `foo`
"#,
);

testcase!(
    test_missing_import_star,
    r#"
from foo import *  # E: Cannot find module `foo`
"#,
);

testcase!(
    test_missing_import_module,
    r#"
import foo, bar.baz  # E: Cannot find module `foo`  # E: Cannot find module `bar.baz`
"#,
);

testcase!(
    test_direct_import_toplevel,
    r#"
import typing

typing.assert_type(None, None)
"#,
);

testcase!(
    test_direct_import_function,
    r#"
import typing

def foo():
    typing.assert_type(None, None)
"#,
);

testcase!(
    test_import_blank_no_reexport_builtins,
    TestEnv::one("blank", ""),
    r#"
from blank import int as int_int # E: Could not import `int` from `blank`
"#,
);

#[test]
fn test_import_fail_to_load() {
    let temp = tempfile::tempdir().unwrap();
    let mut env = TestEnv::new();
    env.add_real_path("foo", temp.path().join("foo.py"));
    env.add("main", "import foo");
    let (state, handle) = env.to_state();
    let errs = state
        .transaction()
        .get_errors([&handle("foo")])
        .collect_errors()
        .ordinary;
    assert_eq!(errs.len(), 1);
    let err = &errs[0];
    assert!(err.msg().contains("Failed to load"));
    assert_eq!(err.display_range().to_string(), "1:1");
    assert_eq!(err.path().as_path().file_name(), Some("foo.py".as_ref()));
}

testcase!(
    test_import_os,
    r#"
import os
from typing import assert_type, LiteralString

x = os.path.join("source")
assert_type(x, LiteralString)
"#,
);

fn env_from_self_import_mod_in_package() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "foo",
        "foo/__init__.py",
        r#"
from . import bar
from . import baz as _baz
baz = _baz
"#,
    );
    env.add_with_path("foo.bar", "foo/bar.py", "");
    env.add_with_path("foo.baz", "foo/baz.py", "");
    env
}

testcase!(
    test_import_from_self,
    env_from_self_import_mod_in_package(),
    r#"
from typing import reveal_type
import foo
reveal_type(foo.bar)  # E: revealed type: Module[foo.bar]
reveal_type(foo.baz)  # E: revealed type: Module[foo.baz]
reveal_type(foo)  # E: revealed type: Module[foo]
"#,
);

fn env_var_leak() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from typing import TypeVar

T = TypeVar("T")

def copy(a: T) -> T: ...

class Interpret:
    @property
    def x(self):
        return copy(y) # E: Could not find name

    @x.setter
    def x(self, x):
        pass
"#,
    )
}

// This test used to crash with Var's leaking between modules
testcase!(
    test_var_leak,
    env_var_leak(),
    r#"
from foo import Interpret
from typing import reveal_type

def test():
    i = Interpret()
    # Deliberately don't specify the type of i.x, as sometimes
    # it works out to None, sometimes Unknown.
    # Plenty of errors here.
    reveal_type(i.x) # E:
"#,
);

fn env_override_typing() -> TestEnv {
    TestEnv::one(
        "typing",
        r#"
# This module uses `Iterator` from the real typeshed typing
for x in [1, 2, 3]:
    pass

custom_thing = 1
"#,
    )
}

testcase!(
    test_override_typing,
    env_override_typing(),
    r#"
# We are importing `typing` from `TestEnv`, which in turn makes use of things
# from the typeshed `typing`.
from typing import custom_thing

for x in [1, 2, 3]:
    pass
"#,
);

// This test was introduced after a regression caused us not to be able to resolve
// these specific names.
testcase!(
    test_some_stdlib_imports,
    r#"
import sys
import time
from typing import assert_type, Never
def f():
    assert_type(time.time(), float)
    assert_type(sys.exit(1), Never)
"#,
);

fn env_import_attribute_init() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.py", "import foo.attribute");
    t.add_with_path("foo.attribute", "foo/attribute.py", "x: int = 42");
    t
}

testcase!(
    test_import_attribute_init,
    env_import_attribute_init(),
    r#"
from typing import assert_type
import foo
assert_type(foo.attribute.x, int)
"#,
);

testcase!(
    test_pyi_ellipsis_no_return_annotation,
    TestEnv::one_with_path("foo", "foo.pyi", "def f(): ..."),
    r#"
from foo import f
from typing import Any, assert_type
assert_type(f(), Any)
    "#,
);

fn env_pyi_docstring() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "foo",
        "foo.pyi",
        r#"
def f():
    """Test."""
"#,
    );
    t
}

testcase!(
    test_pyi_docstring_no_return_annotation,
    env_pyi_docstring(),
    r#"
from foo import f
from typing import Any, assert_type
assert_type(f(), Any)
    "#,
);

fn env_literal_enum_validity() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
import enum
class E(enum.IntEnum):
    X = 0
class F:
    Y = 1
"#,
    )
}

testcase!(
    test_literal_enum_validity,
    env_literal_enum_validity(),
    r#"
import foo
from typing import Literal, assert_type
assert_type(foo.E.X, Literal[foo.E.X])

def derp() -> Literal[foo.E.X]: ...
def test() -> Literal[derp()]: ...  # E:  Invalid literal expression

def test2() -> Literal[foo.F.Y]: ... # E: `foo.F.Y` is not a valid enum member
"#,
);

testcase!(
    test_relative_import_missing_module_attribute,
    r#"
from . import foo  # E: Cannot find module `.`
    "#,
);

#[test]
fn test_interface_has_more() {
    let temp = tempfile::tempdir().unwrap();
    let mut env = TestEnv::new();
    let foo_py = temp.path().join("foo.py");
    let foo_pyi = temp.path().join("foo.pyi");

    fs_anyhow::write(&foo_py, "import foo as X\nX.Extra").unwrap();
    fs_anyhow::write(&foo_pyi, "class Extra: pass").unwrap();
    env.add_real_path("foo", foo_py);
    env.add_real_path("foo", foo_pyi);
    let _ = env.to_state();
}

#[test]
fn test_interface_disagree() {
    let temp = tempfile::tempdir().unwrap();
    let mut env = TestEnv::new();
    let foo_py = temp.path().join("foo.py");
    let foo_pyi = temp.path().join("foo.pyi");

    fs_anyhow::write(
        &foo_py,
        "class Foo:\n  def method(self): pass\nFoo().method()",
    )
    .unwrap();
    fs_anyhow::write(&foo_pyi, "").unwrap();
    env.add_real_path("foo", foo_py.clone());
    env.add_real_path("foo", foo_pyi);
    let h_py = Handle::new(
        ModuleName::from_str("foo"),
        ModulePath::filesystem(foo_py),
        env.sys_info(),
    );
    let (state, _) = env.to_state();
    let errs = state
        .transaction()
        .get_errors([&h_py])
        .collect_errors()
        .ordinary;
    assert_eq!(errs.len(), 0);
}

fn env_class_x_deprecated() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from warnings import deprecated
@deprecated("Don't use this")
class X: ...
x: X = X()
"#,
    )
}

testcase!(
    test_import_deprecated_class_warn,
    env_class_x_deprecated(),
    r#"
from foo import X # E: `X` is deprecated

x = X()
"#,
);

testcase!(
    bug = "When something is imported via *, we should warn on usage not on import",
    test_import_star_deprecated_class_warn,
    env_class_x_deprecated(),
    r#"
from foo import *

x = X()
"#,
);

fn env_func_x_deprecated() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from warnings import deprecated
@deprecated("Don't use this")
def x(): ...
"#,
    )
}

testcase!(
    test_import_deprecated_func_warn,
    env_func_x_deprecated(),
    r#"
from foo import x # E: `x` is deprecated

x()  # E: `foo.x` is deprecated
"#,
);

testcase!(
    test_import_as_deprecated_func_warn,
    env_func_x_deprecated(),
    r#"
from foo import x as y # E: `x` is deprecated

y()  # E: `foo.x` is deprecated
"#,
);

testcase!(
    test_import_star_deprecated_func_warn,
    env_func_x_deprecated(),
    r#"
from foo import *

x()  # E: `foo.x` is deprecated
"#,
);

fn env_func_x_deprecated_conditionally() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from warnings import deprecated
import sys

if sys.version_info >= (3, 10):
    @deprecated("Don't use this")
    def x(): ...
else:
    def x(): ...
"#,
    )
}

fn env_func_x_deprecated_conditionally_no_deprecation() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from warnings import deprecated
import sys

if sys.version_info < (3, 10):
    @deprecated("Don't use this")
    def x(): ...
else:
    def x(): ...
"#,
    )
}

testcase!(
    test_import_conditionally_deprecated_func_warn,
    env_func_x_deprecated_conditionally(),
    r#"
from foo import x # E: `x` is deprecated

x()  # E: `foo.x` is deprecated
"#,
);

testcase!(
    test_import_conditionally_deprecated_func_no_warn,
    env_func_x_deprecated_conditionally_no_deprecation(),
    r#"
from foo import x
# No warning for import, since the function is not deprecated in this context

x()
"#,
);

fn env_func_x_deprecated_overload_only() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from warnings import deprecated
from typing import Any, overload
@overload
def x(y: int) -> int: ...

@deprecated("Don't use this")
@overload
def x(y: str) -> str: ...

def x(y: Any) -> Any: ...
"#,
    )
}

testcase!(
    test_import_deprecated_overload_no_warn,
    env_func_x_deprecated_overload_only(),
    r#"
from foo import x
# No warning for import, since only the overload is deprecated

x("hello")  # E: Call to deprecated overload `foo.x`
"#,
);

fn env_submodule_attribute_name_collision() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("foo", "foo/__init__.py", "from .bar import bar");
    t.add_with_path("foo.bar", "foo/bar.py", "def bar(): pass");
    t
}

testcase!(
    test_submodule_attribute_name_collision,
    env_submodule_attribute_name_collision(),
    r#"
from typing import assert_type, Callable
import foo
assert_type(foo.bar, Callable[[], None])
    "#,
);

testcase!(
    test_unittest_main,
    r#"
import unittest
unittest.main()
    "#,
);

testcase!(
    test_ambiguous_pkg_attribute_vs_submodule,
    r#"
# When a package `__init__` attribute and a same-named submodule collide, the
# runtime result can be order-dependent. When the colliding name is explicitly
# re-exported by the package (as `unittest` does with
# `from .main import main as main`, where `main = TestProgram`), that binding wins
# at runtime even if the submodule is imported directly, so the call below is valid.
# See https://github.com/facebook/pyrefly/issues/322
import unittest.main
unittest.main()
    "#,
);

fn env_reexport_from_same_submodule() -> TestEnv {
    let mut t = TestEnv::new();
    // Synthetic version of the `unittest.main` case, independent of typeshed: `pkg.__init__`
    // re-exports `name` *from* the same-named submodule `pkg.name`. The `from .name` load is a
    // side effect of `__init__`, so the `name = ...` binding wins even after a direct
    // `import pkg.name`, and `pkg.name()` resolves to the callable rather than the module.
    t.add_with_path("pkg", "pkg/__init__.py", "from .name import name as name");
    t.add_with_path("pkg.name", "pkg/name.py", "def name() -> int: ...");
    t
}

testcase!(
    test_reexport_from_same_submodule_shadows,
    env_reexport_from_same_submodule(),
    r#"
# `pkg.__init__` re-exports `name` from the same-named submodule `pkg.name`, so at runtime
# the re-exported callable wins over the submodule binding even after `import pkg.name`, and
# the call below is valid. See https://github.com/facebook/pyrefly/issues/322
import pkg.name
pkg.name()
    "#,
);

fn env_reexport_from_other_submodule() -> TestEnv {
    let mut t = TestEnv::new();
    // `pkg.__init__` re-exports `name` from `pkg.other`, NOT from `pkg.name`. At runtime
    // this does not trigger a side-effect import of `pkg.name`, so a subsequent
    // `import pkg.name` binds the submodule onto `pkg` after `__init__` executes, and
    // the submodule wins over the re-exported attribute.
    t.add_with_path("pkg", "pkg/__init__.py", "from .other import name as name");
    t.add_with_path("pkg.other", "pkg/other.py", "def name() -> int: ...");
    t.add_with_path("pkg.name", "pkg/name.py", "");
    t
}

testcase!(
    test_reexport_from_other_submodule_does_not_shadow,
    env_reexport_from_other_submodule(),
    r#"
# `pkg.__init__` re-exports `name` from `pkg.other` (a *different* module than the
# same-named submodule `pkg.name`), so at runtime the direct `import pkg.name`
# below binds the submodule onto `pkg` after `__init__` and the submodule wins.
# See https://github.com/facebook/pyrefly/issues/322
import pkg.name
pkg.name()  # E: Expected a callable, got `Module[pkg.name]`
    "#,
);

fn env_reexport_from_other_submodule_with_implicit_import() -> TestEnv {
    let mut t = TestEnv::new();
    // `pkg.__init__` re-exports `name` from `pkg.other` and *also* imports the same-named
    // submodule `pkg.name` in an unrelated statement. The re-export does not originate from
    // `pkg.name`, so at runtime the submodule binding still wins even though `pkg.name` is
    // implicitly imported by `__init__`. The two facts "`name` is re-exported" and "`pkg.name`
    // is implicitly imported" must not be combined: only a re-export *sourced from* the
    // same-named submodule shadows it. See https://github.com/facebook/pyrefly/issues/322
    t.add_with_path(
        "pkg",
        "pkg/__init__.py",
        "from .other import name as name\nimport pkg.name",
    );
    t.add_with_path("pkg.other", "pkg/other.py", "def name() -> int: ...");
    t.add_with_path("pkg.name", "pkg/name.py", "");
    t
}

testcase!(
    test_reexport_from_other_submodule_with_implicit_import_does_not_shadow,
    env_reexport_from_other_submodule_with_implicit_import(),
    r#"
# `pkg.__init__` re-exports `name` from `pkg.other` but separately imports the same-named
# submodule `pkg.name`, so the submodule wins at runtime and the call below is invalid.
# The re-export (`name`) and the implicit submodule import (`pkg.name`) come from unrelated
# statements, so they must not be combined to suppress the error.
# See https://github.com/facebook/pyrefly/issues/322
import pkg.name
pkg.name()  # E: Expected a callable, got `Module[pkg.name]`
    "#,
);

fn env_re_export_special() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from typing import TypeVar
from typing import TypeVarTuple as _TypeVarTuple
from typing import TypeAlias
from typing import Annotated as MyAnnotated
"#,
    )
}

testcase!(
    test_re_exported_special,
    env_re_export_special(),
    r#"
from foo import TypeVar
import foo
from foo import TypeAlias as TA, MyAnnotated

T = TypeVar("T")
Ts = foo._TypeVarTuple("Ts")

X: TA = list[int]

x: X = [1, 2, 3]
def test(x: T, ys: tuple[*Ts]) -> tuple[T, *Ts]:
    return (x, *ys)

y: MyAnnotated[int, "hello"] = 1
"#,
);

fn env_extra_builtins() -> TestEnv {
    TestEnv::one_with_path(
        "__builtins__",
        "__builtins__.pyi",
        r#"
class X: ...
"#,
    )
}

testcase!(
    test_extra_builtins,
    env_extra_builtins(),
    r#"
x: X = X()
"#,
);

fn env_extra_builtins_shadows_builtin() -> TestEnv {
    TestEnv::one_with_path(
        "__builtins__",
        "__builtins__.pyi",
        r#"
def abs(x: object) -> str: ...
"#,
    )
}

testcase!(
    // A name defined in both the stdlib `builtins` and the user's `__builtins__.pyi`
    // resolves to the user's `__builtins__` definition, which shadows the stdlib one.
    test_extra_builtins_shadows_builtin,
    env_extra_builtins_shadows_builtin(),
    r#"
from typing import assert_type
assert_type(abs(1), str)
"#,
);

fn env_extra_builtins_in_typings() -> TestEnv {
    TestEnv::one_with_path(
        "__builtins__",
        "typings/__builtins__.pyi",
        r#"
class X: ...
"#,
    )
}

testcase!(
    test_extra_builtins_in_typings,
    env_extra_builtins_in_typings(),
    r#"
x: X = X()
"#,
);

fn env_extra_builtins_single_underscore() -> TestEnv {
    TestEnv::one_with_path(
        "__builtins__",
        "__builtins__.pyi",
        r#"
def gettext(message: str) -> str: ...
def _(message: str) -> str: ...
"#,
    )
}

testcase!(
    test_extra_builtins_single_underscore,
    env_extra_builtins_single_underscore(),
    r#"
from typing import assert_type
# Single underscore `_` is a common alias for gettext and should be exported from builtins
assert_type(gettext("a"), str)
assert_type(_("b"), str)
"#,
);

fn env_assign_to_ellipsis() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "foo_stub",
        "foo_stub.pyi",
        r#"
X = ...
"#,
    );
    env.add_with_path(
        "bar_source",
        "bar_source.py",
        r#"
Y = ...
"#,
    );
    env
}

testcase!(
    test_var_assigned_to_ellipsis,
    env_assign_to_ellipsis(),
    r#"
from foo_stub import X
from bar_source import Y
from typing import Any, assert_type, reveal_type

assert_type(X, Any)
assert_type(X.anything, Any)

reveal_type(Y)  # E: Ellipsis
Y.anything  # E: `EllipsisType` has no attribute `anything`
    "#,
);

fn env_final_value() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from typing import Final
X: Final = 42
Y: Final[int] = 42
"#,
    )
}

testcase!(
    test_modify_imported_final_value,
    env_final_value(),
    r#"
from foo import X, Y
X = 10  # E: Cannot assign to `X` because it is imported as final
Y = 10  # E: Cannot assign to `Y` because it is imported as final
"#,
);

testcase!(
    test_modify_imported_as_final_value,
    env_final_value(),
    r#"
from foo import X as x, Y as y
x = 10  # E: Cannot assign to `x` because it is imported as final
y = 10  # E: Cannot assign to `y` because it is imported as final
"#,
);

testcase!(
    test_duplicate_import_of_final_value,
    env_final_value(),
    r#"
from foo import X
from foo import X
"#,
);

testcase!(
    test_duplicate_import_of_final_value_as,
    env_final_value(),
    r#"
from foo import X as Y
from foo import X as Y
"#,
);

fn env_type_checking_reexport() -> TestEnv {
    TestEnv::one(
        "a",
        r#"
from typing import TYPE_CHECKING
"#,
    )
}

testcase!(
    test_star_import_type_checking,
    env_type_checking_reexport(),
    r#"
from typing import TYPE_CHECKING
from a import *
"#,
);

fn env_final_cross_module() -> TestEnv {
    let mut t = TestEnv::new();
    t.add(
        "a",
        r#"
from typing import Final
X: Final = 42
"#,
    );
    t.add(
        "b",
        r#"
X: int = 10
"#,
    );
    t
}

testcase!(
    test_import_final_then_import_different_value,
    env_final_cross_module(),
    r#"
from a import X
from b import X  # E: Cannot assign to `X` because it is imported as final
"#,
);

fn env_all_binop_add() -> TestEnv {
    let mut t = TestEnv::new();
    t.add(
        "base",
        r#"
__all__ = ["x", "y"]
x: int = 1
y: str = "hello"
"#,
    );
    t.add(
        "combined",
        r#"
import base
from base import *
a: float = 3.14
__all__ = ["a"] + base.__all__
"#,
    );
    t
}

testcase!(
    test_import_star_all_binop_add,
    env_all_binop_add(),
    r#"
from typing import assert_type
from combined import *
assert_type(a, float)
assert_type(x, int)
assert_type(y, str)
"#,
);

fn env_all_starred() -> TestEnv {
    let mut t = TestEnv::new();
    t.add(
        "base",
        r#"
__all__ = ["x", "y"]
x: int = 1
y: str = "hello"
"#,
    );
    t.add(
        "combined",
        r#"
import base
from base import *
a: float = 3.14
__all__ = [*base.__all__, "a"]
"#,
    );
    t
}

testcase!(
    test_import_star_all_starred,
    env_all_starred(),
    r#"
from typing import assert_type
from combined import *
assert_type(a, float)
assert_type(x, int)
assert_type(y, str)
"#,
);

fn env_all_unresolvable() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
def generate_all():
    return ["x"]

x: int = 1
y: str = "hello"
_private: float = 3.14
__all__ = generate_all()  # E: `__all__` could not be statically analyzed
"#,
    )
}

testcase!(
    test_import_star_all_unresolvable,
    env_all_unresolvable(),
    r#"
from typing import assert_type
from foo import *
# Since __all__ is unresolvable, falls back to all public names
assert_type(x, int)
assert_type(y, str)
_private  # E: Could not find name `_private`
"#,
);

testcase!(
    test_import_named_all_unresolvable,
    env_all_unresolvable(),
    r#"
from typing import assert_type
from foo import x, y
assert_type(x, int)
assert_type(y, str)
"#,
);

fn env_all_augassign_unresolvable() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "pkg",
        "pkg/__init__.py",
        r#"
from .sub import *
"#,
    );
    t.add_with_path(
        "pkg.sub",
        "pkg/sub/__init__.py",
        r#"
from ._impl import *

_extra_names = ["x"]
__all__ = ["Base"]
__all__ += _extra_names  # E: `__all__` could not be statically analyzed
"#,
    );
    t.add_with_path(
        "pkg.sub._impl",
        "pkg/sub/_impl.py",
        r#"
class Base: ...
x: int = 1
y: str = "hello"
"#,
    );
    t
}

testcase!(
    test_all_augassign_unresolvable,
    env_all_augassign_unresolvable(),
    r#"
from typing import assert_type
import pkg
# Since __all__ += variable is unresolvable, falls back to all public names,
# which includes names from `from ._impl import *`.
assert_type(pkg.x, int)
assert_type(pkg.y, str)
"#,
);

fn env_all_append_unresolvable() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
x: int = 1
y: str = "hello"
_name = "x"
__all__ = ["y"]
__all__.append(_name)  # E: `__all__` could not be statically analyzed
"#,
    )
}

testcase!(
    test_all_append_unresolvable,
    env_all_append_unresolvable(),
    r#"
from typing import assert_type
from foo import *
# Since __all__.append(variable) is unresolvable, falls back to all public names.
assert_type(x, int)
assert_type(y, str)
"#,
);

fn env_all_relative_module_ref() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "pkg",
        "pkg/__init__.py",
        r#"
from .sub import *
"#,
    );
    t.add_with_path(
        "pkg.sub",
        "pkg/sub/__init__.py",
        r#"
from . import _api
from ._api import *
__all__ = _api.__all__
"#,
    );
    t.add_with_path(
        "pkg.sub._api",
        "pkg/sub/_api.py",
        r#"
__all__ = ["convolve", "medfilt"]
def convolve(x: list[float]) -> list[float]: return x
def medfilt(x: list[float]) -> list[float]: return x
"#,
    );
    t
}

testcase!(
    test_all_relative_module_ref,
    env_all_relative_module_ref(),
    r#"
from typing import assert_type
import pkg
# __all__ = _api.__all__ should resolve _api to pkg.sub._api
assert_type(pkg.convolve([1.0]), list[float])
assert_type(pkg.medfilt([1.0]), list[float])
"#,
);

fn env_relative_import_in_subdirectory() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("test.foo", "test/foo.py", "from .foo2 import bar");
    t.add_with_path("test.foo2", "test/foo2.py", "bar: int = 100");
    t
}

testcase!(
    test_relative_import_in_subdirectory,
    env_relative_import_in_subdirectory(),
    r#"
from typing import assert_type
from test.foo import bar
assert_type(bar, int)
"#,
);

/// Create a test environment with extra-extension `.cinc` modules.
fn env_extra_ext_cinc_import() -> TestEnv {
    let mut t =
        TestEnv::new().with_extra_file_extensions(vec!["cinc".to_owned(), "cconf".to_owned()]);
    t.add_with_path(
        "service.config.cinc",
        "service/config.cinc",
        r#"
x: int = 42
class Config:
    name: str
"#,
    );
    t
}

testcase!(
    test_extra_ext_cinc_import,
    env_extra_ext_cinc_import(),
    r#"
from typing import assert_type
import service.config.cinc as config
assert_type(config.x, int)
"#,
);

// Test importing a `.cconf` file (another extra extension).
fn env_extra_ext_cconf_import() -> TestEnv {
    let mut t =
        TestEnv::new().with_extra_file_extensions(vec!["cinc".to_owned(), "cconf".to_owned()]);
    t.add_with_path(
        "service.settings.cconf",
        "service/settings.cconf",
        r#"
value: str = "hello"
"#,
    );
    t
}

testcase!(
    test_extra_ext_cconf_import,
    env_extra_ext_cconf_import(),
    r#"
from typing import assert_type
import service.settings.cconf as settings
assert_type(settings.value, str)
"#,
);

// Test importing an extra-extension file with dots in the filename.
fn env_extra_ext_dotted_filename_import() -> TestEnv {
    let mut t =
        TestEnv::new().with_extra_file_extensions(vec!["cinc".to_owned(), "cconf".to_owned()]);
    t.add_with_path(
        "foo.file1.name2.cinc",
        "foo/file1.name2.cinc",
        r#"
y: float = 3.14
"#,
    );
    t
}

testcase!(
    test_extra_ext_dotted_filename_import,
    env_extra_ext_dotted_filename_import(),
    r#"
from typing import assert_type
import foo.file1.name2.cinc as mod
assert_type(mod.y, float)
"#,
);

// Test using `from ... import` syntax with extra-extension modules.
testcase!(
    test_extra_ext_from_import,
    env_extra_ext_cinc_import(),
    r#"
from typing import assert_type
from service.config.cinc import x, Config
assert_type(x, int)
c = Config()
assert_type(c.name, str)
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2983
testcase!(
    test_malformed_def_from_star,
    r#"
def # E: Parse error: Expected an identifier
from *a # E: Parse error: Expected `)`, found `from` # E: Parse error: Expected a module name # E: Parse error: Star import must be the only import # E: Parse error: Expected `,`, found name
"#,
);

// Additional regression test for https://github.com/facebook/pyrefly/issues/2983
testcase!(
    test_malformed_class_from_star,
    r#"
class # E: Parse error: Expected an identifier
from *a # E: Parse error: Expected an indented block after `class` definition # E: Parse error: Expected a module name # E: Parse error: Star import must be the only import # E: Parse error: Expected `,`, found name # E: Cannot find module `main`
"#,
);

const PKGUTIL_INIT: &str =
    "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n";

#[test]
fn test_pkgutil_namespace_package_multi_root() {
    // Two search roots both contain `ns/__init__.py` with `pkgutil.extend_path`.
    // Both register as LegacyNamespacePackage and accumulate, so submodules from
    // both roots are importable through the merged namespace.
    let tempdir = tempfile::tempdir().unwrap();
    let root = tempdir.path();
    std::fs::create_dir_all(root.join("root0/ns/baz")).unwrap();
    std::fs::create_dir_all(root.join("root1/ns/bar")).unwrap();
    std::fs::write(root.join("root0/ns/__init__.py"), PKGUTIL_INIT).unwrap();
    std::fs::write(
        root.join("root0/ns/baz/__init__.py"),
        "def helper() -> int: return 0\n",
    )
    .unwrap();
    std::fs::write(root.join("root1/ns/__init__.py"), PKGUTIL_INIT).unwrap();
    std::fs::write(root.join("root1/ns/bar/__init__.py"), "class Foo: ...\n").unwrap();

    let mut env =
        TestEnv::new().with_site_package_paths(vec![root.join("root0"), root.join("root1")]);
    env.add_with_path(
        "main",
        "main.py",
        "from ns.bar import Foo\nfrom ns.baz import helper\n",
    );
    let (state, handle_fn) = env.to_state();
    state
        .transaction()
        .get_errors(&[handle_fn("main")])
        .check_against_expectations()
        .unwrap();
}

#[test]
fn test_pkgutil_namespace_absorbs_implicit_namespace() {
    // An earlier search root contributes only an implicit (PEP 420) `ns/`
    // directory; a later root has `ns/__init__.py` with `pkgutil.extend_path`.
    // The LNP must absorb the prior implicit namespace dir so submodules from
    // both roots are importable. This mirrors the layout produced by editable
    // installs of `extend_path`-style namespace packages (e.g. azure-sdk-for-python).
    let tempdir = tempfile::tempdir().unwrap();
    let root = tempdir.path();
    std::fs::create_dir_all(root.join("root0/ns/baz")).unwrap();
    std::fs::create_dir_all(root.join("root1/ns/bar")).unwrap();
    // root0 has no ns/__init__.py — implicit namespace.
    std::fs::write(
        root.join("root0/ns/baz/__init__.py"),
        "def helper() -> int: return 0\n",
    )
    .unwrap();
    std::fs::write(root.join("root1/ns/__init__.py"), PKGUTIL_INIT).unwrap();
    std::fs::write(root.join("root1/ns/bar/__init__.py"), "class Foo: ...\n").unwrap();

    let mut env =
        TestEnv::new().with_site_package_paths(vec![root.join("root0"), root.join("root1")]);
    env.add_with_path(
        "main",
        "main.py",
        "from ns.bar import Foo\nfrom ns.baz import helper\n",
    );
    let (state, handle_fn) = env.to_state();
    state
        .transaction()
        .get_errors(&[handle_fn("main")])
        .check_against_expectations()
        .unwrap();
}

// ----------------------------------------------------------------------------
// Cross-module class rebind tests: importers should observe whichever class
// the visible result chose. See `assign.rs` for the same-module regressions.
// ----------------------------------------------------------------------------

// Regression test for https://github.com/facebook/pyrefly/issues/1378
fn env_singleton() -> TestEnv {
    TestEnv::one(
        "singleton",
        r#"
class GlobalInventory:
    _instance: GlobalInventory | None = None
    _initialized: bool = False
    initialization_status: bool
    config_file_inventory: dict[str, int]

    def __new__(cls) -> GlobalInventory:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        assert cls._instance is not None
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.initialization_status = False
            self.config_file_inventory = {}
            GlobalInventory._initialized = True

globals_inv = GlobalInventory()
"#,
    )
}

testcase!(
    test_cross_module_singleton_attribute_access,
    env_singleton(),
    r#"
from singleton import globals_inv

class Foo:
    def __init__(self) -> None:
        globals_inv.config_file_inventory = {"foo": 1}
        for _ in globals_inv.config_file_inventory.values():
            if globals_inv.initialization_status:
                break
        globals_inv.initialization_status = True
"#,
);

fn env_class_rebind_incompatible() -> TestEnv {
    TestEnv::one(
        "mod",
        r#"
class Real:
    def __init__(self, host: str, port: int = 0) -> None: ...

class Dummy: ...

def b() -> bool: ...

if b():
    Real = Dummy  # E: `type[Dummy]` is not assignable to variable `Real` with type `type[Real]`
"#,
    )
}

fn env_class_rebind_compatible() -> TestEnv {
    TestEnv::one(
        "mod",
        r#"
class Real:
    def __init__(self, host: str, port: int = 0) -> None: ...

class SubReal(Real):
    def __init__(self, host: str, port: int = 0) -> None: ...

Real = SubReal
"#,
    )
}

testcase!(
    test_class_rebind_import_after_incompatible_write,
    env_class_rebind_incompatible(),
    r#"
from typing import reveal_type
from mod import Real
reveal_type(Real)  # E: revealed type: type[Real]
Real("example.com", port=443)
"#,
);

testcase!(
    test_class_rebind_import_after_compatible_write,
    env_class_rebind_compatible(),
    r#"
from typing import reveal_type
from mod import Real
reveal_type(Real)  # E: revealed type: type[SubReal]
Real("example.com", port=443)
"#,
);

fn env_typevar_param_with_default() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from types import MappingProxyType
from typing import Mapping
class Base[VT]:
    def _update(self, kw: Mapping[str, VT] = MappingProxyType({})) -> None: ...
    "#,
    )
}

testcase!(
    test_import_class_with_typevar_param_with_default,
    env_typevar_param_with_default(),
    r#"
import foo
class Child(foo.Base):
    def put(self):
        return self._update()
    "#,
);

// --- Special import function tests (import_thrift, import_python) ---

/// Create a test environment with a `.thrift` module for import_thrift tests.
fn env_import_thrift() -> TestEnv {
    let mut t =
        TestEnv::new().with_extra_file_extensions(vec!["thrift".to_owned(), "cinc".to_owned()]);
    t.add_with_path(
        "service.types.thrift",
        "service/types.thrift",
        r#"
class MyConfig:
    value: int
"#,
    );
    t
}

// Test import_thrift with wildcard import.
testcase!(
    test_import_thrift_wildcard,
    env_import_thrift(),
    r#"
from typing import assert_type
import_thrift("service/types.thrift", "*")
c = MyConfig()
assert_type(c.value, int)
"#,
);

// Test import_thrift with alias.
testcase!(
    test_import_thrift_alias,
    env_import_thrift(),
    r#"
from typing import assert_type
import_thrift("service/types.thrift", "thrift_mod")
c = thrift_mod.MyConfig()
assert_type(c.value, int)
"#,
);

// Test import_thrift with no second argument (defaults to wildcard).
testcase!(
    test_import_thrift_default_wildcard,
    env_import_thrift(),
    r#"
from typing import assert_type
import_thrift("service/types.thrift")
c = MyConfig()
assert_type(c.value, int)
"#,
);

// Test import_python with wildcard import.
testcase!(
    test_import_python_wildcard,
    env_import_thrift(),
    r#"
from typing import assert_type
import_python("service/types.thrift", "*")
c = MyConfig()
assert_type(c.value, int)
"#,
);

// Test import_thrift with dot separators.
testcase!(
    test_import_thrift_dot_separator,
    env_import_thrift(),
    r#"
from typing import assert_type
import_thrift("service.types.thrift", "*")
c = MyConfig()
assert_type(c.value, int)
"#,
);

// Test two special import alias calls in the same file.
fn env_special_import_two_aliases() -> TestEnv {
    let mut t = TestEnv::new().with_extra_file_extensions(vec!["thrift".to_owned()]);
    t.add_with_path(
        "service.types.thrift",
        "service/types.thrift",
        r#"
class MyConfig:
    value: int
"#,
    );
    t.add_with_path(
        "service.other.thrift",
        "service/other.thrift",
        r#"
class OtherConfig:
    name: str
"#,
    );
    t
}

testcase!(
    test_special_import_two_aliases,
    env_special_import_two_aliases(),
    r#"
from typing import assert_type
import_thrift("service/types.thrift", "types_mod")
import_thrift("service/other.thrift", "other_mod")
c = types_mod.MyConfig()
assert_type(c.value, int)
o = other_mod.OtherConfig()
assert_type(o.name, str)
"#,
);

// Test importing from a module that uses a special import with alias.
fn env_special_import_alias_module() -> TestEnv {
    let mut t =
        TestEnv::new().with_extra_file_extensions(vec!["thrift".to_owned(), "cinc".to_owned()]);
    t.add_with_path(
        "service.types.thrift",
        "service/types.thrift",
        r#"
class MyConfig:
    value: int
"#,
    );
    // This .cinc module uses import_thrift with an alias and re-exports the type.
    t.add_with_path(
        "helper.cinc",
        "helper.cinc",
        r#"import_thrift("service/types.thrift", "types_mod")

def get_config() -> types_mod.MyConfig:
    return types_mod.MyConfig()
"#,
    );
    t
}

testcase!(
    test_special_import_alias_module,
    env_special_import_alias_module(),
    r#"
from typing import assert_type
from helper.cinc import get_config
c = get_config()
assert_type(c.value, int)
"#,
);

// Test special import alias referenced inside a function body with
// an unresolvable module.
fn env_special_import_alias_in_function() -> TestEnv {
    let mut t =
        TestEnv::new().with_extra_file_extensions(vec!["thrift".to_owned(), "cinc".to_owned()]);
    // .cinc module with import_thrift alias used inside a function body.
    // The target module doesn't exist, so the alias is typed as Any.
    t.add_with_path(
        "helper.cinc",
        "helper.cinc",
        r#"import_thrift("nonexistent/types.thrift", "deletion_types")

def get_plan():
    return deletion_types.DeletionPlan()
"#,
    );
    t
}

testcase!(
    test_special_import_alias_in_function,
    env_special_import_alias_in_function(),
    r#"
from helper.cinc import get_plan
x = get_plan()
"#,
);

// Test import_thrift for unresolvable module (no error, names typed as Any).
testcase!(
    test_import_thrift_unresolvable,
    env_import_thrift(),
    r#"
import_thrift("nonexistent/module.thrift", "mod")
"#,
);

// --- Extra extension .pyi type stub tests ---

fn env_pyi_type_stub() -> TestEnv {
    // Module name ends with an extra extension but path is a .pyi stub.
    let mut t = TestEnv::new().with_extra_file_extensions(vec!["thrift".to_owned()]);
    t.add_with_path(
        "service.types.thrift",
        "stubs/service/types.thrift.pyi",
        r#"
class MyRecord:
    name: str
"#,
    );
    t
}

// Test that `from module.ext import Name` works with .ext.pyi stubs.
testcase!(
    test_pyi_type_stub_import,
    env_pyi_type_stub(),
    r#"
from typing import assert_type
from service.types.thrift import MyRecord
r = MyRecord()
assert_type(r.name, str)
"#,
);

// Test that `import module.ext as alias` works with .ext.pyi stubs.
testcase!(
    test_pyi_type_stub_import_alias,
    env_pyi_type_stub(),
    r#"
from typing import assert_type
import service.types.thrift as types_mod
r = types_mod.MyRecord()
assert_type(r.name, str)
"#,
);

// --- __files__ directory import tests ---

// Test that `import foo.__files__ as alias` binds alias without errors.
testcase!(
    test_import_files_directory,
    r#"
import myproject.schemas.__files__ as schema_files
x = schema_files
"#,
);

// Test that `import foo.__recursefiles__ as alias` also works.
testcase!(
    test_import_recursefiles_directory,
    r#"
import some.dir.__recursefiles__ as all_files
x = all_files
"#,
);

// Test that __files__ aliases work inside function bodies without crashing.
testcase!(
    test_import_files_directory_in_function,
    r#"
import myproject.data.__files__ as data_schema_files

def get_files():
    return data_schema_files
"#,
);

fn env_special_export_package_reexport() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path("pkg", "pkg/__init__.py", "");
    t.add_with_path(
        "pkg.my_typing",
        "pkg/my_typing.py",
        "from typing import Annotated",
    );
    t
}

testcase!(
    test_special_export_package_reexport_import,
    env_special_export_package_reexport(),
    r#"
from pkg import my_typing as mt
x: mt.Annotated[int, "metadata"] = 5
"#,
);
