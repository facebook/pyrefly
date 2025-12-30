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
from foo import bad_definition  # E: Could not import `bad_definition` from `foo`
"#,
);

testcase!(
    test_export_all_wrongly_star,
    env_export_all_wrongly(),
    r#"
from foo import *  # E: Could not import `bad_definition` from `foo`
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
from foo import bar  # E: Could not find import of `foo`
"#,
);

testcase!(
    test_missing_import_star,
    r#"
from foo import *  # E: Could not find import of `foo`
"#,
);

testcase!(
    test_missing_import_module,
    r#"
import foo, bar.baz  # E: Could not find import of `foo`  # E: Could not find import of `bar.baz`
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
        .shown;
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
from . import foo  # E: Could not find import of `.`
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
        .shown;
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
    bug = "In the presence of collisions, the runtime result can be order-dependent",
    test_ambiguous_pkg_attribute_vs_submodule,
    r#"
# The behavior when a package `__init__` attribute and a submodule collide can
# be ambiguous. We currently do not model this fully, and instead use the distinction
# between `import pkg.xyz` vs `from pkg import xyz` (which is correlated with the
# runtime behavior but not entirely the same) to approximate.
#
# One example where this leads us astray is with unittest.main; the following
# snippet works at runtime:
import unittest.main
unittest.main()  # E: Expected a callable, got `Module[unittest.main]`
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
