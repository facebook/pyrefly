/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the `State` object.

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::print_errors;
use crate::state::errors::Errors;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::subscriber::TestSubscriber;
use crate::test::util::init_test;

#[derive(Default, Clone, Dupe, Debug)]
struct IncrementalData(Arc<Mutex<SmallMap<ModuleName, Arc<String>>>>);

/// Helper for writing incrementality tests.
struct Incremental {
    data: IncrementalData,
    require: Option<Require>,
    state: State,
    to_set: Vec<(String, String)>,
}

/// What happened when we ran an incremental check.
struct IncrementalResult {
    changed: Vec<String>,
    errors: Errors,
}

impl IncrementalResult {
    fn check_recompute(&self, want: &[&str]) {
        let mut want = want.map(|x| (*x).to_owned());
        want.sort();
        assert_eq!(want, self.changed);
    }

    fn check_recompute_dedup(&self, want: &[&str]) {
        let mut changed = self.changed.clone();
        changed.dedup();
        let mut want = want.map(|x| (*x).to_owned());
        want.sort();
        assert_eq!(want, changed);
    }

    fn check_errors(&self) {
        self.errors.check_against_expectations().unwrap();
    }
}

impl Incremental {
    const USER_FILES: &[&str] = &["main", "foo", "bar", "baz"];

    fn new() -> Self {
        init_test();
        let data = IncrementalData::default();

        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        let mut sourcedb = MapDatabase::new(config.get_sys_info());
        for file in Self::USER_FILES {
            sourcedb.insert(
                ModuleName::from_str(file),
                ModulePath::memory(PathBuf::from(file)),
            );
        }
        config.source_db = Some(ArcId::new(Box::new(sourcedb)));
        config.configure();
        let config = ArcId::new(config);

        Self {
            data: data.dupe(),
            require: None,
            state: State::new(ConfigFinder::new_constant(config)),
            to_set: Vec::new(),
        }
    }

    /// Change this file to these contents, expecting this number of errors.
    fn set(&mut self, file: &str, contents: &str) {
        self.to_set.push((file.to_owned(), contents.to_owned()));
    }

    fn handle(&self, x: &str) -> Handle {
        Handle::new(
            ModuleName::from_str(x),
            ModulePath::memory(PathBuf::from(x)),
            SysInfo::default(),
        )
    }

    fn unchecked(&mut self, want: &[&str]) -> IncrementalResult {
        let subscriber = TestSubscriber::new();
        let mut transaction = self.state.new_committable_transaction(
            self.require.unwrap_or(Require::Errors),
            Some(Box::new(subscriber.dupe())),
        );
        for (file, contents) in mem::take(&mut self.to_set) {
            let contents = Arc::new(contents.to_owned());
            self.data
                .0
                .lock()
                .insert(ModuleName::from_str(&file), contents.dupe());
            transaction.as_mut().set_memory(vec![(
                PathBuf::from(file),
                Some(Arc::new(FileContents::Source(contents))),
            )]);
        }

        let handles = want.map(|x| self.handle(x));
        self.state.run_with_committing_transaction(
            transaction,
            &handles,
            self.require.unwrap_or(Require::Everything),
            None,
        );
        let loaded = Self::USER_FILES.map(|x| self.handle(x));
        let errors = self.state.transaction().get_errors(&loaded);
        let project_root = PathBuf::new();
        print_errors(project_root.as_path(), &errors.collect_errors().shown);

        let mut changed = Vec::new();
        for (x, (count, _)) in subscriber.finish() {
            let m = x.module();
            if self.data.0.lock().contains_key(&m) {
                for _ in 0..count {
                    changed.push(m.as_str().to_owned());
                }
            }
        }
        changed.sort();
        IncrementalResult { changed, errors }
    }

    /// Run a check. Expect to recompute things to have changed and errors from # E: <> comments.
    fn check(&mut self, want: &[&str], recompute: &[&str]) -> IncrementalResult {
        let res = self.unchecked(want);
        res.check_errors();
        res.check_recompute(recompute);
        res
    }

    /// Run a check. Expect to recompute things to have changed, but ignore error comments.
    fn check_ignoring_expectations(
        &mut self,
        want: &[&str],
        recompute: &[&str],
    ) -> IncrementalResult {
        let res = self.unchecked(want);
        res.check_recompute(recompute);
        res
    }
}

#[test]
#[should_panic]
fn test_incremental_inception_errors() {
    let mut i = Incremental::new();
    i.set("main", "i: int = 'test'");
    i.check(&["main"], &["main"]);
}

#[test]
#[should_panic]
fn test_incremental_inception_recompute() {
    let mut i = Incremental::new();
    i.set("main", "i: int = 1");
    i.check(&["main"], &["main", "foo"]);
}

#[test]
fn test_in_memory_updated_content_recheck() {
    let mut i = Incremental::new();
    i.set("main", "unbound_name # E:");
    i.check(&["main"], &["main"]);
    i.set("main", "bound_name = 3");
    i.check(&["main"], &["main"]);
}

#[test]
fn test_incremental_minimal_recompute() {
    let mut i = Incremental::new();
    i.set("main", "import foo; x = foo.x");
    i.set("foo", "x = 7");
    i.check(&["main"], &["main", "foo"]);
    i.set("foo", "x = 'test'");
    i.check(&["main"], &["main", "foo"]);
    i.set("foo", "x = 'test' # still");
    i.check(&["main"], &["foo"]);
    i.set("main", "import foo; x = foo.x # still");
    i.check(&["main"], &["main"]);

    // We stop depending on `foo`, so no longer have to recompute it even though it is dirty.
    // However, our current state algorithm does so anyway as it can be cheaper to compute
    // everything than do careful graph traversal.
    i.set("foo", "x = True");
    i.set("main", "x = 7");
    i.check(&["main"], &["main", "foo"]); // `foo` is not required here
    i.set("main", "import foo; x = foo.x # still");
    i.check(&["main"], &["main"]); // `foo` is required by this point
}

#[test]
fn test_incremental_cyclic() {
    let mut i = Incremental::new();
    i.set("foo", "import bar; x = 1; y = bar.x");
    i.set("bar", "import foo; x = True; y = foo.x");
    i.check(&["foo"], &["foo", "bar"]);
    i.set("foo", "import bar; x = 1; y = bar.x # still");
    i.check(&["foo"], &["foo"]);
    i.set("foo", "import bar; x = 'test'; y = bar.x");
    i.check(&["foo"], &["foo", "bar"]);
}

/// Check that the interface is consistent as we change things.
fn test_interface_consistent(code: &str) {
    let mut i = Incremental::new();
    i.set("main", code);
    i.check(&["main"], &["main"]);
    let base = i
        .state
        .transaction()
        .get_solutions(&i.handle("main"))
        .unwrap();

    i.set("main", &format!("{code} # after"));
    i.check(&["main"], &["main"]);
    let suffix = i
        .state
        .transaction()
        .get_solutions(&i.handle("main"))
        .unwrap();

    i.set("main", &format!("# before\n{code}"));
    i.check(&["main"], &["main"]);
    let prefix = i
        .state
        .transaction()
        .get_solutions(&i.handle("main"))
        .unwrap();

    let same = base.first_difference(&base);
    let suffix = suffix.first_difference(&base);
    let prefix = prefix.first_difference(&base);
    assert!(same.is_none(), "{code:?} led to {same:?}");
    assert!(suffix.is_none(), "{code:?} led to {suffix:?}");
    assert!(prefix.is_none(), "{code:?} led to {prefix:?}");
}

#[test]
fn test_interfaces_simple() {
    test_interface_consistent("x: int = 1\ndef f(y: bool) -> list[str]: return []");

    // Important to have a class with a field, as those also have positions
    test_interface_consistent("class X: y: int");
}

#[test]
fn test_interfaces_generic() {
    // Requires dealing with Forall.
    test_interface_consistent("def f[X](x: X) -> X: ...");
    test_interface_consistent(
        "
from typing import TypeVar, Generic
T = TypeVar('T')
class C(Generic[T]): pass",
    );
    test_interface_consistent("class C[T]: x: T");
}

#[test]
fn test_interfaces_counterexamples() {
    // These all failed at one point or another.

    test_interface_consistent(
        "
from typing import TypeVar, Generic
T = TypeVar('T')
class C(Generic[T]): x: T",
    );

    test_interface_consistent(
        "
from typing import TypeVar, Generic
T = TypeVar('T')
class C(Generic[T]): pass
class D(C[T]): pass",
    );

    test_interface_consistent(
        "
from typing import TypeVar
class C: pass
T = TypeVar('T', bound=C)",
    );

    test_interface_consistent(
        "
class C[R]:
    def __init__(self, field: R) -> None:
        self.field = R
",
    );
}

#[test]
fn test_error_clearing_on_dependency() {
    let mut i = Incremental::new();

    i.set("foo", "def xyz() -> int: ...");
    i.set(
        "main",
        "from foo import x # E: Could not import `x` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();

    assert!(
        !errors.shown.is_empty(),
        "Expected errors before fixing the dependency"
    );

    i.set("foo", "def x() -> int: ...");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after_fix = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after_fix.shown.is_empty(),
        "Expected errors after fixing the dependency"
    );
}

#[test]
fn test_error_clearing_on_dependency_star_import() {
    let mut i = Incremental::new();

    i.set("foo", "def xyz() -> int: ...");
    i.set(
        "main",
        "from foo import *\ny = x # E: Could not find name `x`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();

    assert!(
        !errors.shown.is_empty(),
        "Expected errors before fixing the dependency"
    );

    i.set("foo", "def xyz() -> int: ...\nx = 1");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after_fix = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after_fix.shown.is_empty(),
        "Expected no errors after fixing the dependency"
    );
}

#[test]
fn test_failed_import_invalidation_via_rdeps() {
    // This tests that when a module's exports change to satisfy a previously-failed import,
    // the module with the failed import is invalidated even if it's not explicitly requested.
    //
    // The key difference from test_error_clearing_on_dependency is that we DON'T include
    // the module with the failed import in the `want` list - we only request a third module
    // that depends on it transitively.
    let mut i = Incremental::new();

    // Setup: bar has a failed import from foo
    i.set("foo", "x = 1"); // foo exists but doesn't export `y`
    i.set("bar", "from foo import y"); // bar tries to import y - FAILS
    i.set("main", "import bar"); // main imports bar

    // Initial check - all modules computed
    i.unchecked(&["main"]);

    // Now foo exports `y` - bar's failed import should now succeed
    i.set("foo", "y = 2");

    // Only request main, NOT bar directly.
    // Before the fix: bar wouldn't be invalidated because:
    //   - bar is not in foo's rdeps (the import failed)
    //   - bar is not in the `want` list
    // After the fix: invalidate_failed_imports_from scans for failed imports and invalidates bar
    let res = i.unchecked(&["main"]);

    // bar should be recomputed because its failed import now succeeds
    assert!(
        res.changed.contains(&"bar".to_owned()),
        "bar should have been recomputed, but changed = {:?}",
        res.changed
    );
}

#[test]
fn test_stale_class() {
    let mut i = Incremental::new();
    i.set("foo", "class C: x: int = 1");
    i.set("bar", "from foo import C; c = C");
    i.set("main", "from bar import c; v = c.x");
    i.check(&["main"], &["main", "foo", "bar"]);

    i.set("foo", "");
    i.set("main", "from bar import c; v = c.x # hello");
    let res = i.unchecked(&["main", "foo"]);
    res.check_recompute_dedup(&["main", "foo", "bar"]);
    assert_eq!(res.errors.collect_errors().shown.len(), 1);
}

#[test]
fn test_stale_typed_dict() {
    let mut i = Incremental::new();

    // We need to set up a dep chain of size 4 (i.e. main -> bar -> baz -> foo) to more reliably
    // force `main` to see a stale TypedDict in `bar` during the recheck.
    // It may still be possible to hide the staleness in certain circumstances, but that's fine since
    // the test would still pass in those cases.
    i.set(
        "foo",
        "from typing import TypedDict\nclass D(TypedDict):\n  x: int",
    );
    i.set("bar", "from foo import D\nclass D2:\n  y: D");
    i.set("baz", "from bar import D2\nclass D3:\n  z: D2");
    i.set(
        "main",
        "from baz import D3\ndef test(d: D3) -> None:\n  d.z.y[\'x\']",
    );
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    i.set("foo", "class D: x: int");

    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar", "baz"]);
}

#[test]
fn test_dueling_typevar() {
    // TypeVar (and ParamSpec, TypeVarTuple) are implemented in a way that means
    // grabbing the same value from different modules in conjunction with incremental
    // updates can lead to equal TypeVar's being considered non-equal.
    //
    // Is that a problem? Yes. Is it a real problem? Perhaps no? If you write code
    // that relies on the equality of a single TypeVar imported through two routes,
    // you are really confusing the users.
    //
    // Why does it occur? Because TypeVar has equality via ArcId, so each created
    // TypeVar is different from all others. To check for interface stability
    // we try and find a mapping for equivalent TypeVar values, using TypeEq.
    // So even though your TypeVar changes, it doesn't invalidate your interface.
    // But that means you can construct an example where someone else exports
    // your TypeVar, and they don't invalidate, and then you can have a third
    // module import both and see a discrepancy.
    //
    // How to fix it? Stop TypeVar using ArcId and instead make it identified by
    // an index within the module and the QName, just like we did for class.
    //
    // Should we make that fix? Maybe? But it's not high on the priority list.
    // And the new generic syntax makes it even less important.

    let mut i = Incremental::new();
    i.set("foo", "from typing import TypeVar\nT = TypeVar('T')");
    i.set("bar", "from foo import T");
    i.set(
        "main",
        "import foo\nimport bar\nfrom typing import Any\ndef f() -> Any: ...; x: foo.T = f(); y: bar.T = x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    i.set("foo", "from typing import TypeVar\nT = TypeVar('T') #");
    i.check(&["main"], &["foo"]);

    // Observe that foo.T and bar.T are no longer equal.
    i.set(
        "main",
        "import foo\nimport bar\nfrom typing import Any\ndef f() -> Any: ...; x: foo.T = f(); y: bar.T = x # E: `TypeVar[T]` is not assignable to `TypeVar[T]`",
    );
    i.check(&["main"], &["main"]);
}

#[test]
fn test_incremental_cycle_class() {
    let mut i = Incremental::new();
    i.set("foo", "from bar import Cls");
    i.set(
        "bar",
        r#"
from foo import Cls as C
class Cls:
    def fld(self): return 1
def f(c: C):
    print(c.fld)
"#,
    );
    i.check(&["foo", "bar"], &["foo", "bar"]);

    i.set(
        "bar",
        r#"
from foo import Cls as C
class Cls:
    def fld2(self): return 1
def f(c: C):
    print(c.fld) # E: Object of class `Cls` has no attribute `fld`
"#,
    );
    i.unchecked(&["foo"]); // Used to panic
}

#[test]
fn test_incremental_rdeps() {
    // Make sure we hit the rdeps case
    let mut i = Incremental::new();
    i.require = Some(Require::Everything); // So we don't invalidate based on require
    i.set("foo", "import bar\nclass C: pass\nx = bar.z");
    i.set("bar", "import foo\nz = foo.C\nq: type[foo.C] = foo.x");
    i.check(&["foo"], &["foo", "bar"]);

    i.set("foo", "import bar\nclass Q: pass\nx = bar.z\nclass C: pass");
    i.check(&["foo"], &["bar", "bar", "foo", "foo"]);

    i.check(&["foo", "bar"], &[]); // Nothing appears dirty
}

#[test]
fn test_incremental_rdeps_with_new() {
    // Make sure we hit the rdeps case
    let mut i = Incremental::new();
    i.require = Some(Require::Everything); // So we don't invalidate based on require
    i.set("foo", "import bar\nclass C: pass\nx = bar.z");
    i.set("bar", "import foo\nz = foo.C\nq: type[foo.C] = foo.x");
    i.check(&["foo"], &["foo", "bar"]);

    i.set(
        "foo",
        "import bar\nimport baz\nclass Q: pass\nx = bar.z\nclass C: pass",
    );
    i.set("baz", "import bar");
    i.unchecked(&["foo"]);

    i.check(&["foo", "bar", "baz"], &[]); // Nothing appears dirty
}

/// Test fine-grained dependency tracking: changing an unrelated export should NOT
/// trigger recomputation of a module that only imports a different export.
#[test]
fn test_fine_grained_unrelated_export_no_recompute() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set("main", "from foo import x\nz = x + 1");
    i.check(&["main"], &["main", "foo"]);

    // Change only `y` - main should NOT be recomputed since it only imports `x`
    i.set("foo", "x: int = 1\ny: str = 'changed'");
    i.check(&["main"], &["foo"]);
}

/// Test fine-grained dependency tracking: changing the imported export SHOULD
/// trigger recomputation.
#[test]
fn test_fine_grained_related_export_recompute() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set("main", "from foo import x\nprint(x)");
    i.check(&["main"], &["main", "foo"]);

    // Change `x` - main SHOULD be recomputed since it imports `x`
    i.set("foo", "x: str = 'changed'\ny: int = 2");
    i.check(&["main"], &["foo", "main"]);
}

/// Test fine-grained tracking with unused `import foo` style.
#[test]
fn test_import_module_regular_import_unused() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set("main", "import foo\nz = foo.x");
    i.check(&["main"], &["main", "foo"]);

    i.set("foo", "x: int = 1\ny: str = 'changed'");
    i.check(&["main"], &["foo"]);
}

/// Test mixed import styles: `from foo import x` followed by `import foo`.
/// Should only depend on x.
#[test]
fn test_mixed_import_depends_on_all() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set(
        "main",
        "from foo import x\nimport foo\nprint(x); print(foo.y)",
    );
    i.check(&["main"], &["main", "foo"]);

    // Change only `y` - main SHOULD be recomputed because of `import foo`
    i.set("foo", "x: int = 1\ny: str = 'changed'");
    i.check(&["main"], &["foo", "main"]);
}

/// Test incremental behavior with `import foo; bar(foo)` pattern.
#[test]
fn test_import_module_as_argument() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1");
    i.set(
        "bar",
        "import types\ndef process(m: types.ModuleType) -> int: return m.x",
    );
    i.set("main", "import foo\nimport bar\nresult = bar.process(foo)");
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change `x` in foo - main should not recompute since it doesn't depend on the type
    i.set("foo", "x: str = 'changed'");
    i.check(&["main"], &["foo"]);
}

/// Test transitive export addition: when a module adds an export that a downstream
/// module re-exports, consumers of the re-export should see the error go away.
///
/// Scenario:
/// - foo (a.py): initially empty
/// - bar (b.py): `from foo import *` (re-exports everything from foo)
/// - main (c.py): `from bar import x` (fails because foo doesn't export x)
///
/// After foo adds `x = 1`, main's import should succeed.
#[test]
fn test_transitive_export_addition_clears_error() {
    let mut i = Incremental::new();

    // Initial state: foo is empty, bar re-exports from foo, main tries to import x from bar
    i.set("foo", "");
    i.set("bar", "from foo import *");
    i.set(
        "main",
        "from bar import x # E: Could not import `x` from `bar`",
    );
    i.check(&["main", "foo", "bar"], &["main", "foo", "bar"]);

    let main_handle = i.handle("main");

    // Verify there's an error before the fix
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected errors before foo exports x"
    );

    // Now foo exports x - main's import should succeed
    i.set("foo", "x = 1");
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);

    // Verify the error is gone
    let errors_after_fix = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after_fix.shown.is_empty(),
        "Expected no errors after foo exports x, but got: {:?}",
        errors_after_fix.shown
    );
}

/// Test that when a type is used via inference (not explicitly imported),
/// changes to that type still trigger recomputation.
///
/// Scenario:
/// - foo: exports class A with field x: int
/// - bar: imports A, creates instance, and re-exports it
/// - main: imports the instance from bar (gets type A via inference, not import)
///
/// When A's field type changes, main should see the update.
#[test]
fn test_inferred_type_changes_trigger_recompute() {
    let mut i = Incremental::new();

    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\ninstance = A()");
    i.set("main", "from bar import instance\ny = instance.x + 1");
    i.check(&["main", "foo", "bar"], &["main", "foo", "bar"]);

    // Change A's field type from int to str - main's arithmetic should now fail
    i.set("foo", "class A:\n    x: str = 'hello'");
    i.set(
        "main",
        "from bar import instance\ny = instance.x + 1 # E: `+` is not supported",
    );
    i.check(&["main"], &["foo", "bar", "main"]);
}

/// Test that when a function's return type changes, callers that import only
/// the function (not the return type) still see the update.
///
/// Scenario:
/// - foo: exports class A with field x and function get_a() -> A
/// - main: imports only get_a, uses the returned value's field
///
/// When A's field type changes, main should see the update.
#[test]
fn test_function_return_type_changes_trigger_recompute() {
    let mut i = Incremental::new();

    i.set(
        "foo",
        "class A:\n    x: int = 1\ndef get_a() -> A:\n    return A()",
    );
    i.set(
        "main",
        "from foo import get_a\nval = get_a()\ny = val.x + 1",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    // Change A's field type from int to str - main's arithmetic should now fail
    i.set(
        "foo",
        "class A:\n    x: str = 'hello'\ndef get_a() -> A:\n    return A()",
    );
    i.set(
        "main",
        "from foo import get_a\nval = get_a()\ny = val.x + 1 # E: `+` is not supported",
    );
    i.check(&["main"], &["foo", "main"]);
}

/// Test that non-overlapping export changes do NOT trigger false cycle detection.
///
/// This simulates a pattern similar to PyTorch's torch.distributed.pipelining.stage module,
/// which imports and re-exports from multiple independent sources:
///   - `from torch.distributed._composable.replicate_with_fsdp import replicate`
///   - `from torch.distributed.fsdp import fully_shard`
///
/// When both sources change in different epochs, the same module (stage) appears multiple
/// times in the change propagation - but with DIFFERENT exports. This should NOT be
/// treated as a cycle because:
///   1. The exports don't overlap - they're independent re-exports
///   2. Each export chain will stabilize independently
///   3. There's no infinite loop risk since no single export keeps changing
#[test]
fn test_non_overlapping_exports_no_false_cycle() {
    let mut i = Incremental::new();

    i.set("foo", "x: int = 1");
    i.set("bar", "y: int = 2");
    i.set("baz", "from foo import x\nfrom bar import y");
    i.set("main", "from baz import x, y\nprint(x, y)");
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    // Now change BOTH sources simultaneously. The hub module will:
    // 1. First re-export the changed `x` (from foo)
    // 2. Then re-export the changed `y` (from bar)
    // This should NOT trigger cycle detection since the exports don't overlap.
    i.set("foo", "x: str = 'changed_x'");
    i.set("bar", "y: str = 'changed_y'");
    i.check(&["main"], &["foo", "bar", "baz", "main"]);
}

/// Test that overlapping export changes DO trigger proper cycle detection.
///
/// A true cycle occurs when the same export keeps changing:
///   - Export X in A depends on export Y in B
///   - Export Y in B depends on export X in A
///   - X changes -> Y changes -> X changes -> would loop forever
///
/// The cycle detection should catch this and force invalidation.
#[test]
fn test_overlapping_exports_cycle_detected() {
    let mut i = Incremental::new();

    // Set up a mutual dependency cycle where both modules export
    // values that depend on each other.
    i.set("foo", "import bar\nx: int = 1\ny = bar.x");
    i.set("bar", "import foo\nx: int = 2\ny = foo.x");
    i.check(&["foo"], &["foo", "bar"]);

    // Changing `x` in foo should propagate to bar (which uses foo.x),
    // and potentially back to foo (if bar.x changes). The same export `x`
    // may need to be recomputed multiple times, triggering cycle detection.
    i.set("foo", "import bar\nx: str = 'changed'\ny = bar.x");

    // The cycle detection should handle this gracefully.
    // We use unchecked because the exact recomputation pattern depends on
    // cycle detection behavior.
    let res = i.unchecked(&["foo"]);
    // Both modules should be recomputed to reach stable state
    assert!(res.changed.contains(&"foo".to_owned()));
    assert!(res.changed.contains(&"bar".to_owned()));
}

/// Test a more complex non-overlapping case with a chain of re-exports.
///
/// This models a longer dependency chain where multiple intermediate modules
/// re-export from different sources, similar to:
///   torch.distributed.fsdp -> torch.distributed.fsdp._fully_shard -> _fully_shard.py
#[test]
fn test_reexport_chain_non_overlapping() {
    let mut i = Incremental::new();

    // Create a chain: source -> intermediate -> hub -> main
    // with two parallel chains that don't share exports
    i.set("foo", "a: int = 1\nb: int = 2");
    i.set("bar", "from foo import a"); // bar re-exports only `a`
    i.set("baz", "from foo import b"); // baz re-exports only `b`
    i.set("main", "from bar import a\nfrom baz import b");
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    // Change `a` - only bar and main should be affected (fine-grained tracking)
    i.set("foo", "a: str = 'new_a'\nb: int = 2");
    i.check(&["main"], &["foo", "bar", "main"]);

    // Change `b` - only baz and main should be affected
    i.set("foo", "a: str = 'new_a'\nb: str = 'new_b'");
    i.check(&["main"], &["foo", "baz", "main"]);
}

#[test]
fn test_class_field_type_change_propagates() {
    let mut i = Incremental::new();

    // Set up a chain: main -> baz -> bar -> foo
    // where foo.A has a field, bar.B uses A, baz.C uses B, main uses C
    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set("baz", "from bar import B\nclass C:\n    b: B");
    i.set(
        "main",
        "from baz import C\ndef f(c: C) -> int:\n    return c.b.a.x",
    );
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    // Change A's field type from int to str - modules should be recomputed
    i.set("foo", "class A:\n    x: str = 'hello'");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that class base type change propagates through the dependency chain.
///
/// When a class's base class changes, modules that use the derived class
/// should be recomputed.
#[test]
fn test_class_base_type_change_propagates() {
    let mut i = Incremental::new();

    // foo.Base has method m, bar.Derived extends Base, main uses Derived
    i.set("foo", "class Base:\n    def m(self) -> int: return 1");
    i.set("bar", "from foo import Base\nclass Derived(Base): pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.m()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change Base's method return type - main should be recomputed
    i.set("foo", "class Base:\n    def m(self) -> str: return 'hello'");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that class MRO change propagates when a base class is added/removed.
///
/// Changing the inheritance hierarchy should invalidate dependents.
#[test]
fn test_class_mro_change_propagates() {
    let mut i = Incremental::new();

    // foo.Mixin has method mix, bar.C does NOT inherit from Mixin
    // Note: foo is not in the dependency chain yet since bar doesn't import it
    i.set("foo", "class Mixin:\n    def mix(self) -> int: return 1");
    i.set("bar", "class C: pass");
    i.set(
        "main",
        "from bar import C\ndef f(c: C):\n    c.mix() # E: Object of class `C` has no attribute `mix`",
    );
    i.check(&["main"], &["main", "bar"]);

    // Change C to inherit from Mixin - main's error should go away
    // Now foo is part of the dependency chain since bar imports it
    i.set("bar", "from foo import Mixin\nclass C(Mixin): pass");
    i.set("main", "from bar import C\ndef f(c: C):\n    c.mix()");
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that base class field changes propagate when only the derived class is imported.
///
/// When main imports only Derived (not Base), and Base's fields change,
/// main should still see the updated fields through Derived.
#[test]
fn test_base_class_field_change_derived_import_only() {
    let mut i = Incremental::new();

    // foo.Base has field x, bar.Derived extends Base, main uses Derived.x
    i.set("foo", "class Base:\n    x: int = 1");
    i.set("bar", "from foo import Base\nclass Derived(Base): pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change x to str - main should see the updated type (error)
    i.set("foo", "class Base:\n    x: str = 'hello'");
    i.check_ignoring_expectations(&["foo"], &["main", "foo", "bar"]);
}

/// Test that dataclass field changes propagate through the dependency chain.
///
/// Dataclass synthesized fields (like __init__ parameters) should trigger
/// recomputation when they change.
#[test]
fn test_dataclass_field_change_propagates() {
    let mut i = Incremental::new();

    // foo.Data is a dataclass with field x: int
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: int",
    );
    i.set("bar", "from foo import Data\nclass Wrapper:\n    d: Data");
    i.set(
        "main",
        "from bar import Wrapper\ndef f(w: Wrapper) -> int:\n    return w.d.x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change the dataclass field type - main should be recomputed
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: str",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that adding a new field to a class propagates correctly.
///
/// When a class gains a new field, modules using that class should be
/// recomputed so they can access the new field.
#[test]
fn test_class_field_addition_propagates() {
    let mut i = Incremental::new();

    // foo.A has only field x
    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B):\n    b.a.y # E: Object of class `A` has no attribute `y`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add field y to A - main's error should go away
    i.set("foo", "class A:\n    x: int = 1\n    y: int = 2");
    i.set("main", "from bar import B\ndef f(b: B):\n    b.a.y");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that removing a field from a class propagates correctly.
///
/// When a class loses a field, modules using that field should be
/// recomputed and see the error.
#[test]
fn test_class_field_removal_propagates() {
    let mut i = Incremental::new();

    // foo.A has fields x and y
    i.set("foo", "class A:\n    x: int = 1\n    y: int = 2");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B) -> int:\n    return b.a.y",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Remove field y from A - main should see an error
    i.set("foo", "class A:\n    x: int = 1");
    i.set(
        "main",
        "from bar import B\ndef f(b: B) -> int:\n    return b.a.y # E: Object of class `A` has no attribute `y`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that method signature changes propagate through the dependency chain.
///
/// When a class method's signature changes, callers should be recomputed.
#[test]
fn test_class_method_signature_change_propagates() {
    let mut i = Incremental::new();

    // foo.A has method m(self) -> int
    i.set("foo", "class A:\n    def m(self) -> int: return 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B) -> int:\n    return b.a.m()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change method to require an argument - main should see an error
    i.set("foo", "class A:\n    def m(self, n: int) -> int: return n");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test generic class type parameter changes propagate.
///
/// When a generic class's type parameters change, modules using the class
/// with specific type arguments should be recomputed.
#[test]
fn test_generic_class_type_param_change_propagates() {
    let mut i = Incremental::new();

    // foo.Container is a simple class with field items: list[int]
    i.set("foo", "class Container:\n    items: list[int]");
    i.set(
        "bar",
        "from foo import Container\nclass Wrapper:\n    c: Container",
    );
    i.set(
        "main",
        "from bar import Wrapper\ndef f(w: Wrapper) -> int:\n    return w.c.items[0]",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change items to list[str] - main should see a type error
    i.set("foo", "class Container:\n    items: list[str]");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that NamedTuple field changes propagate correctly.
///
/// NamedTuple has synthesized fields similar to dataclass.
#[test]
fn test_namedtuple_field_change_propagates() {
    let mut i = Incremental::new();

    // foo.Point is a NamedTuple with x: int, y: int
    i.set(
        "foo",
        "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: int\n    y: int",
    );
    i.set(
        "bar",
        "from foo import Point\nclass Line:\n    start: Point",
    );
    i.set(
        "main",
        "from bar import Line\ndef length(l: Line) -> int:\n    return l.start.x + l.start.y",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change field types to str - main should be recomputed
    i.set(
        "foo",
        "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: str\n    y: str",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that Protocol changes propagate correctly.
///
/// When a Protocol's method signature changes, implementors and users
/// should be recomputed.
#[test]
fn test_protocol_change_propagates() {
    let mut i = Incremental::new();

    // foo.Proto is a Protocol with method m() -> int
    i.set(
        "foo",
        "from typing import Protocol\nclass Proto(Protocol):\n    def m(self) -> int: ...",
    );
    i.set(
        "bar",
        "from foo import Proto\nclass Impl:\n    def m(self) -> int: return 1",
    );
    i.set(
        "main",
        "from foo import Proto\nfrom bar import Impl\ndef f(p: Proto) -> int:\n    return p.m()\nx: Proto = Impl()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change Protocol method to return str - main should be recomputed
    i.set(
        "foo",
        "from typing import Protocol\nclass Proto(Protocol):\n    def m(self) -> str: ...",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo"]);
}

/// Test four-level dependency chain with class field change.
///
/// This is a more thorough version of test_stale_typed_dict that verifies
/// propagation through a 4-level chain.
#[test]
fn test_four_level_class_field_chain() {
    let mut i = Incremental::new();

    // Chain: main -> baz -> bar -> foo
    // Each level wraps the previous level's class
    i.set("foo", "class A:\n    val: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set("baz", "from bar import B\nclass C:\n    b: B");
    i.set(
        "main",
        "from baz import C\ndef get_val(c: C) -> int:\n    return c.b.a.val",
    );
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    i.set("foo", "class A:\n    val: str = 'hello'");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that star import properly invalidates on any export change.
///
/// Modules using `from X import *` should be invalidated when ANY export
/// in X changes, including class-related exports.
#[test]
fn test_star_import_invalidates_on_class_change() {
    let mut i = Incremental::new();

    // bar uses star import from foo
    i.set("foo", "class A:\n    x: int = 1\nclass B:\n    y: int = 2");
    i.set("bar", "from foo import *\nclass C:\n    a: A");
    i.set(
        "main",
        "from bar import C\ndef f(c: C) -> int:\n    return c.a.x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change B (not A) - bar should still be recomputed because of star import
    i.set(
        "foo",
        "class A:\n    x: int = 1\nclass B:\n    y: str = 'changed'",
    );
    // bar should be recomputed even though it only uses A, because star import
    // means it depends on all exports
    // Main should not be recomputed because it doesn't use B
    i.check_ignoring_expectations(&["main"], &["foo", "bar"]);
}

/// Test that enum member changes propagate correctly.
#[test]
fn test_enum_member_change_propagates() {
    let mut i = Incremental::new();

    i.set(
        "foo",
        "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2",
    );
    i.set(
        "bar",
        "from foo import Color\nclass Palette:\n    primary: Color",
    );
    i.set(
        "main",
        "from bar import Palette\nfrom foo import Color\ndef f(p: Palette):\n    if p.primary == Color.RED: pass",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add a new enum member - main should be recomputed
    i.set(
        "foo",
        "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test fine-grained tracking when importing a class: changing an unrelated
/// function should NOT trigger recomputation.
#[test]
fn test_class_import_unrelated_function_change_no_recompute() {
    let mut i = Incremental::new();

    // foo has class A and unrelated function f; main imports only A
    i.set("foo", "class A:\n    x: int = 1\ndef f() -> int: return 42");
    i.set(
        "main",
        "from foo import A\ndef use_a(a: A) -> int: return a.x",
    );
    i.check(&["main"], &["main", "foo"]);

    // Change only f - main should NOT be recomputed since it doesn't import f
    i.set(
        "foo",
        "class A:\n    x: int = 1\ndef f() -> str: return 'changed'",
    );
    i.check(&["main"], &["foo"]);
}

#[test]
fn test_mixed_import_failed_export_invalidated() {
    let mut i = Incremental::new();

    // Initial state: foo exports x but not y
    i.set("foo", "x = 1");
    i.set(
        "main",
        "from foo import x, y  # E: Could not import `y` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error before the fix
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error before foo exports y"
    );

    i.set("foo", "x = 1\ny = 2");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);
    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected error after foo exports y"
    );
}

/// Test that abstract class check changes propagate correctly (KeyAbstractClassCheck).
///
/// When a class becomes concrete (by implementing abstract methods), modules
/// that try to instantiate it should be recomputed and errors should clear.
#[test]
fn test_abstract_class_check_change_propagates() {
    let mut i = Incremental::new();

    // foo.Base is abstract (has abstract method), bar.Impl does NOT implement it initially
    i.set(
        "foo",
        "from abc import ABC, abstractmethod\nclass Base(ABC):\n    @abstractmethod\n    def m(self) -> int: ...",
    );
    i.set("bar", "from foo import Base\nclass Impl(Base): pass");
    i.set(
        "main",
        "from bar import Impl\nx = Impl() # E: Cannot instantiate `Impl` because the following members are abstract: `m`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Implement the abstract method - main should be recomputed and error should go away
    i.set(
        "bar",
        "from foo import Base\nclass Impl(Base):\n    def m(self) -> int: return 1",
    );
    i.set("main", "from bar import Impl\nx = Impl()");
    i.check(&["main"], &["bar", "main"]);
}

/// Test that class metadata changes propagate correctly (KeyClassMetadata).
///
/// When a class becomes a Protocol or stops being one, modules using it
/// should be recomputed.
#[test]
fn test_class_metadata_protocol_change_propagates() {
    let mut i = Incremental::new();

    // foo.P is NOT a Protocol initially - main uses it as a regular class
    i.set("foo", "class P:\n    def m(self) -> int: return 1");
    i.set("bar", "class Q:\n    def m(self) -> int: return 2");
    i.set(
        "main",
        "from foo import P\nfrom bar import Q\nx: P = Q() # E: `Q` is not assignable to `P`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Make P a Protocol - main should be recomputed and Q should be assignable
    i.set(
        "foo",
        "from typing import Protocol\nclass P(Protocol):\n    def m(self) -> int: ...",
    );
    i.set("main", "from foo import P\nfrom bar import Q\nx: P = Q()");
    i.check(&["main"], &["foo", "main"]);
}

/// Test that type parameter variance changes propagate correctly (KeyVariance).
///
/// When a generic class's variance changes (e.g., from invariant to covariant),
/// modules using that class should be recomputed.
#[test]
fn test_variance_change_propagates() {
    let mut i = Incremental::new();

    // foo.Container is invariant (default) - main has an error trying to assign Container[Derived] to Container[Base]
    i.set(
        "foo",
        "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass Container(Generic[T]):\n    def get(self) -> T: ...",
    );
    i.set("bar", "class Base: pass\nclass Derived(Base): pass");
    i.set(
        "main",
        "from foo import Container\nfrom bar import Base, Derived\ndef f(c: Container[Derived]) -> Container[Base]:\n    return c # E: Returned type `Container[Derived]` is not assignable to declared return type `Container[Base]`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Make Container covariant - main should be recomputed and error should go away
    i.set(
        "foo",
        "from typing import Generic, TypeVar\nT = TypeVar('T', covariant=True)\nclass Container(Generic[T]):\n    def get(self) -> T: ...",
    );
    i.set(
        "main",
        "from foo import Container\nfrom bar import Base, Derived\ndef f(c: Container[Derived]) -> Container[Base]:\n    return c",
    );
    i.check(&["main"], &["foo", "main"]);
}

/// Test that adding a base class to a derived class propagates (KeyClassBaseType).
///
/// When a class gains a base class, modules that use the derived class
/// should be recomputed and see the inherited methods.
#[test]
fn test_adding_base_class_propagates() {
    let mut i = Incremental::new();

    // foo.Base has method m, bar.Derived does NOT extend Base initially
    i.set("foo", "class Base:\n    def m(self) -> int: return 1");
    i.set("bar", "class Derived: pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.m() # E: Object of class `Derived` has no attribute `m`",
    );
    i.check(&["main"], &["main", "bar"]);

    // Change Derived to extend Base - main should be recomputed and error should go away
    i.set("bar", "from foo import Base\nclass Derived(Base): pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.m()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that adding a dataclass field propagates (KeyClassSynthesizedFields).
///
/// When a dataclass gains a new field, its __init__ parameters change and
/// modules using the class should be recomputed.
#[test]
fn test_adding_dataclass_field_propagates() {
    let mut i = Incremental::new();

    // foo.Data is a dataclass with field x: int - main tries to use missing field y
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: int",
    );
    i.set("bar", "from foo import Data\nclass Wrapper:\n    d: Data");
    i.set(
        "main",
        "from foo import Data\na = Data(x=1, y=2) # E: Unexpected keyword argument `y`",
    );
    i.check(&["main"], &["main", "foo"]);

    // Add field y to the dataclass - main should be recomputed and error should go away
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: int\n    y: int",
    );
    i.set("main", "from foo import Data\na = Data(x=1, y=2)");
    i.check(&["main"], &["foo", "main"]);
}

/// Test that adding a field to a class propagates (KeyClassField).
///
/// When a class gains a new field, modules accessing that field should be
/// recomputed and no longer see an error.
#[test]
fn test_adding_class_field_propagates() {
    let mut i = Incremental::new();

    // foo.A has only field x - main tries to access missing field y
    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B):\n    b.a.y # E: Object of class `A` has no attribute `y`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add field y to A - main should be recomputed and error should go away
    i.set("foo", "class A:\n    x: int = 1\n    y: int = 2");
    i.set("main", "from bar import B\ndef f(b: B):\n    b.a.y");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that making a class generic propagates (KeyTParams).
///
/// When a class gains type parameters (becomes generic), modules using
/// the class with type arguments should be recomputed.
#[test]
fn test_making_class_generic_propagates() {
    let mut i = Incremental::new();

    // foo.Container is not generic initially - main tries to use it with type args
    i.set("foo", "class Container:\n    items: list[int]");
    i.set(
        "bar",
        "from foo import Container\nclass Wrapper:\n    c: Container",
    );
    i.set(
        "main",
        "from foo import Container\nx: Container[int] # E: Expected 0 type arguments for `Container`, got 1",
    );
    i.check(&["main"], &["main", "foo"]);

    // Make Container generic - main should be recomputed and error should go away
    i.set(
        "foo",
        "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass Container(Generic[T]):\n    items: list[int]",
    );
    i.set("main", "from foo import Container\nx: Container[int]");
    i.check(&["main"], &["foo", "main"]);
}

#[test]
fn test_adding_mro_attr_propagates() {
    let mut i = Incremental::new();
    i.set("foo", "class Foo: pass");
    i.set("bar", "from foo import Foo\nclass Bar(Foo): pass");
    i.set("main", "from bar import Bar\nBar().foo");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
    i.set("foo", "class Foo: foo: int = 1");
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that adding a name listed in __all__ clears the error.
///
/// When __all__ lists a name that doesn't exist, there should be an error.
/// After adding the missing name, the error should disappear.
#[test]
fn test_dunder_all_missing_name_error_clears() {
    let mut i = Incremental::new();

    // foo has __all__ = ["test"] but doesn't define test - should error
    i.set(
        "foo",
        "__all__ = [\"test\"] # E: Name `test` is listed in `__all__` but is not defined in the module",
    );
    i.check(&["foo"], &["foo"]);

    let foo_handle = i.handle("foo");

    // Verify there's an error with the expected message
    let errors = i
        .state
        .transaction()
        .get_errors([&foo_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error when __all__ lists undefined name"
    );
    assert!(
        errors.shown.iter().any(|e| e
            .msg()
            .contains("Name `test` is listed in `__all__` but is not defined in the module")),
        "Expected error message about missing __all__ name, but got: {:?}",
        errors.shown.iter().map(|e| e.msg()).collect::<Vec<_>>()
    );

    // Add the missing name - error should disappear
    i.set("foo", "__all__ = [\"test\"]\ntest = 1");
    i.check_ignoring_expectations(&["foo"], &["foo"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&foo_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected no errors after defining the missing name, but got: {:?}",
        errors_after.shown
    );
}

/// Test that modifying __all__ to include a name clears errors in importing files.
///
/// When foo has `x` and `y` but `__all__ = ["x"]`, a star import only brings in `x`.
/// A file using `y` after `from foo import *` will error until `__all__` is updated.
/// This tests export information affecting errors in another file.
///
/// Test that errors are properly cleared when __all__ changes.
/// When a name is added to __all__, star importers should see the change.
#[test]
fn test_dunder_all_star_import_error_clears() {
    let mut i = Incremental::new();

    // foo has x and y, but __all__ only exports x
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\"]");
    // main does star import and tries to use y - should error
    i.set(
        "main",
        "from foo import *\nz = y # E: Could not find name `y`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error with the expected message
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error when using name not in __all__"
    );
    assert!(
        errors
            .shown
            .iter()
            .any(|e| e.msg().contains("Could not find name `y`")),
        "Expected error about missing name y, but got: {:?}",
        errors.shown.iter().map(|e| e.msg()).collect::<Vec<_>>()
    );

    // Update __all__ to include y - error should disappear
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\", \"y\"]");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected no errors after adding y to __all__, but got: {:?}",
        errors_after.shown
    );
}

/// Test that adding a new definition invalidates importers of that name.
/// When a new name is added to a module, modules that try to import it
/// (and previously got an error) should be recomputed.
#[test]
fn test_name_existence_change_invalidates_importer() {
    let mut i = Incremental::new();

    // foo only has x
    i.set("foo", "x = 1");
    // main tries to import y which doesn't exist - should error
    i.set(
        "main",
        "from foo import y # E: Could not import `y` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error when importing non-existent name"
    );

    // Add y to foo - error should disappear
    i.set("foo", "x = 1\ny = 2");
    // main should be recomputed because y now exists
    i.check_ignoring_expectations(&["foo"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected no errors after adding y to foo, but got: {:?}",
        errors_after.shown
    );
}

/// Test that adding a name NOT in __all__ still invalidates importers of that name.
/// __all__ controls what `from M import *` gets, but explicit imports like
/// `from M import name` can access any name in the module. When that name is
/// added or removed, importers should be invalidated.
#[test]
fn test_name_not_in_dunder_all_invalidates_importer() {
    let mut i = Incremental::new();

    // foo has __all__ = ["x"] but only x exists
    i.set("foo", "__all__ = ['x']\nx = 1");
    // main tries to import y which doesn't exist - should error
    i.set(
        "main",
        "from foo import y # E: Could not import `y` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error when importing non-existent name"
    );

    // Add y to foo, but NOT to __all__. The import should still work.
    i.set("foo", "__all__ = ['x']\nx = 1\ny = 2");
    // main should be recomputed because y now exists
    i.check_ignoring_expectations(&["foo"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected no errors after adding y to foo (even though y is not in __all__), but got: {:?}",
        errors_after.shown
    );
}

/// Test that wildcard re-exports properly propagate invalidation.
///
/// When module A re-exports module B's __all__ (e.g., via `from B import *`),
/// and B's __all__ changes, star importers of A should be invalidated.
#[test]
fn test_wildcard_reexport_invalidation() {
    let mut i = Incremental::new();

    // foo exports x via __all__
    i.set("foo", "x = 1\n__all__ = [\"x\"]");
    // bar re-exports foo's __all__ via from foo import *
    i.set("bar", "from foo import *");
    // main uses star import from bar
    i.set("main", "from bar import *\nz = x");
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add y to foo's __all__ - main should be invalidated because
    // bar's effective wildcard set changed (even though bar's __all__ entries didn't)
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\", \"y\"]");
    // bar is invalidated because it depends on foo with wildcard = true
    // main is invalidated because bar's effective wildcard set changed
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);
}

/// Test that type changes propagate through transitive star import chains.
///
/// When module A re-exports module B's exports (via `from B import *`),
/// and a type in B changes, consumers of A that use that type should be invalidated.
#[test]
fn test_wildcard_reexport_type_change_invalidation() {
    let mut i = Incremental::new();

    // foo exports x via __all__
    i.set("foo", "x: int = 1\n__all__ = [\"x\"]");
    // bar re-exports foo's __all__ via from foo import * (pure pass-through)
    i.set("bar", "from foo import *");
    // main uses star import from bar and uses x
    i.set("main", "from bar import *\nz: int = x");
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change x's type from int to str - main should be invalidated because
    // it uses x and the type changed, even though bar is just a pass-through.
    i.set("foo", "x: str = 'hello'\n__all__ = [\"x\"]");
    // foo is invalidated because its source changed
    // bar is invalidated because it depends on foo with wildcard = true
    // main is invalidated because x's type changed and main uses x
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);
}

/// Test that star imports invalidate on type changes to unused variable names.
#[test]
fn test_star_import_unused_variable_type_change() {
    let mut i = Incremental::new();

    // foo exports x and y
    i.set("foo", "x: int = 1\ny: int = 2\n__all__ = ['x', 'y']");
    // bar does star import but only uses x
    i.set("bar", "from foo import *\nz = x + 1");
    i.check(&["bar"], &["bar", "foo"]);

    // Change y's type - bar IS invalidated even though it doesn't use y
    i.set(
        "foo",
        "x: int = 1\ny: str = 'changed'\n__all__ = ['x', 'y']",
    );
    i.check_ignoring_expectations(&["bar"], &["foo", "bar"]);

    // If we change x's type, bar SHOULD be invalidated
    i.set(
        "foo",
        "x: str = 'now a string'\ny: str = 'changed'\n__all__ = ['x', 'y']",
    );
    i.check_ignoring_expectations(&["bar"], &["foo", "bar"]);
}

/// Test that star import errors clear when a missing name is added to the module.
///
/// When `__all__` lists a name that doesn't exist, star importers get an error.
/// Adding the missing definition should clear the error.
#[test]
fn test_dunder_all_star_import_missing_definition_error_clears() {
    let mut i = Incremental::new();

    // foo has __all__ = ["x", "y"] but only defines x - y is missing
    i.set(
        "foo",
        "x = 1\n__all__ = [\"x\", \"y\"] # E: Name `y` is listed in `__all__` but is not defined",
    );
    // main does star import and tries to use y - should error
    i.set(
        "main",
        "from foo import * # E: Could not import `y` from `foo`\nz = y",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error when using name listed in __all__ but not defined"
    );

    // Add the missing definition - error should disappear
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\", \"y\"]");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected no errors after adding y definition, but got: {:?}",
        errors_after.shown
    );
}

/// Test that transitive star import errors clear when a missing name is added.
///
/// Similar to test_dunder_all_star_import_missing_definition_error_clears but:
/// - No __all__ is used (relies on default public exports)
/// - Uses transitive star imports (foo -> bar -> main)
#[test]
fn test_transitive_star_import_missing_name_error_clears() {
    let mut i = Incremental::new();

    // foo initially empty
    i.set("foo", "");
    // bar re-exports from foo via star import
    i.set("bar", "from foo import *");
    // main does star import from bar and tries to use x - should fail
    i.set("main", "from bar import *\nz = x");
    i.check_ignoring_expectations(&["main", "foo", "bar"], &["main", "foo", "bar"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.shown.is_empty(),
        "Expected error when using undefined name via transitive star import"
    );

    // Add x to foo - error should disappear in main
    i.set("foo", "x = 1");
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.shown.is_empty(),
        "Expected no errors after adding x to foo, but got: {:?}",
        errors_after.shown
    );
}

#[test]
fn test_class_index_change_only_invalidates() {
    let mut i = Incremental::new();

    i.set("foo", "class A:\n    attr: int = 1");
    i.set("bar", "from foo import A\nx = A.attr");
    i.check(&["bar"], &["foo", "bar"]);

    // Now add class B before A, shifting A to index 1
    i.set(
        "foo",
        "class B:\n    other: str = 'hello'\nclass A:\n    attr: int = 1",
    );
    i.check(&["foo"], &["foo", "bar"]);
}

#[test]
fn test_class_index_and_key_change_invalidates_dependents() {
    let mut i = Incremental::new();

    i.set("foo", "class A:\n    attr: int = 1");
    i.set("bar", "from foo import A\nx = A.attr");
    i.check(&["bar"], &["foo", "bar"]);

    i.set(
        "foo",
        "class B:\n    other: str = 'hello'\nclass A:\n    attr: bool = True",
    );
    i.check(&["foo"], &["foo", "bar"]);
}

/// Test that when a function's deprecation metadata changes, dependents are invalidated.
///
/// This tests the `get_deprecated` metadata dependency. When a function gains or loses
/// the `@deprecated` decorator, modules that import it should be recomputed so they
/// can show or hide deprecation warnings.
#[test]
fn test_deprecation_metadata_change_invalidates() {
    let mut i = Incremental::new();

    // foo.f is NOT deprecated initially
    i.set("foo", "def f() -> int: return 1");
    // # E: comments are present for the second check when deprecation warnings appear
    i.set("main", "from foo import f # E:\nx = f() # E:");
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    // Add @deprecated decorator - main should be recomputed to show deprecation warning
    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use g instead')\ndef f() -> int: return 1",
    );
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// Test that when a class's deprecation metadata changes, dependents are invalidated.
///
/// Similar to test_deprecation_metadata_change_invalidates but for classes.
#[test]
fn test_class_deprecation_metadata_change_invalidates() {
    let mut i = Incremental::new();

    // foo.C is NOT deprecated initially
    i.set("foo", "class C:\n    x: int = 1");
    // # E: comment is present for the second check when deprecation warning appears
    i.set("main", "from foo import C # E:\nc = C()");
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    // Add @deprecated decorator to class - main should be recomputed
    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use D instead')\nclass C:\n    x: int = 1",
    );
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// Test that when an export changes from direct definition to re-export, dependents are invalidated.
///
/// This tests the `is_reexport` metadata dependency. When an export changes from being
/// defined in this module to being re-exported from another module, dependents should
/// be recomputed.
#[test]
fn test_reexport_status_change_invalidates() {
    let mut i = Incremental::new();

    // foo.x is defined directly in foo
    i.set("foo", "x: int = 1");
    i.set("bar", "y: int = 2");
    i.set("main", "from foo import x\nz = x + 1");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change foo.x to be a re-export from bar
    // Since the type doesn't change (still int), main should still be recomputed
    // so that go-to-definition points to bar.y instead of the old foo.x
    i.set("foo", "from bar import y as x");
    let res = i.unchecked(&["foo", "main"]);
    assert!(res.changed.contains(&"foo".to_owned()));
    assert!(res.changed.contains(&"bar".to_owned()));
    assert!(res.changed.contains(&"main".to_owned()));
}

/// Test that when a re-export's source changes, dependents are invalidated.
///
/// When foo re-exports from bar, changing which module foo re-exports from
/// should invalidate dependents.
#[test]
fn test_reexport_source_change_invalidates() {
    let mut i = Incremental::new();

    // foo re-exports x from bar
    i.set("bar", "x: int = 1");
    i.set("baz", "x: str = 'hello'");
    i.set("foo", "from bar import x");
    i.set("main", "from foo import x\ny = x + 1 # E:");
    // Initial check - no error yet (x is int), but # E: is present for later
    i.check_ignoring_expectations(&["foo", "main"], &["bar", "foo", "main"]);

    // Change foo to re-export from baz instead - main should be recomputed (type changes)
    // Now x is str, so x + 1 is a type error
    i.set("foo", "from baz import x");
    i.check(&["foo", "main"], &["baz", "foo", "main"]);
}

/// Test that when a special export changes, dependents are invalidated.
///
/// This tests the `is_special_export` metadata dependency. When a name gains or loses
/// special export status (like TypeVar, cast, etc.), modules using it should be recomputed.
/// Note: This is a somewhat contrived test since special exports are typically in stdlib,
/// but it validates the dependency tracking.
#[test]
fn test_special_export_usage_change_invalidates() {
    let mut i = Incremental::new();

    // main uses cast from typing
    i.set(
        "main",
        "from typing import cast\nx: int = cast(int, 'hello')",
    );
    i.check(&["main"], &["main"]);

    // Change to use a different cast that's not special - main should be recomputed
    i.set("foo", "def cast(ty, val): return val");
    i.set("main", "from foo import cast\nx: int = cast(int, 'hello')");
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// Test that when a docstring changes, hover info is updated correctly.
///
/// This tests the `docstring_range` metadata dependency. When a function's docstring
/// changes, the hover information should reflect the new docstring.
/// Note: This test primarily validates that docstring changes trigger recomputation
/// for accurate hover info, even if the type signature doesn't change.
#[test]
fn test_docstring_change_invalidates() {
    let mut i = Incremental::new();

    // foo.f has a docstring
    i.set(
        "foo",
        "def f() -> int:\n    \"\"\"Original docstring.\"\"\"\n    return 1",
    );
    i.set("main", "from foo import f\nx = f()");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change the docstring - foo should be recomputed (main is not recomputed since
    // the type signature didn't change)
    i.set(
        "foo",
        "def f() -> int:\n    \"\"\"Updated docstring with new info.\"\"\"\n    return 1",
    );
    i.check(&["foo", "main"], &["foo"]);
}

/// Test that when a class docstring changes, it is properly recomputed.
#[test]
fn test_class_docstring_change_invalidates() {
    let mut i = Incremental::new();

    // foo.C has a docstring
    i.set(
        "foo",
        "class C:\n    \"\"\"Original class docstring.\"\"\"\n    x: int = 1",
    );
    i.set("main", "from foo import C\nc = C()");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change the class docstring - foo should be recomputed (main is not recomputed
    // since the type signature didn't change)
    i.set(
        "foo",
        "class C:\n    \"\"\"Updated class docstring.\"\"\"\n    x: int = 1",
    );
    i.check(&["foo", "main"], &["foo"]);
}

/// Test that when __all__ contains Module entries (like `__all__.extend(other.__all__)`),
/// changing those entries triggers wildcard fallback and invalidates star importers.
///
/// This tests the `wildcard_fallback` behavior in `changed_name_existence`.
/// When __all__ contains Module entries that change, we can't compute the exact
/// set of names that changed without a lookup, so we emit a coarse Wildcard signal.
#[test]
fn test_dunder_all_module_entry_change_invalidates() {
    let mut i = Incremental::new();

    // bar exports x
    i.set("bar", "x = 1\n__all__ = ['x']");
    // baz exports y
    i.set("baz", "y = 2\n__all__ = ['y']");
    // foo imports bar and re-exports bar's __all__
    i.set(
        "foo",
        "import bar\nfrom bar import *\n__all__ = []\n__all__.extend(bar.__all__)",
    );
    // main does star import from foo and uses x
    i.set("main", "from foo import *\nz = x");
    i.check(
        &["main", "foo", "bar", "baz"],
        &["main", "foo", "bar", "baz"],
    );

    // Now change foo to re-export baz's __all__ instead of bar's
    // This changes a Module entry in __all__, triggering wildcard fallback
    i.set(
        "foo",
        "import baz\nfrom baz import *\n__all__ = []\n__all__.extend(baz.__all__)",
    );
    // main should be recomputed because the wildcard set changed
    i.check_ignoring_expectations(&["foo"], &["foo", "main"]);
}
