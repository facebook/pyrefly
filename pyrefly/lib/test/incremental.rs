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
    i.check(&["foo"], &["foo", "foo", "bar"]);
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
    i.check(&["foo"], &["bar", "bar", "foo", "foo", "foo"]);

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
