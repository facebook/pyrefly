/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the `State` object.

use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::prelude::SliceExt;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::print_errors;
use crate::module::finder::find_import;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;

#[test]
fn test_multiple_config() {
    let linux = SysInfo::new(PythonVersion::default(), PythonPlatform::linux());
    let windows = SysInfo::new(PythonVersion::default(), PythonPlatform::windows());

    const LIB: &str = r#"
import sys
if sys.platform == "linux":
    value = 42
else:
    value = "hello"
"#;
    let mut test_env = TestEnv::new();
    test_env.add("lib", LIB);
    test_env.add("windows", "import lib; x: str = lib.value");
    test_env.add("linux", "import lib; x: int = lib.value");
    test_env.add(
        "main",
        "import lib; x: str = lib.value  # E: `Literal[42]` is not assignable to `str`",
    );
    let config_file = test_env.config();
    let state = State::new(test_env.config_finder());

    let f = |name: &str, sys_info: &SysInfo| {
        let name = ModuleName::from_str(name);
        let path = find_import(&config_file, name, None, true).unwrap();
        (
            Handle::new(name, path, sys_info.dupe()),
            Require::Everything,
        )
    };

    let handles = [
        f("linux", &linux),
        f("windows", &windows),
        f("main", &linux),
    ];
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(test_env.get_memory());
    transaction.run(&handles);
    transaction
        .get_errors(handles.iter().map(|(handle, _)| handle))
        .check_against_expectations()
        .unwrap();
}

#[test]
fn test_multiple_path() {
    const LIB_PYI: &str = "x: int";
    const LIB_PY: &str = "x: str = 1  # E: `Literal[1]` is not assignable to `str`";
    const MAIN_PYI: &str =
        "import lib; y: list[int] = lib.x  # E: `int` is not assignable to `list[int]`";
    const MAIN_PY: &str =
        "import lib; y: list[str] = lib.x  # E: `int` is not assignable to `list[str]`";

    const FILES: &[(&str, &str, &str)] = &[
        ("lib", "lib.pyi", LIB_PYI),
        ("lib", "lib.py", LIB_PY),
        ("main", "main.pyi", MAIN_PYI),
        ("main", "main.py", MAIN_PY),
    ];

    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    for (name, path, _) in FILES.iter().rev() {
        config.custom_module_paths.insert(
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(path)),
        );
    }
    config.configure();
    let config = ArcId::new(config);

    let sys_info = SysInfo::default();

    let state = State::new(ConfigFinder::new_constant(config));
    let handles = FILES.map(|(name, path, _)| {
        Handle::new(
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(path)),
            sys_info.dupe(),
        )
    });
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(
        FILES.map(|(_, path, contents)| {
            (PathBuf::from(path), Some(Arc::new((*contents).to_owned())))
        }),
    );
    transaction.run(&handles.map(|x| (x.dupe(), Require::Everything)));
    let loads = transaction.get_errors(handles.iter());
    print_errors(&loads.collect_errors().shown);
    loads.check_against_expectations().unwrap();
    assert_eq!(loads.collect_errors().shown.len(), 3);
}

#[test]
fn test_change_require() {
    let env = TestEnv::one("foo", "x: str = 1\ny: int = 'x'");
    let state = State::new(env.config_finder());
    let handle = Handle::new(
        ModuleName::from_str("foo"),
        ModulePath::memory(PathBuf::from("foo.py")),
        env.sys_info(),
    );

    let mut t = state.new_committable_transaction(Require::Exports, None);
    t.as_mut().set_memory(env.get_memory());
    t.as_mut().run(&[(handle.dupe(), Require::Exports)]);
    state.commit_transaction(t);

    assert_eq!(
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .shown
            .len(),
        0
    );
    assert!(state.transaction().get_bindings(&handle).is_none());
    state.run(&[(handle.dupe(), Require::Errors)], Require::Exports, None);
    assert_eq!(
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .shown
            .len(),
        2
    );
    assert!(state.transaction().get_bindings(&handle).is_none());
    state.run(
        &[(handle.dupe(), Require::Everything)],
        Require::Exports,
        None,
    );
    assert_eq!(
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .shown
            .len(),
        2
    );
    assert!(state.transaction().get_bindings(&handle).is_some());
}

#[test]
fn test_crash_on_search() {
    const REQUIRE: Require = Require::Everything; // Doesn't matter for the test

    let mut t = TestEnv::new();
    t.add("foo", "x = 1");
    let (state, _) = t.to_state();

    // Now we dirty the module `foo`
    let mut t = state.new_committable_transaction(REQUIRE, None);
    t.as_mut().set_memory(vec![(
        PathBuf::from("foo.py"),
        Some(Arc::new("x = 3".to_owned())),
    )]);
    t.as_mut().run(&[]); // This run breaks reproduction (but is now required)
    state.commit_transaction(t);

    // Now we need to increment the step counter.
    let mut t = state.new_committable_transaction(REQUIRE, None);
    t.as_mut().run(&[]);
    state.commit_transaction(t);

    // Now we run two searches, this used to crash
    let t = state.new_transaction(REQUIRE, None);
    eprintln!("First search");
    t.search_exports_exact("x");
    eprintln!("Second search");
    t.search_exports_exact("x");
}
