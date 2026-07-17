/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the `State` object.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::LiveSourceDatabase;
use pyrefly_build::source_db::ModuleEnumerator;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;
use tempfile::TempDir;

use crate::commands::config_finder::default_config_finder;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::finder::ConfigFinder;
use crate::error::error::print_errors;
use crate::lsp::non_wasm::server::resolve_export_location;
use crate::module::finder::DirEntryCache;
use crate::module::finder::find_import;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::require::RequireLevels;
use crate::state::state::State;
use crate::test::util::TestEnv;

#[derive(Debug)]
struct MutableShapeExtensionsSourceDb {
    sys_info: SysInfo,
    contains_shape_extensions: Arc<Mutex<bool>>,
}

impl MutableShapeExtensionsSourceDb {
    fn new(sys_info: SysInfo, contains_shape_extensions: Arc<Mutex<bool>>) -> Self {
        Self {
            sys_info,
            contains_shape_extensions,
        }
    }

    fn module_path(module: ModuleName) -> Option<ModulePath> {
        Some(ModulePath::memory(PathBuf::from(match module.as_str() {
            "main" => "main.py",
            "torch" => "torch.pyi",
            "jaxtyping" => "jaxtyping.pyi",
            "shape_extensions" => "shape_extensions.pyi",
            _ => return None,
        })))
    }
}

impl SourceDatabase for MutableShapeExtensionsSourceDb {
    fn may_contain_module(&self, module: ModuleName) -> bool {
        match module.as_str() {
            "shape_extensions" => *self.contains_shape_extensions.lock(),
            "main" | "torch" | "jaxtyping" => true,
            _ => false,
        }
    }

    fn lookup(
        &self,
        module: ModuleName,
        _: Option<&Path>,
        _: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        if module.as_str() == "shape_extensions" && !*self.contains_shape_extensions.lock() {
            None
        } else {
            Self::module_path(module)
        }
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        let module = match module_path.as_path().to_str()? {
            "main.py" => ModuleName::from_str("main"),
            "torch.pyi" => ModuleName::from_str("torch"),
            "jaxtyping.pyi" => ModuleName::from_str("jaxtyping"),
            "shape_extensions.pyi" => ModuleName::from_str("shape_extensions"),
            _ => return None,
        };
        Some(Handle::new(
            module,
            module_path.dupe(),
            self.sys_info.dupe(),
        ))
    }

    fn as_live_source_database(&self) -> Option<&dyn LiveSourceDatabase> {
        None
    }
}

impl ModuleEnumerator for MutableShapeExtensionsSourceDb {
    fn modules_to_check(&self) -> Vec<Handle> {
        vec![Handle::new(
            ModuleName::from_str("main"),
            ModulePath::memory(PathBuf::from("main.py")),
            self.sys_info.dupe(),
        )]
    }
}

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
        "import lib; x: str = lib.value  # E: `int` is not assignable to `str`",
    );
    let config_file = test_env.config();
    let state = State::new(test_env.config_finder(), TEST_THREAD_COUNT);

    let f = |name: &str, sys_info: &SysInfo| {
        let name = ModuleName::from_str(name);
        let path = find_import(&config_file, name, None, None, &DirEntryCache::new(), None)
            .finding()
            .unwrap();
        Handle::new(name, path, sys_info.dupe())
    };

    let handles = [
        f("linux", &linux),
        f("windows", &windows),
        f("main", &linux),
    ];
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(test_env.get_memory());
    transaction.run(&handles, Require::Everything, None);
    transaction
        .get_errors(&handles)
        .check_against_expectations()
        .unwrap();
}

/// Regression for the TSP type converter's export-location resolution.
///
/// `convert_type_in_transaction` resolves an exported symbol's definition by
/// demanding the target module's exports, which demands its `Stdlib` via
/// `get_stdlib`. The target handle must inherit the *source* file's `SysInfo`
/// (built through `import_handle`), because computing the queried type only
/// populated `compute_stdlib` for the source's `SysInfo`. When the transaction
/// holds more than one `SysInfo`, `get_stdlib` no longer short-circuits to its
/// single entry and looks the `SysInfo` up by key — so a target handle carrying
/// any other `SysInfo` would hit a missing `Stdlib` and panic.
///
/// This test reproduces the multi-`SysInfo` transaction and verifies the warm
/// path: a target resolved via `import_handle` inherits the source `SysInfo`
/// and its export location resolves without panicking.
#[test]
fn test_lookup_export_location_warm_with_multiple_sysinfos() {
    let linux = SysInfo::new(PythonVersion::default(), PythonPlatform::linux());
    let windows = SysInfo::new(PythonVersion::default(), PythonPlatform::windows());

    let mut test_env = TestEnv::new();
    test_env.add("lib", "CONST = 42\n");
    // `main` does not import `lib`, so `lib` is never checked during the run and
    // its exports must be demanded fresh — exercising the `get_stdlib` call.
    test_env.add("main", "x = 1\n");
    let config_file = test_env.config();
    let state = State::new(test_env.config_finder(), TEST_THREAD_COUNT);

    let f = |name: &str, sys_info: &SysInfo| {
        let name = ModuleName::from_str(name);
        let path = find_import(&config_file, name, None, None, &DirEntryCache::new(), None)
            .finding()
            .unwrap();
        Handle::new(name, path, sys_info.dupe())
    };

    // Check `main` under BOTH sys_infos so the transaction's stdlib map has two
    // entries and `get_stdlib`'s single-entry short-circuit no longer applies.
    let handles = [f("main", &linux), f("main", &windows)];
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(test_env.get_memory());
    transaction.run(&handles, Require::Everything, None);

    // Resolve `lib.CONST` the way the converter does: build the target handle via
    // `import_handle` from the source file. It must inherit the source `SysInfo`.
    let source = f("main", &linux);
    let target = transaction
        .import_handle(&source, ModuleName::from_str("lib"), None)
        .finding()
        .unwrap();
    assert_eq!(
        target.sys_info(),
        &linux,
        "import_handle must inherit the source file's SysInfo"
    );
    let location = transaction.lookup_export_location(&target, &Name::new_static("CONST"));
    assert!(
        location.is_some(),
        "expected to resolve the location of lib.CONST without panicking"
    );
}

// Regression for the TSP `get_stdlib` panic (D107960810), exercising the actual
// converter seam `resolve_export_location` rather than `lookup_export_location`
// directly. Before the fix this re-derived the target handle from the module
// path via the config (which defaults to `linux`); when the source file is
// checked under other platforms, that `linux` Stdlib is never computed, so the
// export lookup panicked in `get_stdlib` once the transaction held more than one
// `SysInfo`. The fix reaches the target via `import_handle`, inheriting the
// source's (warm) `SysInfo`, so resolution succeeds.
#[test]
fn test_resolve_export_location_inherits_source_sysinfo() {
    // Two run platforms, both different from the config default (`linux`), so the
    // config-derived handle's Stdlib is never computed and the single-entry
    // shortcut in `get_stdlib` does not apply.
    let windows = SysInfo::new(PythonVersion::default(), PythonPlatform::windows());
    let mac = SysInfo::new(PythonVersion::default(), PythonPlatform::mac());

    let mut test_env = TestEnv::new();
    test_env.add("lib", "CONST = 42\n");
    // `main` does not import `lib`, so `lib` is demanded fresh during resolution.
    test_env.add("main", "x = 1\n");
    let config_file = test_env.config();
    let state = State::new(test_env.config_finder(), TEST_THREAD_COUNT);

    let f = |name: &str, sys_info: &SysInfo| {
        let name = ModuleName::from_str(name);
        let path = find_import(&config_file, name, None, None, &DirEntryCache::new(), None)
            .finding()
            .unwrap();
        Handle::new(name, path, sys_info.dupe())
    };

    let handles = [f("main", &windows), f("main", &mac)];
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(test_env.get_memory());
    transaction.run(&handles, Require::Everything, None);

    // Resolve `lib.CONST` from `main` (checked under `windows`). The fix inherits
    // `windows`; the pre-fix code re-derives `linux` and panics in `get_stdlib`.
    let source = f("main", &windows);
    let location = resolve_export_location(
        &transaction,
        &source,
        ModuleName::from_str("lib"),
        &Name::new_static("CONST"),
    );
    assert!(
        location.is_some(),
        "expected lib.CONST to resolve against the source file's SysInfo"
    );
}

#[test]
fn test_cross_module_literal_promotion() {
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::linux());
    let mut test_env = TestEnv::new();
    test_env.add("lib", "timeout = 100\nMY_CONST = 42");
    test_env.add(
        "main",
        "import lib; x: str = lib.timeout  # E: `int` is not assignable to `str`",
    );
    let config_file = test_env.config();
    let state = State::new(test_env.config_finder(), TEST_THREAD_COUNT);
    let f = |name: &str| {
        let name = ModuleName::from_str(name);
        let path = find_import(&config_file, name, None, None, &DirEntryCache::new(), None)
            .finding()
            .unwrap();
        Handle::new(name, path, sys_info.dupe())
    };
    let handles = [f("main")];
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(test_env.get_memory());
    transaction.run(&handles, Require::Everything, None);
    transaction
        .get_errors(&handles)
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
    let sys_info = config.get_sys_info();
    let mut sourcedb = MapDatabase::new(sys_info.dupe());
    for (name, path, _) in FILES.iter().rev() {
        sourcedb.insert(
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(path)),
        );
    }
    let handles = sourcedb.modules_to_check();
    config.source_db = Some(ArcId::new(Box::new(sourcedb)));
    config.configure();
    let config = ArcId::new(config);

    let state = State::new(
        ConfigFinder::new_constant(config.clone()),
        TEST_THREAD_COUNT,
    );
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.set_memory(FILES.map(|(_, path, contents)| {
        (
            PathBuf::from(path),
            Some(Arc::new(FileContents::from_source((*contents).to_owned()))),
        )
    }));
    transaction.run(&handles, Require::Everything, None);
    let loads = transaction.get_errors(handles.iter());
    let project_root = PathBuf::new();
    print_errors(project_root.as_path(), &loads.collect_display_errors());
    loads.check_against_expectations().unwrap();
    assert_eq!(loads.collect_errors().ordinary.len(), 3);
}

#[test]
fn test_tensor_shapes_availability_uses_origin_sensitive_resolution() {
    let tdir = TempDir::new().unwrap();
    let root = tdir.path();
    let pkg = root.join("pkg");
    let plain = root.join("plain");
    fs::create_dir_all(pkg.join("shape_extensions")).unwrap();
    fs::create_dir_all(&plain).unwrap();
    fs::write(root.join(ConfigFile::PYREFLY_FILE_NAME), "").unwrap();
    fs::write(
        pkg.join("shape_extensions").join("__init__.pyi"),
        r#"
from typing import Any

shaped_array: Any
class IntTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
"#,
    )
    .unwrap();
    fs::write(
        pkg.join("torch.pyi"),
        r#"
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]: ...
"#,
    )
    .unwrap();
    fs::write(
        pkg.join("jaxtyping.pyi"),
        r#"
class Float[*Shape]: ...
"#,
    )
    .unwrap();
    fs::write(
        plain.join("torch.pyi"),
        r#"
class Tensor[*Shape]: ...
"#,
    )
    .unwrap();
    fs::write(
        plain.join("jaxtyping.pyi"),
        r#"
class Float[*Shape]: ...
"#,
    )
    .unwrap();
    let shaped_main_path = pkg.join("main.py");
    fs::write(
        &shaped_main_path,
        r#"
from jaxtyping import Float
from torch import Tensor
from typing import reveal_type

def f(x: Float[Tensor, "batch channels"]) -> None:
    reveal_type(x)  # E: revealed type: Shaped[Tensor, "batch channels"]
"#,
    )
    .unwrap();
    let plain_main_path = plain.join("main.py");
    fs::write(
        &plain_main_path,
        r#"
from jaxtyping import Float
from torch import Tensor

def f(x: Float[Tensor, "batch channels"]) -> None:
    pass
"#,
    )
    .unwrap();

    let mut config = ConfigFile {
        source: ConfigSource::File(root.join(ConfigFile::PYREFLY_FILE_NAME)),
        enable_fallback_search_path: true,
        ..Default::default()
    };
    config.python_environment.set_empty_to_default();
    config.interpreters.skip_interpreter_query = true;
    let mut sourcedb = MapDatabase::new(config.get_sys_info());
    // The source DB can prove `shape_extensions` is absent, but directory-relative
    // fallback still makes the filesystem lookup depend on the importing origin.
    sourcedb.insert(
        ModuleName::from_str("pkg.main"),
        ModulePath::filesystem(shaped_main_path.clone()),
    );
    sourcedb.insert(
        ModuleName::from_str("plain.main"),
        ModulePath::filesystem(plain_main_path.clone()),
    );
    config.source_db = Some(ArcId::new(Box::new(sourcedb)));
    config.configure();
    let config = ArcId::new(config);
    let sys_info = config.get_sys_info();
    let state = State::new(ConfigFinder::new_constant(config.dupe()), TEST_THREAD_COUNT);
    let shaped_handle = Handle::new(
        ModuleName::from_str("pkg.main"),
        ModulePath::filesystem(shaped_main_path),
        sys_info.dupe(),
    );
    let plain_handle = Handle::new(
        ModuleName::from_str("plain.main"),
        ModulePath::filesystem(plain_main_path),
        sys_info,
    );

    let mut transaction = state.new_transaction(Require::Everything, None);
    assert!(transaction.tensor_shapes_available(&config, &shaped_handle, None));
    assert!(!transaction.tensor_shapes_available(&config, &plain_handle, None));

    transaction.run(&[shaped_handle.dupe()], Require::Everything, None);
    let errors = transaction.get_errors([&shaped_handle]);
    print_errors(PathBuf::new().as_path(), &errors.collect_display_errors());
    errors.check_against_expectations().unwrap();
}

/// Regression test: the per-module `tensor_shapes` bit is derived from whether
/// `shape_extensions` is resolvable, but the module never imports it directly.
/// When `shape_extensions` is created on disk (a find invalidation), a module that
/// does not import it must still rebuild and pick up the now-true bit. This exercises
/// the find-only `tensor_shapes` dependency in the `dirty.find()` clean-check.
#[test]
fn test_tensor_shapes_find_invalidation_rebuilds_module() {
    let tdir = TempDir::new().unwrap();
    let root = tdir.path();
    fs::write(root.join(ConfigFile::PYREFLY_FILE_NAME), "").unwrap();
    // `torch` and `jaxtyping` are present from the start, but `shape_extensions` is not,
    // so shapes are initially unavailable for `main`. `torch` references `shaped_array`
    // from `shape_extensions` so the shaped form is only derivable once it resolves.
    fs::write(
        root.join("torch.pyi"),
        r#"
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]: ...
"#,
    )
    .unwrap();
    fs::write(
        root.join("jaxtyping.pyi"),
        r#"
class Float[*Shape]: ...
"#,
    )
    .unwrap();
    let main_path = root.join("main.py");
    fs::write(
        &main_path,
        r#"
from jaxtyping import Float
from torch import Tensor
from typing import reveal_type

def f(x: Float[Tensor, "batch channels"]) -> None:
    reveal_type(x)
"#,
    )
    .unwrap();

    let mut config = ConfigFile {
        source: ConfigSource::File(root.join(ConfigFile::PYREFLY_FILE_NAME)),
        enable_fallback_search_path: true,
        ..Default::default()
    };
    config.python_environment.set_empty_to_default();
    config.interpreters.skip_interpreter_query = true;
    config.configure();
    let config = ArcId::new(config);
    let sys_info = config.get_sys_info();
    let state = State::new(ConfigFinder::new_constant(config.dupe()), TEST_THREAD_COUNT);
    let handle = Handle::new(
        ModuleName::from_str("main"),
        ModulePath::filesystem(main_path),
        sys_info,
    );

    // Before `shape_extensions` exists, shapes are unavailable and the revealed type is
    // the unshaped jaxtyping `Float` form, not the shaped form. We commit this transaction
    // so the stored `tensor_shapes = Some(false)` bit persists into the main state — this
    // is what makes the next transaction exercise the incremental `dirty.find()` re-check.
    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    assert!(
        !transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let before = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        before.contains("revealed type: Float[") && !before.contains("Shaped["),
        "expected unshaped Float type before shape_extensions exists, got: {before}"
    );
    state.commit_transaction(transaction, None);

    // Create `shape_extensions` on disk: this is a find invalidation that the production
    // code triggers via `invalidate_events` on file creation.
    let shape_ext_path = root.join("shape_extensions.pyi");
    fs::write(
        &shape_ext_path,
        r#"
from typing import Any

shaped_array: Any
class IntTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
"#,
    )
    .unwrap();

    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    transaction.as_mut().invalidate_events(&CategorizedEvents {
        created: vec![shape_ext_path.clone()],
        ..Default::default()
    });

    // Now shapes are available, and `main` — which never imports `shape_extensions` —
    // must rebuild and re-derive the shaped type. This only happens if the `dirty.find()`
    // re-check noticed the stored `tensor_shapes` bit flipped from false to true.
    assert!(
        transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let after = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        after.contains(r#"revealed type: Shaped[Tensor, "batch channels"]"#)
            && !after.contains("revealed type: Float["),
        "expected Shaped type (and no stale unshaped Float) after shape_extensions created, got: {after}"
    );

    // Commit the rebuilt transaction, then remove `shape_extensions` again. This proves the
    // `Some(true)` bit written *during the dirty.find() rebuild* (not an initial run) survives
    // `take_and_freeze`/`clone_for_mutation`: the next transaction's dirty.find must see the
    // committed `Some(true)`, notice it flipped back to false, and revert `main` to unshaped.
    state.commit_transaction(transaction, None);
    fs::remove_file(&shape_ext_path).unwrap();

    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    transaction.as_mut().invalidate_events(&CategorizedEvents {
        removed: vec![shape_ext_path],
        ..Default::default()
    });
    assert!(
        !transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let reverted = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        reverted.contains("revealed type: Float[") && !reverted.contains("Shaped["),
        "expected revert to unshaped Float after removing shape_extensions, got: {reverted}"
    );
}

/// Regression test for source-db rebuild invalidation: when the source DB changes
/// from proving `shape_extensions` absent to resolving it, the stored find-only
/// `tensor_shapes` bit must be rechecked and affected modules must rebuild.
#[test]
fn test_tensor_shapes_source_db_rebuild_rechecks_marker_availability() {
    let shape_extensions_available = Arc::new(Mutex::new(false));
    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    let sys_info = config.get_sys_info();
    config.source_db = Some(ArcId::new(Box::new(MutableShapeExtensionsSourceDb::new(
        sys_info.dupe(),
        shape_extensions_available.dupe(),
    ))));
    config.configure();
    let config = ArcId::new(config);
    let state = State::new(ConfigFinder::new_constant(config.dupe()), TEST_THREAD_COUNT);
    let handle = Handle::new(
        ModuleName::from_str("main"),
        ModulePath::memory(PathBuf::from("main.py")),
        sys_info,
    );
    let memory = vec![
        (
            PathBuf::from("main.py"),
            Some(Arc::new(FileContents::from_source(
                r#"
from jaxtyping import Float
from torch import Tensor
from typing import reveal_type

def f(x: Float[Tensor, "batch channels"]) -> None:
    reveal_type(x)
"#
                .to_owned(),
            ))),
        ),
        (
            PathBuf::from("torch.pyi"),
            Some(Arc::new(FileContents::from_source(
                r#"
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]: ...
"#
                .to_owned(),
            ))),
        ),
        (
            PathBuf::from("jaxtyping.pyi"),
            Some(Arc::new(FileContents::from_source(
                r#"
class Float[*Shape]: ...
"#
                .to_owned(),
            ))),
        ),
        (
            PathBuf::from("shape_extensions.pyi"),
            Some(Arc::new(FileContents::from_source(
                r#"
from typing import Any

shaped_array: Any
class IntTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
"#
                .to_owned(),
            ))),
        ),
    ];

    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    transaction.as_mut().set_memory(memory);
    assert!(
        !transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let before = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        before.contains("revealed type: Float[") && !before.contains("Shaped["),
        "expected unshaped Float type before source DB resolves shape_extensions, got: {before}"
    );
    state.commit_transaction(transaction, None);

    *shape_extensions_available.lock() = true;
    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    transaction
        .as_mut()
        .invalidate_find_for_configs(SmallSet::from_iter([config.dupe()]));
    assert!(
        transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let after = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        after.contains(r#"revealed type: Shaped[Tensor, "batch channels"]"#)
            && !after.contains("revealed type: Float["),
        "expected Shaped type after source DB resolves shape_extensions, got: {after}"
    );
}

/// Reverse of `test_tensor_shapes_find_invalidation_rebuilds_module`: when
/// `shape_extensions` is removed from disk (a find invalidation), a module that
/// never imported it directly must still rebuild and drop the now-false bit,
/// reverting to the unshaped form. Guards against only handling the create case.
#[test]
fn test_tensor_shapes_find_invalidation_drops_shapes_on_removal() {
    let tdir = TempDir::new().unwrap();
    let root = tdir.path();
    fs::write(root.join(ConfigFile::PYREFLY_FILE_NAME), "").unwrap();
    // `shape_extensions` is present from the start, so shapes are initially available.
    let shape_ext_path = root.join("shape_extensions.pyi");
    fs::write(
        &shape_ext_path,
        r#"
from typing import Any

shaped_array: Any
class IntTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
"#,
    )
    .unwrap();
    fs::write(
        root.join("torch.pyi"),
        r#"
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]: ...
"#,
    )
    .unwrap();
    fs::write(
        root.join("jaxtyping.pyi"),
        r#"
class Float[*Shape]: ...
"#,
    )
    .unwrap();
    let main_path = root.join("main.py");
    fs::write(
        &main_path,
        r#"
from jaxtyping import Float
from torch import Tensor
from typing import reveal_type

def f(x: Float[Tensor, "batch channels"]) -> None:
    reveal_type(x)
"#,
    )
    .unwrap();

    let mut config = ConfigFile {
        source: ConfigSource::File(root.join(ConfigFile::PYREFLY_FILE_NAME)),
        enable_fallback_search_path: true,
        ..Default::default()
    };
    config.python_environment.set_empty_to_default();
    config.interpreters.skip_interpreter_query = true;
    config.configure();
    let config = ArcId::new(config);
    let sys_info = config.get_sys_info();
    let state = State::new(ConfigFinder::new_constant(config.dupe()), TEST_THREAD_COUNT);
    let handle = Handle::new(
        ModuleName::from_str("main"),
        ModulePath::filesystem(main_path),
        sys_info,
    );

    // While `shape_extensions` exists, shapes are available and the revealed type is the
    // shaped form. We commit so the stored `tensor_shapes = Some(true)` bit persists into
    // the main state, setting up the incremental `dirty.find()` re-check on removal.
    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    assert!(
        transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let before = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        before.contains(r#"revealed type: Shaped[Tensor, "batch channels"]"#),
        "expected Shaped type while shape_extensions exists, got: {before}"
    );
    state.commit_transaction(transaction, None);

    // Remove `shape_extensions` from disk: a find invalidation triggered in production via
    // `invalidate_events` on file removal.
    fs::remove_file(&shape_ext_path).unwrap();

    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    transaction.as_mut().invalidate_events(&CategorizedEvents {
        removed: vec![shape_ext_path],
        ..Default::default()
    });

    // Now shapes are unavailable, and `main` — which never imports `shape_extensions` —
    // must rebuild and revert to the unshaped type. This only happens if the `dirty.find()`
    // re-check noticed the stored `tensor_shapes` bit flipped from true to false.
    assert!(
        !transaction
            .as_mut()
            .tensor_shapes_available(&config, &handle, None)
    );
    transaction
        .as_mut()
        .run(&[handle.dupe()], Require::Everything, None);
    let after = transaction
        .as_mut()
        .get_errors([&handle])
        .collect_display_errors()
        .map(|e| e.msg())
        .join("\n");
    assert!(
        after.contains("revealed type: Float[") && !after.contains("Shaped["),
        "expected unshaped Float type after shape_extensions removed, got: {after}"
    );
}

#[test]
fn test_change_require() {
    let env = TestEnv::one("foo", "x: str = 1\ny: int = 'x'");
    let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
    let handle = Handle::new(
        ModuleName::from_str("foo"),
        ModulePath::memory(PathBuf::from("foo.py")),
        env.sys_info(),
    );

    let mut t = state.new_committable_transaction(Require::Exports, None);
    t.as_mut().set_memory(env.get_memory());
    t.as_mut().run(&[handle.dupe()], Require::Exports, None);
    state.commit_transaction(t, None);

    assert_eq!(
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .ordinary
            .len(),
        0
    );
    assert!(state.transaction().get_bindings(&handle).is_none());
    state.run(
        &[handle.dupe()],
        RequireLevels {
            specified: Require::Errors,
            default: Require::Exports,
        },
        None,
        None,
        None,
    );
    assert_eq!(
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .ordinary
            .len(),
        2
    );
    assert!(state.transaction().get_bindings(&handle).is_none());
    state.run(
        &[handle.dupe()],
        RequireLevels {
            specified: Require::Everything,
            default: Require::Exports,
        },
        None,
        None,
        None,
    );
    assert_eq!(
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .ordinary
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
        Some(Arc::new(FileContents::from_source("x = 3".to_owned()))),
    )]);
    t.as_mut().run(&[], Require::Everything, None); // This run breaks reproduction (but is now required)
    state.commit_transaction(t, None);

    // Now we need to increment the step counter.
    let mut t = state.new_committable_transaction(REQUIRE, None);
    t.as_mut().run(&[], Require::Everything, None);
    state.commit_transaction(t, None);

    // Now we run two searches, this used to crash
    let t = state.new_transaction(REQUIRE, None);
    eprintln!("First search");
    t.search_exports_exact("x", None).unwrap();
    eprintln!("Second search");
    t.search_exports_exact("x", None).unwrap();
}

#[test]
fn test_search_exports_cancellation() {
    let mut t = TestEnv::new();
    t.add("foo", "x = 1");
    let (state, _) = t.to_state();

    let t = state.new_transaction(Require::Everything, None);

    // Cancel the transaction before searching.
    // The cancellation check in search_exports' get_module loop fires immediately.
    t.get_cancellation_handle().cancel();
    assert!(
        t.search_exports_exact("x", None).is_err(),
        "search_exports_exact should return Err(Cancelled) when cancelled"
    );
}

#[test]
fn test_compute_stdlib_uses_bundled_typeshed_even_with_custom_path() {
    use std::fs;

    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let typeshed_path = temp_dir.path().join("custom_typeshed");
    let stdlib_path = typeshed_path.join("stdlib");
    fs::create_dir_all(&stdlib_path).unwrap();

    // Create a minimal builtins.pyi that defines int as a class, but very minimally.
    // The key insight: Stdlib object loads built-in types (int, str, etc.) and their
    // class definitions for type-checking. Since our custom typeshed's int is used
    // for imports but bundled typeshed's int is used for Stdlib, we get a type mismatch.
    let builtins_content = r#"
class object: ...
class type: ...
class int:
    # This is a minimal int that doesn't have many methods
    # The bundled typeshed has a full implementation
    pass
class str: ...
class bool(int): ...
class float: ...
class list: ...
class dict: ...
class tuple: ...
class None: ...
"#;
    fs::write(stdlib_path.join("builtins.pyi"), builtins_content).unwrap();

    fs::write(stdlib_path.join("VERSIONS"), "builtins: 3.0-\n").unwrap();

    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    config.typeshed_path = Some(typeshed_path.clone());
    let sys_info = config.get_sys_info();

    // Create a test module. The bug manifests as follows:
    // - The `int` annotation is resolved from the custom typeshed
    // - The Stdlib's int (used for Literal[1]) comes from bundled typeshed
    // - These are different types, causing "Literal[1] is not assignable to int"
    let test_code = r#"
# This simple assignment demonstrates the bug:
# - The `int` type annotation is resolved from custom typeshed (builtins.int@4:7-10)
# - The Literal[1] type comes from bundled Stdlib (builtins.int@420:7-10)
# - Since these are different types, we get a type error
x: int = 1
"#;
    let module_name = ModuleName::from_str("test_module");
    let module_path = ModulePath::memory(PathBuf::from("test_module.py"));

    let mut sourcedb = MapDatabase::new(sys_info.dupe());
    sourcedb.insert(module_name, module_path.dupe());
    config.source_db = Some(ArcId::new(Box::new(sourcedb)));
    config.configure();
    let config = ArcId::new(config);

    let state = State::new(
        ConfigFinder::new_constant(config.clone()),
        TEST_THREAD_COUNT,
    );
    let handle = Handle::new(module_name, module_path, sys_info);

    let mut transaction = state.new_transaction(Require::Everything, None);
    transaction.set_memory(vec![(
        PathBuf::from("test_module.py"),
        Some(Arc::new(FileContents::from_source(test_code.to_owned()))),
    )]);
    transaction.run(&[handle.dupe()], Require::Everything, None);

    let errors = transaction.get_errors([&handle]).collect_errors();

    assert!(
        config.typeshed_path.is_some(),
        "Test setup error: typeshed_path should be set in config"
    );
    assert_eq!(
        config.typeshed_path.as_ref().unwrap(),
        &typeshed_path,
        "Test setup error: typeshed_path should match the custom path"
    );

    // Verify the specific error is the expected type mismatch. This is specific
    // error is an indication that we are not using the typeshed bundled with Pyrefly
    // and not the typeshed provided through the config.
    let error_messages: Vec<String> = errors
        .ordinary
        .iter()
        .map(|e| e.msg().to_string())
        .collect();
    let has_literal_int_error = error_messages
        .iter()
        .any(|msg| msg.contains("Literal[1]") && msg.contains("int"));
    assert!(
        has_literal_int_error,
        "Expected error about Literal[1] not being assignable to int, but got: {:?}. \
         This error demonstrates the mismatch between bundled Stdlib int and custom typeshed int.",
        error_messages
    );
}

const SEQUENTIAL_COMMITTABLE_TRANSACTIONS_SLEEP_TIME_MS: u64 = 100;

#[test]
fn test_sequential_committable_transactions() {
    let mut t = TestEnv::new();
    t.add("foo", "x = 1");
    let (state, _) = t.to_state();
    let state = Arc::new(state);
    let state1 = state.dupe();
    let state2 = state.dupe();
    let state3 = state.dupe();
    let state4 = state.dupe();
    let state5 = state.dupe();
    let counter = Arc::new(Mutex::new(0));
    let counter1 = counter.dupe();
    let counter2 = counter.dupe();
    let counter3 = counter.dupe();
    let counter4 = counter.dupe();
    let counter5 = counter.dupe();

    /// This function is called in many separate threads.
    /// We want to make sure that state only allow one committable transaction at a time.
    /// We will indirectly verify this by having each thread increment a shared counter twice
    /// during the transaction. Due to the exclusivity of the transaction, we should never be able
    /// to observe that the counter has an odd value.
    fn do_work_and_verify(state: Arc<State>, counter: Arc<Mutex<usize>>) {
        let t = state.new_committable_transaction(Require::Exports, None);
        let initial_value = {
            let mut lock = counter.lock();
            let v = *lock;
            assert!(v.is_multiple_of(2));
            *lock += 1;
            v
        };
        sleep(Duration::from_millis(
            SEQUENTIAL_COMMITTABLE_TRANSACTIONS_SLEEP_TIME_MS,
        ));
        {
            let mut lock = counter.lock();
            let v = *lock;
            *lock += 1;
            assert_eq!(initial_value + 1, v);
            v
        };
        state.commit_transaction(t, None);
    }

    // We rapidly spin up 5 threads, each of which will increment the counter twice.
    let t1 = std::thread::spawn(move || {
        do_work_and_verify(state1, counter1);
    });
    let t2 = std::thread::spawn(move || {
        do_work_and_verify(state2, counter2);
    });
    let t3 = std::thread::spawn(move || {
        do_work_and_verify(state3, counter3);
    });
    let t4 = std::thread::spawn(move || {
        do_work_and_verify(state4, counter4);
    });
    let t5 = std::thread::spawn(move || {
        do_work_and_verify(state5, counter5);
    });

    t1.join().unwrap();
    t2.join().unwrap();
    t3.join().unwrap();
    t4.join().unwrap();
    t5.join().unwrap();

    // When we are here, we are sure that there is no deadlock.
    let lock = counter.lock();
    assert_eq!(10, *lock);
}

/// Test that fixing a previously malformed notebook triggers a rebuild.
/// Regression test for a bug where the reload logic returned `false` when
/// old_load.module_info.notebook() was None, preventing rebuilds.
#[test]
fn test_notebook_reload_after_parse_failure() {
    let temp_dir = TempDir::new().unwrap();
    let notebook_path = temp_dir.path().join("test.ipynb");

    // Start with invalid JSON
    fs::write(&notebook_path, "{ invalid json }").unwrap();

    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    config.configure();
    let config = ArcId::new(config);
    let sys_info = config.get_sys_info();
    let state = State::new(ConfigFinder::new_constant(config), TEST_THREAD_COUNT);
    let module_name = ModuleName::from_str("test");
    let module_path = ModulePath::filesystem(notebook_path.clone());
    let handle = Handle::new(module_name, module_path, sys_info);

    // First run: malformed notebook produces load error
    let mut t = state.new_committable_transaction(Require::Exports, None);
    t.as_mut().run(&[handle.dupe()], Require::Errors, None);
    state.commit_transaction(t, None);
    assert_eq!(
        1,
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .ordinary
            .len()
    );

    // Fix the notebook with valid JSON
    let valid_notebook = r#"{
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }"#;
    fs::write(&notebook_path, valid_notebook).unwrap();

    // Invalidate and re-run - should now have no errors
    let mut t = state.new_committable_transaction(Require::Exports, None);
    t.as_mut()
        .invalidate_disk(std::slice::from_ref(&notebook_path));
    t.as_mut().run(&[handle.dupe()], Require::Errors, None);
    state.commit_transaction(t, None);

    assert_eq!(
        0,
        state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .ordinary
            .len()
    );
}

/// Regression test for a crash where `get_module().finding().unwrap()` used to
/// panic in `TransactionHandle::get()` (state.rs) when resolving a cross-module
/// `TypeAliasRef` and the current module's config cannot find the defining module.
///
/// Scenario:
/// - `baz` defines a recursive type alias `type Tree = int | list[Tree]`
/// - `foo` re-exports `Tree` from `baz`
/// - `main` imports `Tree` from `foo` and uses it in an annotation
/// - `main`'s config can find `foo` but NOT `baz`
/// - `foo`'s config can find both `foo` and `baz`
///
/// When `main` resolves the `TypeAliasRef { module: baz }` embedded in the
/// recursive type, it calls `get_module(baz)` which returns `None` because
/// `main`'s config cannot locate `baz`. This should be handled gracefully
/// instead of panicking.
#[test]
fn test_crash_on_cross_module_type_alias_ref() {
    let temp_dir = TempDir::new().unwrap();

    // main_proj/ contains main.py with a config that can see dep_proj/ but NOT extra/.
    // dep_proj/ contains foo.py with a config that can see extra/.
    // extra/ contains baz.py (no config needed, discovered via dep_proj's search path).
    let main_proj = temp_dir.path().join("main_proj");
    let dep_proj = temp_dir.path().join("dep_proj");
    let extra = temp_dir.path().join("extra");
    fs::create_dir_all(&main_proj).unwrap();
    fs::create_dir_all(&dep_proj).unwrap();
    fs::create_dir_all(&extra).unwrap();

    fs::write(
        main_proj.join("main.py"),
        "from foo import Tree\nx: Tree = [[1]]",
    )
    .unwrap();
    fs::write(dep_proj.join("foo.py"), "from baz import Tree as Tree").unwrap();
    fs::write(extra.join("baz.py"), "type Tree = int | list[Tree]").unwrap();

    // main's config: search-path includes main_proj and dep_proj, but NOT extra.
    fs::write(
        main_proj.join("pyrefly.toml"),
        "search-path = [\".\", \"../dep_proj\"]\nskip-interpreter-query = true\n",
    )
    .unwrap();
    // foo/baz's config: search-path includes dep_proj and extra.
    fs::write(
        dep_proj.join("pyrefly.toml"),
        "search-path = [\".\", \"../extra\"]\nskip-interpreter-query = true\n",
    )
    .unwrap();

    let finder = default_config_finder(None);
    let main_path = ModulePath::filesystem(main_proj.join("main.py"));
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::linux());
    let handle = Handle::new(ModuleName::from_str("main"), main_path, sys_info);

    let state = State::new(finder, TEST_THREAD_COUNT);
    let mut transaction = state.new_transaction(Require::Exports, None);
    transaction.run(&[handle], Require::Everything, None);
}

/// Verify that stdlib computation is cached across transaction runs.
///
/// The first run must compute the stdlib from bundled typeshed stubs (expensive,
/// 80-150ms single-threaded). Subsequent runs within the same transaction, or
/// new transactions created after committing, should reuse the cached stdlib
/// and report `compute_stdlib_cached = true`.
#[test]
fn test_stdlib_cached_on_recheck() {
    let env = TestEnv::one("foo", "x: int = 1");
    let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
    let handle = Handle::new(
        ModuleName::from_str("foo"),
        ModulePath::memory(PathBuf::from("foo.py")),
        env.sys_info(),
    );

    // First run: stdlib must be computed from scratch.
    let mut t1 = state.new_committable_transaction(Require::Exports, None);
    t1.as_mut().set_memory(env.get_memory());
    t1.as_mut().run(&[handle.dupe()], Require::Everything, None);
    assert!(
        !t1.as_ref().compute_stdlib_cached(),
        "First run should compute stdlib, not use cache"
    );
    assert!(
        t1.as_ref().compute_stdlib_prewarm_time() > Duration::ZERO,
        "Pre-warming should take nonzero time on first run"
    );
    state.commit_transaction(t1, None);

    // Second run (recheck): stdlib should be cached because it was committed.
    let mut t2 = state.new_committable_transaction(Require::Exports, None);
    t2.as_mut().set_memory(env.get_memory());
    t2.as_mut().run(&[handle.dupe()], Require::Everything, None);
    assert!(
        t2.as_ref().compute_stdlib_cached(),
        "Recheck should use cached stdlib, not recompute"
    );
    assert_eq!(
        t2.as_ref().compute_stdlib_prewarm_time(),
        Duration::ZERO,
        "Cached stdlib should skip pre-warming entirely"
    );
    state.commit_transaction(t2, None);
}
