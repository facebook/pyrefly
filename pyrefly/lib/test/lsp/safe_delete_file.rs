/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::ClientCapabilities;
use lsp_types::CodeActionResponse;
use lsp_types::Location;
use lsp_types::Uri;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use serde_json::json;
use tempfile::tempdir;

use crate::lsp::non_wasm::safe_delete_file::safe_delete_file_code_actions;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;

fn file_usage_paths(files: &[(&str, &str, &str)], target: &str) -> Vec<String> {
    file_usage_paths_with_indexing(files, target, true)
}

fn file_usage_paths_with_indexing(
    files: &[(&str, &str, &str)],
    target: &str,
    index_bridge: bool,
) -> Vec<String> {
    let root = tempdir().unwrap();
    let mut env = TestEnv::new();
    for (module, relative_path, source) in files {
        let path = root.path().join(relative_path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, source).unwrap();
        env.add_real_path(module, path);
    }
    let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
    let handle = |module: &str| {
        let (_, path, _) = files.iter().find(|(name, _, _)| *name == module).unwrap();
        Handle::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(root.path().join(path)),
            env.sys_info(),
        )
    };
    let roots = files
        .iter()
        .filter(|(name, _, _)| index_bridge || *name != "bridge")
        .map(|(name, _, _)| handle(name))
        .collect::<Vec<_>>();
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().run(&roots, Require::Indexing, None);
    state.commit_transaction(transaction, None);
    let transaction = state.transaction();
    if index_bridge {
        assert!(transaction.get_ast(&handle("bridge")).is_none());
    } else {
        assert!(transaction.get_answers(&handle("bridge")).is_some());
    }
    assert_eq!(
        transaction.get_solutions(&handle("bridge")).is_some(),
        index_bridge
    );
    assert!(
        transaction
            .get_rdeps(&handle(target))
            .iter()
            .any(|handle| handle.module().as_str() == "consumer"),
        "the consumer must be a candidate before import locations are collected"
    );
    let capabilities: ClientCapabilities = serde_json::from_value(json!({
        "workspace": {"workspaceEdit": {
            "documentChanges": true,
            "resourceOperations": ["delete"]
        }}
    }))
    .unwrap();
    let uri = Uri::from_file_path(handle(target).path().as_path()).unwrap();
    let actions = safe_delete_file_code_actions(
        &capabilities,
        true,
        &state,
        &transaction,
        &uri,
        |_| None,
        |usage| {
            Some(Location::new(
                Uri::from_file_path(usage.module.path().as_path()).unwrap(),
                usage.module.to_lsp_range(usage.range),
            ))
        },
    )
    .unwrap();
    assert!(actions.iter().any(|action| {
        matches!(action, CodeActionResponse::CodeAction(action)
            if action.title.starts_with("Delete file ") && action.title.ends_with(" anyway"))
    }));
    let locations: Vec<Location> = actions
        .iter()
        .find_map(|action| {
            let CodeActionResponse::CodeAction(action) = action else {
                return None;
            };
            let command = action.command.as_ref()?;
            (command.command == "pyrefly.findFileUsages")
                .then(|| command.arguments.as_ref().unwrap()[0]["locations"].clone())
        })
        .map(|value| serde_json::from_value(value).unwrap())
        .unwrap_or_default();
    let mut paths = locations
        .iter()
        .map(|location| {
            location
                .uri
                .to_file_path()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn test_safe_delete_file_unaliased_submodule_reexport() {
    let paths = file_usage_paths(
        &[
            ("pkg", "pkg/__init__.py", ""),
            ("pkg.target", "pkg/target.py", "value = 1\n"),
            ("bridge", "bridge.py", "import pkg.target\n"),
            (
                "consumer",
                "consumer.py",
                "from bridge import pkg\nprint(pkg.target.value)\n",
            ),
        ],
        "pkg.target",
    );
    assert_eq!(paths, ["bridge.py", "consumer.py"]);
}

#[test]
fn test_safe_delete_file_excludes_inaccessible_submodules() {
    let paths = file_usage_paths(
        &[
            ("pkg", "pkg/__init__.py", ""),
            ("pkg.target", "pkg/target.py", "value = 1\n"),
            ("pkg.other", "pkg/other.py", "value = 2\n"),
            (
                "bridge",
                "bridge.py",
                "import pkg.target\nimport pkg.other\nexported = pkg.other\n",
            ),
            (
                "consumer",
                "consumer.py",
                "from bridge import exported\nimport pkg.target\nprint(exported.value, pkg.target.value)\n",
            ),
        ],
        "pkg.target",
    );
    assert_eq!(paths, ["bridge.py", "consumer.py"]);
}

#[test]
fn test_safe_delete_file_module_reexports() {
    for (bridge, consumer) in [
        (
            "import target\n",
            "from bridge import target\nprint(target.value)\n",
        ),
        (
            "import target as target\n",
            "from bridge import target\nprint(target.value)\n",
        ),
        (
            "import target as exported\n",
            "from bridge import exported as target\nprint(target.value)\n",
        ),
    ] {
        let paths = file_usage_paths(
            &[
                ("target", "target.py", "value = 1\n"),
                ("bridge", "bridge.py", bridge),
                ("consumer", "consumer.py", consumer),
            ],
            "target",
        );
        assert_eq!(paths, ["bridge.py", "consumer.py"]);
    }
}

#[test]
fn test_safe_delete_file_aliased_submodule_reexport() {
    let paths = file_usage_paths(
        &[
            ("pkg", "pkg/__init__.py", ""),
            ("pkg.target", "pkg/target.py", "value = 1\n"),
            ("bridge", "bridge.py", "import pkg.target as exported\n"),
            (
                "consumer",
                "consumer.py",
                "from bridge import exported\nprint(exported.value)\n",
            ),
        ],
        "pkg.target",
    );
    assert_eq!(paths, ["bridge.py", "consumer.py"]);
}

#[test]
fn test_safe_delete_file_import_only_module_reexports() {
    for bridge in [
        "import target as exported\n",
        "try:\n    import target as exported\nexcept ImportError:\n    exported = None\n",
    ] {
        let paths = file_usage_paths_with_indexing(
            &[
                ("target", "target.py", "value = 1\n"),
                ("bridge", "bridge.py", bridge),
                (
                    "consumer",
                    "consumer.py",
                    "from bridge import exported\nif exported is not None:\n    print(exported.value)\n",
                ),
            ],
            "target",
            false,
        );
        assert_eq!(paths, ["bridge.py", "consumer.py"]);
    }
}

#[test]
fn test_safe_delete_file_wildcard_reexports() {
    for bridge in [
        "from target import f as f, g as g\n__all__ = ['f', 'g']\n",
        "from target import *\n",
    ] {
        let paths = file_usage_paths(
            &[
                (
                    "target",
                    "target.py",
                    "def f():\n    return 1\ndef g():\n    return 2\n",
                ),
                ("bridge", "bridge.py", bridge),
                (
                    "consumer",
                    "consumer.py",
                    "from bridge import *\nprint(f(), g())\n",
                ),
            ],
            "target",
        );
        assert_eq!(paths, ["bridge.py", "consumer.py"]);
    }
}

#[test]
fn test_safe_delete_file_wildcard_excludes_unexported_names() {
    for bridge in [
        "from target import f\nfrom other import g\n__all__ = ['g']\n",
        "from target import f as _hidden\nfrom other import g\n",
    ] {
        let paths = file_usage_paths(
            &[
                ("target", "target.py", "def f():\n    return 1\n"),
                ("other", "other.py", "def g():\n    return 2\n"),
                ("bridge", "bridge.py", bridge),
                (
                    "consumer",
                    "consumer.py",
                    "from bridge import *\nimport target\nprint(g(), target.f())\n",
                ),
            ],
            "target",
        );
        assert_eq!(paths, ["bridge.py", "consumer.py"]);
    }
}
