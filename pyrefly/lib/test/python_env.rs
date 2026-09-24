/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Disk-backed Python environment test harness.
//!
//! Composable types for building realistic Python packaging layouts on disk:
//! workspaces, packages (flat or src-layout), virtual environments, editable
//! installs with PEP 610 metadata, `.pth` files, and mock interpreters.
//!
//! All paths are real filesystem paths backed by `TempDir`. Nothing in this
//! module wraps `LspInteraction` or creates `Memory` handles.

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;

use lsp_types::Url;
use pyrefly_util::fs_anyhow::write;
use serde_json::json;
use tempfile::TempDir;

/// A workspace on disk containing Python packages and virtual environments.
///
/// Owns the `TempDir` whose lifetime governs all paths handed out.
pub struct PythonTestWorkspace {
    dir: TempDir,
}

impl PythonTestWorkspace {
    pub fn new() -> Self {
        Self {
            dir: TempDir::with_prefix("pyrefly_test_ws_").unwrap(),
        }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Create a synthetic (non-interpreter) virtual environment in the workspace.
    pub fn create_venv(&self, relative_path: &str) -> TestVenv {
        TestVenv::new_synthetic(self.dir.path().join(relative_path), true)
    }
}

/// A Python package that can be installed into a `TestVenv`.
pub struct TestPackage {
    name: String,
    version: String,
    import_root: PathBuf,
    project_root: PathBuf,
}

impl TestPackage {
    /// Create a package with a src-layout: `{project_root}/src/{package_name}/`.
    pub fn src_layout(project_root: impl Into<PathBuf>, name: &str, version: &str) -> Self {
        let project_root = project_root.into();
        let normalized = name.replace('-', "_");
        let import_root = project_root.join("src");
        fs::create_dir_all(import_root.join(&normalized)).unwrap();
        write(&import_root.join(&normalized).join("__init__.py"), "").unwrap();
        Self {
            name: name.to_owned(),
            version: version.to_owned(),
            import_root,
            project_root,
        }
    }

    /// Create a package with a flat layout: `{project_root}/{package_name}/`.
    pub fn flat_layout(project_root: impl Into<PathBuf>, name: &str, version: &str) -> Self {
        let project_root = project_root.into();
        let normalized = name.replace('-', "_");
        let import_root = project_root.clone();
        fs::create_dir_all(import_root.join(&normalized)).unwrap();
        write(&import_root.join(&normalized).join("__init__.py"), "").unwrap();
        Self {
            name: name.to_owned(),
            version: version.to_owned(),
            import_root,
            project_root,
        }
    }

    /// The normalized distribution name used for `.dist-info` directories.
    fn dist_name(&self) -> String {
        self.name.replace('-', "_")
    }

    pub fn import_root(&self) -> &Path {
        &self.import_root
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }
}

/// The `sys.platform` value a real interpreter on this host would report,
/// for the unix platforms `create_mock_interpreter` runs on.
#[cfg(unix)]
fn mock_python_platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    }
}

/// Directory names pyrefly's workspace auto-discovery scans for a venv root
/// (mirrors `pyrefly_config::environment::venv`'s private candidate list;
/// that module is `pub(crate)` and unreachable from this test-only crate, so
/// the names are duplicated here rather than imported — a stale copy could
/// only make `TestVenv::new_synthetic_excluded_from_discovery` reject a name
/// it need not have, never let a truly-discoverable one slip through).
const AUTO_DISCOVERED_VENV_DIR_NAMES: &[&str] = &[".venv", "venv", "env"];

/// A virtual environment on disk with a `site-packages` directory.
pub struct TestVenv {
    // Only `create_mock_interpreter` (unix-only, since it shells out to a `#!` script) needs
    // the venv root; cfg it out elsewhere so non-unix builds don't warn that it's dead.
    #[cfg(unix)]
    root: PathBuf,
    site_packages: PathBuf,
}

impl TestVenv {
    /// Create a synthetic venv at the given path (no real interpreter).
    ///
    /// Writes `pyvenv.cfg` and creates the `lib/python3.12/site-packages`
    /// directory tree. Use this when the venv must live at a specific path
    /// (e.g. inside a test-fixture directory); otherwise prefer
    /// `PythonTestWorkspace::create_venv`.
    pub fn synthetic(root: impl Into<PathBuf>) -> Self {
        Self::new_synthetic(root.into(), true)
    }

    /// Like `synthetic`, but never creates the `site-packages` directory, for
    /// tests exercising a client-reported site-packages path that is truly
    /// missing on disk (distinct from an existing-but-empty directory).
    pub fn synthetic_without_site_packages(root: impl Into<PathBuf>) -> Self {
        Self::new_synthetic(root.into(), false)
    }

    fn new_synthetic(root: PathBuf, create_site_packages: bool) -> Self {
        fs::create_dir_all(&root).unwrap();
        write(&root.join("pyvenv.cfg"), "").unwrap();
        let site_packages = root.join("lib/python3.12/site-packages");
        if create_site_packages {
            fs::create_dir_all(&site_packages).unwrap();
        }
        #[cfg(unix)]
        return Self {
            root,
            site_packages,
        };
        #[cfg(not(unix))]
        Self { site_packages }
    }

    /// Create a synthetic venv at `root`, for a venv that must be reachable
    /// only through explicit configuration (`pythonPath` or
    /// `python-interpreter-path`), never through workspace auto-discovery.
    ///
    /// Panics if `root`'s directory name is one auto-discovery scans for, so
    /// that invariant is checked by the harness itself instead of relying on
    /// each call site to pick a safe-looking name and explain why in a
    /// comment.
    pub fn new_synthetic_excluded_from_discovery(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let name = root.file_name().and_then(|name| name.to_str());
        assert!(
            !name.is_some_and(|name| AUTO_DISCOVERED_VENV_DIR_NAMES.contains(&name)),
            "TestVenv: {root:?} would be auto-discovered by pyrefly (directory name is one of \
             {AUTO_DISCOVERED_VENV_DIR_NAMES:?}); this venv must be reachable only through \
             explicit configuration, so it needs a different directory name"
        );
        Self::new_synthetic(root, true)
    }

    pub fn site_packages(&self) -> &Path {
        &self.site_packages
    }

    /// Install a package as editable with UV-equivalent metadata:
    /// - A `.pth` file in site-packages pointing to the package's import root
    /// - A `.dist-info/METADATA` file with package name and version
    /// - A `.dist-info/direct_url.json` with PEP 610 editable marker
    pub fn install_editable(&self, package: &TestPackage) {
        let dist_name = package.dist_name();

        // .pth file: UV writes `_editable_impl_{name}.pth`
        write(
            &self
                .site_packages
                .join(format!("_editable_impl_{dist_name}.pth")),
            format!("{}\n", package.import_root.display()),
        )
        .unwrap();

        // .dist-info directory
        let dist_info = self
            .site_packages
            .join(format!("{dist_name}-{}.dist-info", package.version));
        fs::create_dir_all(&dist_info).unwrap();

        // METADATA
        write(
            &dist_info.join("METADATA"),
            format!(
                "Metadata-Version: 2.1\nName: {}\nVersion: {}\n",
                package.name, package.version
            ),
        )
        .unwrap();

        // PEP 610 direct_url.json
        let project_url = Url::from_file_path(&package.project_root).unwrap();
        write(
            &dist_info.join("direct_url.json"),
            json!({"url": project_url.as_str(), "dir_info": {"editable": true}}).to_string(),
        )
        .unwrap();
    }

    /// Copy a Python module file into site-packages, keeping its filename.
    /// Returns `self` so call sites that immediately follow up with e.g.
    /// `create_mock_interpreter` can chain into one expression.
    pub fn add_site_package_module(&self, source: &Path) -> &Self {
        let filename = source
            .file_name()
            .expect("module source must have a filename");
        fs::copy(source, self.site_packages.join(filename)).unwrap();
        self
    }

    /// Shorthand for `Self::synthetic(root).create_mock_interpreter()`, for
    /// call sites that only need the interpreter path and have no other use
    /// for the venv.
    #[cfg(unix)]
    pub fn mock_interpreter(root: impl Into<PathBuf>) -> PathBuf {
        Self::synthetic(root).create_mock_interpreter()
    }

    /// Like `mock_interpreter`, for an interpreter that must be reachable
    /// only through explicit configuration (`pythonPath` or
    /// `python-interpreter-path`), never through workspace auto-discovery.
    /// See `new_synthetic_excluded_from_discovery` for the enforced invariant.
    #[cfg(unix)]
    pub fn mock_interpreter_excluded_from_discovery(root: impl Into<PathBuf>) -> PathBuf {
        Self::new_synthetic_excluded_from_discovery(root).create_mock_interpreter()
    }

    /// Create a mock Python interpreter at `bin/python` that responds to
    /// pyrefly's environment query with this venv's site-packages path.
    #[cfg(unix)]
    pub fn create_mock_interpreter(&self) -> PathBuf {
        // Serialize through `serde_json` (rather than interpolating
        // `to_str().unwrap()` into a hand-built JSON literal) so a
        // site-packages path containing `"` or `\` still produces valid JSON,
        // and a non-UTF8 path fails with a clear message instead of panicking
        // inside `to_str()`.
        let site_packages = serde_json::to_value(&self.site_packages)
            .expect("mock site-packages path must be valid UTF-8");
        let response = json!({
            "python_platform": mock_python_platform(),
            "python_version": "3.12.0",
            "site_package_path": [site_packages],
            "stdlib_paths": [],
            "distribution_urls": [],
        });

        let script = format!(
            r#"#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == *"import importlib.metadata, json, sys, sysconfig"* ]]; then
    cat << 'EOF'
{response}
EOF
else
    echo "Mock python interpreter - args: $@" >&2
    exit 1
fi
"#
        );

        let bin_dir = self.root.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let interpreter_path = bin_dir.join("python");
        write(&interpreter_path, script).unwrap();
        let mut perms = fs::metadata(&interpreter_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&interpreter_path, perms).unwrap();

        interpreter_path
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::*;

    #[test]
    fn test_new_synthetic_excluded_from_discovery_rejects_auto_discovered_names() {
        let ws = PythonTestWorkspace::new();
        for reserved in AUTO_DISCOVERED_VENV_DIR_NAMES {
            let root = ws.path().join(reserved);
            let result =
                std::panic::catch_unwind(|| TestVenv::new_synthetic_excluded_from_discovery(root));
            assert!(
                result.is_err(),
                "expected a panic for reserved directory name {reserved:?}"
            );
        }
    }

    #[test]
    fn test_editable_install_creates_metadata() {
        let ws = PythonTestWorkspace::new();
        let pkg = TestPackage::src_layout(ws.path().join("my_pkg"), "my-pkg", "1.0.0");
        let venv = ws.create_venv(".venv");
        venv.install_editable(&pkg);

        assert!(
            venv.site_packages()
                .join("_editable_impl_my_pkg.pth")
                .exists()
        );
        assert!(
            venv.site_packages()
                .join("my_pkg-1.0.0.dist-info/METADATA")
                .exists()
        );
        assert!(
            venv.site_packages()
                .join("my_pkg-1.0.0.dist-info/direct_url.json")
                .exists()
        );

        let pth =
            fs::read_to_string(venv.site_packages().join("_editable_impl_my_pkg.pth")).unwrap();
        assert!(pth.contains(&pkg.import_root().to_string_lossy().to_string()));

        let direct_url: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                venv.site_packages()
                    .join("my_pkg-1.0.0.dist-info/direct_url.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(direct_url["dir_info"]["editable"], json!(true));
    }

    #[cfg(unix)]
    #[test]
    fn test_mock_interpreter_responds_to_env_query() {
        let ws = PythonTestWorkspace::new();
        let venv = ws.create_venv(".venv");
        let interpreter_path = venv.create_mock_interpreter();

        assert!(interpreter_path.exists());
        let output = Command::new(&interpreter_path)
            .args(["-c", "import importlib.metadata, json, sys, sysconfig"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(response["python_platform"], mock_python_platform());
        assert_eq!(response["python_version"], "3.12.0");
        let sp = response["site_package_path"][0].as_str().unwrap();
        assert_eq!(sp, venv.site_packages().to_str().unwrap());
    }

    #[test]
    fn test_synthetic_without_site_packages_does_not_create_directory() {
        let ws = PythonTestWorkspace::new();
        let venv = TestVenv::synthetic_without_site_packages(ws.path().join(".venv"));
        assert!(!venv.site_packages().exists());
    }

    // The mock response is JSON-serialized (via `serde_json`) rather than
    // interpolated as a raw string, so the site-packages path round-trips
    // correctly through JSON even when it contains characters like `"` or
    // `\` that would otherwise need escaping.
    #[cfg(unix)]
    #[test]
    fn test_mock_interpreter_escapes_special_characters_in_path() {
        let ws = PythonTestWorkspace::new();
        for dir_name in [r#"weird "quoted" dir"#, r"weird \backslash\ dir"] {
            let venv = TestVenv::synthetic(ws.path().join(dir_name));
            let interpreter_path = venv.create_mock_interpreter();

            let output = Command::new(&interpreter_path)
                .args(["-c", "import importlib.metadata, json, sys, sysconfig"])
                .output()
                .unwrap();
            assert!(output.status.success());
            let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            let sp = response["site_package_path"][0].as_str().unwrap();
            assert_eq!(sp, venv.site_packages().to_str().unwrap());
        }
    }
}
