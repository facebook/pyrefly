/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context as _;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::error::ErrorDisplayConfig;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lock::Mutex;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use tempfile::NamedTempFile;

pub fn set_readonly(path: &Path, value: bool) -> anyhow::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(value);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct BundleFile {
    pub(crate) import_path: PathBuf,
    pub(crate) storage_path: PathBuf,
    pub(crate) contents: String,
}

#[derive(Debug)]
struct Candidate {
    load_index: usize,
    is_package: bool,
}

/// An eagerly resolved view of one virtual import root.
///
/// Package initializers take precedence over same-name modules, and modules block descendants.
#[derive(Debug, Clone)]
pub(crate) struct Bundle {
    find: SmallMap<ModuleName, usize>,
    load: SmallMap<PathBuf, Arc<String>>,
}

impl Bundle {
    pub(crate) fn new<Files>(files: Files) -> anyhow::Result<Self>
    where
        Files: IntoIterator<Item = BundleFile>,
    {
        let mut candidates: SmallMap<ModuleName, Candidate> = SmallMap::new();
        let mut load = SmallMap::new();
        for file in files {
            let module = ModuleName::from_relative_path(&file.import_path)?;
            let is_package = file
                .import_path
                .file_stem()
                .is_some_and(|stem| stem == "__init__");
            let storage_path = Hashed::new(file.storage_path);
            let load_index = load
                .get_index_of_hashed(storage_path.as_ref())
                .unwrap_or(load.len());
            if let Some(candidate) = candidates.get_mut(&module) {
                if is_package && !candidate.is_package {
                    candidate.load_index = load_index;
                    candidate.is_package = true;
                }
            } else {
                candidates.insert(
                    module,
                    Candidate {
                        load_index,
                        is_package,
                    },
                );
            }
            load.insert_hashed(storage_path, Arc::new(file.contents));
        }

        let mut find = SmallMap::new();
        'modules: for (module, candidate) in &candidates {
            let mut ancestor = *module;
            while let Some(parent) = ancestor.parent() {
                ancestor = parent;
                if candidates
                    .get(&parent)
                    .is_some_and(|candidate| !candidate.is_package)
                {
                    continue 'modules;
                }
            }
            find.insert(*module, candidate.load_index);
        }
        Ok(Self { find, load })
    }

    pub(crate) fn find(&self, module: ModuleName) -> Option<&PathBuf> {
        let index = *self.find.get(&module)?;
        let (path, _) = self
            .load
            .get_index(index)
            .expect("bundle find indices refer to immutable load entries");
        Some(path)
    }

    pub(crate) fn load(&self, path: &Path) -> Option<Arc<String>> {
        self.load.get(path).cloned()
    }

    pub(crate) fn modules(&self) -> impl Iterator<Item = ModuleName> + '_ {
        self.find.keys().copied()
    }

    pub(crate) fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)> {
        self.load.iter()
    }
}

#[cfg(test)]
pub(crate) fn assert_bundle_order_independent(files: impl IntoIterator<Item = BundleFile>) {
    fn loaded_modules(bundle: &Bundle) -> Vec<(ModuleName, PathBuf, Arc<String>)> {
        let mut modules = bundle.modules().collect::<Vec<_>>();
        modules.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        modules
            .into_iter()
            .map(|module| {
                let path = bundle
                    .find(module)
                    .expect("bundle modules have selected paths");
                let contents = bundle
                    .load(path)
                    .expect("selected bundle paths have contents");
                (module, path.clone(), contents)
            })
            .collect()
    }

    let files = files.into_iter().collect::<Vec<_>>();
    let forward = Bundle::new(files.clone()).expect("static bundle files should be valid");
    let reverse =
        Bundle::new(files.into_iter().rev()).expect("static bundle files should be valid");
    assert_eq!(loaded_modules(&forward), loaded_modules(&reverse));
}

/// Creates a base config file for bundled stubs with common settings.
///
/// This helper function encapsulates the common configuration logic shared across
/// different bundled stub types (typeshed stdlib, typeshed third-party, and third-party stubs).
pub fn create_bundled_stub_config(
    search_paths: Option<Vec<PathBuf>>,
    error_overrides: Option<HashMap<ErrorKind, Severity>>,
    permissive_ignores: Option<bool>,
) -> ConfigFile {
    let mut config_file = ConfigFile::default();
    config_file.interpreters.skip_interpreter_query = true;
    config_file.python_environment.site_package_path = Some(Vec::new());

    if let Some(paths) = search_paths {
        config_file.search_path_from_file = paths;
    }

    if let Some(overrides) = error_overrides {
        config_file.root.errors = Some(ErrorDisplayConfig::new(overrides));
    }

    config_file.root.disable_type_errors_in_ide = Some(true);
    config_file.root.permissive_ignores = permissive_ignores;
    config_file.configure();
    config_file
}

/// Trait for managing bundled Python stub files (type hints) that are embedded in the binary.
///
/// This trait provides methods for accessing bundled stub files, such as those from typeshed,
/// which are included with the type checker rather than loaded from the file system.
/// Implementations can find modules by name, load their contents, and materialize the bundled
/// files to disk when needed for inspection or debugging.
pub trait BundledStub {
    fn new() -> anyhow::Result<Self>
    where
        Self: Sized;
    fn find(&self, module: ModuleName) -> Option<ModulePath>;
    fn load(&self, path: &Path) -> Option<Arc<String>>;
    fn modules(&self) -> impl Iterator<Item = ModuleName>;
    fn config() -> ArcId<ConfigFile>;
    /// Obtain a materialized path for bundled stubs, writing it all to disk the first time.
    /// This function tracks which paths have been written to disk, so it will only write a path that
    /// has not already been written.
    /// Note: this path is not the source of truth, it simply exists to display typeshed contents
    /// for informative purposes.
    fn materialized_path_on_disk(&self) -> anyhow::Result<PathBuf> {
        static WRITTEN_TO_DISK: LazyLock<Mutex<HashSet<String>>> =
            LazyLock::new(|| Mutex::new(HashSet::new()));

        let path_name = self.get_path_name();
        let temp_dir = env::temp_dir().join(&path_name);

        let mut written_paths = WRITTEN_TO_DISK.lock();
        if !written_paths.contains(&path_name) {
            self.write(&temp_dir)?;
            written_paths.insert(path_name);
        }
        Ok(temp_dir)
    }

    /// Writes all bundled stub files to a directory on disk.
    ///
    /// File writes are atomic (using temp files and rename) to prevent corruption.
    /// Files are made read-only after writing as a guardrail.
    fn write(&self, output_dir: &Path) -> anyhow::Result<()> {
        fs_anyhow::create_dir_all(output_dir)?;

        for (relative_path, contents) in self.load_map() {
            let mut file_path = output_dir.to_owned();
            file_path.push(relative_path);

            if let Some(parent) = file_path.parent() {
                fs_anyhow::create_dir_all(parent)?;
            }

            // Check if the file already exists. If it does, assume another process has already
            // written the file and continue.
            if fs::exists(&file_path).with_context(|| {
                format!("When checking existence of file `{}`", file_path.display())
            })? {
                continue;
            }

            // File writes are not atomic, so we write to a tempfile then atomically _rename_ to
            // the destination file.
            let mut temp_file = NamedTempFile::new().with_context(|| {
                format!("When creating temp file for `{}`", file_path.display())
            })?;
            temp_file.write_all(contents.as_bytes()).with_context(|| {
                format!("When writing to temp file for `{}`", file_path.display())
            })?;
            temp_file.flush().with_context(|| {
                format!("When flushing to temp file for `{}`", file_path.display())
            })?;

            // If we can't persist (atomically rename) the file, check to see if the file exists.
            // If so, assume another process has written the file and made it readonly, causing
            // the error.
            match temp_file.persist(&file_path) {
                Ok(_) => {
                    // Make file readonly as a guardrail, since editing the bundled typeshed files
                    // can lead to surprising behavior. This can fail, but we ignore errors because
                    // this is not critical.
                    let _ = set_readonly(&file_path, true);
                    Ok(())
                }
                Err(e) => {
                    if fs::exists(&file_path).is_ok_and(|b| b) {
                        Ok(())
                    } else {
                        Err(e)
                    }
                }
            }
            .with_context(|| format!("When persisting temp file to `{}`", file_path.display()))?;
        }

        Self::config()
            .as_ref()
            .write_to_toml_in_directory(output_dir)
            .with_context(|| {
                format!(
                    "Failed to write pyrefly config at {:?}",
                    output_dir.display()
                )
            })?;
        Ok(())
    }
    fn get_path_name(&self) -> String;
    fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle(files: &[(&str, &str)]) -> Bundle {
        Bundle::new(files.iter().map(|(module_path, path)| BundleFile {
            import_path: PathBuf::from(module_path),
            storage_path: PathBuf::from(path),
            contents: path.to_string(),
        }))
        .unwrap()
    }

    fn assert_found(bundle: &Bundle, module: &str, path: &str) {
        let path = PathBuf::from(path);
        assert_eq!(bundle.find(ModuleName::from_str(module)), Some(&path));
        assert_eq!(
            bundle.load(&path).as_deref().map(String::as_str),
            path.to_str()
        );
    }

    #[test]
    fn test_bundled_package_initializer_precedes_module_file() {
        let module_then_package = bundle(&[
            ("foo.pyi", "foo.pyi"),
            ("foo/__init__.pyi", "foo/__init__.pyi"),
        ]);
        let package_then_module = bundle(&[
            ("foo/__init__.pyi", "foo/__init__.pyi"),
            ("foo.pyi", "foo.pyi"),
        ]);
        assert_found(&module_then_package, "foo", "foo/__init__.pyi");
        assert_found(&package_then_module, "foo", "foo/__init__.pyi");
    }

    #[test]
    fn test_bundled_module_parent_blocks_child_module() {
        let bundle = bundle(&[("foo/bar.pyi", "foo/bar.pyi"), ("foo.pyi", "foo.pyi")]);
        assert_found(&bundle, "foo", "foo.pyi");
        assert!(bundle.find(ModuleName::from_str("foo.bar")).is_none());
    }

    #[test]
    fn test_bundled_namespace_package_merges_files() {
        let bundle = bundle(&[
            ("ns/left.pyi", "first/ns/left.pyi"),
            ("ns/right.pyi", "second/ns/right.pyi"),
        ]);
        assert_found(&bundle, "ns.left", "first/ns/left.pyi");
        assert_found(&bundle, "ns.right", "second/ns/right.pyi");
    }

    #[test]
    fn test_bundled_equal_kind_candidates_follow_file_order() {
        let bundle = bundle(&[
            ("duplicate.pyi", "first/duplicate.pyi"),
            ("duplicate.pyi", "second/duplicate.pyi"),
        ]);
        assert_found(&bundle, "duplicate", "first/duplicate.pyi");
    }

    #[test]
    fn test_bundled_duplicate_storage_path_keeps_stable_indices() {
        let bundle = Bundle::new([
            BundleFile {
                import_path: PathBuf::from("first.pyi"),
                storage_path: PathBuf::from("shared.pyi"),
                contents: "first".to_owned(),
            },
            BundleFile {
                import_path: PathBuf::from("second.pyi"),
                storage_path: PathBuf::from("shared.pyi"),
                contents: "second".to_owned(),
            },
        ])
        .unwrap();
        let shared = PathBuf::from("shared.pyi");
        assert_eq!(bundle.find(ModuleName::from_str("first")), Some(&shared));
        assert_eq!(bundle.find(ModuleName::from_str("second")), Some(&shared));
        assert_eq!(
            bundle
                .load(Path::new("shared.pyi"))
                .as_deref()
                .map(String::as_str),
            Some("second")
        );
    }

    #[test]
    fn test_bundled_regular_package_includes_all_children_in_the_root() {
        let bundle = bundle(&[
            ("pkg/__init__.pyi", "first/pkg/__init__.pyi"),
            ("pkg/child.pyi", "second/pkg/child.pyi"),
        ]);
        assert_found(&bundle, "pkg", "first/pkg/__init__.pyi");
        assert_found(&bundle, "pkg.child", "second/pkg/child.pyi");
    }

    #[test]
    fn test_bundled_regular_package_preserves_overlaid_namespace_children() {
        let bundle = bundle(&[
            ("ns/left.pyi", "first/ns/left.pyi"),
            ("ns/__init__.pyi", "second/ns/__init__.pyi"),
            ("ns/right.pyi", "second/ns/right.pyi"),
        ]);
        assert_found(&bundle, "ns.left", "first/ns/left.pyi");
        assert_found(&bundle, "ns.right", "second/ns/right.pyi");
    }
}
