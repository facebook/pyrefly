/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
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
use tempfile::NamedTempFile;

pub fn set_readonly(path: &Path, value: bool) -> anyhow::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(value);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

/// Creates a base config file for bundled stubs with common settings.
///
/// This helper function encapsulates the common configuration logic shared across
/// different bundled stub types (typeshed stdlib, typeshed third-party, and third-party stubs).
///
/// # Parameters
/// * `search_paths` - Optional search paths to add to the config
/// * `error_overrides` - Optional error kind overrides (e.g., to ignore certain errors)
pub fn create_bundled_stub_config(
    search_paths: Option<Vec<PathBuf>>,
    error_overrides: Option<HashMap<ErrorKind, Severity>>,
) -> ConfigFile {
    let mut config_file = ConfigFile::default();
    config_file.python_environment.site_package_path = Some(Vec::new());

    if let Some(paths) = search_paths {
        config_file.search_path_from_file = paths;
    }

    if let Some(overrides) = error_overrides {
        config_file.root.errors = Some(ErrorDisplayConfig::new(overrides));
    }

    config_file.root.disable_type_errors_in_ide = Some(true);
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
    /// Obtain a materialized path for bundled typeshed, writing it all to disk the first time.
    /// Note: this path is not the source of truth, it simply exists to display typeshed contents
    /// for informative purposes.
    fn materialized_path_on_disk(&self) -> anyhow::Result<PathBuf> {
        static WRITTEN_TO_DISK: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

        let temp_dir = env::temp_dir().join(self.get_path_name());

        let mut written = WRITTEN_TO_DISK.lock();
        if !*written {
            self.write(&temp_dir)?;
            *written = true;
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
