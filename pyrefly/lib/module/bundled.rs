/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context as _;
use pyrefly_config::config::ConfigFile;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lock::Mutex;

pub fn set_readonly(path: &Path, value: bool) -> anyhow::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(value);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

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
    fn write(&self, temp_dir: &Path) -> anyhow::Result<()> {
        fs_anyhow::create_dir_all(temp_dir)?;

        for (relative_path, contents) in self.load_map() {
            let mut file_path = temp_dir.to_owned();
            file_path.push(relative_path);

            if let Some(parent) = file_path.parent() {
                fs_anyhow::create_dir_all(parent)?;
            }

            // Write the file and set it as read-only in a single logical operation
            let _ = set_readonly(&file_path, false); // Might fail (e.g. file doesn't exist)
            fs::write(&file_path, contents.as_bytes())
                .with_context(|| format!("When writing file `{}`", file_path.display()))?;

            // We try and make the files read-only, since editing them in the IDE won't update.
            let _ = set_readonly(&file_path, true); // If this fails, not a big deal
        }

        Self::config()
            .as_ref()
            .write_to_toml_in_directory(temp_dir)
            .with_context(|| {
                format!("Failed to write pyrefly config at {:?}", temp_dir.display())
            })?;
        Ok(())
    }
    fn get_path_name(&self) -> &'static str;
    fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)>;
}
