/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context as _;
use anyhow::anyhow;
use dupe::Dupe;
use dupe::OptionDupedExt;
use pyrefly_bundled::bundled_typeshed;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lock::Mutex;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::config::error::ErrorDisplayConfig;
use crate::config::error_kind::ErrorKind;
use crate::config::error_kind::Severity;

#[derive(Debug, Clone)]
pub struct BundledTypeshed {
    find: SmallMap<ModuleName, PathBuf>,
    load: SmallMap<PathBuf, Arc<String>>,
}

fn set_readonly(path: &Path, value: bool) -> anyhow::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(value);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

impl BundledTypeshed {
    fn new() -> anyhow::Result<Self> {
        let contents = bundled_typeshed()?;
        let mut res = Self {
            find: SmallMap::new(),
            load: SmallMap::new(),
        };
        for (relative_path, contents) in contents {
            let module_name = ModuleName::from_relative_path(&relative_path)?;
            res.find.insert(module_name, relative_path.clone());
            res.load.insert(relative_path, Arc::new(contents));
        }
        Ok(res)
    }

    pub fn find(&self, module: ModuleName) -> Option<ModulePath> {
        self.find
            .get(&module)
            .map(|path| ModulePath::bundled_typeshed(path.clone()))
    }

    pub fn load(&self, path: &Path) -> Option<Arc<String>> {
        self.load.get(path).duped()
    }

    pub fn modules(&self) -> impl Iterator<Item = ModuleName> {
        self.find.keys().copied()
    }

    pub fn config() -> ArcId<ConfigFile> {
        static CONFIG: LazyLock<ArcId<ConfigFile>> = LazyLock::new(|| {
            let mut config_file = ConfigFile::default();
            config_file.python_environment.site_package_path = Some(Vec::new());
            config_file.search_path_from_file = match stdlib_search_path() {
                Some(path) => vec![path],
                None => Vec::new(),
            };
            config_file.root.errors = Some(ErrorDisplayConfig::new(HashMap::from([
                // The stdlib is full of deliberately incorrect overrides, so ignore them
                (ErrorKind::BadOverride, Severity::Ignore),
                (ErrorKind::BadParamNameOverride, Severity::Ignore),
            ])));
            config_file.root.disable_type_errors_in_ide = Some(true);
            config_file.configure();
            ArcId::new(config_file)
        });
        CONFIG.dupe()
    }

    /// Obtain a materialized path for bundled typeshed, writing it all to disk the first time.
    /// Note: this path is not the source of truth, it simply exists to display typeshed contents
    /// for informative purposes.
    pub fn materialized_path_on_disk(&self) -> anyhow::Result<PathBuf> {
        static WRITTEN_TO_DISK: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

        let temp_dir = env::temp_dir().join("pyrefly_bundled_typeshed");

        let mut written = WRITTEN_TO_DISK.lock();
        if !*written {
            self.write(&temp_dir)?;
            *written = true;
        }
        Ok(temp_dir)
    }

    fn write(&self, temp_dir: &Path) -> anyhow::Result<()> {
        fs_anyhow::create_dir_all(temp_dir)?;

        for (relative_path, contents) in &self.load {
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

        BundledTypeshed::config()
            .as_ref()
            .write_to_toml_in_directory(temp_dir)
            .with_context(|| {
                format!("Failed to write pyrefly config at {:?}", temp_dir.display())
            })?;
        Ok(())
    }
}

static BUNDLED_TYPESHED: LazyLock<anyhow::Result<BundledTypeshed>> =
    LazyLock::new(BundledTypeshed::new);

pub fn typeshed() -> anyhow::Result<&'static BundledTypeshed> {
    match &*BUNDLED_TYPESHED {
        Ok(typeshed) => Ok(typeshed),
        Err(error) => Err(anyhow!("{error:#}")),
    }
}

/// This is a workaround for bundled typeshed incorrectly taking precedence over
/// stubs manually put at the beginning of the search path.
/// See https://typing.python.org/en/latest/spec/distributing.html#import-resolution-ordering.
/// Note that you need to set both the PYREFLY_STDLIB_SEARCH_PATH environment variable AND
/// --search-path/SEARCH_PATH for this workaround to be effective.
pub fn stdlib_search_path() -> Option<PathBuf> {
    env::var_os("PYREFLY_STDLIB_SEARCH_PATH").map(|path| Path::new(&path).to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typeshed_materialize() {
        let typeshed = typeshed().unwrap();
        let path = typeshed.materialized_path_on_disk().unwrap();
        // Do it twice, to check that works.
        typeshed.materialized_path_on_disk().unwrap();
        typeshed.write(&path).unwrap();
    }
}
