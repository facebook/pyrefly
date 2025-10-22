/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::anyhow;
use pyrefly_bundled::bundled_typeshed;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::module::bundled::bundled::Stub;
use crate::module::bundled::bundled::find_bundled_stub_module_path;
use crate::module::bundled::bundled::get_config_file;
use crate::module::bundled::bundled::get_materialized_path_on_disk;
use crate::module::bundled::bundled::get_modules;
use crate::module::bundled::bundled::load_stubs_from_path;
#[cfg(test)]
use crate::module::bundled::bundled::write_stub_files;

#[derive(Debug, Clone)]
pub struct BundledTypeshedStdlib {
    pub find: SmallMap<ModuleName, PathBuf>,
    pub load: SmallMap<PathBuf, Arc<String>>,
}

impl BundledTypeshedStdlib {
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
        find_bundled_stub_module_path(Stub::BundledTypeshedStdlib(self.clone()), module)
    }

    pub fn load(&self, path: &Path) -> Option<Arc<String>> {
        load_stubs_from_path(Stub::BundledTypeshedStdlib(self.clone()), path)
    }

    pub fn modules(&self) -> impl Iterator<Item = ModuleName> {
        let stub = Stub::BundledTypeshedStdlib(self.clone());
        get_modules(&stub).collect::<Vec<_>>().into_iter()
    }

    pub fn config() -> ArcId<ConfigFile> {
        get_config_file()
    }

    /// Obtain a materialized path for bundled typeshed, writing it all to disk the first time.
    /// Note: this path is not the source of truth, it simply exists to display typeshed contents
    /// for informative purposes.
    pub fn materialized_path_on_disk(&self) -> anyhow::Result<PathBuf> {
        get_materialized_path_on_disk(
            Stub::BundledTypeshedStdlib(self.clone()),
            "pyrefly_bundled_typeshed",
        )
    }

    #[cfg(test)]
    fn write(&self, temp_dir: &Path) -> anyhow::Result<()> {
        write_stub_files(Stub::BundledTypeshedStdlib(self.clone()), temp_dir)
    }
}

static BUNDLED_TYPESHED: LazyLock<anyhow::Result<BundledTypeshedStdlib>> =
    LazyLock::new(BundledTypeshedStdlib::new);

pub fn typeshed() -> anyhow::Result<&'static BundledTypeshedStdlib> {
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
