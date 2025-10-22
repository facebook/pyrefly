/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::anyhow;
use pyrefly_bundled::bundled_third_party_stubs;
use pyrefly_config::config::ConfigFile;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::module::bundled::bundled::Stub;
use crate::module::bundled::bundled::find_bundled_stub_module_path;
use crate::module::bundled::bundled::get_config_file;
use crate::module::bundled::bundled::get_materialized_path_on_disk;
use crate::module::bundled::bundled::get_modules;
use crate::module::bundled::bundled::load_stubs_from_path;
#[cfg(test)]
use crate::module::bundled::bundled::write_stub_files;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BundledTypeshedThirdParty {
    pub find: SmallMap<ModuleName, PathBuf>,
    pub load: SmallMap<PathBuf, Arc<String>>,
}

impl BundledTypeshedThirdParty {
    #[allow(dead_code)]
    fn new() -> anyhow::Result<Self> {
        let contents = bundled_third_party_stubs()?;
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

    #[allow(dead_code)]
    pub fn find(&self, module: ModuleName) -> Option<ModulePath> {
        find_bundled_stub_module_path(Stub::BundledTypeshedThirdParty(self.clone()), module)
    }

    pub fn load(&self, path: &Path) -> Option<Arc<String>> {
        load_stubs_from_path(Stub::BundledTypeshedThirdParty(self.clone()), path)
    }

    #[allow(dead_code)]
    pub fn modules(&self) -> impl Iterator<Item = ModuleName> {
        let stub = Stub::BundledTypeshedThirdParty(self.clone());
        get_modules(&stub).collect::<Vec<_>>().into_iter()
    }

    pub fn materialized_path_on_disk(&self) -> anyhow::Result<PathBuf> {
        get_materialized_path_on_disk(
            Stub::BundledTypeshedThirdParty(self.clone()),
            "pyrefly_bundled_typeshed_third_party",
        )
    }

    #[cfg(test)]
    fn write(&self, temp_dir: &Path) -> anyhow::Result<()> {
        write_stub_files(Stub::BundledTypeshedThirdParty(self.clone()), temp_dir)
    }

    #[allow(dead_code)]
    pub fn config() -> ArcId<ConfigFile> {
        get_config_file()
    }
}

static BUNDLED_TYPESHED_THIRD_PARTY: LazyLock<anyhow::Result<BundledTypeshedThirdParty>> =
    LazyLock::new(BundledTypeshedThirdParty::new);

pub fn typeshed_third_party() -> anyhow::Result<&'static BundledTypeshedThirdParty> {
    match &*BUNDLED_TYPESHED_THIRD_PARTY {
        Ok(typeshed) => Ok(typeshed),
        Err(error) => Err(anyhow!("{error:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typeshed_materialize() {
        let typeshed = typeshed_third_party().unwrap();
        let path = typeshed.materialized_path_on_disk().unwrap();
        // Do it twice, to check that works.
        typeshed.materialized_path_on_disk().unwrap();
        typeshed.write(&path).unwrap();
    }
}
