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
use dupe::Dupe;
use pyrefly_config::config::ConfigFile;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::module::bundled::BundledStub;
use crate::module::bundled::create_bundled_stub_config;

#[allow(dead_code)]
pub struct BundledThirdParty {
    pub find: SmallMap<ModuleName, PathBuf>,
    pub load: SmallMap<PathBuf, Arc<String>>,
}

impl BundledStub for BundledThirdParty {
    fn new() -> anyhow::Result<Self> {
        // todo:(jvansch) This is just a mock, actual logic will be implemented later
        Ok(Self {
            find: SmallMap::new(),
            load: SmallMap::new(),
        })
    }

    fn find(&self, _module: ModuleName) -> Option<ModulePath> {
        // todo:(jvansch) This is just a mock, actual logic will be implemented later
        None
    }

    fn load(&self, path: &Path) -> Option<Arc<String>> {
        self.load.get(path).cloned()
    }

    fn modules(&self) -> impl Iterator<Item = ModuleName> {
        self.find.keys().copied()
    }

    fn config() -> ArcId<ConfigFile> {
        static CONFIG: LazyLock<ArcId<ConfigFile>> = LazyLock::new(|| {
            let config_file = create_bundled_stub_config(None, None);
            ArcId::new(config_file)
        });
        CONFIG.dupe()
    }

    fn get_path_name(&self) -> String {
        format!(
            "pyrefly_bundled_third_party_{}",
            faster_hex::hex_string(&pyrefly_bundled::BUNDLED_TYPESHED_DIGEST[0..6])
        )
    }

    fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)> {
        self.load.iter()
    }
}

#[allow(dead_code)]
static BUNDLED_THIRD_PARTY: LazyLock<anyhow::Result<BundledThirdParty>> =
    LazyLock::new(BundledThirdParty::new);

#[allow(dead_code)]
pub fn bundled_third_party() -> anyhow::Result<&'static BundledThirdParty> {
    match &*BUNDLED_THIRD_PARTY {
        Ok(stub) => Ok(stub),
        Err(error) => Err(anyhow!("{error:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typeshed_materialize() {
        let stub = bundled_third_party().unwrap();
        let path = stub.materialized_path_on_disk().unwrap();
        // Do it twice, to check that works.
        stub.materialized_path_on_disk().unwrap();
        stub.write(&path).unwrap();
    }
}
