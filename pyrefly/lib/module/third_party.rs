/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::anyhow;
use dupe::Dupe;
use pyrefly_bundled::bundled_third_party;
use pyrefly_config::config::ConfigFile;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::module::bundled::BundledStub;
use crate::module::bundled::create_bundled_stub_config;

#[derive(Debug, Clone)]
pub struct BundledThirdParty {
    pub find: SmallMap<ModuleName, PathBuf>,
    pub load: SmallMap<PathBuf, Arc<String>>,
}

/// Unlike typeshed stubs, other third-party stubs have -stubs suffixes
/// which are just the name of the package, e.g. `conans-stubs` for `conans`.
/// Strips the `-stubs` suffix from the first component of a path
/// so that we can look it up easier later.
/// e.g., "conans-stubs/errors.pyi" -> "conans/errors.pyi"
fn strip_stubs_suffix_from_path(path: &Path) -> PathBuf {
    let mut components = path.components().peekable();
    if let Some(first) = components.next()
        && let Component::Normal(os_str) = first
        && let Some(s) = os_str.to_str()
        && let Some(stripped) = s.strip_suffix("-stubs")
    {
        let mut new_path = PathBuf::from(stripped);
        for component in components {
            new_path.push(component);
        }
        return new_path;
    }
    path.to_path_buf()
}

impl BundledStub for BundledThirdParty {
    fn new() -> anyhow::Result<Self> {
        let contents = bundled_third_party()?;
        let mut res = Self {
            find: SmallMap::new(),
            load: SmallMap::new(),
        };
        for (relative_path, contents) in contents {
            let adjusted_path = strip_stubs_suffix_from_path(&relative_path);
            let module_name = ModuleName::from_relative_path(&adjusted_path)?;
            res.find.insert(module_name, relative_path.clone());
            res.load.insert(relative_path, Arc::new(contents));
        }
        Ok(res)
    }

    fn find(&self, module: ModuleName) -> Option<ModulePath> {
        self.find
            .get(&module)
            .map(|path| ModulePath::bundled_third_party(path.clone()))
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
            faster_hex::hex_string(&pyrefly_bundled::BUNDLED_THIRD_PARTY_DIGEST[0..6])
        )
    }

    fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)> {
        self.load.iter()
    }
}

static BUNDLED_THIRD_PARTY: LazyLock<anyhow::Result<BundledThirdParty>> =
    LazyLock::new(BundledThirdParty::new);

pub fn get_bundled_third_party() -> anyhow::Result<&'static BundledThirdParty> {
    match &*BUNDLED_THIRD_PARTY {
        Ok(stub) => Ok(stub),
        Err(error) => Err(anyhow!("{error:#}")),
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_python::module_path::ModulePathDetails;

    use super::*;

    #[test]
    fn test_bundled_third_party_materialize() {
        let stub = get_bundled_third_party().unwrap();
        let path = stub.materialized_path_on_disk().unwrap();
        // Do it twice, to check that works.
        stub.materialized_path_on_disk().unwrap();
        stub.write(&path).unwrap();
    }

    #[test]
    fn test_bundled_third_party_find_returns_correct_path_type() {
        let stub = get_bundled_third_party().unwrap();
        // If there are any modules loaded, verify they return the correct path type
        for module in stub.modules().take(5) {
            if let Some(path) = stub.find(module) {
                assert!(
                    matches!(path.details(), ModulePathDetails::BundledThirdParty(_)),
                    "Expected BundledThirdParty path type for module {}",
                    module
                );
            }
        }
    }

    #[test]
    fn test_bundled_third_party_load_works() {
        let stub = get_bundled_third_party().unwrap();
        // Verify that loaded content can be retrieved
        for (path, _) in stub.load_map().take(5) {
            let content = stub.load(path);
            assert!(
                content.is_some(),
                "Should be able to load content for path {:?}",
                path
            );
        }
    }
}
