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
use pyrefly_bundled::bundled_third_party_stubs;
use pyrefly_config::config::ConfigFile;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::module::bundled::BundledStub;
use crate::module::bundled::create_bundled_stub_config;

#[derive(Debug, Clone)]
pub struct BundledTypeshedThirdParty {
    /// Maps module name to (stub path, typeshed package name).
    /// The package name is needed for correct `types-{package}` recommendations,
    /// since the module name often differs from the package name
    /// (e.g., module `dateutil` comes from package `python-dateutil`).
    pub find: SmallMap<ModuleName, (PathBuf, ModuleName)>,
    pub load: SmallMap<PathBuf, Arc<String>>,
}

impl BundledStub for BundledTypeshedThirdParty {
    fn new() -> anyhow::Result<Self> {
        let (contents, path_to_package) = bundled_third_party_stubs()?;
        let mut res = Self {
            find: SmallMap::new(),
            load: SmallMap::new(),
        };
        for (relative_path, contents) in contents {
            let module_name = ModuleName::from_relative_path(&relative_path)?;
            let package_name = path_to_package
                .get(&relative_path)
                .map(|s| ModuleName::from_str(s))
                .unwrap_or_else(|| ModuleName::from_name(&module_name.first_component()));
            res.find
                .insert(module_name, (relative_path.clone(), package_name));
            res.load.insert(relative_path, Arc::new(contents));
        }
        Ok(res)
    }

    fn find(&self, module: ModuleName) -> Option<ModulePath> {
        self.find
            .get(&module)
            .map(|(path, _)| ModulePath::bundled_typeshed_third_party(path.clone()))
    }

    fn load(&self, path: &Path) -> Option<Arc<String>> {
        self.load.get(path).cloned()
    }

    fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)> {
        self.load.iter()
    }

    fn modules(&self) -> impl Iterator<Item = ModuleName> {
        self.find.keys().copied()
    }

    fn get_path_name(&self) -> String {
        format!(
            "pyrefly_bundled_typeshed_third_party_{}",
            faster_hex::hex_string(&pyrefly_bundled::BUNDLED_TYPESHED_DIGEST[0..6])
        )
    }

    fn config() -> ArcId<ConfigFile> {
        static CONFIG: LazyLock<ArcId<ConfigFile>> = LazyLock::new(|| {
            let config_file = create_bundled_stub_config(None, None, None);
            ArcId::new(config_file)
        });
        CONFIG.dupe()
    }
}

impl BundledTypeshedThirdParty {
    pub fn package_name(&self, module: ModuleName) -> Option<&ModuleName> {
        self.find.get(&module).map(|(_, pkg)| pkg)
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

    /// Regression test: namespace packages like `google/` are shared by multiple
    /// typeshed packages (protobuf, google-cloud-ndb). Each module must map to
    /// its own package, not whichever was iterated first.
    #[test]
    fn test_namespace_packages_have_distinct_package_names() {
        let typeshed = typeshed_third_party().unwrap();
        let protobuf_pkg = typeshed
            .package_name(ModuleName::from_str("google.protobuf"))
            .expect("google.protobuf should have a package name");
        let ndb_pkg = typeshed
            .package_name(ModuleName::from_str("google.cloud.ndb"))
            .expect("google.cloud.ndb should have a package name");
        assert_eq!(protobuf_pkg.as_str(), "protobuf");
        assert_eq!(ndb_pkg.as_str(), "google-cloud-ndb");
    }
}
