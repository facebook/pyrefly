/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::anyhow;
use dupe::Dupe;
use pyrefly_bundled::bundled_typeshed;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::module::bundled::BundledStub;
use crate::module::bundled::create_bundled_stub_config;

#[derive(Debug, Clone)]
pub struct BundledTypeshedStdlib {
    pub find: SmallMap<ModuleName, PathBuf>,
    pub load: SmallMap<PathBuf, Arc<String>>,
}

impl BundledStub for BundledTypeshedStdlib {
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

    fn find(&self, module: ModuleName) -> Option<ModulePath> {
        self.find
            .get(&module)
            .map(|path| ModulePath::bundled_typeshed(path.clone()))
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
            "pyrefly_bundled_typeshed_{}",
            faster_hex::hex_string(&pyrefly_bundled::BUNDLED_TYPESHED_DIGEST[0..6])
        )
    }

    fn config() -> ArcId<ConfigFile> {
        static CONFIG: LazyLock<ArcId<ConfigFile>> = LazyLock::new(|| {
            let search_paths = match stdlib_search_path() {
                Some(path) => vec![path],
                None => Vec::new(),
            };
            let error_overrides = HashMap::from([
                // The stdlib is full of deliberately incorrect overrides, so ignore them
                (ErrorKind::BadOverride, Severity::Ignore),
                (ErrorKind::BadParamNameOverride, Severity::Ignore),
            ]);
            let config_file = create_bundled_stub_config(Some(search_paths), Some(error_overrides));
            ArcId::new(config_file)
        });
        CONFIG.dupe()
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
