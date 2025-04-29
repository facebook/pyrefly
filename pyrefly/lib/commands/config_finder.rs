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

use dupe::Dupe;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::module::module_path::ModulePathDetails;
use crate::util::arc_id::ArcId;
use crate::util::lock::Mutex;

/// Create a standard `ConfigFinder`. The `configure` function is expected to set any additional options,
/// then call `configure` and `validate`.
/// The `path` to `configure` is a directory, either to the python file or the config file.
#[allow(clippy::field_reassign_with_default)] // ConfigFile default is dubious
pub fn standard_config_finder(
    configure: Arc<dyn Fn(Option<&Path>, ConfigFile) -> ConfigFile + Send + Sync>,
) -> ConfigFinder {
    let configure2 = configure.dupe();
    let configure3 = configure.dupe();

    // A cache where path `p` maps to config file with `search_path = [p]`. If we can find the root.
    let cache_one: Mutex<SmallMap<PathBuf, ArcId<ConfigFile>>> = Mutex::new(SmallMap::new());
    // A cache where path `p` maps to config file with `search_path = [p, p/.., p/../.., ...]`.
    let cache_parents: Mutex<SmallMap<PathBuf, ArcId<ConfigFile>>> = Mutex::new(SmallMap::new());

    let empty = LazyLock::new(move || ArcId::new(configure3(None, ConfigFile::default())));

    ConfigFinder::new(
        Box::new(move |file| {
            let config = ConfigFile::from_file(file, false)?;
            Ok(ArcId::new(configure(file.parent(), config)))
        }),
        Box::new(move |name, path| match path.root_of(name) {
            Some(path) => cache_one
                .lock()
                .entry(path.clone())
                .or_insert_with(|| {
                    let mut config = ConfigFile::default();
                    config.search_path = vec![path.clone()];
                    ArcId::new(configure2(path.parent(), config))
                })
                .dupe(),

            None => {
                let path = match path.details() {
                    ModulePathDetails::FileSystem(x) | ModulePathDetails::Memory(x) => {
                        Some(x.as_path())
                    }
                    ModulePathDetails::Namespace(x) => x.parent(),
                    ModulePathDetails::BundledTypeshed(_) => None,
                };
                match path {
                    None => empty.dupe(),
                    Some(path) => cache_parents
                        .lock()
                        .entry(path.to_owned())
                        .or_insert_with(|| {
                            let mut config = ConfigFile::default();
                            config.search_path =
                                path.ancestors().skip(1).map(|x| x.to_owned()).collect();
                            ArcId::new(configure2(path.parent(), config))
                        })
                        .dupe(),
                }
            }
        }),
    )
}
