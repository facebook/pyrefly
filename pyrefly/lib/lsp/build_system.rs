/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::sync::Arc;
use std::sync::LazyLock;

use dupe::Dupe as _;
use pyrefly_build::handle::Handle;
use pyrefly_config::config::ConfigFile;
use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lock::Mutex;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::lsp::queue::HeavyTaskQueue;
use crate::lsp::queue::LspEvent;
use crate::lsp::queue::LspQueue;
use crate::state::state::State;

pub fn should_requery_build_system(events: &CategorizedEvents) -> bool {
    static CONFIG_NAMES: LazyLock<SmallSet<OsString>> = LazyLock::new(|| {
        ConfigFile::CONFIG_FILE_NAMES
            .iter()
            .chain(ConfigFile::ADDITIONAL_ROOT_FILE_NAMES.iter())
            .map(OsString::from)
            .collect()
    });
    static PYTHON_SUFFIXES: LazyLock<SmallSet<OsString>> = LazyLock::new(|| {
        PYTHON_EXTENSIONS
            .iter()
            .chain(COMPILED_FILE_SUFFIXES.iter())
            .map(OsString::from)
            .collect()
    });

    events.iter().any(|f| {
        !(f.file_name().is_some_and(|n| CONFIG_NAMES.contains(n))
            || f.extension().is_some_and(|e| PYTHON_SUFFIXES.contains(e)))
    })
}

/// Attempts to requery any open sourced_dbs for open files, and if there are changes,
/// invalidate find and perform a recheck.
pub fn queue_source_db_rebuild_and_recheck(
    state: &State,
    invalidated_configs: Arc<Mutex<SmallSet<ArcId<ConfigFile>>>>,
    sourcedb_queue: HeavyTaskQueue,
    lsp_queue: LspQueue,
    handles: &[Handle],
) {
    let mut configs_to_paths: SmallMap<ArcId<ConfigFile>, SmallSet<ModulePath>> = SmallMap::new();
    let config_finder = state.config_finder();
    for handle in handles {
        let config = config_finder.python_file(handle.module(), handle.path());
        configs_to_paths
            .entry(config)
            .or_default()
            .insert(handle.path().dupe());
    }
    sourcedb_queue.queue_task(Box::new(move || {
        let new_invalidated_configs: SmallSet<ArcId<ConfigFile>> = configs_to_paths
            .into_iter()
            .filter(|(c, files)| match c.requery_source_db(files) {
                Ok(reloaded) => reloaded,
                Err(error) => {
                    eprintln!("Error reloading source database for config: {error}");
                    false
                }
            })
            .map(|(c, _)| c)
            .collect();
        if !new_invalidated_configs.is_empty() {
            let mut lock = invalidated_configs.lock();
            for c in new_invalidated_configs {
                lock.insert(c);
            }
            let _ = lsp_queue.send(LspEvent::InvalidateConfigFind);
        }
    }));
}
