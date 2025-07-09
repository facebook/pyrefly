/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::error::collector::CollectedErrors;
use crate::error::expectation::Expectation;
use crate::state::load::Load;

/// The errors from a collection of modules.
#[derive(Debug)]
pub struct Errors {
    // Sorted by module name and path (so deterministic display order)
    loads: Vec<(Arc<Load>, ArcId<ConfigFile>)>,
}

impl Errors {
    pub fn new(mut loads: Vec<(Arc<Load>, ArcId<ConfigFile>)>) -> Self {
        loads.sort_by_key(|x| (x.0.module_info.name(), x.0.module_info.path().dupe()));
        Self { loads }
    }

    pub fn collect_errors(&self) -> CollectedErrors {
        let mut errors = CollectedErrors::default();
        for (load, config) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            load.errors.collect_into(&error_config, &mut errors);
        }
        errors
    }

    pub fn collect_ignores(&self) -> SmallMap<&ModulePath, &Ignore> {
        let mut ignore_collection: SmallMap<&ModulePath, &Ignore> = SmallMap::new();
        for (load, _) in &self.loads {
            let module_path = load.module_info.path();
            let ignores = load.module_info.ignore();
            ignore_collection.insert(module_path, ignores);
        }
        ignore_collection
    }

    pub fn check_against_expectations(&self) -> anyhow::Result<()> {
        for (load, config) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            Expectation::parse(load.module_info.dupe(), load.module_info.contents())
                .check(&load.errors.collect(&error_config).shown)?;
        }
        Ok(())
    }
}
