/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use dashmap::DashMap;
use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use serde::Serialize;

use crate::report::pysa::step_logger::StepLogger;

/// Represents a unique identifier for a module
#[derive(
    Debug, Clone, Copy, Dupe, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize
)]
pub struct ModuleId(u32);

impl ModuleId {
    pub fn to_int(self) -> u32 {
        self.0
    }
}

/// Represents what makes a module unique
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleKey {
    name: ModuleName,
    path: ModulePath,
}

impl ModuleKey {
    pub fn from_handle(handle: &Handle) -> ModuleKey {
        ModuleKey {
            name: handle.module(),
            path: handle.path().dupe(),
        }
    }

    pub fn from_module(module: &Module) -> ModuleKey {
        ModuleKey {
            name: module.name(),
            path: module.path().dupe(),
        }
    }
}

/// Thread-safe map from `ModuleKey` to `ModuleId`.
///
/// Project handles are pre-assigned deterministic IDs in sorted order.
/// Dependency modules discovered during type checking get IDs assigned
/// lazily on first access via `get_or_insert`.
pub struct ModuleIds {
    map: DashMap<ModuleKey, ModuleId>,
    next_id: AtomicU32,
}

impl ModuleIds {
    /// Pre-assign deterministic IDs for project handles in sorted order.
    /// Dependency modules discovered later get IDs via `get_or_insert`.
    pub fn new(handles: &[Handle]) -> ModuleIds {
        let step = StepLogger::start(
            "Building unique module ids",
            format!("Built unique module ids for {} modules", handles.len()).as_str(),
        );

        let mut modules = handles
            .iter()
            .map(ModuleKey::from_handle)
            .collect::<Vec<_>>();
        modules.sort();

        let map = DashMap::new();
        let mut current_id = 1u32;
        for module in modules {
            assert!(
                map.insert(module, ModuleId(current_id)).is_none(),
                "Found multiple handles with the same module name and path"
            );
            current_id += 1;
        }

        step.finish();
        ModuleIds {
            map,
            next_id: AtomicU32::new(current_id),
        }
    }

    /// Get or lazily assign a `ModuleId` for the given key.
    fn get_or_insert(&self, key: ModuleKey) -> ModuleId {
        *self
            .map
            .entry(key)
            .or_insert_with(|| {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                ModuleId(id)
            })
            .value()
    }

    pub fn get_from_handle(&self, handle: &Handle) -> ModuleId {
        self.get_or_insert(ModuleKey::from_handle(handle))
    }

    pub fn get_from_module(&self, module: &Module) -> ModuleId {
        self.get_or_insert(ModuleKey::from_module(module))
    }
}
