/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use dashmap::DashMap;
use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::sys_info::SysInfo;
use serde::Serialize;

use crate::module::bundled::BundledStub;
use crate::module::typeshed::typeshed;
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

/// Thread-safe map from `Handle` to `ModuleId`.
///
/// Typeshed modules and project handles are pre-assigned deterministic IDs
/// in sorted order. Dependency modules discovered during type checking get
/// IDs assigned lazily on first access via `get_or_insert`.
pub struct ModuleIds {
    map: DashMap<Handle, ModuleId>,
    next_id: AtomicU32,
}

impl ModuleIds {
    /// Pre-assign deterministic IDs for typeshed modules and project handles
    /// in sorted order. Typeshed modules are assigned first (for each unique
    /// SysInfo), then project handles. Dependency modules discovered later
    /// get IDs via `get_or_insert`.
    pub fn new(handles: &[Handle]) -> ModuleIds {
        let step = StepLogger::start("Building unique module ids", "Built unique module ids");

        // Collect unique SysInfo values from project handles.
        let sys_infos: HashSet<SysInfo> = handles.iter().map(|handle| *handle.sys_info()).collect();

        // Build sorted typeshed handles for each SysInfo variant.
        let typeshed = typeshed().expect("Failed to load typeshed");
        let mut typeshed_handles: Vec<Handle> = sys_infos
            .iter()
            .flat_map(|sys_info| {
                typeshed.modules().filter_map(move |module_name| {
                    let path = typeshed.find(module_name)?;
                    Some(Handle::new(module_name, path, *sys_info))
                })
            })
            .collect();
        typeshed_handles.sort();

        // Build sorted project handles.
        let mut sorted_handles = handles.to_vec();
        sorted_handles.sort();

        let map = DashMap::new();
        let mut current_id = 1u32;

        // Assign typeshed IDs first.
        for handle in typeshed_handles {
            assert!(
                map.insert(handle, ModuleId(current_id)).is_none(),
                "Found multiple typeshed modules with the same module name, path, and sys_info"
            );
            current_id += 1;
        }

        // Round up to the next multiple of 1000 so project IDs start at a
        // clean boundary (e.g., 1000, 2000, …). This keeps project IDs stable
        // when the number of typeshed modules changes slightly.
        current_id = current_id.div_ceil(1000) * 1000;

        // Assign project handle IDs, skipping any already assigned as typeshed.
        for handle in sorted_handles {
            if map.contains_key(&handle) {
                continue;
            }
            assert!(
                map.insert(handle, ModuleId(current_id)).is_none(),
                "Found multiple handles with the same module name, path, and sys_info"
            );
            current_id += 1;
        }

        step.finish();
        ModuleIds {
            map,
            next_id: AtomicU32::new(current_id),
        }
    }

    /// Get or lazily assign a `ModuleId` for the given handle.
    fn get_or_insert(&self, handle: Handle) -> ModuleId {
        *self
            .map
            .entry(handle)
            .or_insert_with(|| {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                ModuleId(id)
            })
            .value()
    }

    pub fn get_from_handle(&self, handle: &Handle) -> ModuleId {
        self.get_or_insert(handle.dupe())
    }

    pub fn get_from_module(&self, module: &Module, sys_info: SysInfo) -> ModuleId {
        self.get_or_insert(Handle::new(module.name(), module.path().dupe(), sys_info))
    }
}
