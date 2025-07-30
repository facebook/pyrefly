/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use lsp_types::Location;
use lsp_types::Url;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::absolutize::Absolutize as _;
use tracing::warn;

use crate::module::from_path::module_from_path;
use crate::module::module_info::ModuleInfo;
use crate::module::typeshed::typeshed;
use crate::state::handle::Handle;
use crate::state::state::State;

/// Convert to a path we can show to the user. The contents may not match the disk, but it has
/// to be basically right.
pub fn to_real_path(path: &ModulePath) -> Option<PathBuf> {
    match path.details() {
        ModulePathDetails::FileSystem(path)
        | ModulePathDetails::Memory(path)
        | ModulePathDetails::Namespace(path) => Some(path.to_path_buf()),
        ModulePathDetails::BundledTypeshed(path) => {
            let typeshed = typeshed().ok()?;
            let typeshed_path = match typeshed.materialized_path_on_disk() {
                Ok(typeshed_path) => Some(typeshed_path),
                Err(err) => {
                    warn!("Builtins unable to be loaded on disk, {}", err);
                    None
                }
            }?;
            Some(typeshed_path.join(path))
        }
    }
}

pub fn module_info_to_uri(module_info: &ModuleInfo) -> Option<Url> {
    let path = to_real_path(module_info.path())?;
    let abs_path = path.absolutize();
    Some(Url::from_file_path(abs_path).unwrap())
}

pub fn handle_from_module_path(state: &State, path: ModulePath) -> Handle {
    let unknown = ModuleName::unknown();
    let config = state.config_finder().python_file(unknown, &path);
    let module_name = to_real_path(&path)
        .and_then(|path| module_from_path(&path, config.search_path()))
        .unwrap_or(unknown);
    Handle::new(module_name, path, config.get_sys_info())
}

pub fn make_open_handle(state: &State, path: &Path) -> Handle {
    let path = ModulePath::memory(path.to_owned());
    handle_from_module_path(state, path)
}

pub fn to_lsp_location(location: &TextRangeWithModule) -> Option<Location> {
    let TextRangeWithModule {
        module: definition_module_info,
        range,
    } = location;
    let uri = module_info_to_uri(definition_module_info)?;
    Some(Location {
        uri,
        range: definition_module_info.lined_buffer().to_lsp_range(*range),
    })
}
