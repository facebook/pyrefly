/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe as _;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::absolutize::Absolutize as _;

use crate::lsp::module_helpers::to_real_path;
use crate::module::module_info::ModuleInfo;
use crate::state::state::State;

/// Callback type for remapping paths before converting to URIs.
/// Used for setups where file paths need to be transformed when converting
/// to URIs.
pub type PathRemapper = Arc<dyn Fn(&Path) -> Cow<'_, Path> + Send + Sync>;

/// Convert ModuleInfo to URI with optional path remapping.
/// When a path remapper is provided, the path is transformed before
/// being converted to a URI.
pub fn module_info_to_uri(
    module_info: &ModuleInfo,
    remapper: Option<&PathRemapper>,
) -> Option<Url> {
    let path = to_real_path(module_info.path())?;
    let final_path: PathBuf = match remapper {
        Some(remap_fn) => remap_fn(&path).into_owned(),
        None => path,
    };
    let abs_path = final_path.absolutize();
    Some(Url::from_file_path(abs_path).unwrap())
}

pub(in crate::lsp) fn handle_from_module_path(state: &State, path: ModulePath) -> Handle {
    let unknown = ModuleName::unknown();
    let config = state
        .config_finder()
        .python_file(ModuleNameWithKind::guaranteed(unknown), &path);
    match path.details() {
        ModulePathDetails::BundledTypeshed(_) => {
            let module_name = to_real_path(&path)
                .and_then(|path| ModuleName::from_path(&path, config.search_path()))
                .unwrap_or(unknown);
            Handle::new(module_name, path, config.get_sys_info())
        }
        _ => {
            config.handle_from_module_path_with_fallback(path.dupe(), &config.fallback_search_path)
        }
    }
}

pub fn make_open_handle(state: &State, path: &Path) -> Handle {
    let path = ModulePath::memory(path.to_owned());
    handle_from_module_path(state, path)
}
