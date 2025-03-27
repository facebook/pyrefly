/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use anyhow::anyhow;
use dupe::Dupe;
use ruff_text_size::TextRange;

use crate::error::collector::ErrorCollector;
use crate::error::kind::ErrorKind;
use crate::error::style::ErrorStyle;
use crate::module::bundled::typeshed;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::state::loader::Loader;
use crate::util::fs_anyhow;

#[derive(Debug)]
pub struct Load {
    pub errors: ErrorCollector,
    pub module_info: ModuleInfo,
}

impl Load {
    /// Return the code for this module, and whether there was an error while loading (a self-error).
    pub fn load_from_path(
        path: &ModulePath,
        loader: &dyn Loader,
    ) -> (Arc<String>, Option<anyhow::Error>) {
        let res = match path.details() {
            ModulePathDetails::FileSystem(path) => fs_anyhow::read_to_string(path).map(Arc::new),
            ModulePathDetails::Namespace(_) => Ok(Arc::new("".to_owned())),
            ModulePathDetails::Memory(path) => loader
                .load_from_memory(path)
                .ok_or_else(|| anyhow!("memory path not found")),
            ModulePathDetails::BundledTypeshed(path) => typeshed().and_then(|x| {
                x.load(path)
                    .ok_or_else(|| anyhow!("bundled typeshed problem"))
            }),
        };
        match res {
            Err(err) => (Arc::new(String::new()), Some(err)),
            Ok(res) => (res, None),
        }
    }

    pub fn load_from_data(
        name: ModuleName,
        path: ModulePath,
        error_style: ErrorStyle,
        code: Arc<String>,
        self_error: Option<anyhow::Error>,
    ) -> Self {
        let module_info = ModuleInfo::new(name, path, code);
        let errors = ErrorCollector::new(module_info.dupe(), error_style);
        if let Some(err) = self_error {
            errors.add(
                TextRange::default(),
                format!(
                    "Failed to load `{name}` from `{}`, got {err:#}",
                    module_info.path()
                ),
                ErrorKind::ImportError,
                None,
            );
        }
        Self {
            errors,
            module_info,
        }
    }
}
