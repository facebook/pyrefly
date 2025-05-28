/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use anyhow::anyhow;
use dupe::Dupe;
use itertools::Itertools;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::locked_map::LockedMap;

use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::config::ImportLookupPathPart;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;

#[derive(Debug, Clone, Dupe)]
pub enum FindError {
    /// This module could not be found, and we should emit an error
    NotFound(Arc<anyhow::Error>, ModuleName),
    /// This import could not be found, but the user configured it to be ignored
    Ignored,
    /// This site package path entry was found, but does not have a py.typed entry
    /// and use_untyped_imports is disabled
    NoPyTyped,
    /// We found stubs, but no source files were found. This means it's likely stubs
    /// are installed for a project, but the library is not actually importable
    NoSource(ModuleName),
}

impl FindError {
    pub fn not_found(err: anyhow::Error, module: ModuleName) -> Self {
        Self::NotFound(Arc::new(err), module)
    }

    pub fn no_source(module: ModuleName) -> Self {
        Self::NoSource(module)
    }

    pub fn import_lookup_path(
        path: Vec<ImportLookupPathPart>,
        module: ModuleName,
        config_source: &ConfigSource,
    ) -> FindError {
        let config_suffix = match config_source {
            ConfigSource::File(p) => format!(" (from config in `{}`)", p.display()),
            ConfigSource::Marker(p) => {
                format!(
                    " (from default config for project root marked by `{}`)",
                    p.display()
                )
            }
            _ => "".to_owned(),
        };
        let path_dump = path
            .iter()
            .filter_map(|path| {
                if path.is_empty() {
                    None
                } else {
                    Some(format!("\n  {path}"))
                }
            })
            .join("");
        let explanation = if path_dump.is_empty() {
            format!("no search path or site package path{config_suffix}")
        } else {
            format!("looked in these locations{config_suffix}:{path_dump}")
        };
        FindError::NotFound(Arc::new(anyhow!(explanation)), module)
    }

    pub fn display(&self) -> String {
        match self {
            Self::NotFound(err, module) => {
                format!("Could not find import of `{module}`, {:#}", err)
            }
            Self::Ignored => "Ignored import".to_owned(),
            Self::NoPyTyped => "Imported package does not contain a py.typed file, \
                and therefore cannot be typed. Try installing a `<package name>-stubs` version
                of your package to get the released stubs, or enable `use_untyped_imports` to
                disable this error."
                .to_owned(),
            Self::NoSource(module) => {
                format!(
                    "Found stubs for `{module}`, but no source. This means it's likely not \
                    installed/unimportable. See `ignore_missing_source` to disable this error."
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct LoaderFindCache {
    config: ArcId<ConfigFile>,
    cache: LockedMap<ModuleName, Result<ModulePath, FindError>>,
}

impl LoaderFindCache {
    pub fn new(config: ArcId<ConfigFile>) -> Self {
        Self {
            config,
            cache: Default::default(),
        }
    }

    pub fn find_import(
        &self,
        module: ModuleName,
        path: Option<&Path>,
    ) -> Result<ModulePath, FindError> {
        self.cache
            .ensure(&module, || self.config.find_import(module, path))
            .dupe()
    }
}
