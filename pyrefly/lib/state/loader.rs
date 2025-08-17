/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::locked_map::LockedMap;
use vec1::Vec1;
use vec1::vec1;

use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::config::ImportLookupPathPart;
use crate::error::context::ErrorContext;
use crate::module::finder::find_import;
use crate::module::finder::find_import_filtered;

#[derive(Debug, Clone, Dupe)]
pub enum FindError {
    /// This module could not be found, and we should emit an error
    NotFound(ModuleName, Arc<Vec1<String>>),
    /// This import could not be found, but the user configured it to be ignored
    Ignored,
    /// We found stubs, but no source files were found. This means it's likely stubs
    /// are installed for a project, but the library is not actually importable
    NoSource(ModuleName),
}

impl FindError {
    pub fn not_found(err: anyhow::Error, module: ModuleName) -> Self {
        Self::NotFound(module, Arc::new(vec1![format!("{err:#}")]))
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
        let nonempty_paths = path
            .iter()
            .filter_map(|path| {
                if path.is_empty() {
                    None
                } else {
                    Some(format!("{path}"))
                }
            })
            .collect::<Vec<_>>();
        let mut explanation = vec1![if nonempty_paths.is_empty() {
            format!("No search path or site package path{config_suffix}")
        } else {
            format!("Looked in these locations{config_suffix}:")
        }];
        explanation.extend(nonempty_paths);
        FindError::NotFound(module, Arc::new(explanation))
    }

    pub fn display(&self) -> (Option<Box<dyn Fn() -> ErrorContext + '_>>, Vec1<String>) {
        match self {
            Self::NotFound(module, err) => (
                Some(Box::new(|| ErrorContext::ImportNotFound(*module))),
                (**err).clone(),
            ),
            Self::Ignored => (None, vec1!["Ignored import".to_owned()]),
            Self::NoSource(module) => (
                None,
                vec1![format!(
                    "Found stubs for `{module}`, but no source. This means it's likely not \
                    installed/unimportable. See `ignore-missing-source` to disable this error."
                )],
            ),
        }
    }
}

#[derive(Debug)]
pub struct LoaderFindCache {
    config: ArcId<ConfigFile>,
    cache: LockedMap<ModuleName, Result<ModulePath, FindError>>,
    // If a python executable module (excludes .pyi) exists and differs from the imported python module, store it here
    executable_cache: LockedMap<ModuleName, Option<ModulePath>>,
}

impl LoaderFindCache {
    pub fn new(config: ArcId<ConfigFile>) -> Self {
        Self {
            config,
            cache: Default::default(),
            executable_cache: Default::default(),
        }
    }

    pub fn find_import_prefer_executable(
        &self,
        module: ModuleName,
        path: Option<&Path>,
    ) -> Result<ModulePath, FindError> {
        match self.executable_cache.get(&module) {
            Some(Some(module)) => Ok(module.dupe()),
            Some(None) => self.find_import(module, path),
            None => {
                match find_import_filtered(
                    &self.config,
                    module,
                    path,
                    Some(ModuleStyle::Executable),
                ) {
                    Ok(import) => {
                        self.executable_cache.insert(module, Some(import.dupe()));
                        Ok(import)
                    }
                    Err(_) => {
                        self.executable_cache.insert(module, None);
                        self.find_import(module, path)
                    }
                }
            }
        }
    }

    pub fn find_import(
        &self,
        module: ModuleName,
        path: Option<&Path>,
    ) -> Result<ModulePath, FindError> {
        self.cache
            .ensure(&module, || find_import(&self.config, module, path))
            .dupe()
    }
}
