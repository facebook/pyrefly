/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_config::error_kind::ErrorKind;
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

#[derive(Debug, Clone, Dupe, PartialEq, Eq)]
pub enum FindError {
    /// This module could not be found, and we should emit an error
    NotFound(ModuleName, Arc<Vec1<String>>),
    /// This import could not be found, but the user configured it to be ignored
    Ignored,
    /// We found stubs, but no source files were found. This means it's likely stubs
    /// are installed for a project, but the library is not actually importable
    NoSource(ModuleName),
    /// We have the source files, but do not have the stubs. In this case we should send
    /// a message to the user which will allow them to install the stubs for the package.
    /// The string will hold the name of the pip package that we will tell the user to install.
    MissingStubs(ModuleName, Arc<String>),
    /// This is the condition where we are using stubs but we do not have the source files
    NoSourceForStubs(ModuleName),
}

impl FindError {
    pub fn not_found(err: anyhow::Error, module: ModuleName) -> Self {
        Self::NotFound(module, Arc::new(vec1![format!("{err:#}")]))
    }

    pub fn import_lookup_path(
        path: Vec<ImportLookupPathPart>,
        module: ModuleName,
        config_source: &ConfigSource,
        suggestion: Option<ModuleName>,
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
        if let Some(suggested) = suggestion {
            explanation.insert(0, format!("Did you mean `{suggested}`?"));
        }
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
                    installed/unimportable."
                )],
            ),
            Self::NoSourceForStubs(module) => (
                None,
                vec1![format!(
                    "Stubs for `{module}` are bundled with Pyrefly but the source files for the package are not found."
                )],
            ),
            Self::MissingStubs(source_package, stubs_package) => (
                Some(Box::new(|| ErrorContext::ImportNotTyped(*source_package))),
                vec1![format!("Hint: install the `{stubs_package}` package")],
            ),
        }
    }

    pub fn kind(&self) -> Option<ErrorKind> {
        match self {
            Self::NotFound(..) => Some(ErrorKind::MissingImport),
            Self::NoSource(..) => Some(ErrorKind::MissingSource),
            Self::NoSourceForStubs(..) => Some(ErrorKind::MissingSourceForStubs),
            Self::MissingStubs(..) => Some(ErrorKind::UntypedImport),
            Self::Ignored => None,
        }
    }
}

#[derive(Debug, Clone, Dupe, PartialEq, Eq)]
pub struct Finding<T> {
    pub finding: T,
    pub error: Option<FindError>,
}

/// Result of an attempt to find a module
#[derive(Debug, Clone, Dupe, PartialEq, Eq)]
pub enum FindingOrError<T> {
    /// Information about a found module. May have a non-fatal error attached.
    Finding(Finding<T>),
    /// A fatal error that prevented us from finding a module.
    Error(FindError),
}

impl<T> FindingOrError<T> {
    pub fn new_finding(finding: T) -> Self {
        Self::Finding(Finding {
            finding,
            error: None,
        })
    }

    pub fn finding(self) -> Option<T> {
        match self {
            Self::Finding(finding) => Some(finding.finding),
            Self::Error(_) => None,
        }
    }

    pub fn error(self) -> Option<FindError> {
        match self {
            Self::Finding(Finding { error, finding: _ }) => error,
            Self::Error(error) => Some(error),
        }
    }

    pub fn map<T2>(self, f: impl Fn(T) -> T2) -> FindingOrError<T2> {
        match self {
            Self::Finding(Finding { finding, error }) => FindingOrError::Finding(Finding {
                finding: f(finding),
                error,
            }),
            Self::Error(e) => FindingOrError::Error(e),
        }
    }

    pub fn with_error(self, error: FindError) -> Self {
        match self {
            Self::Finding(x) if x.error.is_none() => Self::Finding(Finding {
                finding: x.finding,
                error: Some(error),
            }),
            x => x,
        }
    }
}

#[derive(Debug)]
pub struct LoaderFindCache {
    config: ArcId<ConfigFile>,
    cache: LockedMap<(ModuleName, Option<ModulePath>), FindingOrError<ModulePath>>,
    // If a python executable module (excludes .pyi) exists and differs from the imported python module, store it here
    executable_cache: LockedMap<(ModuleName, Option<ModulePath>), Option<ModulePath>>,
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
        origin: Option<&ModulePath>,
    ) -> FindingOrError<ModulePath> {
        let key = (module.dupe(), origin.cloned());
        match self.executable_cache.get(&key) {
            Some(Some(module)) => FindingOrError::new_finding(module.dupe()),
            Some(None) => self.find_import(module, origin),
            None => {
                match find_import_filtered(
                    &self.config,
                    module,
                    origin,
                    Some(ModuleStyle::Executable),
                ) {
                    FindingOrError::Finding(import) => {
                        self.executable_cache
                            .insert(key, Some(import.finding.dupe()));
                        FindingOrError::Finding(import)
                    }
                    FindingOrError::Error(_) => {
                        self.executable_cache.insert(key, None);
                        self.find_import(module, origin)
                    }
                }
            }
        }
    }

    pub fn find_import(
        &self,
        module: ModuleName,
        origin: Option<&ModulePath>,
    ) -> FindingOrError<ModulePath> {
        self.cache
            .ensure(&(module.dupe(), origin.cloned()), || {
                find_import(&self.config, module, origin)
            })
            .0
            .dupe()
    }
}
