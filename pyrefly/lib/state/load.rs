/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::sync::Arc;

use anyhow::anyhow;
use dupe::Dupe;
use dupe::OptionDupedExt;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use ruff_notebook::Notebook;
use ruff_text_size::TextRange;
use vec1::vec1;

use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::style::ErrorStyle;
use crate::module::bundled::BundledStub;
use crate::module::third_party::get_bundled_third_party;
use crate::module::typeshed::typeshed;
use crate::module::typeshed_third_party::typeshed_third_party;
use crate::state::memory::MemoryFilesLookup;
use crate::state::notebook::LspNotebook;

#[derive(Clone, Dupe, Debug, Eq, PartialEq)]
pub enum FileContents {
    Source(Arc<String>),
    Notebook(Arc<Notebook>),
}

impl FileContents {
    pub fn from_source(source: String) -> Self {
        Self::Source(Arc::new(source))
    }
}

/// This is the representation of files in the language server
/// It can be converted to `FileContents`
#[derive(Clone, Dupe, Debug, Eq, PartialEq)]
pub enum LspFile {
    Source(Arc<String>),
    Notebook(Arc<LspNotebook>),
}

impl LspFile {
    pub fn get_string(&self) -> &str {
        match self {
            Self::Source(contents) => contents.as_str(),
            Self::Notebook(notebook) => notebook.ruff_notebook().source_code(),
        }
    }

    pub fn from_source(source: String) -> Self {
        Self::Source(Arc::new(source))
    }

    pub fn to_file_contents(&self) -> FileContents {
        match self {
            Self::Source(contents) => FileContents::Source(Arc::clone(contents)),
            Self::Notebook(notebook) => {
                FileContents::Notebook(Arc::clone(notebook.ruff_notebook()))
            }
        }
    }

    pub fn is_notebook(&self) -> bool {
        matches!(self, Self::Notebook(_))
    }
}

/// The result of loading a module, including its `Module` and `ErrorCollector`.
#[derive(Debug)]
pub struct Load {
    pub errors: ErrorCollector,
    pub module_info: Module,
}

impl Load {
    /// Return the code for this module, optional notebook cell mapping, and whether there was an error while loading (a self-error).
    pub fn load_from_path(
        path: &ModulePath,
        memory_lookup: &MemoryFilesLookup,
    ) -> (FileContents, Option<anyhow::Error>) {
        let res = match path.details() {
            ModulePathDetails::FileSystem(path) => Self::load_from_filesystem(path),
            ModulePathDetails::Namespace(_) => Ok(FileContents::from_source("".to_owned())),
            ModulePathDetails::Memory(path) => memory_lookup
                .get(path)
                .duped()
                .as_deref()
                .duped()
                .ok_or_else(|| anyhow!("memory path not found")),
            ModulePathDetails::BundledTypeshed(path) => typeshed().and_then(|x| {
                x.load(path)
                    .ok_or_else(|| anyhow!("bundled typeshed problem"))
                    .map(FileContents::Source)
            }),
            ModulePathDetails::BundledTypeshedThirdParty(path) => {
                typeshed_third_party().and_then(|x| {
                    x.load(path)
                        .ok_or_else(|| anyhow!("bundled typeshed third party problem"))
                        .map(FileContents::Source)
                })
            }
            ModulePathDetails::BundledThirdParty(path) => get_bundled_third_party().and_then(|x| {
                x.load(path)
                    .ok_or_else(|| anyhow!("bundled third party problem"))
                    .map(FileContents::Source)
            }),
        };
        match res {
            Err(err) => (FileContents::from_source(String::new()), Some(err)),
            Ok(file_contents) => (file_contents, None),
        }
    }

    /// Load from filesystem, handling both regular Python files and Jupyter notebooks.
    /// Returns the code and optional notebook.
    fn load_from_filesystem(path: &Path) -> anyhow::Result<FileContents> {
        let content = fs_anyhow::read_to_string(path)?;

        // Check if this is a Jupyter notebook by file extension
        if path.extension().and_then(|s| s.to_str()) == Some("ipynb") {
            // Extract Python code from the notebook
            let notebook = Notebook::from_source_code(&content)?;
            Ok(FileContents::Notebook(Arc::new(notebook)))
        } else {
            // Regular Python file
            Ok(FileContents::from_source(content))
        }
    }

    pub fn load_from_data(
        name: ModuleName,
        path: ModulePath,
        error_style: ErrorStyle,
        file_contents: FileContents,
        self_error: Option<anyhow::Error>,
    ) -> Self {
        let module_info = match file_contents {
            FileContents::Source(code) => Module::new(name, path, code),
            FileContents::Notebook(notebook) => {
                Module::new_notebook(name, path, Arc::clone(&notebook))
            }
        };
        let errors = ErrorCollector::new(module_info.dupe(), error_style);
        if let Some(err) = self_error {
            errors.add(
                TextRange::default(),
                ErrorInfo::Kind(ErrorKind::MissingImport),
                vec1![format!(
                    "Failed to load `{name}` from `{}`, got {err:#}",
                    module_info.path()
                )],
            );
        }
        Self {
            errors,
            module_info,
        }
    }
}
