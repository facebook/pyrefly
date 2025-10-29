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
use crate::module::typeshed::typeshed;
use crate::module::typeshed_third_party::typeshed_third_party;
use crate::state::memory::MemoryFilesLookup;

/// The result of loading a module, including its `Module` and `ErrorCollector`.
#[derive(Debug)]
pub struct Load {
    pub errors: ErrorCollector,
    pub module_info: Module,
}

pub enum CodeOrNotebook {
    Code(Arc<String>),
    Notebook(Box<Notebook>),
}

impl Load {
    /// Return the code for this module, optional notebook cell mapping, and whether there was an error while loading (a self-error).
    pub fn load_from_path(
        path: &ModulePath,
        memory_lookup: &MemoryFilesLookup,
    ) -> (CodeOrNotebook, Option<anyhow::Error>) {
        let res = match path.details() {
            ModulePathDetails::FileSystem(path) => Self::load_from_filesystem(path),
            ModulePathDetails::Namespace(_) => Ok(CodeOrNotebook::Code(Arc::new("".to_owned()))),
            ModulePathDetails::Memory(path) => memory_lookup
                .get(path)
                .duped()
                .ok_or_else(|| anyhow!("memory path not found"))
                .map(CodeOrNotebook::Code),
            ModulePathDetails::BundledTypeshed(path) => typeshed().and_then(|x| {
                x.load(path)
                    .ok_or_else(|| anyhow!("bundled typeshed problem"))
                    .map(CodeOrNotebook::Code)
            }),
            ModulePathDetails::BundledTypeshedThirdParty(path) => {
                typeshed_third_party().and_then(|x| {
                    x.load(path)
                        .ok_or_else(|| anyhow!("bundled typeshed third party problem"))
                        .map(CodeOrNotebook::Code)
                })
            }
        };
        match res {
            Err(err) => (CodeOrNotebook::Code(Arc::new(String::new())), Some(err)),
            Ok(code_or_notebook) => (code_or_notebook, None),
        }
    }

    /// Load from filesystem, handling both regular Python files and Jupyter notebooks.
    /// Returns the code and optional notebook.
    fn load_from_filesystem(path: &Path) -> anyhow::Result<CodeOrNotebook> {
        let content = fs_anyhow::read_to_string(path)?;

        // Check if this is a Jupyter notebook by file extension
        if path.extension().and_then(|s| s.to_str()) == Some("ipynb") {
            // Extract Python code from the notebook
            let notebook = Notebook::from_source_code(&content)?;
            Ok(CodeOrNotebook::Notebook(Box::new(notebook)))
        } else {
            // Regular Python file
            Ok(CodeOrNotebook::Code(Arc::new(content)))
        }
    }

    pub fn load_from_data(
        name: ModuleName,
        path: ModulePath,
        error_style: ErrorStyle,
        code_or_notebook: CodeOrNotebook,
        self_error: Option<anyhow::Error>,
    ) -> Self {
        let module_info = match code_or_notebook {
            CodeOrNotebook::Code(code) => Module::new(name, path, code),
            CodeOrNotebook::Notebook(notebook) => Module::new_notebook(name, path, *notebook),
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
