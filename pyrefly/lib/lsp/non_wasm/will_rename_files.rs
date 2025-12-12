/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;

use lsp_types::DocumentChangeOperation;
use lsp_types::DocumentChanges;
use lsp_types::OneOf;
use lsp_types::OptionalVersionedTextDocumentIdentifier;
use lsp_types::RenameFilesParams;
use lsp_types::TextDocumentEdit;
use lsp_types::TextEdit;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::lined_buffer::LinedBuffer;
use pyrefly_util::lock::RwLock;
use rayon::prelude::*;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use tracing::info;

use crate::lsp::non_wasm::module_helpers::handle_from_module_path;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::state::load::LspFile;
use crate::state::state::State;
use crate::state::state::Transaction;

/// Visitor that looks for imports of an old module name and creates TextEdits to update them
struct RenameUsageVisitor<'a> {
    edits: Vec<TextEdit>,
    old_module_name: &'a ModuleName,
    new_module_name: &'a ModuleName,
    lined_buffer: &'a LinedBuffer,
}

impl<'a> RenameUsageVisitor<'a> {
    fn new(
        old_module_name: &'a ModuleName,
        new_module_name: &'a ModuleName,
        lined_buffer: &'a LinedBuffer,
    ) -> Self {
        Self {
            edits: Vec::new(),
            old_module_name,
            new_module_name,
            lined_buffer,
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Import(import) => {
                for alias in &import.names {
                    let imported_module = ModuleName::from_name(&alias.name.id);
                    if imported_module == *self.old_module_name
                        || imported_module
                            .as_str()
                            .starts_with(&format!("{}.", self.old_module_name.as_str()))
                    {
                        // Replace the module name
                        let new_import_name = if imported_module == *self.old_module_name {
                            self.new_module_name.as_str().to_owned()
                        } else {
                            // Replace the prefix
                            imported_module.as_str().replace(
                                self.old_module_name.as_str(),
                                self.new_module_name.as_str(),
                            )
                        };

                        self.edits.push(TextEdit {
                            range: self.lined_buffer.to_lsp_range(alias.name.range(), None),
                            new_text: new_import_name,
                        });
                    }
                }
            }
            Stmt::ImportFrom(import_from) => {
                if let Some(module) = &import_from.module {
                    let imported_module = ModuleName::from_name(&module.id);
                    if imported_module == *self.old_module_name
                        || imported_module
                            .as_str()
                            .starts_with(&format!("{}.", self.old_module_name.as_str()))
                    {
                        // Replace the module name
                        let new_import_name = if imported_module == *self.old_module_name {
                            self.new_module_name.as_str().to_owned()
                        } else {
                            // Replace the prefix
                            imported_module.as_str().replace(
                                self.old_module_name.as_str(),
                                self.new_module_name.as_str(),
                            )
                        };

                        self.edits.push(TextEdit {
                            range: self.lined_buffer.to_lsp_range(module.range(), None),
                            new_text: new_import_name,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn take_edits(self) -> Vec<TextEdit> {
        self.edits
    }
}

/// Handle workspace/willRenameFiles request to update imports when files are renamed.
///
/// This function:
/// 1. Converts file paths to module names
/// 2. Uses get_transitive_rdeps to find all files that depend on the renamed module
/// 3. Uses a visitor pattern to find imports of the old module and creates TextEdits
/// 4. Returns a WorkspaceEdit with all necessary changes
///
/// If the client supports `workspace.workspaceEdit.documentChanges`, the response will use
/// `document_changes` instead of `changes` for better ordering guarantees and version checking.
pub fn will_rename_files(
    state: &State,
    transaction: &Transaction<'_>,
    _open_files: &RwLock<HashMap<std::path::PathBuf, Arc<LspFile>>>,
    params: RenameFilesParams,
    supports_document_changes: bool,
) -> Option<WorkspaceEdit> {
    info!(
        "will_rename_files called with {} file(s)",
        params.files.len()
    );

    let mut all_changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();

    for file_rename in &params.files {
        info!(
            "  Processing rename: {} -> {}",
            file_rename.old_uri, file_rename.new_uri
        );

        // Convert URLs to paths
        let old_uri = match Url::parse(&file_rename.old_uri) {
            Ok(uri) => uri,
            Err(_) => {
                info!("    Failed to parse old_uri");
                continue;
            }
        };

        let new_uri = match Url::parse(&file_rename.new_uri) {
            Ok(uri) => uri,
            Err(_) => {
                info!("    Failed to parse new_uri");
                continue;
            }
        };

        let old_path = match old_uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                info!("    Failed to convert old_uri to path");
                continue;
            }
        };

        let new_path = match new_uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                info!("    Failed to convert new_uri to path");
                continue;
            }
        };

        // Only process Python files
        if !PYTHON_EXTENSIONS
            .iter()
            .any(|ext| old_path.extension().and_then(|e| e.to_str()) == Some(*ext))
        {
            info!("    Skipping non-Python file");
            continue;
        }

        // Important: only use filesystem handle (never use an in-memory handle)
        let module_path = ModulePath::filesystem(old_path.clone());
        let old_handle = handle_from_module_path(state, module_path.clone());

        // Convert paths to module names
        let old_module_name = old_handle.module();

        let config = state
            .config_finder()
            .python_file(old_module_name, &module_path);
        let new_module_name = ModuleName::from_path(
            &new_path,
            config.search_path().chain(
                config
                    .fallback_search_path
                    .for_directory(new_path.parent())
                    .iter(),
            ),
        );

        let new_module_name = match new_module_name {
            Some(name) => name,
            None => {
                info!("    Could not determine new module name, skipping");
                continue;
            }
        };

        info!(
            "    Module rename: {} -> {}",
            old_module_name, new_module_name
        );

        // If module names are the same, no need to update imports
        if old_module_name == new_module_name {
            info!("    Module names are the same, skipping");
            continue;
        }

        // Use get_transitive_rdeps to find all files that depend on this module
        let rdeps = transaction.get_transitive_rdeps(old_handle.clone());

        info!("    Found {} transitive rdeps", rdeps.len());

        // Deduplicate rdeps by module path string (get_transitive_rdeps might return duplicates
        // with different variants like FileSystem vs Memory for the same path)
        let unique_rdeps: Vec<_> = {
            let mut seen = std::collections::HashSet::new();
            rdeps
                .into_iter()
                .filter(|handle| seen.insert(handle.path().as_path().to_owned()))
                .collect()
        };

        // Visit each dependent file to find and update imports (parallelized)
        let rdeps_changes: Vec<(Url, Vec<TextEdit>)> = unique_rdeps
            .into_par_iter()
            .filter_map(|rdep_handle| {
                let module_info = transaction.get_module_info(&rdep_handle)?;

                let ast = Ast::parse(module_info.contents(), module_info.source_type()).0;
                let mut visitor = RenameUsageVisitor::new(
                    &old_module_name,
                    &new_module_name,
                    module_info.lined_buffer(),
                );

                for stmt in &ast.body {
                    visitor.visit_stmt(stmt);
                }

                let edits_for_file = visitor.take_edits();

                if !edits_for_file.is_empty() {
                    let uri = module_info_to_uri(&module_info)?;
                    info!(
                        "    Found {} import(s) to update in {}",
                        edits_for_file.len(),
                        uri
                    );
                    Some((uri, edits_for_file))
                } else {
                    None
                }
            })
            .collect();

        // Merge results into all_changes
        for (uri, edits) in rdeps_changes {
            all_changes.entry(uri).or_default().extend(edits);
        }
    }

    if all_changes.is_empty() {
        info!("  No import updates needed");
        None
    } else {
        info!(
            "  Returning {} file(s) with import updates",
            all_changes.len()
        );

        if supports_document_changes {
            // Use document_changes for better ordering guarantees and version checking
            // Sort by URI for deterministic ordering
            let mut sorted_changes: Vec<(Url, Vec<TextEdit>)> = all_changes.into_iter().collect();
            sorted_changes.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));

            let document_changes: Vec<DocumentChangeOperation> = sorted_changes
                .into_iter()
                .map(|(uri, edits)| {
                    DocumentChangeOperation::Edit(TextDocumentEdit {
                        text_document: OptionalVersionedTextDocumentIdentifier {
                            uri,
                            version: None, // None means "any version"
                        },
                        edits: edits.into_iter().map(OneOf::Left).collect(),
                    })
                })
                .collect();

            Some(WorkspaceEdit {
                document_changes: Some(DocumentChanges::Operations(document_changes)),
                ..Default::default()
            })
        } else {
            // Fall back to changes for older clients
            Some(WorkspaceEdit {
                changes: Some(all_changes),
                ..Default::default()
            })
        }
    }
}
