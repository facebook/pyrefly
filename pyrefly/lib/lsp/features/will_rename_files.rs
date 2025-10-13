/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;

use lsp_types::RenameFilesParams;
use lsp_types::TextEdit;
use lsp_types::Url;
use lsp_types::WorkspaceEdit;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::lined_buffer::LinedBuffer;
use pyrefly_util::lock::RwLock;
use rayon::prelude::*;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;

use crate::lsp::module_helpers::make_open_handle;
use crate::lsp::module_helpers::module_info_to_uri;
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
                            range: self.lined_buffer.to_lsp_range(alias.name.range()),
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
                            range: self.lined_buffer.to_lsp_range(module.range()),
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
pub fn will_rename_files(
    state: &Arc<State>,
    transaction: &Transaction<'_>,
    _open_files: &Arc<RwLock<HashMap<std::path::PathBuf, Arc<String>>>>,
    params: RenameFilesParams,
) -> Option<WorkspaceEdit> {
    eprintln!(
        "will_rename_files called with {} file(s)",
        params.files.len()
    );

    let mut all_changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();

    for file_rename in &params.files {
        eprintln!(
            "  Processing rename: {} -> {}",
            file_rename.old_uri, file_rename.new_uri
        );

        // Convert URLs to paths
        let old_uri = match Url::parse(&file_rename.old_uri) {
            Ok(uri) => uri,
            Err(_) => {
                eprintln!("    Failed to parse old_uri");
                continue;
            }
        };

        let new_uri = match Url::parse(&file_rename.new_uri) {
            Ok(uri) => uri,
            Err(_) => {
                eprintln!("    Failed to parse new_uri");
                continue;
            }
        };

        let old_path = match old_uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                eprintln!("    Failed to convert old_uri to path");
                continue;
            }
        };

        let new_path = match new_uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                eprintln!("    Failed to convert new_uri to path");
                continue;
            }
        };

        // Only process Python files
        if !PYTHON_EXTENSIONS
            .iter()
            .any(|ext| old_path.extension().and_then(|e| e.to_str()) == Some(*ext))
        {
            eprintln!("    Skipping non-Python file");
            continue;
        }

        // Get the config to find the search path
        let old_handle = make_open_handle(state, &old_path);
        let config = state
            .config_finder()
            .python_file(old_handle.module(), old_handle.path());

        // Convert paths to module names
        let old_module_name =
            ModuleName::from_path(&old_path, config.search_path()).or_else(|| {
                // Fallback: try to get module name from the handle
                Some(old_handle.module())
            });

        // For the new module name, we can't rely on from_path because the file doesn't exist yet.
        // Instead, we compute the relative path from the old to new file and adjust the module name.
        let new_module_name = if let Some(old_parent) = old_path.parent() {
            if let Some(new_parent) = new_path.parent() {
                // If both files are in the same directory, just replace the file name part
                if old_parent == new_parent {
                    // Extract the module name from the new file name
                    if let Some(file_stem) = new_path.file_stem() {
                        if let Some(file_stem_str) = file_stem.to_str() {
                            // If the old file was in a module, replace just the last component
                            if let Some(old_mod) = &old_module_name {
                                let mut components = old_mod.components();
                                if !components.is_empty() {
                                    components.pop();
                                    components.push(Name::new(file_stem_str));
                                    Some(ModuleName::from_parts(components))
                                } else {
                                    Some(ModuleName::from_str(file_stem_str))
                                }
                            } else {
                                Some(ModuleName::from_str(file_stem_str))
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    // Files are in different directories, try from_path
                    ModuleName::from_path(&new_path, config.search_path())
                }
            } else {
                None
            }
        } else {
            None
        };

        let (old_module_name, new_module_name) = match (old_module_name, new_module_name) {
            (Some(old), Some(new)) => (old, new),
            _ => {
                eprintln!(
                    "    Could not determine module names for the rename (old={:?}, new={:?})",
                    old_module_name, new_module_name
                );
                continue;
            }
        };

        eprintln!(
            "    Module rename: {} -> {}",
            old_module_name, new_module_name
        );

        // If module names are the same, no need to update imports
        if old_module_name == new_module_name {
            eprintln!("    Module names are the same, skipping");
            continue;
        }

        // Use get_transitive_rdeps to find all files that depend on this module
        let rdeps = transaction.get_transitive_rdeps(old_handle.clone());

        eprintln!("    Found {} transitive rdeps", rdeps.len());

        // Visit each dependent file to find and update imports (parallelized)
        let rdeps_changes: Vec<(Url, Vec<TextEdit>)> = rdeps
            .into_par_iter()
            .filter_map(|rdep_handle| {
                let module_info = transaction.get_module_info(&rdep_handle)?;

                let ast = Ast::parse(module_info.contents()).0;
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
                    eprintln!(
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
        eprintln!("  No import updates needed");
        None
    } else {
        eprintln!(
            "  Returning {} file(s) with import updates",
            all_changes.len()
        );
        Some(WorkspaceEdit {
            changes: Some(all_changes),
            ..Default::default()
        })
    }
}
