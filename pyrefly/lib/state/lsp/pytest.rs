/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use vec1::Vec1;

use super::DefinitionMetadata;
use super::FindDefinitionItemWithDocstring;
use super::FindPreference;
use crate::state::pytest::find_pytest_fixture_definitions_for_parameter;
use crate::state::pytest::find_pytest_fixture_definitions_in_conftest;
use crate::state::pytest::find_pytest_fixture_parameter_references;
use crate::state::pytest::is_pytest_fixture_parameter_context;
use crate::state::state::Transaction;

pub(crate) fn pytest_conftest_handles(
    transaction: &Transaction<'_>,
    handle: &Handle,
    preference: FindPreference,
) -> Vec<Handle> {
    let module_path = handle.path();
    let Some(mut dir) = module_path.as_path().parent() else {
        return Vec::new();
    };
    let root = module_path
        .root_of(handle.module())
        .unwrap_or_else(|| dir.to_path_buf());
    let filenames = if preference.prefer_pyi {
        ["conftest.pyi", "conftest.py"]
    } else {
        ["conftest.py", "conftest.pyi"]
    };
    let mut handles = Vec::new();
    loop {
        for filename in filenames {
            let path = dir.join(filename);
            let memory_path = ModulePath::memory(path.clone());
            let memory_config = transaction.config_finder().python_file(
                ModuleNameWithKind::guaranteed(ModuleName::unknown()),
                &memory_path,
            );
            let memory_handle = memory_config.handle_from_module_path(memory_path);
            if transaction.get_module_info(&memory_handle).is_some() {
                handles.push(memory_handle);
            } else if path.exists() {
                let filesystem_path = ModulePath::filesystem(path);
                let filesystem_config = transaction.config_finder().python_file(
                    ModuleNameWithKind::guaranteed(ModuleName::unknown()),
                    &filesystem_path,
                );
                handles.push(filesystem_config.handle_from_module_path(filesystem_path));
            }
        }
        if dir == root {
            break;
        }
        let Some(parent) = dir.parent() else {
            break;
        };
        dir = parent;
    }
    handles
}

impl<'a> Transaction<'a> {
    /// Resolve a pytest fixture parameter to the fixture functions that can provide it.
    ///
    /// This runs during definition lookup. The common non-pytest path is cheap because we first
    /// require either a pytest fixture function or a test-named function.
    pub(super) fn pytest_fixture_definitions_for_parameter(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        covering_nodes: &[AnyNodeRef],
        preference: FindPreference,
    ) -> Option<Vec1<FindDefinitionItemWithDocstring>> {
        let mod_module = self.get_ast(handle)?;
        let bindings = self.get_bindings(handle)?;
        if !is_pytest_fixture_parameter_context(&bindings, covering_nodes) {
            return None;
        }
        let matches = find_pytest_fixture_definitions_for_parameter(
            mod_module.as_ref(),
            &bindings,
            identifier,
            covering_nodes,
        );
        let module_info = self.get_module_info(handle)?;
        let mut definitions: Vec<_> = matches
            .into_iter()
            .map(|fixture| FindDefinitionItemWithDocstring {
                metadata: DefinitionMetadata::Variable(Some(SymbolKind::Function)),
                definition_range: fixture.range,
                module: module_info.clone(),
                docstring_range: fixture.docstring_range,
                display_name: Some(fixture.name.as_str().to_owned()),
            })
            .collect();
        if definitions.is_empty() {
            for conftest_handle in pytest_conftest_handles(self, handle, preference) {
                let _ = self.get_exports(&conftest_handle);
                let conftest_module_info = self
                    .get_module_info(&conftest_handle)
                    .expect("conftest module must be loaded after demanding exports");
                let conftest_ast = Ast::parse(
                    conftest_module_info.contents(),
                    conftest_module_info.source_type(),
                )
                .0;
                definitions.extend(
                    find_pytest_fixture_definitions_in_conftest(&conftest_ast, identifier.id())
                        .into_iter()
                        .map(|fixture| FindDefinitionItemWithDocstring {
                            metadata: DefinitionMetadata::Variable(Some(SymbolKind::Function)),
                            definition_range: fixture.range,
                            module: conftest_module_info.clone(),
                            docstring_range: fixture.docstring_range,
                            display_name: Some(fixture.name.as_str().to_owned()),
                        }),
                );
                if !definitions.is_empty() {
                    break;
                }
            }
        }
        Vec1::try_from_vec(definitions).ok()
    }

    /// Find local pytest test/fixture parameters that reference a fixture definition.
    ///
    /// This is only used after the regular reference path has found a definition. Modules without
    /// pytest metadata return before walking the AST.
    pub(super) fn local_pytest_fixture_parameter_references(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        expected_name: &Name,
    ) -> Option<Vec<TextRange>> {
        let mod_module = self.get_ast(handle)?;
        let bindings = self.get_bindings(handle)?;
        find_pytest_fixture_parameter_references(
            mod_module.as_ref(),
            &bindings,
            definition_range,
            expected_name,
        )
    }
}
