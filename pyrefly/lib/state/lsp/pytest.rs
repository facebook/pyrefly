/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use super::DefinitionMetadata;
use super::FindDefinitionItemWithDocstring;
use crate::state::pytest::pytest_fixture_definitions_for_parameter as pytest_fixture_definitions_for_parameter_in_module;
use crate::state::pytest::pytest_fixture_parameter_references as pytest_fixture_parameter_references_in_module;
use crate::state::state::Transaction;

impl<'a> Transaction<'a> {
    pub(super) fn pytest_fixture_definitions_for_parameter(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        covering_nodes: &[AnyNodeRef],
    ) -> Option<Vec<FindDefinitionItemWithDocstring>> {
        let mod_module = self.get_ast(handle)?;
        let bindings = self.get_bindings(handle)?;
        let matches = pytest_fixture_definitions_for_parameter_in_module(
            mod_module.as_ref(),
            &bindings,
            identifier,
            covering_nodes,
        )?;
        let module_info = self.get_module_info(handle)?;
        Some(
            matches
                .into_iter()
                .map(|fixture| FindDefinitionItemWithDocstring {
                    metadata: DefinitionMetadata::Variable(Some(SymbolKind::Function)),
                    definition_range: fixture.range,
                    module: module_info.clone(),
                    docstring_range: fixture.docstring_range,
                    display_name: Some(fixture.name.as_str().to_owned()),
                })
                .collect(),
        )
    }

    pub(super) fn local_pytest_fixture_parameter_references(
        &self,
        handle: &Handle,
        definition_range: TextRange,
        expected_name: &Name,
    ) -> Option<Vec<TextRange>> {
        let mod_module = self.get_ast(handle)?;
        let bindings = self.get_bindings(handle)?;
        pytest_fixture_parameter_references_in_module(
            mod_module.as_ref(),
            &bindings,
            definition_range,
            expected_name,
        )
    }
}
