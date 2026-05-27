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
use vec1::Vec1;

use super::DefinitionMetadata;
use super::FindDefinitionItemWithDocstring;
use crate::state::pytest::find_pytest_fixture_definitions_for_parameter;
use crate::state::pytest::find_pytest_fixture_parameter_references;
use crate::state::state::Transaction;

impl<'a> Transaction<'a> {
    /// Resolve a pytest fixture parameter to the fixture functions that can provide it.
    ///
    /// This runs during definition lookup. The common non-pytest path is cheap because we first
    /// ask bindings for pytest metadata, which is absent in modules that do not import pytest.
    pub(super) fn pytest_fixture_definitions_for_parameter(
        &self,
        handle: &Handle,
        identifier: &Identifier,
        covering_nodes: &[AnyNodeRef],
    ) -> Option<Vec1<FindDefinitionItemWithDocstring>> {
        let mod_module = self.get_ast(handle)?;
        let bindings = self.get_bindings(handle)?;
        let matches = find_pytest_fixture_definitions_for_parameter(
            mod_module.as_ref(),
            &bindings,
            identifier,
            covering_nodes,
        );
        let module_info = self.get_module_info(handle)?;
        let definitions = matches
            .into_iter()
            .map(|fixture| FindDefinitionItemWithDocstring {
                metadata: DefinitionMetadata::Variable(Some(SymbolKind::Function)),
                definition_range: fixture.range,
                module: module_info.clone(),
                docstring_range: fixture.docstring_range,
                display_name: Some(fixture.name.as_str().to_owned()),
            })
            .collect();
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
