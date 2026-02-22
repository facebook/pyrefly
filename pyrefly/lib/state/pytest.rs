/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::binding::binding::KeyClass;
use crate::binding::bindings::Bindings;
use crate::binding::pytest::PytestBindingInfo;

#[derive(Clone)]
pub(crate) struct PytestFixtureDefinition {
    pub(crate) name: Name,
    pub(crate) range: TextRange,
    pub(crate) docstring_range: Option<TextRange>,
}

fn is_pytest_test_function(function_def: &StmtFunctionDef, class_context: Option<bool>) -> bool {
    let name = function_def.name.id.as_str();
    if !name.starts_with("test_") {
        return false;
    }
    match class_context {
        Some(true) | None => true,
        Some(false) => false,
    }
}

fn is_pytest_test_class(class_def: &StmtClassDef) -> bool {
    class_def.name.id.as_str().starts_with("Test")
}

fn class_key_for_definition(
    bindings: &Bindings,
    class_def: &StmtClassDef,
) -> Option<Idx<KeyClass>> {
    bindings.key_to_idx_hashed_opt(Hashed::new(&KeyClass(ShortIdentifier::new(
        &class_def.name,
    ))))
}

fn is_pytest_fixture_function(
    function_def: &StmtFunctionDef,
    class_key: Option<Idx<KeyClass>>,
    pytest_info: &PytestBindingInfo,
) -> bool {
    pytest_info.is_fixture_definition(&function_def.name, class_key.as_ref())
}

fn maybe_add_fixture_param_reference(
    param: &ParameterWithDefault,
    fixture_name: &Name,
    references: &mut Vec<TextRange>,
) {
    let param_name = param.name();
    if param_name != "self" && param_name != "cls" && param_name.id() == fixture_name {
        references.push(param_name.range());
    }
}

fn collect_fixture_param_ranges_from_parameters(
    parameters: &Parameters,
    fixture_name: &Name,
    references: &mut Vec<TextRange>,
) {
    for param in parameters
        .posonlyargs
        .iter()
        .chain(parameters.args.iter())
        .chain(parameters.kwonlyargs.iter())
    {
        maybe_add_fixture_param_reference(param, fixture_name, references);
    }
}

fn collect_pytest_fixture_parameter_ranges(
    stmts: &[Stmt],
    bindings: &Bindings,
    pytest_info: &PytestBindingInfo,
    fixture_name: &Name,
    references: &mut Vec<TextRange>,
    class_key: Option<Idx<KeyClass>>,
    class_context: Option<bool>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if is_pytest_fixture_function(function_def, class_key, pytest_info)
                    || is_pytest_test_function(function_def, class_context)
                {
                    collect_fixture_param_ranges_from_parameters(
                        &function_def.parameters,
                        fixture_name,
                        references,
                    );
                }
            }
            Stmt::ClassDef(class_def) => {
                let nested_class_key = class_key_for_definition(bindings, class_def);
                let class_is_test = is_pytest_test_class(class_def);
                collect_pytest_fixture_parameter_ranges(
                    &class_def.body,
                    bindings,
                    pytest_info,
                    fixture_name,
                    references,
                    nested_class_key,
                    Some(class_is_test),
                );
            }
            _ => {}
        }
    }
}

fn collect_pytest_fixture_definitions_for_name(
    stmts: &[Stmt],
    bindings: &Bindings,
    pytest_info: &PytestBindingInfo,
    fixture_name: &Name,
    out: &mut Vec<PytestFixtureDefinition>,
    class_key: Option<Idx<KeyClass>>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if function_def.name.id() == fixture_name
                    && is_pytest_fixture_function(function_def, class_key, pytest_info)
                {
                    out.push(PytestFixtureDefinition {
                        name: function_def.name.id.clone(),
                        range: function_def.name.range,
                        docstring_range: Docstring::range_from_stmts(&function_def.body),
                    });
                }
            }
            Stmt::ClassDef(class_def) => {
                let nested_class_key = class_key_for_definition(bindings, class_def);
                collect_pytest_fixture_definitions_for_name(
                    &class_def.body,
                    bindings,
                    pytest_info,
                    fixture_name,
                    out,
                    nested_class_key,
                );
            }
            _ => {}
        }
    }
}

fn has_fixture_definition_at_range(
    stmts: &[Stmt],
    bindings: &Bindings,
    pytest_info: &PytestBindingInfo,
    fixture_name: &Name,
    definition_range: TextRange,
    class_key: Option<Idx<KeyClass>>,
) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if function_def.name.id() == fixture_name
                    && function_def.name.range == definition_range
                    && is_pytest_fixture_function(function_def, class_key, pytest_info)
                {
                    return true;
                }
            }
            Stmt::ClassDef(class_def) => {
                let nested_class_key = class_key_for_definition(bindings, class_def);
                if has_fixture_definition_at_range(
                    &class_def.body,
                    bindings,
                    pytest_info,
                    fixture_name,
                    definition_range,
                    nested_class_key,
                ) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Returns fixture definitions for a parameter when the cursor is in a pytest test/fixture.
pub(crate) fn pytest_fixture_definitions_for_parameter(
    module: &ModModule,
    bindings: &Bindings,
    identifier: &Identifier,
    covering_nodes: &[AnyNodeRef],
) -> Option<Vec<PytestFixtureDefinition>> {
    let pytest_info = bindings.pytest_info()?;
    let function_def = covering_nodes.iter().find_map(|node| match node {
        AnyNodeRef::StmtFunctionDef(stmt) => Some(stmt),
        _ => None,
    })?;
    let class_def = covering_nodes.iter().find_map(|node| match node {
        AnyNodeRef::StmtClassDef(stmt) => Some(stmt),
        _ => None,
    });
    let class_is_test = class_def.map(|def| is_pytest_test_class(def));
    let class_key = class_def.and_then(|def| class_key_for_definition(bindings, def));

    if !is_pytest_fixture_function(function_def, class_key, pytest_info)
        && !is_pytest_test_function(function_def, class_is_test)
    {
        return None;
    }

    let mut matches = Vec::new();
    collect_pytest_fixture_definitions_for_name(
        &module.body,
        bindings,
        pytest_info,
        identifier.id(),
        &mut matches,
        None,
    );
    if matches.is_empty() {
        return None;
    }
    Some(matches)
}

/// Returns all parameter ranges that reference the fixture definition in this module.
pub(crate) fn pytest_fixture_parameter_references(
    module: &ModModule,
    bindings: &Bindings,
    definition_range: TextRange,
    expected_name: &Name,
) -> Option<Vec<TextRange>> {
    let pytest_info = bindings.pytest_info()?;
    if !has_fixture_definition_at_range(
        &module.body,
        bindings,
        pytest_info,
        expected_name,
        definition_range,
        None,
    ) {
        return None;
    }
    let mut references = Vec::new();
    collect_pytest_fixture_parameter_ranges(
        &module.body,
        bindings,
        pytest_info,
        expected_name,
        &mut references,
        None,
        None,
    );
    Some(references)
}
