/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::docstring::Docstring;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

// Tracks local names that refer to pytest so we can resolve @pytest.fixture and fixture aliases.
#[derive(Default)]
struct PytestAliases {
    pytest_module_aliases: Vec<String>,
    fixture_aliases: Vec<String>,
}

impl PytestAliases {
    fn from_module(module: &ModModule) -> Self {
        let mut aliases = Self::default();
        for stmt in &module.body {
            match stmt {
                Stmt::Import(import_stmt) => {
                    for alias in &import_stmt.names {
                        if alias.name.id == "pytest" {
                            let local_name = alias
                                .asname
                                .as_ref()
                                .map(|asname| asname.id.as_str())
                                .unwrap_or(alias.name.id.as_str());
                            aliases.pytest_module_aliases.push(local_name.to_owned());
                        }
                    }
                }
                Stmt::ImportFrom(StmtImportFrom {
                    module: Some(module),
                    names,
                    ..
                }) if module.id == "pytest" || module.id.starts_with("pytest.") => {
                    for alias in names {
                        if alias.name.id == "fixture" {
                            let local_name = alias
                                .asname
                                .as_ref()
                                .map(|asname| asname.id.as_str())
                                .unwrap_or(alias.name.id.as_str());
                            aliases.fixture_aliases.push(local_name.to_owned());
                        }
                    }
                }
                _ => {}
            }
        }
        aliases
    }

    fn is_empty(&self) -> bool {
        self.pytest_module_aliases.is_empty() && self.fixture_aliases.is_empty()
    }

    fn is_pytest_module_alias(&self, name: &Name) -> bool {
        self.pytest_module_aliases
            .iter()
            .any(|alias| alias == name.as_str())
    }

    fn is_fixture_alias(&self, name: &Name) -> bool {
        self.fixture_aliases
            .iter()
            .any(|alias| alias == name.as_str())
    }
}

#[derive(Clone)]
pub(crate) struct PytestFixtureDefinition {
    pub(crate) name: Name,
    pub(crate) range: TextRange,
    pub(crate) docstring_range: Option<TextRange>,
}

// Aggregates pytest-related data for a module to avoid repeated scans.
pub(crate) struct PytestModuleInfo {
    aliases: PytestAliases,
    fixtures: Vec<PytestFixtureDefinition>,
}

impl PytestModuleInfo {
    /// Collects pytest aliases and fixture definitions for a module.
    pub(crate) fn from_module(module: &ModModule) -> Option<Self> {
        let aliases = PytestAliases::from_module(module);
        if aliases.is_empty() {
            return None;
        }
        let mut fixtures = Vec::new();
        collect_pytest_fixture_definitions(&module.body, &aliases, &mut fixtures);
        Some(Self { aliases, fixtures })
    }

    fn aliases(&self) -> &PytestAliases {
        &self.aliases
    }

    fn fixture_definitions_for_name(&self, name: &Name) -> Vec<PytestFixtureDefinition> {
        self.fixtures
            .iter()
            .filter(|fixture| fixture.name == *name)
            .cloned()
            .collect()
    }

    fn has_fixture_definition(&self, name: &Name, range: TextRange) -> bool {
        self.fixtures
            .iter()
            .any(|fixture| fixture.name == *name && fixture.range == range)
    }
}

fn is_pytest_fixture_decorator(expr: &Expr, aliases: &PytestAliases) -> bool {
    match expr {
        Expr::Name(name) => aliases.is_fixture_alias(name.id()),
        Expr::Attribute(attr) => {
            attr.attr.id() == "fixture"
                && matches!(attr.value.as_ref(), Expr::Name(base) if aliases.is_pytest_module_alias(base.id()))
        }
        Expr::Call(call) => is_pytest_fixture_decorator(call.func.as_ref(), aliases),
        _ => false,
    }
}

fn is_pytest_fixture_function(function_def: &StmtFunctionDef, aliases: &PytestAliases) -> bool {
    function_def
        .decorator_list
        .iter()
        .any(|decorator| is_pytest_fixture_decorator(&decorator.expression, aliases))
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

fn is_pytest_fixture_or_test_function(
    function_def: &StmtFunctionDef,
    aliases: &PytestAliases,
    class_context: Option<bool>,
) -> bool {
    is_pytest_fixture_function(function_def, aliases)
        || is_pytest_test_function(function_def, class_context)
}

// Collects pytest fixture definitions in a module/class body.
fn collect_pytest_fixture_definitions(
    stmts: &[Stmt],
    aliases: &PytestAliases,
    fixtures: &mut Vec<PytestFixtureDefinition>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if is_pytest_fixture_function(function_def, aliases) {
                    fixtures.push(PytestFixtureDefinition {
                        name: function_def.name.id.clone(),
                        range: function_def.name.range,
                        docstring_range: Docstring::range_from_stmts(&function_def.body),
                    });
                }
            }
            Stmt::ClassDef(class_def) => {
                collect_pytest_fixture_definitions(&class_def.body, aliases, fixtures);
            }
            _ => {}
        }
    }
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

// Collects parameter ranges in test/fixture functions that reference a fixture by name.
fn collect_pytest_fixture_parameter_ranges(
    stmts: &[Stmt],
    aliases: &PytestAliases,
    fixture_name: &Name,
    references: &mut Vec<TextRange>,
    class_context: Option<bool>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                if is_pytest_fixture_or_test_function(function_def, aliases, class_context) {
                    collect_fixture_param_ranges_from_parameters(
                        &function_def.parameters,
                        fixture_name,
                        references,
                    );
                }
            }
            Stmt::ClassDef(class_def) => {
                let class_is_test = is_pytest_test_class(class_def);
                collect_pytest_fixture_parameter_ranges(
                    &class_def.body,
                    aliases,
                    fixture_name,
                    references,
                    Some(class_is_test),
                );
            }
            _ => {}
        }
    }
}

/// Returns fixture definitions for a parameter when the cursor is in a pytest test/fixture.
pub(crate) fn pytest_fixture_definitions_for_parameter(
    module_info: &PytestModuleInfo,
    identifier: &Identifier,
    covering_nodes: &[AnyNodeRef],
) -> Option<Vec<PytestFixtureDefinition>> {
    let function_def = covering_nodes.iter().find_map(|node| match node {
        AnyNodeRef::StmtFunctionDef(stmt) => Some(stmt),
        _ => None,
    })?;
    let class_context = covering_nodes
        .iter()
        .find_map(|node| match node {
            AnyNodeRef::StmtClassDef(stmt) => Some(stmt),
            _ => None,
        })
        .map(|class_def| is_pytest_test_class(*class_def));

    if !is_pytest_fixture_or_test_function(function_def, module_info.aliases(), class_context) {
        return None;
    }

    let matches = module_info.fixture_definitions_for_name(identifier.id());
    if matches.is_empty() {
        return None;
    }
    Some(matches)
}

/// Returns all parameter ranges that reference the fixture definition in this module.
pub(crate) fn pytest_fixture_parameter_references(
    module: &ModModule,
    module_info: &PytestModuleInfo,
    definition_range: TextRange,
    expected_name: &Name,
) -> Option<Vec<TextRange>> {
    if !module_info.has_fixture_definition(expected_name, definition_range) {
        return None;
    }
    let mut references = Vec::new();
    collect_pytest_fixture_parameter_ranges(
        &module.body,
        module_info.aliases(),
        expected_name,
        &mut references,
        None,
    );
    Some(references)
}
