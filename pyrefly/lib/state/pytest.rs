/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::docstring::Docstring;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

// Tracks local names that refer to pytest so we can resolve @pytest.fixture and fixture aliases.
#[derive(Default)]
pub(crate) struct PytestAliases {
    pytest_module_aliases: Vec<String>,
    fixture_aliases: Vec<String>,
}

impl PytestAliases {
    pub(crate) fn from_module(module: &ModModule) -> Self {
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

    pub(crate) fn is_empty(&self) -> bool {
        self.pytest_module_aliases.is_empty() && self.fixture_aliases.is_empty()
    }

    pub(crate) fn is_pytest_module_alias(&self, name: &Name) -> bool {
        self.pytest_module_aliases
            .iter()
            .any(|alias| alias == name.as_str())
    }

    pub(crate) fn is_fixture_alias(&self, name: &Name) -> bool {
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

pub(crate) fn is_pytest_fixture_function(
    function_def: &StmtFunctionDef,
    aliases: &PytestAliases,
) -> bool {
    function_def
        .decorator_list
        .iter()
        .any(|decorator| is_pytest_fixture_decorator(&decorator.expression, aliases))
}

pub(crate) fn is_pytest_test_function(
    function_def: &StmtFunctionDef,
    class_context: Option<bool>,
) -> bool {
    let name = function_def.name.id.as_str();
    if !name.starts_with("test_") {
        return false;
    }
    match class_context {
        Some(true) | None => true,
        Some(false) => false,
    }
}

pub(crate) fn is_pytest_test_class(class_def: &StmtClassDef) -> bool {
    class_def.name.id.as_str().starts_with("Test")
}

// Collects pytest fixture definitions in a module/class body.
pub(crate) fn collect_pytest_fixture_definitions(
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

// Collects parameter ranges in test/fixture functions that reference a fixture by name.
pub(crate) fn collect_pytest_fixture_parameter_ranges(
    stmts: &[Stmt],
    aliases: &PytestAliases,
    fixture_name: &Name,
    references: &mut Vec<TextRange>,
    class_context: Option<bool>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(function_def) => {
                let is_fixture = is_pytest_fixture_function(function_def, aliases);
                let is_test = is_pytest_test_function(function_def, class_context);
                if is_fixture || is_test {
                    for param in function_def.parameters.posonlyargs.iter() {
                        if param.name() != "self"
                            && param.name() != "cls"
                            && param.name().id() == fixture_name
                        {
                            references.push(param.name().range());
                        }
                    }
                    for param in function_def.parameters.args.iter() {
                        if param.name() != "self"
                            && param.name() != "cls"
                            && param.name().id() == fixture_name
                        {
                            references.push(param.name().range());
                        }
                    }
                    for param in function_def.parameters.kwonlyargs.iter() {
                        if param.name() != "self"
                            && param.name() != "cls"
                            && param.name().id() == fixture_name
                        {
                            references.push(param.name().range());
                        }
                    }
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
