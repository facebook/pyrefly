/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::binding::binding::KeyClass;

// Tracks local names that refer to pytest so we can resolve @pytest.fixture and fixture aliases.
#[derive(Clone, Debug, Default)]
pub struct PytestAliases {
    pytest_module_aliases: Vec<String>,
    fixture_aliases: Vec<String>,
}

impl PytestAliases {
    pub fn from_module(module: &ModModule) -> Self {
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

    pub fn is_empty(&self) -> bool {
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

#[derive(Clone, Debug)]
pub struct PytestFixtureDefinitions {
    module: Option<ShortIdentifier>,
    classes: SmallMap<Idx<KeyClass>, ShortIdentifier>,
}

#[derive(Clone, Debug)]
pub struct PytestBindingInfo {
    aliases: PytestAliases,
    fixtures: SmallMap<Name, PytestFixtureDefinitions>,
}

impl PytestBindingInfo {
    pub fn from_module(module: &ModModule) -> Option<Self> {
        let aliases = PytestAliases::from_module(module);
        if aliases.is_empty() {
            return None;
        }
        Some(Self {
            aliases,
            fixtures: SmallMap::new(),
        })
    }

    pub fn aliases(&self) -> &PytestAliases {
        &self.aliases
    }

    pub fn add_fixture_definition(
        &mut self,
        name: Name,
        return_type_key: ShortIdentifier,
        class_key: Option<Idx<KeyClass>>,
    ) {
        let definitions = self
            .fixtures
            .entry(name)
            .or_insert_with(|| PytestFixtureDefinitions {
                module: None,
                classes: SmallMap::new(),
            });
        match class_key {
            Some(class_key) => {
                definitions.classes.insert(class_key, return_type_key);
            }
            None => {
                definitions.module = Some(return_type_key);
            }
        }
    }

    pub fn visible_fixture_class_key(
        &self,
        name: &Name,
        class_key: Option<&Idx<KeyClass>>,
    ) -> Option<Option<Idx<KeyClass>>> {
        let defs = self.fixtures.get(name)?;
        if let Some(class_key) = class_key
            && defs.classes.contains_key(class_key)
        {
            return Some(Some(*class_key));
        }
        defs.module.as_ref()?;
        Some(None)
    }

    pub fn is_fixture_definition(
        &self,
        func_name: &ruff_python_ast::Identifier,
        class_key: Option<&Idx<KeyClass>>,
    ) -> bool {
        let Some(defs) = self.fixtures.get(&func_name.id) else {
            return false;
        };
        let func_id = ShortIdentifier::new(func_name);
        match class_key {
            Some(class_key) => defs.classes.get(class_key) == Some(&func_id),
            None => defs.module.as_ref() == Some(&func_id),
        }
    }
}

pub fn is_pytest_fixture_function(function_def: &StmtFunctionDef, aliases: &PytestAliases) -> bool {
    function_def
        .decorator_list
        .iter()
        .any(|decorator| is_pytest_fixture_decorator(&decorator.expression, aliases))
}

pub fn is_pytest_parametrize_decorator(expr: &Expr, aliases: &PytestAliases) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(parametrize) = call.func.as_ref() else {
        return false;
    };
    if parametrize.attr.id() != "parametrize" {
        return false;
    }
    let Expr::Attribute(mark) = parametrize.value.as_ref() else {
        return false;
    };
    mark.attr.id() == "mark"
        && matches!(
            mark.value.as_ref(),
            Expr::Name(base) if aliases.is_pytest_module_alias(base.id())
        )
}

fn is_pytest_fixture_decorator(expr: &Expr, aliases: &PytestAliases) -> bool {
    match expr {
        Expr::Name(name) => aliases.is_fixture_alias(name.id()),
        Expr::Attribute(attr) => {
            attr.attr.id() == "fixture"
                && matches!(
                    attr.value.as_ref(),
                    Expr::Name(base) if aliases.is_pytest_module_alias(base.id())
                )
        }
        Expr::Call(call) => is_pytest_fixture_decorator(call.func.as_ref(), aliases),
        _ => false,
    }
}
