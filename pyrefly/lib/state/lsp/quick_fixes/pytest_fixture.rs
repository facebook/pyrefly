/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Key;
use crate::state::ide::insert_import_edit;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::state::Transaction;

#[derive(Debug, Default)]
struct PytestAliases {
    pytest_modules: HashSet<Name>,
    fixture_names: HashSet<Name>,
}

impl PytestAliases {
    fn collect(ast: &ModModule) -> Self {
        let mut aliases = Self::default();
        for stmt in &ast.body {
            match stmt {
                Stmt::Import(import_stmt) => {
                    for alias in &import_stmt.names {
                        if alias.name.id.as_str() == "pytest" {
                            let name = alias.asname.as_ref().unwrap_or(&alias.name).id.clone();
                            aliases.pytest_modules.insert(name);
                        }
                    }
                }
                Stmt::ImportFrom(import_from) => {
                    if let Some(module) = &import_from.module
                        && module.id.as_str() == "pytest"
                    {
                        for alias in &import_from.names {
                            if alias.name.id.as_str() == "fixture" {
                                let name = alias.asname.as_ref().unwrap_or(&alias.name).id.clone();
                                aliases.fixture_names.insert(name);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        aliases
    }

    fn is_pytest_module_name(&self, name: &Name) -> bool {
        name.as_str() == "pytest" || self.pytest_modules.contains(name)
    }

    fn is_fixture_name(&self, name: &Name) -> bool {
        self.fixture_names.contains(name)
    }
}

fn is_pytest_fixture_decorator(expr: &Expr, aliases: &PytestAliases) -> bool {
    match expr {
        Expr::Call(ExprCall { func, .. }) => is_pytest_fixture_decorator(func, aliases),
        Expr::Name(name) => aliases.is_fixture_name(name.id()),
        Expr::Attribute(ExprAttribute { value, attr, .. }) => {
            attr.id.as_str() == "fixture"
                && matches!(value.as_ref(), Expr::Name(name) if aliases.is_pytest_module_name(name.id()))
        }
        _ => false,
    }
}

fn is_pytest_fixture_function(func: &StmtFunctionDef, aliases: &PytestAliases) -> bool {
    func.decorator_list
        .iter()
        .any(|decorator| is_pytest_fixture_decorator(&decorator.expression, aliases))
}

fn collect_fixture_functions<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a StmtFunctionDef>) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(func) => out.push(func),
            Stmt::ClassDef(class_def) => collect_fixture_functions(&class_def.body, out),
            _ => {}
        }
    }
}

fn should_skip_annotation(rendered: &str, ty: &Type) -> bool {
    ty.is_any()
        || rendered.contains("Any")
        || rendered.contains("Unknown")
        || rendered.contains("Never")
        || rendered.contains('@')
}

fn is_test_name(name: &Name) -> bool {
    name.as_str().starts_with("test_")
}

fn is_test_function(func: &StmtFunctionDef, class_name: Option<&Name>) -> bool {
    if !is_test_name(&func.name.id) {
        return false;
    }
    class_name.map_or(true, |name| name.as_str().starts_with("Test"))
}

fn collect_test_functions<'a>(
    stmts: &'a [Stmt],
    class_name: Option<&'a Name>,
    out: &mut Vec<&'a StmtFunctionDef>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(func) => {
                if is_test_function(func, class_name) {
                    out.push(func);
                }
            }
            Stmt::ClassDef(class_def) => {
                collect_test_functions(&class_def.body, Some(&class_def.name.id), out);
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
struct FixtureAnnotationEdit {
    def_range: TextRange,
    insert_range: TextRange,
    insert_text: String,
    import_edits: Vec<(TextSize, String)>,
}

#[derive(Debug)]
struct FixtureParamAnnotationEdit {
    insert_range: TextRange,
    insert_text: String,
    import_edits: Vec<(TextSize, String)>,
}

fn fixture_return_type(
    transaction: &Transaction<'_>,
    handle: &Handle,
    func: &StmtFunctionDef,
) -> Option<Type> {
    let return_key = Key::ReturnType(ShortIdentifier::new(&func.name));
    let mut ty = transaction.get_type(handle, &return_key)?;
    if func.is_async {
        if let Some(Some((_, _, return_ty))) =
            transaction.ad_hoc_solve(handle, |solver| solver.unwrap_coroutine(&ty))
        {
            ty = return_ty;
        }
    }
    if let Some(display_ty) =
        transaction.ad_hoc_solve(handle, |solver| solver.for_display(ty.clone()))
    {
        ty = display_ty;
    }
    let stdlib = transaction.get_stdlib(handle);
    Some(
        ty.promote_implicit_literals(&stdlib)
            .explicit_any()
            .clean_var(),
    )
}

fn fixture_types_for_module(transaction: &Transaction<'_>, handle: &Handle) -> HashMap<Name, Type> {
    let Some(ast) = transaction.get_ast(handle) else {
        return HashMap::new();
    };
    let aliases = PytestAliases::collect(&ast);
    let mut fixture_functions = Vec::new();
    collect_fixture_functions(&ast.body, &mut fixture_functions);
    let mut fixtures = HashMap::new();
    for func in fixture_functions {
        if !is_pytest_fixture_function(func, &aliases) {
            continue;
        }
        let Some(ty) = fixture_return_type(transaction, handle, func) else {
            continue;
        };
        let rendered = ty.as_lsp_string(LspDisplayMode::SignatureHelp);
        if should_skip_annotation(&rendered, &ty) {
            continue;
        }
        fixtures.entry(func.name.id.clone()).or_insert(ty);
    }
    fixtures
}

fn conftest_handles(transaction: &Transaction<'_>, handle: &Handle) -> Vec<Handle> {
    let module_path = handle.path();
    let Some(mut dir) = module_path.as_path().parent() else {
        return Vec::new();
    };
    let root = module_path
        .root_of(handle.module())
        .unwrap_or_else(|| dir.to_path_buf());
    let is_memory = matches!(module_path.details(), ModulePathDetails::Memory(_));
    let mut conftest_paths = Vec::new();
    loop {
        let conftest_pyi = dir.join("conftest.pyi");
        let conftest_py = dir.join("conftest.py");
        if is_memory {
            conftest_paths.push(ModulePath::memory(conftest_pyi.clone()));
            conftest_paths.push(ModulePath::memory(conftest_py.clone()));
        } else {
            if conftest_pyi.exists() {
                conftest_paths.push(ModulePath::filesystem(conftest_pyi));
            }
            if conftest_py.exists() {
                conftest_paths.push(ModulePath::filesystem(conftest_py));
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
    let mut handles = Vec::new();
    for path in conftest_paths {
        let config = transaction
            .config_finder()
            .python_file(ModuleNameWithKind::guaranteed(ModuleName::unknown()), &path);
        handles.push(config.handle_from_module_path(path));
    }
    handles
}

fn import_edits_for_type(
    transaction: &Transaction<'_>,
    ast: &ModModule,
    handle: &Handle,
    module_contents: &str,
    import_format: ImportFormat,
    ty: &Type,
) -> Vec<(TextSize, String)> {
    let mut import_edits = Vec::new();
    let mut seen_imports = HashSet::new();
    ty.universe(&mut |ty| {
        let Some(qname) = ty.qname() else {
            return;
        };
        if !qname.parent().is_toplevel() {
            return;
        }
        let module = qname.module_name();
        if module == handle.module() || module.as_str() == "builtins" {
            return;
        }
        let Some(handle_to_import_from) = transaction.import_handle(handle, module, None).finding()
        else {
            return;
        };
        let (position, insert_text, _) = insert_import_edit(
            ast,
            transaction.config_finder(),
            handle.dupe(),
            handle_to_import_from,
            qname.id().as_str(),
            import_format,
        );
        if module_contents.contains(&insert_text) {
            return;
        }
        if seen_imports.insert(insert_text.clone()) {
            import_edits.push((position, insert_text));
        }
    });
    import_edits
}

/// Builds code actions that add inferred return annotations to pytest fixtures.
pub(crate) fn pytest_fixture_type_annotation_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
    import_format: ImportFormat,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let ast = transaction.get_ast(handle)?;
    let module_info = transaction.get_module_info(handle)?;
    let module_contents = module_info.contents();
    let aliases = PytestAliases::collect(&ast);
    let mut fixture_functions = Vec::new();
    collect_fixture_functions(&ast.body, &mut fixture_functions);

    let mut candidates = Vec::new();
    for func in fixture_functions {
        if func.returns.is_some() {
            continue;
        }
        if !is_pytest_fixture_function(func, &aliases) {
            continue;
        }
        let Some(ty) = fixture_return_type(transaction, handle, func) else {
            continue;
        };
        let rendered = ty.as_lsp_string(LspDisplayMode::SignatureHelp);
        if should_skip_annotation(&rendered, &ty) {
            continue;
        }
        let insert_position = func.parameters.range.end();
        let insert_range = TextRange::at(insert_position, TextSize::new(0));
        let insert_text = format!(" -> {rendered}");
        let import_edits = import_edits_for_type(
            transaction,
            &ast,
            handle,
            module_contents.as_str(),
            import_format,
            &ty,
        );
        candidates.push(FixtureAnnotationEdit {
            def_range: func.range(),
            insert_range,
            insert_text,
            import_edits,
        });
    }

    if candidates.is_empty() {
        return None;
    }

    let module = module_info.dupe();
    let selection_matches_fixtures = candidates
        .iter()
        .any(|candidate| candidate.def_range.contains_range(selection));
    let mut actions = Vec::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.def_range.contains_range(selection))
    {
        let mut edits = Vec::new();
        edits.push((
            module.dupe(),
            candidate.insert_range,
            candidate.insert_text.clone(),
        ));
        for (position, text) in &candidate.import_edits {
            edits.push((
                module.dupe(),
                TextRange::at(*position, TextSize::new(0)),
                text.clone(),
            ));
        }
        actions.push(LocalRefactorCodeAction {
            title: "Add pytest fixture type annotation".to_owned(),
            edits,
            kind: lsp_types::CodeActionKind::QUICKFIX,
        });
        break;
    }

    if selection_matches_fixtures && candidates.len() > 1 {
        let mut edits = Vec::new();
        let mut seen_imports = HashSet::new();
        for candidate in &candidates {
            edits.push((
                module.dupe(),
                candidate.insert_range,
                candidate.insert_text.clone(),
            ));
            for (position, text) in &candidate.import_edits {
                if seen_imports.insert(text.clone()) {
                    edits.push((
                        module.dupe(),
                        TextRange::at(*position, TextSize::new(0)),
                        text.clone(),
                    ));
                }
            }
        }
        actions.push(LocalRefactorCodeAction {
            title: "Add all pytest fixture type annotations".to_owned(),
            edits,
            kind: lsp_types::CodeActionKind::QUICKFIX,
        });
    }

    let mut fixture_types = fixture_types_for_module(transaction, handle);
    for conftest_handle in conftest_handles(transaction, handle) {
        let conftest_types = fixture_types_for_module(transaction, &conftest_handle);
        for (name, ty) in conftest_types {
            fixture_types.entry(name).or_insert(ty);
        }
    }

    let mut test_functions = Vec::new();
    collect_test_functions(&ast.body, None, &mut test_functions);
    let mut fixture_param_candidates = Vec::new();
    for func in &test_functions {
        for param in func.parameters.iter() {
            let identifier = param.name();
            let name = identifier.id();
            let name_range = identifier.range();
            if !name_range.contains_range(selection) {
                continue;
            }
            if param.annotation().is_some() {
                continue;
            }
            if name.as_str() == "self" || name.as_str() == "cls" {
                continue;
            }
            let Some(ty) = fixture_types.get(name).cloned() else {
                continue;
            };
            let rendered = ty.as_lsp_string(LspDisplayMode::SignatureHelp);
            if should_skip_annotation(&rendered, &ty) {
                continue;
            }
            let insert_range = TextRange::at(name_range.end(), TextSize::new(0));
            let insert_text = format!(": {rendered}");
            let import_edits = import_edits_for_type(
                transaction,
                &ast,
                handle,
                module_contents.as_str(),
                import_format,
                &ty,
            );
            fixture_param_candidates.push(FixtureParamAnnotationEdit {
                insert_range,
                insert_text,
                import_edits,
            });
        }
    }

    let selection_matches_params = !fixture_param_candidates.is_empty();
    if let Some(candidate) = fixture_param_candidates.first() {
        let mut edits = Vec::new();
        edits.push((
            module.dupe(),
            candidate.insert_range,
            candidate.insert_text.clone(),
        ));
        for (position, text) in &candidate.import_edits {
            edits.push((
                module.dupe(),
                TextRange::at(*position, TextSize::new(0)),
                text.clone(),
            ));
        }
        actions.push(LocalRefactorCodeAction {
            title: "Add pytest fixture parameter type annotation".to_owned(),
            edits,
            kind: lsp_types::CodeActionKind::QUICKFIX,
        });
    }

    if selection_matches_params {
        let mut edits = Vec::new();
        let mut seen_imports = HashSet::new();
        let mut seen_params = HashSet::new();
        for func in &test_functions {
            for param in func.parameters.iter() {
                let identifier = param.name();
                let name = identifier.id();
                if param.annotation().is_some() {
                    continue;
                }
                if name.as_str() == "self" || name.as_str() == "cls" {
                    continue;
                }
                let Some(ty) = fixture_types.get(name).cloned() else {
                    continue;
                };
                let rendered = ty.as_lsp_string(LspDisplayMode::SignatureHelp);
                if should_skip_annotation(&rendered, &ty) {
                    continue;
                }
                if !seen_params.insert(identifier.range()) {
                    continue;
                }
                edits.push((
                    module.dupe(),
                    TextRange::at(identifier.range().end(), TextSize::new(0)),
                    format!(": {rendered}"),
                ));
                for (position, text) in import_edits_for_type(
                    transaction,
                    &ast,
                    handle,
                    module_contents.as_str(),
                    import_format,
                    &ty,
                ) {
                    if seen_imports.insert(text.clone()) {
                        edits.push((
                            module.dupe(),
                            TextRange::at(position, TextSize::new(0)),
                            text,
                        ));
                    }
                }
            }
        }
        if !edits.is_empty() {
            actions.push(LocalRefactorCodeAction {
                title: "Add all pytest fixture parameter type annotations".to_owned(),
                edits,
                kind: lsp_types::CodeActionKind::QUICKFIX,
            });
        }
    }

    if actions.is_empty() {
        None
    } else {
        Some(actions)
    }
}
