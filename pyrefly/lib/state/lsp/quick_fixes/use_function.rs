/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Alias;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprBoolOp;
use ruff_python_ast::ExprBooleanLiteral;
use ruff_python_ast::ExprBytesLiteral;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprEllipsisLiteral;
use ruff_python_ast::ExprIf;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNoneLiteral;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprSet;
use ruff_python_ast::ExprSlice;
use ruff_python_ast::ExprStarred;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::ModuleInfo;
use crate::state::ide::import_regular_import_edit;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_function::LocalRefactorCodeAction;

/// Builds use-function refactor actions for the supplied selection.
pub(crate) fn use_function_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    if transaction.is_third_party_module(&module_info, handle)
        && !transaction.is_source_file(&module_info, handle)
    {
        return None;
    }
    let ast = transaction.get_ast(handle)?;
    let selection_start = selection.start();
    let function_def = find_enclosing_top_level_function(ast.as_ref(), selection_start)?;
    if function_def.is_async || !function_def.decorator_list.is_empty() {
        return None;
    }
    let params = collect_supported_params(&function_def)?;
    let return_expr = extract_return_expr(&function_def)?;
    let mut param_set = HashSet::new();
    for param in &params {
        param_set.insert(param.clone());
    }
    let mut param_counts = HashMap::new();
    if !validate_return_expr(return_expr, &param_set, &mut param_counts) {
        return None;
    }
    if params
        .iter()
        .any(|param| param_counts.get(param).copied().unwrap_or(0) == 0)
    {
        return None;
    }
    let pattern = FunctionPattern {
        name: function_def.name.id.to_string(),
        params,
        param_set,
        param_counts,
        return_expr: return_expr.clone(),
        def_range: function_def.range(),
        module_name: handle.module(),
    };

    let mut edits: Vec<(Module, TextRange, String)> = Vec::new();
    let mut matched_any = false;

    for target_handle in transaction.handles() {
        let target_info = match transaction.get_module_info(&target_handle) {
            Some(info) => info,
            None => continue,
        };
        if transaction.is_third_party_module(&target_info, &target_handle)
            && !transaction.is_source_file(&target_info, &target_handle)
        {
            continue;
        }
        let target_ast = match transaction.get_ast(&target_handle) {
            Some(ast) => ast,
            None => continue,
        };
        let call_target =
            match call_target_for_module(&target_handle, target_ast.as_ref(), &pattern, handle) {
                Some(target) => target,
                None => continue,
            };
        let same_module = target_handle.module() == pattern.module_name;
        let min_start = if same_module {
            Some(pattern.def_range.end())
        } else {
            None
        };
        let matches = collect_matches(
            &pattern,
            target_ast.as_ref(),
            &target_info,
            min_start,
            if same_module {
                Some(pattern.def_range)
            } else {
                None
            },
        );
        if matches.is_empty() {
            continue;
        }
        if let Some((range, text)) = call_target.import_edit {
            edits.push((target_info.dupe(), range, text));
        }
        for matched in matches {
            let call_text = build_call_text(
                &call_target.callee,
                &pattern.params,
                &matched.bindings,
                &target_info,
            )?;
            edits.push((target_info.dupe(), matched.range, call_text));
        }
        matched_any = true;
    }

    if !matched_any {
        return None;
    }

    Some(vec![LocalRefactorCodeAction {
        title: format!("Use function `{}`", pattern.name),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

#[derive(Clone, Debug)]
struct FunctionPattern {
    name: String,
    params: Vec<String>,
    param_set: HashSet<String>,
    param_counts: HashMap<String, usize>,
    return_expr: Expr,
    def_range: TextRange,
    module_name: ModuleName,
}

#[derive(Clone, Debug)]
struct ArgBinding {
    range: TextRange,
    key: Option<String>,
}

#[derive(Clone, Debug)]
struct ExprMatch {
    range: TextRange,
    bindings: HashMap<String, ArgBinding>,
}

#[derive(Clone, Debug)]
struct CallTarget {
    callee: String,
    import_edit: Option<(TextRange, String)>,
}

fn find_enclosing_top_level_function(
    ast: &ModModule,
    position: TextSize,
) -> Option<StmtFunctionDef> {
    for stmt in &ast.body {
        if let Stmt::FunctionDef(function_def) = stmt
            && function_def.range().contains_inclusive(position)
        {
            return Some(function_def.clone());
        }
    }
    None
}

fn collect_supported_params(function_def: &StmtFunctionDef) -> Option<Vec<String>> {
    let parameters = &function_def.parameters;
    if parameters.vararg.is_some()
        || parameters.kwarg.is_some()
        || !parameters.kwonlyargs.is_empty()
    {
        return None;
    }
    let mut params = Vec::new();
    for param in &parameters.posonlyargs {
        params.push(param.name().id.to_string());
    }
    for param in &parameters.args {
        params.push(param.name().id.to_string());
    }
    Some(params)
}

fn extract_return_expr(function_def: &StmtFunctionDef) -> Option<&Expr> {
    let body_iter = function_def.body.iter();
    let mut body: Vec<&Stmt> = Vec::new();
    for stmt in body_iter {
        if body.is_empty() && is_docstring_stmt(stmt) {
            continue;
        }
        body.push(stmt);
    }
    if body.len() != 1 {
        return None;
    }
    match body[0] {
        Stmt::Return(ret) => ret.value.as_deref(),
        _ => None,
    }
}

fn validate_return_expr(
    expr: &Expr,
    param_set: &HashSet<String>,
    param_counts: &mut HashMap<String, usize>,
) -> bool {
    match expr {
        Expr::Name(ExprName { id, .. }) => {
            let name = id.as_str();
            if !param_set.contains(name) {
                return false;
            }
            *param_counts.entry(name.to_owned()).or_insert(0) += 1;
            true
        }
        Expr::BooleanLiteral(ExprBooleanLiteral { .. })
        | Expr::NumberLiteral(ExprNumberLiteral { .. })
        | Expr::StringLiteral(ExprStringLiteral { .. })
        | Expr::BytesLiteral(ExprBytesLiteral { .. })
        | Expr::NoneLiteral(ExprNoneLiteral { .. })
        | Expr::EllipsisLiteral(ExprEllipsisLiteral { .. }) => true,
        Expr::Attribute(ExprAttribute { value, .. }) => {
            validate_return_expr(value, param_set, param_counts)
        }
        Expr::Subscript(ExprSubscript { value, slice, .. }) => {
            validate_return_expr(value, param_set, param_counts)
                && validate_return_expr(slice, param_set, param_counts)
        }
        Expr::Slice(ExprSlice {
            lower, upper, step, ..
        }) => {
            validate_opt_expr(lower, param_set, param_counts)
                && validate_opt_expr(upper, param_set, param_counts)
                && validate_opt_expr(step, param_set, param_counts)
        }
        Expr::Tuple(ExprTuple { elts, .. })
        | Expr::List(ExprList { elts, .. })
        | Expr::Set(ExprSet { elts, .. }) => elts
            .iter()
            .all(|expr| validate_return_expr(expr, param_set, param_counts)),
        Expr::Dict(ExprDict { items, .. }) => items.iter().all(|item| {
            item.key
                .as_ref()
                .map(|key| validate_return_expr(key, param_set, param_counts))
                .unwrap_or(true)
                && validate_return_expr(&item.value, param_set, param_counts)
        }),
        Expr::UnaryOp(ExprUnaryOp { operand, .. }) => {
            validate_return_expr(operand, param_set, param_counts)
        }
        Expr::BinOp(ExprBinOp { left, right, .. }) => {
            validate_return_expr(left, param_set, param_counts)
                && validate_return_expr(right, param_set, param_counts)
        }
        Expr::Compare(ExprCompare {
            left, comparators, ..
        }) => {
            if comparators.len() != 1 {
                return false;
            }
            validate_return_expr(left, param_set, param_counts)
                && comparators
                    .iter()
                    .all(|expr| validate_return_expr(expr, param_set, param_counts))
        }
        Expr::Starred(ExprStarred { value, .. }) => {
            validate_return_expr(value, param_set, param_counts)
        }
        _ => false,
    }
}

fn validate_opt_expr(
    expr: &Option<Box<Expr>>,
    param_set: &HashSet<String>,
    param_counts: &mut HashMap<String, usize>,
) -> bool {
    expr.as_ref()
        .map(|expr| validate_return_expr(expr, param_set, param_counts))
        .unwrap_or(true)
}

fn call_target_for_module(
    target_handle: &Handle,
    ast: &ModModule,
    pattern: &FunctionPattern,
    function_handle: &Handle,
) -> Option<CallTarget> {
    if target_handle.module() == pattern.module_name {
        return Some(CallTarget {
            callee: pattern.name.clone(),
            import_edit: None,
        });
    }
    let module_str = pattern.module_name.as_str();
    if let Some(callee) = find_imported_function_name(ast, module_str, &pattern.name) {
        return Some(CallTarget {
            callee,
            import_edit: None,
        });
    }
    if let Some(module_alias) = find_imported_module_alias(ast, module_str) {
        return Some(CallTarget {
            callee: format!("{module_alias}.{}", pattern.name),
            import_edit: None,
        });
    }
    let module_binding = module_str
        .split('.')
        .next()
        .unwrap_or(module_str)
        .to_owned();
    if module_has_top_level_binding(ast, &module_binding) {
        return None;
    }
    let (position, insert_text) = import_regular_import_edit(ast, function_handle.dupe());
    let range = TextRange::at(position, TextSize::new(0));
    Some(CallTarget {
        callee: format!("{module_str}.{}", pattern.name),
        import_edit: Some((range, insert_text)),
    })
}

fn find_imported_function_name(
    ast: &ModModule,
    module_name: &str,
    function_name: &str,
) -> Option<String> {
    for stmt in &ast.body {
        let Stmt::ImportFrom(StmtImportFrom {
            module,
            names,
            level,
            ..
        }) = stmt
        else {
            continue;
        };
        if *level != 0 {
            continue;
        }
        let Some(module) = module else {
            continue;
        };
        if module.as_str() != module_name {
            continue;
        }
        for alias in names {
            if alias.name.as_str() == function_name {
                return Some(alias_name(alias));
            }
        }
    }
    None
}

fn find_imported_module_alias(ast: &ModModule, module_name: &str) -> Option<String> {
    for stmt in &ast.body {
        let Stmt::Import(StmtImport { names, .. }) = stmt else {
            continue;
        };
        for alias in names {
            if alias.name.as_str() == module_name {
                return Some(alias_name(alias));
            }
        }
    }
    None
}

fn alias_name(alias: &Alias) -> String {
    alias
        .asname
        .as_ref()
        .unwrap_or(&alias.name)
        .as_str()
        .to_owned()
}

fn module_has_top_level_binding(ast: &ModModule, name: &str) -> bool {
    for stmt in &ast.body {
        match stmt {
            Stmt::FunctionDef(def) => {
                if def.name.id.as_str() == name {
                    return true;
                }
            }
            Stmt::ClassDef(def) => {
                if def.name.id.as_str() == name {
                    return true;
                }
            }
            Stmt::Assign(assign) => {
                if assign
                    .targets
                    .iter()
                    .any(|target| matches!(target, Expr::Name(ExprName { id, .. }) if id.as_str() == name))
                {
                    return true;
                }
            }
            Stmt::AnnAssign(ann) => {
                if matches!(&*ann.target, Expr::Name(ExprName { id, .. }) if id.as_str() == name) {
                    return true;
                }
            }
            Stmt::AugAssign(aug) => {
                if matches!(&*aug.target, Expr::Name(ExprName { id, .. }) if id.as_str() == name) {
                    return true;
                }
            }
            Stmt::Import(StmtImport { names, .. }) => {
                for alias in names {
                    if bound_name_for_import(alias) == name {
                        return true;
                    }
                }
            }
            Stmt::ImportFrom(StmtImportFrom { names, .. }) => {
                for alias in names {
                    let bound = alias.asname.as_ref().unwrap_or(&alias.name).as_str();
                    if bound == name {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn bound_name_for_import(alias: &Alias) -> &str {
    if let Some(asname) = &alias.asname {
        return asname.as_str();
    }
    alias.name.as_str().split('.').next().unwrap_or("")
}

fn collect_matches(
    pattern: &FunctionPattern,
    ast: &ModModule,
    module_info: &ModuleInfo,
    min_start: Option<TextSize>,
    skip_range: Option<TextRange>,
) -> Vec<ExprMatch> {
    let mut matches = Vec::new();
    ast.visit(&mut |expr: &Expr| {
        if let Some(skip_range) = skip_range
            && skip_range.contains_range(expr.range())
        {
            return;
        }
        if let Some(min_start) = min_start
            && expr.range().start() < min_start
        {
            return;
        }
        let mut bindings = HashMap::new();
        if !match_expr(
            &pattern.return_expr,
            expr,
            &pattern.param_set,
            &pattern.param_counts,
            &mut bindings,
            module_info,
        ) {
            return;
        }
        let expr_range = expr.range();
        let mut should_add = true;
        matches.retain(|existing: &ExprMatch| {
            if existing.range.contains_range(expr_range) {
                should_add = false;
                true
            } else {
                !expr_range.contains_range(existing.range)
            }
        });
        if should_add {
            matches.push(ExprMatch {
                range: expr_range,
                bindings,
            });
        }
    });
    matches
}

fn match_expr(
    pattern: &Expr,
    target: &Expr,
    param_set: &HashSet<String>,
    param_counts: &HashMap<String, usize>,
    bindings: &mut HashMap<String, ArgBinding>,
    module_info: &ModuleInfo,
) -> bool {
    if let Expr::Name(ExprName { id, .. }) = pattern {
        let name = id.as_str();
        if param_set.contains(name) {
            let duplicate = param_counts.get(name).copied().unwrap_or(0) > 1;
            if let Some(existing) = bindings.get(name) {
                if !duplicate {
                    return true;
                }
                let Some(key) = argument_key(target, module_info) else {
                    return false;
                };
                return existing.key.as_ref() == Some(&key);
            }
            if duplicate {
                let Some(key) = argument_key(target, module_info) else {
                    return false;
                };
                bindings.insert(
                    name.to_owned(),
                    ArgBinding {
                        range: target.range(),
                        key: Some(key),
                    },
                );
            } else {
                bindings.insert(
                    name.to_owned(),
                    ArgBinding {
                        range: target.range(),
                        key: None,
                    },
                );
            }
            return true;
        }
    }
    match (pattern, target) {
        (
            Expr::BooleanLiteral(ExprBooleanLiteral { value: lhs, .. }),
            Expr::BooleanLiteral(ExprBooleanLiteral { value: rhs, .. }),
        ) => lhs == rhs,
        (
            Expr::NumberLiteral(ExprNumberLiteral { value: lhs, .. }),
            Expr::NumberLiteral(ExprNumberLiteral { value: rhs, .. }),
        ) => lhs == rhs,
        (
            Expr::StringLiteral(ExprStringLiteral { value: lhs, .. }),
            Expr::StringLiteral(ExprStringLiteral { value: rhs, .. }),
        ) => lhs == rhs,
        (
            Expr::BytesLiteral(ExprBytesLiteral { value: lhs, .. }),
            Expr::BytesLiteral(ExprBytesLiteral { value: rhs, .. }),
        ) => lhs == rhs,
        (Expr::NoneLiteral(ExprNoneLiteral { .. }), Expr::NoneLiteral(ExprNoneLiteral { .. })) => {
            true
        }
        (
            Expr::EllipsisLiteral(ExprEllipsisLiteral { .. }),
            Expr::EllipsisLiteral(ExprEllipsisLiteral { .. }),
        ) => true,
        (
            Expr::Attribute(ExprAttribute {
                value: lhs_value,
                attr: lhs_attr,
                ..
            }),
            Expr::Attribute(ExprAttribute {
                value: rhs_value,
                attr: rhs_attr,
                ..
            }),
        ) => {
            lhs_attr.id == rhs_attr.id
                && match_expr(
                    lhs_value,
                    rhs_value,
                    param_set,
                    param_counts,
                    bindings,
                    module_info,
                )
        }
        (
            Expr::Subscript(ExprSubscript {
                value: lhs_value,
                slice: lhs_slice,
                ..
            }),
            Expr::Subscript(ExprSubscript {
                value: rhs_value,
                slice: rhs_slice,
                ..
            }),
        ) => {
            match_expr(
                lhs_value,
                rhs_value,
                param_set,
                param_counts,
                bindings,
                module_info,
            ) && match_expr(
                lhs_slice,
                rhs_slice,
                param_set,
                param_counts,
                bindings,
                module_info,
            )
        }
        (
            Expr::Slice(ExprSlice {
                lower: lhs_lower,
                upper: lhs_upper,
                step: lhs_step,
                ..
            }),
            Expr::Slice(ExprSlice {
                lower: rhs_lower,
                upper: rhs_upper,
                step: rhs_step,
                ..
            }),
        ) => {
            match_opt_expr(
                lhs_lower.as_deref(),
                rhs_lower.as_deref(),
                param_set,
                param_counts,
                bindings,
                module_info,
            ) && match_opt_expr(
                lhs_upper.as_deref(),
                rhs_upper.as_deref(),
                param_set,
                param_counts,
                bindings,
                module_info,
            ) && match_opt_expr(
                lhs_step.as_deref(),
                rhs_step.as_deref(),
                param_set,
                param_counts,
                bindings,
                module_info,
            )
        }
        (
            Expr::Tuple(ExprTuple { elts: lhs_elts, .. }),
            Expr::Tuple(ExprTuple { elts: rhs_elts, .. }),
        )
        | (
            Expr::List(ExprList { elts: lhs_elts, .. }),
            Expr::List(ExprList { elts: rhs_elts, .. }),
        )
        | (Expr::Set(ExprSet { elts: lhs_elts, .. }), Expr::Set(ExprSet { elts: rhs_elts, .. })) => {
            lhs_elts.len() == rhs_elts.len()
                && lhs_elts.iter().zip(rhs_elts.iter()).all(|(lhs, rhs)| {
                    match_expr(lhs, rhs, param_set, param_counts, bindings, module_info)
                })
        }
        (
            Expr::Dict(ExprDict {
                items: lhs_items, ..
            }),
            Expr::Dict(ExprDict {
                items: rhs_items, ..
            }),
        ) => {
            lhs_items.len() == rhs_items.len()
                && lhs_items.iter().zip(rhs_items.iter()).all(|(lhs, rhs)| {
                    match_opt_expr(
                        lhs.key.as_ref(),
                        rhs.key.as_ref(),
                        param_set,
                        param_counts,
                        bindings,
                        module_info,
                    ) && match_expr(
                        &lhs.value,
                        &rhs.value,
                        param_set,
                        param_counts,
                        bindings,
                        module_info,
                    )
                })
        }
        (
            Expr::UnaryOp(ExprUnaryOp {
                op: lhs_op,
                operand: lhs_operand,
                ..
            }),
            Expr::UnaryOp(ExprUnaryOp {
                op: rhs_op,
                operand: rhs_operand,
                ..
            }),
        ) => {
            lhs_op == rhs_op
                && match_expr(
                    lhs_operand,
                    rhs_operand,
                    param_set,
                    param_counts,
                    bindings,
                    module_info,
                )
        }
        (
            Expr::BinOp(ExprBinOp {
                op: lhs_op,
                left: lhs_left,
                right: lhs_right,
                ..
            }),
            Expr::BinOp(ExprBinOp {
                op: rhs_op,
                left: rhs_left,
                right: rhs_right,
                ..
            }),
        ) => {
            lhs_op == rhs_op
                && match_expr(
                    lhs_left,
                    rhs_left,
                    param_set,
                    param_counts,
                    bindings,
                    module_info,
                )
                && match_expr(
                    lhs_right,
                    rhs_right,
                    param_set,
                    param_counts,
                    bindings,
                    module_info,
                )
        }
        (
            Expr::BoolOp(ExprBoolOp {
                op: lhs_op,
                values: lhs_values,
                ..
            }),
            Expr::BoolOp(ExprBoolOp {
                op: rhs_op,
                values: rhs_values,
                ..
            }),
        ) => {
            lhs_op == rhs_op
                && lhs_values.len() == rhs_values.len()
                && lhs_values.iter().zip(rhs_values.iter()).all(|(lhs, rhs)| {
                    match_expr(lhs, rhs, param_set, param_counts, bindings, module_info)
                })
        }
        (
            Expr::Compare(ExprCompare {
                left: lhs_left,
                ops: lhs_ops,
                comparators: lhs_comparators,
                ..
            }),
            Expr::Compare(ExprCompare {
                left: rhs_left,
                ops: rhs_ops,
                comparators: rhs_comparators,
                ..
            }),
        ) => {
            lhs_ops == rhs_ops
                && lhs_comparators.len() == rhs_comparators.len()
                && match_expr(
                    lhs_left,
                    rhs_left,
                    param_set,
                    param_counts,
                    bindings,
                    module_info,
                )
                && lhs_comparators
                    .iter()
                    .zip(rhs_comparators.iter())
                    .all(|(lhs, rhs)| {
                        match_expr(lhs, rhs, param_set, param_counts, bindings, module_info)
                    })
        }
        (
            Expr::Starred(ExprStarred {
                value: lhs_value, ..
            }),
            Expr::Starred(ExprStarred {
                value: rhs_value, ..
            }),
        ) => match_expr(
            lhs_value,
            rhs_value,
            param_set,
            param_counts,
            bindings,
            module_info,
        ),
        _ => false,
    }
}

fn match_opt_expr(
    left: Option<&Expr>,
    right: Option<&Expr>,
    param_set: &HashSet<String>,
    param_counts: &HashMap<String, usize>,
    bindings: &mut HashMap<String, ArgBinding>,
    module_info: &ModuleInfo,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(lhs), Some(rhs)) => {
            match_expr(lhs, rhs, param_set, param_counts, bindings, module_info)
        }
        _ => false,
    }
}

fn argument_key(expr: &Expr, module_info: &ModuleInfo) -> Option<String> {
    match expr {
        Expr::Name(ExprName { id, .. }) => Some(id.as_str().to_owned()),
        Expr::BooleanLiteral(ExprBooleanLiteral { .. })
        | Expr::NumberLiteral(ExprNumberLiteral { .. })
        | Expr::StringLiteral(ExprStringLiteral { .. })
        | Expr::BytesLiteral(ExprBytesLiteral { .. })
        | Expr::NoneLiteral(ExprNoneLiteral { .. })
        | Expr::EllipsisLiteral(ExprEllipsisLiteral { .. }) => {
            Some(module_info.code_at(expr.range()).to_owned())
        }
        _ => None,
    }
}

fn build_call_text(
    callee: &str,
    param_order: &[String],
    bindings: &HashMap<String, ArgBinding>,
    module_info: &ModuleInfo,
) -> Option<String> {
    let mut args = Vec::new();
    for param in param_order {
        let binding = bindings.get(param)?;
        let text = module_info.code_at(binding.range).to_owned();
        args.push(text);
    }
    Some(format!("{callee}({})", args.join(", ")))
}
