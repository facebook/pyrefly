/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::module::Module;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

#[derive(Debug, Clone)]
pub(crate) enum AttributeKind {
    Function,
    Class,
    Assignment,
}

#[derive(Debug, Clone)]
pub(crate) struct AttributeContext {
    pub(crate) parent_classes: Vec<ruff_python_ast::name::Name>,
    pub(crate) kind: AttributeKind,
}

impl AttributeContext {
    pub(crate) fn from_module(
        module: &Module,
        target_range: TextRange,
    ) -> Option<AttributeContext> {
        let ast = Ast::parse(module.contents(), module.source_type()).0;
        let mut parents = Vec::new();
        Self::from_body(ast.body.as_slice(), &mut parents, target_range)
    }

    fn from_body(
        body: &[Stmt],
        parents: &mut Vec<ruff_python_ast::name::Name>,
        target_range: TextRange,
    ) -> Option<AttributeContext> {
        for stmt in body {
            match stmt {
                Stmt::ClassDef(class_def) => {
                    parents.push(class_def.name.id.clone());
                    if let Some(ctx) = Self::from_body(&class_def.body, parents, target_range) {
                        return Some(ctx);
                    }
                    parents.pop();
                    if class_def.range().contains_range(target_range) {
                        return Some(AttributeContext {
                            parent_classes: parents.clone(),
                            kind: AttributeKind::Class,
                        });
                    }
                }
                Stmt::FunctionDef(_) if stmt.range().contains_range(target_range) => {
                    return Some(AttributeContext {
                        parent_classes: parents.clone(),
                        kind: AttributeKind::Function,
                    });
                }
                Stmt::Assign(_) | Stmt::AnnAssign(_)
                    if stmt.range().contains_range(target_range) =>
                {
                    return Some(AttributeContext {
                        parent_classes: parents.clone(),
                        kind: AttributeKind::Assignment,
                    });
                }
                _ => {}
            }
        }
        None
    }
}

pub(crate) fn expr_matches_name(expr: &Expr, attr_name: &ruff_python_ast::name::Name) -> bool {
    match expr {
        Expr::Name(name) => &name.id == attr_name,
        _ => false,
    }
}

pub(crate) fn definition_from_executable_ast(
    ast: &ModModule,
    context: &AttributeContext,
    attr_name: &ruff_python_ast::name::Name,
) -> Option<(TextRange, Option<TextRange>)> {
    let mut body = ast.body.as_slice();
    for class_name in &context.parent_classes {
        let class_def = body.iter().find_map(|stmt| match stmt {
            Stmt::ClassDef(class_def) if &class_def.name.id == class_name => Some(class_def),
            _ => None,
        })?;
        body = class_def.body.as_slice();
    }
    match context.kind {
        AttributeKind::Function => definition_from_function(body, attr_name),
        AttributeKind::Class => definition_from_class(body, attr_name),
        AttributeKind::Assignment => definition_from_assignment(body, attr_name),
    }
}

fn definition_from_function(
    body: &[Stmt],
    attr_name: &ruff_python_ast::name::Name,
) -> Option<(TextRange, Option<TextRange>)> {
    for stmt in body {
        if let Stmt::FunctionDef(func_def) = stmt
            && &func_def.name.id == attr_name
        {
            let docstring_range = Docstring::range_from_stmts(&func_def.body);
            return Some((func_def.name.range, docstring_range));
        }
    }
    None
}

fn definition_from_class(
    body: &[Stmt],
    attr_name: &ruff_python_ast::name::Name,
) -> Option<(TextRange, Option<TextRange>)> {
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt
            && &class_def.name.id == attr_name
        {
            let docstring_range = Docstring::range_from_stmts(&class_def.body);
            return Some((class_def.name.range, docstring_range));
        }
    }
    None
}

fn definition_from_assignment(
    body: &[Stmt],
    attr_name: &ruff_python_ast::name::Name,
) -> Option<(TextRange, Option<TextRange>)> {
    for (idx, stmt) in body.iter().enumerate() {
        let (matches, def_range) = match stmt {
            Stmt::Assign(assign) => {
                let target_opt = assign
                    .targets
                    .iter()
                    .find(|target| expr_matches_name(target, attr_name));
                (target_opt.is_some(), target_opt.map(|t| t.range()))
            }
            Stmt::AnnAssign(assign) => {
                let matches = expr_matches_name(assign.target.as_ref(), attr_name);
                (matches, Some(assign.target.range()))
            }
            _ => (false, None),
        };
        if matches {
            let docstring_range = if let Some(Stmt::Expr(expr_stmt)) = body.get(idx + 1)
                && matches!(expr_stmt.value.as_ref(), Expr::StringLiteral(_))
            {
                Some(expr_stmt.range())
            } else {
                None
            };
            return Some((def_range?, docstring_range));
        }
    }
    None
}
