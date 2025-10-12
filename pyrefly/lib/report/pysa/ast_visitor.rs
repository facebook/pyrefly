/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::class::Class;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::identifier::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyDecoratedFunction;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::should_export_function;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;

pub enum Scope {
    TopLevel,
    ExportedFunction {
        function_id: FunctionId,
        function_name: Name,
        #[allow(dead_code)]
        location: TextRange,
        #[allow(dead_code)]
        decorated_function: DecoratedFunction,
    },
    ExportedClass {
        #[allow(dead_code)]
        class_id: ClassId,
        #[allow(dead_code)]
        class_name: Name,
        #[allow(dead_code)]
        location: TextRange,
        #[allow(dead_code)]
        class: Class,
    },
    NonExportedFunction {
        #[allow(dead_code)]
        function_name: Name,
        #[allow(dead_code)]
        location: TextRange,
    },
    NonExportedClass {
        #[allow(dead_code)]
        class_name: Name,
        #[allow(dead_code)]
        location: TextRange,
    },
    FunctionDecorators,
    FunctionTypeParams,
    FunctionParameters,
    FunctionReturnAnnotation,
    ClassDecorators,
    ClassTypeParams,
    ClassArguments,
}

pub struct Scopes {
    stack: Vec<Scope>,
}

impl Scopes {
    pub fn current_exported_function(
        &self,
        module_id: ModuleId,
        module_name: ModuleName,
        include_top_level: bool,
        include_class_top_level: bool,
        include_decorators_in_decorated_definition: bool,
    ) -> Option<FunctionRef> {
        let mut iterator = self.stack.iter().rev();
        loop {
            match iterator.next().unwrap() {
                Scope::TopLevel => {
                    if include_top_level {
                        return Some(FunctionRef {
                            module_id,
                            module_name,
                            function_id: FunctionId::ModuleTopLevel,
                            function_name: Name::from("$toplevel"),
                        });
                    } else {
                        return None;
                    }
                }
                Scope::ExportedFunction {
                    function_id,
                    function_name,
                    ..
                } => {
                    return Some(FunctionRef {
                        module_id,
                        module_name,
                        function_id: function_id.clone(),
                        function_name: function_name.clone(),
                    });
                }
                Scope::NonExportedFunction { .. } => {
                    return None;
                }
                Scope::ExportedClass { class_id, .. } => {
                    if include_class_top_level {
                        return Some(FunctionRef {
                            module_id,
                            module_name,
                            function_id: FunctionId::ClassTopLevel {
                                class_id: *class_id,
                            },
                            function_name: Name::from("$class_toplevel"),
                        });
                    } else {
                        return None;
                    }
                }
                Scope::NonExportedClass { .. } => {
                    return None;
                }
                Scope::FunctionTypeParams
                | Scope::FunctionParameters
                | Scope::FunctionReturnAnnotation
                | Scope::ClassTypeParams
                | Scope::ClassArguments => {
                    // These are not true "semantic" scopes.
                    // We need to skip the parent scope, which is the wrapping function/class scope.
                    iterator.next().unwrap();
                }
                Scope::FunctionDecorators | Scope::ClassDecorators => {
                    if !include_decorators_in_decorated_definition {
                        iterator.next().unwrap();
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeId(TextRange);

impl ScopeId {
    pub fn top_level() -> Self {
        ScopeId(TextRange::default())
    }

    pub fn from_scopes(scopes: &Scopes) -> Self {
        let mut iterator = scopes.stack.iter().rev();
        loop {
            match iterator.next().unwrap() {
                Scope::TopLevel => return ScopeId::top_level(),
                Scope::ExportedFunction { location, .. } => return ScopeId(*location),
                Scope::ExportedClass { location, .. } => return ScopeId(*location),
                Scope::NonExportedFunction { location, .. } => return ScopeId(*location),
                Scope::NonExportedClass { location, .. } => return ScopeId(*location),
                Scope::FunctionDecorators
                | Scope::FunctionTypeParams
                | Scope::FunctionParameters
                | Scope::FunctionReturnAnnotation
                | Scope::ClassDecorators
                | Scope::ClassTypeParams
                | Scope::ClassArguments => {
                    // These are not true "semantic" scopes.
                    // We need to skip the parent scope, which is the wrapping function/class scope.
                    iterator.next().unwrap();
                }
            }
        }
    }
}

pub trait AstScopedVisitor {
    fn visit_statement(&mut self, _stmt: &Stmt, _scopes: &Scopes) {}
    fn visit_expression(&mut self, _expr: &Expr, _scopes: &Scopes) {}
    fn enter_function_scope(
        &mut self,
        _function_def: &StmtFunctionDef,
        _scopes_in_function: &Scopes,
    ) {
    }
    fn exit_function_scope(
        &mut self,
        _function_def: &StmtFunctionDef,
        _scopes_outside_function: &Scopes,
    ) {
    }
    fn enter_class_scope(&mut self, _class_def: &StmtClassDef, _scopes_in_class: &Scopes) {}
    fn exit_class_scope(&mut self, _function_def: &StmtClassDef, _scopes_outside_class: &Scopes) {}
    fn enter_toplevel_scope(&mut self, _ast: &ModModule, _scopes_in_toplevel: &Scopes) {}
    fn exit_toplevel_scope(&mut self, _ast: &ModModule, _scopes_in_toplevel: &Scopes) {}
    fn on_scope_update(&mut self, _scopes: &Scopes) {}
}

fn visit_expression<V: AstScopedVisitor>(expr: &Expr, visitor: &mut V, scopes: &mut Scopes) {
    visitor.visit_expression(expr, scopes);
    expr.recurse(&mut |e| visit_expression(e, visitor, scopes));
}

fn visit_statement<V: AstScopedVisitor>(
    stmt: &Stmt,
    visitor: &mut V,
    scopes: &mut Scopes,
    module_context: &ModuleContext,
) {
    visitor.visit_statement(stmt, scopes);

    match stmt {
        Stmt::FunctionDef(function_def) => {
            let key = KeyDecoratedFunction(ShortIdentifier::new(&function_def.name));
            let function_scope = if let Some(idx) = module_context
                .bindings
                .key_to_idx_hashed_opt(Hashed::new(&key))
            {
                let decorated_function = DecoratedFunction::from_bindings_answers(
                    idx,
                    &module_context.bindings,
                    &module_context.answers,
                );
                if should_export_function(&decorated_function, module_context) {
                    Scope::ExportedFunction {
                        function_id: FunctionId::Function {
                            location: PysaLocation::new(
                                module_context
                                    .module_info
                                    .display_range(function_def.identifier()),
                            ),
                        },
                        location: function_def.identifier().range(),
                        function_name: function_def.name.id().clone(),
                        decorated_function,
                    }
                } else {
                    Scope::NonExportedFunction {
                        function_name: function_def.name.id().clone(),
                        location: function_def.identifier().range(),
                    }
                }
            } else {
                Scope::NonExportedFunction {
                    function_name: function_def.name.id().clone(),
                    location: function_def.identifier().range(),
                }
            };
            scopes.stack.push(function_scope);
            visitor.enter_function_scope(function_def, scopes);
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::FunctionDecorators);
            visitor.on_scope_update(scopes);
            function_def
                .decorator_list
                .recurse(&mut |e| visit_expression(e, visitor, scopes));
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::FunctionTypeParams);
            visitor.on_scope_update(scopes);
            function_def.type_params.recurse(&mut |e| {
                visit_expression(e, visitor, scopes);
            });
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::FunctionParameters);
            visitor.on_scope_update(scopes);
            function_def.parameters.recurse(&mut |e| {
                visit_expression(e, visitor, scopes);
            });
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::FunctionReturnAnnotation);
            visitor.on_scope_update(scopes);
            function_def.returns.recurse(&mut |e| {
                visit_expression(e, visitor, scopes);
            });
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            for stmt in &function_def.body {
                visit_statement(stmt, visitor, scopes, module_context);
            }

            scopes.stack.pop();
            visitor.exit_function_scope(function_def, scopes);
            visitor.on_scope_update(scopes);
        }
        Stmt::ClassDef(class_def) => {
            let key = KeyClass(ShortIdentifier::new(&class_def.name));
            let class_scope = if let Some(idx) = module_context
                .bindings
                .key_to_idx_hashed_opt(Hashed::new(&key))
            {
                let class = module_context
                    .answers
                    .get_idx(idx)
                    .unwrap()
                    .0
                    .dupe()
                    .unwrap();
                Scope::ExportedClass {
                    class_id: ClassId::from_class(&class),
                    class_name: class_def.name.id().clone(),
                    location: class_def.identifier().range(),
                    class,
                }
            } else {
                Scope::NonExportedClass {
                    class_name: class_def.name.id().clone(),
                    location: class_def.identifier().range(),
                }
            };
            scopes.stack.push(class_scope);
            visitor.enter_class_scope(class_def, scopes);
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::ClassDecorators);
            visitor.on_scope_update(scopes);
            class_def
                .decorator_list
                .recurse(&mut |e| visit_expression(e, visitor, scopes));
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::ClassTypeParams);
            visitor.on_scope_update(scopes);
            class_def.type_params.recurse(&mut |e| {
                visit_expression(e, visitor, scopes);
            });
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::ClassArguments);
            visitor.on_scope_update(scopes);
            class_def.arguments.recurse(&mut |e| {
                visit_expression(e, visitor, scopes);
            });
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            for stmt in &class_def.body {
                visit_statement(stmt, visitor, scopes, module_context);
            }

            scopes.stack.pop();
            visitor.exit_class_scope(class_def, scopes);
            visitor.on_scope_update(scopes);
        }
        _ => {
            // Use the ruff python ast visitor to find the first reachable statements and expressions from this statement.
            // By overriding visit_stmt and visit_expr, the visitor doesn't automatically visit nested statements/expressions.
            struct X<'a, V> {
                visitor: &'a mut V,
                scopes: &'a mut Scopes,
                module_context: &'a ModuleContext<'a>,
            }
            impl<'v, 'e, V: AstScopedVisitor>
                ruff_python_ast::visitor::source_order::SourceOrderVisitor<'e> for X<'v, V>
            {
                fn visit_stmt(&mut self, stmt: &'e Stmt) {
                    visit_statement(stmt, self.visitor, self.scopes, self.module_context);
                }
                fn visit_expr(&mut self, expr: &'e Expr) {
                    visit_expression(expr, self.visitor, self.scopes);
                }
            }
            ruff_python_ast::visitor::source_order::walk_stmt(
                &mut X {
                    visitor,
                    scopes,
                    module_context,
                },
                stmt,
            );
        }
    }
}

pub fn visit_module_ast<V: AstScopedVisitor>(
    visitor: &mut V,
    module_context: &ModuleContext,
) -> Scopes {
    let mut scopes = Scopes {
        stack: vec![Scope::TopLevel],
    };
    visitor.enter_toplevel_scope(&module_context.ast, &scopes);
    visitor.on_scope_update(&scopes);
    for stmt in &module_context.ast.body {
        visit_statement(stmt, visitor, &mut scopes, module_context);
    }
    visitor.exit_toplevel_scope(&module_context.ast, &scopes);
    scopes
}
