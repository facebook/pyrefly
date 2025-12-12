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
use crate::report::pysa::function::should_export_decorated_function;
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

pub enum ExportFunctionDecorators {
    #[allow(dead_code)]
    InDecoratedFunction,
    InParentScope,
    InDecoratedTarget,
    #[allow(dead_code)]
    Ignore,
}

pub enum ExportClassDecorators {
    #[allow(dead_code)]
    InDecoratedClassTopLevel,
    InParentScope,
    #[allow(dead_code)]
    Ignore,
}

pub enum ExportDefaultArguments {
    InFunction,
    InParentScope,
    #[allow(dead_code)]
    Ignore,
}

pub struct ScopeExportedFunctionFlags {
    pub include_top_level: bool,
    pub include_class_top_level: bool,
    pub include_function_decorators: ExportFunctionDecorators,
    pub include_class_decorators: ExportClassDecorators,
    pub include_default_arguments: ExportDefaultArguments,
}

impl Scopes {
    pub fn current_exported_function(
        &self,
        module_id: ModuleId,
        module_name: ModuleName,
        flags: &ScopeExportedFunctionFlags,
    ) -> Option<FunctionRef> {
        Self::current_exported_function_impl(self.stack.iter().rev(), module_id, module_name, flags)
    }

    fn current_exported_function_impl<'a>(
        mut iterator: impl Iterator<Item = &'a Scope>,
        module_id: ModuleId,
        module_name: ModuleName,
        flags: &ScopeExportedFunctionFlags,
    ) -> Option<FunctionRef> {
        match iterator.next().unwrap() {
            Scope::TopLevel => {
                if flags.include_top_level {
                    Some(FunctionRef {
                        module_id,
                        module_name,
                        function_id: FunctionId::ModuleTopLevel,
                        function_name: Name::from("$toplevel"),
                    })
                } else {
                    None
                }
            }
            Scope::ExportedFunction {
                function_id,
                function_name,
                ..
            } => Some(FunctionRef {
                module_id,
                module_name,
                function_id: function_id.clone(),
                function_name: function_name.clone(),
            }),
            Scope::NonExportedFunction { .. } => None,
            Scope::ExportedClass { class_id, .. } => {
                if flags.include_class_top_level {
                    Some(FunctionRef {
                        module_id,
                        module_name,
                        function_id: FunctionId::ClassTopLevel {
                            class_id: *class_id,
                        },
                        function_name: Name::from("$class_toplevel"),
                    })
                } else {
                    None
                }
            }
            Scope::NonExportedClass { .. } => None,
            Scope::FunctionParameters => match flags.include_default_arguments {
                ExportDefaultArguments::InFunction => {
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                }
                ExportDefaultArguments::InParentScope => {
                    iterator.next().unwrap();
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                }
                ExportDefaultArguments::Ignore => None,
            },
            Scope::FunctionTypeParams
            | Scope::FunctionReturnAnnotation
            | Scope::ClassTypeParams
            | Scope::ClassArguments => {
                // These are not true "semantic" scopes.
                // We need to skip the parent scope, which is the wrapping function/class scope.
                iterator.next().unwrap();
                Self::current_exported_function_impl(iterator, module_id, module_name, flags)
            }
            Scope::ClassDecorators => match flags.include_class_decorators {
                ExportClassDecorators::InDecoratedClassTopLevel => {
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                }
                ExportClassDecorators::InParentScope => {
                    iterator.next().unwrap();
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                }
                ExportClassDecorators::Ignore => None,
            },
            Scope::FunctionDecorators => match flags.include_function_decorators {
                ExportFunctionDecorators::InDecoratedFunction => {
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                }
                ExportFunctionDecorators::InParentScope => {
                    iterator.next().unwrap();
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                }
                ExportFunctionDecorators::InDecoratedTarget => {
                    Self::current_exported_function_impl(iterator, module_id, module_name, flags)
                        .and_then(|function_ref| function_ref.get_decorated_target())
                }
                ExportFunctionDecorators::Ignore => None,
            },
        }
    }
}

pub trait AstScopedVisitor {
    fn visit_statement(&mut self, stmt: &Stmt, scopes: &Scopes) {
        let _ = stmt;
        let _ = scopes;
    }
    fn visit_expression(
        &mut self,
        expr: &Expr,
        scopes: &Scopes,
        parent_expression: Option<&Expr>,
        current_statement: Option<&Stmt>,
    ) {
        let _ = expr;
        let _ = scopes;
        let _ = parent_expression;
        let _ = current_statement;
    }
    fn enter_function_scope(
        &mut self,
        function_def: &StmtFunctionDef,
        scopes_in_function: &Scopes,
    ) {
        let _ = function_def;
        let _ = scopes_in_function;
    }
    fn exit_function_scope(
        &mut self,
        function_def: &StmtFunctionDef,
        scopes_outside_function: &Scopes,
    ) {
        let _ = function_def;
        let _ = scopes_outside_function;
    }
    fn enter_class_scope(&mut self, class_def: &StmtClassDef, scopes_in_class: &Scopes) {
        let _ = class_def;
        let _ = scopes_in_class;
    }
    fn exit_class_scope(&mut self, function_def: &StmtClassDef, scopes_outside_class: &Scopes) {
        let _ = function_def;
        let _ = scopes_outside_class;
    }
    fn enter_toplevel_scope(&mut self, ast: &ModModule, scopes_in_toplevel: &Scopes) {
        let _ = ast;
        let _ = scopes_in_toplevel;
    }
    fn exit_toplevel_scope(&mut self, ast: &ModModule, scopes_in_toplevel: &Scopes) {
        let _ = ast;
        let _ = scopes_in_toplevel;
    }
    fn on_scope_update(&mut self, scopes: &Scopes) {
        let _ = scopes;
    }
    fn visit_type_annotations() -> bool;
}

fn visit_statement<V: AstScopedVisitor>(
    stmt: &Stmt,
    visitor: &mut V,
    scopes: &mut Scopes,
    module_context: &ModuleContext,
) {
    // By overriding visit_stmt and visit_expr, the visitor doesn't automatically visit nested statements/expressions.
    struct CustomSourceOrderVisitor<'a, V> {
        visitor: &'a mut V,
        scopes: &'a mut Scopes,
        module_context: &'a ModuleContext<'a>,
        parent_expression: Option<&'a Expr>,
        current_statement: Option<&'a Stmt>,
    }
    impl<'v, 'e: 'v, V: AstScopedVisitor>
        ruff_python_ast::visitor::source_order::SourceOrderVisitor<'e>
        for CustomSourceOrderVisitor<'v, V>
    {
        fn visit_stmt(&mut self, stmt: &'e Stmt) {
            visit_statement(stmt, self.visitor, self.scopes, self.module_context);
        }
        fn visit_expr(&mut self, expr: &'e Expr) {
            self.visitor.visit_expression(
                expr,
                self.scopes,
                self.parent_expression,
                self.current_statement,
            );
            let current_parent_expression = self.parent_expression;
            self.parent_expression = Some(expr);
            ruff_python_ast::visitor::source_order::walk_expr(self, expr);
            self.parent_expression = current_parent_expression;
        }
        fn visit_annotation(&mut self, expr: &'e Expr) {
            if V::visit_type_annotations() {
                ruff_python_ast::visitor::source_order::walk_annotation(self, expr);
            }
        }
    }

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
                if should_export_decorated_function(&decorated_function, module_context) {
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
            visitor.on_scope_update(scopes);
            visitor.enter_function_scope(function_def, scopes);

            if !function_def.decorator_list.is_empty() {
                scopes.stack.push(Scope::FunctionDecorators);
                visitor.on_scope_update(scopes);
                function_def.decorator_list.iter().for_each(&mut |e| {
                    ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_decorator(
                        &mut CustomSourceOrderVisitor {
                            visitor,
                            scopes,
                            module_context,
                            parent_expression: None,
                            current_statement: Some(stmt),
                        },
                        e,
                    )
                });
                scopes.stack.pop();
                visitor.on_scope_update(scopes);
            }

            if let Some(type_params) = &function_def.type_params {
                scopes.stack.push(Scope::FunctionTypeParams);
                visitor.on_scope_update(scopes);
                ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_type_params(
                    &mut CustomSourceOrderVisitor {
                        visitor,
                        scopes,
                        module_context,
                        parent_expression: None,
                        current_statement: Some(stmt),
                    },
                    type_params,
                );
                scopes.stack.pop();
                visitor.on_scope_update(scopes);
            }

            scopes.stack.push(Scope::FunctionParameters);
            visitor.on_scope_update(scopes);
            ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_parameters(
                &mut CustomSourceOrderVisitor {
                    visitor,
                    scopes,
                    module_context,
                    parent_expression: None,
                    current_statement: Some(stmt),
                },
                &function_def.parameters,
            );
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            if function_def.returns.is_some() {
                scopes.stack.push(Scope::FunctionReturnAnnotation);
                visitor.on_scope_update(scopes);
                function_def.returns.iter().for_each(|return_annotation| {
                    ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_annotation(
                        &mut CustomSourceOrderVisitor {
                            visitor,
                            scopes,
                            module_context,
                            parent_expression: None,
                            current_statement: Some(stmt),
                        },
                        return_annotation,
                    );
                });
                scopes.stack.pop();
                visitor.on_scope_update(scopes);
            }

            for stmt in &function_def.body {
                visit_statement(stmt, visitor, scopes, module_context);
            }

            visitor.exit_function_scope(function_def, scopes);
            scopes.stack.pop();
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
            visitor.on_scope_update(scopes);
            visitor.enter_class_scope(class_def, scopes);

            scopes.stack.push(Scope::ClassDecorators);
            visitor.on_scope_update(scopes);
            class_def.decorator_list.iter().for_each(&mut |e| {
                ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_decorator(
                    &mut CustomSourceOrderVisitor {
                        visitor,
                        scopes,
                        module_context,
                        parent_expression: None,
                        current_statement: Some(stmt),
                    },
                    e,
                )
            });
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::ClassTypeParams);
            visitor.on_scope_update(scopes);
            if let Some(type_params) = &class_def.type_params {
                ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_type_params(
                    &mut CustomSourceOrderVisitor {
                        visitor,
                        scopes,
                        module_context,
                        parent_expression: None,
                        current_statement: Some(stmt),
                    },
                    type_params,
                );
            }
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            scopes.stack.push(Scope::ClassArguments);
            visitor.on_scope_update(scopes);
            if let Some(arguments) = &class_def.arguments {
                ruff_python_ast::visitor::source_order::SourceOrderVisitor::visit_arguments(
                    &mut CustomSourceOrderVisitor {
                        visitor,
                        scopes,
                        module_context,
                        parent_expression: None,
                        current_statement: Some(stmt),
                    },
                    arguments,
                );
            }
            scopes.stack.pop();
            visitor.on_scope_update(scopes);

            for stmt in &class_def.body {
                visit_statement(stmt, visitor, scopes, module_context);
            }

            visitor.exit_class_scope(class_def, scopes);
            scopes.stack.pop();
            visitor.on_scope_update(scopes);
        }
        _ => {
            // Use the ruff python ast visitor to find the first reachable statements and expressions from this statement.
            ruff_python_ast::visitor::source_order::walk_stmt(
                &mut CustomSourceOrderVisitor {
                    visitor,
                    scopes,
                    module_context,
                    parent_expression: None,
                    current_statement: Some(stmt),
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
