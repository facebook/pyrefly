/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_text_size::Ranged;

use crate::report::pysa::FunctionTypeOfExpressions;
use crate::report::pysa::LocalTypeId;
use crate::report::pysa::ast_visitor::AstScopedVisitor;
use crate::report::pysa::ast_visitor::ExportClassDecorators;
use crate::report::pysa::ast_visitor::ExportDefaultArguments;
use crate::report::pysa::ast_visitor::ExportFunctionDecorators;
use crate::report::pysa::ast_visitor::ScopeExportedFunctionFlags;
use crate::report::pysa::ast_visitor::Scopes;
use crate::report::pysa::ast_visitor::visit_module_ast;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::types::PysaType;

/// Builder for constructing per-function type data with deduplication.
struct FunctionTypeOfExpressionsBuilder {
    type_table: Vec<PysaType>,
    type_to_id: HashMap<PysaType, LocalTypeId>,
    locations: HashMap<PysaLocation, LocalTypeId>,
}

impl FunctionTypeOfExpressionsBuilder {
    fn new() -> Self {
        Self {
            type_table: Vec::new(),
            type_to_id: HashMap::new(),
            locations: HashMap::new(),
        }
    }

    /// Insert a type, deduplicating against previously seen types.
    /// Returns the LocalTypeId for the type.
    fn insert_type(&mut self, pysa_type: PysaType) -> LocalTypeId {
        if let Some(&id) = self.type_to_id.get(&pysa_type) {
            return id;
        }
        let id = LocalTypeId(self.type_table.len() as u32);
        self.type_to_id.insert(pysa_type.clone(), id);
        self.type_table.push(pysa_type);
        id
    }

    /// Add a location-to-type mapping. Skips duplicates.
    fn add_location(&mut self, location: PysaLocation, pysa_type: PysaType) {
        if !self.locations.contains_key(&location) {
            let id = self.insert_type(pysa_type);
            self.locations.insert(location, id);
        }
    }

    fn build(self) -> FunctionTypeOfExpressions {
        FunctionTypeOfExpressions {
            type_table: self.type_table,
            locations: self.locations,
        }
    }
}

struct TypeOfExpressionVisitor<'a> {
    module_context: &'a ModuleContext<'a>,
    current_function: Option<FunctionId>,
    result: HashMap<FunctionId, FunctionTypeOfExpressionsBuilder>,
}

impl<'a> TypeOfExpressionVisitor<'a> {
    /// Export the type of a single expression, if it has one.
    fn maybe_export_type(&mut self, e: &Expr) {
        let function_id = match &self.current_function {
            Some(id) => id,
            None => return,
        };
        let range = e.range();
        if let Some(type_) = self
            .module_context
            .answers_context
            .answers
            .get_type_trace(range)
        {
            let location = PysaLocation::from_text_range(
                range,
                &self.module_context.answers_context.module_info,
            );
            let pysa_type = PysaType::from_type(&type_, self.module_context);
            self.result
                .entry(function_id.clone())
                .or_insert_with(FunctionTypeOfExpressionsBuilder::new)
                .add_location(location, pysa_type);
        }
    }
}

impl AstScopedVisitor for TypeOfExpressionVisitor<'_> {
    fn on_scope_update(&mut self, scopes: &Scopes) {
        self.current_function = scopes
            .current_exported_function(
                self.module_context.answers_context.module_id,
                self.module_context.answers_context.module_info.name(),
                &ScopeExportedFunctionFlags {
                    include_top_level: true,
                    include_class_top_level: true,
                    include_function_decorators: ExportFunctionDecorators::InParentScope,
                    include_class_decorators: ExportClassDecorators::InParentScope,
                    include_default_arguments: ExportDefaultArguments::InFunction,
                },
            )
            .map(|func_ref| func_ref.function_id);
        if let Some(function_id) = &self.current_function {
            // Always insert an empty entry for the function.
            // This way we can error on missing type-of-expressions in pysa.
            self.result
                .entry(function_id.clone())
                .or_insert_with(FunctionTypeOfExpressionsBuilder::new);
        }
    }

    /// We only export types for expressions that Pysa needs:
    /// - `Expr::Name`: simple variable references (e.g. `x`)
    /// - `Expr::Attribute`: the base of an attribute access (e.g. type of `x` in `x.foo`)
    /// - `Expr::Call`: each positional and keyword argument
    fn visit_expression(
        &mut self,
        expr: &Expr,
        _scopes: &Scopes,
        _parent_expression: Option<&Expr>,
        _current_statement: Option<&ruff_python_ast::Stmt>,
    ) {
        match expr {
            Expr::Name(_) => self.maybe_export_type(expr),
            Expr::Attribute(ExprAttribute { value, .. }) => self.maybe_export_type(value),
            Expr::Call(ExprCall { arguments, .. }) => {
                for arg in &arguments.args {
                    self.maybe_export_type(arg);
                }
                for keyword in &arguments.keywords {
                    self.maybe_export_type(&keyword.value);
                }
            }
            _ => {}
        }
    }

    fn visit_type_annotations() -> bool {
        false
    }
}

pub fn export_type_of_expressions(
    context: &ModuleContext,
) -> HashMap<FunctionId, FunctionTypeOfExpressions> {
    let mut visitor = TypeOfExpressionVisitor {
        module_context: context,
        current_function: None,
        result: HashMap::new(),
    };

    visit_module_ast(&mut visitor, context);

    visitor
        .result
        .into_iter()
        .map(|(id, builder)| (id, builder.build()))
        .collect()
}
