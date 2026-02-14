/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CallHierarchyIncomingCall;
use lsp_types::CallHierarchyItem;
use lsp_types::CallHierarchyOutgoingCall;
use lsp_types::SymbolKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::state::CancellableTransaction;

/// Finds a function definition at a specific position in an AST.
///
/// This is used by call hierarchy to identify the function being queried.
/// Returns None if no function is found at the position.
///
/// Uses `locate_node` for efficient single-pass traversal, consistent with
/// other LSP position-based queries. Handles nested functions and methods.
pub fn find_function_at_position_in_ast(
    ast: &ModModule,
    position: TextSize,
) -> Option<&StmtFunctionDef> {
    let covering_nodes = Ast::locate_node(ast, position);

    // Find the first StmtFunctionDef in the covering nodes
    // This returns the innermost function containing the position
    for node in covering_nodes.iter() {
        if let AnyNodeRef::StmtFunctionDef(func_def) = node {
            return Some(func_def);
        }
    }
    None
}

/// Creates call hierarchy information for the function containing a call site.
///
/// Given a call site position, finds the enclosing function and returns
/// information needed to represent it in the call hierarchy.
/// For module-level code (e.g., `if __name__ == "__main__":`), returns
/// the module name with `<module>` suffix.
pub fn find_containing_function_for_call(
    handle: &Handle,
    ast: &ModModule,
    position: TextSize,
) -> Option<(String, TextRange)> {
    let covering_nodes = Ast::locate_node(ast, position);

    // Look through the node chain for the containing function
    for (i, node) in covering_nodes.iter().enumerate() {
        if let AnyNodeRef::StmtFunctionDef(func_def) = node {
            // Check if this is a method (next node is a ClassDef)
            if let Some(AnyNodeRef::StmtClassDef(class_def)) = covering_nodes.get(i + 1) {
                let name = format!(
                    "{}.{}.{}",
                    handle.module(),
                    class_def.name.id,
                    func_def.name.id
                );
                return Some((name, func_def.name.range()));
            } else {
                // Top-level function
                let name = format!("{}.{}", handle.module(), func_def.name.id);
                return Some((name, func_def.name.range()));
            }
        }
    }

    // No containing function found - this is module-level code.
    // Use "<module>" as the caller name with the module's range.
    let name = format!("{}.<module>", handle.module());
    Some((name, ast.range()))
}

/// Converts raw incoming call data to LSP CallHierarchyIncomingCall items.
///
/// Takes the output from `find_global_incoming_calls_from_function_definition`
/// and transforms it into the LSP response format.
pub fn transform_incoming_calls(
    callers: Vec<(Module, Vec<(TextRange, String, TextRange)>)>,
    path_remapper: Option<&PathRemapper>,
) -> Vec<CallHierarchyIncomingCall> {
    let mut incoming_calls = Vec::new();
    for (caller_module, call_sites) in callers {
        for (call_range, caller_name, caller_def_range) in call_sites {
            let Some(caller_uri) = module_info_to_uri(&caller_module, path_remapper) else {
                continue;
            };

            let from = CallHierarchyItem {
                name: caller_name
                    .split('.')
                    .next_back()
                    .unwrap_or(&caller_name)
                    .to_owned(),
                kind: SymbolKind::FUNCTION,
                tags: None,
                detail: Some(caller_name),
                uri: caller_uri,
                range: caller_module.to_lsp_range(caller_def_range),
                selection_range: caller_module.to_lsp_range(caller_def_range),
                data: None,
            };

            incoming_calls.push(CallHierarchyIncomingCall {
                from,
                from_ranges: vec![caller_module.to_lsp_range(call_range)],
            });
        }
    }
    incoming_calls
}

/// Converts raw outgoing call data to LSP CallHierarchyOutgoingCall items.
///
/// Takes the output from `find_global_outgoing_calls_from_function_definition`
/// and transforms it into the LSP response format.
pub fn transform_outgoing_calls(
    callees: Vec<(Module, Vec<(TextRange, TextRange)>)>,
    source_module: &Module,
    fallback_uri: &lsp_types::Url,
) -> Vec<CallHierarchyOutgoingCall> {
    let mut outgoing_calls = Vec::new();
    for (target_module, calls) in callees {
        let target_uri = lsp_types::Url::from_file_path(target_module.path().as_path())
            .unwrap_or_else(|()| fallback_uri.clone());

        for (call_range, target_def_range) in calls {
            let target_name_short = target_module.code_at(target_def_range);
            let target_name = format!("{}.{}", target_module.name(), target_name_short);

            let to = CallHierarchyItem {
                name: target_name_short.to_owned(),
                kind: SymbolKind::FUNCTION,
                tags: None,
                detail: Some(target_name),
                uri: target_uri.clone(),
                range: target_module.to_lsp_range(target_def_range),
                selection_range: target_module.to_lsp_range(target_def_range),
                data: None,
            };

            outgoing_calls.push(CallHierarchyOutgoingCall {
                to,
                from_ranges: vec![source_module.to_lsp_range(call_range)],
            });
        }
    }
    outgoing_calls
}

/// Prepares a CallHierarchyItem for a function definition.
///
/// Creates the LSP CallHierarchyItem representation for a function,
/// including its name, fully qualified detail, and range information.
pub fn prepare_call_hierarchy_item(
    func_def: &StmtFunctionDef,
    module: &Module,
    uri: lsp_types::Url,
) -> CallHierarchyItem {
    let name = func_def.name.id.to_string();
    let detail = Some(format!("{}.{}", module.name(), name));

    CallHierarchyItem {
        name,
        kind: SymbolKind::FUNCTION,
        tags: None,
        detail,
        uri,
        range: module.to_lsp_range(func_def.range()),
        selection_range: module.to_lsp_range(func_def.name.range()),
        data: None,
    }
}

impl CancellableTransaction<'_> {
    /// Finds all incoming calls (functions that call this function) of a function across the entire codebase.
    ///
    /// This searches transitive reverse dependencies to find all locations where
    /// the target function is called. For each call site, it identifies the
    /// containing function.
    ///
    /// Returns a vector of tuples containing:
    /// - Module where the call occurs
    /// - Vector of (call_site_range, containing_function_name, containing_function_range)
    ///
    /// Returns Err if the request is canceled during execution.
    pub fn find_global_incoming_calls_from_function_definition(
        &mut self,
        sys_info: &SysInfo,
        definition_kind: DefinitionMetadata,
        target_definition: &TextRangeWithModule,
    ) -> Result<Vec<(Module, Vec<(TextRange, String, TextRange)>)>, Cancelled> {
        // Use process_rdeps_with_definition to find references and filter to call sites in a single pass
        let results = self.process_rdeps_with_definition(
            sys_info,
            target_definition,
            |transaction, handle, patched_definition| {
                let module_info = transaction.as_ref().get_module_info(handle)?;
                let ast = transaction.as_ref().get_ast(handle)?;

                let references = transaction
                    .as_ref()
                    .local_references_from_definition(
                        handle,
                        definition_kind.clone(),
                        patched_definition.range,
                        &patched_definition.module,
                    )
                    .unwrap_or_default();

                if references.is_empty() {
                    return None;
                }

                let ref_set: std::collections::HashSet<TextRange> =
                    references.into_iter().collect();

                let mut callers_in_file = Vec::new();

                /// Recursively collects Call expressions that match references to the target function.
                fn collect_calls_from_expr(
                    expr: &Expr,
                    ref_set: &std::collections::HashSet<TextRange>,
                    handle: &Handle,
                    ast: &ModModule,
                    callers: &mut Vec<(TextRange, String, TextRange)>,
                ) {
                    if let Expr::Call(call) = expr
                        && ref_set
                            .iter()
                            .any(|ref_range| call.func.range().contains(ref_range.start()))
                        && let Some((containing_func_name, containing_func_range)) =
                            find_containing_function_for_call(handle, ast, call.range().start())
                    {
                        callers.push((call.range(), containing_func_name, containing_func_range));
                    }
                    expr.recurse(&mut |child| {
                        collect_calls_from_expr(child, ref_set, handle, ast, callers)
                    });
                }

                ast.visit(&mut |expr| {
                    collect_calls_from_expr(expr, &ref_set, handle, &ast, &mut callers_in_file)
                });

                if callers_in_file.is_empty() {
                    None
                } else {
                    Some((module_info, callers_in_file))
                }
            },
        )?;

        Ok(results)
    }

    /// Finds outgoing calls (functions called by) of a function, resolving across files.
    ///
    /// Given a function definition, this method finds all Call expressions within
    /// that function's body and resolves each call target to its definition,
    /// which may be in other files.
    ///
    /// Returns a vector of tuples containing:
    /// - Target module (where the callee is defined)
    /// - Vector of (call_site_range, callee_definition_range)
    ///
    /// Results are grouped by target module for consistency with find_global_callers_from_definition.
    ///
    /// Returns Err if the request is canceled during execution.
    pub fn find_global_outgoing_calls_from_function_definition(
        &mut self,
        handle: &Handle,
        position: TextSize,
    ) -> Result<Vec<(Module, Vec<(TextRange, TextRange)>)>, Cancelled> {
        // Resolve cursor position to function definition (handles both definitions and call sites)
        let definitions =
            self.as_ref()
                .find_definition(handle, position, FindPreference::default());
        let Some(target_def) = definitions.into_iter().next() else {
            return Ok(Vec::new());
        };

        // Get handle for module where target function is defined (may differ from original handle)
        let target_handle = Handle::new(
            target_def.module.name(),
            target_def.module.path().dupe(),
            handle.sys_info().dupe(),
        );

        let Some(target_ast) = self.as_ref().get_ast(&target_handle) else {
            return Ok(Vec::new());
        };

        let Some(func_def) =
            find_function_at_position_in_ast(&target_ast, target_def.definition_range.start())
        else {
            return Ok(Vec::new());
        };

        let mut callees_by_module: std::collections::HashMap<
            ModulePath,
            (Module, Vec<(TextRange, TextRange)>),
        > = std::collections::HashMap::new();

        for stmt in &func_def.body {
            stmt.visit(&mut |expr| {
                if let Expr::Call(call) = expr {
                    let call_pos = call
                        .func
                        .range()
                        .end()
                        .checked_sub(TextSize::from(1))
                        .unwrap_or(call.func.range().start());

                    let definitions = self.as_ref().find_definition(
                        &target_handle,
                        call_pos,
                        FindPreference::default(),
                    );

                    for def in definitions {
                        let module_path = def.module.path().dupe();
                        callees_by_module
                            .entry(module_path)
                            .or_insert_with(|| (def.module.clone(), Vec::new()))
                            .1
                            .push((call.range(), def.definition_range));
                    }
                }
            });
        }

        let mut result: Vec<(Module, Vec<(TextRange, TextRange)>)> =
            callees_by_module.into_values().collect();
        result.sort_by_key(|(module, _)| module.path().dupe());

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_build::handle::Handle;
    use pyrefly_python::ast::Ast;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::sys_info::SysInfo;
    use ruff_python_ast::PySourceType;
    use ruff_text_size::TextSize;

    use super::find_containing_function_for_call;

    #[test]
    fn test_find_containing_function_for_call() {
        let source = r#"
def my_function():
    x = call()

class MyClass:
    def method(self):
        y = call()
"#;
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);
        let handle = Handle::new(
            ModuleName::from_str("test"),
            ModulePath::memory(PathBuf::from("test.py")),
            SysInfo::default(),
        );

        // Returns qualified name for top-level function
        let pos_in_func = TextSize::from(30);
        let (name, _) = find_containing_function_for_call(&handle, &ast, pos_in_func).unwrap();
        assert_eq!(name, "test.my_function");

        // Returns qualified name for class method
        let pos_in_method = TextSize::from(85);
        let (name, _) = find_containing_function_for_call(&handle, &ast, pos_in_method).unwrap();
        assert_eq!(name, "test.MyClass.method");
    }

    #[test]
    fn test_find_function_at_position_in_ast() {
        use super::find_function_at_position_in_ast;

        let source = r#"
def top_level():
    pass

class MyClass:
    def method(self):
        pass
"#;
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);

        // Finds top-level function
        let pos_in_func = TextSize::from(5);
        let result = find_function_at_position_in_ast(&ast, pos_in_func);
        assert_eq!(result.unwrap().name.id.as_str(), "top_level");

        // Finds class method
        let pos_in_method = TextSize::from(50);
        let result = find_function_at_position_in_ast(&ast, pos_in_method);
        assert_eq!(result.unwrap().name.id.as_str(), "method");
    }

    #[test]
    fn test_find_function_at_position_in_ast_returns_none_outside_functions() {
        use super::find_function_at_position_in_ast;

        let source = "x = 10\n";
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);

        let pos_outside = TextSize::from(0);
        let result = find_function_at_position_in_ast(&ast, pos_outside);
        assert!(result.is_none());
    }
}
