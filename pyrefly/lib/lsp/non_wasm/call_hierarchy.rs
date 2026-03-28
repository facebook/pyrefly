/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::sync::Arc;

use dupe::Dupe;
use lsp_types::CallHierarchyIncomingCall;
use lsp_types::CallHierarchyItem;
use lsp_types::CallHierarchyOutgoingCall;
use lsp_types::Range;
use lsp_types::SymbolKind;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::PySourceType;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use tracing::debug;
use vec1::Vec1;

use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::module_info_to_uri;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::state::CancellableTransaction;

pub struct CallerInfo {
    pub call_range: TextRange,
    pub name: String,
    pub full_range: TextRange,
    pub name_range: TextRange,
    pub kind: SymbolKind,
}

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
    module_name: ModuleName,
    ast: &ModModule,
    position: TextSize,
) -> (String, TextRange, TextRange, SymbolKind) {
    let covering_nodes = Ast::locate_node(ast, position);

    for (i, node) in covering_nodes.iter().enumerate() {
        if let AnyNodeRef::StmtFunctionDef(func_def) = node {
            if let Some(AnyNodeRef::StmtClassDef(class_def)) = covering_nodes.get(i + 1) {
                let name = format!("{}.{}.{}", module_name, class_def.name.id, func_def.name.id);
                return (
                    name,
                    func_def.range(),
                    func_def.name.range(),
                    SymbolKind::METHOD,
                );
            } else {
                let name = format!("{}.{}", module_name, func_def.name.id);
                return (
                    name,
                    func_def.range(),
                    func_def.name.range(),
                    SymbolKind::FUNCTION,
                );
            }
        }
    }

    let name = format!("{}.<module>", module_name);
    (name, ast.range(), ast.range(), SymbolKind::FUNCTION)
}

/// Converts raw incoming call data to LSP CallHierarchyIncomingCall items.
///
/// Takes the output from `find_global_incoming_calls_from_function_definition`
/// and transforms it into the LSP response format.
pub fn transform_incoming_calls(
    callers: Vec<(Module, Vec<CallerInfo>)>,
    path_remapper: Option<&PathRemapper>,
) -> Vec<CallHierarchyIncomingCall> {
    let mut incoming_calls = Vec::new();
    for (caller_module, call_sites) in callers {
        for caller in call_sites {
            let Some(caller_uri) = module_info_to_uri(&caller_module, path_remapper) else {
                continue;
            };

            let from = CallHierarchyItem {
                name: caller
                    .name
                    .split('.')
                    .next_back()
                    .unwrap_or(&caller.name)
                    .to_owned(),
                kind: caller.kind,
                tags: None,
                detail: Some(caller.name),
                uri: caller_uri,
                range: caller_module.to_lsp_range(caller.full_range),
                selection_range: caller_module.to_lsp_range(caller.name_range),
                data: None,
            };

            incoming_calls.push(CallHierarchyIncomingCall {
                from,
                from_ranges: vec![caller_module.to_lsp_range(caller.call_range)],
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

/// Checks whether a position is inside a `Call` expression's func range.
///
/// Returns the call's full range if the position is inside the func part
/// of a call expression, `None` otherwise. This filters out non-call
/// references (imports, type annotations, etc.).
fn find_enclosing_call_range(ast: &ModModule, position: TextSize) -> Option<TextRange> {
    for node in Ast::locate_node(ast, position) {
        if let AnyNodeRef::ExprCall(call) = node {
            if call.func.range().contains(position) {
                return Some(call.range());
            }
            return None;
        }
    }
    None
}

/// Derives a Python module name from a file path by walking up parent
/// directories looking for `__init__.py` package markers.
fn module_name_from_path(path: &Path) -> ModuleName {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("__unknown__");

    let mut parts = if stem == "__init__" {
        vec![]
    } else {
        vec![stem]
    };

    let mut current = path.parent();
    while let Some(dir) = current {
        if dir.join("__init__.py").exists() {
            if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
                parts.push(name);
                current = dir.parent();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    parts.reverse();

    if parts.is_empty() {
        ModuleName::from_str(stem)
    } else {
        ModuleName::from_string(parts.join("."))
    }
}

/// Converts external Glean references into LSP `CallHierarchyIncomingCall` items.
///
/// For each external file, reads the source, parses the AST, filters to actual
/// call expressions (via `find_enclosing_call_range`), and finds the enclosing
/// function (via `find_containing_function_for_call`).
pub fn convert_external_references_to_incoming_calls(
    external_refs: Vec<(Url, Vec<Range>)>,
) -> Vec<CallHierarchyIncomingCall> {
    let mut results = Vec::new();

    for (url, ranges) in external_refs {
        let Ok(path) = url.to_file_path() else {
            debug!("Skipping external reference: cannot convert URL to file path: {url}");
            continue;
        };
        let Ok(content) = std::fs::read_to_string(&path) else {
            debug!(
                "Skipping external reference: cannot read file: {}",
                path.display()
            );
            continue;
        };

        let module_name = module_name_from_path(&path);
        let source_type = if path.extension().is_some_and(|ext| ext == "pyi") {
            PySourceType::Stub
        } else {
            PySourceType::Python
        };
        let module_path = ModulePath::filesystem(path.to_path_buf());
        let module = Module::new(module_name, module_path, Arc::new(content));

        let (ast, _, _) = Ast::parse(module.contents(), source_type);

        for range in ranges {
            let position = module.from_lsp_position(range.start, None);

            let Some(call_range) = find_enclosing_call_range(&ast, position) else {
                continue;
            };

            let (caller_name, caller_full_range, caller_name_range, kind) =
                find_containing_function_for_call(module_name, &ast, position);

            results.push(CallHierarchyIncomingCall {
                from: CallHierarchyItem {
                    name: caller_name
                        .split('.')
                        .next_back()
                        .unwrap_or(&caller_name)
                        .to_owned(),
                    kind,
                    tags: None,
                    detail: Some(caller_name),
                    uri: url.clone(),
                    range: module.to_lsp_range(caller_full_range),
                    selection_range: module.to_lsp_range(caller_name_range),
                    data: None,
                },
                from_ranges: vec![module.to_lsp_range(call_range)],
            });
        }
    }

    results
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
        sys_info: SysInfo,
        definition_kind: DefinitionMetadata,
        target_definition: &TextRangeWithModule,
    ) -> Result<Vec<(Module, Vec<CallerInfo>)>, Cancelled> {
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
                        true,
                    )
                    .unwrap_or_default();

                if references.is_empty() {
                    return None;
                }

                let ref_set: std::collections::HashSet<TextRange> =
                    references.into_iter().collect();

                let mut callers_in_file = Vec::new();

                fn collect_calls_from_expr(
                    expr: &Expr,
                    ref_set: &std::collections::HashSet<TextRange>,
                    module_name: ModuleName,
                    ast: &ModModule,
                    callers: &mut Vec<CallerInfo>,
                ) {
                    if let Expr::Call(call) = expr
                        && ref_set
                            .iter()
                            .any(|ref_range| call.func.range().contains(ref_range.start()))
                    {
                        let (name, full_range, name_range, kind) =
                            find_containing_function_for_call(
                                module_name,
                                ast,
                                call.range().start(),
                            );
                        callers.push(CallerInfo {
                            call_range: call.range(),
                            name,
                            full_range,
                            name_range,
                            kind,
                        });
                    }
                    expr.recurse(&mut |child| {
                        collect_calls_from_expr(child, ref_set, module_name, ast, callers)
                    });
                }

                ast.visit(&mut |expr| {
                    collect_calls_from_expr(
                        expr,
                        &ref_set,
                        handle.module(),
                        &ast,
                        &mut callers_in_file,
                    )
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
        let definitions = self
            .as_ref()
            .find_definition(handle, position, FindPreference::default())
            .map(Vec1::into_vec)
            .unwrap_or_default();
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

                    let definitions = self
                        .as_ref()
                        .find_definition(&target_handle, call_pos, FindPreference::default())
                        .map(Vec1::into_vec)
                        .unwrap_or_default();

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
    use pyrefly_python::ast::Ast;
    use pyrefly_python::module_name::ModuleName;
    use ruff_python_ast::PySourceType;
    use ruff_text_size::TextSize;

    use super::find_containing_function_for_call;

    #[test]
    fn test_find_containing_function_for_call() {
        use lsp_types::SymbolKind;

        let source = r#"
def my_function():
    x = call()

class MyClass:
    def method(self):
        y = call()
"#;
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);
        let module_name = ModuleName::from_str("test");

        let pos_in_func = TextSize::from(30);
        let (name, _full_range, _name_range, kind) =
            find_containing_function_for_call(module_name, &ast, pos_in_func);
        assert_eq!(name, "test.my_function");
        assert_eq!(kind, SymbolKind::FUNCTION);

        let pos_in_method = TextSize::from(85);
        let (name, _full_range, _name_range, kind) =
            find_containing_function_for_call(module_name, &ast, pos_in_method);
        assert_eq!(name, "test.MyClass.method");
        assert_eq!(kind, SymbolKind::METHOD);
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

    #[test]
    fn test_find_enclosing_call_range() {
        use super::find_enclosing_call_range;

        let source = "foo(bar)\nfoo(bar())\nobj.method()\n";
        let (ast, _, _) = Ast::parse(source, PySourceType::Python);

        // Position on `foo` in `foo(bar)` — inside func range of the call
        let pos_foo = TextSize::from(0);
        assert!(find_enclosing_call_range(&ast, pos_foo).is_some());

        // Position on `bar` in `foo(bar)` — argument, not inside func range
        let pos_bar_arg = TextSize::from(4);
        assert!(find_enclosing_call_range(&ast, pos_bar_arg).is_none());

        // Position on inner `bar` in `foo(bar())` — inside func range of inner call
        let pos_inner_bar = TextSize::from(13);
        assert!(find_enclosing_call_range(&ast, pos_inner_bar).is_some());

        // Position on `method` in `obj.method()` — inside func range (attribute is part of func)
        let pos_method = TextSize::from(24);
        assert!(find_enclosing_call_range(&ast, pos_method).is_some());
    }

    #[test]
    fn test_module_name_from_path() {
        use std::fs;

        use tempfile::tempdir;

        use super::module_name_from_path;

        let dir = tempdir().unwrap();
        let root = dir.path();

        // Standalone file: bar.py with no __init__.py → bar
        let bar_py = root.join("bar.py");
        fs::write(&bar_py, "").unwrap();
        assert_eq!(module_name_from_path(&bar_py).as_str(), "bar");

        // Package: pkg/__init__.py + pkg/baz.py → pkg.baz
        let pkg = root.join("pkg");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(pkg.join("__init__.py"), "").unwrap();
        let baz_py = pkg.join("baz.py");
        fs::write(&baz_py, "").unwrap();
        assert_eq!(module_name_from_path(&baz_py).as_str(), "pkg.baz");

        // __init__.py itself → package name
        let init_py = pkg.join("__init__.py");
        assert_eq!(module_name_from_path(&init_py).as_str(), "pkg");

        // Nested: foo/__init__.py + foo/bar/__init__.py + foo/bar/qux.py → foo.bar.qux
        let nested = root.join("foo").join("bar");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("foo").join("__init__.py"), "").unwrap();
        fs::write(nested.join("__init__.py"), "").unwrap();
        let qux_py = nested.join("qux.py");
        fs::write(&qux_py, "").unwrap();
        assert_eq!(module_name_from_path(&qux_py).as_str(), "foo.bar.qux");

        // .pyi stub → same as .py
        let stub = root.join("stub.pyi");
        fs::write(&stub, "").unwrap();
        assert_eq!(module_name_from_path(&stub).as_str(), "stub");
    }

    #[test]
    fn test_convert_external_references_to_incoming_calls() {
        use std::io::Write;

        use lsp_types::Url;
        use tempfile::NamedTempFile;

        use super::convert_external_references_to_incoming_calls;

        let source = r#"from other import target

def caller_func():
    target()

x: target = None
"#;
        let mut file = NamedTempFile::with_suffix(".py").unwrap();
        write!(file, "{}", source).unwrap();
        let url = Url::from_file_path(file.path()).unwrap();

        let call_range = lsp_types::Range {
            start: lsp_types::Position {
                line: 3,
                character: 4,
            },
            end: lsp_types::Position {
                line: 3,
                character: 10,
            },
        };
        let import_range = lsp_types::Range {
            start: lsp_types::Position {
                line: 0,
                character: 18,
            },
            end: lsp_types::Position {
                line: 0,
                character: 24,
            },
        };

        let external_refs = vec![(url.clone(), vec![call_range, import_range])];
        let results = convert_external_references_to_incoming_calls(external_refs);

        // Only the call expression should produce an incoming call, not the import
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .from
                .detail
                .as_ref()
                .unwrap()
                .ends_with("caller_func")
        );
    }

    #[test]
    fn test_convert_external_references_filters_non_call() {
        use std::io::Write;

        use lsp_types::Url;
        use tempfile::NamedTempFile;

        use super::convert_external_references_to_incoming_calls;

        let source = r#"from other import target
x: target = None
"#;
        let mut file = NamedTempFile::with_suffix(".py").unwrap();
        write!(file, "{}", source).unwrap();
        let url = Url::from_file_path(file.path()).unwrap();

        let import_range = lsp_types::Range {
            start: lsp_types::Position {
                line: 0,
                character: 18,
            },
            end: lsp_types::Position {
                line: 0,
                character: 24,
            },
        };
        let annotation_range = lsp_types::Range {
            start: lsp_types::Position {
                line: 1,
                character: 3,
            },
            end: lsp_types::Position {
                line: 1,
                character: 9,
            },
        };

        let external_refs = vec![(url, vec![import_range, annotation_range])];
        let results = convert_external_references_to_incoming_calls(external_refs);
        assert!(results.is_empty());
    }
}
