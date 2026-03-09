/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Generates PEP 484 `.pyi` stub files from Python source.
//!
//! Phase 1 (this module): structure-only stubs using the AST and source
//! annotations. No type inference -- unannotated items get `Any`.

use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::sys_info::SysInfo;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;

use crate::export::definitions::Definitions;
use crate::export::definitions::DunderAllKind;

/// Options controlling stub generation.
pub struct StubgenOptions {
    /// Include private names (those starting with `_`) in generated stubs.
    pub include_private: bool,
}

/// The set of names that should appear in the generated stub, based on
/// `__all__` (if present) or public-name heuristics.
enum VisibilityFilter {
    /// `__all__` was explicitly defined; only these names are exported.
    Explicit(Vec<String>),
    /// No `__all__`; export all names that pass the privacy check.
    Inferred,
}

impl VisibilityFilter {
    fn from_definitions(defs: &Definitions) -> Self {
        match defs.dunder_all.kind {
            DunderAllKind::Specified => {
                let names = defs
                    .dunder_all
                    .entries
                    .iter()
                    .filter_map(|entry| match entry {
                        crate::export::definitions::DunderAllEntry::Name(_, name) => {
                            Some(name.to_string())
                        }
                        _ => None,
                    })
                    .collect();
                VisibilityFilter::Explicit(names)
            }
            _ => VisibilityFilter::Inferred,
        }
    }

    /// Should `name` appear in the stub?
    fn should_include(&self, name: &str, include_private: bool) -> bool {
        match self {
            VisibilityFilter::Explicit(names) => names.iter().any(|n| n == name),
            VisibilityFilter::Inferred => {
                // Dunder names (__init__, __str__, etc.) are always public.
                // Single underscore `_` is also public (commonly used as gettext alias).
                // Only single-leading-underscore names (_helper) are private.
                if name.starts_with("__") && name.ends_with("__") {
                    true
                } else if name.starts_with('_') && name != "_" {
                    include_private
                } else {
                    true
                }
            }
        }
    }
}

/// Generate a `.pyi` stub string from Python source code.
///
/// This is the main entry point. It parses the source, builds definitions
/// for `__all__` filtering, and walks the AST to emit stub syntax.
pub fn generate_stub(source: &str, module_name: &str, options: &StubgenOptions) -> String {
    let (ast, _parse_errors, _) = Ast::parse(source, PySourceType::Python);
    let defs = Definitions::new(
        &ast.body,
        ModuleName::from_str(module_name),
        module_name.is_empty() || module_name.ends_with(".__init__"),
        &SysInfo::default(),
    );
    let filter = VisibilityFilter::from_definitions(&defs);

    let mut out = String::new();
    let mut needs_any_import = false;

    for stmt in &ast.body {
        let emitted = emit_stmt(source, stmt, &filter, options, &mut out, "", &defs);
        if emitted {
            needs_any_import = needs_any_import || stmt_uses_any(stmt);
        }
    }

    // Prepend `from typing import Any` if we emitted any `Any` placeholders.
    if needs_any_import {
        let mut header = String::from("from typing import Any\n");
        if !out.is_empty() {
            header.push('\n');
        }
        header.push_str(&out);
        header
    } else {
        out
    }
}

/// Extract the source text for an AST node's range.
fn source_at<'a>(source: &'a str, node: &impl Ranged) -> &'a str {
    let range = node.range();
    &source[range.start().to_usize()..range.end().to_usize()]
}

/// Emit a single top-level (or class-body) statement into the output buffer.
/// Returns `true` if something was written.
fn emit_stmt(
    source: &str,
    stmt: &Stmt,
    filter: &VisibilityFilter,
    options: &StubgenOptions,
    out: &mut String,
    indent: &str,
    defs: &Definitions,
) -> bool {
    match stmt {
        Stmt::Import(_) | Stmt::ImportFrom(_) => {
            out.push_str(indent);
            out.push_str(source_at(source, stmt));
            out.push('\n');
            true
        }
        Stmt::FunctionDef(func) => {
            let name = func.name.as_str();
            if !filter.should_include(name, options.include_private) {
                return false;
            }
            emit_function(source, func, out, indent);
            true
        }
        Stmt::ClassDef(class) => {
            let name = class.name.as_str();
            if !filter.should_include(name, options.include_private) {
                return false;
            }
            emit_class(source, class, filter, options, out, indent, defs);
            true
        }
        Stmt::AnnAssign(ann) => {
            if let Expr::Name(ExprName { id, .. }) = &*ann.target {
                if !filter.should_include(id.as_str(), options.include_private) {
                    return false;
                }
                out.push_str(indent);
                out.push_str(id.as_str());
                out.push_str(": ");
                out.push_str(source_at(source, &*ann.annotation));
                if ann.value.is_some() {
                    out.push_str(" = ...");
                }
                out.push('\n');
                return true;
            }
            false
        }
        Stmt::Assign(assign) => {
            if let [Expr::Name(ExprName { id, .. })] = assign.targets.as_slice() {
                // __all__ assignments are always included -- stub consumers need
                // them regardless of the visibility filter.
                if id.as_str() == "__all__" {
                    out.push_str(indent);
                    out.push_str(source_at(source, stmt));
                    out.push('\n');
                    return true;
                }
                if !filter.should_include(id.as_str(), options.include_private) {
                    return false;
                }
                out.push_str(indent);
                out.push_str(id.as_str());
                out.push_str(": Any = ...\n");
                return true;
            }
            false
        }
        Stmt::TypeAlias(_) => {
            out.push_str(indent);
            out.push_str(source_at(source, stmt));
            out.push('\n');
            true
        }
        _ => false,
    }
}

/// Emit a function stub: `def name(params) -> RetType: ...`
fn emit_function(source: &str, func: &StmtFunctionDef, out: &mut String, indent: &str) {
    for decorator in &func.decorator_list {
        out.push_str(indent);
        out.push('@');
        out.push_str(source_at(source, &decorator.expression));
        out.push('\n');
    }

    out.push_str(indent);
    if func.is_async {
        out.push_str("async ");
    }
    out.push_str("def ");
    out.push_str(func.name.as_str());
    out.push('(');

    emit_parameters(source, &func.parameters, out);

    out.push(')');

    if let Some(returns) = &func.returns {
        out.push_str(" -> ");
        out.push_str(source_at(source, &**returns));
    }

    out.push_str(": ...\n");
}

/// Emit function parameters, preserving source annotations and defaults as `...`.
fn emit_parameters(source: &str, params: &ruff_python_ast::Parameters, out: &mut String) {
    let mut first = true;
    let mut needs_separator = false;

    // Positional-only parameters
    for param in &params.posonlyargs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        emit_param(source, param, out);
    }
    if !params.posonlyargs.is_empty() {
        needs_separator = true;
    }

    // Regular (positional-or-keyword) parameters
    for param in &params.args {
        if needs_separator {
            out.push_str(", /");
            needs_separator = false;
        }
        if !first {
            out.push_str(", ");
        }
        first = false;
        emit_param(source, param, out);
    }
    if needs_separator && params.args.is_empty() {
        // All params were positional-only, emit the separator
        if !first {
            out.push_str(", ");
        }
        out.push('/');
        first = false;
    }

    // *args
    if let Some(vararg) = &params.vararg {
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('*');
        out.push_str(vararg.name.as_str());
        if let Some(ann) = &vararg.annotation {
            out.push_str(": ");
            out.push_str(source_at(source, &**ann));
        }
    } else if !params.kwonlyargs.is_empty() {
        // Bare `*` to signal keyword-only params follow
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('*');
    }

    // Keyword-only parameters
    for param in &params.kwonlyargs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        emit_param(source, param, out);
    }

    // **kwargs
    if let Some(kwarg) = &params.kwarg {
        if !first {
            out.push_str(", ");
        }
        out.push_str("**");
        out.push_str(kwarg.name.as_str());
        if let Some(ann) = &kwarg.annotation {
            out.push_str(": ");
            out.push_str(source_at(source, &**ann));
        }
    }
}

/// Emit a single parameter with its annotation (from source) and default as `...`.
fn emit_param(source: &str, param: &ParameterWithDefault, out: &mut String) {
    out.push_str(param.parameter.name.as_str());
    if let Some(ann) = &param.parameter.annotation {
        out.push_str(": ");
        out.push_str(source_at(source, &**ann));
    }
    if param.default.is_some() {
        if param.parameter.annotation.is_some() {
            out.push_str(" = ...");
        } else {
            out.push_str("=...");
        }
    }
}

/// Emit a class stub with its bases and nested method/variable stubs.
fn emit_class(
    source: &str,
    class: &StmtClassDef,
    _filter: &VisibilityFilter,
    options: &StubgenOptions,
    out: &mut String,
    indent: &str,
    defs: &Definitions,
) {
    for decorator in &class.decorator_list {
        out.push_str(indent);
        out.push('@');
        out.push_str(source_at(source, &decorator.expression));
        out.push('\n');
    }

    out.push_str(indent);
    out.push_str("class ");
    out.push_str(class.name.as_str());

    if let Some(type_params) = &class.type_params {
        out.push_str(source_at(source, &**type_params));
    }

    if let Some(arguments) = &class.arguments
        && (!arguments.args.is_empty() || !arguments.keywords.is_empty())
    {
        // Arguments range already includes the parentheses in the source.
        out.push_str(source_at(source, &**arguments));
    }

    out.push_str(":\n");

    let child_indent = format!("{indent}    ");
    let mut has_body = false;

    for stmt in &class.body {
        // Inside a class, we don't filter by module-level __all__ -- all
        // class members are part of the class's stub.
        let class_filter = VisibilityFilter::Inferred;
        if emit_stmt(
            source,
            stmt,
            &class_filter,
            options,
            out,
            &child_indent,
            defs,
        ) {
            has_body = true;
        }
    }

    if !has_body {
        out.push_str(&child_indent);
        out.push_str("...\n");
    }
}

/// Check whether a statement will produce `Any` in its stub output, so
/// we know to add `from typing import Any` at the top.
fn stmt_uses_any(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::FunctionDef(func) => {
            if func.returns.is_none() {
                return true;
            }
            for p in func
                .parameters
                .posonlyargs
                .iter()
                .chain(&func.parameters.args)
                .chain(&func.parameters.kwonlyargs)
            {
                let name = p.parameter.name.as_str();
                if name == "self" || name == "cls" {
                    continue;
                }
                if p.parameter.annotation.is_none() {
                    return true;
                }
            }
            false
        }
        Stmt::Assign(assign) => {
            if let [Expr::Name(ExprName { id, .. })] = assign.targets.as_slice() {
                id.as_str() != "__all__"
            } else {
                false
            }
        }
        Stmt::ClassDef(class) => class.body.iter().any(stmt_uses_any),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub(source: &str) -> String {
        generate_stub(
            source,
            "test_module",
            &StubgenOptions {
                include_private: false,
            },
        )
    }

    fn stub_with_private(source: &str) -> String {
        generate_stub(
            source,
            "test_module",
            &StubgenOptions {
                include_private: true,
            },
        )
    }

    #[test]
    fn test_simple_function() {
        let result = stub("def foo(x: int, y: str) -> bool:\n    return x > 0\n");
        assert_eq!(result, "def foo(x: int, y: str) -> bool: ...\n");
    }

    #[test]
    fn test_function_no_annotations() {
        let result = stub("def foo(x, y):\n    return x\n");
        assert_eq!(result, "from typing import Any\n\ndef foo(x, y): ...\n");
    }

    #[test]
    fn test_function_partial_annotations() {
        let result = stub("def foo(x: int, y) -> str:\n    pass\n");
        assert_eq!(
            result,
            "from typing import Any\n\ndef foo(x: int, y) -> str: ...\n"
        );
    }

    #[test]
    fn test_async_function() {
        let result = stub("async def fetch(url: str) -> bytes:\n    pass\n");
        assert_eq!(result, "async def fetch(url: str) -> bytes: ...\n");
    }

    #[test]
    fn test_decorated_function() {
        let result = stub("@staticmethod\ndef foo(x: int) -> int:\n    return x\n");
        assert_eq!(result, "@staticmethod\ndef foo(x: int) -> int: ...\n");
    }

    #[test]
    fn test_function_with_defaults() {
        let result = stub("def foo(x: int = 0, y: str = 'hello') -> None:\n    pass\n");
        assert_eq!(result, "def foo(x: int = ..., y: str = ...) -> None: ...\n");
    }

    #[test]
    fn test_function_varargs_kwargs() {
        let result = stub("def foo(*args: int, **kwargs: str) -> None:\n    pass\n");
        assert_eq!(result, "def foo(*args: int, **kwargs: str) -> None: ...\n");
    }

    #[test]
    fn test_function_keyword_only() {
        let result = stub("def foo(x: int, *, key: str) -> None:\n    pass\n");
        assert_eq!(result, "def foo(x: int, *, key: str) -> None: ...\n");
    }

    #[test]
    fn test_function_positional_only() {
        let result = stub("def foo(x: int, y: int, /, z: int) -> None:\n    pass\n");
        assert_eq!(result, "def foo(x: int, y: int, /, z: int) -> None: ...\n");
    }

    #[test]
    fn test_annotated_variable() {
        let result = stub("x: int = 42\n");
        assert_eq!(result, "x: int = ...\n");
    }

    #[test]
    fn test_annotated_variable_no_value() {
        let result = stub("x: int\n");
        assert_eq!(result, "x: int\n");
    }

    #[test]
    fn test_unannotated_variable() {
        let result = stub("x = 42\n");
        assert_eq!(result, "from typing import Any\n\nx: Any = ...\n");
    }

    #[test]
    fn test_class_simple() {
        let result = stub("class Foo:\n    x: int\n    def method(self) -> None:\n        pass\n");
        assert_eq!(
            result,
            "class Foo:\n    x: int\n    def method(self) -> None: ...\n"
        );
    }

    #[test]
    fn test_class_with_bases() {
        let result = stub("class Foo(Bar, Baz):\n    pass\n");
        assert_eq!(result, "class Foo(Bar, Baz):\n    ...\n");
    }

    #[test]
    fn test_class_with_decorator() {
        let result = stub("@dataclass\nclass Foo:\n    x: int\n    y: str = 'hello'\n");
        assert_eq!(
            result,
            "@dataclass\nclass Foo:\n    x: int\n    y: str = ...\n"
        );
    }

    #[test]
    fn test_imports_preserved() {
        let result = stub("import os\nfrom typing import List\n");
        assert_eq!(result, "import os\nfrom typing import List\n");
    }

    #[test]
    fn test_private_names_excluded() {
        let result = stub("def public() -> None:\n    pass\ndef _private() -> None:\n    pass\n");
        assert_eq!(result, "def public() -> None: ...\n");
    }

    #[test]
    fn test_private_names_included() {
        let result = stub_with_private(
            "def public() -> None:\n    pass\ndef _private() -> None:\n    pass\n",
        );
        assert_eq!(
            result,
            "def public() -> None: ...\ndef _private() -> None: ...\n"
        );
    }

    #[test]
    fn test_dunder_all_filtering() {
        let source =
            "__all__ = ['foo']\ndef foo() -> None:\n    pass\ndef bar() -> None:\n    pass\n";
        let result = stub(source);
        assert!(result.contains("def foo() -> None: ..."));
        assert!(!result.contains("bar"));
    }

    #[test]
    fn test_type_alias() {
        let result = stub("type Vector = list[float]\n");
        assert_eq!(result, "type Vector = list[float]\n");
    }

    #[test]
    fn test_dunder_all_preserved() {
        let source = "__all__ = ['foo']\ndef foo() -> None:\n    pass\n";
        let result = stub(source);
        assert!(result.contains("__all__ = ['foo']"));
    }

    #[test]
    fn test_mixed_module() {
        let source = "\
import os
from typing import Optional

VERSION: str = '1.0'

class Config:
    debug: bool
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

def get_config() -> Config:
    return Config()
";
        let result = stub(source);
        assert!(result.contains("import os"));
        assert!(result.contains("from typing import Optional"));
        assert!(result.contains("VERSION: str = ..."));
        assert!(result.contains("class Config:"));
        assert!(result.contains("    debug: bool"));
        assert!(result.contains("    def __init__(self, debug: bool = ...) -> None: ..."));
        assert!(result.contains("def get_config() -> Config: ..."));
    }
}
