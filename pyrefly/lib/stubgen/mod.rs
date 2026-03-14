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
//!
//! - [`emit`]: Writes `.pyi` syntax from AST nodes.
//! - [`visibility`]: Decides which names to include based on `__all__` / privacy.

mod emit;
pub(crate) mod visibility;

use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::sys_info::SysInfo;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Stmt;

use crate::export::definitions::Definitions;
use crate::stubgen::emit::collect_overloaded_names;
use crate::stubgen::emit::emit_stmt;
use crate::stubgen::emit::is_overload_impl;
use crate::stubgen::emit::stmt_uses_any;
use crate::stubgen::visibility::VisibilityFilter;

/// Options controlling stub generation.
pub struct StubgenOptions {
    /// Include private names (those starting with `_`) in generated stubs.
    pub include_private: bool,
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
    let overloaded = collect_overloaded_names(&ast.body);

    let mut out = String::new();
    let mut needs_any_import = false;
    let mut last_emitted_class = false;

    for stmt in &ast.body {
        if is_overload_impl(stmt, &overloaded) {
            continue;
        }

        let is_class = matches!(stmt, Stmt::ClassDef(_));
        let pos_before = out.len();

        let emitted = emit_stmt(source, stmt, &filter, options, &mut out, "", &defs);
        if emitted {
            // Blank line before/after class definitions (PEP 8 style).
            if (is_class || last_emitted_class) && pos_before > 0 {
                out.insert(pos_before, '\n');
            }
            needs_any_import = needs_any_import || stmt_uses_any(stmt);
            last_emitted_class = is_class;
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
        assert_eq!(result, "def foo(x, y): ...\n");
    }

    #[test]
    fn test_function_partial_annotations() {
        let result = stub("def foo(x: int, y) -> str:\n    pass\n");
        assert_eq!(result, "def foo(x: int, y) -> str: ...\n");
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

    // --- PR 4: complex annotation tests ---

    #[test]
    fn test_union_annotation() {
        let result = stub("def foo(x: int | str) -> int | None:\n    pass\n");
        assert_eq!(result, "def foo(x: int | str) -> int | None: ...\n");
    }

    #[test]
    fn test_optional_annotation() {
        let result = stub(
            "from typing import Optional\ndef foo(x: Optional[str]) -> Optional[int]:\n    pass\n",
        );
        assert!(result.contains("def foo(x: Optional[str]) -> Optional[int]: ..."));
    }

    #[test]
    fn test_generic_annotations() {
        let result = stub(
            "\
from typing import List, Dict, Tuple, Set
def foo(a: List[int], b: Dict[str, int]) -> Tuple[str, ...]:
    pass
x: Set[float] = set()
",
        );
        assert!(
            result.contains("def foo(a: List[int], b: Dict[str, int]) -> Tuple[str, ...]: ...")
        );
        assert!(result.contains("x: Set[float] = ..."));
    }

    #[test]
    fn test_callable_annotation() {
        let result = stub(
            "\
from typing import Callable
def apply(fn: Callable[[int, str], bool]) -> None:
    pass
",
        );
        assert!(result.contains("def apply(fn: Callable[[int, str], bool]) -> None: ..."));
    }

    #[test]
    fn test_nested_generic_annotation() {
        let result = stub("def foo(x: dict[str, list[tuple[int, ...]]]) -> None:\n    pass\n");
        assert_eq!(
            result,
            "def foo(x: dict[str, list[tuple[int, ...]]]) -> None: ...\n"
        );
    }

    #[test]
    fn test_class_with_generic_base() {
        let result = stub(
            "\
from typing import Generic, TypeVar
T = TypeVar('T')
class Stack(Generic[T]):
    def push(self, item: T) -> None:
        pass
    def pop(self) -> T:
        pass
",
        );
        assert!(result.contains("class Stack(Generic[T]):"));
        assert!(result.contains("    def push(self, item: T) -> None: ..."));
        assert!(result.contains("    def pop(self) -> T: ..."));
    }

    #[test]
    fn test_overloaded_function() {
        let result = stub(
            "\
from typing import overload
@overload
def process(x: int) -> int: ...
@overload
def process(x: str) -> str: ...
def process(x):
    return x
",
        );
        assert!(result.contains("@overload\ndef process(x: int) -> int: ..."));
        assert!(result.contains("@overload\ndef process(x: str) -> str: ..."));
    }

    #[test]
    fn test_reexport_import() {
        let result = stub("from os.path import join\nfrom typing import List\n");
        assert_eq!(
            result,
            "from os.path import join\nfrom typing import List\n"
        );
    }

    #[test]
    fn test_import_as() {
        let result = stub("import numpy as np\nfrom pathlib import Path as P\n");
        assert_eq!(
            result,
            "import numpy as np\nfrom pathlib import Path as P\n"
        );
    }

    #[test]
    fn test_multiple_decorators() {
        let result = stub(
            "\
class Foo:
    @property
    def name(self) -> str:
        return ''
    @name.setter
    def name(self, value: str) -> None:
        pass
",
        );
        assert!(result.contains("    @property\n    def name(self) -> str: ..."));
        assert!(result.contains("    @name.setter\n    def name(self, value: str) -> None: ..."));
    }

    #[test]
    fn test_class_variable_vs_instance_annotation() {
        let result = stub(
            "\
from typing import ClassVar
class Foo:
    count: ClassVar[int] = 0
    name: str
",
        );
        assert!(result.contains("    count: ClassVar[int] = ..."));
        assert!(result.contains("    name: str\n"));
    }

    #[test]
    fn test_string_literal_annotation() {
        let result = stub("def foo(x: 'ForwardRef') -> 'ForwardRef':\n    pass\n");
        assert_eq!(result, "def foo(x: 'ForwardRef') -> 'ForwardRef': ...\n");
    }
}
