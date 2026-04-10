/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the `typeServer/resolveImport` TSP request.
//!
//! Resolves a Python import (absolute or relative) to the file URI that
//! contains the target module.  The heavy lifting is delegated to
//! pyrefly's existing `find_import` infrastructure via
//! [`Transaction::import_handle`].

use lsp_server::RequestId;
use lsp_server::ResponseError;
use lsp_types::Url;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use ruff_python_ast::name::Name;
use tsp_types::protocol::ResolveImportParams;

use crate::lsp::module_helpers::to_real_path;
use crate::lsp::non_wasm::server::TspInterface;
use crate::lsp::non_wasm::transaction_manager::TransactionManager;
use crate::tsp::server::TspServer;
use crate::tsp::validation::invalid_params_error;
use crate::tsp::validation::parse_file_uri;

impl<T: TspInterface> TspServer<T> {
    /// Handle a `typeServer/resolveImport` request.
    ///
    /// Converts the TSP [`ResolveImportParams`] into pyrefly's internal
    /// [`ModuleName`], resolves via [`Transaction::import_handle`], and returns
    /// the resolved module's file URI as a string (or JSON `null` if the
    /// module cannot be found).
    pub fn handle_resolve_import<'a>(
        &'a self,
        id: RequestId,
        params: ResolveImportParams,
        ide_transaction_manager: &mut TransactionManager<'a>,
    ) {
        // --- 1. Validate snapshot ---
        if let Err(err) = self.validate_snapshot(params.snapshot) {
            self.send_err(id, err);
            return;
        }

        // --- 2. Parse source URI ---
        let source_url = match parse_file_uri(&params.source_uri) {
            Ok(url) => url,
            Err(err) => {
                self.send_err(id, err);
                return;
            }
        };
        let source_path = match source_url.to_file_path() {
            Ok(p) => p,
            Err(_) => {
                self.send_err(id, invalid_params_error("sourceUri is not a file:// URI"));
                return;
            }
        };

        // --- 3. Build source handle and resolve the module name ---
        let source_module_path = ModulePath::filesystem(source_path.clone());
        let source_handle = self.inner.handle_from_module_path(source_module_path);

        let module_name = match resolve_module_name(
            &params.module_descriptor.name_parts,
            params.module_descriptor.leading_dots,
            source_handle.module(),
            source_path
                .file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f == "__init__.py" || f == "__init__.pyi"),
        ) {
            Ok(name) => name,
            Err(err) => {
                self.send_err(id, err);
                return;
            }
        };

        // --- 4. Resolve the import via existing infrastructure ---
        let transaction = self
            .inner
            .non_committable_transaction(ide_transaction_manager);
        let result = transaction.import_handle(&source_handle, module_name, None);

        // --- 5. Convert result to URI string (or null) ---
        let uri_string: Option<String> = result.finding().and_then(|handle| {
            to_real_path(handle.path()).and_then(|path| {
                Url::from_file_path(path.canonicalize().unwrap_or(path))
                    .ok()
                    .map(|u| u.to_string())
            })
        });

        self.send_ok(id, uri_string);
    }
}

/// Resolve a TSP module descriptor into a pyrefly [`ModuleName`].
///
/// For absolute imports (`leading_dots == 0`), the name parts are joined
/// with dots (e.g., `["os", "path"]` → `ModuleName("os.path")`).
///
/// For relative imports (`leading_dots > 0`), the source module's name
/// and `is_init` status are used with [`ModuleName::new_maybe_relative`]
/// to resolve to an absolute module name. For example, if the source
/// module is `a.b.c` and the import is `from ..foo import bar`
/// (leading_dots=2, name_parts=["foo"]), the result is `a.foo`.
fn resolve_module_name(
    name_parts: &[String],
    leading_dots: i32,
    source_module: ModuleName,
    is_init: bool,
) -> Result<ModuleName, ResponseError> {
    if leading_dots < 0 {
        return Err(invalid_params_error("leadingDots must be non-negative"));
    }

    if leading_dots == 0 {
        // Absolute import: name parts must be non-empty.
        if name_parts.is_empty() {
            return Err(invalid_params_error(
                "nameParts must be non-empty for absolute imports",
            ));
        }
        Ok(ModuleName::from_parts(name_parts))
    } else {
        // Relative import: resolve against the source module.
        let suffix = if name_parts.is_empty() {
            None
        } else {
            Some(Name::new(name_parts.join(".")))
        };
        source_module
            .new_maybe_relative(is_init, leading_dots as u32, suffix.as_ref())
            .ok_or_else(|| {
                invalid_params_error("relative import level exceeds source module depth")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- resolve_module_name unit tests ---

    #[test]
    fn test_absolute_import_single_part() {
        let name =
            resolve_module_name(&["os".to_owned()], 0, ModuleName::from_str("any"), false).unwrap();
        assert_eq!(name.as_str(), "os");
    }

    #[test]
    fn test_absolute_import_multi_part() {
        let name = resolve_module_name(
            &["os".to_owned(), "path".to_owned()],
            0,
            ModuleName::from_str("any"),
            false,
        )
        .unwrap();
        assert_eq!(name.as_str(), "os.path");
    }

    #[test]
    fn test_absolute_import_empty_parts_is_error() {
        let result = resolve_module_name(&[], 0, ModuleName::from_str("any"), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("nameParts must be non-empty"));
    }

    #[test]
    fn test_negative_leading_dots_is_error() {
        let result =
            resolve_module_name(&["foo".to_owned()], -1, ModuleName::from_str("any"), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("leadingDots must be non-negative"));
    }

    #[test]
    fn test_relative_import_single_dot_with_name() {
        // `from .utils import ...` in module a.b.c
        // leading_dots=1 → strip 1 component from a.b.c → a.b, then append "utils" → a.b.utils
        let name = resolve_module_name(
            &["utils".to_owned()],
            1,
            ModuleName::from_str("a.b.c"),
            false,
        )
        .unwrap();
        assert_eq!(name.as_str(), "a.b.utils");
    }

    #[test]
    fn test_relative_import_two_dots_with_name() {
        // `from ..foo import ...` in module a.b.c
        // leading_dots=2 → strip 2 → a, then append "foo" → a.foo
        let name =
            resolve_module_name(&["foo".to_owned()], 2, ModuleName::from_str("a.b.c"), false)
                .unwrap();
        assert_eq!(name.as_str(), "a.foo");
    }

    #[test]
    fn test_relative_import_from_init() {
        // `from .utils import ...` in module a.b.__init__
        // is_init=true, leading_dots=1 → effective dots = 0 (subtract 1),
        // so no components stripped from a.b, then append "utils" → a.b.utils
        let name = resolve_module_name(&["utils".to_owned()], 1, ModuleName::from_str("a.b"), true)
            .unwrap();
        assert_eq!(name.as_str(), "a.b.utils");
    }

    #[test]
    fn test_relative_import_bare_dot_no_parts() {
        // `from . import *` in module a.b.c → leading_dots=1, name_parts=[]
        // Strip 1 component from a.b.c → a.b (the package itself)
        let name = resolve_module_name(&[], 1, ModuleName::from_str("a.b.c"), false).unwrap();
        assert_eq!(name.as_str(), "a.b");
    }

    #[test]
    fn test_relative_import_too_many_dots_is_error() {
        // `from ....foo import ...` in module a.b.c → 4 dots exceeds depth 3
        let result =
            resolve_module_name(&["foo".to_owned()], 4, ModuleName::from_str("a.b.c"), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("relative import level exceeds"));
    }

    #[test]
    fn test_deeply_nested_absolute_import() {
        let parts: Vec<String> = vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ];
        let name = resolve_module_name(&parts, 0, ModuleName::from_str("any"), false).unwrap();
        assert_eq!(name.as_str(), "a.b.c.d");
    }

    #[test]
    fn test_relative_import_multi_part_suffix() {
        // `from ..parent.module import ...` in module a.b.c
        // leading_dots=2 → strip 2 → a, then append "parent.module" → a.parent.module
        let name = resolve_module_name(
            &["parent".to_owned(), "module".to_owned()],
            2,
            ModuleName::from_str("a.b.c"),
            false,
        )
        .unwrap();
        assert_eq!(name.as_str(), "a.parent.module");
    }
}
