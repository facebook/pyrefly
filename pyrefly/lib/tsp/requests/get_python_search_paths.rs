/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the `typeServer/getPythonSearchPaths` TSP request.
//!
//! Returns the list of directories that pyrefly uses to resolve Python
//! imports for a given source file. The result includes user-configured
//! search paths, inferred import roots, and site-packages directories.

use lsp_server::RequestId;
use lsp_types::Url;
use tsp_types::protocol::GetPythonSearchPathsParams;

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspConnection;
use crate::tsp::validation::internal_error;
use crate::tsp::validation::parse_uri;

impl<T: TspInterface> TspConnection<T> {
    /// Handle a `typeServer/getPythonSearchPaths` request.
    ///
    /// Validates the snapshot, parses the `from_uri`, and delegates to
    /// [`TspInterface::get_python_search_paths`] to collect the ordered
    /// list of directories used for import resolution.
    ///
    /// For notebook cell URIs, resolves to the parent notebook's filesystem
    /// path so that search paths reflect the notebook's project context.
    pub fn handle_get_python_search_paths(
        &self,
        id: RequestId,
        params: GetPythonSearchPathsParams,
    ) {
        // --- 1. Validate snapshot ---
        if let Err(err) = self.validate_snapshot(params.snapshot) {
            self.send_err(id, err);
            return;
        }

        // --- 2. Parse and resolve from_uri ---
        let url = match parse_uri(&params.from_uri) {
            Ok(url) => url,
            Err(err) => {
                self.send_err(id, err);
                return;
            }
        };

        // For non-file URIs (e.g. notebook cells), resolve to the parent
        // notebook's filesystem path so we return the right search paths.
        let resolved_url = if url.scheme() != "file" {
            match self
                .inner()
                .resolve_uri_to_path(&url)
                .and_then(|p| Url::from_file_path(p).ok())
            {
                Some(file_url) => file_url,
                None => {
                    // Cannot resolve to a filesystem path — return empty list.
                    self.send_ok::<Vec<String>>(id, vec![]);
                    return;
                }
            }
        } else {
            url
        };

        match self.inner().get_python_search_paths(&resolved_url) {
            Ok(paths) => self.send_ok(id, paths),
            Err(detail) => self.send_err(id, internal_error(&detail)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tsp::validation::parse_uri;

    #[test]
    fn test_valid_file_uri() {
        let url = parse_uri("file:///home/user/project/main.py").unwrap();
        assert_eq!(url.scheme(), "file");
    }

    #[test]
    fn test_valid_file_uri_windows_style() {
        let url = parse_uri("file:///C:/Users/test/project/main.py").unwrap();
        assert_eq!(url.scheme(), "file");
        assert!(url.to_file_path().is_ok());
    }

    #[test]
    fn test_empty_uri_is_error() {
        let result = parse_uri("");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("not valid"));
    }

    #[test]
    fn test_relative_path_is_error() {
        assert!(parse_uri("some/path/main.py").is_err());
    }

    #[test]
    fn test_notebook_cell_uri_is_valid() {
        let url =
            parse_uri("vscode-notebook-cell:/Users/kylei/projects/test/test.ipynb#W0sZmlsZQ%3D%3D")
                .unwrap();
        assert_eq!(url.scheme(), "vscode-notebook-cell");
    }

    #[test]
    fn test_uri_with_spaces_encoded() {
        let url = parse_uri("file:///home/user/my%20project/main.py").unwrap();
        assert_eq!(url.scheme(), "file");
    }
}
