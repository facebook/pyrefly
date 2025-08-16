/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getPythonSearchPaths TSP request

use lsp_types::Url;
use tsp_types as tsp;

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_python_search_paths(
        &self,
        _transaction: &Transaction<'_>,
        params: tsp::GetPythonSearchPathsParams,
    ) -> Vec<Url> {
        // Get the URI directly from params
        let uri = match lsp_types::Url::parse(&params.from_uri) {
            Ok(u) => u,
            Err(_) => return Vec::new(),
        };

        // Convert URI to file path
        let path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Vec::new(), // Return empty vector on error
        };

        // Check if language services are disabled for this workspace
        let workspace_disabled = self.inner.workspaces.get_with(path.clone(), |workspace| {
            workspace.disable_language_services
        });

        if workspace_disabled {
            return Vec::new();
        }

        // Try to get configuration from config finder first
        let config_opt = if path.is_dir() {
            // For directories, use the directory method directly
            self.inner.state.config_finder().directory(&path)
        } else {
            // For files, try to get config from python_file method
            let module_path = if self.inner.open_files.read().contains_key(&path) {
                pyrefly_python::module_path::ModulePath::memory(path.clone())
            } else {
                pyrefly_python::module_path::ModulePath::filesystem(path.clone())
            };

            // python_file always returns a config, but check if it's synthetic
            let config = self.inner.state.config_finder().python_file(
                pyrefly_python::module_name::ModuleName::unknown(),
                &module_path,
            );

            // If it's a real config file (not synthetic), use it
            match &config.source {
                crate::config::config::ConfigSource::File(_)
                | crate::config::config::ConfigSource::Marker(_) => Some(config),
                crate::config::config::ConfigSource::Synthetic => None,
            }
        };

        if let Some(config) = config_opt {
            // We found a real config file, use its search paths
            let mut search_paths = Vec::new();

            // Add search paths from config
            for path in config.search_path() {
                if let Ok(uri) = Url::from_file_path(path) {
                    search_paths.push(uri);
                }
            }

            // Add site package paths from config
            for path in config.site_package_path() {
                if let Ok(uri) = Url::from_file_path(path) {
                    search_paths.push(uri);
                }
            }

            search_paths
        } else {
            // No config file found, use workspace python_info as fallback
            self.inner.workspaces.get_with(path, |workspace| {
                let mut search_paths = Vec::new();

                // Add workspace-specific search paths if available
                if let Some(workspace_search_paths) = &workspace.search_path {
                    for path in workspace_search_paths {
                        if let Ok(uri) = Url::from_file_path(path) {
                            search_paths.push(uri);
                        }
                    }
                }

                // Add Python environment site package paths if available
                if let Some(python_info) = &workspace.python_info {
                    let env = python_info.env();

                    // Add all site package paths from the Python environment
                    for path in env.all_site_package_paths() {
                        if let Ok(uri) = Url::from_file_path(path) {
                            search_paths.push(uri);
                        }
                    }
                }

                search_paths
            })
        }
    }
}
