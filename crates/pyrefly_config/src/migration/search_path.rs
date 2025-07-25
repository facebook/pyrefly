/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::mypy::util;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for search path
pub struct SearchPath;

impl ConfigOptionMigrater for SearchPath {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // https://mypy.readthedocs.io/en/latest/config_file.html#import-discovery
        // mypy_path can only be set in the top level `[mypy]` global section
        let mypy_path = match mypy_cfg.get("mypy", "mypy_path") {
            Some(path) => path,
            None => return Err(anyhow::anyhow!("No mypy_path found in mypy config")),
        };

        let paths = util::string_to_paths(&mypy_path);
        if paths.is_empty() {
            return Err(anyhow::anyhow!("Empty search paths found in mypy config"));
        }
        pyrefly_cfg.search_path_from_file = paths;
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, search path is specified in the "extraPaths" field
        let search_path = match &pyright_cfg.search_path {
            Some(path) => path,
            None => return Err(anyhow::anyhow!("No search paths found in pyright config")),
        };
        if search_path.is_empty() {
            return Err(anyhow::anyhow!(
                "Empty search paths found in pyright config"
            ));
        }

        pyrefly_cfg.search_path_from_file = search_path.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::migration::test_util::default_pyright_config;

    // Constants for test paths
    const FILE_PATH1: &str = "/path/to/stubs";
    const FILE_PATH2: &str = "/another/path";
    const FILE_PATH3: &str = "/third/path";

    // Constants for mypy config
    const MYPY_SECTION: &str = "mypy";
    const MYPY_PATH_KEY: &str = "mypy_path";

    // Constant vector of all paths
    const ALL_PATHS: [&str; 3] = [FILE_PATH1, FILE_PATH2, FILE_PATH3];

    #[test]
    fn test_migrate_from_mypy_with_single_path() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(MYPY_SECTION, MYPY_PATH_KEY, Some(FILE_PATH1.to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let search_path = SearchPath;
        let _ = search_path.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.search_path_from_file,
            vec![PathBuf::from(FILE_PATH1)]
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_comma_separated_paths() {
        let mut mypy_cfg = Ini::new();
        let comma_separated = format!("{FILE_PATH1}, {FILE_PATH2}, {FILE_PATH3}");
        mypy_cfg.set(MYPY_SECTION, MYPY_PATH_KEY, Some(comma_separated));

        let mut pyrefly_cfg = ConfigFile::default();

        let search_path = SearchPath;
        let _ = search_path.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        // Verify that the ConfigFile was updated correctly
        assert_eq!(
            pyrefly_cfg.search_path_from_file,
            ALL_PATHS
                .iter()
                .map(|&p| PathBuf::from(p))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_colon_separated_paths() {
        let mut mypy_cfg = Ini::new();
        let colon_separated = format!("{FILE_PATH1}:{FILE_PATH2}:{FILE_PATH3}");
        mypy_cfg.set(MYPY_SECTION, MYPY_PATH_KEY, Some(colon_separated));

        let mut pyrefly_cfg = ConfigFile::default();

        let search_path = SearchPath;
        let _ = search_path.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.search_path_from_file,
            ALL_PATHS
                .iter()
                .map(|&p| PathBuf::from(p))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_mixed_separators() {
        let mut mypy_cfg = Ini::new();
        let mixed_separators = format!("{FILE_PATH1}, {FILE_PATH2}:{FILE_PATH3}");
        mypy_cfg.set(MYPY_SECTION, MYPY_PATH_KEY, Some(mixed_separators));

        let mut pyrefly_cfg = ConfigFile::default();

        let search_path = SearchPath;
        let _ = search_path.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.search_path_from_file,
            ALL_PATHS
                .iter()
                .map(|&p| PathBuf::from(p))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_empty_path() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(MYPY_SECTION, MYPY_PATH_KEY, Some("".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();
        let default_search_path = pyrefly_cfg.search_path_from_file.clone();

        let search_path = SearchPath;
        let _ = search_path.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.search_path_from_file, default_search_path);
    }

    #[test]
    fn test_migrate_from_mypy_with_no_path() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_search_path = pyrefly_cfg.search_path_from_file.clone();

        let search_path = SearchPath;
        let _ = search_path.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.search_path_from_file, default_search_path);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.search_path = Some(vec![
            PathBuf::from(FILE_PATH1),
            PathBuf::from(FILE_PATH2),
            PathBuf::from(FILE_PATH3),
        ]);

        let mut pyrefly_cfg = ConfigFile::default();

        let search_path = SearchPath;
        let result = search_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.search_path_from_file,
            ALL_PATHS
                .iter()
                .map(|&p| PathBuf::from(p))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_search_path = pyrefly_cfg.search_path_from_file.clone();

        let search_path = SearchPath;
        let result = search_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.search_path_from_file, default_search_path);
    }

    #[test]
    fn test_migrate_from_pyright_empty_paths() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.search_path = Some(vec![]);

        let mut pyrefly_cfg = ConfigFile::default();
        let default_search_path = pyrefly_cfg.search_path_from_file.clone();

        let search_path = SearchPath;
        let result = search_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.search_path_from_file, default_search_path);
    }

    #[test]
    fn test_migrate_from_pyright_with_paths() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.search_path = Some(vec![
            PathBuf::from(FILE_PATH1),
            PathBuf::from(FILE_PATH2),
            PathBuf::from(FILE_PATH3),
        ]);

        let mut pyrefly_cfg = ConfigFile::default();

        let search_path = SearchPath;
        let result = search_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.search_path_from_file,
            ALL_PATHS
                .iter()
                .map(|&p| PathBuf::from(p))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_migrate_from_pyright_with_no_paths() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_search_path = pyrefly_cfg.search_path_from_file.clone();

        let search_path = SearchPath;
        let result = search_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.search_path_from_file, default_search_path);
    }

    #[test]
    fn test_migrate_from_pyright_with_empty_paths() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.search_path = Some(vec![]);

        let mut pyrefly_cfg = ConfigFile::default();
        let default_search_path = pyrefly_cfg.search_path_from_file.clone();

        let search_path = SearchPath;
        let result = search_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.search_path_from_file, default_search_path);
    }
}
