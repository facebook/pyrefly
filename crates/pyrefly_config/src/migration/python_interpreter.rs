/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::str::FromStr;

use configparser::ini::Ini;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::pyright::PyrightConfig;
use crate::util::ConfigOrigin;

/// Configuration option for Python interpreter
pub struct PythonInterpreter;

impl ConfigOptionMigrater for PythonInterpreter {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // https://mypy.readthedocs.io/en/latest/config_file.html#import-discovery
        // python_executable can only be set in the top level `[mypy]` global section
        let python_executable = mypy_cfg.get("mypy", "python_executable");

        if python_executable.is_none() {
            return Err(anyhow::anyhow!("No python_executable found in mypy config"));
        }

        let python_interpreter = python_executable.unwrap();
        pyrefly_cfg.interpreters.python_interpreter = PathBuf::from_str(&python_interpreter)
            .ok()
            .map(ConfigOrigin::config);
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        _pyright_cfg: &PyrightConfig,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "Pyright does not have a direct equivalent for python_interpreter"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy",
            "python_executable",
            Some("/usr/bin/python3".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let python_interpreter = PythonInterpreter;
        let _ = python_interpreter.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.interpreters.python_interpreter,
            Some(ConfigOrigin::config(PathBuf::from("/usr/bin/python3")))
        );
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_interpreter = pyrefly_cfg.interpreters.python_interpreter.clone();

        let python_interpreter = PythonInterpreter;
        let _ = python_interpreter.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.interpreters.python_interpreter,
            default_interpreter
        );
    }

    #[test]
    fn test_migrate_from_pyright() {
        let pyright_cfg = default_pyright_config();
        let mut pyrefly_cfg = ConfigFile::default();
        let default_interpreter = pyrefly_cfg.interpreters.python_interpreter.clone();

        let python_interpreter = PythonInterpreter;
        let result = python_interpreter.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.interpreters.python_interpreter,
            default_interpreter
        );
    }
}
