/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The very basic version of this script takes the path to the config file
// as an argument, reads it, parses it, converts it, and writes it out.
// Future features:
// - find the configs to convert rather than taking a path
// - match up the error configurations (best-effort)
// - extract configs from pyproject.toml
// - write configs to pyproject.toml
// This script does not otherwise invoke pyrefly. This gives the user time to change anything by hand if needed.
use std::path::PathBuf;

use clap::Parser;
use tracing::info;

use crate::commands::run::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::mypy;
use crate::config::mypy::MypyConfig;
use crate::config::pyright;
use crate::config::pyright::PyrightConfig;
use crate::util::fs_anyhow;

#[derive(Clone, Debug, Parser)]
pub struct Args {
    /// The path to the mypy or pyright config file to convert.
    input_path: PathBuf,
    /// Optional path to write the converted pyre.toml config file to. If not provided, the config will be written to the same directory as the input file.
    output_path: Option<PathBuf>,
}

impl Args {
    fn load_from_pyproject(raw_file: &str) -> anyhow::Result<ConfigFile> {
        info!("Attempting to load [tool.mypy] config");
        match mypy::parse_pyrproject_config(raw_file) {
            ok @ Ok(_) => {
                info!("Successfully loaded [tool.mypy] config from pyproject.toml");
                return ok;
            }
            Err(e) => {
                info!("Failed to load [tool.mypy] config from pyproject.toml:\n  {e}");
                info!("Attempting to load [tool.pyright] config.");
            }
        }
        pyright::parse_pyproject_toml(raw_file)
            .inspect(|_| info!("Successfully loaded [tool.pyright] config from pyproject.toml"))
            .inspect_err(|e| {
                info!("failed to load [tool.pyright] config from pyproject.toml:\n  {e}")
            })
    }

    pub fn run(&self) -> anyhow::Result<CommandExitStatus> {
        info!("Looking for {}", self.input_path.display());
        let config = if self.input_path.file_name() == Some("pyrightconfig.json".as_ref()) {
            let raw_file = fs_anyhow::read_to_string(&self.input_path)?;
            info!("Detected pyright config file");
            let pyr = serde_json::from_str::<PyrightConfig>(&raw_file)?;
            pyr.convert()
        } else if self.input_path.file_name() == Some("mypy.ini".as_ref()) {
            info!("Detected mypy config file");
            MypyConfig::parse_mypy_config(&self.input_path)?
        } else if self.input_path.file_name() == Some("pyproject.toml".as_ref()) {
            let raw_file = fs_anyhow::read_to_string(&self.input_path)?;
            info!("Detected pyproject.toml file.");
            Self::load_from_pyproject(&raw_file)?
        } else {
            eprintln!(
                "Currently only migration from pyrightconfig.json, mypy.ini, and pyproject.toml is supported"
            );
            return Ok(CommandExitStatus::UserError);
        };
        info!("Conversion finished");

        let serialized = toml::to_string_pretty(&config)?;
        if let Some(output_path) = &self.output_path {
            fs_anyhow::write(output_path, serialized.as_bytes())?;
            info!("New config written to {}", output_path.display());
        } else {
            let output_path = self.input_path.with_file_name(ConfigFile::CONFIG_FILE_NAME);
            fs_anyhow::write(&output_path, serialized.as_bytes())?;
            info!("New config written to {}", output_path.display());
        }
        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::util::globs::Globs;

    #[test]
    fn test_run_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let args = Args {
            input_path: tmp.path().join("pyrightconfig.json"),
            output_path: None,
        };
        let pyr = br#"{
    "include": ["src/**/*.py"]
}
"#;
        fs_anyhow::write(&args.input_path, pyr)?;
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output_path = args.input_path.with_file_name(ConfigFile::CONFIG_FILE_NAME);
        let output = fs_anyhow::read_to_string(&output_path)?;
        // We're not going to check the whole output because most of it will be default values, which may change.
        // We only actually care about the includes.
        let output_lines = output.lines().collect::<Vec<_>>();
        assert_eq!(output_lines[0], r#"project_includes = ["src/**/*.py"]"#);
        assert!(output_lines.len() > 1);
        ConfigFile::from_file(&output_path, false)?;
        Ok(())
    }

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let args = Args {
            input_path: tmp.path().join("mypy.ini"),
            output_path: None,
        };
        // This config is derived from the pytorch mypy.ini.
        let mypy = br#"[mypy]
files =
    src,
    other_src,
    test/some_test.py,

mypy_path = some_paths:comma,separated

unknown_option = True

exclude = src/include/|other_src/include/|src/specific/bad/file.py

[mypy-some.*.project]
ignore_missing_imports = True

[mypy-some.specific.project.subdir]
ignore_missing_imports = True

[mypy-stricter.on.this.*]
check_untyped_defs = True
"#;
        fs_anyhow::write(&args.input_path, mypy)?;
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));

        // We care about the config getting serialized in a way that can be checked-in to a repo,
        // i.e. without absolutized paths. So we need to check the raw file.
        #[derive(Deserialize)]
        struct CheckConfig {
            project_includes: Vec<String>,
            search_path: Vec<String>,
        }
        let output_path = args.input_path.with_file_name(ConfigFile::CONFIG_FILE_NAME);
        let raw_output = fs_anyhow::read_to_string(&output_path)?;
        let CheckConfig {
            project_includes,
            search_path,
        } = toml::from_str::<CheckConfig>(&raw_output)?;
        assert_eq!(
            project_includes,
            vec!["src", "other_src", "test/some_test.py"]
        );
        assert_eq!(search_path, vec!["some_paths", "comma", "separated"]);
        Ok(())
    }

    #[test]
    fn test_run_pyproject_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let args = Args {
            input_path: tmp.path().join("pyproject.toml"),
            output_path: None,
        };
        let pyproject = br#"[tool.mypy]
files = ["a.py"]
"#;
        fs_anyhow::write(&args.input_path, pyproject)?;
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let args = Args {
            input_path: tmp.path().join("pyproject.toml"),
            output_path: None,
        };
        let pyproject = br#"[tool.pyright]
include = ["a.py"]
"#;
        fs_anyhow::write(&args.input_path, pyproject)?;
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_bad_mypy_into_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let args = Args {
            input_path: tmp.path().join("pyproject.toml"),
            output_path: None,
        };
        let pyproject = br#"[tool.pyright]
include = ["a.py"]

[tool.mypy]
files = 1
"#;
        fs_anyhow::write(&args.input_path, pyproject)?;
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_mypy_over_pyright() -> anyhow::Result<()> {
        // The current implementation favors mypy over pyright. This test documents that.
        // However, we may want to change this in the future, so it's OK to break this test.
        let pyproject = r#"[tool.pyright]
include = ["pyright.py"]

[tool.mypy]
files = ["mypy.py"]
"#;
        let cfg = Args::load_from_pyproject(pyproject)?;
        assert_eq!(cfg.project_includes, Globs::new(vec!["mypy.py".to_owned()]));
        Ok(())
    }
}
