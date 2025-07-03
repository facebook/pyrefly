/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This has been deprecated in favor of `pyrefly init`.
// The very basic version of this script takes the path to the config file
// as an argument, reads it, parses it, converts it, and writes it out.
// Future features:
// - match up the error configurations (best-effort)
// This script does not otherwise invoke pyrefly. This gives the user time to change anything by hand if needed.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use pyrefly_util::fs_anyhow;
use pyrefly_util::upward_search::UpwardSearch;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::commands::run::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::mypy;
use crate::config::mypy::MypyConfig;
use crate::config::pyright;
use crate::config::pyright::PyrightConfig;
use crate::config::util::PyProject;

/// Arguments for configuration migration from other type checkers
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Clone, Debug, Parser)]
pub struct Args {
    /// The path to the mypy or pyright config file to convert. Optional.
    /// If not provided, or if it's a directory, pyrefly will search upwards for a
    /// mypy.ini, pyrightconfig.json, or pyproject.toml.
    pub original_config_path: Option<PathBuf>,
}

impl Args {
    fn load_from_pyproject(original_config_path: &Path) -> anyhow::Result<ConfigFile> {
        let raw_file = fs_anyhow::read_to_string(original_config_path)?;
        match mypy::parse_pyproject_config(&raw_file) {
            ok @ Ok(_) => {
                info!(
                    "Migrating [tool.mypy] config from pyproject.toml in `{}`",
                    original_config_path.parent().unwrap().display()
                );
                return ok;
            }
            Err(_) => {
                // Try to parse [tool.pyright] instead.
            }
        }
        pyright::parse_pyproject_toml(&raw_file).inspect(|_| {
            info!(
                "Migrating [tool.pyright] config from pyproject.toml in `{}`",
                original_config_path.parent().unwrap().display()
            )
        })
    }

    fn find_config(start: &Path) -> anyhow::Result<PathBuf> {
        let searcher = UpwardSearch::new(
            // Search for pyproject.toml last, because we're only going to find 1 config.
            vec![
                "mypy.ini".into(),
                "pyrightconfig.json".into(),
                "pyproject.toml".into(),
            ],
            |p| std::sync::Arc::new(p.to_path_buf()),
        );
        searcher.directory(start).map_or_else(
            || Err(anyhow::anyhow!("Failed to find config")),
            |p| Ok(std::sync::Arc::unwrap_or_clone(p)),
        )
    }

    /// Check for certain conditions and warn the user that they may need to edit the config.
    fn check_and_warn(config: &ConfigFile) {
        if toml::to_string(&config).is_ok_and(|s| s.is_empty()) {
            warn!(
                "The generated config is empty. This likely means that none of the config options in the migrated config have pyrefly equivalents."
            );
        }
    }

    pub fn run(&self) -> anyhow::Result<(CommandExitStatus, Option<PathBuf>)> {
        if let Some(path) = self.original_config_path.as_ref()
            && !path.exists()
        {
            error!("Could not find or cannot access `{}`", path.display());
            return Ok((CommandExitStatus::InfraError, None));
        }

        let original_config_path = match &self.original_config_path {
            Some(path) if path.is_file() => path,
            Some(path) => &Self::find_config(path)?,
            None => {
                let cwd = std::env::current_dir()
                    .context("Could not find dir to start search for configs from")?;
                &Self::find_config(&cwd)?
            }
        };
        let config = if original_config_path.file_name() == Some("pyrightconfig.json".as_ref()) {
            let raw_file = fs_anyhow::read_to_string(original_config_path)?;
            info!(
                "Migrating pyright config file from: `{}`",
                original_config_path.display()
            );
            let pyr = serde_jsonrc::from_str::<PyrightConfig>(&raw_file)?;
            pyr.convert()
        } else if original_config_path.file_name() == Some("mypy.ini".as_ref()) {
            info!(
                "Migrating mypy config file from: `{}`",
                original_config_path.display()
            );
            MypyConfig::parse_mypy_config(original_config_path)?
        } else if original_config_path.file_name() == Some("pyproject.toml".as_ref()) {
            match Self::load_from_pyproject(original_config_path) {
                Ok(config) => config,
                Err(e) => {
                    error!("Failed to load config from pyproject.toml: {}", e);
                    return Ok((CommandExitStatus::UserError, None));
                }
            }
        } else {
            error!(
                "Currently only migration from pyrightconfig.json, mypy.ini, and pyproject.toml is supported, not `{}`",
                original_config_path.display(),
            );
            return Ok((CommandExitStatus::UserError, None));
        };

        Self::check_and_warn(&config);

        let pyrefly_config_path = {
            if original_config_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
                original_config_path
            } else {
                &original_config_path.with_file_name(ConfigFile::PYREFLY_FILE_NAME)
            }
        };
        if !pyrefly_config_path
            .parent()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not check if parent directories of `{}` exist",
                    pyrefly_config_path.display()
                )
            })?
            .exists()
        {
            std::fs::create_dir_all(pyrefly_config_path.parent().unwrap())
                .with_context(|| "While trying to write the migrated config file")?;
        }
        if pyrefly_config_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
            write_pyproject(pyrefly_config_path, config)?;
            info!("Config written to `{}`", pyrefly_config_path.display());
        } else {
            let serialized = toml::to_string_pretty(&config)?;
            fs_anyhow::write(pyrefly_config_path, serialized.as_bytes())?;
            info!("New config written to `{}`", pyrefly_config_path.display());
        }
        Ok((
            CommandExitStatus::Success,
            Some(pyrefly_config_path.clone()),
        ))
    }
}

pub fn write_pyproject(pyproject_path: &Path, config: ConfigFile) -> anyhow::Result<()> {
    const ERR_WRITE_CONFIG: &str = "While trying to write Pyrefly config to pyproject.toml file";
    let config_pyproject = PyProject::new(config);
    if pyproject_path.exists() {
        let original_content = fs_anyhow::read_to_string(pyproject_path)?;
        let mut doc = original_content
            .parse::<toml_edit::DocumentMut>()
            .with_context(|| {
                format!(
                    "Failed to parse {} as TOML document",
                    pyproject_path.display()
                )
            })?;
        let toml_string = toml::to_string_pretty(&config_pyproject)?;
        let config_doc = toml_string.parse::<toml_edit::DocumentMut>()?;
        if let Some(tool_table) = config_doc.get("tool")
            && let Some(pyrefly_table) = tool_table.get("pyrefly")
        {
            let tool_entry = doc
                .entry("tool")
                .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
            if let Some(tool_table_mut) = tool_entry.as_table_mut() {
                tool_table_mut.remove("pyrefly");
                let max_tool_pos = tool_table_mut
                    .iter()
                    .filter_map(|(_, v)| v.as_table().and_then(|t| t.position()))
                    .max()
                    .unwrap_or(0);
                tool_table_mut.insert("pyrefly", pyrefly_table.clone());
                if let Some(pyrefly_item) = tool_table_mut.get_mut("pyrefly")
                    && let Some(pyrefly_table_mut) = pyrefly_item.as_table_mut()
                {
                    pyrefly_table_mut.decor_mut().set_prefix("\n");
                    pyrefly_table_mut.set_position(max_tool_pos + 1);
                }
            }
        }
        fs_anyhow::write(pyproject_path, doc.to_string().as_bytes())
            .with_context(|| ERR_WRITE_CONFIG)
    } else {
        let mut serialized_toml = toml::to_string_pretty(&config_pyproject)?;
        if !serialized_toml.contains("[tool.pyrefly]") {
            serialized_toml = String::from("[tool.pyrefly]\n");
        }
        fs_anyhow::write(pyproject_path, serialized_toml.as_bytes())
            .with_context(|| ERR_WRITE_CONFIG)
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::globs::Globs;
    use serde::Deserialize;

    use super::*;

    // helper function for ConfigFile::from_file
    fn from_file(path: &Path) -> anyhow::Result<()> {
        let (_, errs) = ConfigFile::from_file(path);
        if errs.is_empty() {
            Ok(())
        } else {
            for e in errs {
                e.print();
            }
            Err(anyhow::anyhow!(format!(
                "ConfigFile::from_file({}) failed",
                path.display(),
            )))
        }
    }

    #[test]
    fn test_run_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyrightconfig.json");
        let pyr = br#"{
    "include": ["src/**/*.py"]
}
"#;
        fs_anyhow::write(&original_config_path, pyr)?;

        let args = Args {
            original_config_path: Some(original_config_path),
        };
        let (status, pyrefly_config_path) = args.run()?;

        assert!(matches!(status, CommandExitStatus::Success));
        assert!(pyrefly_config_path.is_some());
        let pyrefly_config_path_unwrapped = pyrefly_config_path.unwrap();
        let output = fs_anyhow::read_to_string(&(pyrefly_config_path_unwrapped))?;
        // We're not going to check the whole output because most of it will be default values, which may change.
        // We only actually care about the includes.
        let output_lines = output.lines().collect::<Vec<_>>();
        assert_eq!(output_lines[0], r#"project-includes = ["src/**/*.py"]"#);
        from_file(&pyrefly_config_path_unwrapped)
    }

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("mypy.ini");
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
        fs_anyhow::write(&original_config_path, mypy)?;

        let args = Args {
            original_config_path: Some(original_config_path),
        };
        let (status, pyrefly_config_path) = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        assert!(pyrefly_config_path.is_some());
        let pyrefly_config_path_unwrapped = pyrefly_config_path.unwrap();

        // We care about the config getting serialized in a way that can be checked-in to a repo,
        // i.e. without absolutized paths. So we need to check the raw file.
        #[derive(Deserialize)]
        #[serde(rename_all = "kebab-case")]
        struct CheckConfig {
            project_includes: Vec<String>,
            search_path: Vec<String>,
        }
        let raw_output = fs_anyhow::read_to_string(&pyrefly_config_path_unwrapped)?;
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
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.mypy]
files = ["a.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let args = Args {
            original_config_path: Some(original_config_path.clone()),
        };
        let (status, pyrefly_config_path) = args.run()?;
        assert_eq!(status, CommandExitStatus::Success);
        assert_eq!(pyrefly_config_path.unwrap(), original_config_path);
        let pyproject = fs_anyhow::read_to_string(&original_config_path)?;
        assert_eq!(pyproject.lines().next().unwrap(), "[tool.mypy]");
        assert!(pyproject.contains("[tool.pyrefly]"));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.pyright]
include = ["a.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let args = Args {
            original_config_path: Some(original_config_path.clone()),
        };
        let (status, pyrefly_config_path) = args.run()?;
        assert_eq!(status, CommandExitStatus::Success);
        assert_eq!(pyrefly_config_path.unwrap(), original_config_path);
        let pyproject = fs_anyhow::read_to_string(&original_config_path)?;
        assert_eq!(pyproject.lines().next().unwrap(), "[tool.pyright]");
        assert!(pyproject.contains("[tool.pyrefly]"));
        assert!(!tmp.path().join("pyrefly.toml").exists());
        Ok(())
    }

    #[test]
    fn test_run_pyproject_without_tools() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[project]
name = "test-project"
version = "0.1.0"
description = "A test project"
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let args = Args {
            original_config_path: Some(original_config_path.clone()),
        };
        let result = args.run();
        assert!(matches!(result.unwrap().0, CommandExitStatus::UserError));
        let content = fs_anyhow::read_to_string(&original_config_path)?;
        assert_eq!(content, std::str::from_utf8(pyproject)?);
        Ok(())
    }

    #[test]
    fn test_run_pyproject_bad_mypy_into_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.pyright]
include = ["a.py"]

[tool.mypy]
files = 1
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let args = Args {
            original_config_path: Some(original_config_path),
        };
        let (status, _) = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_mypy_over_pyright() -> anyhow::Result<()> {
        // The current implementation favors mypy over pyright. This test documents that.
        // However, we may want to change this in the future, so it's OK to break this test.
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.pyright]
include = ["pyright.py"]

[tool.mypy]
files = ["mypy.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let cfg = Args::load_from_pyproject(&original_config_path)?;
        assert_eq!(cfg.project_includes, Globs::new(vec!["mypy.py".to_owned()]));
        Ok(())
    }

    #[test]
    fn test_find_config_find_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b/c/");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(&tmp.path().join("a/mypy.ini"), b"[mypy]\n")?;
        fs_anyhow::write(&tmp.path().join("a/pyproject.toml"), b"")?;
        let found = Args::find_config(&bottom)?;
        assert!(found.ends_with("mypy.ini"));
        Ok(())
    }

    #[test]
    fn test_find_config_find_from_dir() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b/c/");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(&tmp.path().join("a/mypy.ini"), b"[mypy]\n")?;
        let (status, _) = Args {
            original_config_path: Some(bottom),
        }
        .run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        assert!(tmp.path().join("a/pyrefly.toml").try_exists()?);
        Ok(())
    }

    #[test]
    fn test_empty_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("mypy.ini");
        let pyrefly_config_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&original_config_path, b"[mypy]\nfake_option = True\n")?;
        let args = Args {
            original_config_path: Some(original_config_path),
        };
        let (status, _) = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&pyrefly_config_path)?;
        assert_eq!(output, "");
        Ok(())
    }

    #[test]
    fn test_empty_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyrightconfig.json");
        let pyrefly_config_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&original_config_path, b"{}")?;
        let args = Args {
            original_config_path: Some(original_config_path),
        };
        let (status, _) = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&pyrefly_config_path)?;
        assert_eq!(output, "");
        Ok(())
    }

    #[test]
    fn test_report_trailing_commas() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py",
                ],
                "pythonVersion": "3.11",
                "reportMissingImports": "none"
            }
            "#;
        let pyr = serde_jsonrc::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert!(!config.project_includes.is_empty());
        Ok(())
    }

    #[test]
    fn test_replace_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let pyproject_path = tmp.path().join("pyproject.toml");

        let existing_content = r#"[project]
name = "test-project"
version = "0.1.0"

[tool.poetry]
dependencies = { python = "^3.8" }

[tool.pyrefly]
project_includes = ["old/path/**/*.py"]
project_excludes = ["should/be/removed.py"]

[tool.black]
line-length = 88
"#;
        fs_anyhow::write(&pyproject_path, existing_content.as_bytes())?;

        let config = ConfigFile {
            project_includes: Globs::new(vec!["new/path/**/*.py".to_owned()]),
            ..Default::default()
        };
        write_pyproject(&pyproject_path, config)?;

        let updated_content = fs_anyhow::read_to_string(&pyproject_path)?;

        assert!(updated_content.contains("[project]"));
        assert!(updated_content.contains("name = \"test-project\""));
        assert!(updated_content.contains("[tool.poetry]"));
        assert!(updated_content.contains("[tool.black]"));

        assert!(updated_content.contains("[tool.pyrefly]"));
        assert!(updated_content.contains("project-includes = [\"new/path/**/*.py\"]"));
        assert!(!updated_content.contains("project_includes = [\"old/path/**/*.py\"]"));
        assert!(!updated_content.contains("project_excludes"));

        Ok(())
    }

    #[test]
    fn test_pyrefly_section_ordering() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let ordering_path = tmp.path().join("ordering_test.toml");
        let existing_content = r#"[project]
name = "test-project"
version = "0.1.0"

# Comment before tool section
[tool]
# Comment within tool section

[tool.black]
line-length = 88

[tool.pytest]
testpaths = ["tests"]

[tool.ruff]
line-length = 88

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
"#;
        fs_anyhow::write(&ordering_path, existing_content.as_bytes())?;

        let config = ConfigFile {
            project_includes: Globs::new(vec!["ordering_test.py".to_owned()]),
            ..Default::default()
        };
        write_pyproject(&ordering_path, config)?;

        let toml_content = fs_anyhow::read_to_string(&ordering_path)?;

        let toml_expected = concat!(
            "[project]\n",
            "name = \"test-project\"\n",
            "version = \"0.1.0\"\n",
            "\n",
            "# Comment before tool section\n",
            "[tool]\n",
            "# Comment within tool section\n",
            "\n",
            "[tool.black]\n",
            "line-length = 88\n",
            "\n",
            "[tool.pytest]\n",
            "testpaths = [\"tests\"]\n",
            "\n",
            "[tool.ruff]\n",
            "line-length = 88\n",
            "\n",
            "[tool.pyrefly]\n",
            "project-includes = [\"ordering_test.py\"]\n",
            "\n",
            "[build-system]\n",
            "requires = [\"setuptools\"]\n",
            "build-backend = \"setuptools.build_meta\"\n",
        );

        assert_eq!(toml_content.trim(), toml_expected.trim());

        Ok(())
    }
}
