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
use crate::util::fs_anyhow;
use crate::util::upward_search::UpwardSearch;

#[derive(Clone, Debug, Parser)]
pub struct Args {
    /// The path to the mypy or pyright config file to convert. Optional.
    /// If not provided, or if it's a directory, pyrefly will search upwards for a
    /// mypy.ini, pyrightconfig.json, or pyproject.toml.
    pub input_path: Option<PathBuf>,
    /// Optional path to write the pyrefly config file to. If not provided, the config will be written to the same directory as the input file.
    /// Must end in pyrefly.toml or pyproject.toml.
    pub output_path: Option<PathBuf>,
}

impl Args {
    fn load_from_pyproject(raw_file: &str) -> anyhow::Result<ConfigFile> {
        match mypy::parse_pyrproject_config(raw_file) {
            ok @ Ok(_) => {
                info!("Migrating [tool.mypy] config from pyproject.toml");
                return ok;
            }
            Err(_) => {
                // Try to parse [tool.pyright] instead.
            }
        }
        pyright::parse_pyproject_toml(raw_file)
            .inspect(|_| info!("Migrating [tool.pyright] config from pyproject.toml"))
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

    pub fn run(&self) -> anyhow::Result<CommandExitStatus> {
        if let Some(path) = self.input_path.as_ref() {
            if !path.exists() {
                error!("Could not find or cannot access `{}`", path.display());
                return Ok(CommandExitStatus::InfraError);
            }
        }

        if self.output_path.as_ref().is_some_and(|p| {
            !(p.ends_with(ConfigFile::PYREFLY_FILE_NAME)
                || p.ends_with(ConfigFile::PYPROJECT_FILE_NAME))
        }) {
            error!("Output path must end in pyrefly.toml or pyproject.toml");
            return Ok(CommandExitStatus::UserError);
        }

        let input_path = match &self.input_path {
            Some(path) if path.is_file() => path,
            Some(path) => &Self::find_config(path)?,
            None => {
                let cwd = std::env::current_dir()
                    .context("Could not find dir to start search for configs from")?;
                &Self::find_config(&cwd)?
            }
        };
        let config = if input_path.file_name() == Some("pyrightconfig.json".as_ref()) {
            let raw_file = fs_anyhow::read_to_string(input_path)?;
            info!("Migrating pyright config file");
            let pyr = serde_jsonrc::from_str::<PyrightConfig>(&raw_file)?;
            pyr.convert()
        } else if input_path.file_name() == Some("mypy.ini".as_ref()) {
            info!("Migrating mypy config file");
            MypyConfig::parse_mypy_config(input_path)?
        } else if input_path.file_name() == Some("pyproject.toml".as_ref()) {
            let raw_file = fs_anyhow::read_to_string(input_path)?;
            Self::load_from_pyproject(&raw_file)?
        } else {
            error!(
                "Currently only migration from pyrightconfig.json, mypy.ini, and pyproject.toml is supported, not `{}`",
                input_path.display(),
            );
            return Ok(CommandExitStatus::UserError);
        };

        Self::check_and_warn(&config);

        let output_path = match &self.output_path {
            Some(path) => path,
            None => {
                if input_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
                    input_path
                } else {
                    &input_path.with_file_name(ConfigFile::PYREFLY_FILE_NAME)
                }
            }
        };
        if !output_path
            .parent()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not check if parent directories of `{}` exist",
                    output_path.display()
                )
            })?
            .exists()
        {
            std::fs::create_dir_all(output_path.parent().unwrap())
                .with_context(|| "While trying to write the migrated config file")?;
        }
        if output_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
            write_pyproject(output_path, config)?;
            info!("Config written to `{}`", output_path.display());
        } else {
            let serialized = toml::to_string_pretty(&config)?;
            fs_anyhow::write(output_path, serialized.as_bytes())?;
            info!("New config written to `{}`", output_path.display());
        }
        Ok(CommandExitStatus::Success)
    }
}

pub fn write_pyproject(pyproject_path: &Path, config: ConfigFile) -> anyhow::Result<()> {
    const ERR_WRITE_CONFIG: &str = "While trying to write pyrefly config to pyproject.toml file";

    if pyproject_path.exists() {
        let original_content = fs_anyhow::read_to_string(pyproject_path)?;
        let doc = original_content
            .parse::<toml_edit::DocumentMut>()
            .with_context(|| {
                format!(
                    "Failed to parse {} as TOML document",
                    pyproject_path.display()
                )
            })?;

        let config_pyproject = PyProject::new(config.clone());
        let toml = toml::to_string_pretty(&config_pyproject)?;
        let pyrefly_section = if let Some(start) = toml.find("[tool.pyrefly]") {
            toml[start..].trim_end().to_string()
        } else {
            String::from("[tool.pyrefly]")
        };
        let mut updated_content = doc.to_string();
        updated_content = updated_content.trim_end().to_string();
        loop {
            if let Some(start) = updated_content.find("[tool.pyrefly]") {
                let after = &updated_content[start + 1..];
                let next_section = after.find('\n').and_then(|nl| {
                    let rest = &after[nl + 1..];
                    rest.find("\n[").map(|i| nl + 1 + i)
                });
                let end = next_section
                    .map(|i| start + 1 + i + 1)
                    .unwrap_or(updated_content.len());
                let mut cleaned = String::new();
                cleaned.push_str(&updated_content[..start]);
                cleaned.push_str(&updated_content[end..]);
                updated_content = cleaned.trim_end().to_string();
            } else {
                break;
            }
        }
        if pyrefly_section.trim() == "[tool.pyrefly]" {
            if updated_content.is_empty() {
                updated_content = pyrefly_section;
            } else {
                updated_content.push_str("\n\n");
                updated_content.push_str(&pyrefly_section);
            }
        } else {
            if updated_content.is_empty() {
                updated_content = pyrefly_section;
            } else {
                updated_content.push_str("\n\n");
                updated_content.push_str(&pyrefly_section);
            }
        }
        updated_content.push('\n');
        fs_anyhow::write(pyproject_path, updated_content.as_bytes())
            .with_context(|| ERR_WRITE_CONFIG)
    } else {
        let config_pyproject = PyProject::new(config);
        let mut serialized = toml::to_string_pretty(&config_pyproject)?;
        if !serialized.contains("[tool.pyrefly]") {
            serialized = String::from("[tool.pyrefly]\n");
        } else {
            serialized = serialized.trim().to_string();
            serialized = format!("{}\n", serialized);
            if serialized
                .trim_start_matches('\n')
                .starts_with("[tool.pyrefly]")
                && !serialized.trim_start_matches('\n').contains('\n')
            {
                serialized = String::from("[tool.pyrefly]\n");
            }
        }
        fs_anyhow::write(pyproject_path, serialized.as_bytes()).with_context(|| ERR_WRITE_CONFIG)
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::util::globs::Globs;

    // helper function for ConfigFile::from_file
    fn from_file(path: &Path) -> anyhow::Result<()> {
        let (_, errs) = ConfigFile::from_file(path);
        if errs.is_empty() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(format!(
                "ConfigFile::from_file({}) failed: {:#?}",
                path.display(),
                ConfigFile::from_file(path).1
            )))
        }
    }

    #[test]
    fn test_run_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        let pyr = br#"{
    "include": ["src/**/*.py"]
}
"#;
        fs_anyhow::write(&input_path, pyr)?;
        let output_path = input_path.with_file_name(ConfigFile::PYREFLY_FILE_NAME);

        let args = Args {
            input_path: Some(input_path),
            output_path: None,
        };
        let status = args.run()?;

        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&output_path)?;
        // We're not going to check the whole output because most of it will be default values, which may change.
        // We only actually care about the includes.
        let output_lines = output.lines().collect::<Vec<_>>();
        assert_eq!(output_lines[0], r#"project_includes = ["src/**/*.py"]"#);
        from_file(&output_path)
    }

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let output_path = input_path.with_file_name(ConfigFile::PYREFLY_FILE_NAME);
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
        fs_anyhow::write(&input_path, mypy)?;

        let args = Args {
            input_path: Some(input_path),
            output_path: None,
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));

        // We care about the config getting serialized in a way that can be checked-in to a repo,
        // i.e. without absolutized paths. So we need to check the raw file.
        #[derive(Deserialize)]
        struct CheckConfig {
            project_includes: Vec<String>,
            search_path: Vec<String>,
        }
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
        let input_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.mypy]
files = ["a.py"]
"#;
        fs_anyhow::write(&input_path, pyproject)?;
        let args = Args {
            input_path: Some(input_path),
            output_path: None,
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let input_path = args.input_path.unwrap();
        let pyproject = fs_anyhow::read_to_string(&input_path)?;
        assert_eq!(pyproject.lines().next().unwrap(), "[tool.mypy]");
        assert!(pyproject.contains("[tool.pyrefly]"));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.pyright]
include = ["a.py"]
"#;
        fs_anyhow::write(&input_path, pyproject)?;
        let args = Args {
            input_path: Some(input_path),
            output_path: None,
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let input_path = args.input_path.unwrap();
        let pyproject = fs_anyhow::read_to_string(&input_path)?;
        assert_eq!(pyproject.lines().next().unwrap(), "[tool.pyright]");
        assert!(pyproject.contains("[tool.pyrefly]"));
        assert!(!tmp.path().join("pyrefly.toml").exists());
        Ok(())
    }

    #[test]
    fn test_run_pyproject_bad_mypy_into_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyproject.toml");
        let pyproject = br#"[tool.pyright]
include = ["a.py"]

[tool.mypy]
files = 1
"#;
        fs_anyhow::write(&input_path, pyproject)?;
        let args = Args {
            input_path: Some(input_path),
            output_path: None,
        };
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
        let status = Args {
            input_path: Some(bottom),
            output_path: None,
        }
        .run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        assert!(tmp.path().join("a/pyrefly.toml").try_exists()?);
        Ok(())
    }

    #[test]
    fn test_existing_file_to_new_pyproject() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        let pyr = br#"{
    "include": ["src/**/*.py"]
}
"#;
        fs_anyhow::write(&input_path, pyr)?;
        let output_path = input_path.with_file_name(ConfigFile::PYPROJECT_FILE_NAME);

        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;

        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&output_path)?;
        assert!(output.contains("[tool.pyrefly]"));
        assert!(output.contains(r#"src/**/*.py"#));
        assert!(!tmp.path().join("pyrefly.toml").exists());
        Ok(())
    }

    /*
    #[ignore = "Cannot edit toml files in a way that preserves order and comments"]
    #[test]
    fn test_overwrite_existing_toolpyrefly() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        fs_anyhow::write(&input_path, b"{}")?;
        let output_path = tmp.path().join("pyproject.toml");
        fs_anyhow::write(
            &output_path,
            br#"[project]
name = "test"

[tool.a]
test = true

[tool.z]
test = true

# I'm a comment!

[tool.pyrefly]
incomplete = true

[tool.b]
test = true
"#,
        )?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));

        let output = fs_anyhow::read_to_string(&output_path)?;
        let headers = output
            .lines()
            .filter(|l| l.starts_with(['[', '#']))
            .collect::<Vec<_>>();
        assert_eq!(
            headers,
            vec![
                "[project]",
                "[tool.a]",
                "[tool.z]",
                "# I'm a comment!",
                "[tool.pyrefly]",
                "[tool.b]",
            ]
        );

        let config = toml::from_str::<toml::Table>(&output)?;
        assert!(
            !config["tool"]["pyrefly"]
                .as_table()
                .is_some_and(|table| table.contains_key("incomplete"))
        );

        Ok(())
    }
    */

    #[test]
    fn test_output_path_must_be_file() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        fs_anyhow::write(&input_path, b"{}")?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(tmp.path().to_path_buf()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::UserError));
        Ok(())
    }

    #[test]
    fn test_output_path_create_dir_if_needed() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        let output_path = tmp.path().join("a").join("pyrefly.toml");
        fs_anyhow::write(&input_path, b"{}")?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        assert!(output_path.exists());
        from_file(&output_path)
    }

    #[test]
    fn test_empty_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let output_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&input_path, b"[mypy]\nfake_option = True\n")?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&output_path)?;
        assert_eq!(output, "");
        Ok(())
    }

    #[test]
    fn test_empty_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        let output_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&input_path, b"{}")?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&output_path)?;
        assert_eq!(output, "");
        Ok(())
    }

    #[test]
    fn test_empty_mypy_pyproject() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let output_path = tmp.path().join("pyproject.toml");
        fs_anyhow::write(&input_path, b"[mypy]\nfake_option = True\n")?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&output_path)?;
        assert_eq!(output, "[tool.pyrefly]\n");
        Ok(())
    }

    #[test]
    fn test_empty_pyright_pyproject() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("pyrightconfig.json");
        let output_path = tmp.path().join("pyproject.toml");
        fs_anyhow::write(&input_path, b"{}")?;
        let args = Args {
            input_path: Some(input_path),
            output_path: Some(output_path.clone()),
        };
        let status = args.run()?;
        assert!(matches!(status, CommandExitStatus::Success));
        let output = fs_anyhow::read_to_string(&output_path)?;
        assert_eq!(output, "[tool.pyrefly]\n");
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

[tool.black]
line-length = 88
"#;
        fs_anyhow::write(&pyproject_path, existing_content.as_bytes())?;

        let mut config = ConfigFile::default();
        config.project_includes = Globs::new(vec!["new/path/**/*.py".to_owned()]);

        write_pyproject(&pyproject_path, config)?;

        let updated_content = fs_anyhow::read_to_string(&pyproject_path)?;

        assert!(updated_content.contains("[project]"));
        assert!(updated_content.contains("name = \"test-project\""));
        assert!(updated_content.contains("[tool.poetry]"));
        assert!(updated_content.contains("[tool.black]"));

        assert!(updated_content.contains("[tool.pyrefly]"));
        assert!(updated_content.contains("project_includes = [\"new/path/**/*.py\"]"));
        assert!(!updated_content.contains("project_includes = [\"old/path/**/*.py\"]"));

        Ok(())
    }

    #[test]
    fn test_pyrefly_section_ordering() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let new_pyproject_path = tmp.path().join("new_pyproject.toml");

        let mut config = ConfigFile::default();
        config.project_includes = Globs::new(vec!["test/**/*.py".to_owned()]);

        let config_pyproject = PyProject::new(config);
        let serialized = toml::to_string_pretty(&config_pyproject)?;
        fs_anyhow::write(&new_pyproject_path, serialized.as_bytes())?;

        let content = fs_anyhow::read_to_string(&new_pyproject_path)?;
        println!("NEW FILE CONTENT:\n{}", content);
        assert!(content.contains("[tool.pyrefly]"));

        let existing_file_path = tmp.path().join("existing_file.toml");
        fs_anyhow::write(&existing_file_path, b"[tool]\n# Some comment\n")?;

        let mut config = ConfigFile::default();
        config.project_includes = Globs::new(vec!["test/**/*.py".to_owned()]);
        write_pyproject(&existing_file_path, config)?;

        let content = fs_anyhow::read_to_string(&existing_file_path)?;
        println!("EXISTING FILE CONTENT:\n{}", content);
        assert!(content.contains("[tool.pyrefly]"));

        let mypy_path = tmp.path().join("mypy_test.toml");
        fs_anyhow::write(&mypy_path, b"[tool.mypy]\nfiles = [\"a.py\"]\n")?;
        let mut config = ConfigFile::default();
        config.project_includes = Globs::new(vec!["mypy_test.py".to_owned()]);
        write_pyproject(&mypy_path, config)?;

        let content = fs_anyhow::read_to_string(&mypy_path)?;
        println!("MYPY TEST CONTENT:\n{}", content);
        assert!(content.contains("[tool.pyrefly]"));

        let ordering_path = tmp.path().join("ordering_test.toml");
        let existing_content = r#"[project]
name = "test-project"
version = "0.1.0"

# Comment before tool section
[tool]
# Comment within tool section

[tool.a]
option = "first"
"#;
        fs_anyhow::write(&ordering_path, existing_content.as_bytes())?;

        let mut config = ConfigFile::default();
        config.project_includes = Globs::new(vec!["ordering_test.py".to_owned()]);
        write_pyproject(&ordering_path, config)?;

        let content = fs_anyhow::read_to_string(&ordering_path)?;
        println!("ORDERING TEST CONTENT:\n{}", content);
        assert!(content.contains("[tool.pyrefly]"));

        Ok(())
    }
}
