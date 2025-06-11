/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use parse_display::Display;
use path_absolutize::Absolutize;
use pyrefly_util::fs_anyhow;
use tracing::error;
use tracing::info;

use crate::commands::config_migration;
use crate::commands::config_migration::write_pyproject;
use crate::commands::run::CommandExitStatus;
use crate::config::config::ConfigFile;

// This should likely be moved into config.rs
#[derive(Clone, Debug, Parser, Copy, Display)]
pub enum ConfigFileKind {
    MyPy,
    Pyright,
    Pyrefly,
    Pyproject,
}

impl ConfigFileKind {
    fn file_name(&self) -> &str {
        match self {
            Self::MyPy => "mypy.ini",
            Self::Pyright => "pyrightconfig.json",
            Self::Pyrefly => "pyrefly.toml",
            Self::Pyproject => "pyproject.toml",
        }
    }

    fn toml_identifier(self) -> String {
        match self {
            // This makes me question if pyproject should be a part of the enum at all
            Self::Pyproject => "".to_owned(),
            _ => format!("[tool.{self}]").to_lowercase(),
        }
    }
}

/// Initialize a new pyrefly config in the given directory. Can also be used to run pyrefly config-migration on a given project.
#[derive(Clone, Debug, Parser)]
#[command(after_help = "Examples:
   `pyrefly init`: Create a new pyrefly.toml config in the current directory
 ")]
pub struct Args {
    /// The path to the project to initialize. Optional. If not present, will create a new pyrefly.toml config in the current directory.
    /// If this is the path to a pyproject.toml, the config will be written as a `[tool.pyrefly]` entry in that file.
    #[arg(default_value_os_t = PathBuf::from("."))]
    path: PathBuf,
}

impl Args {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn new_migration(path: PathBuf) -> Self {
        Self { path }
    }

    fn check_for_pyproject_file(path: &Path) -> bool {
        if path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
            return true;
        }
        let pyproject_path = &path.join(ConfigFile::PYPROJECT_FILE_NAME);
        pyproject_path.exists()
    }

    fn check_for_existing_config(path: &Path, kind: ConfigFileKind) -> anyhow::Result<bool> {
        let file_name = kind.file_name();
        if path.ends_with(file_name) && path.exists() {
            return Ok(true);
        }
        if path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) && path.exists() {
            let raw_pyproject = fs_anyhow::read_to_string(path).with_context(|| {
                format!(
                    "While trying to check for an existing {} config in `{}`",
                    kind,
                    path.display()
                )
            })?;
            return Ok(raw_pyproject.contains(&kind.toml_identifier()));
        }
        if path.is_dir() {
            let custom_file = Args::check_for_existing_config(&path.join(file_name), kind);

            let pyproject =
                Args::check_for_existing_config(&path.join(ConfigFile::PYPROJECT_FILE_NAME), kind);
            return Ok(custom_file? || pyproject?);
        }
        Ok(false)
    }

    fn prompt_user_confirmation(prompt: &str) -> bool {
        if cfg!(test) {
            // decline confirmation, mocking user input
            return false;
        }
        print!("{prompt}");
        std::io::stdout().flush().ok();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).ok();
        let input = input.trim();
        input == "y" || input == "Y"
    }

    pub fn run(&self) -> anyhow::Result<CommandExitStatus> {
        let path = self.path.absolutize()?.to_path_buf();

        let dir: Option<&Path> = if path.is_dir() {
            Some(&path)
        } else {
            path.parent()
        };
        if let Some(dir) = dir
            && Args::check_for_existing_config(dir, ConfigFileKind::Pyrefly)?
        {
            let prompt = format!(
                "The project at `{}` has already been initialized for pyrefly. Run `pyrefly check` to see type errors. Re-initialize and write a new section? (y/N): ",
                dir.display()
            );
            if !Self::prompt_user_confirmation(&prompt) {
                return Ok(CommandExitStatus::UserError);
            }
        }

        // 1. Check for mypy configuration
        let found_mypy = Args::check_for_existing_config(&path, ConfigFileKind::MyPy)?;
        let found_pyright = Args::check_for_existing_config(&path, ConfigFileKind::Pyright)?;
        // 2. Pyrefly configuration

        if found_mypy || found_pyright {
            println!("Found an existing type checking configuration - setting up pyrefly ...");
            let args = config_migration::Args {
                original_config_path: Some(path),
            };
            return args.run();
        }

        // Generate a basic config with a couple sensible defaults.
        // This prevents us from simply outputting an empty file, and gives the user somewhere to start if they want to customize.
        let cfg = ConfigFile {
            project_includes: ConfigFile::default_project_includes(),
            project_excludes: ConfigFile::default_project_excludes(),
            ..Default::default()
        };

        // 3. pyproject.toml configuration
        if Args::check_for_pyproject_file(&path) {
            let config_path = if path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
                path
            } else {
                path.join(ConfigFile::PYPROJECT_FILE_NAME)
            };
            write_pyproject(&config_path, cfg)?;
            info!("Config written to `{}`", config_path.display());
            return Ok(CommandExitStatus::Success);
        }

        let config_path = if path.is_dir() {
            path.join(ConfigFile::PYREFLY_FILE_NAME)
        } else if path.ends_with(ConfigFile::PYREFLY_FILE_NAME) {
            path
        } else if !path.exists() {
            error!("Path `{}` does not exist", path.display());
            return Ok(CommandExitStatus::UserError);
        } else {
            error!(
                "Pyrefly configs must reside in `pyrefly.toml` or `pyproject.toml`, not `{}`",
                path.display()
            );
            return Ok(CommandExitStatus::UserError);
        };
        let serialized = toml::to_string_pretty(&cfg)?;
        fs_anyhow::write(&config_path, serialized.as_bytes())?;
        info!("New config written to `{}`", config_path.display());
        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod test {
    use tempfile;
    use tempfile::TempDir;

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

    fn run_init_on_dir(dir: &TempDir) -> anyhow::Result<CommandExitStatus> {
        let args = Args::new(dir.path().to_path_buf());
        args.run()
    }

    fn run_init_on_file(dir: &TempDir, file: &str) -> anyhow::Result<CommandExitStatus> {
        let args = Args::new(dir.path().join(file));
        args.run()
    }

    fn assert_success(status: CommandExitStatus) {
        assert!(matches!(status, CommandExitStatus::Success), "{status:#?}");
    }

    fn assert_user_error(status: CommandExitStatus) {
        assert!(
            matches!(status, CommandExitStatus::UserError),
            "{status:#?}"
        );
    }

    fn create_file_in(dir: &Path, filename: &str, contents: Option<&[u8]>) -> anyhow::Result<()> {
        let pyrefly_toml = dir.join(filename);
        let contents = contents.unwrap_or(b"");
        fs_anyhow::write(&pyrefly_toml, contents)
    }

    fn check_file_in(dir: &Path, filename: &str, contents: &[&str]) -> anyhow::Result<()> {
        let fi = dir.join(filename);
        let raw_cfg = fs_anyhow::read_to_string(&fi)?;
        for snippet in contents {
            assert!(raw_cfg.contains(snippet), "{snippet}");
        }
        from_file(&fi)
    }

    #[test]
    fn test_empty_dir() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let status = run_init_on_dir(&tmp)?;
        assert_success(status);
        check_file_in(tmp.path(), "pyrefly.toml", &["project-includes"])
    }

    #[test]
    fn test_path_to_new_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let status = run_init_on_file(&tmp, "pyrefly.toml")?;
        assert_success(status);
        check_file_in(tmp.path(), "pyrefly.toml", &["project-includes"])
    }

    #[test]
    fn test_dir_with_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "pyrefly.toml", None)?;
        let status = run_init_on_dir(&tmp)?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_path_to_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "pyrefly.toml", None)?;
        let status = run_init_on_file(&tmp, "pyrefly.toml")?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_dir_with_mypy_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "mypy.ini", Some(b"[mypy]\nfiles = abc"))?;
        let status = run_init_on_dir(&tmp)?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyrefly.toml",
            &["project-includes = [\"abc\"]"],
        )
    }

    #[test]
    fn test_path_to_mypy_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "mypy.ini", Some(b"[mypy]\nfiles = abc"))?;
        let status = run_init_on_file(&tmp, "mypy.ini")?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyrefly.toml",
            &["project-includes = [\"abc\"]"],
        )
    }

    #[test]
    fn test_path_to_nonexistent_mypy_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let status = run_init_on_file(&tmp, "mypy.ini")?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_dir_with_pyright_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyrightconfig.json",
            Some(
                b"\
{
    \"include\": [\"abc\"]
}",
            ),
        )?;
        let status = run_init_on_dir(&tmp)?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyrefly.toml",
            &["project-includes = [\"abc\"]"],
        )
    }

    #[test]
    fn test_path_to_pyright_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyrightconfig.json",
            Some(
                b"\
{
    \"include\": [\"abc\"]
}",
            ),
        )?;
        let status = run_init_on_file(&tmp, "pyrightconfig.json")?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyrefly.toml",
            &["project-includes = [\"abc\"]"],
        )
    }

    #[test]
    fn test_path_to_nonexistent_pyright_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let status = run_init_on_file(&tmp, "pyrightconfig.json")?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_dir_with_pyproject_toml_no_typechecking_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyproject.toml",
            Some(
                b"\
[tool.random_project]
k = \"v\"
",
            ),
        )?;
        let status = run_init_on_dir(&tmp)?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyproject.toml",
            &["tool.random_project", "tool.pyrefly"],
        )
    }

    #[test]
    fn test_dir_with_pyproject_toml_mypy_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyproject.toml",
            Some(
                b"\
[tool.mypy]
files = \"abc\"
",
            ),
        )?;
        let status = run_init_on_dir(&tmp)?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyproject.toml",
            &["tool.mypy", "tool.pyrefly", "project-includes = [\"abc\"]"],
        )
    }

    #[test]
    fn test_dir_with_pyproject_toml_pyright_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyproject.toml",
            Some(
                b"\
[tool.pyright]
include = [\"abc\"]
",
            ),
        )?;
        let status = run_init_on_dir(&tmp)?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyproject.toml",
            &[
                "tool.pyright",
                "tool.pyrefly",
                "project-includes = [\"abc\"]",
            ],
        )
    }

    #[test]
    fn test_path_to_pyproject_toml_pyright_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyproject.toml",
            Some(
                b"\
[tool.pyright]
include = [\"abc\"]
",
            ),
        )?;
        let status = run_init_on_file(&tmp, "pyproject.toml")?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyproject.toml",
            &[
                "tool.pyright",
                "tool.pyrefly",
                "project-includes = [\"abc\"]",
            ],
        )
    }

    #[test]
    fn test_path_to_pyproject_toml_no_typechecking_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(
            tmp.path(),
            "pyproject.toml",
            Some(
                b"\
[tool.random_project]
k = [\"v\"]
",
            ),
        )?;
        let status = run_init_on_file(&tmp, "pyproject.toml")?;
        assert_success(status);
        check_file_in(
            tmp.path(),
            "pyproject.toml",
            &["tool.random_project", "tool.pyrefly"],
        )
    }

    #[test]
    fn test_path_to_nonexistent_pyproject_toml() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let status = run_init_on_file(&tmp, "pyproject.toml")?;
        assert_success(status);
        check_file_in(tmp.path(), "pyproject.toml", &["tool.pyrefly"])
    }

    #[test]
    fn test_dir_with_pyproject_toml_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "pyproject.toml", Some(b"[tool.pyrefly]"))?;
        let status = run_init_on_dir(&tmp)?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_path_to_pyproject_toml_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "pyproject.toml", Some(b"[tool.pyrefly]"))?;
        let status = run_init_on_file(&tmp, "pyproject.toml")?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_config_file_kinds() -> anyhow::Result<()> {
        let kind = ConfigFileKind::MyPy;
        assert_eq!(kind.toml_identifier(), "[tool.mypy]".to_owned());
        Ok(())
    }

    #[test]
    fn test_bad_path() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let status = run_init_on_file(&tmp, "personal_configs.json")?;
        assert_user_error(status);
        Ok(())
    }

    #[test]
    fn test_mypy_config_twice() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "mypy.ini", None)?;
        let status1 = run_init_on_file(&tmp, "mypy.ini")?;
        assert_success(status1);
        let status2 = run_init_on_file(&tmp, "mypy.ini")?;
        assert_user_error(status2);
        Ok(())
    }

    #[test]
    fn test_pyproject_toml_with_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        create_file_in(tmp.path(), "pyrefly.toml", None)?;
        let status = run_init_on_file(&tmp, "pyproject.toml")?;
        assert_user_error(status);
        Ok(())
    }
}
