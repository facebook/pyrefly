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

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::ValueEnum;
use pyrefly_util::fs_anyhow;
use pyrefly_util::upward_search::UpwardSearch;
use tracing::info;
use tracing::warn;

use crate::config::ConfigFile;
use crate::migration::mypy;
use crate::migration::mypy::ini::parse_mypy_config;
use crate::migration::pyright;
use crate::migration::pyright::PyrightConfig;
use crate::pyproject::PyProject;

/// Which type checker config to migrate from.
///
/// When set to a specific source (`MyPy` or `Pyright`), only that source is
/// tried — there is no fallback. `Auto` (the default) tries mypy first, then
/// pyright, preserving the historical behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum MigrationSource {
    /// Automatically detect: try mypy first, then pyright.
    #[default]
    #[value(name = "auto")]
    Auto,
    /// Migrate only from mypy configuration.
    #[value(name = "mypy")]
    MyPy,
    /// Migrate only from pyright configuration.
    #[value(name = "pyright")]
    Pyright,
}

/// Migrate the config file at a given location (pyproject, mypy, pyright etc), producing a new file.
/// In some cases, e.g. pyproject, we will modify the original file in-place.
///
/// When `dry_run` is true, no files are created or modified: the migrated config
/// is computed and printed to the log, and the returned `PathBuf` is the path
/// where the config *would* have been written.
///
/// When `print_config` is true, the migrated config TOML is also written to
/// stdout (independent of `dry_run`), so downstream tooling can capture it.
pub fn config_migration(
    path: &Path,
    migrate_from: MigrationSource,
    dry_run: bool,
    print_config: bool,
) -> anyhow::Result<PathBuf> {
    // TODO: This code is written in a fairly weird style. Give it a nicer interface
    //       without bothering to refactor the internals just yet.
    Args {
        original_config_path: path.to_owned(),
        migrate_from,
        dry_run,
        print_config,
    }
    .run()
}

/// Arguments for configuration migration from other type checkers
#[derive(Clone, Debug)]
struct Args {
    /// The path to the mypy or pyright config file to convert. Optional.
    /// If not provided, or if it's a directory, pyrefly will search upwards for a
    /// mypy.ini, pyrightconfig.json, or pyproject.toml.
    original_config_path: PathBuf,
    /// Which type checker config to migrate from.
    migrate_from: MigrationSource,
    /// When true, do not write any files; just print what would be written.
    dry_run: bool,
    /// When true, also emit the migrated config TOML to stdout.
    print_config: bool,
}

impl Args {
    /// Load config from a pyproject.toml file. When `migrate_from` is set to a
    /// specific source, only try that one. When `Auto`, try mypy first, then
    /// pyright (the historical default).
    fn load_from_pyproject(
        original_config_path: &Path,
        migrate_from: MigrationSource,
    ) -> anyhow::Result<ConfigFile> {
        let raw_file = fs_anyhow::read_to_string(original_config_path)?;
        let parent = original_config_path.parent().unwrap().display();

        let try_mypy = || {
            mypy::parse_pyproject_config(&raw_file).inspect(|_| {
                info!("Migrating [tool.mypy] config from pyproject.toml in `{parent}`")
            })
        };
        let try_pyright = || {
            pyright::parse_pyproject_toml(&raw_file).inspect(|_| {
                info!("Migrating [tool.pyright] config from pyproject.toml in `{parent}`")
            })
        };

        match migrate_from {
            MigrationSource::MyPy => try_mypy(),
            MigrationSource::Pyright => try_pyright(),
            MigrationSource::Auto => try_mypy().or_else(|_| try_pyright()),
        }
    }

    /// Search upward for a config file. When `migrate_from` is set to a
    /// specific source, only look for files matching that source (plus
    /// pyproject.toml). When `Auto`, look for all three in the historical order.
    fn find_config(start: &Path, migrate_from: MigrationSource) -> anyhow::Result<PathBuf> {
        let filenames = match migrate_from {
            MigrationSource::MyPy => {
                vec!["mypy.ini".into(), "pyproject.toml".into()]
            }
            MigrationSource::Pyright => {
                vec!["pyrightconfig.json".into(), "pyproject.toml".into()]
            }
            MigrationSource::Auto => vec![
                "mypy.ini".into(),
                "pyrightconfig.json".into(),
                "pyproject.toml".into(),
            ],
        };
        let searcher = UpwardSearch::new(filenames, |p| std::sync::Arc::new(p.to_path_buf()));
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

    /// This function handles finding the config file if needed, loading it, and converting it to a Pyrefly config.
    /// It returns the config and the path to the original config file.
    fn load_config(&self) -> anyhow::Result<(ConfigFile, PathBuf)> {
        if !self.original_config_path.exists() {
            return Err(anyhow::anyhow!(
                "Could not find or access config file `{}`",
                self.original_config_path.display()
            ));
        }

        let original_config_path = if self.original_config_path.is_file() {
            self.original_config_path.clone()
        } else {
            Self::find_config(&self.original_config_path, self.migrate_from)?
        };

        let config = if original_config_path.file_name() == Some("pyrightconfig.json".as_ref()) {
            info!(
                "Migrating pyright config file from: `{}`",
                original_config_path.display()
            );
            let raw_file = fs_anyhow::read_to_string(&original_config_path)?;
            let pyr = PyrightConfig::parse(&raw_file)?;
            pyr.convert()
        } else if original_config_path.file_name() == Some("mypy.ini".as_ref()) {
            info!(
                "Migrating mypy config file from: `{}`",
                original_config_path.display()
            );
            parse_mypy_config(&original_config_path)?
        } else if original_config_path.file_name() == Some("pyproject.toml".as_ref()) {
            Self::load_from_pyproject(&original_config_path, self.migrate_from)
                .context("Failed to load config from pyproject.toml")?
        } else {
            return Err(anyhow::anyhow!(
                "Currently only migration from pyrightconfig.json, mypy.ini, and pyproject.toml is supported, not `{}`",
                original_config_path.display(),
            ));
        };

        Self::check_and_warn(&config);

        Ok((config, original_config_path))
    }

    fn run(&self) -> anyhow::Result<PathBuf> {
        let (config, original_config_path) = self.load_config()?;

        let pyrefly_config_path = {
            if original_config_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
                original_config_path
            } else {
                original_config_path.with_file_name(ConfigFile::PYREFLY_FILE_NAME)
            }
        };
        if !self.dry_run
            && !pyrefly_config_path
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
        // The pyproject non-dry-run path doesn't need a local serialization:
        // PyProject::update re-serializes internally. Handle it as a fast
        // path so the default migration doesn't pay for an unused TOML
        // serialize.
        if pyrefly_config_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) && !self.dry_run {
            if self.print_config {
                let serialized = toml::to_string_pretty(&config)?;
                print!("{serialized}");
                std::io::stdout().flush().ok();
            }
            PyProject::update(&pyrefly_config_path, config)?;
            info!("Config written to `{}`", pyrefly_config_path.display());
            return Ok(pyrefly_config_path);
        }

        let serialized = toml::to_string_pretty(&config)?;
        if pyrefly_config_path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) {
            info!(
                "Dry run: would insert [tool.pyrefly] section into `{}`:\n{}",
                pyrefly_config_path.display(),
                serialized
            );
        } else if self.dry_run {
            info!(
                "Dry run: would write new config to `{}`:\n{}",
                pyrefly_config_path.display(),
                serialized
            );
        } else {
            fs_anyhow::write(&pyrefly_config_path, &serialized)?;
            info!("New config written to `{}`", pyrefly_config_path.display());
        }
        if self.print_config {
            print!("{serialized}");
            std::io::stdout().flush().ok();
        }
        Ok(pyrefly_config_path)
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::globs::Globs;
    use serde::Deserialize;

    use super::*;
    use crate::error_kind::ErrorKind;
    use crate::error_kind::Severity;

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
        let pyr = r#"{
    "include": ["src/**/*.py"]
}
"#;
        fs_anyhow::write(&original_config_path, pyr)?;

        let pyrefly_config_path =
            config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
        let output = fs_anyhow::read_to_string(&pyrefly_config_path)?; // We're not going to check the whole output because most of it will be default values, which may change.
        // We only actually care about the includes.
        let output_lines = output.lines().collect::<Vec<_>>();
        assert_eq!(output_lines[0], r#"project-includes = ["src/**/*.py"]"#);
        from_file(&pyrefly_config_path)
    }

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = r#"[mypy]
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

        let pyrefly_config_path =
            config_migration(&original_config_path, MigrationSource::Auto, false, false)?;

        // We care about the config getting serialized in a way that can be checked-in to a repo,
        // i.e. without absolutized paths. So we need to check the raw file.
        #[derive(Deserialize)]
        #[serde(rename_all = "kebab-case")]
        struct CheckConfig {
            project_includes: Vec<String>,
            search_path: Vec<String>,
        }
        let raw_output = fs_anyhow::read_to_string(&pyrefly_config_path)?;
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
        let pyproject = r#"[tool.mypy]
files = ["a.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let pyrefly_config_path =
            config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
        assert_eq!(pyrefly_config_path, original_config_path);
        let pyproject = fs_anyhow::read_to_string(&original_config_path)?;
        assert_eq!(pyproject.lines().next().unwrap(), "[tool.mypy]");
        assert!(pyproject.contains("[tool.pyrefly]"));
        Ok(())
    }

    #[test]
    fn test_run_pyproject_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = r#"[tool.pyright]
include = ["a.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
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
        let pyproject = r#"[project]
name = "test-project"
version = "0.1.0"
description = "A test project"
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        assert!(
            config_migration(&original_config_path, MigrationSource::Auto, false, false).is_err()
        );
        let content = fs_anyhow::read_to_string(&original_config_path)?;
        assert_eq!(content, pyproject);
        Ok(())
    }

    #[test]
    fn test_run_pyproject_bad_mypy_into_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = r#"[tool.pyright]
include = ["a.py"]

[tool.mypy]
files = 1
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
        Ok(())
    }

    #[test]
    fn test_run_pyproject_mypy_over_pyright() -> anyhow::Result<()> {
        // The default (Auto) favors mypy over pyright. This test documents that.
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = r#"[tool.pyright]
include = ["pyright.py"]

[tool.mypy]
files = ["mypy.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let cfg = Args::load_from_pyproject(&original_config_path, MigrationSource::Auto)?;
        assert_eq!(
            cfg.project_includes,
            Globs::new(vec!["mypy.py".to_owned()]).unwrap()
        );
        Ok(())
    }

    #[test]
    fn test_run_pyproject_migrate_from_pyright() -> anyhow::Result<()> {
        // When migrate_from is Pyright, pyright config is picked even when mypy is also present.
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = r#"[tool.pyright]
include = ["pyright.py"]

[tool.mypy]
files = ["mypy.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject)?;
        let cfg = Args::load_from_pyproject(&original_config_path, MigrationSource::Pyright)?;
        assert_eq!(
            cfg.project_includes,
            Globs::new(vec!["pyright.py".to_owned()]).unwrap()
        );
        Ok(())
    }

    #[test]
    fn test_run_pyproject_migrate_from_mypy_missing() {
        // When migrate_from is MyPy but only pyright config exists, migration should fail
        // (no fallback to pyright).
        let tmp = tempfile::tempdir().unwrap();
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = r#"[tool.pyright]
include = ["pyright.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject).unwrap();
        assert!(Args::load_from_pyproject(&original_config_path, MigrationSource::MyPy).is_err());
    }

    #[test]
    fn test_run_pyproject_migrate_from_pyright_missing() {
        // When migrate_from is Pyright but only mypy config exists, migration should fail
        // (no fallback to mypy).
        let tmp = tempfile::tempdir().unwrap();
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = r#"[tool.mypy]
files = ["mypy.py"]
"#;
        fs_anyhow::write(&original_config_path, pyproject).unwrap();
        assert!(
            Args::load_from_pyproject(&original_config_path, MigrationSource::Pyright).is_err()
        );
    }

    #[test]
    fn test_find_config_find_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b/c/");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(&tmp.path().join("a/mypy.ini"), b"[mypy]\n")?;
        fs_anyhow::write(&tmp.path().join("a/pyproject.toml"), b"")?;
        let found = Args::find_config(&bottom, MigrationSource::Auto)?;
        assert!(found.ends_with("mypy.ini"));
        Ok(())
    }

    #[test]
    fn test_find_config_find_from_dir() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b/c/");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(&tmp.path().join("a/mypy.ini"), b"[mypy]\n")?;
        config_migration(&bottom, MigrationSource::Auto, false, false)?;
        assert!(tmp.path().join("a/pyrefly.toml").try_exists()?);
        Ok(())
    }

    #[test]
    fn test_empty_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("mypy.ini");
        let pyrefly_config_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&original_config_path, b"[mypy]\nfake_option = True\n")?;
        config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
        let output = fs_anyhow::read_to_string(&pyrefly_config_path)?;
        // mutable-override is disabled by default when migrating from mypy
        assert_eq!(
            output.trim(),
            "check-unannotated-defs = false\ninfer-return-types = \"never\"\n\n[errors]\nbad-override-mutable-attribute = \"ignore\""
        );
        Ok(())
    }

    /// Tests that a migrated mypy config with both root-level and per-module
    /// error codes produces a pyrefly config where the sub-config inherits the
    /// root's error overrides. Before the sub-config error merging fix, the
    /// sub-config's error map would completely replace the root's, causing the
    /// root-level `override` disable to be lost for matching files.
    #[test]
    fn test_mypy_subconfig_inherits_root_errors() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("mypy.ini");
        let pyrefly_config_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(
            &original_config_path,
            b"[mypy]\ndisable_error_code = override\n\n[mypy-app.models]\ndisable_error_code = name-defined\n",
        )?;
        config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
        let output = fs_anyhow::read_to_string(&pyrefly_config_path)?;

        // Parse the generated config and verify
        let mut config: ConfigFile = toml::from_str(&output)?;
        config.configure();

        // Sub-config file should inherit root's bad-override=ignore
        let sub_errors = config.errors(Path::new("app/models/foo.py"));
        assert_eq!(
            sub_errors.severity(ErrorKind::BadOverride),
            Severity::Ignore
        );
        assert_eq!(
            sub_errors.severity(ErrorKind::UnknownName),
            Severity::Ignore
        );

        // Non-matching file should only have root errors
        let root_errors = config.errors(Path::new("other.py"));
        assert_eq!(
            root_errors.severity(ErrorKind::BadOverride),
            Severity::Ignore
        );
        assert_eq!(
            root_errors.severity(ErrorKind::UnknownName),
            Severity::Error
        );

        Ok(())
    }

    #[test]
    fn test_empty_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyrightconfig.json");
        let pyrefly_config_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&original_config_path, b"{}")?;
        config_migration(&original_config_path, MigrationSource::Auto, false, false)?;
        let output = fs_anyhow::read_to_string(&pyrefly_config_path)?;
        assert_eq!(output, "infer-with-first-use = false\n");
        Ok(())
    }

    #[test]
    fn test_dry_run_does_not_write_pyrefly_toml() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("mypy.ini");
        let pyrefly_config_path = tmp.path().join("pyrefly.toml");
        fs_anyhow::write(&original_config_path, b"[mypy]\nfiles = abc\n")?;

        let returned = config_migration(&original_config_path, MigrationSource::Auto, true, false)?;
        assert_eq!(returned, pyrefly_config_path);
        assert!(!pyrefly_config_path.exists());
        Ok(())
    }

    #[test]
    fn test_dry_run_does_not_modify_pyproject_toml() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let original_config_path = tmp.path().join("pyproject.toml");
        let pyproject = "[tool.mypy]\nfiles = [\"a.py\"]\n";
        fs_anyhow::write(&original_config_path, pyproject)?;

        config_migration(&original_config_path, MigrationSource::Auto, true, false)?;
        let unchanged = fs_anyhow::read_to_string(&original_config_path)?;
        assert_eq!(unchanged, pyproject);
        Ok(())
    }
}
