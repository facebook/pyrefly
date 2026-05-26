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
use std::sync::Arc;

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

/// Which type checker config we successfully migrated from, plus the
/// kind of file we read it out of. Used by the LSP/CLI to label the
/// synthesized config and explain to the user where the imported
/// settings came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MigratedFromKind {
    Mypy(MigratedConfigSource),
    Pyright(MigratedConfigSource),
}

/// Where the migrated mypy / pyright settings physically lived: a
/// dedicated config file (`mypy.ini` / `pyrightconfig.json`), or a
/// `[tool.mypy]` / `[tool.pyright]` section of a `pyproject.toml`. The
/// surfaced wording differs ("`mypy.ini`" vs "`[tool.mypy]` in
/// `pyproject.toml`"), but the migrated settings themselves don't.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MigratedConfigSource {
    /// `mypy.ini` or `pyrightconfig.json`.
    DedicatedFile,
    /// `[tool.mypy]` or `[tool.pyright]` section of a `pyproject.toml`.
    PyprojectToml,
}

/// Search upward from `start` for a mypy or pyright config (`mypy.ini`,
/// `pyrightconfig.json`, or a `pyproject.toml` with `[tool.mypy]` /
/// `[tool.pyright]`) and migrate it to a Pyrefly `ConfigFile` entirely in
/// memory — no files are written. The result is equivalent to what
/// `pyrefly init` would produce, only without touching disk.
///
/// Returns `Ok(None)` if no migrate-able config exists. Returns `Err` if a
/// candidate config was found but couldn't be parsed; callers that want to
/// fall back to a plain `Basic` preset on parse failure should catch the
/// error themselves (see `resolve_unconfigured_config`).
pub fn find_and_migrate_in_memory(
    start: &Path,
) -> anyhow::Result<Option<(ConfigFile, MigratedFromKind)>> {
    let Some(path) = find_upward_config(start, MigrationSource::Auto) else {
        return Ok(None);
    };
    if path.file_name() == Some("pyrightconfig.json".as_ref()) {
        let raw_file = fs_anyhow::read_to_string(&path)
            .with_context(|| format!("While reading pyright config at {}", path.display()))?;
        let pyr = PyrightConfig::parse(&raw_file)
            .with_context(|| format!("While parsing pyright config at {}", path.display()))?;
        Ok(Some((
            pyr.convert(),
            MigratedFromKind::Pyright(MigratedConfigSource::DedicatedFile),
        )))
    } else if path.file_name() == Some("mypy.ini".as_ref()) {
        let cfg = parse_mypy_config(&path)?;
        Ok(Some((
            cfg,
            MigratedFromKind::Mypy(MigratedConfigSource::DedicatedFile),
        )))
    } else if path.file_name() == Some("pyproject.toml".as_ref()) {
        let raw_file = fs_anyhow::read_to_string(&path)
            .with_context(|| format!("While reading pyproject.toml at {}", path.display()))?;
        // Try mypy first, then pyright. Matches the historical Auto order.
        //
        // TODO: this falls back to pyright on *any* mypy parse error,
        // not just "section missing" — so a malformed `[tool.mypy]`
        // silently becomes a pyright migration when both sections are
        // present. `Args::load_from_pyproject` (used by `pyrefly init`)
        // has the same shape. Ideally section presence (not parse
        // success) would decide which tool migrates; change both
        // together.
        match mypy::parse_pyproject_config(&raw_file) {
            Ok(cfg) => Ok(Some((
                cfg,
                MigratedFromKind::Mypy(MigratedConfigSource::PyprojectToml),
            ))),
            Err(mypy_err) => match pyright::parse_pyproject_toml(&raw_file) {
                Ok(cfg) => Ok(Some((
                    cfg,
                    MigratedFromKind::Pyright(MigratedConfigSource::PyprojectToml),
                ))),
                Err(pyright_err) => {
                    let has_mypy = raw_file.contains("[tool.mypy]");
                    let has_pyright = raw_file.contains("[tool.pyright]");
                    if !has_mypy && !has_pyright {
                        // No tool sections at all — this isn't a
                        // migrate-able config, not a parse error.
                        // Treat as "nothing nearby."
                        Ok(None)
                    } else if has_pyright && !has_mypy {
                        // Only `[tool.pyright]` is present, so a mypy
                        // parse error is "section missing" — surface
                        // the pyright error instead, since that's
                        // what the user actually has and needs to
                        // fix.
                        Err(pyright_err.context(format!(
                            "While parsing pyproject.toml at {}",
                            path.display()
                        )))
                    } else {
                        // `[tool.mypy]` is present (alone or
                        // alongside pyright). Mypy was tried first;
                        // surface its error.
                        Err(mypy_err.context(format!(
                            "While parsing pyproject.toml at {}",
                            path.display()
                        )))
                    }
                }
            },
        }
    } else {
        // `find_upward_config(_, Auto)` only returns one of the three
        // filenames handled above, so no other filename can reach
        // here. Panic loudly if a future change adds a new search
        // candidate without a matching arm.
        unreachable!(
            "find_upward_config returned an unexpected filename: {}",
            path.display()
        )
    }
}

/// Search upward from `start` for the first matching config file. Returns
/// `None` if nothing is found. Used by both the migration command (which
/// requires a config to be present) and the in-memory helper (which is
/// fine with `None`).
fn find_upward_config(start: &Path, migrate_from: MigrationSource) -> Option<PathBuf> {
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
    let searcher = UpwardSearch::new(filenames, |p| Arc::new(p.to_path_buf()));
    searcher.directory(start).map(Arc::unwrap_or_clone)
}

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
            find_upward_config(&self.original_config_path, self.migrate_from)
                .ok_or_else(|| anyhow::anyhow!("Failed to find config"))?
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
        let found =
            find_upward_config(&bottom, MigrationSource::Auto).expect("config should be found");
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
        // mypy migration uses the `legacy` preset; no redundant explicit settings
        assert_eq!(output.trim(), "preset = \"legacy\"");
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

    /// Tests for the in-memory loader used by `resolve_unconfigured_config`.
    /// These mirror the on-disk `config_migration` tests but assert that no
    /// `pyrefly.toml` is written and that the resulting `ConfigFile` carries
    /// the expected migrated settings.
    #[test]
    fn test_in_memory_finds_mypy_ini() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(
            &tmp.path().join("a/mypy.ini"),
            b"[mypy]\ncheck_untyped_defs = True\n",
        )?;

        let (cfg, kind) = find_and_migrate_in_memory(&bottom)?.expect("should find mypy.ini");
        assert_eq!(
            kind,
            MigratedFromKind::Mypy(MigratedConfigSource::DedicatedFile)
        );
        // Mypy migration sets the legacy preset.
        assert_eq!(cfg.preset, Some(crate::base::Preset::Legacy));
        // Full migration also sets check_unannotated_defs (mypy's
        // `check_untyped_defs` maps to pyrefly's `check_unannotated_defs`).
        assert_eq!(cfg.root.check_unannotated_defs, Some(true));
        // No file written.
        assert!(!tmp.path().join("a/pyrefly.toml").exists());
        Ok(())
    }

    #[test]
    fn test_in_memory_finds_pyrightconfig() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(
            &tmp.path().join("a/pyrightconfig.json"),
            br#"{ "include": ["src/**/*.py"] }"#,
        )?;

        let (cfg, kind) =
            find_and_migrate_in_memory(&bottom)?.expect("should find pyrightconfig.json");
        assert_eq!(
            kind,
            MigratedFromKind::Pyright(MigratedConfigSource::DedicatedFile)
        );
        assert_eq!(
            cfg.project_includes,
            Globs::new(vec!["src/**/*.py".to_owned()]).unwrap()
        );
        assert!(!tmp.path().join("a/pyrefly.toml").exists());
        Ok(())
    }

    #[test]
    fn test_in_memory_finds_pyproject_with_tool_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(
            &tmp.path().join("a/pyproject.toml"),
            b"[tool.mypy]\nfiles = [\"a.py\"]\n",
        )?;

        let (_cfg, kind) = find_and_migrate_in_memory(&bottom)?.expect("should find pyproject");
        assert_eq!(
            kind,
            MigratedFromKind::Mypy(MigratedConfigSource::PyprojectToml)
        );
        Ok(())
    }

    #[test]
    fn test_in_memory_finds_pyproject_with_tool_pyright() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b");
        std::fs::create_dir_all(&bottom)?;
        fs_anyhow::write(
            &tmp.path().join("a/pyproject.toml"),
            b"[tool.pyright]\ninclude = [\"a.py\"]\n",
        )?;

        let (_cfg, kind) = find_and_migrate_in_memory(&bottom)?.expect("should find pyproject");
        assert_eq!(
            kind,
            MigratedFromKind::Pyright(MigratedConfigSource::PyprojectToml)
        );
        Ok(())
    }

    #[test]
    fn test_in_memory_returns_none_when_no_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&bottom)?;
        // No mypy/pyright/pyproject anywhere.
        let result = find_and_migrate_in_memory(&bottom)?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn test_in_memory_pyproject_without_tool_sections_is_none() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b");
        std::fs::create_dir_all(&bottom)?;
        // pyproject.toml exists but has neither [tool.mypy] nor [tool.pyright].
        fs_anyhow::write(
            &tmp.path().join("a/pyproject.toml"),
            b"[project]\nname = \"x\"\n",
        )?;

        let result = find_and_migrate_in_memory(&bottom)?;
        assert!(
            result.is_none(),
            "bare pyproject.toml without tool sections is not migrate-able"
        );
        Ok(())
    }

    #[test]
    fn test_in_memory_malformed_mypy_returns_err() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b");
        std::fs::create_dir_all(&bottom)?;
        // mypy.ini with content that confuses the configparser library.
        fs_anyhow::write(
            &tmp.path().join("a/mypy.ini"),
            b"this is not a valid ini file\nno equals signs\n[unclosed",
        )?;

        let result = find_and_migrate_in_memory(&bottom);
        assert!(
            result.is_err(),
            "malformed mypy.ini should propagate as Err so callers can decide whether to fall back",
        );
        Ok(())
    }

    /// When `pyproject.toml` has only `[tool.pyright]` (no
    /// `[tool.mypy]`) and the pyright section fails to parse, the
    /// surfaced error should reference the pyright section's data —
    /// not the mypy "section missing" failure that happens to be
    /// tried first. This pins the pyright-vs-mypy error precedence
    /// rule.
    #[test]
    fn test_in_memory_pyright_only_parse_error_surfaces_pyright_error() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bottom = tmp.path().join("a/b");
        std::fs::create_dir_all(&bottom)?;
        // `[tool.pyright]` present but malformed (`include` should be a list).
        fs_anyhow::write(
            &tmp.path().join("a/pyproject.toml"),
            b"[tool.pyright]\ninclude = 42\n",
        )?;

        let err = find_and_migrate_in_memory(&bottom)
            .expect_err("malformed [tool.pyright] should propagate as Err");
        let msg = format!("{err:#}");
        // The pyright parser fails on `include = 42` (expects a list).
        // Mypy would have failed earlier with a generic "section
        // missing" message that doesn't mention `include`. Asserting
        // the message names the offending field is the cleanest way
        // to confirm the pyright error is the one being surfaced.
        assert!(
            msg.contains("include"),
            "expected error to reference the pyright `include` field, got: {msg}",
        );
        Ok(())
    }
}
