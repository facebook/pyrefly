/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::args::clap_env;
use pyrefly_util::fs_anyhow;
use tracing::warn;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::stubgen::StubgenOptions;
use crate::stubgen::generate_stub;

/// Arguments for the stubgen command, which generates PEP 484 `.pyi` stub files
/// from Python source.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct StubgenArgs {
    /// Which files to generate stubs for.
    #[command(flatten)]
    files: FilesArgs,

    /// Type checking arguments and configuration.
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Output directory for generated `.pyi` files.
    #[arg(
        long,
        short,
        default_value = "out",
        env = clap_env("STUBGEN_OUTPUT")
    )]
    output: PathBuf,

    /// Include private names (those starting with `_`) in generated stubs.
    #[arg(long, default_value = "false")]
    include_private: bool,

    /// Continue generating stubs even if errors occur for some files.
    #[arg(long, default_value = "false")]
    ignore_errors: bool,
}

/// Derive a module name from a file path (e.g. `foo/bar.py` -> `foo.bar`).
fn module_name_from_path(path: &Path) -> String {
    let stem = path.with_extension("");
    stem.components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Map an input `.py` path to its output `.pyi` path under the output directory.
/// Uses the common ancestor of all input roots to compute relative paths, so
/// `src/foo/bar.py` with root `src/` becomes `out/foo/bar.pyi`.
fn output_path_for(input: &Path, roots: &[PathBuf], output_dir: &Path) -> PathBuf {
    let relative = roots
        .iter()
        .find_map(|root| input.strip_prefix(root).ok())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            // No root matched. Use the file name alone to avoid
            // `Path::join` discarding `output_dir` when `input` is absolute.
            input
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| input.to_path_buf())
        });
    output_dir.join(relative).with_extension("pyi")
}

/// Read one `.py` file, generate its stub, and write the `.pyi` output.
fn generate_one(
    path: &Path,
    roots: &[PathBuf],
    output_dir: &Path,
    options: &StubgenOptions,
) -> anyhow::Result<()> {
    let source = fs_anyhow::read_to_string(path)?;
    let module_name = module_name_from_path(path);
    let stub = generate_stub(&source, &module_name, options);

    let out_path = output_path_for(path, roots, output_dir);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    fs_anyhow::write(&out_path, stub)?;
    Ok(())
}

impl StubgenArgs {
    /// Resolve inputs, create the output directory, and generate stubs.
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;

        // Extract owned fields before `resolve` consumes `self.files` and
        // `self.config_override`. After the move, `self` is partially
        // consumed and can no longer be borrowed.
        let output = self.output;
        let include_private = self.include_private;
        let ignore_errors = self.ignore_errors;

        let (files_to_check, _config_finder) = self.files.resolve(self.config_override, wrapper)?;

        let roots = files_to_check.roots();
        let all_files = files_to_check.files()?;
        let py_files: Vec<&PathBuf> = all_files
            .iter()
            .filter(|p| p.extension().is_some_and(|ext| ext == "py"))
            .collect();

        if py_files.is_empty() {
            eprintln!("stubgen: no .py files found");
            return Ok(CommandExitStatus::Success);
        }

        std::fs::create_dir_all(&output)?;

        let options = StubgenOptions { include_private };

        let mut generated = 0usize;
        let mut failed = 0usize;

        for path in &py_files {
            match generate_one(path, &roots, &output, &options) {
                Ok(()) => generated += 1,
                Err(e) => {
                    failed += 1;
                    if ignore_errors {
                        warn!("stubgen: skipping {}: {e}", path.display());
                    } else {
                        return Err(e.context(format!(
                            "stubgen: failed to generate stub for {}",
                            path.display()
                        )));
                    }
                }
            }
        }

        eprintln!(
            "stubgen: generated {generated} stub{} in {}{}",
            if generated == 1 { "" } else { "s" },
            output.display(),
            if failed > 0 {
                format!(" ({failed} failed)")
            } else {
                String::new()
            },
        );

        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn test_module_name_from_path() {
        assert_eq!(module_name_from_path(Path::new("foo.py")), "foo");
        assert_eq!(module_name_from_path(Path::new("foo/bar.py")), "foo.bar");
        assert_eq!(
            module_name_from_path(Path::new("foo/bar/__init__.py")),
            "foo.bar.__init__"
        );
    }

    #[test]
    fn test_output_path_for() {
        let roots = vec![PathBuf::from("/src")];
        let out = Path::new("/out");

        assert_eq!(
            output_path_for(Path::new("/src/foo.py"), &roots, out),
            PathBuf::from("/out/foo.pyi")
        );
        assert_eq!(
            output_path_for(Path::new("/src/pkg/mod.py"), &roots, out),
            PathBuf::from("/out/pkg/mod.pyi")
        );
        assert_eq!(
            output_path_for(Path::new("/src/pkg/__init__.py"), &roots, out),
            PathBuf::from("/out/pkg/__init__.pyi")
        );
    }

    #[test]
    fn test_output_path_no_matching_root() {
        let roots = vec![PathBuf::from("/other")];
        let out = Path::new("/out");
        // Falls back to using just the file name when no root matches
        assert_eq!(
            output_path_for(Path::new("/src/foo.py"), &roots, out),
            PathBuf::from("/out/foo.pyi")
        );
    }
}
