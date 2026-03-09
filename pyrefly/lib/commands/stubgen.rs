/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_types::types::Union;
use pyrefly_util::args::clap_env;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use ruff_text_size::TextSize;
use tracing::warn;

use crate::commands::check::Handles;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::lsp::wasm::inlay_hints::ParameterAnnotation;
use crate::state::lsp::AnnotationKind;
use crate::state::require::Require;
use crate::state::state::CancellableTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::stubgen::StubgenOptions;
use crate::stubgen::generate_stub;
use crate::types::class::Class;
use crate::types::heap::TypeHeap;
use crate::types::simplify::unions_with_literals;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

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

// ---------------------------------------------------------------------------
// Type inference helpers
// ---------------------------------------------------------------------------

/// Convert a `Type` to its string representation, simplifying literal unions.
fn type_to_string(
    ty: Type,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
    heap: &TypeHeap,
) -> String {
    let ty = ty.promote_implicit_literals(stdlib);
    let ty = ty.explicit_any().clean_var();
    let ty = match ty {
        Type::Union(box Union { members, .. }) => {
            unions_with_literals(members, stdlib, enum_members, heap)
        }
        _ => ty,
    };
    ty.to_string()
}

/// Convert raw inferred-type tuples to `(position, annotation_string)` pairs,
/// filtering out types that would be unhelpful in a stub (`Any`, internal
/// `@`-types, `Unknown`, `Never`).
fn format_type_hints<'a>(
    hints: Vec<(TextSize, Type, AnnotationKind)>,
    stdlib: &Stdlib,
    transaction: &Transaction<'a>,
    handle: &pyrefly_build::handle::Handle,
    heap: &TypeHeap,
) -> Vec<(TextSize, String)> {
    let enum_members = |cls: &Class| -> Option<usize> {
        transaction
            .ad_hoc_solve(handle, "stubgen_enum_metadata", |solver| {
                let meta = solver.get_metadata_for_class(cls);
                if meta.is_enum() {
                    Some(solver.get_enum_members(cls).len())
                } else {
                    None
                }
            })
            .flatten()
    };

    let mut result = Vec::new();
    for (position, ty, kind) in hints {
        let text = type_to_string(ty, stdlib, &enum_members, heap);
        if text.contains("Any")
            || text.contains('@')
            || text.contains("Unknown")
            || text.contains("Never")
        {
            continue;
        }
        if text == "None" && kind == AnnotationKind::Parameter {
            continue;
        }
        let annotation = match kind {
            AnnotationKind::Parameter => format!(": {text}"),
            AnnotationKind::Return => format!(" -> {text}"),
            AnnotationKind::Variable => format!(": {text}"),
        };
        result.push((position, annotation));
    }
    result
}

/// Apply annotation insertions to `source` in-memory. Insertions are applied
/// from back to front so earlier positions remain valid.
fn apply_annotations(source: &str, mut hints: Vec<(TextSize, String)>) -> String {
    hints.sort_by(|(a, _), (b, _)| b.cmp(a));
    let mut result = source.to_owned();
    for (position, annotation) in hints {
        let offset: usize = position.into();
        if offset <= result.len() {
            result.insert_str(offset, &annotation);
        }
    }
    result
}

impl StubgenArgs {
    /// Resolve inputs, set up the type-checker, and generate stubs.
    ///
    /// For each input file we run Pyrefly's solver to infer types for
    /// unannotated items, then feed the annotated source into the stub
    /// generator for a high-quality `.pyi`.
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;

        let output = self.output;
        let include_private = self.include_private;
        let ignore_errors = self.ignore_errors;

        let (files_to_check, config_finder) = self.files.resolve(self.config_override, wrapper)?;

        let roots: Vec<PathBuf> = files_to_check.roots().to_vec();
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;

        if expanded_file_list.is_empty() {
            eprintln!("stubgen: no .py files found");
            return Ok(CommandExitStatus::Success);
        }

        // Set up the solver
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);
        let handles_set = Handles::new(expanded_file_list);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );
        let mut cancellable = holder.as_ref().cancellable_transaction();
        let transaction = forgetter.as_mut();

        let (handles, _, sourcedb_errors) = handles_set.all(holder.as_ref().config_finder());
        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("stubgen: failed to load source files."));
        }

        std::fs::create_dir_all(&output)?;
        let options = StubgenOptions { include_private };
        let mut generated = 0usize;
        let mut failed = 0usize;

        for handle in &handles {
            let path = handle.path().as_path();
            if path.extension().is_none_or(|e| e != "py") {
                continue;
            }

            let result = generate_one_with_solver(
                handle,
                &roots,
                &output,
                &options,
                transaction,
                &mut cancellable,
            );
            match result {
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

/// Run the solver on a single file, apply inferred annotations, then generate
/// and write the `.pyi` stub.
fn generate_one_with_solver(
    handle: &pyrefly_build::handle::Handle,
    roots: &[PathBuf],
    output_dir: &Path,
    options: &StubgenOptions,
    transaction: &mut Transaction<'_>,
    cancellable: &mut CancellableTransaction<'_>,
) -> anyhow::Result<()> {
    let path = handle.path().as_path();

    // Run the solver for this file
    transaction.run(&[handle.dupe()], Require::Everything, None);

    // Collect inferred types and parameter annotations
    let stdlib = transaction.get_stdlib(handle);
    let inferred_types = transaction.inferred_types(handle, true, true);
    let parameter_annotations = transaction.infer_parameter_annotations(handle, cancellable);

    let mut all_hints: Vec<(TextSize, Type, AnnotationKind)> = parameter_annotations
        .into_iter()
        .filter_map(|p: ParameterAnnotation| p.to_inlay_hint())
        .collect();
    if let Some(inferred) = inferred_types {
        all_hints.extend(inferred);
    }

    // Read the original source
    let source = fs_anyhow::read_to_string(path)?;

    // Format and apply inferred annotations in-memory
    let heap = TypeHeap::new();
    let formatted = format_type_hints(all_hints, &stdlib, transaction, handle, &heap);
    let annotated = apply_annotations(&source, formatted);

    // Generate stub from the (now annotated) source
    let module_name = module_name_from_path(path);
    let stub = generate_stub(&annotated, &module_name, options);

    let out_path = output_path_for(path, roots, output_dir);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    fs_anyhow::write(&out_path, stub)?;
    Ok(())
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
        assert_eq!(
            output_path_for(Path::new("/src/foo.py"), &roots, out),
            PathBuf::from("/out/foo.pyi")
        );
    }
}
