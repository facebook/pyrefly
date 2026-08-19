/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::io::Write as _;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

use anstream::stderr;
use anyhow::Context as _;
use anyhow::bail;
use clap::Parser;
use dupe::Dupe as _;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::LiveSourceDatabase;
use pyrefly_build::source_db::ModuleEnumerator;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_config::base::Preset;
use pyrefly_config::error::ErrorDisplayConfig;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::thread_pool::ThreadCount;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;
use yansi::Paint;

use crate::commands::check::write_errors_to_stderr;
use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::legacy::LegacyError;
use crate::state::require::Require;
use crate::state::state::State;

/// Arguments for Bazel-powered type checking.
#[derive(Debug, Clone, Parser)]
pub struct BazelCheckArgs {
    /// Path to Bazel input JSON.
    input_path: PathBuf,

    /// Path to output JSON file containing Pyrefly type check results.
    #[arg(long = "output", short = 'o', value_name = "FILE")]
    output_path: PathBuf,

    /// Minimum severity level for errors to be displayed.
    /// Errors below this severity will not be shown. Defaults to "error".
    #[arg(long, value_enum, default_value_t = Severity::Error)]
    min_severity: Severity,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelCheckInput {
    target: BazelTargetInfo,
    check_roots: BazelCheckRoots,
    search_path: BazelSearchPath,
    #[serde(default)]
    path_overlays: Vec<BazelPathOverlay>,
    config: BazelConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct BazelTargetInfo {
    label: String,
    workspace_name: String,
    package: String,
    name: String,
    rule_kind: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelCheckRoots {
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    stubs: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelSearchPath {
    main_file: Option<String>,
    main_file_directory: Option<String>,
    #[serde(default)]
    explicit_imports: Vec<String>,
    workspace_name: String,
    #[serde(default)]
    python_import_all_repositories: bool,
    #[serde(default)]
    repository_roots: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelPathOverlay {
    short_path: String,
    path: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelConfig {
    python_version: Option<String>,
    system_platform: Option<String>,
    preset: Option<String>,
    error_severities: Option<Value>,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedBazelConfig {
    python_version: PythonVersion,
    system_platform: PythonPlatform,
    preset: Option<Preset>,
    errors: Option<ErrorDisplayConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BazelSearchRoot {
    logical: PathBuf,
    physical: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BazelFileInput {
    raw_short_path: String,
    logical_path: PathBuf,
    physical_path: PathBuf,
    is_check_root: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExplicitModuleEntry {
    module_name: ModuleName,
    module_path: ModulePath,
    logical_path: PathBuf,
    is_check_root: bool,
    rank: ExplicitModuleRank,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExplicitModuleRank {
    is_stub_package: Reverse<bool>,
    search_root_order: usize,
    extension_priority: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedModuleName {
    module_name: ModuleName,
    search_root_order: usize,
    root_relative_path: PathBuf,
}

#[derive(Debug)]
struct BazelCheckSourceDatabase {
    // Entries for each module are sorted best-first; lookup relies on first()
    // being the default candidate when no style filter matches.
    candidates: SmallMap<ModuleName, Vec1<ExplicitModuleEntry>>,
    modules_to_check: Vec<Handle>,
    path_to_handle: SmallMap<ModulePath, Handle>,
}

impl BazelCheckInput {
    fn from_json_bytes(path: &Path, data: &[u8]) -> anyhow::Result<Self> {
        let input: Self = serde_json::from_slice(data)
            .with_context(|| format!("failed to parse Bazel input JSON `{}`", path.display()))?;
        if input.target.workspace_name != input.search_path.workspace_name {
            bail!(
                "Bazel input target workspace `{}` does not match search path workspace `{}` for target `{}`",
                input.target.workspace_name,
                input.search_path.workspace_name,
                input.target.label,
            );
        }
        Ok(input)
    }
}

impl BazelConfig {
    fn parse(&self) -> anyhow::Result<ParsedBazelConfig> {
        let python_version = self
            .python_version
            .as_deref()
            .map(PythonVersion::from_str)
            .transpose()?
            .unwrap_or_default();
        let system_platform = self
            .system_platform
            .as_deref()
            .map(PythonPlatform::new)
            .unwrap_or_default();
        let preset = self
            .preset
            .as_ref()
            .map(|preset| {
                serde_json::from_value::<Preset>(Value::String(preset.clone()))
                    .with_context(|| "invalid Bazel input config `preset`")
            })
            .transpose()?;
        let errors = self
            .error_severities
            .clone()
            .map(|errors| {
                serde_json::from_value::<ErrorDisplayConfig>(errors)
                    .with_context(|| "invalid Bazel input config `error_severities`")
            })
            .transpose()?;

        Ok(ParsedBazelConfig {
            python_version,
            system_platform,
            preset,
            errors,
        })
    }
}

fn has_python_extension(path: &str) -> bool {
    path.ends_with(".py") || path.ends_with(".pyi")
}

fn validate_relative_path(
    path_for_error: &str,
    description: &str,
    allow_empty: bool,
    allow_leading_parent: bool,
) -> anyhow::Result<()> {
    if path_for_error.is_empty() {
        if allow_empty {
            return Ok(());
        }
        bail!("Bazel {description} must not be empty");
    }
    let path = Path::new(path_for_error);
    if path.is_absolute() {
        bail!("Bazel {description} `{path_for_error}` must be relative, not absolute");
    }
    let mut components = path.components();
    if allow_leading_parent && let Some(Component::ParentDir) = components.clone().next() {
        // Allow exactly one leading "../" for external repos, no other parent or absolute components.
        // consume leading ../
        components.next();
        // after leading ../, next component must exist and not be ParentDir, CurDir, Root, Prefix
        let Some(Component::Normal(_)) = components.clone().next() else {
            bail!(
                "Bazel {description} `{path_for_error}` has invalid external prefix; expected `../<repo>/...`"
            );
        };
    }
    for component in components {
        match component {
            Component::Normal(_) => {}
            Component::ParentDir => {
                if allow_leading_parent {
                    bail!(
                        "Bazel {description} `{path_for_error}` must not contain `..` except leading `../` for external repos"
                    )
                } else {
                    bail!("Bazel {description} `{path_for_error}` must not contain `..`")
                }
            }
            Component::CurDir => {
                bail!("Bazel {description} `{path_for_error}` must not contain `.` components")
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("Bazel {description} `{path_for_error}` must be relative")
            }
        }
    }
    Ok(())
}

fn validate_short_path(short_path: &str) -> anyhow::Result<()> {
    validate_relative_path(short_path, "short path", false, true)?;
    validate_no_empty_or_cur_dir_components(short_path, "short path")
}

fn validate_search_root_path(
    path_for_error: &str,
    description: &str,
    allow_empty: bool,
) -> anyhow::Result<()> {
    validate_relative_path(path_for_error, description, allow_empty, false)?;
    if allow_empty && path_for_error.is_empty() {
        Ok(())
    } else {
        validate_no_empty_or_cur_dir_components(path_for_error, description)
    }
}

fn validate_no_empty_or_cur_dir_components(
    path_for_error: &str,
    description: &str,
) -> anyhow::Result<()> {
    for component in path_for_error.split('/') {
        match component {
            "" => bail!(
                "Bazel {description} `{path_for_error}` must not contain empty components or trailing slashes"
            ),
            "." => {
                bail!("Bazel {description} `{path_for_error}` must not contain `.` components")
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_path_overlay_physical_path(path_for_error: &str) -> anyhow::Result<()> {
    validate_relative_path(path_for_error, "path overlay physical path", false, false)?;
    validate_no_empty_or_cur_dir_components(path_for_error, "path overlay physical path")
}

fn logical_path_from_short_path(short_path: &str, workspace_name: &str) -> PathBuf {
    // External repository short paths escape the main workspace as `../repo/...`;
    // runfiles expose those files under `repo/...`.
    // Caller must validate short_path with validate_short_path first.
    short_path
        .strip_prefix("../")
        .map_or_else(|| Path::new(workspace_name).join(short_path), PathBuf::from)
}

fn default_physical_from_logical(logical: &Path, workspace: &str) -> anyhow::Result<PathBuf> {
    if logical.is_absolute() {
        bail!("logical path must be relative, got `{}`", logical.display());
    }
    Ok(match logical.strip_prefix(workspace) {
        Ok(relative) if relative.as_os_str().is_empty() => PathBuf::from("."),
        Ok(relative) => relative.to_path_buf(),
        Err(_) => Path::new("external").join(logical),
    })
}

fn physical_path_from_logical_import(import: &str, workspace_name: &str) -> PathBuf {
    let import_path = Path::new(import);
    if import_path == Path::new(workspace_name) {
        PathBuf::from(".")
    } else if let Ok(relative) = import_path.strip_prefix(workspace_name) {
        relative.to_path_buf()
    } else {
        Path::new("external").join(import_path)
    }
}

fn build_search_roots(
    search_path: &BazelSearchPath,
    python_version: PythonVersion,
) -> anyhow::Result<Vec<BazelSearchRoot>> {
    if search_path.python_import_all_repositories && search_path.repository_roots.is_empty() {
        bail!(
            "`python_import_all_repositories` is true but `repository_roots` is empty for workspace `{}`",
            search_path.workspace_name,
        );
    }
    search_path
        .main_file_directory
        .iter()
        .try_for_each(|main_file_directory| {
            validate_search_root_path(main_file_directory, "main_file_directory", true)
        })?;
    search_path
        .explicit_imports
        .iter()
        .try_for_each(|import| validate_search_root_path(import, "explicit import", false))?;
    search_path
        .repository_roots
        .iter()
        .try_for_each(|repository_root| {
            validate_search_root_path(repository_root, "repository root", false)
        })?;

    let mut roots = Vec::new();
    let mut add_root = |logical: PathBuf, physical: PathBuf| {
        if !roots
            .iter()
            .any(|root: &BazelSearchRoot| root.logical == logical && root.physical == physical)
        {
            roots.push(BazelSearchRoot { logical, physical });
        }
    };

    // Python 3.11's safe-path behavior omits the script directory prepend.
    if python_version < PythonVersion::new(3, 11, 0)
        && let Some(main_file_directory) = &search_path.main_file_directory
    {
        if main_file_directory.is_empty() {
            add_root(
                PathBuf::from(&search_path.workspace_name),
                PathBuf::from("."),
            );
        } else {
            add_root(
                Path::new(&search_path.workspace_name).join(main_file_directory),
                PathBuf::from(main_file_directory),
            );
        }
    }

    search_path.explicit_imports.iter().for_each(|import| {
        add_root(
            PathBuf::from(import),
            physical_path_from_logical_import(import, &search_path.workspace_name),
        );
    });

    if search_path.python_import_all_repositories {
        let mut repository_roots = search_path.repository_roots.clone();
        // rules_python treats import-all repository roots as an unordered
        // runfiles directory set and appends them in lexicographic order.
        repository_roots.sort();
        repository_roots.into_iter().for_each(|repository_root| {
            add_root(
                PathBuf::from(&repository_root),
                physical_path_from_logical_import(&repository_root, &search_path.workspace_name),
            );
        });
    } else {
        add_root(
            PathBuf::from(&search_path.workspace_name),
            PathBuf::from("."),
        );
    }

    Ok(roots)
}

fn build_bazel_file_inputs(input: &BazelCheckInput) -> anyhow::Result<Vec<BazelFileInput>> {
    let mut overlay_by_logical_path = SmallMap::new();
    input.path_overlays.iter().try_for_each(|overlay| {
        if !has_python_extension(&overlay.short_path) {
            bail!(
                "Bazel path overlay `{}` for target `{}` is not a Python file",
                overlay.short_path,
                input.target.label,
            );
        }
        validate_short_path(&overlay.short_path).with_context(|| {
            format!(
                "invalid Bazel path overlay `{}` for target `{}`",
                overlay.short_path, input.target.label
            )
        })?;
        validate_path_overlay_physical_path(overlay.path.as_str()).with_context(|| {
            format!(
                "invalid Bazel path overlay physical path `{}` for target `{}`",
                overlay.path, input.target.label
            )
        })?;
        let logical_path =
            logical_path_from_short_path(&overlay.short_path, &input.search_path.workspace_name);
        let physical_path = PathBuf::from(overlay.path.as_str());
        match overlay_by_logical_path.get(&logical_path) {
            Some((_, existing_physical_path)) if existing_physical_path != &physical_path => {
                bail!(
                    "Bazel path overlay `{}` for target `{}` conflicts with another overlay for normalized path `{}`",
                    overlay.short_path,
                    input.target.label,
                    logical_path.display(),
                );
            }
            Some(_) => {}
            None => {
                overlay_by_logical_path.insert(
                    logical_path,
                    (overlay.short_path.clone(), physical_path),
                );
            }
        }
        Ok(())
    })?;

    let check_roots = input
        .check_roots
        .sources
        .iter()
        .chain(input.check_roots.stubs.iter())
        .map(|check_root| {
            if !has_python_extension(check_root) {
                bail!(
                    "Bazel check root `{}` for target `{}` is not a Python file",
                    check_root,
                    input.target.label,
                );
            }
            validate_short_path(check_root).with_context(|| {
                format!(
                    "invalid Bazel check root `{}` for target `{}`",
                    check_root, input.target.label
                )
            })?;
            let logical_path =
                logical_path_from_short_path(check_root, &input.search_path.workspace_name);
            let physical_path = match overlay_by_logical_path.get(&logical_path) {
                Some((_, physical_path)) => physical_path.clone(),
                None => {
                    default_physical_from_logical(&logical_path, &input.search_path.workspace_name)
                        .with_context(|| {
                            format!(
                                "invalid Bazel check root `{}` for target `{}`",
                                check_root, input.target.label
                            )
                        })?
                }
            };
            Ok(BazelFileInput {
                raw_short_path: check_root.clone(),
                logical_path,
                physical_path,
                is_check_root: true,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let overlay_entries = overlay_by_logical_path
        .into_iter()
        .map(|(logical_path, (raw_short_path, physical_path))| {
            let default_physical =
                default_physical_from_logical(&logical_path, &input.search_path.workspace_name)
                    .with_context(|| {
                        format!(
                            "invalid Bazel path overlay `{}` for target `{}`",
                            raw_short_path, input.target.label
                        )
                    });
            default_physical.map(|default_physical| {
                (physical_path != default_physical
                    && !check_roots
                        .iter()
                        .any(|check_root| check_root.logical_path == logical_path))
                .then_some(BazelFileInput {
                    raw_short_path,
                    logical_path,
                    physical_path,
                    is_check_root: false,
                })
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    Ok(check_roots.into_iter().chain(overlay_entries).collect())
}

fn strip_top_level_stubs_suffix(path: &Path) -> PathBuf {
    let mut components = path.components();
    let Some(first) = components.next() else {
        return path.to_path_buf();
    };
    let mut stripped_path = match first {
        Component::Normal(os_str) => os_str
            .to_str()
            .and_then(|name| name.strip_suffix("-stubs"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(os_str)),
        _ => PathBuf::from(first.as_os_str()),
    };
    components.for_each(|component| stripped_path.push(component));
    stripped_path
}

fn module_name_for_logical_path_under_search_roots(
    logical_path: &Path,
    ordered_search_roots: &[BazelSearchRoot],
) -> anyhow::Result<ResolvedModuleName> {
    let Some((search_root_order, search_root, root_relative_path)) = ordered_search_roots
        .iter()
        .enumerate()
        .find_map(|(order, root)| {
            logical_path
                .strip_prefix(&root.logical)
                .ok()
                .map(|relative| (order, root, relative.to_path_buf()))
        })
    else {
        bail!(
            "Bazel logical path `{}` is not under any search root",
            logical_path.display(),
        );
    };
    let module_relative_path = strip_top_level_stubs_suffix(&root_relative_path);
    let module_name = ModuleName::from_relative_path(&module_relative_path).with_context(|| {
        format!(
            "failed to derive module name from Bazel logical path `{}` under search root `{}`",
            logical_path.display(),
            search_root.logical.display(),
        )
    })?;
    Ok(ResolvedModuleName {
        module_name,
        search_root_order,
        root_relative_path,
    })
}

fn is_top_level_stub_package(root_relative_path: &Path) -> bool {
    root_relative_path.components().next().is_some_and(|component| {
        matches!(component, Component::Normal(os_str) if os_str.to_str().is_some_and(|name| name.ends_with("-stubs")))
    })
}

fn ensure_logical_and_physical_styles_match(
    logical_path: &Path,
    module_path: &ModulePath,
) -> anyhow::Result<()> {
    let logical_style = ModuleStyle::of_path(logical_path);
    let physical_style = module_path.style();
    if logical_style != physical_style {
        bail!(
            "Bazel logical path `{}` has style `{:?}`, but physical path `{}` has style `{:?}`",
            logical_path.display(),
            logical_style,
            module_path.as_path().display(),
            physical_style,
        );
    }
    Ok(())
}

fn extension_priority(style: ModuleStyle) -> u8 {
    match style {
        ModuleStyle::Interface => 0,
        ModuleStyle::Executable => 1,
    }
}

fn compare_explicit_module_entries(
    left: &ExplicitModuleEntry,
    right: &ExplicitModuleEntry,
) -> Ordering {
    left.rank
        .cmp(&right.rank)
        .then_with(|| left.logical_path.cmp(&right.logical_path))
        .then_with(|| left.module_path.as_path().cmp(right.module_path.as_path()))
}

fn build_explicit_entries(
    file_inputs: Vec<BazelFileInput>,
    search_roots: &[BazelSearchRoot],
    target_label: &str,
) -> anyhow::Result<Vec<ExplicitModuleEntry>> {
    let mut entries = Vec::new();
    for file_input in file_inputs {
        match module_name_for_logical_path_under_search_roots(
            &file_input.logical_path,
            search_roots,
        ) {
            Ok(resolved) => {
                let module_path = ModulePath::filesystem(file_input.physical_path.clone());
                ensure_logical_and_physical_styles_match(&file_input.logical_path, &module_path)?;
                let module_style = module_path.style();
                entries.push(ExplicitModuleEntry {
                    module_name: resolved.module_name,
                    module_path,
                    logical_path: file_input.logical_path.clone(),
                    is_check_root: file_input.is_check_root,
                    rank: ExplicitModuleRank {
                        is_stub_package: Reverse(is_top_level_stub_package(
                            &resolved.root_relative_path,
                        )),
                        search_root_order: resolved.search_root_order,
                        extension_priority: extension_priority(module_style),
                    },
                });
            }
            Err(error) if file_input.is_check_root => {
                bail!(
                    "Bazel check root `{}` for target `{}` is not importable: {error:#}",
                    file_input.raw_short_path,
                    target_label,
                );
            }
            Err(_) => {
                // Non-check-root overlays only participate when importable under the target search roots.
            }
        }
    }

    let mut by_key: SmallMap<(ModuleName, ModulePath), ExplicitModuleEntry> = SmallMap::new();
    for entry in entries {
        match by_key.get_mut(&(entry.module_name, entry.module_path.dupe())) {
            Some(existing) => {
                let is_check_root = existing.is_check_root || entry.is_check_root;
                if compare_explicit_module_entries(&entry, existing).is_lt() {
                    *existing = entry;
                }
                existing.is_check_root = is_check_root;
            }
            None => {
                by_key.insert((entry.module_name, entry.module_path.dupe()), entry);
            }
        }
    }

    Ok(by_key.into_values().collect())
}

impl BazelCheckSourceDatabase {
    fn new(entries: Vec<ExplicitModuleEntry>, sys_info: SysInfo) -> Self {
        let mut accumulated: SmallMap<ModuleName, Vec<ExplicitModuleEntry>> = SmallMap::new();
        entries.into_iter().for_each(|entry| {
            accumulated
                .entry(entry.module_name)
                .or_default()
                .push(entry);
        });

        let candidates = accumulated
            .into_iter()
            .map(|(module_name, mut entries)| {
                entries.sort_by(compare_explicit_module_entries);
                (
                    module_name,
                    Vec1::try_from_vec(entries).expect("inserted at least one entry"),
                )
            })
            .collect::<SmallMap<_, _>>();
        let handle = |module_name: &ModuleName, module_path: &ModulePath| {
            Handle::new(module_name.dupe(), module_path.dupe(), sys_info.dupe())
        };
        let modules_to_check = candidates
            .iter()
            .flat_map(|(module_name, entries)| {
                entries
                    .iter()
                    .filter(|entry| entry.is_check_root)
                    .map(|entry| handle(module_name, &entry.module_path))
            })
            .collect::<Vec<_>>();
        // TODO: This matches buck-check behavior, but still collapses the rare
        // case where multiple logical modules intentionally share one physical artifact.
        let path_to_handle = candidates
            .iter()
            .flat_map(|(module_name, entries)| {
                entries.iter().map(|entry| {
                    (
                        entry.module_path.dupe(),
                        handle(module_name, &entry.module_path),
                    )
                })
            })
            .collect::<SmallMap<_, _>>();

        Self {
            candidates,
            modules_to_check,
            path_to_handle,
        }
    }
}

impl SourceDatabase for BazelCheckSourceDatabase {
    fn may_contain_module(&self, module: ModuleName) -> bool {
        self.candidates.contains_key(&module)
    }

    fn lookup(
        &self,
        module: ModuleName,
        _: Option<&Path>,
        style_filter: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        let candidates = self.candidates.get(&module)?;
        style_filter
            .and_then(|style| {
                candidates
                    .iter()
                    .find(|entry| entry.module_path.style() == style)
            })
            .or_else(|| Some(candidates.first()))
            .map(|entry| entry.module_path.dupe())
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        self.path_to_handle
            .get(module_path)
            .map(|handle| handle.dupe())
    }

    fn as_live_source_database(&self) -> Option<&dyn LiveSourceDatabase> {
        None
    }
}

impl ModuleEnumerator for BazelCheckSourceDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        self.modules_to_check.clone()
    }
}

#[derive(Debug, Serialize)]
struct BazelDiagnostics<'a> {
    diagnostics: Vec<BazelDiagnostic<'a>>,
}

#[derive(Debug, Serialize)]
struct BazelDiagnostic<'a> {
    target: &'a str,
    #[serde(flatten)]
    error: LegacyError,
}

fn keep_in_bazel_output(error: &Error, min_severity: Severity) -> bool {
    error.error_kind().is_directive() || error.severity() >= min_severity
}

fn compute_bazel_errors(
    parsed_config: &ParsedBazelConfig,
    search_roots: &[BazelSearchRoot],
    source_db: BazelCheckSourceDatabase,
    thread_count: ThreadCount,
) -> anyhow::Result<Vec<Error>> {
    let modules_to_check = source_db.modules_to_check();
    let mut config = ConfigFile::default();
    config.python_environment.python_platform = Some(parsed_config.system_platform.clone());
    config.python_environment.python_version = Some(parsed_config.python_version);
    config.python_environment.site_package_path = Some(Vec::new());
    config.search_path_from_file = search_roots
        .iter()
        .map(|root| root.physical.clone())
        .collect();
    config.source_db = Some(ArcId::new(Box::new(source_db)));
    config.interpreters.skip_interpreter_query = true;
    config.disable_search_path_heuristics = true;
    config.preset = parsed_config.preset;
    if let Some(errors) = &parsed_config.errors {
        config.root.errors = Some(errors.clone());
    }
    config.configure();

    let config = ArcId::new(config);
    let state = State::new(ConfigFinder::new_constant(config), thread_count);
    let mut transaction = state.new_transaction(Require::Errors, None);
    transaction.run(&modules_to_check, Require::Errors, None);

    Ok(transaction
        .get_errors(&modules_to_check)
        .collect_display_errors_with_unused_ignores())
}

fn bazel_diagnostics_from_errors<'a>(target: &'a str, errors: &[Error]) -> BazelDiagnostics<'a> {
    BazelDiagnostics {
        diagnostics: errors
            .iter()
            .map(|error| BazelDiagnostic {
                target,
                error: LegacyError::from_error(Path::new(""), error),
            })
            .collect(),
    }
}

fn display_target_label(input_label: &str) -> &str {
    if input_label.starts_with("@@//") {
        &input_label[2..]
    } else {
        input_label
    }
}

fn read_input_file(path: &Path) -> anyhow::Result<BazelCheckInput> {
    let data = fs_anyhow::read(path)
        .with_context(|| format!("failed to read Bazel input JSON `{}`", path.display()))?;
    BazelCheckInput::from_json_bytes(path, &data)
}

fn write_output(path: &Path, diagnostics: &BazelDiagnostics<'_>) -> anyhow::Result<()> {
    let output_bytes = serde_json::to_vec(diagnostics)
        .with_context(|| "failed to serialize Bazel diagnostic JSON value to bytes")?;
    fs_anyhow::write(path, &output_bytes)
}

impl BazelCheckArgs {
    pub fn run(self, thread_count: ThreadCount) -> anyhow::Result<CommandExitStatus> {
        match self.run_inner(thread_count) {
            Ok(status) => Ok(status),
            Err(error) => {
                eprintln!("{error:?}");
                Ok(CommandExitStatus::InfraError)
            }
        }
    }

    fn run_inner(self, thread_count: ThreadCount) -> anyhow::Result<CommandExitStatus> {
        let Self {
            input_path,
            output_path,
            min_severity,
        } = self;
        let input = read_input_file(&input_path)?;
        let file_inputs = build_bazel_file_inputs(&input)?;
        let config = input.config.parse()?;
        let search_roots = build_search_roots(&input.search_path, config.python_version)?;
        let explicit_entries =
            build_explicit_entries(file_inputs, &search_roots, &input.target.label)?;
        let source_db = BazelCheckSourceDatabase::new(
            explicit_entries,
            SysInfo::new(config.python_version, config.system_platform.clone()),
        );
        let type_errors = compute_bazel_errors(&config, &search_roots, source_db, thread_count)?;
        let displayed_errors = type_errors
            .into_iter()
            .filter(|error| keep_in_bazel_output(error, min_severity))
            .collect::<Vec<_>>();
        let target_label = display_target_label(&input.target.label);
        let diagnostics = bazel_diagnostics_from_errors(target_label, &displayed_errors);
        write_output(&output_path, &diagnostics)?;
        if !displayed_errors.is_empty() {
            // `stderr()` is an anstream stream, so it strips the color codes itself
            // when color is disabled; no need to branch on the color choice here.
            writeln!(
                stderr(),
                "{}\n",
                Paint::magenta(&format!("Target {target_label}"))
                    .bold()
                    .underline()
            )?;
            write_errors_to_stderr(Path::new(""), &displayed_errors, true)?;
        }
        if displayed_errors
            .iter()
            .any(|error| !error.error_kind().is_directive())
        {
            Ok(CommandExitStatus::UserError)
        } else {
            Ok(CommandExitStatus::Success)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_input_json() -> &'static str {
        r#"{
  "target": {
    "label": "//pkg:app",
    "workspace_name": "_main",
    "package": "pkg",
    "name": "app",
    "rule_kind": "py_binary"
  },
  "check_roots": {
    "sources": ["pkg/app.py"],
    "stubs": ["pkg/app.pyi"]
  },
  "search_path": {
    "main_file": "pkg/app.py",
    "main_file_directory": "pkg",
    "explicit_imports": [
      "_main/pkg",
      "rules_python++pip+pypi_311/site-packages"
    ],
    "workspace_name": "_main",
    "python_import_all_repositories": false,
    "repository_roots": ["_main"]
  },
  "path_overlays": [
    {
      "short_path": "pkg/generated.py",
      "path": "bazel-out/darwin-fastbuild/bin/pkg/generated.py"
    }
  ],
  "config": {
    "python_version": "3.12",
    "system_platform": "linux",
    "preset": null,
    "error_severities": null
  }
}"#
    }

    fn parse(json: &str) -> anyhow::Result<BazelCheckInput> {
        BazelCheckInput::from_json_bytes(Path::new("input.json"), json.as_bytes())
    }

    fn parse_config_object(config: &str) -> anyhow::Result<ParsedBazelConfig> {
        let json = format!(
            r#"{{
  "target": {{"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"}},
  "check_roots": {{"sources": ["pkg/lib.py"]}},
  "search_path": {{"workspace_name": "_main"}},
  "config": {config}
}}"#
        );
        parse(&json)?.config.parse()
    }

    #[test]
    fn target_metadata_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.target.label, "//pkg:app");
        assert_eq!(input.target.workspace_name, "_main");
        assert_eq!(input.target.package, "pkg");
        assert_eq!(input.target.name, "app");
        assert_eq!(input.target.rule_kind, "py_binary");
    }

    #[test]
    fn display_target_label_normalizes_root_workspace_canonical_label_only() {
        assert_eq!(display_target_label("@@//:app"), "//:app");
        assert_eq!(display_target_label("//pkg:app"), "//pkg:app");
        assert_eq!(display_target_label("@@repo//pkg:app"), "@@repo//pkg:app");
    }

    #[test]
    fn check_root_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.check_roots.sources, vec!["pkg/app.py".to_owned()]);
        assert_eq!(input.check_roots.stubs, vec!["pkg/app.pyi".to_owned()]);
    }

    #[test]
    fn search_path_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.search_path.main_file.as_deref(), Some("pkg/app.py"));
        assert_eq!(
            input.search_path.main_file_directory.as_deref(),
            Some("pkg")
        );
        assert_eq!(
            input
                .search_path
                .explicit_imports
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["_main/pkg", "rules_python++pip+pypi_311/site-packages"]
        );
        assert_eq!(input.search_path.workspace_name, "_main");
        assert!(!input.search_path.python_import_all_repositories);
        assert_eq!(input.search_path.repository_roots, vec!["_main".to_owned()]);
    }

    #[test]
    fn path_overlay_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.path_overlays.len(), 1);
        assert_eq!(input.path_overlays[0].short_path, "pkg/generated.py");
        assert_eq!(
            input.path_overlays[0].path,
            "bazel-out/darwin-fastbuild/bin/pkg/generated.py"
        );
    }

    #[test]
    fn empty_package_and_main_file_directory_are_preserved() {
        let input = parse(
            r#"{
  "target": {"label": "//:app", "workspace_name": "_main", "package": "", "name": "app", "rule_kind": "py_binary"},
  "check_roots": {},
  "search_path": {"main_file_directory": "", "workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input with empty string fields should parse");
        assert_eq!(input.target.package, "");
        assert_eq!(input.search_path.main_file_directory.as_deref(), Some(""));
    }

    #[test]
    fn omitted_repeated_fields_default_to_empty() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input with omitted repeated fields should parse");
        assert!(input.check_roots.sources.is_empty());
        assert!(input.check_roots.stubs.is_empty());
        assert!(input.search_path.explicit_imports.is_empty());
        assert!(input.search_path.repository_roots.is_empty());
        assert!(input.path_overlays.is_empty());
    }

    #[test]
    fn omitted_optional_search_path_fields_default_to_none_or_false() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input with omitted optional search path fields should parse");
        assert!(input.search_path.main_file.is_none());
        assert!(input.search_path.main_file_directory.is_none());
        assert!(!input.search_path.python_import_all_repositories);
    }

    #[test]
    fn null_config_values_use_pyrefly_defaults() {
        let config = parse_config_object(
            r#"{"python_version": null, "system_platform": null, "preset": null, "error_severities": null}"#,
        );
        let config = config.expect("explicit null config values should parse");
        assert_eq!(config.python_version, PythonVersion::default());
        assert_eq!(config.system_platform, PythonPlatform::default());
        assert!(config.preset.is_none());
        assert!(config.errors.is_none());
    }

    #[test]
    fn python_version_config_parses_to_pyrefly_type() {
        let config =
            parse_config_object(r#"{"python_version": "python3.10", "system_platform": null}"#)
                .expect("valid Python version config should parse");
        assert_eq!(config.python_version, PythonVersion::new(3, 10, 0));
    }

    #[test]
    fn system_platform_config_parses_to_pyrefly_type() {
        let config = parse_config_object(r#"{"python_version": null, "system_platform": "Linux"}"#)
            .expect("valid system platform config should parse");
        assert_eq!(config.system_platform, PythonPlatform::linux());
    }

    #[test]
    fn preset_config_parses_to_pyrefly_type() {
        let config = parse_config_object(
            r#"{"python_version": null, "system_platform": null, "preset": "strict"}"#,
        )
        .expect("valid preset config should parse");
        assert_eq!(config.preset, Some(Preset::Strict));
    }

    #[test]
    fn error_severities_config_parses_to_pyrefly_type() {
        let config = parse_config_object(
            r#"{"python_version": null, "system_platform": null, "error_severities": {"missing-attribute": "warn"}}"#,
        )
        .expect("valid error severities config should parse");
        assert!(config.errors.is_some());
    }

    #[test]
    fn minimal_input_uses_schema_and_config_defaults() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("minimal input should parse");
        assert!(input.check_roots.sources.is_empty());
        assert!(input.check_roots.stubs.is_empty());
        assert!(input.search_path.main_file.is_none());
        assert!(input.search_path.main_file_directory.is_none());
        assert!(input.search_path.explicit_imports.is_empty());
        assert!(!input.search_path.python_import_all_repositories);
        assert!(input.search_path.repository_roots.is_empty());
        assert!(input.path_overlays.is_empty());

        let config = input.config.parse().expect("minimal config should parse");
        assert_eq!(config.python_version, PythonVersion::default());
        assert_eq!(config.system_platform, PythonPlatform::default());
        assert!(config.preset.is_none());
        assert!(config.errors.is_none());
    }

    #[test]
    fn config_parses_to_pyrefly_types() {
        let config = parse_config_object(
            r#"{"python_version": "python3.10", "system_platform": "Linux", "preset": "strict", "error_severities": {"missing-attribute": "warn"}}"#,
        )
        .expect("valid config should parse");
        assert_eq!(config.python_version, PythonVersion::new(3, 10, 0));
        assert_eq!(config.system_platform, PythonPlatform::linux());
        assert_eq!(config.preset, Some(Preset::Strict));
        assert!(config.errors.is_some());
    }

    fn path(path: &str) -> PathBuf {
        PathBuf::from(path)
    }

    #[test]
    fn invalid_short_paths_are_rejected_before_logical_path_conversion() {
        assert!(validate_short_path("../repo/foo.py").is_ok());

        let invalid_short_path = |short_path: &str, expected: &str| {
            assert!(
                validate_short_path(short_path)
                    .expect_err("invalid short path should be rejected")
                    .to_string()
                    .contains(expected),
            );
        };

        invalid_short_path("/tmp/foo.py", "short path `/tmp/foo.py` must be relative");
        invalid_short_path(
            "pkg/../foo.py",
            "short path `pkg/../foo.py` must not contain `..` except leading `../` for external repos",
        );
        invalid_short_path(
            "pkg/./foo.py",
            "short path `pkg/./foo.py` must not contain `.` components",
        );
        invalid_short_path(
            "pkg//foo.py",
            "short path `pkg//foo.py` must not contain empty components or trailing slashes",
        );
        invalid_short_path(
            "pkg/foo.py/",
            "short path `pkg/foo.py/` must not contain empty components or trailing slashes",
        );
    }

    #[test]
    fn short_paths_translate_to_logical_and_physical_paths() {
        assert_eq!(
            logical_path_from_short_path("pkg/foo.py", "_main"),
            path("_main/pkg/foo.py"),
        );
        assert_eq!(
            logical_path_from_short_path("../rules_python+/python/runfiles.py", "_main"),
            path("rules_python+/python/runfiles.py"),
        );
        assert_eq!(
            default_physical_from_logical(&PathBuf::from("_main/pkg/foo.py"), "_main").unwrap(),
            path("pkg/foo.py"),
        );
        assert_eq!(
            default_physical_from_logical(
                &PathBuf::from("rules_python+/python/runfiles.py"),
                "_main"
            )
            .unwrap(),
            path("external/rules_python+/python/runfiles.py"),
        );
        assert_eq!(
            default_physical_from_logical(&PathBuf::from("_main"), "_main").unwrap(),
            path("."),
        );
        assert_eq!(
            physical_path_from_logical_import("_main", "_main"),
            path(".")
        );
        assert_eq!(
            physical_path_from_logical_import("_main/pkg", "_main"),
            path("pkg"),
        );
        assert_eq!(
            physical_path_from_logical_import("_main//pkg", "_main"),
            path("pkg"),
        );
        assert_eq!(
            physical_path_from_logical_import("rules_python+/python/runfiles.py", "_main"),
            path("external/rules_python+/python/runfiles.py"),
        );
    }

    #[test]
    fn check_root_uses_overlay_physical_path_when_present() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:generated", "workspace_name": "_main", "package": "pkg", "name": "generated", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/generated.py"]},
  "search_path": {"workspace_name": "_main"},
  "path_overlays": [{"short_path": "pkg/generated.py", "path": "bazel-out/darwin-fastbuild/bin/pkg/generated.py"}],
  "config": {}
}"#,
        )
        .expect("input should parse");

        let files = build_bazel_file_inputs(&input).expect("Bazel file inputs should build");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].raw_short_path, "pkg/generated.py");
        assert_eq!(files[0].logical_path, path("_main/pkg/generated.py"));
        assert_eq!(
            files[0].physical_path,
            path("bazel-out/darwin-fastbuild/bin/pkg/generated.py"),
        );
        assert!(files[0].is_check_root);
    }

    #[test]
    fn unchanged_overlay_physical_path_does_not_create_explicit_overlay_entry() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/lib.py"]},
  "search_path": {"workspace_name": "_main"},
  "path_overlays": [{"short_path": "../rules_python+/python/runfiles.py", "path": "external/rules_python+/python/runfiles.py"}],
  "config": {}
}"#,
        )
        .expect("input should parse");

        let files = build_bazel_file_inputs(&input).expect("Bazel file inputs should build");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].raw_short_path, "pkg/lib.py");
    }

    #[test]
    fn search_roots_match_rules_python_import_all_order() {
        let search_path = BazelSearchPath {
            main_file: Some("pkg/app.py".to_owned()),
            main_file_directory: Some("pkg".to_owned()),
            explicit_imports: vec![
                "_main/src".to_owned(),
                "rules_python++pip+pypi/site-packages".to_owned(),
            ],
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: true,
            repository_roots: vec!["z_repo".to_owned(), "a_repo".to_owned()],
        };

        let roots = build_search_roots(&search_path, PythonVersion::new(3, 10, 0))
            .expect("search roots should build");
        let logical_physical = roots
            .iter()
            .map(|root| (root.logical.clone(), root.physical.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            logical_physical,
            vec![
                (path("_main/pkg"), path("pkg")),
                (path("_main/src"), path("src")),
                (
                    path("rules_python++pip+pypi/site-packages"),
                    path("external/rules_python++pip+pypi/site-packages"),
                ),
                (path("a_repo"), path("external/a_repo")),
                (path("z_repo"), path("external/z_repo")),
            ],
        );
    }

    #[test]
    fn empty_main_file_directory_searches_workspace_root() {
        let search_path = BazelSearchPath {
            main_file: Some("app.py".to_owned()),
            main_file_directory: Some("".to_owned()),
            explicit_imports: vec!["_main/src".to_owned()],
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };

        let roots = build_search_roots(&search_path, PythonVersion::new(3, 10, 0))
            .expect("search roots should build");
        assert_eq!(
            roots
                .iter()
                .map(|root| (root.logical.clone(), root.physical.clone()))
                .collect::<Vec<_>>(),
            vec![(path("_main"), path(".")), (path("_main/src"), path("src"))],
        );
    }

    #[test]
    fn python_safe_path_omits_main_file_directory_search_root() {
        let search_path = BazelSearchPath {
            main_file: Some("pkg/app.py".to_owned()),
            main_file_directory: Some("pkg".to_owned()),
            explicit_imports: vec!["_main/src".to_owned()],
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };

        let roots = build_search_roots(&search_path, PythonVersion::new(3, 11, 0))
            .expect("search roots should build");
        assert_eq!(
            roots
                .iter()
                .map(|root| (root.logical.clone(), root.physical.clone()))
                .collect::<Vec<_>>(),
            vec![(path("_main/src"), path("src")), (path("_main"), path("."))],
        );
    }

    #[test]
    fn invalid_search_root_paths_are_rejected() {
        let absolute_explicit_import = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: vec!["/tmp".to_owned()],
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(&absolute_explicit_import, PythonVersion::new(3, 12, 0))
                .expect_err("absolute explicit import should be rejected")
                .to_string()
                .contains("explicit import `/tmp` must be relative"),
        );

        let absolute_repository_root = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: true,
            repository_roots: vec!["/tmp".to_owned()],
        };
        assert!(
            build_search_roots(&absolute_repository_root, PythonVersion::new(3, 12, 0))
                .expect_err("absolute repository root should be rejected")
                .to_string()
                .contains("repository root `/tmp` must be relative"),
        );

        let parent_prefixed_explicit_import = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: vec!["../repo".to_owned()],
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(
                &parent_prefixed_explicit_import,
                PythonVersion::new(3, 12, 0)
            )
            .expect_err("parent-prefixed explicit import should be rejected")
            .to_string()
            .contains("explicit import `../repo` must not contain `..`"),
        );

        let parent_prefixed_repository_root = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: true,
            repository_roots: vec!["../repo".to_owned()],
        };
        assert!(
            build_search_roots(
                &parent_prefixed_repository_root,
                PythonVersion::new(3, 12, 0)
            )
            .expect_err("parent-prefixed repository root should be rejected")
            .to_string()
            .contains("repository root `../repo` must not contain `..`"),
        );

        let absolute_main_file_directory = BazelSearchPath {
            main_file: Some("/tmp/app.py".to_owned()),
            main_file_directory: Some("/tmp".to_owned()),
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(&absolute_main_file_directory, PythonVersion::new(3, 10, 0))
                .expect_err("absolute main file directory should be rejected")
                .to_string()
                .contains("main_file_directory `/tmp` must be relative"),
        );

        let parent_traversal_main_file_directory = BazelSearchPath {
            main_file: Some("../app.py".to_owned()),
            main_file_directory: Some("../pkg".to_owned()),
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(
                &parent_traversal_main_file_directory,
                PythonVersion::new(3, 10, 0),
            )
            .expect_err("parent traversal main file directory should be rejected")
            .to_string()
            .contains("main_file_directory `../pkg` must not contain `..`"),
        );

        let repeated_separator_explicit_import = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: vec!["_main//pkg".to_owned()],
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(
                &repeated_separator_explicit_import,
                PythonVersion::new(3, 12, 0),
            )
            .expect_err("repeated separator explicit import should be rejected")
            .to_string()
            .contains(
                "explicit import `_main//pkg` must not contain empty components or trailing slashes"
            ),
        );

        let trailing_slash_repository_root = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: true,
            repository_roots: vec!["repo/".to_owned()],
        };
        assert!(
            build_search_roots(
                &trailing_slash_repository_root,
                PythonVersion::new(3, 12, 0)
            )
            .expect_err("trailing slash repository root should be rejected")
            .to_string()
            .contains(
                "repository root `repo/` must not contain empty components or trailing slashes"
            ),
        );

        let leading_cur_dir_main_file_directory = BazelSearchPath {
            main_file: Some("./pkg/app.py".to_owned()),
            main_file_directory: Some("./pkg".to_owned()),
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: false,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(
                &leading_cur_dir_main_file_directory,
                PythonVersion::new(3, 10, 0),
            )
            .expect_err("leading current directory main file directory should be rejected")
            .to_string()
            .contains("main_file_directory `./pkg` must not contain `.` components"),
        );
    }

    #[test]
    fn workspace_mismatch_is_rejected() {
        let workspace_mismatch = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/lib.py"]},
  "search_path": {"workspace_name": "other"},
  "config": {}
}"#,
        )
        .expect_err("workspace mismatch should be rejected");
        assert!(
            workspace_mismatch
                .to_string()
                .contains("does not match search path workspace"),
        );
    }

    #[test]
    fn import_all_requires_repository_roots() {
        let import_all_without_roots = BazelSearchPath {
            main_file: None,
            main_file_directory: None,
            explicit_imports: Vec::new(),
            workspace_name: "_main".to_owned(),
            python_import_all_repositories: true,
            repository_roots: Vec::new(),
        };
        assert!(
            build_search_roots(&import_all_without_roots, PythonVersion::new(3, 12, 0))
                .expect_err("import-all without repository roots should be rejected")
                .to_string()
                .contains("repository_roots"),
        );
    }

    #[test]
    fn non_python_check_roots_are_rejected() {
        let non_python_check_root = parse(
            r#"{
  "target": {"label": "//pkg:data", "workspace_name": "_main", "package": "pkg", "name": "data", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/data.txt"]},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input should parse");
        assert!(
            build_bazel_file_inputs(&non_python_check_root)
                .expect_err("non-Python check root should be rejected")
                .to_string()
                .contains("not a Python file"),
        );
    }

    #[test]
    fn invalid_check_root_paths_are_rejected() {
        let absolute_check_root = parse(
            r#"{
  "target": {"label": "//pkg:abs", "workspace_name": "_main", "package": "pkg", "name": "abs", "rule_kind": "py_library"},
  "check_roots": {"sources": ["/tmp/foo.py"]},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input should parse");
        assert!(
            format!(
                "{:#}",
                build_bazel_file_inputs(&absolute_check_root)
                    .expect_err("absolute check root should be rejected")
            )
            .contains("must be relative"),
        );

        let parent_traversal_check_root = parse(
            r#"{
  "target": {"label": "//pkg:traversal", "workspace_name": "_main", "package": "pkg", "name": "traversal", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/../evil.py"]},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input should parse");
        assert!(
            format!(
                "{:#}",
                build_bazel_file_inputs(&parent_traversal_check_root)
                    .expect_err("parent traversal check root should be rejected")
            )
            .contains("must not contain"),
        );
    }

    #[test]
    fn invalid_path_overlay_physical_paths_are_rejected() {
        let invalid_overlay_physical_path = |physical_path: &str, expected: &str| {
            let input = parse(&format!(
                r#"{{
  "target": {{"label": "//pkg:generated", "workspace_name": "_main", "package": "pkg", "name": "generated", "rule_kind": "py_library"}},
  "check_roots": {{"sources": ["pkg/generated.py"]}},
  "search_path": {{"workspace_name": "_main"}},
  "path_overlays": [{{"short_path": "pkg/generated.py", "path": "{physical_path}"}}],
  "config": {{}}
}}"#,
            ))
            .expect("input should parse");

            assert!(
                format!(
                    "{:#}",
                    build_bazel_file_inputs(&input)
                        .expect_err("invalid overlay physical path should be rejected")
                )
                .contains(expected),
            );
        };

        invalid_overlay_physical_path(
            "/tmp/generated.py",
            "path overlay physical path `/tmp/generated.py` must be relative",
        );
        invalid_overlay_physical_path(
            "../generated.py",
            "path overlay physical path `../generated.py` must not contain `..`",
        );
        invalid_overlay_physical_path("", "path overlay physical path must not be empty");
        invalid_overlay_physical_path(
            "bazel-out/./generated.py",
            "path overlay physical path `bazel-out/./generated.py` must not contain `.` components",
        );
        invalid_overlay_physical_path(
            "bazel-out//generated.py",
            "path overlay physical path `bazel-out//generated.py` must not contain empty components or trailing slashes",
        );
        invalid_overlay_physical_path(
            "bazel-out/generated.py/",
            "path overlay physical path `bazel-out/generated.py/` must not contain empty components or trailing slashes",
        );
    }

    #[test]
    fn identical_path_overlay_normalized_short_path_is_deduped() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:generated", "workspace_name": "_main", "package": "pkg", "name": "generated", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/app.py"]},
  "search_path": {"workspace_name": "_main"},
  "path_overlays": [
    {"short_path": "pkg/generated.py", "path": "bazel-out/bin/pkg/generated.py"},
    {"short_path": "pkg/generated.py", "path": "bazel-out/bin/pkg/generated.py"}
  ],
  "config": {}
}"#,
        )
        .expect("input should parse");

        let files = build_bazel_file_inputs(&input).expect("identical overlay should dedupe");
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].raw_short_path, "pkg/app.py");
        assert!(files[0].is_check_root);
        assert_eq!(files[1].raw_short_path, "pkg/generated.py");
        assert_eq!(files[1].logical_path, path("_main/pkg/generated.py"));
        assert_eq!(
            files[1].physical_path,
            path("bazel-out/bin/pkg/generated.py")
        );
        assert!(!files[1].is_check_root);
    }

    #[test]
    fn conflicting_path_overlay_normalized_short_path_is_rejected() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:generated", "workspace_name": "_main", "package": "pkg", "name": "generated", "rule_kind": "py_library"},
  "check_roots": {"sources": ["pkg/generated.py"]},
  "search_path": {"workspace_name": "_main"},
  "path_overlays": [
    {"short_path": "pkg/generated.py", "path": "bazel-out/bin/pkg/generated.py"},
    {"short_path": "../_main/pkg/generated.py", "path": "bazel-out/other/pkg/generated.py"}
  ],
  "config": {}
}"#,
        )
        .expect("input should parse");

        assert!(
            build_bazel_file_inputs(&input)
                .expect_err("conflicting normalized overlay short path should be rejected")
                .to_string()
                .contains("conflicts with another overlay"),
        );
    }

    fn search_root(logical: &str, physical: &str) -> BazelSearchRoot {
        BazelSearchRoot {
            logical: path(logical),
            physical: path(physical),
        }
    }

    fn file_input(
        raw_short_path: &str,
        logical_path: &str,
        physical_path: &str,
        is_check_root: bool,
    ) -> BazelFileInput {
        BazelFileInput {
            raw_short_path: raw_short_path.to_owned(),
            logical_path: path(logical_path),
            physical_path: path(physical_path),
            is_check_root,
        }
    }

    fn explicit_module_entry(
        module_name: &str,
        logical_path: &str,
        physical_path: &str,
        is_check_root: bool,
        rank: ExplicitModuleRank,
    ) -> ExplicitModuleEntry {
        ExplicitModuleEntry {
            module_name: ModuleName::from_str(module_name),
            module_path: ModulePath::filesystem(path(physical_path)),
            logical_path: path(logical_path),
            is_check_root,
            rank,
        }
    }

    #[test]
    fn module_name_uses_first_containing_search_root_and_top_level_stubs_suffix() {
        let roots = vec![search_root("_main/pkg", "pkg"), search_root("_main", ".")];
        let resolved =
            module_name_for_logical_path_under_search_roots(Path::new("_main/pkg/app.py"), &roots)
                .expect("module should resolve");
        assert_eq!(resolved.module_name, ModuleName::from_str("app"));
        assert_eq!(resolved.search_root_order, 0);
        assert_eq!(resolved.root_relative_path, path("app.py"));

        let roots = vec![search_root("rules_python+", "external/rules_python+")];
        let resolved = module_name_for_logical_path_under_search_roots(
            Path::new("rules_python+/foo-stubs/bar/baz.pyi"),
            &roots,
        )
        .expect("stub package module should resolve");
        assert_eq!(resolved.module_name, ModuleName::from_str("foo.bar.baz"));
        assert_eq!(resolved.root_relative_path, path("foo-stubs/bar/baz.pyi"));
    }

    #[test]
    fn source_db_style_filter_prefers_owned_style_with_fallback() {
        let entries = vec![
            explicit_module_entry(
                "pkg.mod",
                "_main/pkg/mod.py",
                "runtime/pkg/mod.py",
                true,
                ExplicitModuleRank {
                    is_stub_package: Reverse(false),
                    search_root_order: 0,
                    extension_priority: 1,
                },
            ),
            explicit_module_entry(
                "pkg.mod",
                "_main/pkg/mod.pyi",
                "stubs/pkg/mod.pyi",
                false,
                ExplicitModuleRank {
                    is_stub_package: Reverse(false),
                    search_root_order: 1,
                    extension_priority: 0,
                },
            ),
        ];
        let source_db = BazelCheckSourceDatabase::new(entries, SysInfo::default());

        assert_eq!(
            source_db.lookup(ModuleName::from_str("pkg.mod"), None, None),
            Some(ModulePath::filesystem(path("runtime/pkg/mod.py"))),
        );
        assert_eq!(
            source_db.lookup(
                ModuleName::from_str("pkg.mod"),
                None,
                Some(ModuleStyle::Interface),
            ),
            Some(ModulePath::filesystem(path("stubs/pkg/mod.pyi"))),
        );
        assert_eq!(
            source_db.lookup(
                ModuleName::from_str("pkg.mod"),
                None,
                Some(ModuleStyle::Executable),
            ),
            Some(ModulePath::filesystem(path("runtime/pkg/mod.py"))),
        );
        assert_eq!(
            source_db.lookup(
                ModuleName::from_str("pkg.missing"),
                None,
                Some(ModuleStyle::Interface),
            ),
            None,
        );
        assert_eq!(source_db.modules_to_check().len(), 1);
    }

    #[test]
    fn explicit_entries_reject_logical_physical_style_mismatch() {
        let roots = vec![search_root("_main", ".")];
        let error = build_explicit_entries(
            vec![file_input(
                "pkg/mod.pyi",
                "_main/pkg/mod.pyi",
                "bazel-out/pkg/mod.py",
                true,
            )],
            &roots,
            "//pkg:generated",
        )
        .expect_err("logical and physical module styles should agree");

        let message = format!("{error:#}");
        assert!(
            message.contains("_main/pkg/mod.pyi") && message.contains("bazel-out/pkg/mod.py"),
            "error should name both conflicting paths: {message}",
        );
    }

    #[test]
    fn source_db_rank_prefers_stub_packages_before_search_root_order() {
        let entries = vec![
            explicit_module_entry(
                "foo.bar",
                "_main/foo/bar.py",
                "runtime/foo/bar.py",
                false,
                ExplicitModuleRank {
                    is_stub_package: Reverse(false),
                    search_root_order: 0,
                    extension_priority: 1,
                },
            ),
            explicit_module_entry(
                "foo.bar",
                "external/foo-stubs/bar.pyi",
                "stubs/foo-stubs/bar.pyi",
                false,
                ExplicitModuleRank {
                    is_stub_package: Reverse(true),
                    search_root_order: 1,
                    extension_priority: 0,
                },
            ),
        ];
        let source_db = BazelCheckSourceDatabase::new(entries, SysInfo::default());
        assert_eq!(
            source_db.lookup(ModuleName::from_str("foo.bar"), None, None),
            Some(ModulePath::filesystem(path("stubs/foo-stubs/bar.pyi"))),
        );
    }

    #[test]
    fn check_root_outside_search_roots_is_rejected_but_overlay_is_skipped() {
        let roots = vec![search_root("_main/pkg", "pkg")];
        let explicit = build_explicit_entries(
            vec![file_input(
                "other/dep.py",
                "_main/other/dep.py",
                "other/dep.py",
                false,
            )],
            &roots,
            "//pkg:lib",
        )
        .expect("non-importable overlay should be skipped");
        assert!(explicit.is_empty());

        let error = build_explicit_entries(
            vec![file_input(
                "other/app.py",
                "_main/other/app.py",
                "other/app.py",
                true,
            )],
            &roots,
            "//pkg:app",
        )
        .expect_err("non-importable check root should fail");
        assert!(format!("{error:#}").contains("//pkg:app"));
        assert!(format!("{error:#}").contains("other/app.py"));
    }

    #[test]
    fn invalid_python_version_is_rejected() {
        let error = parse_config_object(r#"{"python_version": "python"}"#)
            .expect_err("invalid Python version should fail");
        assert!(
            error.to_string().contains("Invalid version string"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn invalid_preset_is_rejected() {
        let error =
            parse_config_object(r#"{"preset": "nope"}"#).expect_err("invalid preset should fail");
        assert!(
            format!("{error:#}").contains("invalid Bazel input config `preset`"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn invalid_error_severity_name_is_rejected() {
        let error = parse_config_object(r#"{"error_severities": {"not-a-real-error": "warn"}}"#)
            .expect_err("invalid error severity config should fail");
        assert!(
            format!("{error:#}").contains("invalid Bazel input config `error_severities`"),
            "unexpected error: {error:#}"
        );
    }
}
