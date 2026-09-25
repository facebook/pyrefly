/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::iter::successors;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context as _;
use anyhow::anyhow;
use dupe::Dupe;
use pyrefly_bundled::bundled_typeshed;
use pyrefly_bundled::bundled_typeshed_versions;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::arc_id::ArcId;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::module::bundled::Bundle;
use crate::module::bundled::BundleFile;
use crate::module::bundled::BundledStub;
use crate::module::bundled::create_bundled_stub_config;

#[derive(Debug, Clone, Copy)]
struct VersionRange {
    min: PythonVersion,
    max: Option<PythonVersion>,
}

/// Parse a `<major>.<minor>` bound from `stdlib/VERSIONS`.
///
/// Deliberately not `PythonVersion::from_str`: that is an unanchored regex, recompiled on
/// every call, which defaults a missing minor to the *current default* version. Under it a
/// truncated bound like `3` parses as 3.13 rather than failing, which would silently hide
/// whole ranges of modules.
fn parse_version_bound(bound: &str) -> anyhow::Result<PythonVersion> {
    fn component(part: &str) -> Option<u32> {
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        part.parse().ok()
    }

    let invalid =
        || anyhow!("Invalid typeshed version bound `{bound}`, expected `<major>.<minor>`");
    let (major, minor) = bound.split_once('.').ok_or_else(invalid)?;
    let (Some(major), Some(minor)) = (component(major), component(minor)) else {
        return Err(invalid());
    };
    Ok(PythonVersion::new(major, minor, 0))
}

impl VersionRange {
    /// typeshed writes `<min>-` for a module that still exists and `<min>-<max>` for one that
    /// was removed after `max`.
    fn parse(range: &str) -> anyhow::Result<Self> {
        let (min, max) = range
            .split_once('-')
            .with_context(|| format!("Invalid typeshed version range `{range}`"))?;
        Ok(Self {
            min: parse_version_bound(min)?,
            max: if max.is_empty() {
                None
            } else {
                Some(parse_version_bound(max)?)
            },
        })
    }

    fn contains(self, version: PythonVersion) -> bool {
        version.cmp_ignore_patch(self.min).is_ge()
            && self
                .max
                .is_none_or(|max| version.cmp_ignore_patch(max).is_le())
    }
}

#[derive(Debug, Clone)]
pub struct BundledTypeshedStdlib {
    bundle: Bundle,
    /// Availability of every module in `bundle`, resolved at load. Every key of `bundle` is a
    /// key here, so a version lookup is one map read and cannot fail.
    versions: SmallMap<ModuleName, VersionRange>,
}

impl BundledStub for BundledTypeshedStdlib {
    fn new() -> anyhow::Result<Self> {
        let declared = parse_versions(bundled_typeshed_versions())?;
        let provider = bundled_typeshed()?
            .into_iter()
            .map(|(relative_path, contents)| BundleFile {
                import_path: relative_path.clone(),
                storage_path: relative_path,
                contents,
            });
        let bundle = Bundle::new(provider)?;
        let versions = resolve_versions(bundle.modules(), &declared)?;
        Ok(Self { bundle, versions })
    }

    fn find(&self, module: ModuleName) -> Option<ModulePath> {
        self.bundle
            .find(module)
            .map(|path| ModulePath::bundled_typeshed(path.clone()))
    }

    fn load(&self, path: &Path) -> Option<Arc<String>> {
        self.bundle.load(path)
    }

    fn load_map(&self) -> impl Iterator<Item = (&PathBuf, &Arc<String>)> {
        self.bundle.load_map()
    }

    fn modules(&self) -> impl Iterator<Item = ModuleName> {
        self.bundle.modules()
    }

    fn get_path_name(&self) -> String {
        format!(
            "pyrefly_bundled_typeshed_{}",
            faster_hex::hex_string(&pyrefly_bundled::BUNDLED_TYPESHED_DIGEST[0..6])
        )
    }

    fn config() -> ArcId<ConfigFile> {
        static CONFIG: LazyLock<ArcId<ConfigFile>> = LazyLock::new(|| {
            let config_file = create_bundled_stub_config(
                Some(Vec::new()),
                Some(stdlib_error_overrides()),
                Some(true),
            );
            ArcId::new(config_file)
        });
        CONFIG.dupe()
    }
}

/// Error kinds that must be ignored when type-checking the stdlib stubs themselves.
/// The stdlib deliberately contains incorrect overrides and variance violations
/// (e.g. in `typing.pyi`) that are not real errors for our purposes.
fn stdlib_error_overrides() -> HashMap<ErrorKind, Severity> {
    HashMap::from([
        (ErrorKind::BadOverride, Severity::Ignore),
        (ErrorKind::BadOverrideParamName, Severity::Ignore),
        (ErrorKind::InvalidVariance, Severity::Ignore),
    ])
}

/// Config used to load the `Stdlib` from a user-provided typeshed directory from the
/// `typeshed_path` config option. Stdlib modules will be resolved from
/// `<typeshed_path>/stdlib` on disk; missing modules fall back to the bundled typeshed,
/// matching how `typeshed_path` already behaves for ordinary import resolution.
pub fn custom_typeshed_stdlib_config(typeshed_path: PathBuf) -> ArcId<ConfigFile> {
    let mut config_file =
        create_bundled_stub_config(None, Some(stdlib_error_overrides()), Some(true));
    config_file.typeshed_path = Some(typeshed_path);
    config_file.configure();
    ArcId::new(config_file)
}

/// The entries `stdlib/VERSIONS` states outright, before inheritance is applied.
fn parse_versions(contents: &str) -> anyhow::Result<SmallMap<ModuleName, VersionRange>> {
    let mut versions = SmallMap::new();
    for line in contents.lines() {
        let line = line.split_once('#').map_or(line, |(line, _)| line).trim();
        if line.is_empty() {
            continue;
        }
        let (module, range) = line
            .split_once(':')
            .with_context(|| format!("Invalid typeshed VERSIONS entry `{line}`"))?;
        versions.insert(
            ModuleName::from_str(module.trim()),
            VersionRange::parse(range.trim())?,
        );
    }
    Ok(versions)
}

/// Give every bundled module an explicit version range. typeshed declares a module in
/// `stdlib/VERSIONS` only when its availability differs from its parent's, so a module
/// inherits its nearest declared ancestor.
///
/// Resolving at load means a gap in the metadata surfaces once, as a load error naming the
/// modules, rather than as a failure on whichever lookup happens to reach it first.
fn resolve_versions(
    modules: impl Iterator<Item = ModuleName>,
    declared: &SmallMap<ModuleName, VersionRange>,
) -> anyhow::Result<SmallMap<ModuleName, VersionRange>> {
    /// Enough to recognize the pattern in a failure without pasting hundreds of names.
    const MAX_REPORTED: usize = 10;

    let mut resolved = SmallMap::new();
    let mut missing = Vec::new();
    for module in modules {
        match successors(Some(module), ModuleName::parent)
            .find_map(|ancestor| declared.get(&ancestor))
        {
            Some(range) => {
                resolved.insert(module, *range);
            }
            None => missing.push(module),
        }
    }

    if !missing.is_empty() {
        missing.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        let shown = missing
            .iter()
            .take(MAX_REPORTED)
            .map(ModuleName::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        let ellipsis = if missing.len() > MAX_REPORTED {
            ", ..."
        } else {
            ""
        };
        return Err(anyhow!(
            "Bundled typeshed stdlib/VERSIONS has no entry for {} module(s) or any of their parents: {shown}{ellipsis}",
            missing.len()
        ));
    }
    Ok(resolved)
}

impl BundledTypeshedStdlib {
    pub fn has_module(&self, module: ModuleName) -> bool {
        self.versions.contains_key(&module)
    }

    pub fn is_available_for_python_version(
        &self,
        module: ModuleName,
        version: PythonVersion,
    ) -> bool {
        self.versions
            .get(&module)
            .is_some_and(|range| range.contains(version))
    }

    pub fn find_for_python_version(
        &self,
        module: ModuleName,
        version: PythonVersion,
    ) -> Option<ModulePath> {
        if !self.is_available_for_python_version(module, version) {
            return None;
        }
        self.find(module)
    }

    pub fn modules_for_python_version(
        &self,
        version: PythonVersion,
    ) -> impl Iterator<Item = ModuleName> + '_ {
        self.versions
            .iter()
            .filter_map(move |(module, range)| range.contains(version).then_some(*module))
    }
}

static BUNDLED_TYPESHED: LazyLock<anyhow::Result<BundledTypeshedStdlib>> =
    LazyLock::new(BundledTypeshedStdlib::new);

pub fn typeshed() -> anyhow::Result<&'static BundledTypeshedStdlib> {
    match &*BUNDLED_TYPESHED {
        Ok(typeshed) => Ok(typeshed),
        Err(error) => Err(anyhow!("{error:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::bundled::assert_bundle_order_independent;

    #[test]
    fn test_typeshed_materialize() {
        let typeshed = typeshed().unwrap();
        let path = typeshed.materialized_path_on_disk().unwrap();
        // Do it twice, to check that works.
        typeshed.materialized_path_on_disk().unwrap();
        typeshed.write(&path).unwrap();
    }

    #[test]
    fn test_typeshed_lookup_is_file_order_independent() {
        let typeshed = typeshed().unwrap();
        assert_bundle_order_independent(typeshed.load_map().map(|(path, contents)| BundleFile {
            import_path: path.clone(),
            storage_path: path.clone(),
            contents: contents.as_str().to_owned(),
        }));
    }

    #[test]
    fn test_typeshed_respects_versions_file() {
        let typeshed = typeshed().unwrap();
        assert!(
            typeshed
                .find_for_python_version(
                    ModuleName::from_str("distutils"),
                    PythonVersion::new(3, 11, 9)
                )
                .is_some()
        );
        assert!(
            typeshed
                .find_for_python_version(
                    ModuleName::from_str("distutils"),
                    PythonVersion::new(3, 12, 1)
                )
                .is_none()
        );
        assert!(
            typeshed
                .find_for_python_version(
                    ModuleName::from_str("distutils.version"),
                    PythonVersion::new(3, 12, 0)
                )
                .is_none()
        );
        assert!(
            typeshed
                .find_for_python_version(
                    ModuleName::from_str("graphlib"),
                    PythonVersion::new(3, 8, 0)
                )
                .is_none()
        );
        assert!(
            typeshed
                .find_for_python_version(
                    ModuleName::from_str("graphlib"),
                    PythonVersion::new(3, 9, 0)
                )
                .is_some()
        );
    }

    /// Construction proves every bundled module has metadata, but not the converse: an entry
    /// naming a module we no longer ship is silently ignored, and signals that the stubs and
    /// their metadata have drifted apart.
    #[test]
    fn test_every_versions_entry_names_a_bundled_module() {
        let typeshed = typeshed().unwrap();
        let declared = parse_versions(bundled_typeshed_versions()).unwrap();

        let unbundled = declared
            .keys()
            .filter(|module| !typeshed.has_module(**module))
            .collect::<Vec<_>>();

        assert!(
            unbundled.is_empty(),
            "stdlib/VERSIONS lists modules that the bundled stdlib does not provide: {unbundled:?}"
        );
    }

    #[test]
    fn test_parse_version_bound_requires_exact_major_minor() {
        assert_eq!(
            parse_version_bound("3.11").unwrap(),
            PythonVersion::new(3, 11, 0)
        );
        // `PythonVersion::from_str` accepts all of these; a version bound must not.
        for bound in [
            "3", "", "3.", ".11", "3.11.2", "3.x", "junk3.11", " 3.11", "+3.11",
        ] {
            assert!(
                parse_version_bound(bound).is_err(),
                "`{bound}` should not parse as a version bound"
            );
        }
    }

    #[test]
    fn test_parse_versions_rejects_a_truncated_bound() {
        // Under the regex parser this silently became 3.13, hiding every module below it.
        assert!(parse_versions("distutils: 3-3.11").is_err());
    }

    fn declared(entries: &[(&str, &str)]) -> SmallMap<ModuleName, VersionRange> {
        entries
            .iter()
            .map(|(module, range)| {
                (
                    ModuleName::from_str(module),
                    VersionRange::parse(range).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn test_resolve_versions_inherits_the_nearest_declared_ancestor() {
        let declared = declared(&[("email", "3.0-"), ("email.mime.text", "3.4-3.11")]);
        let modules = ["email", "email.utils", "email.mime", "email.mime.text"]
            .map(ModuleName::from_str)
            .into_iter();

        let resolved = resolve_versions(modules, &declared).unwrap();

        // `email.utils` and `email.mime` are undeclared, so they take `email`'s range, while
        // `email.mime.text` keeps its own rather than its parent's.
        for module in ["email", "email.utils", "email.mime"] {
            let range = resolved.get(&ModuleName::from_str(module)).unwrap();
            assert!(range.contains(PythonVersion::new(3, 12, 0)), "{module}");
        }
        let text = resolved
            .get(&ModuleName::from_str("email.mime.text"))
            .unwrap();
        assert!(text.contains(PythonVersion::new(3, 11, 9)));
        assert!(!text.contains(PythonVersion::new(3, 12, 0)));
        assert!(!text.contains(PythonVersion::new(3, 3, 0)));
    }

    #[test]
    fn test_resolve_versions_reports_modules_with_no_metadata() {
        let declared = declared(&[("email", "3.0-")]);
        let modules = ["email.utils", "zoneinfo", "tomllib"]
            .map(ModuleName::from_str)
            .into_iter();

        let error = resolve_versions(modules, &declared)
            .unwrap_err()
            .to_string();

        assert!(error.contains("2 module(s)"), "{error}");
        assert!(
            error.contains("tomllib") && error.contains("zoneinfo"),
            "{error}"
        );
        assert!(!error.contains("email.utils"), "{error}");
    }
}
