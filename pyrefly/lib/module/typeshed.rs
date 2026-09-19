/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::Context as _;
use anyhow::anyhow;
use dupe::Dupe;
use pyrefly_bundled::bundled_typeshed;
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

impl VersionRange {
    fn parse(range: &str) -> anyhow::Result<Self> {
        let (min, max) = range
            .split_once('-')
            .with_context(|| format!("Invalid typeshed version range `{range}`"))?;
        Ok(Self {
            min: min.parse()?,
            max: if max.is_empty() {
                None
            } else {
                Some(max.parse()?)
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
    versions: SmallMap<ModuleName, VersionRange>,
}

impl BundledStub for BundledTypeshedStdlib {
    fn new() -> anyhow::Result<Self> {
        let (contents, versions) = bundled_typeshed()?;
        let versions = parse_versions(&versions)?;
        let provider = contents
            .into_iter()
            .map(|(relative_path, contents)| BundleFile {
                import_path: relative_path.clone(),
                storage_path: relative_path,
                contents,
            });
        Ok(Self {
            bundle: Bundle::new(provider)?,
            versions,
        })
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

impl BundledTypeshedStdlib {
    pub fn has_module(&self, module: ModuleName) -> bool {
        self.bundle.find(module).is_some()
    }

    pub fn is_available_for_python_version(
        &self,
        module: ModuleName,
        version: PythonVersion,
    ) -> bool {
        self.has_module(module) && self.version_range(module).contains(version)
    }

    fn version_range(&self, module: ModuleName) -> VersionRange {
        let mut current = Some(module);
        while let Some(module) = current {
            if let Some(range) = self.versions.get(&module) {
                return *range;
            }
            current = module.parent();
        }
        unreachable!("Bundled typeshed module `{module}` missing stdlib/VERSIONS metadata");
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
        self.bundle
            .modules()
            .filter(move |module| self.version_range(*module).contains(version))
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
}
