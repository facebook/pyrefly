/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::base::ConfigBase;
use pyrefly_config::config::DirectoryRelativeFallbackSearchPathCache;
use pyrefly_config::config::FallbackSearchPath;
use pyrefly_config::config::GENERATED_FILE_CONFIG_OVERRIDE;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lock::Mutex;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::config::config::ProjectLayout;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::config::finder::debug_log;
use crate::module::bundled::BundledStub;
use crate::module::third_party::BundledThirdParty;
use crate::module::typeshed::BundledTypeshedStdlib;
use crate::module::typeshed_third_party::BundledTypeshedThirdParty;

/// Finalizes a config before being returned by a [`ConfigFinder`].
pub trait ConfigConfigurer: Send + Sync + 'static {
    /// Sets additional options on, and calls configure to finalize and validate
    /// [`ConfigFile`]s before being returned.
    ///
    /// `root` is the root of the project (directory the config/marker file was found in or
    /// directory of the Python file we're loading this config for, if no config/marker).
    /// This may be `None` if [`Path::parent()`] doesn't exist or if it's irrelevant
    /// (bundled typeshed).
    ///
    /// `config` is the configuration loaded from disk or constructed by the
    /// [`standard_config_finder`] when no config can be found.
    ///
    /// `errors` are any errors that occurred while parsing the config. Any
    /// new errors that occur as a result of configuring should be added to `errors`
    /// and handled in the same manner. In most cases, this means appending any
    /// errors to `errors` then returning `errors`. Note:
    /// - If `configure` handles outputting error information, it is recommended
    ///   to [`Vec::clear()`] the errors to avoid duplicate error message output.
    /// - If the `configure` function should handle outputting error information,
    ///   ensure any pre-existing parse errors in `errors` are also output.
    ///
    /// Returns a tuple containing the completed [`ConfigFile`] and any [`ConfigError`]s
    /// that occurred during parsing or configuring that weren't already output during
    /// this [`Self::configure`] call.
    fn configure(
        &self,
        root: Option<&Path>,
        config: ConfigFile,
        errors: Vec<ConfigError>,
    ) -> (ArcId<ConfigFile>, Vec<ConfigError>);
}

/// A basic [`ConfigConfigurer`] implementation that only calls [`ConfigFile::configure()`]
/// and returns the configured config. Any errors are ignored, and an empty [`Vec<ConfigError>`]
/// is always returned.
pub struct DefaultConfigConfigurer {}

impl ConfigConfigurer for DefaultConfigConfigurer {
    fn configure(
        &self,
        _: Option<&std::path::Path>,
        mut config: ConfigFile,
        _: Vec<pyrefly_config::finder::ConfigError>,
    ) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
        config.configure();
        (ArcId::new(config), Vec::new())
    }
}

pub fn default_config_finder() -> ConfigFinder {
    standard_config_finder(Arc::new(DefaultConfigConfigurer {}))
}

struct DefaultConfigConfigurerWithOverrides {
    args: ConfigOverrideArgs,
    ignore_errors: bool,
}

impl DefaultConfigConfigurerWithOverrides {
    fn new(args: ConfigOverrideArgs, ignore_errors: bool) -> Self {
        Self {
            args,
            ignore_errors,
        }
    }
}

impl ConfigConfigurer for DefaultConfigConfigurerWithOverrides {
    fn configure(
        &self,
        _: Option<&Path>,
        config: ConfigFile,
        mut errors: Vec<ConfigError>,
    ) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
        let (c, mut configure_errors) = self.args.override_config(config);
        if self.ignore_errors {
            errors.clear();
        } else {
            errors.append(&mut configure_errors);
        }
        (c, errors)
    }
}

pub fn default_config_finder_with_overrides(
    args: ConfigOverrideArgs,
    ignore_errors: bool,
) -> ConfigFinder {
    standard_config_finder(Arc::new(DefaultConfigConfigurerWithOverrides::new(
        args,
        ignore_errors,
    )))
}

/// Create a standard `ConfigFinder`, using the provided [`ConfigConfigurer`] to finalize
/// the config before caching/returning it.
pub fn standard_config_finder(configure: Arc<dyn ConfigConfigurer>) -> ConfigFinder {
    let configure2 = configure.dupe();
    let configure3 = configure.dupe();

    // A cache where path `p` maps to config file with `search_path = [p]`. If we can find the root.
    let cache_one: Arc<Mutex<SmallMap<PathBuf, ArcId<ConfigFile>>>> =
        Arc::new(Mutex::new(SmallMap::new()));
    // A cache where path `p` maps to config file with
    // `fallback_search_path = [p, p/.., p/../.., ...]`.
    let cache_parents: Arc<Mutex<SmallMap<PathBuf, ArcId<ConfigFile>>>> =
        Arc::new(Mutex::new(SmallMap::new()));
    // A cache where path `p` maps to search paths from `p` to the nearest on-disk config,
    // marker file, or filesystem root. Used as the `fallback_search_path` in `cache_parents`.
    let cache_ancestors: Arc<DirectoryRelativeFallbackSearchPathCache> =
        Arc::new(DirectoryRelativeFallbackSearchPathCache::new(None));

    let clear_extra_caches = {
        let cache_one = cache_one.dupe();
        let cache_parents = cache_parents.dupe();
        let cache_ancestors = cache_ancestors.dupe();
        Box::new(move || {
            cache_one.lock().clear();
            cache_parents.lock().clear();
            cache_ancestors.clear();
            GENERATED_FILE_CONFIG_OVERRIDE.write().clear();
        })
    };

    let empty = LazyLock::new(move || {
        let (config, errors) = configure3.configure(None, ConfigFile::default(), vec![]);
        // Since this is a config we generated, these are likely internal errors.
        debug_log(errors);
        config
    });

    ConfigFinder::new_custom(
        Box::new(move |_, path| {
            Ok(GENERATED_FILE_CONFIG_OVERRIDE
                .read()
                .get(&path.module_path_buf())
                .cloned())
        }),
        Box::new(move |file| {
            let (file_config, parse_errors) = ConfigFile::from_file(file);
            let (config, validation_errors) =
                configure.configure(file.parent(), file_config, parse_errors);
            (config, validation_errors)
        }),
        // Fall back to using a default config, but let's see if we can make the `search_path` somewhat useful
        // based on a few heuristics.
        Box::new(move |name, path| match path.root_of(name) {
            // We were able to walk up `path` and match each component of `name` to a directory until we ran out.
            // That means the resulting path is likely the root of the 'project', and should therefore be its `search_path`.
            Some(path) => cache_one
                .lock()
                .entry(path.clone())
                .or_insert_with(|| {
                    let (config, errors) = configure2.configure(
                        path.parent(),
                        ConfigFile::init_at_root(&path, &ProjectLayout::Flat, true),
                        vec![],
                    );
                    // Since this is a config we generated, these are likely internal errors.
                    debug_log(errors);
                    config
                })
                .dupe(),

            // We couldn't walk up and find a possible root of the project, so let's try to create a search
            // path that is still useful for this import by including all of its parents.
            None => {
                let parent = match path.details() {
                    ModulePathDetails::FileSystem(x) | ModulePathDetails::Memory(x) => {
                        if let Some(path) = x.parent() {
                            path
                        } else {
                            return empty.dupe();
                        }
                    }
                    ModulePathDetails::Namespace(x) => x.as_path(),
                    ModulePathDetails::BundledTypeshed(_) => {
                        return BundledTypeshedStdlib::config();
                    }
                    ModulePathDetails::BundledTypeshedThirdParty(_) => {
                        return BundledTypeshedThirdParty::config();
                    }
                    ModulePathDetails::BundledThirdParty(_) => {
                        return BundledThirdParty::config();
                    }
                };
                cache_parents
                    .lock()
                    .entry(parent.to_owned())
                    .or_insert_with(|| {
                        let fallback_search_path =
                            FallbackSearchPath::Explicit(cache_ancestors.get_ancestors(parent));
                        let mut config = ConfigFile {
                            project_includes: ConfigFile::default_project_includes(),
                            // We use `fallback_search_path` because otherwise a user with `/sys` on their
                            // computer (all of them) will override `sys.version` in preference to typeshed.
                            fallback_search_path,
                            root: ConfigBase::default_for_ide_without_config(),
                            ..Default::default()
                        };
                        config.rewrite_with_path_to_config(parent);
                        let (config, errors) = configure2.configure(Some(parent), config, vec![]);
                        // Since this is a config we generated, these are likely internal errors.
                        debug_log(errors);
                        config
                    })
                    .dupe()
            }
        }),
        clear_extra_caches,
    )
}

#[cfg(test)]
mod tests {

    use std::ops::Deref as _;

    use pretty_assertions::assert_eq;
    use pyrefly_config::args::ConfigOverrideArgs;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_util::test_path::TestPath;

    use super::*;
    use crate::config::config::ConfigSource;
    use crate::config::environment::environment::PythonEnvironment;

    struct TestConfigurer(
        Box<
            dyn Fn(
                    Option<&Path>,
                    ConfigFile,
                    Vec<ConfigError>,
                ) -> (ArcId<ConfigFile>, Vec<ConfigError>)
                + Send
                + Sync,
        >,
    );

    impl TestConfigurer {
        fn new_standard(
            f: impl Fn(
                Option<&Path>,
                ConfigFile,
                Vec<ConfigError>,
            ) -> (ArcId<ConfigFile>, Vec<ConfigError>)
            + Send
            + Sync
            + 'static,
        ) -> ConfigFinder {
            standard_config_finder(Arc::new(TestConfigurer(Box::new(f))))
        }
    }

    impl ConfigConfigurer for TestConfigurer {
        fn configure(
            &self,
            root: Option<&Path>,
            config: ConfigFile,
            errors: Vec<ConfigError>,
        ) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
            (self.0)(root, config, errors)
        }
    }

    #[test]
    fn test_site_package_path_from_environment() {
        let args = ConfigOverrideArgs::default();
        let config = TestConfigurer::new_standard(move |_, x, _| args.override_config(x))
            .python_file(ModuleName::unknown(), &ModulePath::filesystem("".into()));
        let env = PythonEnvironment::get_default_interpreter_env();
        if let Some(paths) = env.site_package_path {
            for p in paths {
                assert!(config.site_package_path().collect::<Vec<_>>().contains(&&p));
            }
        }
    }

    #[test]
    fn test_fallback_search_path_fallback() {
        fn finder(
            expect_dir: Option<&Path>,
            module_name: ModuleName,
            module_path: ModulePath,
        ) -> ArcId<ConfigFile> {
            let expect_dir = expect_dir.map(|p| p.to_path_buf());
            let module_path2 = module_path.clone();
            TestConfigurer::new_standard(move |dir, x, _| {
                assert_eq!(
                    dir.map(|p| p.to_path_buf()),
                    expect_dir,
                    "failed for {expect_dir:?}, {module_name}, {module_path}"
                );
                (ArcId::new(x), Vec::new())
            })
            .python_file(module_name, &module_path2)
        }

        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "with_config",
                    vec![
                        TestPath::file("pyrefly.toml"),
                        TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                    ],
                ),
                TestPath::dir(
                    "no_config",
                    vec![TestPath::dir("foo", vec![TestPath::file("bar.py")])],
                ),
            ],
        );

        // we shouldn't do anything to the search path when we found a config file on disk
        let config_file = finder(
            Some(&root.join("with_config")),
            ModuleName::from_str("foo.bar"),
            ModulePath::filesystem(root.join("with_config/foo/bar.py")),
        );
        assert_eq!(
            config_file.source,
            ConfigSource::File(root.join("with_config/pyrefly.toml"))
        );
        assert_eq!(
            config_file.search_path().cloned().collect::<Vec<_>>(),
            vec![root.join("with_config")]
        );
        assert_eq!(config_file.fallback_search_path, FallbackSearchPath::Empty);

        // we should get a synthetic config with a search path = project_root/..
        let config_file = finder(
            Some(root),
            ModuleName::from_str("foo.bar"),
            ModulePath::filesystem(root.join("no_config/foo/bar.py")),
        );
        assert_eq!(config_file.source, ConfigSource::Synthetic);
        assert_eq!(
            config_file.search_path().cloned().collect::<Vec<_>>(),
            Vec::<PathBuf>::new()
        );
        assert_eq!(
            config_file.fallback_search_path,
            FallbackSearchPath::Explicit(Arc::new(vec![root.join("no_config")])),
        );

        // check invalid module path parent
        assert_eq!(
            finder(
                None,
                ModuleName::from_str("foo.bar"),
                ModulePath::filesystem(PathBuf::from("/")),
            )
            .deref(),
            &ConfigFile::default(),
        );

        // check typeshed
        assert_eq!(
            finder(
                None,
                ModuleName::from_str("foo.bar"),
                ModulePath::bundled_typeshed(PathBuf::from("bundled_typeshed")),
            ),
            BundledTypeshedStdlib::config(),
        );

        // check namespace
        let config_file = finder(
            Some(&root.join("no_config/foo")),
            ModuleName::unknown(),
            ModulePath::namespace(root.join("no_config/foo")),
        );
        assert_eq!(config_file.source, ConfigSource::Synthetic);
        assert_eq!(config_file.search_path_from_file, Vec::<PathBuf>::new());
        assert_eq!(
            config_file.fallback_search_path,
            FallbackSearchPath::Explicit(Arc::new(
                [root.join("no_config/foo"), root.join("no_config")]
                    .into_iter()
                    .chain(root.ancestors().map(PathBuf::from))
                    .collect::<Vec<PathBuf>>()
            )),
        );

        // check filesystem/memory
        let config_file = finder(
            Some(&root.join("no_config/foo")),
            ModuleName::unknown(),
            ModulePath::filesystem(root.join("no_config/foo/bar.py")),
        );
        assert_eq!(config_file.source, ConfigSource::Synthetic);
        assert_eq!(config_file.search_path_from_file, Vec::<PathBuf>::new());
        assert_eq!(
            config_file.fallback_search_path,
            FallbackSearchPath::Explicit(Arc::new(
                [root.join("no_config/foo"), root.join("no_config")]
                    .into_iter()
                    .chain(root.ancestors().map(PathBuf::from))
                    .collect::<Vec<PathBuf>>()
            )),
        );
    }
}
