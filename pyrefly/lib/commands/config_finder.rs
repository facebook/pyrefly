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
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lock::Mutex;
use starlark_map::small_map::SmallMap;

use crate::config::config::ConfigFile;
use crate::config::config::ProjectLayout;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::config::finder::debug_log;
use crate::module::bundled::BundledTypeshed;

/// Create a standard `ConfigFinder`. The `configure` function is expected to set any additional options,
/// then call `configure` and `validate`.
/// The `path` to `configure` is a directory, either to the python file or the config file.
/// In the case we can't find a config by walking up, we create a default config, setting the `search_path`
/// to a sensible default if possible.
///
/// `configure` is a function that accepts a `Option<&Path>` representing the root of the project
/// (the directory the config was found in), and the `ConfigFile` that was loaded/created. The root
/// of the project may be `None` if it can't be found (no [`Path::parent()`]) or it's irrelevant (bundled typeshed).
pub fn standard_config_finder(
    configure: Arc<
        dyn Fn(Option<&Path>, ConfigFile) -> (ArcId<ConfigFile>, Vec<ConfigError>) + Send + Sync,
    >,
) -> ConfigFinder {
    let configure2 = configure.dupe();
    let configure3 = configure.dupe();

    // A cache where path `p` maps to config file with `search_path = [p]`. If we can find the root.
    let cache_one: Arc<Mutex<SmallMap<PathBuf, ArcId<ConfigFile>>>> =
        Arc::new(Mutex::new(SmallMap::new()));
    // A cache where path `p` maps to config file with `search_path = [p, p/.., p/../.., ...]`.
    let cache_parents: Arc<Mutex<SmallMap<PathBuf, ArcId<ConfigFile>>>> =
        Arc::new(Mutex::new(SmallMap::new()));

    let clear_extra_caches = {
        let cache_one = cache_one.dupe();
        let cache_parents = cache_parents.dupe();
        Box::new(move || {
            cache_one.lock().clear();
            cache_parents.lock().clear();
        })
    };

    let empty = LazyLock::new(move || {
        let (config, errors) = configure3(None, ConfigFile::default());
        // Since this is a config we generated, these are likely internal errors.
        debug_log(errors);
        config
    });

    ConfigFinder::new(
        Box::new(move |file| {
            let (file_config, parse_errors) = ConfigFile::from_file(file);
            let (config, validation_errors) = configure(file.parent(), file_config);
            (
                config,
                parse_errors.into_iter().chain(validation_errors).collect(),
            )
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
                    let (config, errors) = configure2(
                        path.parent(),
                        ConfigFile::init_at_root(&path, &ProjectLayout::Flat),
                    );
                    // Since this is a config we generated, these are likely internal errors.
                    debug_log(errors);
                    config
                })
                .dupe(),

            // We couldn't walk up and find a possible root of the project, so let's try to create a search
            // path that is still useful for this import by including all of its parents.
            None => {
                let path = match path.details() {
                    ModulePathDetails::FileSystem(x) | ModulePathDetails::Memory(x) => {
                        if let Some(path) = x.parent() {
                            path
                        } else {
                            return empty.dupe();
                        }
                    }
                    ModulePathDetails::Namespace(x) => x.as_path(),
                    ModulePathDetails::BundledTypeshed(_) => {
                        return BundledTypeshed::config();
                    }
                };
                cache_parents
                    .lock()
                    .entry(path.to_owned())
                    .or_insert_with(|| {
                        let (config, errors) = configure2(
                            path.parent(),
                            ConfigFile {
                                // We use `fallback_search_path` because otherwise a user with `/sys` on their
                                // computer (all of them) will override `sys.version` in preference to typeshed.
                                fallback_search_path: path
                                    .ancestors()
                                    .map(|x| x.to_owned())
                                    .collect::<Vec<_>>(),
                                ..Default::default()
                            },
                        );
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

    use std::ffi::OsString;
    use std::ops::Deref as _;

    use clap::Parser;
    use pretty_assertions::assert_eq;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_util::test_path::TestPath;

    use super::*;
    use crate::commands::check::Args;
    use crate::config::config::ConfigSource;
    use crate::config::environment::environment::PythonEnvironment;

    #[test]
    fn test_site_package_path_from_environment() {
        let args = Args::parse_from(Vec::<OsString>::new().iter());
        let config = standard_config_finder(Arc::new(move |_, x| args.override_config(x)))
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
            standard_config_finder(Arc::new(move |dir, x| {
                assert_eq!(
                    dir.map(|p| p.to_path_buf()),
                    expect_dir,
                    "failed for {expect_dir:?}, {module_name}, {module_path}"
                );
                (ArcId::new(x), Vec::new())
            }))
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
        assert_eq!(config_file.fallback_search_path, Vec::<PathBuf>::new());

        // we should get a synthetic config with a search path = project_root/..
        let config_file = finder(
            Some(root),
            ModuleName::from_str("foo.bar"),
            ModulePath::filesystem(root.join("no_config/foo/bar.py")),
        );
        assert_eq!(config_file.source, ConfigSource::Synthetic);
        assert_eq!(
            config_file.search_path().cloned().collect::<Vec<_>>(),
            vec![root.join("no_config")]
        );
        assert_eq!(config_file.fallback_search_path, Vec::<PathBuf>::new());

        // check invalid module path parent
        assert_eq!(
            finder(
                None,
                ModuleName::from_str("foo.bar"),
                ModulePath::filesystem(PathBuf::new())
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
            BundledTypeshed::config(),
        );

        // check namespace
        let config_file = finder(
            Some(&root.join("no_config")),
            ModuleName::unknown(),
            ModulePath::namespace(root.join("no_config/foo")),
        );
        assert_eq!(config_file.source, ConfigSource::Synthetic);
        assert_eq!(config_file.search_path_from_file, Vec::<PathBuf>::new());
        assert_eq!(
            config_file.fallback_search_path,
            [root.join("no_config/foo"), root.join("no_config")]
                .into_iter()
                .chain(root.ancestors().map(PathBuf::from))
                .collect::<Vec<PathBuf>>(),
        );

        // check filesystem/memory
        let config_file = finder(
            Some(&root.join("no_config")),
            ModuleName::unknown(),
            ModulePath::filesystem(root.join("no_config/foo/bar.py")),
        );
        assert_eq!(config_file.source, ConfigSource::Synthetic);
        assert_eq!(config_file.search_path_from_file, Vec::<PathBuf>::new());
        assert_eq!(
            config_file.fallback_search_path,
            [root.join("no_config/foo"), root.join("no_config")]
                .into_iter()
                .chain(root.ancestors().map(PathBuf::from))
                .collect::<Vec<PathBuf>>(),
        );
    }
}
