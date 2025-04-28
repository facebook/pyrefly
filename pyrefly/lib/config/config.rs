/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::Context;
use anyhow::anyhow;
use dupe::Dupe;
use itertools::Itertools;
use path_absolutize::Absolutize;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use tracing::debug;
use tracing::warn;

use crate::config::base::ConfigBase;
use crate::config::environment::PythonEnvironment;
use crate::config::error::ErrorDisplayConfig;
use crate::module::bundled::typeshed;
use crate::module::finder::find_module_in_search_path;
use crate::module::finder::find_module_in_site_package_path;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::wildcard::ModuleWildcard;
use crate::state::loader::FindError;
use crate::sys_info::PythonPlatform;
use crate::sys_info::PythonVersion;
use crate::sys_info::SysInfo;
use crate::util::arc_id::ArcId;
use crate::util::fs_anyhow;
use crate::util::globs::Glob;
use crate::util::globs::Globs;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone)]
pub struct SubConfig {
    matches: Glob,
    #[serde(flatten)]
    settings: ConfigBase,
}

impl SubConfig {
    fn rewrite_with_path_to_config(&mut self, config_root: &Path) {
        self.matches = self.matches.clone().from_root(config_root);
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone)]
pub struct ConfigFile {
    /// Files that should be counted as sources (e.g. user-space code).
    /// NOTE: unlike other args, this is never replaced with CLI arg overrides
    /// in this config, but may be overridden by CLI args where used.
    #[serde(default = "ConfigFile::default_project_includes")]
    pub project_includes: Globs,

    /// Files that should be excluded as sources (e.g. user-space code). These take
    /// precedence over `project_includes`.
    /// NOTE: unlike other configs, this is never replaced with CLI arg overrides
    /// in this config, but may be overridden by CLI args where used.
    #[serde(default = "ConfigFile::default_project_excludes")]
    pub project_excludes: Globs,

    /// The list of directories where imports are
    /// imported from, including type checked files.
    #[serde(default = "ConfigFile::default_search_path")]
    pub search_path: Vec<PathBuf>,

    // TODO(connernilsen): make this mutually exclusive with venv/conda env
    /// The python executable that will be queried for `python_version`,
    /// `python_platform`, or `site_package_path` if any of the values are missing.
    #[serde(default = "PythonEnvironment::get_default_interpreter")]
    pub python_interpreter: Option<PathBuf>,

    /// Values representing the environment of the Python interpreter
    /// (which platform, Python version, ...). When we parse, these values
    /// are default to false so we know to query the `python_interpreter` before falling
    /// back to Pyrefly's defaults.
    #[serde(flatten)]
    pub python_environment: PythonEnvironment,

    /// The `ConfigBase` values for the whole project.
    #[serde(default, flatten)]
    pub root: ConfigBase,

    /// Sub-configs that can override specific `ConfigBase` settings
    /// based on path matching.
    #[serde(default)]
    pub sub_configs: Vec<SubConfig>,

    /// Skips any `py.typed` checks we do when resolving `site_package_path` imports.
    #[serde(default)]
    pub use_untyped_imports: bool,

    /// Completely custom module to path mappings. Currently not exposed to the user.
    #[serde(skip)]
    pub custom_module_paths: SmallMap<ModuleName, ModulePath>,

    /// Skips the check to ensure any `-stubs` `site_package_path` entries have an
    /// installed non-stubs package.
    #[serde(default)]
    pub ignore_missing_source: bool,
}

impl Default for ConfigFile {
    /// Gets a default ConfigFile, with all paths rewritten relative to cwd.
    fn default() -> ConfigFile {
        let mut result = Self::default_no_path_rewrite();
        match std::env::current_dir() {
            Ok(cwd) => {
                result.rewrite_with_path_to_config(&cwd);
            }
            Err(err) => {
                debug!(
                    "Cannot identify current dir for default config path rewriting: {}",
                    err
                );
            }
        }
        result
    }
}

impl ConfigFile {
    /// Gets a default ConfigFile, with no path rewriting. This should only be used for unit testing,
    /// since it may have strange runtime behavior. Prefer to use `ConfigFile::default()` instead.
    fn default_no_path_rewrite() -> Self {
        ConfigFile {
            project_includes: Self::default_project_includes(),
            project_excludes: Self::default_project_excludes(),
            python_interpreter: PythonEnvironment::get_default_interpreter(),
            search_path: Self::default_search_path(),
            python_environment: PythonEnvironment {
                python_platform: None,
                python_version: None,
                site_package_path: None,
                site_package_path_from_interpreter: false,
            },
            root: Default::default(),
            sub_configs: Default::default(),
            custom_module_paths: Default::default(),
            use_untyped_imports: false,
            ignore_missing_source: false,
        }
    }

    /// Return `Err` to indicate the module could not be found.
    pub fn find_import(&self, module: ModuleName) -> Result<ModulePath, FindError> {
        if let Some(path) = self.custom_module_paths.get(&module) {
            Ok(path.clone())
        } else if self
            .replace_imports_with_any()
            .iter()
            .any(|p| p.matches(module))
        {
            Err(FindError::Ignored)
        } else if let Some(path) = find_module_in_search_path(module, &self.search_path) {
            Ok(path)
        } else if let Some(path) = typeshed()
            .map_err(|err| FindError::not_found(err, module))?
            .find(module)
        {
            Ok(path)
        } else if let Some(path) = find_module_in_site_package_path(
            module,
            self.site_package_path(),
            self.use_untyped_imports,
            self.ignore_missing_source,
        )? {
            Ok(path)
        } else {
            Err(FindError::search_path(
                &self.search_path,
                self.site_package_path(),
                module,
            ))
        }
    }
}

impl ConfigFile {
    pub const CONFIG_FILE_NAME: &str = "pyrefly.toml";
    pub const PYPROJECT_FILE_NAME: &str = "pyproject.toml";
    pub const CONFIG_FILE_NAMES: &[&str] = &[Self::CONFIG_FILE_NAME, Self::PYPROJECT_FILE_NAME];

    /// An empty `ConfigFile` with no search path at all. Always returns the same `ArcId`.
    #[allow(clippy::field_reassign_with_default)] // Default doesn't do what a normal Default does
    pub fn empty() -> ArcId<Self> {
        static EMPTY: LazyLock<ArcId<ConfigFile>> = LazyLock::new(|| {
            let mut config = ConfigFile::default();
            config.search_path = Vec::new();
            config.python_environment.site_package_path = Some(Vec::new());
            config.configure();
            ArcId::new(config)
        });
        EMPTY.dupe()
    }

    pub fn default_project_includes() -> Globs {
        Globs::new(vec!["**/*".to_owned()])
    }

    pub fn default_project_excludes() -> Globs {
        Globs::new(vec![
            "**/__pycache__/**".to_owned(),
            // match any hidden file, but don't match `.` or `..` (equivalent to regex: `\.[^/\.]{0,1}.*`)
            "**/.[!/.]*".to_owned(),
        ])
    }

    pub fn default_search_path() -> Vec<PathBuf> {
        vec![PathBuf::from("")]
    }

    pub fn python_version(&self) -> PythonVersion {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.python_environment.python_version.unwrap()
    }

    pub fn python_platform(&self) -> &PythonPlatform {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.python_environment.python_platform.as_ref().unwrap()
    }

    pub fn site_package_path(&self) -> &[PathBuf] {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.python_environment
            .site_package_path
            .as_deref()
            .unwrap()
    }

    pub fn get_sys_info(&self) -> SysInfo {
        SysInfo::new(self.python_version(), self.python_platform().clone())
    }

    pub fn errors(&self) -> &ErrorDisplayConfig {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.root.errors.as_ref().unwrap()
    }

    pub fn replace_imports_with_any(&self) -> &[ModuleWildcard] {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.root.replace_imports_with_any.as_deref().unwrap()
    }

    pub fn skip_untyped_functions(&self) -> bool {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.root.skip_untyped_functions.unwrap()
    }

    pub fn ignore_errors_in_generated_code(&self) -> bool {
        // we can use unwrap here, because the value in the root config must
        // be set in `ConfigFile::configure()`.
        self.root.ignore_errors_in_generated_code.unwrap()
    }

    #[expect(dead_code)]
    /// Filter to sub configs whose matches succeed for the given `path`,
    /// then return the first non-None value the getter returns, or None
    /// if a non-empty value can't be found.
    fn get_from_sub_configs<'a, T>(
        &'a self,
        getter: impl Fn(&'a ConfigBase) -> Option<T>,
        path: &Path,
    ) -> Option<T> {
        self.sub_configs.iter().find_map(|c| {
            if c.matches.matches(path).ok()? {
                return getter(&c.settings);
            }
            None
        })
    }

    /// Configures values that must be updated *after* overwriting with CLI flag values,
    /// which should probably be everything except for `PathBuf` or `Globs` types.
    pub fn configure(&mut self) {
        if self.python_environment.any_empty() {
            if let Some(interpreter) = &self.python_interpreter {
                let system_env = PythonEnvironment::get_interpreter_env(interpreter);
                self.python_environment.override_empty(system_env);
            } else {
                self.python_environment.set_empty_to_default();
                warn!(
                    "Python environment (version, platform, or site_package_path) has value unset, \
                but no Python interpreter could be found to query for values. Falling back to \
                Pyrefly defaults for missing values."
                )
            }
        }

        if self.root.errors.is_none() {
            self.root.errors = Some(Default::default());
        }

        if self.root.replace_imports_with_any.is_none() {
            self.root.replace_imports_with_any = Some(Default::default());
        }

        if self.root.skip_untyped_functions.is_none() {
            self.root.skip_untyped_functions = Some(Default::default());
        }

        if self.root.ignore_errors_in_generated_code.is_none() {
            self.root.ignore_errors_in_generated_code = Some(Default::default());
        }
    }

    /// Rewrites any config values that must be updated *before* applying CLI flag values, namely
    /// rewriting any `PathBuf`s and `Globs` to be relative to `config_root`.
    /// We do this as a step separate from `configure()` because CLI args may override some of these
    /// values, but CLI args will always be relative to CWD, whereas config values should be relative
    /// to the config root.
    fn rewrite_with_path_to_config(&mut self, config_root: &Path) {
        // TODO(connernilsen): store root as part of config to make it easier to rewrite later on
        self.project_includes = self.project_includes.clone().from_root(config_root);
        self.search_path.iter_mut().for_each(|search_root| {
            let mut base = config_root.to_path_buf();
            base.push(search_root.as_path());
            *search_root = base;
        });
        // push config to search path to make sure we can fall back to the config directory as an import path
        // if users forget to add it
        self.search_path.push(config_root.to_path_buf());
        self.python_environment
            .site_package_path
            .iter_mut()
            .for_each(|v| {
                v.iter_mut().for_each(|site_package_path| {
                    let mut with_base = config_root.to_path_buf();
                    with_base.push(site_package_path.as_path());
                    *site_package_path = with_base;
                });
            });
        self.project_excludes = self.project_excludes.clone().from_root(config_root);
        self.python_interpreter = self
            .python_interpreter
            .as_ref()
            .map(|i| config_root.join(i));
        self.sub_configs
            .iter_mut()
            .for_each(|c| c.rewrite_with_path_to_config(config_root));
    }

    pub fn validate(&self) {
        fn warn_on_invalid(p: &Path, field: &str) {
            match p.try_exists() {
                Ok(true) => return,
                Err(err) => {
                    debug!(
                        "Error checking for existence of path {}: {}",
                        p.display(),
                        err
                    );
                    return;
                }
                _ => (),
            }
            let p = if p == Path::new("") {
                Path::new("./")
            } else {
                p
            };
            warn!("Nonexistent `{field}` found: {}", p.display());
        }
        if !self.python_environment.site_package_path_from_interpreter {
            self.python_environment
                .site_package_path
                .as_ref()
                .inspect(|p| {
                    p.iter()
                        .for_each(|p| warn_on_invalid(p, "site_package_path"))
                });
        }
        self.search_path
            .iter()
            .for_each(|p| warn_on_invalid(p, "search_path"));
    }

    pub fn from_file(config_path: &Path, error_on_extras: bool) -> anyhow::Result<ConfigFile> {
        fn f(config_path: &Path, error_on_extras: bool) -> anyhow::Result<ConfigFile> {
            let config_path = config_path
                .absolutize()
                .with_context(|| format!("Path `{}` cannot be absolutized", config_path.display()))?
                .into_owned();
            let config_str = fs_anyhow::read_to_string(&config_path)?;
            let mut config =
                if config_path.file_name() == Some(OsStr::new(ConfigFile::PYPROJECT_FILE_NAME)) {
                    ConfigFile::parse_pyproject_toml(&config_str)
                } else {
                    ConfigFile::parse_config(&config_str)
                }?;

            if error_on_extras {
                if !config.root.extras.0.is_empty() {
                    let extra_keys = config.root.extras.0.keys().join(", ");
                    return Err(anyhow!("Extra keys found in config: {extra_keys}"));
                }
                for sub_config in &config.sub_configs {
                    if !sub_config.settings.extras.0.is_empty() {
                        let extra_keys = config.root.extras.0.keys().join(", ");
                        return Err(anyhow!(
                            "Extra keys found in sub config matching {}: {extra_keys}",
                            sub_config.matches
                        ));
                    }
                }
            }

            if let Some(config_root) = config_path.parent() {
                config.rewrite_with_path_to_config(config_root);
            }

            Ok(config)
        }
        f(config_path, error_on_extras).map_err(|err| {
            let file_str = config_path.display();
            anyhow!("Failed to parse configuration at {file_str}: {err}")
        })
    }

    fn parse_config(config_str: &str) -> anyhow::Result<ConfigFile> {
        toml::from_str::<ConfigFile>(config_str).map_err(|err| anyhow::Error::msg(err.to_string()))
    }

    fn parse_pyproject_toml(config_str: &str) -> anyhow::Result<ConfigFile> {
        #[derive(Debug, Deserialize)]
        struct PyProject {
            #[serde(default)]
            pub tool: Option<Tool>,
        }

        #[derive(Debug, Deserialize)]
        struct Tool {
            #[serde(default)]
            pub pyrefly: Option<ConfigFile>,
        }

        let maybe_config = toml::from_str::<PyProject>(config_str)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?
            .tool
            .and_then(|c| c.pyrefly);
        Ok(maybe_config.unwrap_or_else(ConfigFile::default))
    }
}

impl Display for ConfigFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{project_includes: {}, project_excludes: {}, search_path: [{}], python_interpreter: {:?}, python_environment: {}, replace_imports_with_any: [{}]}}",
            self.project_includes,
            self.project_excludes,
            self.search_path.iter().map(|p| p.display()).join(", "),
            self.python_interpreter,
            self.python_environment,
            self.root
                .replace_imports_with_any
                .as_ref()
                .map(|r| { r.iter().map(|p| p.as_str()).join(", ") })
                .unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path;

    use toml::Table;
    use toml::Value;

    use super::*;
    use crate::error::kind::ErrorKind;

    #[test]
    fn deserialize_pyrefly_config() {
        let config_str = r#"
            project_includes = ["tests", "./implementation"]
            project_excludes = ["tests/untyped/**"]
            skip_untyped_functions = false
            search_path = ["../.."]
            python_platform = "darwin"
            python_version = "1.2.3"
            site_package_path = ["venv/lib/python1.2.3/site-packages"]
            python_interpreter = "venv/my/python"
            replace_imports_with_any = ["fibonacci"]
            ignore_errors_in_generated_code = true
            use_untyped_imports = true
            ignore_missing_source = true

            [errors]
            assert-type = true
            bad-return = false

            [[sub_configs]]
            matches = "sub/project/**"

            skip_untyped_functions = true
            replace_imports_with_any = []
            ignore_errors_in_generated_code = false
            [sub_configs.errors]
            assert-type = false
            invalid-yield = false
        "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "tests".to_owned(),
                    "./implementation".to_owned()
                ]),
                project_excludes: Globs::new(vec!["tests/untyped/**".to_owned()]),
                search_path: vec![PathBuf::from("../..")],
                python_environment: PythonEnvironment::new(
                    PythonPlatform::mac(),
                    PythonVersion::new(1, 2, 3),
                    vec![PathBuf::from("venv/lib/python1.2.3/site-packages")],
                ),
                python_interpreter: Some(PathBuf::from("venv/my/python")),
                root: ConfigBase {
                    extras: Default::default(),
                    errors: Some(ErrorDisplayConfig::new(HashMap::from_iter([
                        (ErrorKind::AssertType, true),
                        (ErrorKind::BadReturn, false)
                    ]))),
                    ignore_errors_in_generated_code: Some(true),
                    replace_imports_with_any: Some(vec![ModuleWildcard::new("fibonacci").unwrap()]),
                    skip_untyped_functions: Some(false),
                },
                custom_module_paths: Default::default(),
                sub_configs: vec![SubConfig {
                    matches: Glob::new("sub/project/**".to_owned()),
                    settings: ConfigBase {
                        extras: Default::default(),
                        errors: Some(ErrorDisplayConfig::new(HashMap::from_iter([
                            (ErrorKind::AssertType, false),
                            (ErrorKind::InvalidYield, false)
                        ]))),
                        ignore_errors_in_generated_code: Some(false),
                        replace_imports_with_any: Some(Vec::new()),
                        skip_untyped_functions: Some(true),
                    }
                }],
                use_untyped_imports: true,
                ignore_missing_source: true,
            }
        );
    }

    #[test]
    fn deserialize_pyrefly_config_defaults() {
        let config_str = "";
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(config, ConfigFile::default_no_path_rewrite());
    }

    #[test]
    fn deserialize_pyrefly_config_with_unknown() {
        let config_str = r#"
            laszewo = "good kids"
            python_platform = "windows"

            [[sub_configs]]
            matches = "abcd"
            
            atliens = 1
        "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config.root.extras.0,
            Table::from_iter([("laszewo".to_owned(), Value::String("good kids".to_owned())),])
        );
        assert_eq!(
            config.sub_configs[0].settings.extras.0,
            Table::from_iter([("atliens".to_owned(), Value::Integer(1))])
        );
    }

    #[test]
    fn deserialize_pyproject_toml() {
        let config_str = r#"
            [tool.pyrefly]
            project_includes = ["./tests", "./implementation"]
            python_platform = "darwin"
            python_version = "1.2.3"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "./tests".to_owned(),
                    "./implementation".to_owned()
                ]),
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::mac()),
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    site_package_path: None,
                    site_package_path_from_interpreter: false,
                },
                ..ConfigFile::default_no_path_rewrite()
            }
        );
    }

    #[test]
    fn deserialize_pyproject_toml_defaults() {
        let config_str = "";
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(config, ConfigFile::default());
    }

    #[test]
    fn deserialize_pyproject_toml_with_unknown() {
        let config_str = r#"
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
            [tool.pyrefly]
            python_version = "1.2.3"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                python_environment: PythonEnvironment {
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    python_platform: None,
                    site_package_path: None,
                    // this won't be set until after `configure()`
                    site_package_path_from_interpreter: false,
                },
                ..ConfigFile::default_no_path_rewrite()
            }
        );
    }

    #[test]
    fn deserialize_pyproject_toml_without_pyrefly() {
        let config_str = "
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
        ";
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(config, ConfigFile::default());
    }

    #[test]
    fn deserialize_pyproject_toml_with_unknown_in_pyrefly() {
        let config_str = r#"
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
            [tool.pyrefly]
            python_version = "1.2.3"
            inzo = "overthinker"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(
            config.root.extras.0,
            Table::from_iter([("inzo".to_owned(), Value::String("overthinker".to_owned()))])
        );
    }

    #[test]
    fn test_rewrite_with_path_to_config() {
        fn with_sep(s: &str) -> String {
            s.replace("/", path::MAIN_SEPARATOR_STR)
        }
        let mut python_environment = PythonEnvironment {
            site_package_path: Some(vec![PathBuf::from("venv/lib/python1.2.3/site-packages")]),
            ..PythonEnvironment::default()
        };
        let interpreter = "venv/bin/python3".to_owned();
        let mut config = ConfigFile {
            project_includes: Globs::new(vec!["path1/**".to_owned(), "path2/path3".to_owned()]),
            project_excludes: Globs::new(vec!["tests/untyped/**".to_owned()]),
            search_path: vec![PathBuf::from("../..")],
            python_environment: python_environment.clone(),
            python_interpreter: Some(PathBuf::from(interpreter.clone())),
            root: Default::default(),
            custom_module_paths: Default::default(),
            sub_configs: vec![SubConfig {
                matches: Glob::new("sub/project/**".to_owned()),
                settings: Default::default(),
            }],
            use_untyped_imports: false,
            ignore_missing_source: false,
        };

        let path_str = with_sep("path/to/my/config");
        let test_path = PathBuf::from(path_str.clone());

        let project_includes_vec = vec![
            path_str.clone() + &with_sep("/path1/**"),
            path_str.clone() + &with_sep("/path2/path3"),
        ];
        let project_excludes_vec = vec![path_str.clone() + &with_sep("/tests/untyped/**")];
        let search_path = vec![test_path.join("../.."), test_path.clone()];
        python_environment.site_package_path =
            Some(vec![test_path.join("venv/lib/python1.2.3/site-packages")]);

        let sub_config_matches = Glob::new(path_str.clone() + &with_sep("/sub/project/**"));

        config.rewrite_with_path_to_config(&test_path);

        let expected_config = ConfigFile {
            project_includes: Globs::new(project_includes_vec),
            project_excludes: Globs::new(project_excludes_vec),
            python_interpreter: Some(test_path.join(interpreter)),
            search_path,
            python_environment,
            root: Default::default(),
            custom_module_paths: Default::default(),
            sub_configs: vec![SubConfig {
                matches: sub_config_matches,
                settings: Default::default(),
            }],
            use_untyped_imports: false,
            ignore_missing_source: false,
        };
        assert_eq!(config, expected_config);
    }

    #[test]
    fn test_deserializing_unknown_error_errors() {
        let config_str = "
            [errors]
            subtronics = true
            zeds_dead = false
            GRiZ = true
        ";
        let err = ConfigFile::parse_config(config_str).unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
    }

    #[test]
    fn test_deserializing_sub_config_missing_matches() {
        let config_str = r#"
            [[sub_configs]]
            search_path = ["../../.."]
        "#;
        let err = ConfigFile::parse_config(config_str).unwrap_err();
        assert!(err.to_string().contains("missing field `matches`"));
    }

    #[test]
    fn test_expect_all_fields_set_in_root_config() {
        let mut config = ConfigFile::default_no_path_rewrite();
        config.configure();

        let table: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();

        let ignore_keys: Vec<String> = vec![
            // top level configs, where null values (if possible), should be allowed
            "project_includes",
            "project_excludes",
            "python_interpreter",
            // values we won't be geting
            "extras",
            // values that must be Some (if flattend, their contents will be checked)
            "python_environment",
        ]
        .into_iter()
        .map(|k| k.to_owned())
        .collect();

        table.keys().for_each(|k| {
            if ignore_keys.contains(k) {
                return;
            }

            assert!(
                table.get(k).is_some_and(|v| !v.is_null()),
                "Value for {k} is None after ConfigFile::configure()"
            );
        });
    }

    #[test]
    fn test_site_package_path_default_after_configure() {
        let mut config = ConfigFile::parse_config("").unwrap();
        assert!(!config.python_environment.site_package_path_from_interpreter);

        if config.python_interpreter.is_none() {
            // we don't really need to test anything else here if the interpreter isn't
            // available, since the rest of the test only makes sense with it present
            return;
        }

        config.configure();
        assert!(config.python_environment.site_package_path_from_interpreter);

        config = ConfigFile::parse_config("site_package_path = []").unwrap();
        assert!(!config.python_environment.site_package_path_from_interpreter);
        config.configure();
        assert!(!config.python_environment.site_package_path_from_interpreter);
    }
}
