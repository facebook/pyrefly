/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context as _;
use dupe::Dupe as _;
use itertools::Itertools as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathBuf;
use pyrefly_python::sys_info::SysInfo;
use serde::Deserialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tempfile::NamedTempFile;
use vec1::Vec1;
#[allow(unused_imports)]
use vec1::vec1;

use crate::source_db::Target;

pub mod buck;
pub mod custom;

/// An enum representing something that has been included by the build system, and
/// which the build system should query for when building the sourcedb.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) enum Include {
    #[allow(unused)]
    Target(Target),
    Path(ModulePathBuf),
}

impl Include {
    pub fn path(path: ModulePathBuf) -> Self {
        Self::Path(path)
    }

    fn to_cli_arg(&self) -> impl Iterator<Item = &OsStr> {
        match self {
            Include::Target(target) => [OsStr::new("--target"), target.to_os_str()].into_iter(),
            Include::Path(path) => [OsStr::new("--file"), path.as_os_str()].into_iter(),
        }
    }
}

pub trait SourceDbQuerier: Send + Sync + fmt::Debug {
    fn query_source_db(
        &self,
        files: &SmallSet<Include>,
        cwd: &Path,
    ) -> anyhow::Result<TargetManifestDatabase> {
        if files.is_empty() {
            return Ok(TargetManifestDatabase {
                db: SmallMap::new(),
                root: cwd.to_path_buf(),
            });
        }

        let mut argfile =
            NamedTempFile::with_prefix("pyrefly_build_query_").with_context(|| {
                "Failed to create temporary argfile for querying source DB".to_owned()
            })?;
        let mut argfile_args = OsString::from("--");
        files.iter().flat_map(Include::to_cli_arg).for_each(|arg| {
            argfile_args.push("\n");
            argfile_args.push(arg);
        });

        argfile
            .as_file_mut()
            .write_all(argfile_args.as_encoded_bytes())
            .with_context(|| "Could not write to argfile when querying source DB".to_owned())?;

        let mut cmd = self.construct_command();
        cmd.arg(format!("@{}", argfile.path().display()));
        cmd.current_dir(cwd);

        let result = cmd.output()?;
        if !result.status.success() {
            let stdout = String::from_utf8(result.stdout)
                .unwrap_or_else(|_| "<Failed to parse stdout from source DB query>".to_owned());
            let stderr = String::from_utf8(result.stderr).unwrap_or_else(|_| {
                "<Failed to parse stderr from Buck source DB query>".to_owned()
            });

            return Err(anyhow::anyhow!(
                "Source DB query failed...\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            ));
        }

        match serde_json::from_slice(&result.stdout)
                .with_context(|| {
                    format!(
                        "Failed to construct valid `TargetManifestDatabase` from querier result. Command run: {} {}",
                        cmd.get_program().display(),
                        cmd.get_args().map(|a| a.to_string_lossy()).join(" "),
                    )
                }) {
            Err(e) => {
                let Some(downcast) = e.downcast_ref::<serde_json::error::Error>() else {
                    return Err(e);
                };
                let Ok(content) = String::from_utf8(result.stdout) else {
                    return Err(e);
                };
                let lines = content.lines().collect::<Vec<_>>();
                let error_line = downcast.line();
                let start = std::cmp::max(0, error_line - 30);
                let end = std::cmp::min(lines.len() - 1, error_line + 20);
                let cont = std::cmp::min(error_line + 1, end);

                let e = e.context(
                    format!(
                        "Context: ```\n{} # THIS LINE HAS A PROBLEM\n{}\n```",
                        lines[start..=error_line].iter().join("\n"),
                        lines[cont..=end].iter().join("\n"),
                    )
                );

                Err(e)
            },
            ok => ok,
        }
    }

    fn construct_command(&self) -> Command;
}

#[derive(Debug, PartialEq, Eq, Deserialize, Clone)]
pub(crate) struct PythonLibraryManifest {
    pub deps: SmallSet<Target>,
    pub srcs: SmallMap<ModuleName, Vec1<ModulePathBuf>>,
    #[serde(default)]
    pub relative_to: Option<PathBuf>,
    #[serde(flatten)]
    pub sys_info: SysInfo,
    pub buildfile_path: PathBuf,
    #[serde(default, skip)]
    pub packages: SmallMap<ModuleName, Vec1<ModulePathBuf>>,
}

impl PythonLibraryManifest {
    fn replace_alias_deps(&mut self, aliases: &SmallMap<Target, Target>) {
        self.deps = self
            .deps
            .iter()
            .map(|t| {
                if let Some(replace) = aliases.get(t) {
                    replace.dupe()
                } else {
                    t.dupe()
                }
            })
            .collect();
    }

    fn rewrite_relative_to_root(&mut self, root: &Path) {
        if let Some(relative_to) = &mut self.relative_to {
            *relative_to = root.join(&relative_to);
        }
        let relative_to = self.relative_to.as_deref().unwrap_or(root);
        self.srcs.iter_mut().for_each(|(_, paths)| {
            paths.iter_mut().for_each(|p| {
                let resolved = relative_to.join(&**p);
                // canonicalize it, since VSCode will do it for us anyway and
                // then path lookups won't work
                // TODO(connernilsen): do we need to store canonicalized files and
                // buck-provided files?
                let file = resolved.canonicalize().unwrap_or(resolved);
                *p = ModulePathBuf::new(file)
            })
        });
        self.packages.iter_mut().for_each(|(_, paths)| {
            paths.iter_mut().for_each(|p| {
                let resolved = relative_to.join(&**p);
                // canonicalize it, since VSCode will do it for us anyway and
                // then path lookups won't work
                // TODO(connernilsen): do we need to store canonicalized files and
                // buck-provided files?
                let file = resolved.canonicalize().unwrap_or(resolved);
                *p = ModulePathBuf::new(file)
            })
        });
        self.buildfile_path = root.join(&self.buildfile_path);
    }

    /// Synthesize packages for all source modules.
    ///
    /// Uses a simple algorithm:
    /// 1. For each source that is an explicit __init__.py/pyi, add it as a package
    /// 2. For each source module, walk up the module hierarchy and create namespace
    ///    packages for each parent
    ///
    /// This matches the behavior in `BuckCheckSourceDatabase` and follows Buck's
    /// convention that any module parent is implicitly an empty `__init__.py` file.
    ///
    /// See: <https://github.com/facebook/buck2/blob/main/prelude/python/tools/wheel.py>
    fn fill_implicit_packages(&mut self) {
        let mut packages: SmallMap<ModuleName, Vec1<ModulePathBuf>> = SmallMap::new();

        for (module_name, paths) in &self.srcs {
            let first_path = paths.first();
            let path_ref = first_path.as_path();
            let is_init = path_ref.file_stem() == Some("__init__".as_ref());

            // If this is an explicit init file (__init__.py or __init__.pyi), add ALL paths
            // (there may be both .py and .pyi variants)
            if is_init {
                packages.insert(*module_name, paths.clone());
            }

            // Walk up and synthesize parent packages (as directories)
            let mut name = *module_name;
            let mut path = path_ref.to_path_buf();

            // For init files, pop once first since the __init__.py file corresponds
            // to the directory that contains it (e.g., `foo/bar/__init__.py` -> `foo/bar`)
            if is_init {
                path.pop();
            }

            while let Some(parent) = name.parent() {
                path.pop();
                packages
                    .entry(parent)
                    .or_insert_with(|| vec1![ModulePathBuf::from_path(&path)]);
                name = parent;
            }
        }

        self.packages = packages;
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize, Clone)]
#[serde(untagged)]
pub(crate) enum TargetManifest {
    Library(PythonLibraryManifest),
    Alias { alias: Target },
}

#[derive(Debug, PartialEq, Eq, Deserialize, Clone)]
pub(crate) struct TargetManifestDatabase {
    db: SmallMap<Target, TargetManifest>,
    pub root: PathBuf,
}

impl TargetManifestDatabase {
    pub fn produce_map(mut self) -> SmallMap<Target, PythonLibraryManifest> {
        let mut result = SmallMap::new();
        let aliases: SmallMap<Target, Target> = self
            .db
            .iter()
            .filter_map(|(t, manifest)| match manifest {
                TargetManifest::Alias { alias } => Some((t.dupe(), alias.dupe())),
                _ => None,
            })
            .collect();

        for manifest in self.db.values_mut() {
            match manifest {
                TargetManifest::Alias { .. } => continue,
                TargetManifest::Library(lib) => {
                    lib.replace_alias_deps(&aliases);
                    lib.rewrite_relative_to_root(&self.root);
                }
            }
        }

        for (target, manifest) in self.db {
            match manifest {
                TargetManifest::Alias { .. } => continue,
                TargetManifest::Library(mut lib) => {
                    lib.fill_implicit_packages();
                    result.insert(target, lib);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;
    use starlark_map::smallmap;

    use super::*;

    impl TargetManifestDatabase {
        pub fn new(db: SmallMap<Target, TargetManifest>, root: PathBuf) -> Self {
            TargetManifestDatabase { db, root }
        }

        /// This is a simplified sourcedb taken from the BXL output run on pyre/client/log/log.py.
        /// We also add a few extra entries to model some of the behavior around multiple entries
        /// (i.e. multiple file paths corresponding to a module path, multiple module paths in
        /// different targets).
        pub fn get_test_database() -> Self {
            TargetManifestDatabase::new(
                smallmap! {
                    Target::from_string("//colorama:py-stubs".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "colorama",
                            &[
                            "colorama/__init__.pyi",
                            ],
                        ),
                        ],
                        &[],
                        "colorama/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//colorama:py".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "colorama",
                            &[
                            "colorama/__init__.py",
                            ]
                        ),
                        ],
                        &["//colorama:py-stubs"],
                        "colorama/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//colorama:colorama".to_owned()) => TargetManifest::alias(
                        "//colorama:py"
                    ),
                    Target::from_string("//click:py".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "click",
                            &[
                            "click/__init__.pyi",
                            "click/__init__.py",
                            ],
                        )
                        ],
                        &[
                        "//colorama:colorama"
                        ],
                        "click/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//click:click".to_owned()) => TargetManifest::alias(
                        "//click:py"
                    ),
                    Target::from_string("//pyre/client/log:log".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "pyre.client.log",
                            &[
                            "pyre/client/log/__init__.py"
                            ]
                        ),
                        (
                            "pyre.client.log.log",
                            &[
                            "pyre/client/log/log.py",
                            "pyre/client/log/log.pyi",
                            ],
                        ),
                        (
                            "pyre.client.log.format",
                            &[
                            "pyre/client/log/format.py",
                            ],
                        ),
                        ],
                        &[
                        "//click:click"
                        ],
                        "pyre/client/log/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//pyre/client/log:log2".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "log",
                            &[
                            "pyre/client/log/__init__.py"
                            ]
                        ),
                        (
                            "log.log",
                            &[
                            "pyre/client/log/log.py",
                            "pyre/client/log/log.pyi",
                            ]
                        ),
                        (
                            "log.format",
                            &[
                            "pyre/client/log/format.py",
                            ]
                        ),
                        ],
                        &[
                        "//click:click"
                        ],
                        "pyre/client/log/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//implicit_package/test:main".to_owned()) => TargetManifest::lib(
                            &[
                            (
                                "implicit_package.main",
                                &[
                                "implicit_package/test/main.py",
                                ],
                            ),
                            ],
                            &[
                            "//implicit_package/test:lib",
                            ],
                            "implicit_package/test/BUCK",
                            &[],
                            None,
                    ),
                    Target::from_string("//implicit_package/test:lib".to_owned()) => TargetManifest::lib(
                            &[
                            (
                                "implicit_package.lib.utils",
                                &[
                                "implicit_package/test/lib/utils.py",
                                ],
                            ),
                            (
                                "implicit_package.package_boundary_violation",
                                &[
                                "implicit_package/package_boundary_violation.py",
                                ]
                            ),
                            (
                                "implicit_package.deeply.nested.package.file",
                                &[
                                "implicit_package/test/deeply/nested/package/file.py",
                                ],
                            ),
                            ],
                            &["//external:package"],
                            "implicit_package/test/BUCK",
                            &[],
                            None,
                    ),
                    Target::from_string("//external:package".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "external_package.main",
                            &[
                            "/path/to/another/repository/package/external_package/main.py",
                            ]
                        ),
                        (
                            "external_package.non_python_file",
                            &[
                            "/path/to/another/repository/package/external_package/non_python_file.thrift",
                            ],
                        ),
                        ],
                        &[],
                        "/path/to/another/repository/package/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//generated:main".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "generated.main",
                            &[
                            "generated/main.py"
                            ]
                        ),
                        ],
                        &["//generated:lib"],
                        "generated/BUCK",
                        &[],
                        None,
                    ),
                    Target::from_string("//generated:lib".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "generated",
                            &[
                            "generated/__init__.py"
                            ]
                        ),
                        ],
                        &[],
                        "generated/BUCK",
                        &[],
                        Some("build-out/materialized"),
                    ),
                },
                PathBuf::from("/path/to/this/repository"),
            )
        }
    }

    fn map_srcs(
        srcs: &[(&str, &[&str])],
        prefix_paths: Option<&str>,
    ) -> SmallMap<ModuleName, Vec1<ModulePathBuf>> {
        let prefix = prefix_paths.map(Path::new);
        let map_path = |p| {
            prefix.map_or_else(
                || ModulePathBuf::from_path(Path::new(p)),
                |prefix| ModulePathBuf::new(prefix.join(p)),
            )
        };
        srcs.iter()
            .map(|(n, paths)| {
                (
                    ModuleName::from_str(n),
                    Vec1::try_from_vec(paths.iter().map(map_path).collect()).unwrap(),
                )
            })
            .collect()
    }

    fn map_deps(deps: &[&str]) -> SmallSet<Target> {
        deps.iter()
            .map(|s| Target::from_string((*s).to_owned()))
            .collect()
    }

    fn map_implicit_packages(
        inits: &[(&str, &[&str])],
        prefix_paths: Option<&str>,
    ) -> SmallMap<ModuleName, Vec1<ModulePathBuf>> {
        let prefix = prefix_paths.map(Path::new);
        let map_path = |p| {
            prefix.map_or_else(
                || ModulePathBuf::from_path(Path::new(p)),
                |prefix| ModulePathBuf::new(prefix.join(p)),
            )
        };
        inits
            .iter()
            .map(|(n, paths)| {
                (
                    ModuleName::from_str(n),
                    Vec1::try_from_vec(paths.iter().map(map_path).collect()).unwrap(),
                )
            })
            .collect()
    }

    impl TargetManifest {
        fn alias(target: &str) -> Self {
            TargetManifest::Alias {
                alias: Target::from_string(target.to_owned()),
            }
        }

        pub fn lib(
            srcs: &[(&str, &[&str])],
            deps: &[&str],
            buildfile: &str,
            implicit_packages: &[(&str, &[&str])],
            relative_to: Option<&str>,
        ) -> Self {
            TargetManifest::Library(PythonLibraryManifest {
                srcs: map_srcs(srcs, None),
                relative_to: relative_to.map(PathBuf::from),
                deps: map_deps(deps),
                sys_info: SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                buildfile_path: PathBuf::from(buildfile),
                packages: map_implicit_packages(implicit_packages, None),
            })
        }
    }

    impl PythonLibraryManifest {
        fn new(
            srcs: &[(&str, &[&str])],
            deps: &[&str],
            buildfile: &str,
            inits: &[(&str, &[&str])],
            relative_to: Option<&str>,
        ) -> Self {
            let root = "/path/to/this/repository";
            Self {
                srcs: map_srcs(srcs, Some(root)),
                relative_to: relative_to.map(PathBuf::from),
                deps: map_deps(deps),
                sys_info: SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
                buildfile_path: PathBuf::from(root).join(buildfile),
                packages: map_implicit_packages(inits, Some(root)),
            }
        }
    }

    #[test]
    fn example_json_parses() {
        const EXAMPLE_JSON: &str = r#"
{
  "db": {
    "//colorama:py-stubs": {
      "srcs": {
        "colorama": [
          "colorama/__init__.pyi"
        ]
      },
      "deps": [],
      "buildfile_path": "colorama/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//colorama:py": {
      "srcs": {
        "colorama": [
          "colorama/__init__.py"
        ]
      },
      "deps": ["//colorama:py-stubs"],
      "buildfile_path": "colorama/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//colorama:colorama": {
      "alias": "//colorama:py"
    },
    "//click:py": {
      "srcs": {
        "click": [
          "click/__init__.pyi",
          "click/__init__.py"
        ]
      },
      "deps": [
        "//colorama:colorama"
      ],
      "buildfile_path": "click/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//click:click": {
      "alias": "//click:py"
    },
    "//pyre/client/log:log": {
      "srcs": {
        "pyre.client.log": [
          "pyre/client/log/__init__.py"
        ],
        "pyre.client.log.log": [
          "pyre/client/log/log.py",
          "pyre/client/log/log.pyi"
        ],
        "pyre.client.log.format": [
          "pyre/client/log/format.py"
        ]
      },
      "deps": [
        "//click:click"
      ],
      "buildfile_path": "pyre/client/log/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//pyre/client/log:log2": {
      "srcs": {
        "log": [
          "pyre/client/log/__init__.py"
        ],
        "log.log": [
          "pyre/client/log/log.py",
          "pyre/client/log/log.pyi"
        ],
        "log.format": [
          "pyre/client/log/format.py"
        ]
      },
      "deps": [
        "//click:click"
      ],
      "buildfile_path": "pyre/client/log/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//implicit_package/test:main": {
      "srcs": {
          "implicit_package.main": [
              "implicit_package/test/main.py"
          ]
      },
      "deps": ["//implicit_package/test:lib"],
      "buildfile_path": "implicit_package/test/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//implicit_package/test:lib": {
      "srcs": {
          "implicit_package.lib.utils": [
              "implicit_package/test/lib/utils.py"
          ],
          "implicit_package.package_boundary_violation": [
              "implicit_package/package_boundary_violation.py"
          ],
          "implicit_package.deeply.nested.package.file": [
              "implicit_package/test/deeply/nested/package/file.py"
          ]
      }, 
      "deps": ["//external:package"],
      "buildfile_path": "implicit_package/test/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//external:package": {
      "srcs": {
          "external_package.main": [
              "/path/to/another/repository/package/external_package/main.py"
          ],
          "external_package.non_python_file": [
              "/path/to/another/repository/package/external_package/non_python_file.thrift"
          ]
      }, 
      "deps": [],
      "buildfile_path": "/path/to/another/repository/package/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//generated:main": {
      "srcs": {
          "generated.main": [
              "generated/main.py"
          ]
      },
      "deps": ["//generated:lib"],
      "buildfile_path": "generated/BUCK",
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "//generated:lib": {
      "srcs": {
          "generated": [
              "generated/__init__.py"
          ]
      },
      "deps": [],
      "buildfile_path": "generated/BUCK",
      "python_version": "3.12",
      "python_platform": "linux",
      "relative_to": "build-out/materialized"
    }
  },
  "root": "/path/to/this/repository"
}
        "#;
        let parsed: TargetManifestDatabase = serde_json::from_str(EXAMPLE_JSON).unwrap();
        assert_eq!(parsed, TargetManifestDatabase::get_test_database());
    }

    #[test]
    fn test_produce_db() {
        let expected = smallmap! {
            Target::from_string("//colorama:py-stubs".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "colorama",
                        &[
                            "colorama/__init__.pyi",
                        ]
                    ),
                ],
                &[],
                "colorama/BUCK",
                &[
                    ("colorama", &[
                        "colorama/__init__.pyi",
                    ]),
                ],
                None,
            ),
            Target::from_string("//colorama:py".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "colorama",
                        &[
                            "colorama/__init__.py",
                        ]
                    ),
                ],
                &["//colorama:py-stubs"],
                "colorama/BUCK",
                &[
                    ("colorama", &[
                        "colorama/__init__.py",
                    ]),
                ],
                None,
            ),
            Target::from_string("//click:py".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "click",
                        &[
                            "click/__init__.pyi",
                            "click/__init__.py",
                        ],
                    )
                ],
                &[
                    "//colorama:py"
                ],
                "click/BUCK",
                &[
                    ("click", &[
                        "click/__init__.pyi",
                        "click/__init__.py",
                    ]),
                ],
                None,
            ),
            Target::from_string("//pyre/client/log:log".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "pyre.client.log",
                        &[
                            "pyre/client/log/__init__.py"
                        ]
                    ),
                    (
                        "pyre.client.log.log",
                        &[
                            "pyre/client/log/log.py",
                            "pyre/client/log/log.pyi",
                        ]
                    ),
                    (
                        "pyre.client.log.format",
                        &[
                            "pyre/client/log/format.py",
                        ],
                    ),
                ],
                &[
                    "//click:py"
                ],
                "pyre/client/log/BUCK",
                &[
                    ("pyre.client.log", &[
                     "pyre/client/log/__init__.py",
                    ]),
                    // Synthesized parent packages
                    ("pyre.client", &[
                     "pyre/client",
                    ]),
                    ("pyre", &[
                     "pyre",
                    ]),
                ],
                None,
            ),
            Target::from_string("//pyre/client/log:log2".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "log",
                        &[
                            "pyre/client/log/__init__.py"
                        ]
                    ),
                    (
                        "log.log",
                        &[
                            "pyre/client/log/log.py",
                            "pyre/client/log/log.pyi",
                        ]
                    ),
                    (
                        "log.format",
                        &[
                            "pyre/client/log/format.py",
                        ],
                    ),
                ],
                &[
                    "//click:py"
                ],
                "pyre/client/log/BUCK",
                &[
                    ("log", &[
                        "pyre/client/log/__init__.py",
                    ]),
                ],
                None,
            ),
            Target::from_string("//implicit_package/test:main".to_owned()) => PythonLibraryManifest::new(
                &[
                (
                    "implicit_package.main", &[
                        "implicit_package/test/main.py"
                    ]
                )
                ],
                &[
                    "//implicit_package/test:lib"
                ],
                "implicit_package/test/BUCK",
                &[
                ("implicit_package", &[
                        "implicit_package/test",
                    ],
                )
                ],
                None,
            ),
            Target::from_string("//implicit_package/test:lib".to_owned()) => PythonLibraryManifest::new(
                &[
                (
                    "implicit_package.lib.utils", &[
                        "implicit_package/test/lib/utils.py"
                    ],
                ),
                (
                    "implicit_package.package_boundary_violation", &[
                        "implicit_package/package_boundary_violation.py",
                    ],
                ),
                (
                    "implicit_package.deeply.nested.package.file", &[
                        "implicit_package/test/deeply/nested/package/file.py",
                    ],
                ),
                ],
                &["//external:package"],
                "implicit_package/test/BUCK",
                &[
                ("implicit_package", &[
                        "implicit_package/test",
                    ],
                ),
                ("implicit_package.lib", &[
                        "implicit_package/test/lib",
                    ],
                ),
                ("implicit_package.deeply.nested.package", &[
                        "implicit_package/test/deeply/nested/package",
                    ],
                ),
                ("implicit_package.deeply.nested", &[
                        "implicit_package/test/deeply/nested",
                    ],
                ),
                ("implicit_package.deeply", &[
                        "implicit_package/test/deeply",
                    ],
                ),
                ],
                None,
            ),
            Target::from_string("//external:package".to_owned()) => PythonLibraryManifest::new(
                &[
                ("external_package.main", &[
                 "/path/to/another/repository/package/external_package/main.py"
                    ]
                ),
                (
                    "external_package.non_python_file", &[
                        "/path/to/another/repository/package/external_package/non_python_file.thrift",
                    ],
                ),
                ],
                &[],
                "/path/to/another/repository/package/BUCK",
                &[
                ("external_package", &[
                 "/path/to/another/repository/package/external_package",
                    ],
                ),
                ],
                None,
            ),
            Target::from_string("//generated:main".to_owned()) => PythonLibraryManifest::new(
                &[
                ("generated.main", &[
                    "/path/to/this/repository/generated/main.py"
                    ],
                ),
                ],
                &["//generated:lib"],
                "/path/to/this/repository/generated/BUCK",
                &[
                ("generated", &[
                    "/path/to/this/repository/generated",
                    ],
                ),
                ],
                None,
            ),
            Target::from_string("//generated:lib".to_owned()) => PythonLibraryManifest::new(
                &[
                ("generated", &[
                    "/path/to/this/repository/build-out/materialized/generated/__init__.py"
                    ],
                ),
                ],
                &[],
                "/path/to/this/repository/generated/BUCK",
                &[
                ("generated", &[
                    "/path/to/this/repository/build-out/materialized/generated/__init__.py",
                    ],
                ),
                ],
                Some("/path/to/this/repository/build-out/materialized"),
            ),
        };
        assert_eq!(
            TargetManifestDatabase::get_test_database().produce_map(),
            expected
        );
    }

    #[test]
    fn test_package_finding() {
        // Test that fill_implicit_packages correctly synthesizes packages for all targets.
        // This tests explicit __init__ files are preserved and parent packages are synthesized.
        let db = TargetManifestDatabase::get_test_database();
        let result = db.produce_map();

        // Test colorama:py-stubs - explicit __init__.pyi
        let colorama_stubs = result
            .get(&Target::from_string("//colorama:py-stubs".to_owned()))
            .unwrap();
        assert!(
            colorama_stubs
                .packages
                .contains_key(&ModuleName::from_str("colorama")),
            "colorama:py-stubs should have 'colorama' package"
        );
        // Verify it's the explicit init file, not a synthesized directory
        assert!(
            colorama_stubs
                .packages
                .get(&ModuleName::from_str("colorama"))
                .unwrap()
                .first()
                .as_path()
                .ends_with("__init__.pyi"),
            "colorama package should point to explicit __init__.pyi"
        );

        // Test colorama:py - explicit __init__.py
        let colorama_py = result
            .get(&Target::from_string("//colorama:py".to_owned()))
            .unwrap();
        assert!(
            colorama_py
                .packages
                .contains_key(&ModuleName::from_str("colorama")),
            "colorama:py should have 'colorama' package"
        );
        assert!(
            colorama_py
                .packages
                .get(&ModuleName::from_str("colorama"))
                .unwrap()
                .first()
                .as_path()
                .ends_with("__init__.py"),
            "colorama package should point to explicit __init__.py"
        );

        // Test click:py - explicit __init__ with both .py and .pyi variants
        let click = result
            .get(&Target::from_string("//click:py".to_owned()))
            .unwrap();
        assert!(
            click.packages.contains_key(&ModuleName::from_str("click")),
            "click:py should have 'click' package"
        );
        assert_eq!(
            click
                .packages
                .get(&ModuleName::from_str("click"))
                .unwrap()
                .len(),
            2,
            "click package should have both .pyi and .py variants"
        );

        // Test pyre/client/log:log - explicit init plus synthesized parent packages
        let pyre_log = result
            .get(&Target::from_string("//pyre/client/log:log".to_owned()))
            .unwrap();
        assert!(
            pyre_log
                .packages
                .contains_key(&ModuleName::from_str("pyre.client.log")),
            "pyre/client/log:log should have 'pyre.client.log' package"
        );
        assert!(
            pyre_log
                .packages
                .contains_key(&ModuleName::from_str("pyre.client")),
            "pyre/client/log:log should have synthesized 'pyre.client' package"
        );
        assert!(
            pyre_log
                .packages
                .contains_key(&ModuleName::from_str("pyre")),
            "pyre/client/log:log should have synthesized 'pyre' package"
        );

        // Test implicit_package/test:main - synthesized implicit package
        let implicit_main = result
            .get(&Target::from_string(
                "//implicit_package/test:main".to_owned(),
            ))
            .unwrap();
        assert!(
            implicit_main
                .packages
                .contains_key(&ModuleName::from_str("implicit_package")),
            "implicit_package/test:main should have synthesized 'implicit_package' package"
        );

        // Test implicit_package/test:lib - deeply nested synthesized packages
        let implicit_lib = result
            .get(&Target::from_string(
                "//implicit_package/test:lib".to_owned(),
            ))
            .unwrap();
        assert!(
            implicit_lib
                .packages
                .contains_key(&ModuleName::from_str("implicit_package")),
            "implicit_package/test:lib should have 'implicit_package' package"
        );
        assert!(
            implicit_lib
                .packages
                .contains_key(&ModuleName::from_str("implicit_package.lib")),
            "implicit_package/test:lib should have 'implicit_package.lib' package"
        );
        assert!(
            implicit_lib
                .packages
                .contains_key(&ModuleName::from_str("implicit_package.deeply")),
            "implicit_package/test:lib should have 'implicit_package.deeply' package"
        );
        assert!(
            implicit_lib
                .packages
                .contains_key(&ModuleName::from_str("implicit_package.deeply.nested")),
            "implicit_package/test:lib should have 'implicit_package.deeply.nested' package"
        );
        assert!(
            implicit_lib.packages.contains_key(&ModuleName::from_str(
                "implicit_package.deeply.nested.package"
            )),
            "implicit_package/test:lib should have 'implicit_package.deeply.nested.package' package"
        );

        // Test external:package - external path packages
        let external = result
            .get(&Target::from_string("//external:package".to_owned()))
            .unwrap();
        assert!(
            external
                .packages
                .contains_key(&ModuleName::from_str("external_package")),
            "external:package should have 'external_package' package"
        );

        // Test generated:lib - explicit init with relative_to
        let generated_lib = result
            .get(&Target::from_string("//generated:lib".to_owned()))
            .unwrap();
        assert!(
            generated_lib
                .packages
                .contains_key(&ModuleName::from_str("generated")),
            "generated:lib should have 'generated' package"
        );
        assert!(
            generated_lib
                .packages
                .get(&ModuleName::from_str("generated"))
                .unwrap()
                .first()
                .as_path()
                .to_string_lossy()
                .contains("__init__.py"),
            "generated package should point to explicit __init__.py"
        );

        // Test generated:main - synthesized package
        let generated_main = result
            .get(&Target::from_string("//generated:main".to_owned()))
            .unwrap();
        assert!(
            generated_main
                .packages
                .contains_key(&ModuleName::from_str("generated")),
            "generated:main should have synthesized 'generated' package"
        );
    }

    #[test]
    fn test_implicit_packages_for_generated_files() {
        let db = TargetManifestDatabase::new(
            smallmap! {
                Target::from_string("//thrift:types".to_owned()) => TargetManifest::lib(
                    &[("foo.bar.types", &["buck-out/gen/foo/bar/types.pyi"])],
                    &[],
                    "thrift/BUCK",
                    &[],
                    None
                ),
            },
            PathBuf::from("/repo"),
        );

        let result = db.produce_map();
        let manifest = result
            .get(&Target::from_string("//thrift:types".to_owned()))
            .unwrap();

        assert!(
            manifest
                .packages
                .contains_key(&ModuleName::from_str("foo.bar")),
            "Expected packages to contain 'foo.bar', but got: {:?}",
            manifest.packages.keys().collect::<Vec<_>>()
        );
        assert!(
            manifest.packages.contains_key(&ModuleName::from_str("foo")),
            "Expected packages to contain 'foo', but got: {:?}",
            manifest.packages.keys().collect::<Vec<_>>()
        );
    }
}
