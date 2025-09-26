/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt::Debug;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context as _;
use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::sys_info::SysInfo;
use serde::Deserialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::source_db::Target;

/// An enum representing something that has been included by the build system, and
/// which the build system should query for when building the sourcedb.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Include {
    #[expect(unused)]
    Target(Target),
    Path(PathBuf),
}

impl Include {
    pub fn path(path: PathBuf) -> Self {
        Self::Path(path)
    }

    fn to_bxl_args(&self) -> impl Iterator<Item = &OsStr> {
        match self {
            Include::Target(target) => [OsStr::new("--target"), target.to_os_str()].into_iter(),
            Include::Path(path) => [OsStr::new("--file"), path.as_os_str()].into_iter(),
        }
    }
}

pub fn query_source_db<'a>(
    files: impl Iterator<Item = &'a Include>,
    cwd: &Path,
) -> anyhow::Result<TargetManifestDatabase> {
    let mut files = files.peekable();
    if files.peek().is_none() {
        return Ok(TargetManifestDatabase {
            db: SmallMap::new(),
            root: cwd.to_path_buf(),
        });
    }

    let mut cmd = Command::new("buck2");
    cmd.arg("bxl");
    cmd.arg("--reuse-current-config");
    cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
    cmd.arg("--");
    cmd.args(files.flat_map(Include::to_bxl_args));
    cmd.current_dir(cwd);

    let result = cmd.output()?;
    if !result.status.success() {
        let stdout = String::from_utf8(result.stdout)
            .unwrap_or_else(|_| "<Failed to parse stdout from Buck source db query>".to_owned());
        let stderr = String::from_utf8(result.stderr)
            .unwrap_or_else(|_| "<Failed to parse stderr from Buck source db query>".to_owned());

        return Err(anyhow::anyhow!(
            "Buck source db query failed...\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        ));
    }

    serde_json::from_slice(&result.stdout).with_context(|| {
        "Failed to construct valid `TargetManifestDatabase` from BXL query result".to_owned()
    })
}

#[derive(Debug, PartialEq, Eq, Deserialize, Clone)]
pub(crate) struct PythonLibraryManifest {
    pub deps: SmallSet<Target>,
    pub srcs: SmallMap<ModuleName, Vec1<PathBuf>>,
    #[serde(flatten)]
    pub sys_info: SysInfo,
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
        self.srcs
            .iter_mut()
            .for_each(|(_, paths)| paths.iter_mut().for_each(|p| *p = root.join(&**p)));
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
    pub fn produce_map(self) -> SmallMap<Target, PythonLibraryManifest> {
        let mut result = SmallMap::new();
        let aliases: SmallMap<Target, Target> = self
            .db
            .iter()
            .filter_map(|(t, manifest)| match manifest {
                TargetManifest::Alias { alias } => Some((t.dupe(), alias.dupe())),
                _ => None,
            })
            .collect();
        for (target, manifest) in self.db {
            match manifest {
                TargetManifest::Alias { .. } => continue,
                TargetManifest::Library(mut lib) => {
                    lib.replace_alias_deps(&aliases);
                    lib.rewrite_relative_to_root(&self.root);
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
                    Target::from_string("//colorama:py".to_owned()) => TargetManifest::lib(
                        &[
                        (
                            "colorama",
                            &[
                            "colorama/__init__.py",
                            "colorama/__init__.pyi",
                            ]
                        ),
                        ],
                        &[],
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
                            ]
                        ),
                        ],
                        &[
                        "//click:click"
                        ],
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
                        )
                        ],
                        &[
                        "//click:click"
                        ],
                    )
                },
                PathBuf::from("/path/to/this/repository"),
            )
        }
    }

    fn map_srcs(
        srcs: &[(&str, &[&str])],
        prefix_paths: Option<&str>,
    ) -> SmallMap<ModuleName, Vec1<PathBuf>> {
        let prefix = prefix_paths.map(Path::new);
        let map_path = |p| prefix.map_or_else(|| PathBuf::from(p), |prefix| prefix.join(p));
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

    impl TargetManifest {
        fn alias(target: &str) -> Self {
            TargetManifest::Alias {
                alias: Target::from_string(target.to_owned()),
            }
        }

        pub fn lib(srcs: &[(&str, &[&str])], deps: &[&str]) -> Self {
            TargetManifest::Library(PythonLibraryManifest {
                srcs: map_srcs(srcs, None),
                deps: map_deps(deps),
                sys_info: SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
            })
        }
    }

    impl PythonLibraryManifest {
        fn new(srcs: &[(&str, &[&str])], deps: &[&str]) -> Self {
            Self {
                srcs: map_srcs(srcs, Some("/path/to/this/repository")),
                deps: map_deps(deps),
                sys_info: SysInfo::new(PythonVersion::new(3, 12, 0), PythonPlatform::linux()),
            }
        }
    }

    #[test]
    fn example_json_parses() {
        const EXAMPLE_JSON: &str = r#"
{
  "db": {
    "//colorama:py": {
      "srcs": {
        "colorama": [
          "colorama/__init__.py",
          "colorama/__init__.pyi"
        ]
      },
      "deps": [],
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
        ]
      },
      "deps": [
        "//click:click"
      ],
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
        ]
      },
      "deps": [
        "//click:click"
      ],
      "python_version": "3.12",
      "python_platform": "linux"
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
            Target::from_string("//colorama:py".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "colorama",
                        &[
                            "colorama/__init__.py",
                            "colorama/__init__.pyi",
                        ]
                    ),
                ],
                &[],
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
                ],
                &[
                    "//click:py"
                ],
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
                    )
                ],
                &[
                    "//click:py"
                ],
            )
        };
        assert_eq!(
            TargetManifestDatabase::get_test_database().produce_map(),
            expected
        );
    }
}
