/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
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

pub fn query_source_db<'a>(
    files: impl Iterator<Item = &'a PathBuf>,
    cwd: &Path,
) -> anyhow::Result<TargetManifestDatabase> {
    // TODO(connernilsen): handle querying targets too later on
    let mut cmd = Command::new("buck2");
    cmd.arg("bxl");
    cmd.arg("--reuse-current-config");
    cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
    cmd.arg("--");
    cmd.args(files.flat_map(|f| [OsStr::new("--file"), f.as_os_str()].into_iter()));
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
            .for_each(|(_, paths)| paths.iter_mut().for_each(|p| *p = root.join(&p)));
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize, Clone)]
#[serde(untagged)]
enum TargetManifest {
    Library(PythonLibraryManifest),
    Alias { alias: Target },
}

#[derive(Debug, PartialEq, Eq, Deserialize, Clone)]
pub(crate) struct TargetManifestDatabase {
    db: SmallMap<Target, TargetManifest>,
    root: PathBuf,
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
    use std::sync::LazyLock;

    use pretty_assertions::assert_eq;
    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;
    use starlark_map::smallmap;

    use super::*;

    impl TargetManifestDatabase {
        fn new(db: SmallMap<Target, TargetManifest>, root: PathBuf) -> Self {
            TargetManifestDatabase { db, root }
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

        fn lib(srcs: &[(&str, &[&str])], deps: &[&str]) -> Self {
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

    // This is a simplified sourcedb taken from the BXL output run on pyre/client/log/log.py.
    // We also add a few extra entries to model some of the behavior around multiple entries
    // (i.e. multiple file paths corresponding to a module path, multiple module paths in
    // different targets).
    static EXAMPLE_BUILD: LazyLock<TargetManifestDatabase> = LazyLock::new(|| {
        TargetManifestDatabase::new(
            smallmap! {
                Target::from_string("build_root//third-party/pypi/colorama/0.4.6:py".to_owned()) => TargetManifest::lib(
                    &[
                        (
                            "colorama",
                            &[
                                "third-party/pypi/colorama/0.4.6/colorama/__init__.py",
                                "third-party/pypi/colorama/0.4.6/colorama/__init__.pyi",
                            ]
                        ),
                    ],
                    &[],
                ),
                Target::from_string("build_root//third-party/pypi/colorama:colorama".to_owned()) => TargetManifest::alias(
                    "build_root//third-party/pypi/colorama/0.4.6:py"
                ),
                Target::from_string("build_root//third-party/pypi/click/8.1.7:py".to_owned()) => TargetManifest::lib(
                    &[
                        (
                            "click",
                            &[
                                "third-party/pypi/click/8.1.7/src/click/__init__.pyi",
                                "third-party/pypi/click/8.1.7/src/click/__init__.py",
                            ],
                        )
                    ],
                    &[
                        "build_root//third-party/pypi/colorama:colorama"
                    ],
                ),
                Target::from_string("build_root//third-party/pypi/click:click".to_owned()) => TargetManifest::alias(
                    "build_root//third-party/pypi/click/8.1.7:py"
                ),
                Target::from_string("sub_root//pyre/client/log:log".to_owned()) => TargetManifest::lib(
                    &[
                        (
                            "pyre.client.log",
                            &[
                                "sub_root/pyre/client/log/__init__.py"
                            ]
                        ),
                        (
                            "pyre.client.log.log",
                            &[
                                "sub_root/pyre/client/log/log.py",
                                "sub_root/pyre/client/log/log.pyi",
                            ]
                        ),
                    ],
                    &[
                        "build_root//third-party/pypi/click:click"
                    ],
                ),
                Target::from_string("sub_root//pyre/client/log:log2".to_owned()) => TargetManifest::lib(
                    &[
                        (
                            "log",
                            &[
                                "sub_root/pyre/client/log/__init__.py"
                            ]
                        ),
                        (
                            "log.log",
                            &[
                                "sub_root/pyre/client/log/log.py",
                                "sub_root/pyre/client/log/log.pyi",
                            ]
                        )
                    ],
                    &[
                        "build_root//third-party/pypi/click:click"
                    ],
                )
            },
            PathBuf::from("/path/to/this/repository"),
        )
    });

    #[test]
    fn example_json_parses() {
        const EXAMPLE_JSON: &str = r#"
{
  "db": {
    "build_root//third-party/pypi/colorama/0.4.6:py": {
      "srcs": {
        "colorama": [
          "third-party/pypi/colorama/0.4.6/colorama/__init__.py",
          "third-party/pypi/colorama/0.4.6/colorama/__init__.pyi"
        ]
      },
      "deps": [],
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "build_root//third-party/pypi/colorama:colorama": {
      "alias": "build_root//third-party/pypi/colorama/0.4.6:py"
    },
    "build_root//third-party/pypi/click/8.1.7:py": {
      "srcs": {
        "click": [
          "third-party/pypi/click/8.1.7/src/click/__init__.pyi",
          "third-party/pypi/click/8.1.7/src/click/__init__.py"
        ]
      },
      "deps": [
        "build_root//third-party/pypi/colorama:colorama"
      ],
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "build_root//third-party/pypi/click:click": {
      "alias": "build_root//third-party/pypi/click/8.1.7:py"
    },
    "sub_root//pyre/client/log:log": {
      "srcs": {
        "pyre.client.log": [
          "sub_root/pyre/client/log/__init__.py"
        ],
        "pyre.client.log.log": [
          "sub_root/pyre/client/log/log.py",
          "sub_root/pyre/client/log/log.pyi"
        ]
      },
      "deps": [
        "build_root//third-party/pypi/click:click"
      ],
      "python_version": "3.12",
      "python_platform": "linux"
    },
    "sub_root//pyre/client/log:log2": {
      "srcs": {
        "log": [
          "sub_root/pyre/client/log/__init__.py"
        ],
        "log.log": [
          "sub_root/pyre/client/log/log.py",
          "sub_root/pyre/client/log/log.pyi"
        ]
      },
      "deps": [
        "build_root//third-party/pypi/click:click"
      ],
      "python_version": "3.12",
      "python_platform": "linux"
    }
  },
  "root": "/path/to/this/repository"
}
        "#;
        let parsed: TargetManifestDatabase = serde_json::from_str(EXAMPLE_JSON).unwrap();
        assert_eq!(parsed, *EXAMPLE_BUILD);
    }

    #[test]
    fn test_produce_db() {
        let expected = smallmap! {
            Target::from_string("build_root//third-party/pypi/colorama/0.4.6:py".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "colorama",
                        &[
                            "third-party/pypi/colorama/0.4.6/colorama/__init__.py",
                            "third-party/pypi/colorama/0.4.6/colorama/__init__.pyi",
                        ]
                    ),
                ],
                &[],
            ),
            Target::from_string("build_root//third-party/pypi/click/8.1.7:py".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "click",
                        &[
                            "third-party/pypi/click/8.1.7/src/click/__init__.pyi",
                            "third-party/pypi/click/8.1.7/src/click/__init__.py",
                        ],
                    )
                ],
                &[
                    "build_root//third-party/pypi/colorama/0.4.6:py"
                ],
            ),
            Target::from_string("sub_root//pyre/client/log:log".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "pyre.client.log",
                        &[
                            "sub_root/pyre/client/log/__init__.py"
                        ]
                    ),
                    (
                        "pyre.client.log.log",
                        &[
                            "sub_root/pyre/client/log/log.py",
                            "sub_root/pyre/client/log/log.pyi",
                        ]
                    ),
                ],
                &[
                    "build_root//third-party/pypi/click/8.1.7:py"
                ],
            ),
            Target::from_string("sub_root//pyre/client/log:log2".to_owned()) => PythonLibraryManifest::new(
                &[
                    (
                        "log",
                        &[
                            "sub_root/pyre/client/log/__init__.py"
                        ]
                    ),
                    (
                        "log.log",
                        &[
                            "sub_root/pyre/client/log/log.py",
                            "sub_root/pyre/client/log/log.pyi",
                        ]
                    )
                ],
                &[
                    "build_root//third-party/pypi/click/8.1.7:py"
                ],
            )
        };
        assert_eq!(EXAMPLE_BUILD.clone().produce_map(), expected);
    }
}
