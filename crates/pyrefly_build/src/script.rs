/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::VecDeque;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;

use anyhow::Context as _;
use anyhow::Result;
use dupe::Dupe as _;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;
use vec1::Vec1;

use crate::buck::query::PythonLibraryManifest;
use crate::handle::Handle;
use crate::source_db::SourceDatabase;
use crate::source_db::Target;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub struct QueryScript {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct Inner {
    db: SmallMap<Target, PythonLibraryManifest>,
    path_lookup: SmallMap<PathBuf, Target>,
}

impl Inner {
    fn new() -> Self {
        Inner {
            db: SmallMap::new(),
            path_lookup: SmallMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct ScriptSourceDatabase {
    inner: RwLock<Inner>,
    queried_files: Mutex<SmallSet<PathBuf>>,
    cwd: PathBuf,
    script: QueryScript,
}

impl ScriptSourceDatabase {
    pub fn new(cwd: PathBuf, script: QueryScript) -> Self {
        ScriptSourceDatabase {
            inner: RwLock::new(Inner::new()),
            queried_files: Mutex::new(SmallSet::new()),
            cwd,
            script,
        }
    }

    fn update_with_target_manifest(&self, new_db: SmallMap<Target, PythonLibraryManifest>) -> bool {
        let read = self.inner.read();
        if new_db == read.db {
            debug!("No source DB changes from script query");
            return false;
        }
        drop(read);
        let mut write = self.inner.write();
        write.path_lookup = SmallMap::new();
        for (target, manifest) in new_db.iter() {
            for source in manifest.srcs.values().flatten() {
                if let Some(old_target) = write.path_lookup.get_mut(&**source) {
                    let new_target = (&*old_target).min(target);
                    *old_target = new_target.dupe();
                } else {
                    write.path_lookup.insert(source.clone(), target.dupe());
                }
            }
        }
        write.db = new_db;
        debug!("Finished updating source DB with script response");
        true
    }

    fn run_query(
        &self,
        files: &SmallSet<PathBuf>,
    ) -> anyhow::Result<SmallMap<Target, PythonLibraryManifest>> {
        let mut command = Command::new(&self.script.command);
        command.args(&self.script.args);
        for file in files {
            command.arg("--file");
            command.arg(file);
        }
        command.current_dir(&self.cwd);

        let output = command
            .output()
            .with_context(|| format!("Failed to execute query script `{}`", self.script.command))?;
        if !output.status.success() {
            let stdout = String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "<Failed to parse stdout from query script>".to_owned());
            let stderr = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "<Failed to parse stderr from query script>".to_owned());
            return Err(anyhow::anyhow!(
                "Query script failed...\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            ));
        }

        let parsed: ScriptManifestDatabase =
            serde_json::from_slice(&output.stdout).with_context(|| {
                "Failed to construct valid manifest database from query script output".to_owned()
            })?;
        parsed.into_python_manifest_map()
    }
}

impl SourceDatabase for ScriptSourceDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        // TODO: consider allowing scripts to pre-populate modules to check.
        vec![]
    }

    fn lookup(&self, module: &ModuleName, origin: Option<&Path>) -> Option<ModulePath> {
        let origin = origin?;
        let read = self.inner.read();
        let start_target = read.path_lookup.get(&origin.to_path_buf())?;
        let mut queue = VecDeque::new();
        let mut visited = SmallSet::new();
        queue.push_front(start_target);

        while let Some(current_target) = queue.pop_front() {
            if !visited.insert(current_target) {
                continue;
            }
            let Some(manifest) = read.db.get(current_target) else {
                continue;
            };

            if let Some(paths) = manifest.srcs.get(module) {
                return Some(ModulePath::filesystem(paths.first().to_path_buf()));
            }

            manifest.deps.iter().for_each(|t| queue.push_back(t));
        }

        None
    }

    fn handle_from_module_path(&self, module_path: ModulePath) -> Option<Handle> {
        let read = self.inner.read();
        let target = read.path_lookup.get(&module_path.as_path().to_path_buf())?;
        let manifest = read.db.get(target)?;
        let module_name = manifest
            .srcs
            .iter()
            .find(|(_, paths)| paths.iter().any(|p| p == module_path.as_path()))?
            .0;
        Some(Handle::new(
            module_name.dupe(),
            module_path.dupe(),
            manifest.sys_info.dupe(),
        ))
    }

    fn requery_source_db(&self, files: SmallSet<PathBuf>) -> anyhow::Result<bool> {
        let mut queried_files = self.queried_files.lock();
        if *queried_files == files {
            debug!("Not querying script source DB, since no inputs have changed");
            return Ok(false);
        }
        *queried_files = files.clone();
        info!("Querying script for source DB");
        let raw_db = self.run_query(&files)?;
        info!("Finished querying script for source DB");
        Ok(self.update_with_target_manifest(raw_db))
    }

    fn get_critical_files(&self) -> SmallSet<PathBuf> {
        let read = self.inner.read();
        read.db
            .values()
            .map(|m| m.buildfile_path.to_path_buf())
            .chain(
                read.db
                    .values()
                    .flat_map(|m| m.srcs.values().flatten().map(|p| p.to_path_buf())),
            )
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct ScriptManifestDatabase {
    db: SmallMap<Target, ScriptManifest>,
    root: PathBuf,
}

impl ScriptManifestDatabase {
    fn into_python_manifest_map(self) -> anyhow::Result<SmallMap<Target, PythonLibraryManifest>> {
        self.db
            .into_iter()
            .map(|(target, manifest)| {
                manifest
                    .into_python_manifest(&self.root)
                    .map(|manifest| (target, manifest))
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct ScriptManifest {
    #[serde(default)]
    deps: Vec<Target>,
    srcs: SmallMap<ModuleName, Vec<String>>,
    python_version: String,
    python_platform: String,
    #[serde(default)]
    buildfile_path: Option<String>,
}

impl ScriptManifest {
    fn into_python_manifest(self, root: &Path) -> Result<PythonLibraryManifest> {
        let version = PythonVersion::from_str(&self.python_version).with_context(|| {
            format!(
                "Invalid python_version `{}` from build script",
                self.python_version
            )
        })?;
        let platform = PythonPlatform::new(&self.python_platform);
        let sys_info = SysInfo::new(version, platform);

        let mut srcs = SmallMap::new();
        for (module, paths) in self.srcs {
            let mut joined_paths = Vec::with_capacity(paths.len());
            for path in paths {
                let candidate = PathBuf::from(&path);
                let abs = if candidate.is_absolute() {
                    candidate
                } else {
                    root.join(candidate)
                };
                joined_paths.push(abs);
            }
            let vec1 = Vec1::try_from_vec(joined_paths).map_err(|_| {
                anyhow::anyhow!("Module `{module}` in build script output had no source paths")
            })?;
            srcs.insert(module, vec1);
        }

        let deps = self.deps.into_iter().collect();

        let buildfile_path = self
            .buildfile_path
            .map(PathBuf::from)
            .map(|p| if p.is_absolute() { p } else { root.join(p) })
            .or_else(|| {
                srcs.values()
                    .next()
                    .and_then(|paths| paths.first().parent().map(|p| p.join("BUILD")))
            })
            .unwrap_or_else(|| root.to_path_buf());

        Ok(PythonLibraryManifest {
            deps,
            srcs,
            sys_info,
            buildfile_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;
    use pyrefly_python::sys_info::SysInfo;
    use starlark_map::smallset;

    use super::*;

    fn sample_manifest_json(root: &Path) -> String {
        format!(
            r#"{{
  "db": {{
    "//pkg:lib": {{
      "deps": [],
      "srcs": {{
        "pkg": ["pkg/__init__.py"],
        "pkg.module": ["pkg/module.py"]
      }},
      "python_version": "3.11",
      "python_platform": "linux",
      "buildfile_path": "pkg/BUILD"
    }}
  }},
  "root": "{}"
}}"#,
            root.display()
        )
    }

    #[cfg(unix)]
    fn write_command(tempdir: &Path, json: &str) -> anyhow::Result<PathBuf> {
        use std::os::unix::fs::PermissionsExt;

        let script_path = tempdir.join("script.sh");
        let output_path = tempdir.join("out.json");
        fs::write(&output_path, json)?;
        fs::write(
            &script_path,
            format!("#!/usr/bin/env sh\ncat \"{}\"\n", output_path.display()),
        )?;
        let mut perms = fs::metadata(&script_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms)?;
        Ok(script_path)
    }

    #[cfg(windows)]
    fn write_command(tempdir: &Path, json: &str) -> anyhow::Result<PathBuf> {
        let script_path = tempdir.join("script.cmd");
        let output_path = tempdir.join("out.json");
        fs::write(&output_path, json)?;
        fs::write(
            &script_path,
            format!("@echo off\ntype \"{}\"\n", output_path.display()),
        )?;
        Ok(script_path)
    }

    #[test]
    fn script_source_db_queries_and_updates() -> anyhow::Result<()> {
        let tempdir = tempfile::tempdir()?;
        let json = sample_manifest_json(tempdir.path());
        let script_path = write_command(tempdir.path(), &json)?;

        let sys_info = SysInfo::new(PythonVersion::new(3, 11, 0), PythonPlatform::new("linux"));

        let db = ScriptSourceDatabase::new(
            tempdir.path().to_path_buf(),
            QueryScript {
                command: script_path.to_string_lossy().into_owned(),
                args: vec![],
            },
        );

        let files = smallset![PathBuf::from("pkg/module.py")];
        assert!(db.requery_source_db(files.clone())?);

        let module = ModuleName::from_str("pkg.module");
        let lookup = db.lookup(
            &module,
            Some(tempdir.path().join("pkg/module.py").as_path()),
        );
        assert_eq!(
            lookup,
            Some(ModulePath::filesystem(tempdir.path().join("pkg/module.py")))
        );

        let handle = db
            .handle_from_module_path(ModulePath::filesystem(tempdir.path().join("pkg/module.py")))
            .expect("handle for module path");
        assert_eq!(handle.module(), module);
        assert_eq!(handle.sys_info(), &sys_info);

        assert!(!db.requery_source_db(files)?);

        Ok(())
    }
}
