/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use serde::Deserialize;

use crate::config::config::ConfigFile;
use crate::config::environment::PythonEnvironment;
use crate::config::mypy::regex_converter;
use crate::module::wildcard::ModuleWildcard;
use crate::sys_info::PythonPlatform;
use crate::sys_info::PythonVersion;
use crate::util::globs::Globs;

#[derive(Clone, Debug, Deserialize)]
pub struct MypyConfig {
    files: Option<Vec<String>>,
    packages: Option<Vec<String>>,
    modules: Option<Vec<String>>,
    #[allow(dead_code)]
    #[serde(rename = "exclude")]
    exclude_regex: Option<String>,
    #[serde(rename = "mypy_path")]
    search_path: Option<Vec<PathBuf>>,
    #[serde(rename = "platform")]
    python_platform: Option<String>,
    python_version: Option<PythonVersion>,
    #[serde(rename = "python_executable")]
    python_interpreter: Option<PathBuf>,
    #[serde(rename = "follow_untyped_imports")]
    use_untyped_imports: bool,
}

#[derive(Deserialize)]
struct MypyOutput {
    mypy: MypyConfig,
    replace_imports: Vec<ModuleWildcard>,
}

impl MypyConfig {
    pub fn parse_mypy_config(ini_path: &Path) -> anyhow::Result<ConfigFile> {
        let script = "\
import configparser, json, re, sys
cp = configparser.ConfigParser()
with open(sys.argv[1]) as f:
    cp.read_file(f)
cfg = {}
replace_imports = []
follow_untyped_imports = False
for section in cp.sections():
    cfg[section] = {}
    for key, value in cp.items(section):
        if key == 'ignore_missing_imports' and section.startswith('mypy-') and value:
            replace_imports.extend(section[5:].split(','))
            continue
        if key in ('files', 'packages', 'modules'):
            value = [x.strip() for x in value.split(',') if x.strip()]
        elif key == 'mypy_path':
            value = [x.strip() for x in re.split('[,:]', value) if x.strip()]
        elif key == 'follow_untyped_imports':
            follow_untyped_imports |= value == 'True'
        elif value in ('True', 'False'):
            value = value == 'True'
        cfg[section][key] = value
    if not cfg[section]:
        del cfg[section]
mypy = cfg.pop('mypy', {})
mypy['follow_untyped_imports'] = follow_untyped_imports
print(json.dumps({'mypy': mypy, 'per_module': cfg, 'replace_imports': replace_imports}))
";
        let mut cmd = Command::new(
            PythonEnvironment::get_default_interpreter()
                .ok_or_else(|| anyhow::anyhow!("Failed to find python interpreter"))?,
        );
        cmd.arg("-c");
        cmd.arg(script);
        cmd.arg(ini_path);
        let output = cmd.output()?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to parse mypy config: {}",
                String::from_utf8_lossy(&output.stderr),
            ));
        }
        let raw_config = String::from_utf8(output.stdout)?;
        let MypyOutput {
            mypy,
            replace_imports,
        } = serde_json::from_str(&raw_config)?;

        let mut cfg = ConfigFile::default();

        let project_includes = [mypy.files, mypy.packages, mypy.modules]
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<_>>();
        if !project_includes.is_empty() {
            cfg.project_includes = Globs::new(project_includes);
        }

        if let Some(exclude_regex) = mypy.exclude_regex {
            let patterns = regex_converter::convert(&exclude_regex)?;
            if !patterns.is_empty() {
                cfg.project_excludes = Globs::new(patterns);
            }
        }

        if let Some(search_path) = mypy.search_path {
            cfg.search_path = search_path;
        }
        if let Some(platform) = mypy.python_platform {
            cfg.python_environment.python_platform = Some(PythonPlatform::new(&platform));
        }
        if mypy.python_version.is_some() {
            cfg.python_environment.python_version = mypy.python_version;
        }
        if mypy.python_interpreter.is_some() {
            cfg.python_interpreter = mypy.python_interpreter;
        }
        cfg.use_untyped_imports = mypy.use_untyped_imports;
        cfg.root.replace_imports_with_any = Some(replace_imports);

        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::fs_anyhow;

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = br#"[mypy]
files =
    src,
    other_src,
    test/some_test.py,

mypy_path = some_paths:comma,separated

unknown_option = True

exclude = src/include/|other_src/include/|src/specific/bad/file.py

[mypy-some.*.project]
ignore_missing_imports = True

[mypy-some.specific.project.subdir]
ignore_missing_imports = True

[mypy-stricter.on.this.*]
check_untyped_defs = True

[mypy-do.follow.*]
follow_untyped_imports = True

[mypy-comma,separated,projects]
ignore_missing_imports = True
"#;
        fs_anyhow::write(&input_path, mypy)?;

        let cfg = MypyConfig::parse_mypy_config(&input_path)?;

        let project_includes = Globs::new(vec![
            "src".to_owned(),
            "other_src".to_owned(),
            "test/some_test.py".to_owned(),
        ]);
        assert_eq!(cfg.project_includes, project_includes);

        assert_eq!(
            cfg.search_path,
            vec![
                PathBuf::from("some_paths"),
                PathBuf::from("comma"),
                PathBuf::from("separated"),
            ]
        );

        let expected_excludes = Globs::new(vec![
            "**/src/include/".to_owned(),
            "**/other_src/include/".to_owned(),
            "**/src/specific/bad/file.py".to_owned(),
        ]);
        assert_eq!(cfg.project_excludes, expected_excludes);

        assert_eq!(cfg.replace_imports_with_any().len(), 5);
        assert!(cfg.use_untyped_imports);
        Ok(())
    }

    #[test]
    fn test_no_overwrite_missing_includes() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = br#"[mypy]
unknown_option = True
"#;
        fs_anyhow::write(&input_path, mypy)?;

        let cfg = MypyConfig::parse_mypy_config(&input_path)?;
        let default = ConfigFile::default();
        assert_eq!(cfg.project_includes, default.project_includes);
        assert_eq!(cfg.project_excludes, default.project_excludes);
        Ok(())
    }
}
