/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::anyhow;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use serde::Deserialize;
use serde_json::Value;
use tracing::warn;
use url::Url;

use crate::environment::environment::PythonEnvironment;

/// The JSON contract emitted by the interpreter-query script run in [`query`].
/// `distribution_urls` is left untyped because its per-entry shape is best-effort
/// PEP 610 metadata that [`editable_paths_from_query`] tolerates being partial.
#[derive(Deserialize)]
struct QueryOutput {
    python_platform: Option<PythonPlatform>,
    python_version: Option<PythonVersion>,
    site_package_path: Option<Vec<PathBuf>>,
    #[serde(default)]
    stdlib_paths: Vec<PathBuf>,
    #[serde(default)]
    distribution_urls: Value,
}

/// Given a path to a Python interpreter executable, query that interpreter for its
/// version, platform, and site package path. Return an error in the case of failure during
/// execution, parsing, or deserializing.
pub fn query(interpreter: &Path) -> anyhow::Result<PythonEnvironment> {
    if let Ok(pythonpath) = std::env::var("PYTHONPATH") {
        warn!(
            "PYTHONPATH environment variable is set to `{}`. Checks in other environments may not include these paths.",
            pythonpath
        );
    }

    let script = "\
import importlib.metadata, json, sys, sysconfig
platform = sys.platform
v = sys.version_info
version = '{}.{}.{}'.format(v.major, v.minor, v.micro)
stdlib_paths = [p for p in [sysconfig.get_path('stdlib')] if p is not None]
site_package_path = [p for p in sys.path if p != '' and '.zip' not in p and not p.endswith('/lib-dynload') and p not in stdlib_paths]
distribution_urls = []
for distribution in importlib.metadata.distributions():
    try:
        distribution_url = json.loads(distribution.read_text('direct_url.json') or 'null')
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue
    if isinstance(distribution_url, dict):
        distribution_urls.append(distribution_url)
print(json.dumps({'python_platform': platform, 'python_version': version, 'site_package_path': site_package_path, 'stdlib_paths': stdlib_paths, 'distribution_urls': distribution_urls}))
";

    let mut command = Command::new(interpreter);
    command.arg("-c");
    command.arg(script);

    let python_info = command.output()?;

    let stdout = String::from_utf8(python_info.stdout).with_context(|| {
        format!(
            "while parsing Python interpreter (`{}`) stdout for environment configuration",
            interpreter.display()
        )
    })?;
    if !python_info.status.success() {
        let stderr = String::from_utf8(python_info.stderr)
            .unwrap_or("<Failed to parse STDOUT from UTF-8 string>".to_owned());
        return Err(anyhow::anyhow!(
            "Unable to query interpreter {} for environment info:\nSTDOUT: {}\nSTDERR: {}",
            interpreter.display(),
            stdout,
            stderr
        ));
    }

    let query_output: QueryOutput = serde_json::from_str(&stdout)?;

    let python_platform = query_output.python_platform.ok_or_else(|| {
        anyhow!("Expected `python_platform` from Python interpreter query to be non-empty")
    })?;
    let python_version = query_output.python_version.ok_or_else(|| {
        anyhow!("Expected `python_version` from Python interpreter query to be non-empty")
    })?;
    let interpreter_site_package_path = query_output.site_package_path.ok_or_else(|| {
        anyhow!("Expected `site_package_path` from Python interpreter query to be non-empty")
    })?;
    let interpreter_editable_path = editable_paths_from_query(
        interpreter,
        &query_output.distribution_urls,
        &interpreter_site_package_path,
    )?;

    Ok(PythonEnvironment {
        python_platform: Some(python_platform),
        python_version: Some(python_version),
        // The queried value lives in `interpreter_site_package_path`; this is left
        // `Some(empty)` (rather than `None`) so `set_empty_to_default` treats it as
        // already resolved instead of re-applying Pyrefly's default.
        site_package_path: Some(Vec::new()),
        interpreter_site_package_path,
        interpreter_editable_path,
        interpreter_stdlib_path: query_output.stdlib_paths,
    })
}

fn editable_paths_from_query(
    interpreter: &Path,
    distribution_urls: &Value,
    interpreter_paths: &[PathBuf],
) -> anyhow::Result<Vec<PathBuf>> {
    let distribution_urls = distribution_urls.as_array().ok_or_else(|| {
        anyhow!(
            "Expected `distribution_urls` from Python interpreter (`{}`) query to be an array",
            interpreter.display()
        )
    })?;
    let editable_roots: HashSet<PathBuf> = distribution_urls
        .iter()
        .filter(|distribution_url| {
            distribution_url
                .pointer("/dir_info/editable")
                .and_then(Value::as_bool)
                == Some(true)
        })
        .filter_map(|distribution_url| distribution_url.get("url").and_then(Value::as_str))
        .filter_map(|url| Url::parse(url).ok()?.to_file_path().ok())
        // The direct URL points at the project root, but a `src`-layout
        // package (e.g. a UV workspace member) puts importable code
        // under `<root>/src` instead, so both must be accepted.
        .flat_map(|root| [root.clone(), root.join("src")])
        .filter_map(|path| path.canonicalize().ok())
        .collect();
    Ok(interpreter_paths
        .iter()
        .filter(|path| {
            path.canonicalize()
                .is_ok_and(|canonical_path| editable_roots.contains(&canonical_path))
        })
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn interpreter_query_classifies_editable_file_urls() {
        let temp = tempdir().unwrap();
        let project = temp.path().join("project");
        let source = project.join("src");
        let project_alias = source.join("..");
        let site_packages = temp.path().join("site-packages");
        let vendor = project.join("vendor");
        for path in [&source, &site_packages, &vendor] {
            std::fs::create_dir_all(path).unwrap();
        }
        let interpreter_paths = vec![
            project_alias.clone(),
            source.clone(),
            site_packages.clone(),
            vendor,
        ];
        let distribution_urls = json!([
            {
                "url": Url::from_file_path(&project).unwrap().to_string(),
                "dir_info": {"editable": true},
            },
            {
                "url": Url::from_file_path(&site_packages).unwrap().to_string(),
                "dir_info": {"editable": false},
            },
            {
                "url": "https://example.com/vendor",
                "dir_info": {"editable": true},
            },
            {
                "url": "not a URL",
                "dir_info": {"editable": true},
            },
            {"dir_info": {"editable": true}},
            {"url": Url::from_file_path(&project).unwrap().to_string()},
        ]);

        let paths =
            editable_paths_from_query(Path::new("python"), &distribution_urls, &interpreter_paths)
                .unwrap();

        assert_eq!(paths, vec![project_alias, source]);
    }

    #[test]
    fn interpreter_query_rejects_missing_distribution_urls() {
        let message = editable_paths_from_query(Path::new("python"), &Value::Null, &[])
            .unwrap_err()
            .to_string();

        assert!(message.contains("distribution_urls"));
        assert!(message.contains("Python interpreter (`python`)"));
    }
}
