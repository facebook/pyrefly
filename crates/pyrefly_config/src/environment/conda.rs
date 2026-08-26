/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context as _;
use itertools::Itertools as _;
use serde::Deserialize;

pub const ENV_VAR: &str = "CONDA_PREFIX";

#[cfg(windows)]
fn interpreter_candidates(root: &Path) -> [PathBuf; 2] {
    [root.join("python.exe"), root.join("Scripts/python.exe")]
}

#[cfg(not(windows))]
fn interpreter_candidates(root: &Path) -> [PathBuf; 2] {
    [root.join("bin/python"), root.join("bin/python3")]
}

pub fn find(env_path: &Path) -> Option<PathBuf> {
    interpreter_candidates(env_path)
        .into_iter()
        .find(|path| path.is_file())
}

pub fn get_env_path(env_name: &str) -> anyhow::Result<PathBuf> {
    let mut cmd = Command::new("conda");
    cmd.args(["info", "--envs", "--json"]);

    let output = cmd
        .output()
        .with_context(|| "While running query: `conda info --envs --json`.")?;

    let stdout = String::from_utf8(output.stdout)
        .with_context(|| "While parsing output from query: `conda info --envs --json`.")?;

    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)
            .unwrap_or("<Failed to parse STDOUT from UTF-8 string>".to_owned());
        return Err(anyhow::anyhow!(
            "Unable to conda for interpreter:\nSTDOUT: {}\nSTDERR: {}",
            stdout,
            stderr
        ));
    }

    #[derive(Deserialize)]
    struct CondaEnvOutput {
        envs: Vec<String>,
    }

    let conda_output: CondaEnvOutput =
        serde_json::from_str(&stdout).with_context(|| "While deserializing conda query output")?;

    conda_output.envs.iter().find(|env_path| {
        env_path.ends_with(env_name)
    }).map(PathBuf::from).ok_or_else(|| {
        let found_environments = conda_output.envs.iter().filter_map(|e| Path::new(e).file_name()?.to_str()).join(", ");
        anyhow::anyhow!(
                "Could not find provided Conda environment (`{env_name}`) when querying Conda. Found environments: `{found_environments}`"
        )
    })
}

pub fn find_interpreter_from_env(env_name: &str) -> anyhow::Result<PathBuf> {
    get_env_path(env_name).and_then(|p| {
        find(&p).ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find interpreter for environment named `{env_name}` at `{}`",
                p.display(),
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use pyrefly_util::test_path::TestPath;

    use super::*;

    #[test]
    fn test_find_interpreter() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let relative_interpreter = if cfg!(windows) {
            "python.exe"
        } else {
            "bin/python"
        };
        let layout = if cfg!(windows) {
            vec![TestPath::file("python.exe")]
        } else {
            vec![TestPath::dir("bin", vec![TestPath::file("python")])]
        };
        TestPath::setup_test_directory(root, layout);

        assert_eq!(find(root), Some(root.join(relative_interpreter)));
    }
}
