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

use crate::environment::finder::walk_interpreter;

const SEARCH_DEPTH: usize = 1;
pub const ENV_VAR: &str = "CONDA_PREFIX";

pub fn find(env_path: &Path) -> Option<PathBuf> {
    walk_interpreter(env_path, SEARCH_DEPTH).next()
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
    get_env_path(env_name).and_then(|mut p| {
        p.push("bin");
        find(&p).ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find interpreter for environment named `{env_name}` at `{}`",
                p.display(),
            )
        })
    })
}
