/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Python environments of PEP 723 scripts, read from `uv workspace metadata`.

use std::path::Path;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context as _;
use serde::Deserialize;

/// The subset of `uv workspace metadata` output that names a script's environment.
#[derive(Deserialize)]
struct Metadata {
    environment: Option<Environment>,
}

#[derive(Deserialize)]
struct Environment {
    python: Python,
}

#[derive(Deserialize)]
struct Python {
    path: PathBuf,
}

/// Returns the contents of the PEP 723 `script` metadata block in `source`, if it has one.
pub fn script_metadata(source: &str) -> Option<&str> {
    let mut start = None;
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let text = line.trim_end_matches(['\n', '\r']);
        match start {
            None if text == "# /// script" => start = Some(offset + line.len()),
            Some(start) if text == "# ///" => return Some(&source[start..offset]),
            Some(_) if text != "#" && !text.starts_with("# ") => start = None,
            _ => {}
        }
        offset += line.len();
    }
    None
}

/// Returns the interpreter of the environment that uv runs `script` in, or `None` if that
/// environment does not exist. This never creates or updates the environment.
///
/// The uv executable is `$UV`, which uv sets for the commands that it runs, or `uv` on `PATH`.
#[cfg(not(target_arch = "wasm32"))]
pub fn script_interpreter(script: &Path) -> anyhow::Result<Option<PathBuf>> {
    let mut command = Command::new(std::env::var_os("UV").unwrap_or_else(|| "uv".into()));
    command
        .args([
            "workspace",
            "metadata",
            "--preview-features",
            "workspace-metadata",
            "--script",
        ])
        .arg(script);
    if let Some(directory) = script.parent() {
        command.current_dir(directory);
    }
    let output = command
        .output()
        .context("Failed to run `uv workspace metadata`")?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "`uv workspace metadata --script {}` failed: {}",
            script.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout)
        .context("Failed to parse `uv workspace metadata` output")?;
    Ok(metadata
        .environment
        .map(|environment| environment.python.path))
}

#[cfg(target_arch = "wasm32")]
pub fn script_interpreter(_script: &Path) -> anyhow::Result<Option<PathBuf>> {
    Err(anyhow::anyhow!(
        "uv script environments are not supported on WebAssembly"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_metadata() {
        assert_eq!(
            script_metadata(
                "#!/usr/bin/env python\n# /// script\n# dependencies = [\"six\"]\n#\n# ///\nimport six\n"
            ),
            Some("# dependencies = [\"six\"]\n#\n"),
        );
        assert_eq!(
            script_metadata("# /// script\r\n# requires-python = \">=3.12\"\r\n# ///\r\n"),
            Some("# requires-python = \">=3.12\"\r\n"),
        );
        assert_eq!(script_metadata("# /// script\nimport six\n# ///\n"), None);
        assert_eq!(script_metadata("# /// pyproject\n# ///\n"), None);
        assert_eq!(script_metadata("x = 1\n"), None);
    }
}
