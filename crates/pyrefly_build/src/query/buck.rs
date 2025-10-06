/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::process::Command;

use serde::Deserialize;
use serde::Serialize;

use crate::query::SourceDbQuerier;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct BxlArgs {
    isolation_dir: Option<String>,
    extras: Option<Vec<String>>,
}

#[derive(Debug)]
pub struct BxlQuerier(BxlArgs);

impl BxlQuerier {
    pub fn new(args: BxlArgs) -> Self {
        Self(args)
    }
}

impl SourceDbQuerier for BxlQuerier {
    fn construct_command(&self) -> Command {
        let mut cmd = Command::new("buck2");
        if let Some(isolation_dir) = &self.0.isolation_dir {
            cmd.arg("--isolation-dir");
            cmd.arg(isolation_dir);
        }
        cmd.arg("bxl");
        cmd.arg("--reuse-current-config");
        if let Some(metadata) = &self.0.extras {
            cmd.args(metadata);
        }
        cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
        cmd
    }
}
