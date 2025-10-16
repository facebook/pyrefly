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
use vec1::Vec1;

use crate::query::SourceDbQuerier;

/// Args and settings for querying a custom source DB.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct CustomQueryArgs {
    /// The command to run.
    /// Pyrefly will call this in the form `<command> @<argfile>`,
    /// where `<argfile>` has the format
    /// ```text
    /// --
    /// <arg-flag>
    /// <arg>
    /// ...
    /// ```
    /// and `<arg-flag>` is either `--file` or `--target`, depending on the type
    /// of `<arg>`
    /// and `<arg>` is an absolute path to a file or a build system's target.
    command: Vec1<String>,
}

/// A querier allowing for a custom command when querying and constructing source DB.
#[derive(Debug)]
pub struct CustomQuerier(CustomQueryArgs);

impl CustomQuerier {
    pub fn new(args: CustomQueryArgs) -> Self {
        Self(args)
    }
}

impl SourceDbQuerier for CustomQuerier {
    fn construct_command(&self) -> Command {
        let mut cmd = Command::new(self.0.command.first());
        cmd.args(self.0.command.iter().skip(1));
        cmd
    }
}
