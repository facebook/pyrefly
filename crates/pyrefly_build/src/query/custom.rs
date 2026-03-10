/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use serde::Deserialize;
use serde::Serialize;
use vec1::Vec1;

use crate::query::SourceDbQuerier;

/// Args and settings for querying a custom source DB.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default, Hash)]
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
    ///
    /// `<arg-flag>` is either `--file` or `--target`, depending on the type
    /// of `<arg>`, and `<arg>` is an absolute path to a file or a build system's target.
    pub command: Vec1<String>,

    /// The root of the repository. Repo roots here will be shared between configs.
    #[serde(default)]
    pub repo_root: Option<PathBuf>,
}

#[cfg(feature = "jsonschema")]
impl schemars::JsonSchema for CustomQueryArgs {
    fn schema_name() -> String {
        "CustomQueryArgs".to_owned()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::*;

        let mut properties = schemars::Map::new();

        // command is Vec1<String> — array of strings with minItems: 1
        properties.insert(
            "command".to_owned(),
            SchemaObject {
                instance_type: Some(InstanceType::Array.into()),
                array: Some(Box::new(ArrayValidation {
                    items: Some(SingleOrVec::Single(Box::new(
                        generator.subschema_for::<String>(),
                    ))),
                    min_items: Some(1),
                    ..Default::default()
                })),
                metadata: Some(Box::new(Metadata {
                    description: Some(
                        "The command executed to query the build system about available targets."
                            .to_owned(),
                    ),
                    ..Default::default()
                })),
                ..Default::default()
            }
            .into(),
        );

        properties.insert(
            "repo-root".to_owned(),
            SchemaObject {
                instance_type: Some(InstanceType::String.into()),
                metadata: Some(Box::new(Metadata {
                    description: Some(
                        "The root directory of the repository for the build system.".to_owned(),
                    ),
                    ..Default::default()
                })),
                ..Default::default()
            }
            .into(),
        );

        SchemaObject {
            instance_type: Some(InstanceType::Object.into()),
            object: Some(Box::new(ObjectValidation {
                properties,
                required: ["command".to_owned()].into_iter().collect(),
                ..Default::default()
            })),
            ..Default::default()
        }
        .into()
    }
}

impl CustomQueryArgs {
    pub fn get_repo_root(&self, cwd: &std::path::Path) -> anyhow::Result<PathBuf> {
        Ok(self.repo_root.as_deref().unwrap_or(cwd).to_path_buf())
    }
}

impl Display for CustomQueryArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "command: {:?}, repo_root: {:?}",
            self.command, self.repo_root
        )
    }
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
    fn construct_command(&self, _: Option<&Path>) -> Command {
        let mut cmd = Command::new(self.0.command.first());
        cmd.args(self.0.command.iter().skip(1));
        cmd
    }
}
