/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::prelude::SliceExt;
use starlark_map::small_map::SmallMap;

use crate::commands::check::FullCheckArgs;
use crate::commands::check::Handles;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::state::require::Require;

// We intentionally make DumpConfig take the same arguments as Check so that dumping the
// config is as easy as changing the command name.
#[derive(Debug, Clone, Parser)]
pub struct DumpConfigArgs {
    #[command(flatten)]
    args: FullCheckArgs,
}

impl DumpConfigArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        // Pass on just the subset of args we use, the rest are irrelevant
        dump_config(self.args.files, self.args.args.config_override)
    }
}

fn dump_config(
    files: FilesArgs,
    config_override: ConfigOverrideArgs,
) -> anyhow::Result<CommandExitStatus> {
    config_override.validate()?;
    let (files_to_check, config_finder) = files.resolve(&config_override)?;

    let mut configs_to_files: SmallMap<ArcId<ConfigFile>, Vec<ModulePath>> = SmallMap::new();
    let handles = Handles::new(
        config_finder.checkpoint(files_to_check.files())?,
        &config_finder,
    );
    let mut handles = handles.all(Require::Everything).map(|x| x.0.dupe());
    handles.sort_by(|a, b| a.path().cmp(b.path()));
    for handle in handles {
        let path = handle.path();
        let config = config_finder.python_file(handle.module(), path);
        configs_to_files
            .entry(config)
            .or_default()
            .push(path.clone());
    }
    for error in config_finder.errors() {
        error.print();
    }
    for (config, files) in configs_to_files.into_iter() {
        match &config.source {
            ConfigSource::Synthetic => {
                println!("Default configuration");
            }
            ConfigSource::Marker(path) => {
                println!(
                    "Default configuration for project root marked by `{}`",
                    path.display()
                );
            }
            ConfigSource::File(path) => {
                println!("Configuration at `{}`", path.display());
            }
        }
        println!("  Using interpreter: {}", config.interpreters);
        println!("  Covered files:");
        for (i, fi) in files.iter().enumerate() {
            if i < 10 {
                println!("    {fi}");
            } else {
                println!("    ...and {} more", files.len() - 10);
                break;
            }
        }
        for path_part in config.structured_import_lookup_path() {
            if !path_part.is_empty() {
                println!("  {path_part}");
            }
        }
    }
    Ok(CommandExitStatus::Success)
}
