/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::globs::FilteredGlobs;
use starlark_map::small_map::SmallMap;

use crate::commands::check::CheckArgs;
use crate::commands::run::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::finder::ConfigFinder;

pub fn dump_config(
    files_to_check: FilteredGlobs,
    config_finder: ConfigFinder,
    args: CheckArgs,
) -> anyhow::Result<CommandExitStatus> {
    let mut configs_to_files: SmallMap<ArcId<ConfigFile>, Vec<ModulePath>> = SmallMap::new();
    let mut handles = args
        .get_handles(files_to_check, &config_finder)?
        .into_iter()
        .map(|(handle, _)| handle)
        .collect::<Vec<_>>();
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
