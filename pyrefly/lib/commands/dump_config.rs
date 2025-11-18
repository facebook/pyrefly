/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;

use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use starlark_map::small_map::SmallMap;

use crate::commands::check::FullCheckArgs;
use crate::commands::check::Handles;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;

#[derive(Debug, Clone)]
enum MaxFiles {
    All,
    Count(usize),
}

impl MaxFiles {
    fn should_print(&self, iteration: usize) -> bool {
        match self {
            Self::All => true,
            Self::Count(count) => iteration < *count,
        }
    }

    fn remaining(&self, total: usize) -> usize {
        match self {
            Self::All => 0,
            Self::Count(count) => total - *count,
        }
    }
}

impl FromStr for MaxFiles {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.to_lowercase() == "all" {
            Ok(Self::All)
        } else if let Ok(count) = usize::from_str(s) {
            Ok(Self::Count(count))
        } else {
            Err("expected `all` or positive integer")
        }
    }
}

impl Display for MaxFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "all"),
            Self::Count(count) => write!(f, "{count}"),
        }
    }
}

// We intentionally make DumpConfig take the same arguments as Check so that dumping the
// config is as easy as changing the command name.
#[derive(Debug, Clone, Parser)]
pub struct DumpConfigArgs {
    /// When running `dump-config`, the number of files covered by the found
    /// config(s) to print. Pass "all" to output all files.
    #[arg(long, default_value_t = MaxFiles::Count(10))]
    max_files: MaxFiles,
    #[command(flatten)]
    args: FullCheckArgs,
}

impl DumpConfigArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        // Pass on just the subset of args we use, the rest are irrelevant
        dump_config(self.args.files, self.args.config_override, self.max_files)
    }
}

fn dump_config(
    files: FilesArgs,
    config_override: ConfigOverrideArgs,
    max_files: MaxFiles,
) -> anyhow::Result<CommandExitStatus> {
    config_override.validate()?;
    let (files_to_check, config_finder) = files.resolve(config_override)?;

    let mut configs_to_files: SmallMap<ArcId<ConfigFile>, Vec<ModulePath>> = SmallMap::new();
    let handles = Handles::new(config_finder.checkpoint(files_to_check.files())?);
    let (mut handles, _, sourcedb_errors) = handles.all(&config_finder);
    for error in sourcedb_errors {
        error.print();
    }
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
        let config_env = clap_env("CONFIG");
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
                let config_from = if std::env::var(&config_env)
                    .is_ok_and(|f| &Path::new(&f).absolutize() == path)
                {
                    format!(" (from env {})", config_env)
                } else {
                    "".to_owned()
                };
                println!("Configuration at `{}`{}", path.display(), config_from);
            }
        }
        println!("  Using interpreter: {}", config.interpreters);
        println!("  Covered files:");
        for (i, fi) in files.iter().enumerate() {
            if max_files.should_print(i) {
                println!("    {fi}");
            } else {
                println!("    ...and {} more", max_files.remaining(files.len()));
                break;
            }
        }
        println!("  Resolving imports from:");
        let origin = if files.len() == 1 {
            files.first().map(|p| p.as_path())
        } else {
            None
        };
        for path_part in config.structured_import_lookup_path(origin) {
            if !path_part.is_empty() {
                println!("    {path_part}");
            }
        }
    }
    Ok(CommandExitStatus::Success)
}
