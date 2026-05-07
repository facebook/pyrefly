/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Resolves a `ConfigFile` for project roots that have no `pyrefly.toml`
//! / `[tool.pyrefly]` section. Auto-detects nearby mypy/pyright configs
//! and migrates them in memory, or falls back to the basic preset.

use std::path::Path;

use tracing::debug;

use crate::base::Preset;
use crate::config::ConfigFile;
use crate::config::SynthesizedPresetReason;
use crate::migration::run::find_and_migrate_in_memory;

/// User-facing override for the unconfigured-config resolver. Maps to the
/// values of the `python.pyrefly.typeCheckingMode` VS Code setting plus
/// the implicit `Auto` (which means "let the resolver auto-detect").
///
/// Anything other than `Auto` skips auto-detection entirely and produces
/// an empty config with the corresponding preset.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum UnconfiguredOverride {
    /// Auto-detect from nearby mypy/pyright config; otherwise fall back to
    /// the basic preset.
    #[default]
    Auto,
    /// Force `Off`.
    Off,
    /// Force `Basic`.
    Basic,
    /// Force `Legacy`.
    Legacy,
    /// Force `Default`.
    Default,
    /// Force `Strict`.
    Strict,
}

impl UnconfiguredOverride {
    /// Returns the preset this override forces, or `None` for `Auto`.
    fn explicit_preset(self) -> Option<Preset> {
        match self {
            Self::Auto => None,
            Self::Off => Some(Preset::Off),
            Self::Basic => Some(Preset::Basic),
            Self::Legacy => Some(Preset::Legacy),
            Self::Default => Some(Preset::Default),
            Self::Strict => Some(Preset::Strict),
        }
    }
}

/// Build a `ConfigFile` for a project root that has no `pyrefly.toml` /
/// `[tool.pyrefly]`.
///
/// - Any `over` value other than `Auto` produces an empty `ConfigFile`
///   with that preset and `synthesized_preset_reason = IdeOverride`. The
///   user has explicitly chosen a behavior; we don't auto-detect.
/// - `Auto` searches for a nearby mypy/pyright config and runs the
///   in-memory migration. The migrated result already carries the right
///   preset (`Legacy` for mypy, `None`/Default for pyright); we just
///   stamp the matching `synthesized_preset_reason` on it.
/// - If nothing is found, falls back to `Preset::Basic` /
///   `NoNearbyConfig`.
/// - If migration fails (malformed config), logs at debug and falls back
///   to `Basic` / `NoNearbyConfig`. A broken nearby mypy.ini must not
///   prevent Pyrefly from running.
pub fn resolve_unconfigured_config(start: &Path, over: UnconfiguredOverride) -> ConfigFile {
    if let Some(preset) = over.explicit_preset() {
        return ConfigFile {
            preset: Some(preset),
            project_includes: ConfigFile::default_project_includes(),
            synthesized_preset_reason: Some(SynthesizedPresetReason::IdeOverride),
            ..Default::default()
        };
    }

    match find_and_migrate_in_memory(start) {
        Ok(Some((mut cfg, kind))) => {
            // Migrated mypy configs that didn't set `files` come back
            // with empty `project_includes`; substitute pyrefly's
            // defaults so the project's source files are still
            // discovered.
            if cfg.project_includes.is_empty() {
                cfg.project_includes = ConfigFile::default_project_includes();
            }
            cfg.synthesized_preset_reason = Some(SynthesizedPresetReason::Migrated(kind));
            cfg
        }
        Ok(None) => basic_fallback(),
        Err(e) => {
            debug!(
                "Failed to migrate nearby mypy/pyright config; falling back to Basic preset: {e:#}",
            );
            basic_fallback()
        }
    }
}

fn basic_fallback() -> ConfigFile {
    ConfigFile {
        preset: Some(Preset::Basic),
        project_includes: ConfigFile::default_project_includes(),
        synthesized_preset_reason: Some(SynthesizedPresetReason::NoNearbyConfig),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::fs_anyhow;

    use super::*;
    use crate::migration::run::MigratedConfigSource;
    use crate::migration::run::MigratedFromKind;

    #[test]
    fn test_explicit_override_skips_detection() -> anyhow::Result<()> {
        // Even though a mypy.ini is present, an explicit `Strict` override
        // wins and yields an empty config with that preset and the
        // `IdeOverride` reason.
        let tmp = tempfile::tempdir()?;
        fs_anyhow::write(
            &tmp.path().join("mypy.ini"),
            b"[mypy]\ncheck_untyped_defs = True\n",
        )?;

        let cfg = resolve_unconfigured_config(tmp.path(), UnconfiguredOverride::Strict);
        assert_eq!(cfg.preset, Some(Preset::Strict));
        assert_eq!(
            cfg.synthesized_preset_reason,
            Some(SynthesizedPresetReason::IdeOverride)
        );
        // Explicit overrides skip migration — no mypy values present.
        assert_eq!(cfg.root.check_unannotated_defs, None);
        // Pin the default-globs backfill on the IDE-override path too:
        // Pyrefly must still discover project files when the user picks
        // a preset directly.
        assert!(!cfg.project_includes.is_empty());
        Ok(())
    }

    #[test]
    fn test_off_override() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = resolve_unconfigured_config(tmp.path(), UnconfiguredOverride::Off);
        assert_eq!(cfg.preset, Some(Preset::Off));
        assert_eq!(
            cfg.synthesized_preset_reason,
            Some(SynthesizedPresetReason::IdeOverride)
        );
        assert!(!cfg.project_includes.is_empty());
    }

    #[test]
    fn test_auto_no_nearby_config_falls_back_to_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = resolve_unconfigured_config(tmp.path(), UnconfiguredOverride::Auto);
        assert_eq!(cfg.preset, Some(Preset::Basic));
        assert_eq!(
            cfg.synthesized_preset_reason,
            Some(SynthesizedPresetReason::NoNearbyConfig)
        );
        // Pin the default-globs backfill: Pyrefly must still discover
        // project files even when no nearby config supplies them.
        assert!(!cfg.project_includes.is_empty());
    }

    /// Auto + mypy.ini → resolved config is a *full migration*: not just a
    /// preset choice, but also the migrated mypy options. Pinning the
    /// migrated value catches regressions where someone accidentally drops
    /// the migration and only sets the preset.
    #[test]
    fn test_auto_with_mypy_does_full_migration() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        fs_anyhow::write(
            &tmp.path().join("mypy.ini"),
            b"[mypy]\ncheck_untyped_defs = True\n",
        )?;

        let cfg = resolve_unconfigured_config(tmp.path(), UnconfiguredOverride::Auto);
        assert_eq!(cfg.preset, Some(Preset::Legacy));
        assert_eq!(
            cfg.synthesized_preset_reason,
            Some(SynthesizedPresetReason::Migrated(MigratedFromKind::Mypy(
                MigratedConfigSource::DedicatedFile
            )))
        );
        // Full migration: mypy's `check_untyped_defs = True` flows through
        // to pyrefly's `check_unannotated_defs = Some(true)`.
        assert_eq!(cfg.root.check_unannotated_defs, Some(true));
        // mypy.ini didn't set `files`, so the migrated config has no
        // includes — the backfill must kick in so Pyrefly still
        // discovers project files.
        assert!(!cfg.project_includes.is_empty());
        Ok(())
    }

    #[test]
    fn test_auto_with_pyright_does_full_migration() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        fs_anyhow::write(
            &tmp.path().join("pyrightconfig.json"),
            br#"{ "include": ["src/**/*.py"], "reportMissingImports": "warning" }"#,
        )?;

        let cfg = resolve_unconfigured_config(tmp.path(), UnconfiguredOverride::Auto);
        // Pyright migration leaves preset as None (== Default behavior).
        assert_eq!(cfg.preset, None);
        assert_eq!(
            cfg.synthesized_preset_reason,
            Some(SynthesizedPresetReason::Migrated(
                MigratedFromKind::Pyright(MigratedConfigSource::DedicatedFile)
            ))
        );
        // Pin a migrated value so a regression in pyright's `include`
        // migration would actually fail. `assert!(!is_empty())` alone
        // would still pass because the default-globs backfill fires
        // whether or not migration ran.
        assert!(
            cfg.project_includes
                .globs()
                .iter()
                .any(|g| g.as_str().contains("src/**/*.py")),
            "pyright `include` should have migrated to project_includes; got: {:?}",
            cfg.project_includes,
        );
        Ok(())
    }

    /// Malformed nearby mypy.ini must not break the entire config load —
    /// we log debug and fall back to Basic.
    #[test]
    fn test_auto_with_malformed_mypy_falls_back_to_basic() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        fs_anyhow::write(
            &tmp.path().join("mypy.ini"),
            b"this is not valid ini\nno equals signs\n[unclosed",
        )?;

        let cfg = resolve_unconfigured_config(tmp.path(), UnconfiguredOverride::Auto);
        assert_eq!(cfg.preset, Some(Preset::Basic));
        assert_eq!(
            cfg.synthesized_preset_reason,
            Some(SynthesizedPresetReason::NoNearbyConfig)
        );
        // Pin the backfill on the malformed-config fallback path too.
        assert!(!cfg.project_includes.is_empty());
        Ok(())
    }
}
