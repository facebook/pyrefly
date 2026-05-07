// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Type definitions and derivation logic for the custom
//! `pyrefly/textDocument/typeErrorDisplayStatus` LSP request.
//!
//! The status bar UI is versioned: the client declares which wire shape it
//! can parse via `initializationOptions.pyrefly.typeErrorDisplayStatusVersion`,
//! and the server responds with the negotiated shape. V1 is the legacy
//! bare-string format; V2 is the richer struct (label, tooltip, docs link)
//! used by the current status-bar implementation.

use lsp_types::TextDocumentIdentifier;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::config::ConfigSource;
use pyrefly_config::config::SynthesizedPresetReason;
use pyrefly_config::migration::run::MigratedConfigSource;
use pyrefly_config::migration::run::MigratedFromKind;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::state::lsp::TypeCheckingMode;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TypeErrorDisplayStatus {
    DisabledInIdeConfig,
    EnabledInIdeConfig,
    DisabledInConfigFile,
    EnabledInConfigFile,
    NoConfigFile,
}

/// Versioned wire shape for the custom
/// `pyrefly/textDocument/typeErrorDisplayStatus` LSP request. The client
/// declares which version it can parse via
/// `initializationOptions.pyrefly.typeErrorDisplayStatusVersion`. The
/// server resolves the requested value as follows: missing / null
/// becomes `V1` (the legacy bare-string shape, the safest default for
/// clients that didn't opt in); a known version is honored as-is; an
/// unknown future version is clamped to `LATEST` (the newest variant
/// this server knows about), since the client opted into a richer
/// shape and falling back to V1 would silently strip off a feature it
/// asked for.
///
/// We use an enum so the variants are exhaustive at the type level —
/// adding a new wire shape is a Rust-side change, not a magic-number
/// bump.
#[derive(Clone, Copy, Debug, Deserialize, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TypeErrorDisplayStatusVersion {
    /// Legacy bare-string response (the original
    /// `TypeErrorDisplayStatus` enum). Returned when the client doesn't
    /// declare a version or declares `v1`.
    #[default]
    V1,
    /// Rich response: `{ version: "v2", label, tooltip, docsUrl }`.
    V2,
}

impl TypeErrorDisplayStatusVersion {
    /// The latest version this server can produce. A client that asks
    /// for an unknown future version (e.g. `v3` from a VSIX newer than
    /// the binary) gets clamped to this — the closest thing we can
    /// offer to what the client opted into.
    pub const LATEST: Self = Self::V2;
}

/// Resolve the wire shape from
/// `initializationOptions.pyrefly.typeErrorDisplayStatusVersion`:
///
/// - field absent or explicit JSON `null` → [`V1`](TypeErrorDisplayStatusVersion::V1)
///   (the client didn't opt in; it almost certainly only knows the
///   legacy bare-string shape).
/// - field is a known kebab-case version (`"v1"`, `"v2"`) → that
///   version.
/// - field is an unknown future version (`"v3"` from a newer VSIX) →
///   [`LATEST`](TypeErrorDisplayStatusVersion::LATEST). The client
///   explicitly opted into a richer shape, so V1 would silently strip
///   off a feature they asked for.
///
/// Pulled out as a free function so the negotiation logic is unit-
/// testable without constructing a full `InitializeParams`.
pub fn negotiate_type_error_display_status_version(
    initialization_options: Option<&Value>,
) -> TypeErrorDisplayStatusVersion {
    initialization_options
        .and_then(|opts| opts.get("pyrefly"))
        .and_then(|pyrefly| pyrefly.get("typeErrorDisplayStatusVersion"))
        .filter(|v| !v.is_null())
        .map(|v| {
            serde_json::from_value::<TypeErrorDisplayStatusVersion>(v.clone())
                .unwrap_or(TypeErrorDisplayStatusVersion::LATEST)
        })
        .unwrap_or_default()
}

/// V2 wire shape for the status-bar response. `label` drives the
/// status-bar parenthetical (`Pyrefly (Basic)`, `Pyrefly (Legacy)`,
/// …); `null` means show plain `Pyrefly`. `tooltip` is markdown.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TypeErrorDisplayStatusV2 {
    /// Always `"v2"`. Lets the client dispatch on `response.version`
    /// once it has decoded the response as an object.
    pub version: String,
    /// Preset name to render in parentheses, or `None` for plain
    /// `Pyrefly`. The parenthetical is only shown when the preset was
    /// chosen automatically — if the user explicitly set
    /// `typeCheckingMode` themselves, this is `None` because the user
    /// already knows what they picked.
    pub label: Option<String>,
    /// Markdown tooltip explaining the current state.
    pub tooltip: String,
    /// URL referenced from the tooltip — the IDE typically renders this
    /// as the trailing "Docs" link in the hover.
    pub docs_url: String,
}

/// Internal sum type covering both wire shapes. `#[serde(untagged)]`
/// dispatches by shape: V1 deserializes from / serializes to the V1
/// kebab-case bare string, V2 from / to the V2 struct.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TypeErrorDisplayStatusResponse {
    V1(TypeErrorDisplayStatus),
    V2(TypeErrorDisplayStatusV2),
}

/// Type-level binding for the custom
/// `pyrefly/textDocument/typeErrorDisplayStatus` LSP request. Mirrors
/// the `lsp_types::request::Request` pattern used by stock methods
/// (e.g. `Completion`, `ProvideType`) so the wire method name lives in
/// one place — call sites reference `TypeErrorDisplayStatusRequest::METHOD`
/// instead of duplicating the string literal.
pub enum TypeErrorDisplayStatusRequest {}

impl lsp_types::request::Request for TypeErrorDisplayStatusRequest {
    type Params = TextDocumentIdentifier;
    type Result = TypeErrorDisplayStatusResponse;
    const METHOD: &'static str = "pyrefly/textDocument/typeErrorDisplayStatus";
}

/// URL referenced from the V2 tooltip / docs link. Module-level so the
/// derivation logic and tests share the exact string the user sees.
const STATUS_BAR_DOCS_URL: &str = "https://pyrefly.org/en/docs/IDE/";

/// Silent V2 response — plain `Pyrefly` (no parenthetical), no
/// tooltip. Used for fallback cases where there is nothing useful to
/// surface: the requested file isn't covered by any handle (no path
/// resolves), or a non-`File` config source has no synthesized preset
/// reason (a configured project shouldn't be nudged).
///
/// The onboarding nudge for genuine no-config states lives in the
/// `Some(SynthesizedPresetReason::NoNearbyConfig)` branch of
/// `derive_v2_response` (with `Basic` label + `pyrefly init` tooltip).
pub fn default_v2_response() -> TypeErrorDisplayStatusV2 {
    TypeErrorDisplayStatusV2 {
        version: "v2".to_owned(),
        label: None,
        tooltip: String::new(),
        docs_url: STATUS_BAR_DOCS_URL.to_owned(),
    }
}

/// Pure derivation of the V2 status-bar payload. Factored out of
/// `Server::type_error_display_status_v2` so it can be unit-tested
/// against synthetic inputs without standing up an LSP transport.
///
/// Precedence (mirrors the LSP filter): the workspace kill switch is
/// checked first, then the per-config-source rules. Both kill-switch
/// branches surface the same `Errors Off` label so users learn the
/// state from the status bar itself; the tooltip is the place to
/// distinguish workspace-level from config-level disable.
pub fn derive_v2_response(
    reason: Option<SynthesizedPresetReason>,
    source: &ConfigSource,
    disable_type_errors_in_ide: bool,
    workspace_disable_type_errors: bool,
    workspace_type_checking_mode: Option<TypeCheckingMode>,
) -> TypeErrorDisplayStatusV2 {
    if workspace_disable_type_errors {
        return TypeErrorDisplayStatusV2 {
            version: "v2".to_owned(),
            label: Some("Errors Off".to_owned()),
            tooltip:
                "Pyrefly diagnostics are suppressed by [`python.pyrefly.disableTypeErrors`](command:workbench.action.openSettings?[\"python.pyrefly.disableTypeErrors\"]).\n\nUnset this setting to re-enable diagnostics."
                    .to_owned(),
            docs_url: STATUS_BAR_DOCS_URL.to_owned(),
        };
    }
    match reason {
        Some(SynthesizedPresetReason::IdeOverride) => {
            // The IdeOverride reason is set by the unconfigured resolver
            // when the user explicitly chose a non-`Auto` value for the
            // `python.pyrefly.typeCheckingMode` workspace setting AND no
            // nearby `pyrefly.toml` was found, so we surface both facts.
            // Fall back to `<unknown>` only if the workspace state and
            // the reason somehow disagree — shouldn't happen in practice.
            let value = workspace_type_checking_mode
                .map(type_checking_mode_kebab)
                .unwrap_or("<unknown>");
            TypeErrorDisplayStatusV2 {
                version: "v2".to_owned(),
                label: None,
                tooltip: format!(
                    "Pyrefly is using the [`python.pyrefly.typeCheckingMode`](command:workbench.action.openSettings?[\"python.pyrefly.typeCheckingMode\"]) setting (currently: `{value}`) because no `pyrefly.toml` was found.\n\nRun `pyrefly init` to continue setting up Pyrefly.",
                ),
                docs_url: STATUS_BAR_DOCS_URL.to_owned(),
            }
        }
        Some(SynthesizedPresetReason::Migrated(kind)) => {
            let (location, label, preset) = match kind {
                MigratedFromKind::Mypy(MigratedConfigSource::DedicatedFile) => {
                    ("your `mypy.ini`", "Legacy", "legacy")
                }
                MigratedFromKind::Mypy(MigratedConfigSource::PyprojectToml) => (
                    "`[tool.mypy]` in your `pyproject.toml`",
                    "Legacy",
                    "legacy",
                ),
                MigratedFromKind::Pyright(MigratedConfigSource::DedicatedFile) => {
                    ("your `pyrightconfig.json`", "Default", "default")
                }
                MigratedFromKind::Pyright(MigratedConfigSource::PyprojectToml) => (
                    "`[tool.pyright]` in your `pyproject.toml`",
                    "Default",
                    "default",
                ),
            };
            TypeErrorDisplayStatusV2 {
                version: "v2".to_owned(),
                label: Some(label.to_owned()),
                tooltip: format!(
                    "Pyrefly is using settings imported from {location} (preset: {preset}).\n\nRun `pyrefly init` to continue setting up Pyrefly.",
                ),
                docs_url: STATUS_BAR_DOCS_URL.to_owned(),
            }
        }
        Some(SynthesizedPresetReason::NoNearbyConfig) => TypeErrorDisplayStatusV2 {
            version: "v2".to_owned(),
            label: Some("Basic".to_owned()),
            tooltip:
                "Pyrefly is running with the `basic` preset because no `pyrefly.toml` was found.\n\nRun `pyrefly init` to continue setting up Pyrefly."
                    .to_owned(),
            docs_url: STATUS_BAR_DOCS_URL.to_owned(),
        },
        None => match source {
            ConfigSource::File(path) if disable_type_errors_in_ide => {
                // The in-config disable lives at one of two paths: a
                // dedicated `pyrefly.toml` (the `disable-type-errors-in-ide`
                // key sits at the top level), or `[tool.pyrefly]` inside
                // a `pyproject.toml` (the key sits inside that section).
                // Tooltip distinguishes so users know what file to open.
                let location = if path
                    .file_name()
                    .is_some_and(|n| n == ConfigFile::PYPROJECT_FILE_NAME)
                {
                    "`[tool.pyrefly]` in this project's `pyproject.toml`"
                } else {
                    "this project's `pyrefly.toml`"
                };
                TypeErrorDisplayStatusV2 {
                    version: "v2".to_owned(),
                    label: Some("Errors Off".to_owned()),
                    tooltip: format!(
                        "Pyrefly diagnostics are suppressed by `disable-type-errors-in-ide` in {location}.\n\nRemove this config to re-enable diagnostics.",
                    ),
                    docs_url: STATUS_BAR_DOCS_URL.to_owned(),
                }
            }
            ConfigSource::File(_) => TypeErrorDisplayStatusV2 {
                version: "v2".to_owned(),
                label: None,
                tooltip: String::new(),
                docs_url: STATUS_BAR_DOCS_URL.to_owned(),
            },
            _ => default_v2_response(),
        },
    }
}

/// Kebab-case spelling of a `TypeCheckingMode`. Matches the
/// `#[serde(rename_all = "kebab-case")]` form the user types in
/// `settings.json`, so the value we surface in tooltips is exactly what
/// they would set the configuration key to.
fn type_checking_mode_kebab(mode: TypeCheckingMode) -> &'static str {
    match mode {
        TypeCheckingMode::Auto => "auto",
        TypeCheckingMode::Off => "off",
        TypeCheckingMode::Basic => "basic",
        TypeCheckingMode::Legacy => "legacy",
        TypeCheckingMode::Default => "default",
        TypeCheckingMode::Strict => "strict",
    }
}

impl TypeErrorDisplayStatus {
    pub fn is_enabled(self) -> bool {
        match self {
            TypeErrorDisplayStatus::DisabledInIdeConfig
            | TypeErrorDisplayStatus::DisabledInConfigFile => false,
            TypeErrorDisplayStatus::EnabledInIdeConfig
            | TypeErrorDisplayStatus::EnabledInConfigFile
            | TypeErrorDisplayStatus::NoConfigFile => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit tests for the V2 status-bar response derivation. The test
    /// matrix covers the four `SynthesizedPresetReason` cases plus the
    /// configured-file branches. The full LSP integration (parsing the
    /// version from init options, dispatching the request) is exercised
    /// by the existing notebook_type_error_display_status tests.
    mod v2_response {
        use std::path::PathBuf;

        use pyrefly_config::config::ConfigSource;
        use pyrefly_config::config::SynthesizedPresetReason;
        use pyrefly_config::migration::run::MigratedConfigSource;
        use pyrefly_config::migration::run::MigratedFromKind;

        use super::super::TypeErrorDisplayStatusVersion;
        use super::super::derive_v2_response;
        use crate::state::lsp::TypeCheckingMode;

        #[test]
        fn ide_override_yields_null_label() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::IdeOverride),
                &ConfigSource::Synthetic,
                false,
                false,
                Some(TypeCheckingMode::Strict),
            );
            assert_eq!(r.label, None);
            assert!(r.tooltip.contains("typeCheckingMode"));
            // The current value (`strict` in kebab-case) is rendered so
            // the user sees what the setting is currently set to.
            assert!(
                r.tooltip.contains("currently: `strict`"),
                "tooltip should embed the active `typeCheckingMode` value, got: {}",
                r.tooltip
            );
        }

        #[test]
        fn migrated_from_mypy_ini_yields_legacy_label() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::Migrated(MigratedFromKind::Mypy(
                    MigratedConfigSource::DedicatedFile,
                ))),
                &ConfigSource::Synthetic,
                false,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Legacy"));
            assert!(r.tooltip.contains("your `mypy.ini`"));
            assert!(r.tooltip.contains("pyrefly init"));
        }

        #[test]
        fn migrated_from_mypy_pyproject_yields_legacy_label() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::Migrated(MigratedFromKind::Mypy(
                    MigratedConfigSource::PyprojectToml,
                ))),
                &ConfigSource::Synthetic,
                false,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Legacy"));
            assert!(r.tooltip.contains("`[tool.mypy]` in your `pyproject.toml`"));
            assert!(!r.tooltip.contains("your `mypy.ini`"));
        }

        #[test]
        fn migrated_from_pyrightconfig_yields_default_label() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::Migrated(
                    MigratedFromKind::Pyright(MigratedConfigSource::DedicatedFile),
                )),
                &ConfigSource::Synthetic,
                false,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Default"));
            assert!(r.tooltip.contains("your `pyrightconfig.json`"));
        }

        #[test]
        fn migrated_from_pyright_pyproject_yields_default_label() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::Migrated(
                    MigratedFromKind::Pyright(MigratedConfigSource::PyprojectToml),
                )),
                &ConfigSource::Synthetic,
                false,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Default"));
            assert!(
                r.tooltip
                    .contains("`[tool.pyright]` in your `pyproject.toml`")
            );
            assert!(!r.tooltip.contains("your `pyrightconfig.json`"));
        }

        #[test]
        fn no_nearby_config_yields_basic_label() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::NoNearbyConfig),
                &ConfigSource::Synthetic,
                false,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Basic"));
            assert!(r.tooltip.contains("basic"));
            assert!(r.tooltip.contains("pyrefly init"));
        }

        /// A real config file with errors enabled → no parenthetical, no
        /// onboarding tooltip. The existence of a `pyrefly.toml` means
        /// the user is already configured; nudging them is noise.
        #[test]
        fn configured_file_yields_null_label_no_tooltip() {
            let r = derive_v2_response(
                None,
                &ConfigSource::File(PathBuf::from("/proj/pyrefly.toml")),
                false,
                false,
                None,
            );
            assert_eq!(r.label, None);
            assert!(r.tooltip.is_empty());
        }

        /// `disable_type_errors_in_ide` set in a dedicated `pyrefly.toml`
        /// → status bar shows `Pyrefly (Errors Off)` with a tooltip
        /// pointing at `disable-type-errors-in-ide` in the project's
        /// `pyrefly.toml`.
        #[test]
        fn disabled_in_pyrefly_toml_yields_errors_off_label() {
            let r = derive_v2_response(
                None,
                &ConfigSource::File(PathBuf::from("/proj/pyrefly.toml")),
                true,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Errors Off"));
            assert!(r.tooltip.contains("disable-type-errors-in-ide"));
            assert!(r.tooltip.contains("this project's `pyrefly.toml`"));
            assert!(
                !r.tooltip.contains("[tool.pyrefly]"),
                "dedicated-file tooltip shouldn't mention the pyproject form: {}",
                r.tooltip
            );
        }

        /// Same `disable-type-errors-in-ide` mechanism but the config
        /// lives in a `pyproject.toml` `[tool.pyrefly]` section. The
        /// tooltip distinguishes so users know which file to open.
        #[test]
        fn disabled_in_pyproject_toml_yields_errors_off_label() {
            let r = derive_v2_response(
                None,
                &ConfigSource::File(PathBuf::from("/proj/pyproject.toml")),
                true,
                false,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Errors Off"));
            assert!(r.tooltip.contains("disable-type-errors-in-ide"));
            assert!(
                r.tooltip
                    .contains("`[tool.pyrefly]` in this project's `pyproject.toml`"),
                "pyproject tooltip should distinguish from the dedicated-file form: {}",
                r.tooltip
            );
        }

        /// Workspace `disableTypeErrors = true` is the kill switch.
        /// Status bar shows `Pyrefly (Errors Off)`; tooltip points the
        /// user at the workspace setting (not the config) so they
        /// know which knob to flip.
        #[test]
        fn workspace_kill_switch_yields_errors_off_label() {
            let r = derive_v2_response(None, &ConfigSource::Synthetic, false, true, None);
            assert_eq!(r.label.as_deref(), Some("Errors Off"));
            assert!(r.tooltip.contains("python.pyrefly.disableTypeErrors"));
        }

        /// Workspace kill switch wins over every other branch — even
        /// when a `SynthesizedPresetReason` would otherwise pick a
        /// preset label or the in-config disable would point at the
        /// config.
        #[test]
        fn workspace_kill_switch_wins_over_preset_reason() {
            let r = derive_v2_response(
                Some(SynthesizedPresetReason::NoNearbyConfig),
                &ConfigSource::Synthetic,
                false,
                true,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Errors Off"));
            assert!(r.tooltip.contains("python.pyrefly.disableTypeErrors"));
        }

        #[test]
        fn workspace_kill_switch_wins_over_in_config_disable() {
            let r = derive_v2_response(
                None,
                &ConfigSource::File(PathBuf::from("/proj/pyrefly.toml")),
                true,
                true,
                None,
            );
            assert_eq!(r.label.as_deref(), Some("Errors Off"));
            assert!(r.tooltip.contains("python.pyrefly.disableTypeErrors"));
            assert!(
                !r.tooltip.contains("disable-type-errors-in-ide"),
                "workspace tooltip should not mention the in-config flag when it's the workspace setting that's responsible"
            );
        }

        /// Forward-compat regression: `TypeErrorDisplayStatusVersion`
        /// must round-trip through serde with the kebab-case spelling
        /// the protocol uses. If a future client sends `"v3"` and the
        /// server doesn't know it yet, deserialization fails — the
        /// init-time negotiator catches that and clamps to
        /// [`LATEST`](TypeErrorDisplayStatusVersion::LATEST), since the
        /// client explicitly opted into a richer shape and V1 would
        /// silently strip off a feature they asked for.
        #[test]
        fn version_unknown_value_fails_to_deserialize() {
            let v: Result<TypeErrorDisplayStatusVersion, _> =
                serde_json::from_value(serde_json::json!("v3"));
            assert!(v.is_err());
        }

        /// Pin the clamping target so a future commit that adds `V3`
        /// remembers to bump `LATEST`.
        #[test]
        fn version_latest_is_v2() {
            assert_eq!(
                TypeErrorDisplayStatusVersion::LATEST,
                TypeErrorDisplayStatusVersion::V2
            );
        }

        #[test]
        fn version_v1_v2_round_trip() {
            let v1: TypeErrorDisplayStatusVersion =
                serde_json::from_value(serde_json::json!("v1")).unwrap();
            assert_eq!(v1, TypeErrorDisplayStatusVersion::V1);
            let v2: TypeErrorDisplayStatusVersion =
                serde_json::from_value(serde_json::json!("v2")).unwrap();
            assert_eq!(v2, TypeErrorDisplayStatusVersion::V2);
        }

        /// Negotiation pinned end-to-end against the documented contract:
        /// missing or explicit `null` → V1; known version → that version;
        /// unknown future version → LATEST.
        mod negotiate {
            use super::super::super::TypeErrorDisplayStatusVersion;
            use super::super::super::negotiate_type_error_display_status_version;

            #[test]
            fn missing_initialization_options_resolves_to_v1() {
                assert_eq!(
                    negotiate_type_error_display_status_version(None),
                    TypeErrorDisplayStatusVersion::V1
                );
            }

            #[test]
            fn missing_pyrefly_namespace_resolves_to_v1() {
                let opts = serde_json::json!({});
                assert_eq!(
                    negotiate_type_error_display_status_version(Some(&opts)),
                    TypeErrorDisplayStatusVersion::V1
                );
            }

            #[test]
            fn missing_field_resolves_to_v1() {
                let opts = serde_json::json!({ "pyrefly": {} });
                assert_eq!(
                    negotiate_type_error_display_status_version(Some(&opts)),
                    TypeErrorDisplayStatusVersion::V1
                );
            }

            /// Explicit JSON `null` must resolve to V1, not LATEST.
            /// Without the `.filter(|v| !v.is_null())` guard, deserialize
            /// fails and the `.unwrap_or(LATEST)` branch clamps `null` to
            /// V2 — silently upgrading older clients that send an
            /// explicit `null` instead of omitting the field.
            #[test]
            fn explicit_null_resolves_to_v1() {
                let opts =
                    serde_json::json!({ "pyrefly": { "typeErrorDisplayStatusVersion": null } });
                assert_eq!(
                    negotiate_type_error_display_status_version(Some(&opts)),
                    TypeErrorDisplayStatusVersion::V1
                );
            }

            #[test]
            fn known_v1_resolves_to_v1() {
                let opts =
                    serde_json::json!({ "pyrefly": { "typeErrorDisplayStatusVersion": "v1" } });
                assert_eq!(
                    negotiate_type_error_display_status_version(Some(&opts)),
                    TypeErrorDisplayStatusVersion::V1
                );
            }

            #[test]
            fn known_v2_resolves_to_v2() {
                let opts =
                    serde_json::json!({ "pyrefly": { "typeErrorDisplayStatusVersion": "v2" } });
                assert_eq!(
                    negotiate_type_error_display_status_version(Some(&opts)),
                    TypeErrorDisplayStatusVersion::V2
                );
            }

            /// A version newer than this server knows about clamps to
            /// LATEST — the client opted into a richer shape, so falling
            /// back to V1 would silently strip off a feature.
            #[test]
            fn unknown_future_version_clamps_to_latest() {
                let opts =
                    serde_json::json!({ "pyrefly": { "typeErrorDisplayStatusVersion": "v3" } });
                assert_eq!(
                    negotiate_type_error_display_status_version(Some(&opts)),
                    TypeErrorDisplayStatusVersion::LATEST
                );
            }
        }
    }
}
