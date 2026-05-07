/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use clap::ValueEnum;
use enum_iterator::Sequence;
use enum_iterator::all;
use pyrefly_python::ignore::Tool;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_set::SmallSet;
use toml::Table;

use crate::error::ErrorDisplayConfig;
use crate::error_kind::ErrorKind;
use crate::error_kind::Severity;
use crate::module_wildcard::ModuleWildcard;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Copy, Default)]
#[derive(ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum UntypedDefBehavior {
    #[default]
    CheckAndInferReturnType,
    CheckAndInferReturnAny,
    SkipAndInferReturnAny,
}

/// Controls when Pyrefly infers return types for functions without explicit return annotations.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Copy, Default)]
#[derive(ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum InferReturnTypes {
    /// Never infer return types; unannotated returns are treated as `Any`.
    Never,
    /// Infer return types only for functions with at least one parameter or return annotation.
    Annotated,
    /// Infer return types for all checked functions, including completely unannotated ones.
    #[default]
    Checked,
}

/// How to handle when recursion depth limit is exceeded.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Copy, Default)]
#[derive(ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RecursionOverflowHandler {
    /// Return a placeholder type and emit an internal error. Safe for IDE use.
    #[default]
    BreakWithPlaceholder,
    /// Dump debug info to stderr and panic. For debugging stack overflow issues.
    PanicWithDebugInfo,
}

/// Internal configuration struct combining depth limit and handler.
/// Not serialized directly - constructed from flat config fields.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct RecursionLimitConfig {
    /// Maximum recursion depth before triggering overflow protection.
    pub limit: u32,
    /// How to handle when the depth limit is exceeded.
    pub handler: RecursionOverflowHandler,
}

/// A named collection of error severities and behavior settings that serves as
/// the base configuration. User-specified settings merge on top, overriding
/// the preset. Explicit configuration always wins over the preset regardless
/// of order in the config file.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Copy, Sequence)]
#[derive(ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    /// Silences every error kind. Other settings (scalars, behavior flags) are
    /// left at their defaults. Useful when Pyrefly is running only for IDE
    /// features like hover and go-to-definition, without diagnostics.
    Off,
    /// Minimal checking for LSP users. Raises clear/obvious type errors but
    /// disables stricter checks like override validation and unannotated def checking.
    Basic,
    /// A looser, less-strict preset useful for codebases migrating from mypy.
    /// Pyrefly does not aim to mimic mypy's behavior precisely — this preset
    /// just disables a few checks that mypy does not have, so migrating users
    /// aren't hit with new errors for classes of issues mypy never flagged.
    Legacy,
    /// The default Pyrefly configuration. Equivalent to having no preset at all.
    Default,
    /// Enables additional error codes on top of the default for stricter checking.
    Strict,
}

impl Preset {
    /// Returns a `ConfigBase` carrying this preset's defaults. Only sets fields
    /// the preset explicitly controls — leaves others as `None` so the
    /// per-field defaults in `configure()` still apply. Applied to
    /// `ConfigFile::root` only; sub-configs inherit these values through the
    /// usual root-fallback pattern in the per-field accessors.
    pub fn apply(self) -> ConfigBase {
        match self {
            Preset::Off => {
                // Silence every error kind. Leave all other settings at their
                // defaults so behavior flags still apply — only diagnostics
                // are disabled.
                let errors: HashMap<ErrorKind, Severity> = all::<ErrorKind>()
                    .map(|kind| (kind, Severity::Ignore))
                    .collect();
                ConfigBase {
                    errors: Some(ErrorDisplayConfig::new(errors)),
                    ..Default::default()
                }
            }
            Preset::Basic => {
                // Basic is an opt-in preset: only a small set of high-confidence
                // diagnostics — crashes and clearly broken code — fire. Every
                // other error kind is silenced so unconfigured projects and
                // LSP users see a low-noise baseline.
                let mut errors = HashMap::from([
                    (ErrorKind::DivisionByZero, Severity::Error),
                    (ErrorKind::InvalidSyntax, Severity::Error),
                    (ErrorKind::MissingImport, Severity::Error),
                    (ErrorKind::ParseError, Severity::Error),
                    (ErrorKind::UnexpectedKeyword, Severity::Error),
                    (ErrorKind::UnknownName, Severity::Error),
                    (ErrorKind::InvalidAnnotation, Severity::Error),
                    (ErrorKind::NotAsync, Severity::Error),
                    (ErrorKind::UnusedCoroutine, Severity::Error),
                ]);
                // Silence every other error kind. Explicitly setting each one
                // (rather than relying on `severity()`'s default fallback) is
                // required because the preset's errors map becomes the sole
                // source of truth after merging with user overrides.
                for kind in all::<ErrorKind>() {
                    errors.entry(kind).or_insert(Severity::Ignore);
                }
                ConfigBase {
                    errors: Some(ErrorDisplayConfig::new(errors)),
                    check_unannotated_defs: Some(false),
                    infer_return_types: Some(InferReturnTypes::Never),
                    infer_with_first_use: Some(false),
                    permissive_ignores: Some(true),
                    ..Default::default()
                }
            }
            Preset::Legacy => {
                let errors = HashMap::from([
                    (ErrorKind::BadOverrideMutableAttribute, Severity::Ignore),
                    (ErrorKind::BadOverrideParamName, Severity::Ignore),
                    (ErrorKind::UnboundName, Severity::Ignore),
                ]);
                ConfigBase {
                    errors: Some(ErrorDisplayConfig::new(errors)),
                    check_unannotated_defs: Some(false),
                    infer_return_types: Some(InferReturnTypes::Never),
                    ..Default::default()
                }
            }
            Preset::Default => ConfigBase::default(),
            Preset::Strict => {
                let errors = HashMap::from([
                    (ErrorKind::ImplicitAny, Severity::Error),
                    (ErrorKind::MissingOverrideDecorator, Severity::Error),
                    (ErrorKind::UnusedIgnore, Severity::Error),
                ]);
                ConfigBase {
                    errors: Some(ErrorDisplayConfig::new(errors)),
                    strict_callable_subtyping: Some(true),
                    ..Default::default()
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct ConfigBase {
    /// Errors to silence (or not) when printing errors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errors: Option<ErrorDisplayConfig>,

    /// Consider any ignore (including from other tools) to ignore an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissive_ignores: Option<bool>,

    /// Respect ignore directives from only these tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled_ignores: Option<SmallSet<Tool>>,

    /// Modules from which import errors should be ignored
    /// and the module should always be replaced with `typing.Any`
    #[serde(
        default,
        skip_serializing_if = "crate::util::none_or_empty",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "replace_imports_with_any"
    )]
    pub(crate) replace_imports_with_any: Option<Vec<ModuleWildcard>>,

    /// Modules from which import errors should be
    /// ignored. The module is only replaced with `typing.Any` if it can't be found.
    #[serde(default, skip_serializing_if = "crate::util::none_or_empty")]
    pub(crate) ignore_missing_imports: Option<Vec<ModuleWildcard>>,

    /// Deprecated: use `check-unannotated-defs` and `infer-return-types` instead.
    /// How should we handle analyzing and inferring the function signature if it's untyped?
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "untyped_def_behavior"
    )]
    pub untyped_def_behavior: Option<UntypedDefBehavior>,

    /// Whether to type check the bodies of unannotated function definitions.
    /// Defaults to true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check_unannotated_defs: Option<bool>,

    /// Controls when Pyrefly infers return types for functions without explicit return annotations.
    /// - `never`: unannotated returns are always treated as `Any`.
    /// - `annotated`: infer return types only for functions with at least one annotation.
    /// - `checked`: infer return types for all checked functions (default).
    ///   Only applies to functions whose bodies are checked; unannotated functions
    ///   are only eligible when `check-unannotated-defs` is true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub infer_return_types: Option<InferReturnTypes>,

    /// Whether to disable type errors in language server. By default errors will be shown in IDEs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_type_errors_in_ide: Option<bool>,

    /// Whether to ignore type errors in generated code. By default this is disabled.
    /// Generated code is defined as code that contains the marker string `@` immediately followed by `generated`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "ignore_errors_in_generated_code"
    )]
    pub ignore_errors_in_generated_code: Option<bool>,

    /// Whether to infer empty container types as Any instead of creating type variables.
    /// By default this is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub infer_with_first_use: Option<bool>,

    /// (Experimental) Enable tensor shape type inference.
    /// Supports both native (Tensor[N, M]) and jaxtyping (Float[Tensor, "batch channels"]) syntax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_shapes: Option<bool>,

    /// Maximum recursion depth before triggering overflow protection.
    /// Set to 0 to disable (default). This helps detect potential stack overflow situations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursion_depth_limit: Option<u32>,

    /// How to handle when recursion depth limit is exceeded.
    /// Only used when `recursion-depth-limit` is set to a non-zero value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursion_overflow_handler: Option<RecursionOverflowHandler>,

    /// Whether to strictly check callable subtyping for signatures with `*args: Any, **kwargs: Any`.
    /// When false (the default), callables with `*args: Any, **kwargs: Any` are treated as
    /// compatible with any signature (similar to `...` behavior).
    /// When true, parameter list compatibility is checked strictly even when `*args: Any, **kwargs: Any` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_callable_subtyping: Option<bool>,

    /// Whether to use spec-compliant overload evaluation semantics.
    /// When false (the default), Pyrefly attempts to resolve ambiguous calls precisely.
    /// When true, overload evaluation follows the typing spec exactly, falling back to `Any` more frequently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_compliant_overloads: Option<bool>,

    /// Any unknown config items
    #[serde(default, flatten)]
    pub(crate) extras: ExtraConfigs,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(transparent)]
pub(crate) struct ExtraConfigs(pub(crate) Table);

// `Value` types in `Table` might not be `Eq`, but we don't actually care about that w.r.t. `ConfigFile`
impl Eq for ExtraConfigs {}

impl PartialEq for ExtraConfigs {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl ConfigBase {
    pub fn default_for_ide_without_config() -> Self {
        Self {
            disable_type_errors_in_ide: Some(true),
            ..Default::default()
        }
    }

    /// Resolve the deprecated `untyped_def_behavior` field into the two new fields
    /// (`check_unannotated_defs` and `infer_return_types`).
    /// New fields take precedence; the old field only fills in unset values.
    pub fn resolve_legacy_untyped_def_behavior(&mut self) {
        let Some(behavior) = self.untyped_def_behavior else {
            return;
        };
        if self.check_unannotated_defs.is_none() {
            self.check_unannotated_defs = Some(!matches!(
                behavior,
                UntypedDefBehavior::SkipAndInferReturnAny
            ));
        }
        if self.infer_return_types.is_none() {
            self.infer_return_types = Some(match behavior {
                UntypedDefBehavior::CheckAndInferReturnType => InferReturnTypes::Checked,
                UntypedDefBehavior::CheckAndInferReturnAny
                | UntypedDefBehavior::SkipAndInferReturnAny => InferReturnTypes::Never,
            });
        }
    }

    pub fn get_errors(base: &Self) -> Option<&ErrorDisplayConfig> {
        base.errors.as_ref()
    }

    pub(crate) fn get_replace_imports_with_any(base: &Self) -> Option<&[ModuleWildcard]> {
        base.replace_imports_with_any.as_deref()
    }

    pub(crate) fn get_ignore_missing_imports(base: &Self) -> Option<&[ModuleWildcard]> {
        base.ignore_missing_imports.as_deref()
    }

    pub fn get_check_unannotated_defs(base: &Self) -> Option<bool> {
        base.check_unannotated_defs
    }

    pub fn get_infer_return_types(base: &Self) -> Option<InferReturnTypes> {
        base.infer_return_types
    }

    pub fn get_disable_type_errors_in_ide(base: &Self) -> Option<bool> {
        base.disable_type_errors_in_ide
    }

    pub fn get_ignore_errors_in_generated_code(base: &Self) -> Option<bool> {
        base.ignore_errors_in_generated_code
    }

    pub fn get_infer_with_first_use(base: &Self) -> Option<bool> {
        base.infer_with_first_use
    }

    pub fn get_tensor_shapes(base: &Self) -> Option<bool> {
        base.tensor_shapes
    }

    pub fn get_enabled_ignores(base: &Self) -> Option<&SmallSet<Tool>> {
        base.enabled_ignores.as_ref()
    }

    /// Get the recursion limit configuration, if enabled.
    /// Returns None if recursion_depth_limit is not set or is 0.
    pub fn get_recursion_limit_config(base: &Self) -> Option<RecursionLimitConfig> {
        base.recursion_depth_limit.and_then(|limit| {
            if limit == 0 {
                None
            } else {
                Some(RecursionLimitConfig {
                    limit,
                    handler: base
                        .recursion_overflow_handler
                        .unwrap_or(RecursionOverflowHandler::BreakWithPlaceholder),
                })
            }
        })
    }

    pub fn get_strict_callable_subtyping(base: &Self) -> Option<bool> {
        base.strict_callable_subtyping
    }

    pub fn get_spec_compliant_overloads(base: &Self) -> Option<bool> {
        base.spec_compliant_overloads
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use enum_iterator::all;
    use pulldown_cmark::Event;
    use pulldown_cmark::HeadingLevel;
    use pulldown_cmark::Parser;
    use pulldown_cmark::Tag;
    use pulldown_cmark::TagEnd;

    use super::*;

    /// Canonical kebab-case name for a preset, matching the serde/clap form
    /// (e.g., `StrictPlus` → `"strict-plus"`). Derived from clap's `ValueEnum`
    /// rather than `Debug` so multi-word variants work correctly.
    fn preset_name(preset: Preset) -> String {
        preset
            .to_possible_value()
            .expect("Preset is a ValueEnum")
            .get_name()
            .to_owned()
    }

    /// Verifies that every Preset variant has a corresponding `#### Preset: \`name\``
    /// section in the configuration docs and that the documented error codes match
    /// what `Preset::apply()` actually produces.
    #[test]
    fn test_preset_doc() {
        let doc_path = std::env::var("CONFIG_DOC_PATH")
            .expect("CONFIG_DOC_PATH env var not set: cargo or buck should set this automatically");
        let doc_contents = std::fs::read_to_string(&doc_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        // Parse the doc to collect preset names and their error codes. We only
        // treat an H4 as a preset section if its heading text starts with
        // `Preset:` — that way unrelated H4s elsewhere in the doc can't be
        // mistaken for preset declarations.
        #[derive(Default)]
        struct H4Content {
            text: String,
            code: Option<String>,
        }
        let mut documented_presets: Vec<String> = Vec::new();
        let mut preset_error_codes: HashMap<String, HashSet<String>> = HashMap::new();
        let mut current_preset: Option<String> = None;
        let mut h4_content: Option<H4Content> = None;

        for event in Parser::new(&doc_contents) {
            match event {
                Event::Start(Tag::Heading {
                    level: HeadingLevel::H1 | HeadingLevel::H2 | HeadingLevel::H3,
                    ..
                }) => {
                    // Any higher-level heading ends the current preset section.
                    current_preset = None;
                }
                Event::Start(Tag::Heading {
                    level: HeadingLevel::H4,
                    ..
                }) => {
                    // Entering a new H4 ends any previous preset section and
                    // starts accumulating this heading's content.
                    current_preset = None;
                    h4_content = Some(H4Content::default());
                }
                Event::End(TagEnd::Heading(HeadingLevel::H4)) => {
                    if let Some(content) = h4_content.take()
                        && content.text.trim_start().starts_with("Preset:")
                        && let Some(name) = content.code
                    {
                        documented_presets.push(name.clone());
                        preset_error_codes.entry(name.clone()).or_default();
                        current_preset = Some(name);
                    }
                }
                Event::Text(t) if h4_content.is_some() => {
                    h4_content.as_mut().unwrap().text.push_str(&t);
                }
                Event::Code(c) if h4_content.is_some() => {
                    let content = h4_content.as_mut().unwrap();
                    content.text.push_str(&c);
                    // The first inline code span inside a `Preset: `...`` heading
                    // is the preset name.
                    if content.code.is_none() {
                        content.code = Some(c.to_string());
                    }
                }
                // Collect error code names from links like [bad-override](./error-kinds.mdx#bad-override)
                Event::Start(Tag::Link { dest_url, .. })
                    if current_preset.is_some() && dest_url.contains("error-kinds") =>
                {
                    if let Some(fragment) = dest_url.split('#').nth(1)
                        && let Some(preset_name) = &current_preset
                    {
                        preset_error_codes
                            .entry(preset_name.clone())
                            .or_default()
                            .insert(fragment.to_string());
                    }
                }
                _ => {}
            }
        }

        // Verify every preset variant is documented
        for preset in all::<Preset>() {
            let name = preset_name(preset);
            assert!(
                documented_presets.contains(&name),
                "Preset `{name}` is not documented in {doc_path}. \
                 Add a `#### Preset: \\`{name}\\`` section."
            );
        }

        // Verify no extra presets are documented
        for doc_name in &documented_presets {
            assert!(
                all::<Preset>().any(|p| preset_name(p) == *doc_name),
                "Documentation has preset `{doc_name}` but no such Preset variant exists."
            );
        }

        // Verify documented error codes are consistent with Preset::apply().
        //
        // Direction 1 (doc → code): every documented code must exist as an
        // entry in the preset's errors map. Catches docs referencing a code
        // that the preset doesn't actually touch.
        //
        // Direction 2 (code → doc): every code with a non-Ignore severity must
        // be documented. Catches presets that enable or raise a new error kind
        // without updating the doc. We intentionally skip Ignore-severity
        // entries because opt-in presets like Basic exhaustively set every
        // other kind to Ignore, and documenting all of them would be noise.
        for preset in all::<Preset>() {
            let name = preset_name(preset);
            let config = preset.apply();
            let (all_codes, enabled_codes): (HashSet<String>, HashSet<String>) = config
                .errors
                .as_ref()
                .map(|e| {
                    let all: HashSet<String> =
                        e.iter().map(|(k, _)| k.to_name().to_owned()).collect();
                    let enabled: HashSet<String> = e
                        .iter()
                        .filter(|(_, s)| *s != Severity::Ignore)
                        .map(|(k, _)| k.to_name().to_owned())
                        .collect();
                    (all, enabled)
                })
                .unwrap_or_default();
            let documented_codes = preset_error_codes.get(&name).cloned().unwrap_or_default();

            for code in &documented_codes {
                assert!(
                    all_codes.contains(code),
                    "Preset `{name}`: error code `{code}` is documented in {doc_path} \
                     but not in Preset::apply()."
                );
            }
            for code in &enabled_codes {
                assert!(
                    documented_codes.contains(code),
                    "Preset `{name}`: error code `{code}` is enabled by Preset::apply() \
                     but not documented in {doc_path}."
                );
            }
        }
    }
}
