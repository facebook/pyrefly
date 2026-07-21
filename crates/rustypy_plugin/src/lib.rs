/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the LICENSE
 * file in the root directory of this source tree.
 */

//! Plugin extension points for rustypy.
//!
//! Upstream pyrefly has no plugin protocol: every framework integration
//! (Django, Pydantic, attrs) is hardcoded at compile time as match-based
//! dispatch inside the type engine. rustypy exposes a small, stable trait
//! surface so new framework integrations can be added without forking the
//! engine internals for each one.
//!
//! The trait mirrors mypy's `Plugin` hook names where the semantics line up,
//! so existing mypy plugins can be ported with minimal friction. Not every
//! mypy hook is exposed yet; the surface grows as integrations demand it.
//!
//! v1 ships a static registry (built-in plugins compiled in). Dynamic
//! loading via `libloading` is a future extension; the trait is `Send +
//! Sync` to support it.

use pyrefly_python::module_name::ModuleName;
use pyrefly_python::qname::QName;
use pyrefly_types::class::ClassKind;

/// A framework integration that augments rustypy's type inference for
/// library-specific patterns the standard typing spec cannot express.
///
/// All hooks are optional; a plugin returns `None` from a hook when it has
/// no opinion, and rustypy falls through to the next plugin or the builtin
/// behavior. Plugins are consulted in registration order; the first
/// non-`None` answer wins.
pub trait Plugin: Send + Sync {
    /// Stable identifier shown in `rustypy plugins list` and used for the
    /// `[[plugin]]` config table. Must be unique within a registry.
    fn name(&self) -> &str;

    /// Classify a class-shaped qname the engine is constructing.
    ///
    /// Return `Some(kind)` to override the default `ClassKind::Class` for a
    /// qname this plugin owns (e.g. an SQLAlchemy `Mapped` marker class).
    /// Return `None` to defer to the next plugin or the builtin table.
    fn class_kind_for_qname(&self, _qname: &QName) -> Option<ClassKind> {
        None
    }

    /// Modules this plugin claims ownership of. Used by `plugins list` and
    /// to short-circuit plugin dispatch when a qname's module is unrelated.
    fn owned_modules(&self) -> &[&str] {
        &[]
    }
}

/// Registry of plugins consulted by the type engine.
///
/// Built once at startup from the static built-in list plus any plugins
/// enabled in `rustypy.toml` (`[[plugin]] name = "..."`). Lookups are
/// linear over the registered plugins; plugin counts are expected to stay
/// in the single digits, so a vector beats a hashmap.
pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginRegistry {
    /// Empty registry. Tests and minimal builds use this.
    pub fn empty() -> Self {
        Self { plugins: Vec::new() }
    }

    /// Build the default registry: every built-in plugin compiled into this
    /// binary. Disabled-by-default plugins are omitted here; they opt in via
    /// config.
    pub fn builtins() -> Self {
        let mut reg = Self::empty();
        for plugin in crate::builtin::default_plugins() {
            reg.register(plugin);
        }
        reg
    }

    /// Register a plugin. Order matters: earlier-registered plugins win on
    /// hook conflicts.
    pub fn register(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    /// Iterate registered plugins.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Plugin> {
        self.plugins.iter().map(|p| p.as_ref())
    }

    /// Look up the `ClassKind` for a qname by consulting every plugin in
    /// order. The first non-`None` answer wins; if no plugin claims the
    /// qname, the caller falls back to the builtin `ClassKind::from_qname`.
    pub fn class_kind_for_qname(&self, qname: &QName) -> Option<ClassKind> {
        for plugin in &self.plugins {
            if let Some(kind) = plugin.class_kind_for_qname(qname) {
                return Some(kind);
            }
        }
        None
    }

    /// Whether any registered plugin claims ownership of `module`.
    pub fn owns_module(&self, module: &ModuleName) -> bool {
        let s = module.as_str();
        self.plugins
            .iter()
            .any(|p| p.owned_modules().contains(&s))
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::builtins()
    }
}

pub mod builtin;

/// Install the built-in plugin registry as the global `ClassKindOverride`.
///
/// The engine calls this once at startup (CLI entry and LSP entry). After the
/// call, every `Class::kind()` lookup consults the registered plugins before
/// falling back to the builtin `ClassKind::from_qname` table.
///
/// Returns `true` if this call installed the hook, `false` if a previous call
/// already did (the first registration wins).
///
/// The registry is intentionally leaked to give it process-lifetime storage
/// matching the `OnceLock` it backs into. It is small (a handful of plugin
/// boxes) and rustypy is a short-lived CLI / long-lived LSP process, so the
/// leak is bounded.
pub fn install_default_override() -> bool {
    let registry: &'static PluginRegistry = Box::leak(Box::new(PluginRegistry::builtins()));
    pyrefly_types::class::set_class_kind_override(Box::new(StaticOverride { registry }))
}

struct StaticOverride {
    registry: &'static PluginRegistry,
}

impl pyrefly_types::class::ClassKindOverride for StaticOverride {
    fn class_kind_for_qname(
        &self,
        qname: &QName,
    ) -> Option<pyrefly_types::class::ClassKind> {
        self.registry.class_kind_for_qname(qname)
    }
}
