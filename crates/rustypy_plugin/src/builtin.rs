/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the LICENSE
 * file in the root directory of this source tree.
 */

//! Built-in plugins shipped with rustypy.
//!
//! These fill framework gaps upstream pyrefly leaves open. Each is a small,
//! self-contained `Plugin` impl. The list grows as new integrations are
//! ported; v1 ships a SQLAlchemy 2.0 skeleton and a Celery skeleton so the
//! extension surface is exercised end-to-end. Legacy `@attr.s` already
//! works through pyrefly's PEP 681 path and needs no plugin here.

use pyrefly_python::qname::QName;
use pyrefly_types::class::ClassKind;

use crate::Plugin;

/// Built-in plugins enabled by default. Returned as `Box<dyn Plugin>` so the
/// registry owns them uniformly.
pub fn default_plugins() -> Vec<Box<dyn Plugin>> {
    vec![Box::new(SqlAlchemyPlugin), Box::new(CeleryPlugin)]
}

/// SQLAlchemy 2.0 integration.
///
/// Upstream pyrefly has no SQLAlchemy support: declarative model columns
/// typed as `Mapped[T]` / `mapped_column(T)` are not reflected into the
/// synthesized `__init__`, so passing a `str` to an `int` field goes
/// unchecked. This plugin is the seam for that work. v1 only registers
/// ownership of the `sqlalchemy` module so the engine knows to consult it;
/// the field-type synthesis is wired in subsequent iterations.
pub struct SqlAlchemyPlugin;

impl Plugin for SqlAlchemyPlugin {
    fn name(&self) -> &str {
        "sqlalchemy"
    }

    fn owned_modules(&self) -> &[&str] {
        &["sqlalchemy", "sqlalchemy.orm", "sqlalchemy.sql"]
    }

    fn class_kind_for_qname(&self, qname: &QName) -> Option<ClassKind> {
        // `sqlalchemy.orm.Mapped` and `sqlalchemy.orm.relationship` mark
        // declarative field specifiers, not plain classes. We classify them
        // as dataclass-field-shaped so the synthesis pass treats their
        // annotations as model fields.
        match (qname.module_name().as_str(), qname.id().as_str()) {
            ("sqlalchemy.orm", "Mapped") | ("sqlalchemy.orm", "MappedAsDataclass") => {
                Some(ClassKind::DataclassField)
            }
            _ => None,
        }
    }
}

/// Celery task integration.
///
/// pyrefly and ty both miss Celery: `.delay()` / `.apply_async()` on a
/// `@shared_task`-decorated function accept `*args, **kwargs` of type `Any`,
/// so passing the wrong argument types to a task is unchecked. This plugin
/// is the seam for signature-aware task-call checking. v1 registers
/// ownership only; the method-signature hook lands in a follow-up.
pub struct CeleryPlugin;

impl Plugin for CeleryPlugin {
    fn name(&self) -> &str {
        "celery"
    }

    fn owned_modules(&self) -> &[&str] {
        &["celery", "celery.app.task"]
    }

    fn class_kind_for_qname(&self, _qname: &QName) -> Option<ClassKind> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::PluginRegistry;

    #[test]
    fn registry_returns_builtin_plugins() {
        let reg = PluginRegistry::builtins();
        let names: Vec<&str> = reg.iter().map(|p| p.name()).collect();
        assert!(names.contains(&"sqlalchemy"));
        assert!(names.contains(&"celery"));
    }

    #[test]
    fn empty_registry_has_no_plugins() {
        let reg = PluginRegistry::empty();
        assert_eq!(reg.iter().count(), 0);
    }

    #[test]
    fn owns_module_checks_all_plugins() {
        let reg = PluginRegistry::builtins();
        assert!(reg.owns_module(&pyrefly_python::module_name::ModuleName::from_str(
            "sqlalchemy"
        )));
        assert!(reg.owns_module(&pyrefly_python::module_name::ModuleName::from_str(
            "celery.app.task"
        )));
        assert!(!reg.owns_module(
            &pyrefly_python::module_name::ModuleName::from_str("unrelated_pkg")
        ));
    }
}
