/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! CinderX type report.
//!
//! Produces a flat JSON report with per-expression structured type data,
//! intended for consumption by CinderX's static Python compiler.
//!
//! Output structure:
//! ```text
//! <output_dir>/
//!   index.json              — lists all modules with paths
//!   types/<module>.json      — per-module type table + located types
//!   class_metadata.json      — global class metadata (MRO + semantic tags)
//! ```

pub mod collect;
pub mod convert;
pub(crate) mod display;
pub mod types;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_types::class::Class;
use pyrefly_util::display::Fmt;
use pyrefly_util::fs_anyhow;
use serde::Serialize;

use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::report::cinderx::collect::collect_module_types;
use crate::report::cinderx::convert::canonicalize_class_qname;
use crate::report::cinderx::convert::qname_to_full_string;
use crate::report::cinderx::types::LocatedType;
use crate::report::cinderx::types::TypeTableEntry;
use crate::state::state::Transaction;

/// A module entry in the cinderx index.
#[derive(Debug, Serialize)]
struct ModuleEntry {
    /// Fully-qualified module name.
    module_name: String,
    /// Filesystem path to the source file.
    path: String,
}

/// Top-level index written to `index.json`.
#[derive(Debug, Serialize)]
struct CinderxIndex {
    /// Report format version.
    version: String,
    /// One entry per type-checked module.
    modules: Vec<ModuleEntry>,
}

/// Per-module type report written to `types/<module>.json`.
#[derive(Debug, Serialize)]
struct ModuleReport {
    /// Deduplicated type table; indices in `locations` refer into this.
    type_table: Vec<TypeTableEntry>,
    /// Per-expression type annotations keyed by source location.
    locations: Vec<LocatedType>,
}

/// One entry in the global class metadata table.
#[derive(Debug, Serialize)]
struct ClassMetadataEntry {
    /// Canonicalized qualified name of the class.
    qname: String,
    /// Ancestor classes in method resolution order (excludes `self` and `object`).
    ancestors: Vec<String>,
    /// Semantic tags for this class.
    ///
    /// Current tags:
    /// - `"protocol"` — the class itself is a Protocol
    /// - `"inherits_protocol"` — the class is not a Protocol, but has a
    ///   Protocol ancestor in its runtime MRO (pyrefly excludes Protocol
    ///   from the reported MRO, but at runtime it is present)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
}

/// Global class metadata written to `class_metadata.json`.
#[derive(Debug, Serialize)]
struct ClassMetadataTable {
    /// One entry per unique class encountered in the type report.
    entries: Vec<ClassMetadataEntry>,
}

/// Collect class metadata entries for all classes, deduplicated by qname.
///
/// For each unique class, looks up its MRO and metadata via `KeyClassMro`
/// and `KeyClassMetadata` from the defining module's solutions. Classes
/// whose defining module is not available are silently skipped.
fn collect_class_metadata(
    classes: Vec<Class>,
    transaction: &Transaction,
) -> Vec<ClassMetadataEntry> {
    // Build a handle lookup by (module name, module path) for cross-module
    // resolution. Keying by both name and path avoids collisions when
    // multiple handles exist for the same module name (e.g. a real package
    // and a typeshed stub).
    let handle_by_module: HashMap<_, _> = transaction
        .handles()
        .into_iter()
        .map(|h| ((h.module(), h.path().dupe()), h))
        .collect();

    // Deduplicate classes by canonicalized qname, keeping the first seen.
    let mut seen_qnames: HashMap<String, Class> = HashMap::new();
    for cls in classes {
        let raw = format!("{}", Fmt(|f| cls.qname().fmt_with_module(f)));
        let qname = canonicalize_class_qname(&raw);
        seen_qnames.entry(qname).or_insert(cls);
    }

    let mut entries: Vec<ClassMetadataEntry> = Vec::with_capacity(seen_qnames.len());
    for (qname, cls) in &seen_qnames {
        let key = (cls.module_name(), cls.module_path().dupe());
        let Some(defining_handle) = handle_by_module.get(&key) else {
            continue;
        };
        let Some(solutions) = transaction.get_solutions(defining_handle) else {
            continue;
        };

        // Look up MRO.
        let mro: &Arc<ClassMro> = solutions.get(&KeyClassMro(cls.index()));
        let ancestors: Vec<String> = mro
            .ancestors_no_object()
            .iter()
            .map(|ancestor| {
                let raw = qname_to_full_string(ancestor.qname());
                canonicalize_class_qname(&raw)
            })
            .collect();

        // Look up class metadata for semantic tags.
        let metadata: &Arc<ClassMetadata> = solutions.get(&KeyClassMetadata(cls.index()));
        let mut tags = Vec::new();
        if metadata.is_protocol() {
            tags.push("protocol".to_owned());
        } else if has_protocol_ancestor(mro, &handle_by_module, transaction) {
            tags.push("inherits_protocol".to_owned());
        }

        entries.push(ClassMetadataEntry {
            qname: qname.clone(),
            ancestors,
            tags,
        });
    }

    // Sort by qname for deterministic output.
    entries.sort_by(|a, b| a.qname.cmp(&b.qname));
    entries
}

/// Check whether any ancestor in the MRO is a Protocol.
///
/// This detects classes that aren't themselves Protocols but inherit from one
/// at runtime (e.g. `class Foo(MyProtocol): ...` where `MyProtocol` is a
/// Protocol). Pyrefly excludes `Protocol` from the MRO, but at runtime it is
/// present in `__mro__`.
fn has_protocol_ancestor(
    mro: &ClassMro,
    handle_by_module: &HashMap<(ModuleName, ModulePath), Handle>,
    transaction: &Transaction,
) -> bool {
    for ancestor in mro.ancestors_no_object() {
        let cls = ancestor.class_object();
        let key = (cls.module_name(), cls.module_path().dupe());
        let Some(handle) = handle_by_module.get(&key) else {
            continue;
        };
        let Some(solutions) = transaction.get_solutions(handle) else {
            continue;
        };
        let metadata: &Arc<ClassMetadata> = solutions.get(&KeyClassMetadata(cls.index()));
        if metadata.is_protocol() {
            return true;
        }
    }
    false
}

/// Write a CinderX type report to `output_dir`.
///
/// Reports on all modules in the transaction (project files and dependencies
/// alike), not just the explicitly-checked project modules. This gives the
/// static compiler a complete view of the type information across the entire
/// dependency graph.
///
/// Writes:
/// - `index.json` listing every module
/// - `types/<module>.json` for each module with type table + located types
/// - `class_metadata.json` with global class metadata (MRO + semantic tags)
///
/// When `readable` is true, also writes `types/<module>.txt` alongside each
/// JSON file. The `.txt` format inlines all type-table indices so that each
/// expression location is followed by its fully-resolved type string (no
/// index cross-referencing required). Intended for human debugging; mirrors
/// the output of the `view_types.py` script.
pub fn write_results(
    output_dir: &Path,
    transaction: &Transaction,
    readable: bool,
) -> anyhow::Result<()> {
    fs_anyhow::create_dir_all(output_dir)?;
    let types_dir = output_dir.join("types");
    fs_anyhow::create_dir_all(&types_dir)?;

    let all_handles = transaction.handles();
    let mut modules = Vec::with_capacity(all_handles.len());
    let mut all_classes: Vec<Class> = Vec::new();

    for handle in &all_handles {
        // Some modules (e.g. namespace packages) may not have full type data;
        // skip them rather than failing the entire report.
        let Some(data) = collect_module_types(transaction, handle) else {
            continue;
        };

        modules.push(ModuleEntry {
            module_name: handle.module().to_string(),
            path: handle.path().as_path().display().to_string(),
        });

        all_classes.extend(data.classes);

        let report = ModuleReport {
            type_table: data.entries,
            locations: data.locations,
        };
        let module_json = serde_json::to_string_pretty(&report)?;
        let stem = handle.module().to_string();
        fs_anyhow::write(&types_dir.join(format!("{}.json", stem)), module_json)?;
        if readable {
            let txt = display::format_module_types(&report.type_table, &report.locations);
            fs_anyhow::write(&types_dir.join(format!("{}.txt", stem)), txt)?;
        }
    }

    let index = CinderxIndex {
        version: "0.1".to_owned(),
        modules,
    };
    let json = serde_json::to_string_pretty(&index)?;
    fs_anyhow::write(&output_dir.join("index.json"), json)?;

    // Write global class metadata.
    let class_metadata = ClassMetadataTable {
        entries: collect_class_metadata(all_classes, transaction),
    };
    let metadata_json = serde_json::to_string_pretty(&class_metadata)?;
    fs_anyhow::write(&output_dir.join("class_metadata.json"), metadata_json)?;

    Ok(())
}
