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
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassDefIndex;
use pyrefly_util::fs_anyhow;
use serde::Serialize;

use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::bindings::Bindings;
use crate::report::cinderx::collect::ModuleTypeData;
use crate::report::cinderx::collect::collect_module_types;
use crate::report::cinderx::convert::canonicalize_class_qname;
use crate::report::cinderx::convert::qname_to_full_string;
use crate::report::cinderx::types::LocatedType;
use crate::report::cinderx::types::TypeTableEntry;
use crate::state::state::Transaction;

/// A module entry in the cinderx index.
#[derive(Debug, Serialize)]
#[derive(Clone)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CinderxClassRef {
    module_name: ModuleName,
    module_path: ModulePath,
    class_def_index: ClassDefIndex,
}

impl CinderxClassRef {
    fn from_class(class: &Class) -> Self {
        Self {
            module_name: class.module_name(),
            module_path: class.module_path().dupe(),
            class_def_index: class.index(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CinderxClassInfo {
    qname: String,
    ancestors: Vec<CinderxClassRef>,
    is_protocol: bool,
}

#[derive(Debug)]
pub struct CinderxSolutions {
    classes: HashMap<CinderxClassRef, CinderxClassInfo>,
}

impl CinderxSolutions {
    fn build_class_info(
        class: &Class,
        metadata_is_protocol: bool,
        ancestor_classes: &[Class],
    ) -> CinderxClassInfo {
        CinderxClassInfo {
            qname: canonicalize_class_qname(&qname_to_full_string(class.qname())),
            ancestors: ancestor_classes
                .iter()
                .map(CinderxClassRef::from_class)
                .collect(),
            is_protocol: metadata_is_protocol,
        }
    }

    pub fn build<Ans: LookupAnswer>(
        bindings: &Bindings,
        answers: &AnswersSolver<Ans>,
    ) -> Arc<Self> {
        let classes = bindings
            .keys::<KeyClass>()
            .map(|idx| {
                let class = answers
                    .get_idx(idx)
                    .0
                    .dupe()
                    .expect("class binding must resolve to a class");
                let class_ref = CinderxClassRef::from_class(&class);
                let metadata =
                    answers.get_idx(bindings.key_to_idx(&KeyClassMetadata(class.index())));
                let mro = answers.get_idx(bindings.key_to_idx(&KeyClassMro(class.index())));
                let ancestors: Vec<Class> = mro
                    .ancestors_no_object()
                    .iter()
                    .map(|ancestor| ancestor.class_object().clone())
                    .collect();
                (
                    class_ref,
                    Self::build_class_info(&class, metadata.is_protocol(), &ancestors),
                )
            })
            .collect();
        Arc::new(Self { classes })
    }

    pub fn build_from_answers(bindings: &Bindings, answers: &Answers) -> Arc<Self> {
        let classes = bindings
            .keys::<KeyClass>()
            .map(|idx| {
                let class = answers
                    .get_idx(idx)
                    .expect("class answers must be available for cinderx")
                    .0
                    .dupe()
                    .expect("class binding must resolve to a class");
                let class_ref = CinderxClassRef::from_class(&class);
                let metadata = answers
                    .get_idx(bindings.key_to_idx(&KeyClassMetadata(class.index())))
                    .expect("class metadata answers must be available for cinderx");
                let mro = answers
                    .get_idx(bindings.key_to_idx(&KeyClassMro(class.index())))
                    .expect("class MRO answers must be available for cinderx");
                let ancestors: Vec<Class> = mro
                    .ancestors_no_object()
                    .iter()
                    .map(|ancestor| ancestor.class_object().clone())
                    .collect();
                (
                    class_ref,
                    Self::build_class_info(&class, metadata.is_protocol(), &ancestors),
                )
            })
            .collect();
        Arc::new(Self { classes })
    }

    pub fn get_class_info(&self, class_ref: &CinderxClassRef) -> Option<&CinderxClassInfo> {
        self.classes.get(class_ref)
    }
}

/// Inline writer for CinderX report output during type checking.
pub struct CinderxReporter {
    output_dir: PathBuf,
    types_dir: PathBuf,
    readable: bool,
    report_handles: Option<HashSet<(ModuleName, ModulePath)>>,
    modules: Mutex<HashMap<(ModuleName, ModulePath), ModuleEntry>>,
    classes: Mutex<HashMap<(ModuleName, ModulePath), Vec<CinderxClassRef>>>,
}

impl CinderxReporter {
    pub fn new(
        output_dir: &Path,
        handles: Option<&[Handle]>,
        readable: bool,
    ) -> anyhow::Result<Box<Self>> {
        fs_anyhow::create_dir_all(output_dir)?;
        let types_dir = output_dir.join("types");
        fs_anyhow::create_dir_all(&types_dir)?;
        let report_handles = handles.map(|handles| {
            handles
                .iter()
                .map(|handle| (handle.module(), handle.path().dupe()))
                .collect()
        });
        let capacity = handles.map_or(0, |handles| handles.len());
        Ok(Box::new(Self {
            output_dir: output_dir.to_path_buf(),
            types_dir,
            readable,
            report_handles,
            modules: Mutex::new(HashMap::with_capacity(capacity)),
            classes: Mutex::new(HashMap::with_capacity(capacity)),
        }))
    }

    fn should_report(&self, handle: &Handle) -> bool {
        self.report_handles.as_ref().is_none_or(|report_handles| {
            report_handles.contains(&(handle.module(), handle.path().dupe()))
        })
    }

    fn report_collected_module(&self, handle: &Handle, data: ModuleTypeData) -> anyhow::Result<()> {
        let key = (handle.module(), handle.path().dupe());
        self.modules.lock().unwrap().insert(
            key.clone(),
            ModuleEntry {
                module_name: handle.module().to_string(),
                path: handle.path().as_path().display().to_string(),
            },
        );
        self.classes.lock().unwrap().insert(
            key,
            data.classes
                .iter()
                .map(CinderxClassRef::from_class)
                .collect(),
        );

        let report = ModuleReport {
            type_table: data.entries,
            locations: data.locations,
        };
        let module_json = serde_json::to_string_pretty(&report)?;
        let stem = handle.module().to_string();
        fs_anyhow::write(&self.types_dir.join(format!("{}.json", stem)), module_json)?;
        if self.readable {
            let txt = display::format_module_types(&report.type_table, &report.locations);
            fs_anyhow::write(&self.types_dir.join(format!("{}.txt", stem)), txt)?;
        }
        Ok(())
    }

    /// Write the per-module type report for the given handle, if it is selected.
    pub fn report_module(&self, handle: &Handle, transaction: &Transaction) -> anyhow::Result<()> {
        if !self.should_report(handle) {
            return Ok(());
        }
        let Some(data) = collect_module_types(transaction, handle) else {
            return Ok(());
        };
        self.report_collected_module(handle, data)
    }

    /// Write final project-level CinderX files after all module reports are done.
    pub fn write_project_files(&self, transaction: &Transaction) -> anyhow::Result<()> {
        let mut modules: Vec<_> = self.modules.lock().unwrap().values().cloned().collect();
        modules.sort_by(|a, b| a.module_name.cmp(&b.module_name).then(a.path.cmp(&b.path)));
        let index = CinderxIndex {
            version: "0.1".to_owned(),
            modules,
        };
        let json = serde_json::to_string_pretty(&index)?;
        fs_anyhow::write(&self.output_dir.join("index.json"), json)?;

        let class_metadata = ClassMetadataTable {
            entries: collect_class_metadata(
                self.classes
                    .lock()
                    .unwrap()
                    .values()
                    .flatten()
                    .cloned()
                    .collect(),
                transaction,
            ),
        };
        let metadata_json = serde_json::to_string_pretty(&class_metadata)?;
        fs_anyhow::write(&self.output_dir.join("class_metadata.json"), metadata_json)?;
        Ok(())
    }
}

fn get_class_info(
    class_ref: &CinderxClassRef,
    handle_by_module: &HashMap<(ModuleName, ModulePath), Handle>,
    transaction: &Transaction,
) -> Option<CinderxClassInfo> {
    let key = (class_ref.module_name, class_ref.module_path.dupe());
    let handle = handle_by_module.get(&key)?;
    let solutions = transaction.resolve_cinderx_solutions(handle);
    solutions.get_class_info(class_ref).cloned()
}

/// Collect class metadata entries for all classes, deduplicated by qname.
fn collect_class_metadata(
    classes: Vec<CinderxClassRef>,
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
    let mut seen_qnames: HashMap<String, CinderxClassRef> = HashMap::new();
    for class_ref in classes {
        let Some(class_info) = get_class_info(&class_ref, &handle_by_module, transaction) else {
            continue;
        };
        seen_qnames.entry(class_info.qname).or_insert(class_ref);
    }

    let mut entries: Vec<ClassMetadataEntry> = Vec::with_capacity(seen_qnames.len());
    for (qname, class_ref) in &seen_qnames {
        let Some(class_info) = get_class_info(class_ref, &handle_by_module, transaction) else {
            continue;
        };
        let ancestors: Vec<String> = class_info
            .ancestors
            .iter()
            .filter_map(|ancestor| {
                get_class_info(ancestor, &handle_by_module, transaction).map(|info| info.qname)
            })
            .collect();

        let mut tags = Vec::new();
        if class_info.is_protocol {
            tags.push("protocol".to_owned());
        } else if has_protocol_ancestor(&class_info.ancestors, &handle_by_module, transaction) {
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
    ancestors: &[CinderxClassRef],
    handle_by_module: &HashMap<(ModuleName, ModulePath), Handle>,
    transaction: &Transaction,
) -> bool {
    for ancestor in ancestors {
        let Some(class_info) = get_class_info(ancestor, handle_by_module, transaction) else {
            continue;
        };
        if class_info.is_protocol {
            return true;
        }
    }
    false
}

/// Write a CinderX type report to `output_dir`.
///
/// `handles` controls which modules appear in the report. Pass the
/// explicitly-checked project handles for a project-only report, or
/// `&transaction.handles()` to include all transitive dependencies.
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
#[cfg(test)]
pub fn write_results(
    output_dir: &Path,
    transaction: &Transaction,
    handles: &[Handle],
    readable: bool,
) -> anyhow::Result<()> {
    let reporter = CinderxReporter::new(output_dir, Some(handles), readable)?;
    for handle in handles {
        reporter.report_module(handle, transaction)?;
    }
    reporter.write_project_files(transaction)
}
