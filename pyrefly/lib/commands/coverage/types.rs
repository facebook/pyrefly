/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ignore::Tool;
use pyrefly_types::callable::PropertyRole;
use ruff_text_size::TextRange;
use serde::Serialize;

/// Slot-level annotation counts for a symbol.
///
/// A "slot" is a single annotation site: a function return type, a function
/// parameter, or a module-level variable. Each slot falls into exactly one of
/// three buckets: typed (concrete annotation with no `Any`), any (annotation
/// that resolves to or contains `Any`), or untyped (no annotation at all).
/// `n_typable` is always the sum of the other three.
#[derive(Debug, Serialize, Default, Clone, Copy)]
pub struct SlotCounts {
    /// Total number of annotation sites.
    pub n_typable: usize,
    /// Sites with a concrete annotation containing no `Any`.
    pub n_typed: usize,
    /// Sites annotated but whose resolved type contains `Any`.
    pub n_any: usize,
    /// Sites with no annotation at all.
    pub n_untyped: usize,
}

impl SlotCounts {
    pub fn merge(self, other: SlotCounts) -> SlotCounts {
        SlotCounts {
            n_typable: self.n_typable + other.n_typable,
            n_typed: self.n_typed + other.n_typed,
            n_any: self.n_any + other.n_any,
            n_untyped: self.n_untyped + other.n_untyped,
        }
    }

    pub fn typed() -> SlotCounts {
        SlotCounts {
            n_typable: 1,
            n_typed: 1,
            n_any: 0,
            n_untyped: 0,
        }
    }

    pub fn any() -> SlotCounts {
        SlotCounts {
            n_typable: 1,
            n_typed: 0,
            n_any: 1,
            n_untyped: 0,
        }
    }

    pub fn untyped() -> SlotCounts {
        SlotCounts {
            n_typable: 1,
            n_typed: 0,
            n_any: 0,
            n_untyped: 1,
        }
    }

    /// Coverage: (n_typed + n_any) / n_typable. Treats Any-annotated slots as covered.
    pub fn coverage(&self) -> f64 {
        if self.n_typable == 0 {
            100.0
        } else {
            ((self.n_typed + self.n_any) as f64 / self.n_typable as f64) * 100.0
        }
    }

    /// Strict coverage: n_typed / n_typable. Only concrete types count.
    pub fn strict_coverage(&self) -> f64 {
        if self.n_typable == 0 {
            100.0
        } else {
            (self.n_typed as f64 / self.n_typable as f64) * 100.0
        }
    }
}

/// Annotation quality for a single slot, ordered so that `max()` gives "best-wins" semantics across
/// overloads. `Skip` is the neutral element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlotRank {
    Skip,
    Untyped,
    Any,
    Typed,
}

impl SlotRank {
    /// Map (has_annotation, is_type_known) to a rank.
    pub fn classify(has_annotation: bool, is_type_known: bool) -> Self {
        match (has_annotation, is_type_known) {
            (false, _) => SlotRank::Untyped,
            (true, true) => SlotRank::Typed,
            (true, false) => SlotRank::Any,
        }
    }
}

impl From<&Parameter> for SlotRank {
    fn from(param: &Parameter) -> Self {
        SlotRank::classify(param.annotation.is_some(), param.is_type_known)
    }
}

impl From<SlotRank> for SlotCounts {
    fn from(rank: SlotRank) -> Self {
        match rank {
            SlotRank::Typed => SlotCounts::typed(),
            SlotRank::Any => SlotCounts::any(),
            SlotRank::Untyped => SlotCounts::untyped(),
            SlotRank::Skip => SlotCounts::default(),
        }
    }
}

/// Parameter dedup key for overload merging.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParamKey {
    Positional(usize),
    Named(String),
    VarPositional,
    VarKeyword,
}

#[derive(Debug, Copy, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Serialize)]
/// Information about a single function parameter.
pub struct Parameter {
    pub name: String,
    pub annotation: Option<String>,
    /// Whether the resolved type contains no `Any`.
    pub is_type_known: bool,
    /// Overload merge key (`None` for self/cls and implicit params).
    #[serde(skip_serializing)]
    pub merge_key: Option<ParamKey>,
    pub location: Location,
}

/// Renamed from `Suppression` to avoid collision with `pyrefly_python::ignore::Suppression`.
#[derive(Debug, Serialize)]
pub struct ReportSuppression {
    /// The suppression tool (e.g. pyrefly, mypy, pyre).
    pub kind: Tool,
    pub codes: Vec<String>,
    pub location: Location,
}

#[derive(Debug, Serialize)]
pub struct Function {
    pub name: String,
    pub return_annotation: Option<String>,
    /// Whether the return type contains no `Any`.
    pub is_return_type_known: bool,
    pub parameters: Vec<Parameter>,
    pub is_type_known: bool,
    /// Property role if this function is a property accessor, `None` otherwise.
    #[serde(skip)]
    pub property_role: Option<PropertyRole>,
    /// Number of non-self/cls, non-implicit params (for symbol counting).
    pub n_params: usize,
    pub slots: SlotCounts,
    pub location: Location,
    /// Byte span of the symbol, for rendering diagnostics; not serialized.
    #[serde(skip)]
    pub range: TextRange,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct IncompleteAttribute {
    pub name: String,
    pub declared_in: String,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReportClass {
    pub name: String,
    pub incomplete_attributes: Vec<IncompleteAttribute>,
    pub location: Location,
}

#[derive(Debug, Serialize)]
pub struct Variable {
    pub name: String,
    pub annotation: Option<String>,
    pub slots: SlotCounts,
    pub location: Location,
    /// Byte span of the symbol, for rendering diagnostics; not serialized.
    #[serde(skip)]
    pub range: TextRange,
}

/// Per-symbol report with kind discriminator matching typestats' model.
#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum SymbolReport {
    #[serde(rename = "attr")]
    Attr {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        #[serde(skip)]
        n_params: usize,
        location: Location,
    },
    #[serde(rename = "class")]
    Class {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
    #[serde(rename = "property")]
    Property {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
}

impl SymbolReport {
    pub fn name(&self) -> &str {
        match self {
            Self::Attr { name, .. }
            | Self::Function { name, .. }
            | Self::Class { name, .. }
            | Self::Property { name, .. } => name,
        }
    }

    pub fn name_mut(&mut self) -> &mut String {
        match self {
            Self::Attr { name, .. }
            | Self::Function { name, .. }
            | Self::Class { name, .. }
            | Self::Property { name, .. } => name,
        }
    }

    pub fn slots(&self) -> &SlotCounts {
        match self {
            Self::Attr { slots, .. }
            | Self::Function { slots, .. }
            | Self::Class { slots, .. }
            | Self::Property { slots, .. } => slots,
        }
    }
}

/// Per-kind counts of the symbols in a report.
#[derive(Debug, Serialize, Default, Clone, Copy)]
pub struct SymbolCounts {
    pub n_functions: usize,
    pub n_methods: usize,
    pub n_function_params: usize,
    pub n_method_params: usize,
    pub n_classes: usize,
    pub n_attrs: usize,
    pub n_properties: usize,
    pub n_type_ignores: usize,
}

impl SymbolCounts {
    pub fn merge(self, other: SymbolCounts) -> SymbolCounts {
        SymbolCounts {
            n_functions: self.n_functions + other.n_functions,
            n_methods: self.n_methods + other.n_methods,
            n_function_params: self.n_function_params + other.n_function_params,
            n_method_params: self.n_method_params + other.n_method_params,
            n_classes: self.n_classes + other.n_classes,
            n_attrs: self.n_attrs + other.n_attrs,
            n_properties: self.n_properties + other.n_properties,
            n_type_ignores: self.n_type_ignores + other.n_type_ignores,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModuleReport {
    /// Fully-qualified module name (e.g. "mypackage.submodule").
    pub name: String,
    /// Filesystem path to the module source file.
    pub path: String,
    /// Names of symbols defined in this module.
    pub names: Vec<String>,
    pub line_count: usize,
    pub symbol_reports: Vec<SymbolReport>,
    pub type_ignores: Vec<ReportSuppression>,
    #[serde(flatten)]
    pub slots: SlotCounts,
    pub coverage: f64,
    pub strict_coverage: f64,
    #[serde(flatten)]
    pub symbols: SymbolCounts,
}

#[derive(Debug, Serialize)]
pub struct ReportSummary {
    pub n_modules: usize,
    #[serde(flatten)]
    pub slots: SlotCounts,
    pub coverage: f64,
    pub strict_coverage: f64,
    #[serde(flatten)]
    pub symbols: SymbolCounts,
}

#[derive(Debug, Serialize)]
pub struct FullReport {
    /// `"{major}.{minor}"` version for this report format.
    pub schema_version: String,
    pub module_reports: Vec<ModuleReport>,
    pub summary: ReportSummary,
}
