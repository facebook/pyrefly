/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SARIF diagnostic output for the `check` command.
//!
//! Only the subset of the SARIF 2.1.0 object model that Pyrefly populates is
//! modelled here; the report is purely diagnostic and carries no fixes,
//! fingerprints or code flows.
//! <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anstream::stdout;
use anyhow::Context as _;
use itertools::Itertools as _;
use lsp_types::Url;
use pyrefly_config::error_kind::Severity;
use pyrefly_util::absolutize::Absolutize;
use pyrefly_util::lined_buffer::DisplayRange;
use serde::Serialize;

use crate::error::error::BaselineStatus;
use crate::error::error::Error;

const SCHEMA_URL: &str = "https://json.schemastore.org/sarif-2.1.0.json";
const SARIF_VERSION: &str = "2.1.0";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    #[serde(rename = "$schema")]
    schema: &'static str,
    version: &'static str,
    runs: Vec<Run>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Run {
    tool: Tool,
    /// Pyrefly counts columns in Unicode code points, not UTF-16 code units.
    column_kind: &'static str,
    results: Vec<SarifResult>,
}

#[derive(Serialize)]
struct Tool {
    driver: ToolComponent,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolComponent {
    name: &'static str,
    /// The Pyrefly version string. Deliberately `version` rather than
    /// `semanticVersion`, because internal builds are not semver.
    version: String,
    information_uri: &'static str,
    rules: Vec<ReportingDescriptor>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportingDescriptor {
    id: &'static str,
    name: &'static str,
    help_uri: String,
    default_configuration: ReportingConfiguration,
}

#[derive(Serialize)]
struct ReportingConfiguration {
    level: Level,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SarifResult {
    rule_id: &'static str,
    rule_index: usize,
    level: Level,
    message: Message,
    locations: Vec<Location>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    related_locations: Vec<Location>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_state: Option<SarifBaselineState>,
}

#[derive(Serialize)]
struct Message {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Location {
    physical_location: PhysicalLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<Message>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PhysicalLocation {
    artifact_location: ArtifactLocation,
    region: Region,
}

#[derive(Serialize)]
struct ArtifactLocation {
    uri: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Region {
    start_line: u32,
    start_column: u32,
    end_line: u32,
    end_column: u32,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Level {
    None,
    Note,
    Warning,
    Error,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum SarifBaselineState {
    New,
    Unchanged,
}

/// `Ignore` maps to `none`, which in a rule's `defaultConfiguration` means the rule
/// is off by default. Ignored diagnostics never reach a `result`, which is filtered
/// out before conversion.
fn severity_to_level(severity: Severity) -> Level {
    match severity {
        Severity::Ignore => Level::None,
        Severity::Info => Level::Note,
        Severity::Warn => Level::Warning,
        Severity::Error => Level::Error,
    }
}

fn region(range: &DisplayRange) -> Region {
    let start_line = range.start.line_within_cell().get();
    let start_column = range.start.column().get();
    // `line_within_cell` is relative to each cell, but the URI fragment names only
    // the start cell. A range that spans two cells (e.g. a parse error that reaches
    // the next cell) would otherwise report an end line measured against a different
    // cell, giving a region that is inconsistent with the fragment or outright
    // invalid (`end_line < start_line`). Clamp such a range to its start so the
    // region stays within the first cell.
    let (end_line, end_column) = if range.start.cell() == range.end.cell() {
        (range.end.line_within_cell().get(), range.end.column().get())
    } else {
        (start_line, start_column)
    };
    Region {
        start_line,
        start_column,
        end_line,
        end_column,
    }
}

/// `msg_details` is stored pre-indented for terminal pretty-printing. Strip that
/// indentation so report consumers get the message as written.
fn message_text(error: &Error) -> String {
    match error.msg_details() {
        None => error.msg_header().to_owned(),
        Some(details) => {
            let details = details
                .lines()
                .map(|line| line.strip_prefix("  ").unwrap_or(line))
                .join("\n");
            format!("{}\n{details}", error.msg_header())
        }
    }
}

/// The URI of the file an error lives in, relative to `relative_to` when it is
/// underneath it and absolute otherwise. `None` means the caller asked for
/// absolute paths.
fn artifact_uri(path: &Path, relative_to: Option<&Path>) -> anyhow::Result<String> {
    let path = path.absolutize();
    let path_uri = Url::from_file_path(&path)
        .map_err(|()| anyhow::anyhow!("cannot convert `{}` to a file URI", path.display()))?;

    let Some(relative_to) = relative_to else {
        return Ok(path_uri.to_string());
    };
    // `absolutize` leaves the path alone if it cannot resolve the current directory,
    // so an absolute root is not guaranteed.
    if !relative_to.is_absolute() {
        return Err(anyhow::anyhow!(
            "cannot resolve `{}` to an absolute SARIF root",
            relative_to.display()
        ));
    }
    if path.strip_prefix(relative_to).is_err() {
        return Ok(path_uri.to_string());
    }

    let base_uri = Url::from_directory_path(relative_to).map_err(|()| {
        anyhow::anyhow!(
            "cannot convert `{}` to a directory URI",
            relative_to.display()
        )
    })?;
    base_uri.make_relative(&path_uri).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot make `{}` relative to `{}`",
            path.display(),
            relative_to.display()
        )
    })
}

fn errors_to_sarif(version: &str, relative_to: &Path, errors: &[Error]) -> anyhow::Result<Report> {
    // `--relative-to ""` asks for absolute paths. Absolutizing an empty path would
    // resolve it to the current directory and silently make paths relative to that.
    let relative_to: Option<PathBuf> = if relative_to.as_os_str().is_empty() {
        None
    } else {
        Some(relative_to.absolutize())
    };
    let relative_to = relative_to.as_deref();

    // An ignored diagnostic is one the user suppressed, so it is not a finding.
    let errors = errors
        .iter()
        .filter(|error| error.severity().is_enabled())
        .collect::<Vec<_>>();

    let rule_kinds = errors
        .iter()
        .map(|error| error.error_kind())
        .sorted_unstable_by_key(|kind| kind.to_name())
        .dedup()
        .collect::<Vec<_>>();
    let rule_indices = rule_kinds
        .iter()
        .enumerate()
        .map(|(index, kind)| (*kind, index))
        .collect::<HashMap<_, _>>();

    let mut artifact_uris = HashMap::new();
    for error in &errors {
        let path = error.path().as_path();
        if let Entry::Vacant(entry) = artifact_uris.entry(path.to_path_buf()) {
            entry.insert(artifact_uri(path, relative_to)?);
        }
    }

    let rules = rule_kinds
        .iter()
        .map(|kind| ReportingDescriptor {
            id: kind.to_name(),
            name: kind.to_name(),
            help_uri: kind.docs_url(),
            default_configuration: ReportingConfiguration {
                level: severity_to_level(kind.default_severity()),
            },
        })
        .collect();

    let results = errors
        .iter()
        .map(|error| {
            let kind = error.error_kind();
            let base_uri = artifact_uris
                .get(error.path().as_path())
                .expect("error path was collected into the SARIF artifact URI table");
            // A notebook line number is relative to its cell, so name the cell in the URI
            // fragment to keep the position unambiguous.
            let location = |range: &DisplayRange, message| Location {
                physical_location: PhysicalLocation {
                    artifact_location: ArtifactLocation {
                        uri: match range.start.cell() {
                            Some(cell) => format!("{base_uri}#{cell}"),
                            None => base_uri.clone(),
                        },
                    },
                    region: region(range),
                },
                message,
            };
            SarifResult {
                rule_id: kind.to_name(),
                rule_index: *rule_indices
                    .get(&kind)
                    .expect("error kind was collected into the SARIF rule table"),
                level: severity_to_level(error.severity()),
                message: Message {
                    text: message_text(error),
                },
                locations: vec![location(error.display_range(), None)],
                related_locations: error
                    .secondary_annotations()
                    .iter()
                    .map(|annotation| {
                        location(
                            &error.module().display_range(annotation.range),
                            Some(Message {
                                text: annotation.label.to_string(),
                            }),
                        )
                    })
                    .collect(),
                baseline_state: match error.baseline_status() {
                    BaselineStatus::Unmatched => Some(SarifBaselineState::New),
                    BaselineStatus::Matched => Some(SarifBaselineState::Unchanged),
                    BaselineStatus::NotConfigured | BaselineStatus::NotCompared => None,
                },
            }
        })
        .collect();

    Ok(Report {
        schema: SCHEMA_URL,
        version: SARIF_VERSION,
        runs: vec![Run {
            tool: Tool {
                driver: ToolComponent {
                    name: "Pyrefly",
                    version: version.to_owned(),
                    information_uri: "https://pyrefly.org/",
                    rules,
                },
            },
            column_kind: "unicodeCodePoints",
            results,
        }],
    })
}

fn buffered_write_error_sarif(
    writer: impl Write,
    version: &str,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer_pretty(&mut writer, &errors_to_sarif(version, relative_to, errors)?)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn write_error_sarif_to_file(
    path: &Path,
    version: &str,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    buffered_write_error_sarif(File::create(path)?, version, relative_to, errors)
        .with_context(|| format!("while writing SARIF errors to `{}`", path.display()))
}

pub(crate) fn write_error_sarif_to_console(
    version: &str,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    buffered_write_error_sarif(stdout(), version, relative_to, errors)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_config::error_kind::ErrorKind;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_notebook::Cell;
    use ruff_notebook::CellMetadata;
    use ruff_notebook::CodeCell;
    use ruff_notebook::Notebook;
    use ruff_notebook::RawNotebook;
    use ruff_notebook::RawNotebookMetadata;
    use ruff_notebook::SourceValue;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;

    const VERSION: &str = "0.0.0";

    fn sample_error(
        path: PathBuf,
        source: &str,
        start: u32,
        end: u32,
        msg: &str,
        kind: ErrorKind,
    ) -> Error {
        let module = Module::new(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(path),
            Arc::new(source.to_owned()),
        );
        Error::new(
            module,
            TextRange::new(TextSize::from(start), TextSize::from(end)),
            msg.to_owned(),
            Vec::new(),
            kind,
        )
    }

    fn to_json(relative_to: &Path, errors: &[Error]) -> serde_json::Value {
        serde_json::to_value(errors_to_sarif(VERSION, relative_to, errors).unwrap()).unwrap()
    }

    #[test]
    fn conversion_maps_rules_and_severities() {
        let errors = vec![
            sample_error(
                PathBuf::from("/repo/foo.py"),
                "x\n",
                0,
                1,
                "warning",
                ErrorKind::BadAssignment,
            )
            .with_severity(Severity::Warn),
            sample_error(
                PathBuf::from("/repo/foo.py"),
                "x\n",
                0,
                1,
                "error",
                ErrorKind::BadAssignment,
            ),
            sample_error(
                PathBuf::from("/repo/info.py"),
                "x\n",
                0,
                1,
                "info",
                ErrorKind::RevealType,
            )
            .with_severity(Severity::Info),
        ];

        let sarif = errors_to_sarif(VERSION, Path::new("/repo"), &errors).unwrap();
        assert_eq!(sarif.runs.len(), 1);
        let run = &sarif.runs[0];
        assert_eq!(run.tool.driver.version, VERSION);
        assert_eq!(
            run.tool
                .driver
                .rules
                .iter()
                .map(|rule| rule.id)
                .collect::<Vec<_>>(),
            vec!["bad-assignment", "reveal-type"]
        );
        // `reveal-type` is a directive, so its rule default is `note` rather than `error`.
        assert_eq!(
            run.tool
                .driver
                .rules
                .iter()
                .map(|rule| rule.default_configuration.level)
                .collect::<Vec<_>>(),
            vec![Level::Error, Level::Note]
        );
        assert_eq!(
            run.results
                .iter()
                .map(|result| result.rule_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        assert_eq!(
            run.results
                .iter()
                .map(|result| result.level)
                .collect::<Vec<_>>(),
            vec![Level::Warning, Level::Error, Level::Note]
        );

        let empty = errors_to_sarif(VERSION, Path::new("/repo"), &[]).unwrap();
        assert!(empty.runs[0].tool.driver.rules.is_empty());
        assert!(empty.runs[0].results.is_empty());
    }

    #[test]
    fn conversion_drops_ignored_diagnostics() {
        let errors = vec![
            sample_error(
                PathBuf::from("/repo/ignored.py"),
                "x\n",
                0,
                1,
                "ignored",
                ErrorKind::ExplicitAny,
            )
            .with_severity(Severity::Ignore),
        ];

        let sarif = errors_to_sarif(VERSION, Path::new("/repo"), &errors).unwrap();
        assert!(sarif.runs[0].results.is_empty());
        // A dropped diagnostic must not leave its rule behind either.
        assert!(sarif.runs[0].tool.driver.rules.is_empty());
    }

    #[test]
    fn conversion_maps_baseline_provenance() {
        let errors = vec![
            sample_error(
                PathBuf::from("/repo/matched.py"),
                "x\n",
                0,
                1,
                "matched",
                ErrorKind::BadAssignment,
            )
            .with_baseline_status(BaselineStatus::Matched),
            sample_error(
                PathBuf::from("/repo/new.py"),
                "x\n",
                0,
                1,
                "new",
                ErrorKind::BadAssignment,
            )
            .with_baseline_status(BaselineStatus::Unmatched),
        ];

        let json = to_json(Path::new("/repo"), &errors);
        assert_eq!(json["runs"][0]["results"][0]["baselineState"], "unchanged");
        assert_eq!(json["runs"][0]["results"][1]["baselineState"], "new");

        let not_compared = sample_error(
            PathBuf::from("/repo/uncompared.py"),
            "x\n",
            0,
            1,
            "uncompared",
            ErrorKind::BadAssignment,
        )
        .with_baseline_status(BaselineStatus::NotCompared);
        let json = to_json(Path::new("/repo"), &[not_compared]);
        assert!(json["runs"][0]["results"][0].get("baselineState").is_none());
    }

    #[test]
    fn conversion_maps_locations() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        let inside = root.join("space dir").join("café.py");
        let outside = temp.path().join("outside.py");
        let errors = vec![
            sample_error(
                inside,
                "αβ\nxyz\n",
                2,
                7,
                "inside",
                ErrorKind::BadAssignment,
            ),
            sample_error(
                outside.clone(),
                "x\n",
                0,
                1,
                "outside",
                ErrorKind::BadAssignment,
            ),
        ];

        let sarif = errors_to_sarif(VERSION, &root, &errors).unwrap();
        let results = &sarif.runs[0].results;
        let inside_location = &results[0].locations[0].physical_location;
        assert_eq!(
            inside_location.artifact_location.uri,
            // @lint-ignore SPELL exact-word-misspell
            "space%20dir/caf%C3%A9.py"
        );
        let region = &inside_location.region;
        assert_eq!(
            (
                region.start_line,
                region.start_column,
                region.end_line,
                region.end_column,
            ),
            (1, 2, 2, 3)
        );

        assert_eq!(
            results[1].locations[0]
                .physical_location
                .artifact_location
                .uri,
            Url::from_file_path(outside).unwrap().to_string()
        );
    }

    #[test]
    fn empty_relative_to_keeps_absolute_paths() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("foo.py");
        let errors = vec![sample_error(
            path.clone(),
            "x\n",
            0,
            1,
            "absolute",
            ErrorKind::BadAssignment,
        )];

        // `--relative-to ""` is documented as "show absolute paths", so the current
        // directory must not be used as an implicit root.
        let sarif = errors_to_sarif(VERSION, Path::new(""), &errors).unwrap();
        assert_eq!(
            sarif.runs[0].results[0].locations[0]
                .physical_location
                .artifact_location
                .uri,
            Url::from_file_path(&path).unwrap().to_string()
        );
    }

    #[test]
    fn notebook_uri_names_the_cell_the_line_belongs_to() {
        let code = |src: &str| {
            Cell::Code(CodeCell {
                execution_count: None,
                id: None,
                metadata: CellMetadata::default(),
                outputs: vec![],
                source: SourceValue::String(src.to_owned()),
            })
        };
        let notebook = Notebook::from_raw_notebook(
            RawNotebook {
                cells: vec![code("x = 1"), code("y = 2\nz = 3")],
                metadata: RawNotebookMetadata::default(),
                nbformat: 4,
                nbformat_minor: 5,
            },
            false,
        )
        .unwrap();
        let module = Module::new_notebook(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.ipynb")),
            Arc::new(notebook),
        );
        // "z = 3" is line 2 of the second cell, but line 3 of the concatenated source.
        let error = Error::new(
            module,
            TextRange::new(TextSize::from(12), TextSize::from(13)),
            "second line of the second cell".to_owned(),
            Vec::new(),
            ErrorKind::BadAssignment,
        );

        let sarif = errors_to_sarif(VERSION, Path::new("/repo"), &[error]).unwrap();
        let location = &sarif.runs[0].results[0].locations[0].physical_location;
        // Cell numbers in the fragment are 1-based, matching the other output formats.
        assert_eq!(location.artifact_location.uri, "foo.ipynb#2");
        assert_eq!(location.region.start_line, 2);
    }

    #[test]
    fn notebook_range_spanning_cells_is_clamped_to_the_first_cell() {
        let code = |src: &str| {
            Cell::Code(CodeCell {
                execution_count: None,
                id: None,
                metadata: CellMetadata::default(),
                outputs: vec![],
                source: SourceValue::String(src.to_owned()),
            })
        };
        let notebook = Notebook::from_raw_notebook(
            RawNotebook {
                cells: vec![code("a = 1\nb = 2\nc = 3"), code("d = 4")],
                metadata: RawNotebookMetadata::default(),
                nbformat: 4,
                nbformat_minor: 5,
            },
            false,
        )
        .unwrap();
        let module = Module::new_notebook(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.ipynb")),
            Arc::new(notebook),
        );
        // Start on "c = 3" (line 3 of the first cell), end on "d = 4" (line 1 of the
        // second cell). Reported against their own cells the end line (1) would be
        // less than the start line (3), an invalid region.
        let error = Error::new(
            module,
            TextRange::new(TextSize::from(12), TextSize::from(19)),
            "range that spills into the next cell".to_owned(),
            Vec::new(),
            ErrorKind::BadAssignment,
        );

        let sarif = errors_to_sarif(VERSION, Path::new("/repo"), &[error]).unwrap();
        let location = &sarif.runs[0].results[0].locations[0].physical_location;
        // The fragment names the start cell, so the region must stay inside it.
        assert_eq!(location.artifact_location.uri, "foo.ipynb#1");
        let region = &location.region;
        assert_eq!(region.start_line, 3);
        assert_eq!(region.end_line, 3);
        assert!(region.end_line >= region.start_line);
        assert_eq!(region.start_column, region.end_column);
    }

    #[test]
    fn relative_root_is_rejected() {
        // `absolutize` hands a path back unchanged when the current directory cannot
        // be resolved, so a non-absolute root must be reported rather than asserted on.
        assert!(artifact_uri(Path::new("/repo/foo.py"), Some(Path::new("root"))).is_err());
    }

    #[test]
    fn message_keeps_details_without_terminal_indentation() {
        let module = Module::new(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("x\n".to_owned()),
        );
        let error = Error::new(
            module,
            TextRange::new(TextSize::from(0), TextSize::from(1)),
            "`Literal[0]` is not assignable to `str`".to_owned(),
            vec!["Attempted to assign to a variable declared as `str`".to_owned()],
            ErrorKind::BadAssignment,
        );

        assert_eq!(
            message_text(&error),
            "`Literal[0]` is not assignable to `str`\nAttempted to assign to a variable declared as `str`"
        );
    }

    #[test]
    fn secondary_annotations_become_related_locations() {
        let module = Module::new(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("val * 2".to_owned()),
        );
        let error = Error::new(
            module,
            TextRange::new(TextSize::from(0), TextSize::from(7)),
            "`*` is not supported".to_owned(),
            Vec::new(),
            ErrorKind::UnsupportedOperation,
        )
        .with_annotation(
            TextRange::new(TextSize::from(0), TextSize::from(3)),
            "has type `int | str`".to_owned(),
        );

        let sarif = errors_to_sarif(VERSION, Path::new("/repo"), &[error]).unwrap();
        let related = &sarif.runs[0].results[0].related_locations;
        assert_eq!(related.len(), 1);
        assert_eq!(
            related[0].message.as_ref().unwrap().text,
            "has type `int | str`"
        );
        let region = &related[0].physical_location.region;
        assert_eq!((region.start_column, region.end_column), (1, 4));
    }

    #[test]
    fn results_omit_related_locations_when_there_are_none() {
        let errors = vec![sample_error(
            PathBuf::from("/repo/foo.py"),
            "x\n",
            0,
            1,
            "error",
            ErrorKind::BadAssignment,
        )];

        let json = to_json(Path::new("/repo"), &errors);
        let result = &json["runs"][0]["results"][0];
        assert!(result.get("relatedLocations").is_none());
        assert_eq!(result["locations"][0].get("message"), None);
    }

    #[test]
    fn report_uses_version_not_semantic_version() {
        let json = to_json(Path::new("/repo"), &[]);
        assert_eq!(json["version"], "2.1.0");
        let driver = &json["runs"][0]["tool"]["driver"];
        assert_eq!(driver["version"], VERSION);
        assert!(driver.get("semanticVersion").is_none());
    }
}
