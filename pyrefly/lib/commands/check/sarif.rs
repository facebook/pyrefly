/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SARIF diagnostic output for the `check` command.

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use anstream::stdout;
use anyhow::Context as _;
use lsp_types::Url;
use pyrefly_config::error_kind::Severity;
use pyrefly_util::absolutize::Absolutize;
use serde_sarif::sarif::ArtifactLocation;
use serde_sarif::sarif::Location as SarifLocation;
use serde_sarif::sarif::Message as SarifMessage;
use serde_sarif::sarif::PhysicalLocation;
use serde_sarif::sarif::Region;
use serde_sarif::sarif::ReportingConfiguration;
use serde_sarif::sarif::ReportingDescriptor;
use serde_sarif::sarif::Result as SarifResult;
use serde_sarif::sarif::ResultColumnKind;
use serde_sarif::sarif::ResultLevel;
use serde_sarif::sarif::Run as SarifRun;
use serde_sarif::sarif::SCHEMA_URL;
use serde_sarif::sarif::Sarif;
use serde_sarif::sarif::Tool;
use serde_sarif::sarif::ToolComponent;
use serde_sarif::sarif::Version;

use crate::error::error::Error;

fn severity_to_sarif_level(severity: Severity) -> ResultLevel {
    match severity {
        Severity::Ignore => ResultLevel::None,
        Severity::Info => ResultLevel::Note,
        Severity::Warn => ResultLevel::Warning,
        Severity::Error => ResultLevel::Error,
    }
}

fn sarif_artifact_uri(path: &Path, relative_to: &Path) -> anyhow::Result<String> {
    let path = path.absolutize();
    let relative_to = relative_to.absolutize();
    let path_uri = Url::from_file_path(&path)
        .map_err(|()| anyhow::anyhow!("cannot convert `{}` to a file URI", path.display()))?;

    if path.strip_prefix(&relative_to).is_ok() {
        let base_uri = Url::from_directory_path(&relative_to).map_err(|()| {
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
    } else {
        Ok(path_uri.to_string())
    }
}

fn errors_to_sarif(relative_to: &Path, errors: &[Error]) -> anyhow::Result<Sarif> {
    let rule_kinds = {
        let mut kinds = errors
            .iter()
            .map(|error| error.error_kind())
            .collect::<Vec<_>>();
        kinds.sort_unstable_by_key(|kind| kind.to_name());
        kinds.dedup();
        kinds
    };

    let rules = rule_kinds
        .iter()
        .map(|kind| {
            Ok(ReportingDescriptor::builder()
                .id(kind.to_name())
                .name(kind.to_name())
                .help_uri(kind.docs_url())
                .default_configuration(
                    ReportingConfiguration::builder()
                        .level(serde_json::to_value(severity_to_sarif_level(
                            kind.default_severity(),
                        ))?)
                        .build(),
                )
                .build())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let results = errors
        .iter()
        .map(|error| {
            let range = error.display_range();
            let kind = error.error_kind();
            let rule_index = rule_kinds
                .binary_search_by_key(&kind.to_name(), |kind| kind.to_name())
                .expect("error kind was collected into the SARIF rule table");
            let location = SarifLocation::builder()
                .physical_location(
                    PhysicalLocation::builder()
                        .artifact_location(
                            ArtifactLocation::builder()
                                .uri(sarif_artifact_uri(error.path().as_path(), relative_to)?)
                                .build(),
                        )
                        .region(
                            Region::builder()
                                .start_line(i64::from(range.start.line_within_cell().get()))
                                .start_column(i64::from(range.start.column().get()))
                                .end_line(i64::from(range.end.line_within_cell().get()))
                                .end_column(i64::from(range.end.column().get()))
                                .build(),
                        )
                        .build(),
                )
                .build();
            Ok(SarifResult::builder()
                .rule_id(kind.to_name())
                .rule_index(
                    i64::try_from(rule_index).expect("SARIF rule index does not fit in i64"),
                )
                .level(severity_to_sarif_level(error.severity()))
                .message(SarifMessage::from(error.msg().as_str()))
                .locations(vec![location])
                .build())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let driver = ToolComponent::builder()
        .name("Pyrefly")
        .information_uri("https://pyrefly.org/")
        .semantic_version(env!("CARGO_PKG_VERSION"))
        .rules(rules)
        .build();
    let run = SarifRun::builder()
        .column_kind(serde_json::to_value(ResultColumnKind::UnicodeCodePoints)?)
        .results(results)
        .tool(Tool::from(driver))
        .build();
    Ok(Sarif::builder()
        .schema(SCHEMA_URL)
        .runs(vec![run])
        .version(serde_json::to_value(Version::V2_1_0)?)
        .build())
}

fn write_error_sarif(
    writer: &mut impl Write,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    serde_json::to_writer_pretty(&mut *writer, &errors_to_sarif(relative_to, errors)?)?;
    writeln!(writer)?;
    Ok(())
}

fn buffered_write_error_sarif(
    writer: impl Write,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(writer);
    write_error_sarif(&mut writer, relative_to, errors)?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn write_error_sarif_to_file(
    path: &Path,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    fn f(path: &Path, relative_to: &Path, errors: &[Error]) -> anyhow::Result<()> {
        let file = File::create(path)?;
        buffered_write_error_sarif(file, relative_to, errors)
    }
    f(path, relative_to, errors)
        .with_context(|| format!("while writing SARIF errors to `{}`", path.display()))
}

pub(crate) fn write_error_sarif_to_console(
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    buffered_write_error_sarif(stdout(), relative_to, errors)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_config::error_kind::ErrorKind;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;

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

    #[test]
    fn conversion_maps_rules_and_severities() {
        let errors = vec![
            sample_error(
                PathBuf::from("/repo/foo.py"),
                "x\n",
                0,
                1,
                "error",
                ErrorKind::BadAssignment,
            ),
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
                PathBuf::from("/repo/info.py"),
                "x\n",
                0,
                1,
                "info",
                ErrorKind::RevealType,
            )
            .with_severity(Severity::Info),
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

        let sarif = errors_to_sarif(Path::new("/repo"), &errors).unwrap();
        assert_eq!(sarif.runs.len(), 1);
        let run = &sarif.runs[0];
        let rules = run.tool.driver.rules.as_ref().unwrap();
        assert_eq!(
            rules
                .iter()
                .map(|rule| rule.id.as_str())
                .collect::<Vec<_>>(),
            vec!["bad-assignment", "explicit-any", "reveal-type"]
        );
        assert_eq!(
            rules
                .iter()
                .map(|rule| {
                    rule.default_configuration
                        .as_ref()
                        .unwrap()
                        .level
                        .as_ref()
                        .unwrap()
                        .clone()
                })
                .collect::<Vec<_>>(),
            vec![
                serde_json::json!("error"),
                serde_json::json!("none"),
                serde_json::json!("note"),
            ]
        );

        let results = run.results.as_ref().unwrap();
        assert_eq!(
            results
                .iter()
                .map(|result| result.rule_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(0), Some(2), Some(1)]
        );
        assert_eq!(
            results
                .iter()
                .map(|result| result.level)
                .collect::<Vec<_>>(),
            vec![
                Some(ResultLevel::Error),
                Some(ResultLevel::Warning),
                Some(ResultLevel::Note),
                Some(ResultLevel::None),
            ]
        );

        let empty = errors_to_sarif(Path::new("/repo"), &[]).unwrap();
        assert!(empty.runs[0].tool.driver.rules.as_ref().unwrap().is_empty());
        assert!(empty.runs[0].results.as_ref().unwrap().is_empty());
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

        let sarif = errors_to_sarif(&root, &errors).unwrap();
        let results = sarif.runs[0].results.as_ref().unwrap();
        let inside_location = results[0].locations.as_ref().unwrap()[0]
            .physical_location
            .as_ref()
            .unwrap();
        assert_eq!(
            inside_location
                .artifact_location
                .as_ref()
                .unwrap()
                .uri
                .as_deref(),
            Some("space%20dir/caf%C3%A9.py")
        );
        let region = inside_location.region.as_ref().unwrap();
        assert_eq!(
            (
                region.start_line,
                region.start_column,
                region.end_line,
                region.end_column,
            ),
            (Some(1), Some(2), Some(2), Some(3))
        );

        let outside_uri = results[1].locations.as_ref().unwrap()[0]
            .physical_location
            .as_ref()
            .unwrap()
            .artifact_location
            .as_ref()
            .unwrap()
            .uri
            .as_ref()
            .unwrap();
        assert_eq!(
            outside_uri,
            &Url::from_file_path(outside).unwrap().to_string()
        );
    }
}
