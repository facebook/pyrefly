
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::lsp::ImportFormat;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

#[derive(Default)]
struct ResultsFilter {
    include_keywords: bool,
    include_builtins: bool,
}

fn get_default_test_report() -> impl Fn(&State, &Handle, TextSize) -> String {
    get_test_report(ResultsFilter::default(), ImportFormat::Absolute)
}

fn get_test_report(
    filter: ResultsFilter,
    import_format: ImportFormat,
) -> impl Fn(&State, &Handle, TextSize) -> String {
    move |state: &State, handle: &Handle, position: TextSize| {
        let mut report = "Completion Results:".to_owned();
        for CompletionItem {
            label,
            detail,
            kind,
            insert_text,
            data,
            tags,
            text_edit,
            documentation,
            ..
        } in state
            .transaction()
            .completion(handle, position, import_format, true)
        {
            let is_deprecated = if let Some(tags) = tags {
                tags.contains(&lsp_types::CompletionItemTag::DEPRECATED)
            } else {
                false
            };
            if (filter.include_keywords || kind != Some(CompletionItemKind::KEYWORD))
                && (filter.include_builtins || data != Some(serde_json::json!("builtin")))
            {
                report.push_str("\n- (");
                report.push_str(&format!("{:?}", kind.unwrap()));
                report.push_str(") ");
                if is_deprecated {
                    report.push_str("[DEPRECATED] ");
                }
                report.push_str(&label);
                if let Some(detail) = detail {
                    report.push_str(": ");
                    report.push_str(&detail);
                }
                if let Some(insert_text) = insert_text {
                    report.push_str(" inserting `");
                    report.push_str(&insert_text);
                    report.push('`');
                }
                if let Some(text_edit) = text_edit {
                    report.push_str(" with text edit: ");
                    report.push_str(&format!("{:?}", &text_edit));
                }
                if let Some(documentation) = documentation {
                    report.push('\n');
                    match documentation {
                        lsp_types::Documentation::String(s) => {
                            report.push_str(&s);
                        }
                        lsp_types::Documentation::MarkupContent(content) => {
                            report.push_str(&content.value);
                        }
                    }
                }
            }
        }
        report
    }
}

#[test]
fn completion_literal_quote_test() {
    let code = r#"
from typing import Literal
def foo(fruit: Literal["apple", "pear"]) -> None: ...
foo('
#    ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    
    // We expect the completion to NOT insert extra quotes if we are already in a quote.
    // Currently it likely inserts quotes.
    println!("{}", report);
    assert!(report.contains("inserting `apple`"), "Should insert unquoted apple");
    assert!(report.contains("inserting `pear`"), "Should insert unquoted pear");
}
