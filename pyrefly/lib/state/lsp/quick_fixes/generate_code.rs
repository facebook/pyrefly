/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::find_enclosing_statement_range;
use super::extract_shared::line_indent_and_start;

const BODY_INDENT: &str = "    ";

pub(crate) fn generate_code_actions(
    module_info: &Module,
    ast: &ModModule,
    selection: TextRange,
    name: &str,
) -> Option<Vec<(String, Module, TextRange, String)>> {
    if name.is_empty() {
        return None;
    }
    let (statement_indent, insert_position) = match find_enclosing_statement_range(ast, selection)
        .and_then(|range| line_indent_and_start(module_info.contents(), range.start()))
    {
        Some(result) => result,
        None => line_indent_and_start(module_info.contents(), selection.start())?,
    };
    let insert_range = TextRange::at(insert_position, TextSize::new(0));

    let variable_text = format!("{statement_indent}{name} = None\n");
    let function_text =
        format!("{statement_indent}def {name}():\n{statement_indent}{BODY_INDENT}pass\n");
    let class_text =
        format!("{statement_indent}class {name}:\n{statement_indent}{BODY_INDENT}pass\n");

    Some(vec![
        (
            format!("Generate variable `{name}`"),
            module_info.dupe(),
            insert_range,
            variable_text,
        ),
        (
            format!("Generate function `{name}`"),
            module_info.dupe(),
            insert_range,
            function_text,
        ),
        (
            format!("Generate class `{name}`"),
            module_info.dupe(),
            insert_range,
            class_text,
        ),
    ])
}
