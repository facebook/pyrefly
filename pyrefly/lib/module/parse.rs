/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::ModModule;
use ruff_python_ast::PySourceType;
use ruff_python_ast::token::Tokens;
use ruff_python_parser::Parsed;

use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

pub fn module_parse(
    contents: &str,
    version: PythonVersion,
    source_type: PySourceType,
    errors: &ErrorCollector,
) -> ModModule {
    parse_with_errors(contents, version, source_type, errors).into_syntax()
}

pub fn module_parse_with_tokens(
    contents: &str,
    version: PythonVersion,
    source_type: PySourceType,
    errors: &ErrorCollector,
) -> (ModModule, Tokens) {
    let parsed = parse_with_errors(contents, version, source_type, errors);
    let tokens = parsed.tokens().clone();
    (parsed.into_syntax(), tokens)
}

fn parse_with_errors(
    contents: &str,
    version: PythonVersion,
    source_type: PySourceType,
    errors: &ErrorCollector,
) -> Parsed<ModModule> {
    let parsed = Ast::parse_full_with_version(contents, version, source_type);
    for err in parsed.errors() {
        errors
            .error_builder(
                err.location,
                ErrorKind::ParseError,
                format!("Parse error: {}", err.error),
            )
            .emit();
    }
    for err in parsed.unsupported_syntax_errors() {
        errors
            .error_builder(err.range, ErrorKind::InvalidSyntax, format!("{err}"))
            .emit();
    }
    parsed
}
