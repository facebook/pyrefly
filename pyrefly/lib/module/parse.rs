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

use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

pub fn module_parse(
    contents: &str,
    version: PythonVersion,
    source_type: PySourceType,
    errors: &ErrorCollector,
) -> ModModule {
    let (module, parse_errors, unsupported_syntax_errors) =
        Ast::parse_with_version(contents, version, source_type);
    for err in parse_errors {
        errors
            .error_builder(
                err.location,
                ErrorKind::ParseError,
                format!("Parse error: {}", err.error),
            )
            .emit();
    }
    for err in unsupported_syntax_errors {
        errors
            .error_builder(err.range, ErrorKind::InvalidSyntax, format!("{err}"))
            .emit();
    }
    module
}
