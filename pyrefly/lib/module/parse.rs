/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::ModModule;
use vec1::vec1;

use crate::cython;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::module::module_info::ModuleInfo;

pub fn module_parse(
    module_info: &ModuleInfo,
    contents: &str,
    version: PythonVersion,
    errors: &ErrorCollector,
) -> ModModule {
    if cython::is_cython_module(module_info) {
        for range in cython::syntax_error_ranges(contents) {
            errors.add(
                range,
                ErrorInfo::Kind(ErrorKind::ParseError),
                vec1!["Cython parse error".to_owned()],
            );
        }
        return Ast::parse_with_version("", version, module_info.source_type()).0;
    }
    let (module, parse_errors, unsupported_syntax_errors) =
        Ast::parse_with_version(contents, version, module_info.source_type());
    for err in parse_errors {
        errors.add(
            err.location,
            ErrorInfo::Kind(ErrorKind::ParseError),
            vec1![format!("Parse error: {}", err.error)],
        );
    }
    for err in unsupported_syntax_errors {
        errors.add(
            err.range,
            ErrorInfo::Kind(ErrorKind::InvalidSyntax),
            vec1![format!("{err}")],
        )
    }
    module
}
