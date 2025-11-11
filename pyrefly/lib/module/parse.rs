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
use ruff_text_size::TextRange;
use vec1::vec1;

use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;

pub fn module_parse(
    contents: &str,
    version: PythonVersion,
    source_type: PySourceType,
    errors: &ErrorCollector,
) -> ModModule {
    let (module, parse_errors, unsupported_syntax_errors) =
        Ast::parse_with_version(contents, version, source_type);
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
    SemanticSyntaxContext::new(contents, version, errors).visit(&module);
    module
}

pub struct SemanticSyntaxContext<'me> {
    content: &'me str,
    version: ruff_python_ast::PythonVersion,
    errors: &'me ErrorCollector,
}

impl<'me> SemanticSyntaxContext<'me> {
    pub fn new(content: &'me str, version: PythonVersion, errors: &'me ErrorCollector) -> Self {
        Self {
            content,
            version: ruff_python_ast::PythonVersion {
                major: version.major as u8,
                minor: version.minor as u8,
            },
            errors,
        }
    }

    pub fn visit(&self, module: &ModModule) {
        let mut checker = ruff_python_parser::semantic_errors::SemanticSyntaxChecker::new();
        module.body.iter().for_each(|stmt| {
            checker.visit_stmt(stmt, self);
        });
    }
}

impl<'me> ruff_python_parser::semantic_errors::SemanticSyntaxContext
    for SemanticSyntaxContext<'me>
{
    fn python_version(&self) -> ruff_python_ast::PythonVersion {
        self.version
    }

    fn report_semantic_error(
        &self,
        error: ruff_python_parser::semantic_errors::SemanticSyntaxError,
    ) {
        self.errors.add(
            error.range,
            ErrorInfo::Kind(ErrorKind::InvalidSyntax),
            vec1![error.to_string()],
        );
    }

    fn future_annotations_or_stub(&self) -> bool {
        false
    }

    fn source(&self) -> &str {
        self.content
    }

    fn global(&self, _: &str) -> Option<TextRange> {
        // TODO: Properly implement this
        None
    }

    fn in_async_context(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_await_allowed_context(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_yield_allowed_context(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_sync_comprehension(&self) -> bool {
        // TODO: Properly implement this
        false
    }

    fn in_module_scope(&self) -> bool {
        // TODO: Properly implement this
        false
    }

    fn in_function_scope(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_generator_scope(&self) -> bool {
        // TODO: Properly implement this
        false
    }

    fn in_notebook(&self) -> bool {
        // TODO: Properly implement this
        false
    }
}
