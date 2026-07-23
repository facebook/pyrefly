/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::bindings::BindingsBuilder;
use crate::config::error_kind::ErrorKind;
use crate::export::special::SpecialExport;
use crate::types::class::AttrsFieldSpecifier;
use crate::types::class::AttrsFieldSpecifierKind;

/// `@<field>.default` / `@<field>.validator` / `@<field>.converter` methods found in a class body.
#[derive(Default)]
pub(crate) struct AttrsDecoratorMethods {
    defaults: SmallMap<Name, TextRange>,
    duplicate_defaults: SmallSet<Name>,
    /// attrs `pipe`s repeated `@<field>.converter` methods, so only the first-defined one's input
    /// type reaches `__init__`; we keep just that first range per field.
    converters: SmallMap<Name, TextRange>,
    bad_default_signatures: Vec<BadAttrsMethod>,
    bad_validator_signatures: Vec<BadAttrsMethod>,
    bad_converter_signatures: Vec<BadAttrsMethod>,
}

/// Why attrs cannot call a `@<field>.default` / `@<field>.validator` method, given that it
/// invokes them positionally with a fixed number of arguments.
enum AttrsMethodSignatureError {
    TooFewParameters,
    TooManyRequiredParameters,
    RequiredKeywordOnly,
}

impl AttrsMethodSignatureError {
    fn describe(&self) -> &'static str {
        match self {
            Self::TooFewParameters => "it accepts too few positional parameters",
            Self::TooManyRequiredParameters => {
                "it has required parameters that attrs does not pass"
            }
            Self::RequiredKeywordOnly => {
                "it has a required keyword-only parameter that attrs cannot pass"
            }
        }
    }
}

struct BadAttrsMethod {
    name: Name,
    range: TextRange,
    reason: AttrsMethodSignatureError,
}

struct MethodArity {
    required_positional: usize,
    total_positional: usize,
    has_varargs: bool,
    /// attrs calls decorator methods positionally, so a required keyword-only parameter can never
    /// be filled.
    has_required_kwonly: bool,
}

fn method_arity(parameters: &Parameters) -> MethodArity {
    let mut required_positional = 0;
    let mut total_positional = 0;
    for p in parameters.posonlyargs.iter().chain(&parameters.args) {
        total_positional += 1;
        if p.default.is_none() {
            required_positional += 1;
        }
    }
    MethodArity {
        required_positional,
        total_positional,
        has_varargs: parameters.vararg.is_some(),
        has_required_kwonly: parameters.kwonlyargs.iter().any(|p| p.default.is_none()),
    }
}

/// Collect `@<field>.default` / `@<field>.validator` methods. Recurses through class-body control
/// flow but not into nested `def`/`class` scopes, where `<name>.default` is unrelated.
pub(crate) fn collect_attrs_decorator_methods(body: &[Stmt], out: &mut AttrsDecoratorMethods) {
    for stmt in body {
        match stmt {
            Stmt::FunctionDef(func_def) => {
                for decorator in &func_def.decorator_list {
                    let Expr::Attribute(attr) = &decorator.expression else {
                        continue;
                    };
                    let Some(name) = attr.value.as_name_expr() else {
                        continue;
                    };
                    let arity = method_arity(&func_def.parameters);
                    match attr.attr.id.as_str() {
                        "default" => {
                            if out
                                .defaults
                                .insert(name.id.clone(), func_def.name.range)
                                .is_some()
                            {
                                out.duplicate_defaults.insert(name.id.clone());
                            }
                            // attrs calls the method as `meth(self)`: anything beyond `self` is
                            // unfilled (`*args` does not satisfy a required parameter).
                            let reason = if arity.required_positional > 1 {
                                Some(AttrsMethodSignatureError::TooManyRequiredParameters)
                            } else if arity.has_required_kwonly {
                                Some(AttrsMethodSignatureError::RequiredKeywordOnly)
                            } else {
                                None
                            };
                            if let Some(reason) = reason {
                                out.bad_default_signatures.push(BadAttrsMethod {
                                    name: name.id.clone(),
                                    range: func_def.name.range,
                                    reason,
                                });
                            }
                        }
                        // attrs calls the validator as `validator(self, attribute, value)`.
                        "validator" => {
                            let reason = if arity.total_positional < 3 && !arity.has_varargs {
                                Some(AttrsMethodSignatureError::TooFewParameters)
                            } else if arity.required_positional > 3 {
                                Some(AttrsMethodSignatureError::TooManyRequiredParameters)
                            } else if arity.has_required_kwonly {
                                Some(AttrsMethodSignatureError::RequiredKeywordOnly)
                            } else {
                                None
                            };
                            if let Some(reason) = reason {
                                out.bad_validator_signatures.push(BadAttrsMethod {
                                    name: name.id.clone(),
                                    range: func_def.name.range,
                                    reason,
                                });
                            }
                        }
                        // attrs invokes the converter as `converter(self, field, value)`.
                        "converter" => {
                            out.converters
                                .entry(name.id.clone())
                                .or_insert(func_def.name.range);
                            let reason = if arity.total_positional < 3 && !arity.has_varargs {
                                Some(AttrsMethodSignatureError::TooFewParameters)
                            } else if arity.required_positional > 3 {
                                Some(AttrsMethodSignatureError::TooManyRequiredParameters)
                            } else if arity.has_required_kwonly {
                                Some(AttrsMethodSignatureError::RequiredKeywordOnly)
                            } else {
                                None
                            };
                            if let Some(reason) = reason {
                                out.bad_converter_signatures.push(BadAttrsMethod {
                                    name: name.id.clone(),
                                    range: func_def.name.range,
                                    reason,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            Stmt::If(x) => {
                collect_attrs_decorator_methods(&x.body, out);
                for clause in &x.elif_else_clauses {
                    collect_attrs_decorator_methods(&clause.body, out);
                }
            }
            Stmt::For(x) => {
                collect_attrs_decorator_methods(&x.body, out);
                collect_attrs_decorator_methods(&x.orelse, out);
            }
            Stmt::While(x) => {
                collect_attrs_decorator_methods(&x.body, out);
                collect_attrs_decorator_methods(&x.orelse, out);
            }
            Stmt::With(x) => collect_attrs_decorator_methods(&x.body, out),
            Stmt::Try(x) => {
                collect_attrs_decorator_methods(&x.body, out);
                for ExceptHandler::ExceptHandler(h) in &x.handlers {
                    collect_attrs_decorator_methods(&h.body, out);
                }
                collect_attrs_decorator_methods(&x.orelse, out);
                collect_attrs_decorator_methods(&x.finalbody, out);
            }
            Stmt::Match(x) => {
                for case in &x.cases {
                    collect_attrs_decorator_methods(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    /// The attrs field-specifier kind for a call's callee: `Attrib` for legacy `attr.ib()`/`attrib()`,
    /// `Field` for next-gen `field()`, and `None` for any other call.
    pub(crate) fn attrs_field_specifier_kind(
        &self,
        func: &Expr,
    ) -> Option<AttrsFieldSpecifierKind> {
        match self.as_special_export(func) {
            Some(SpecialExport::AttrsLegacyAttrib) => Some(AttrsFieldSpecifierKind::Attrib),
            Some(SpecialExport::AttrsNextGenField) => Some(AttrsFieldSpecifierKind::Field),
            _ => None,
        }
    }

    /// Whether `expr` is a call to an attrs field specifier (`attr.ib()`/`attrib()`/`field()`).
    pub(crate) fn is_attrs_specifier_call(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Call(call) if self.attrs_field_specifier_kind(&call.func).is_some())
    }

    /// If `target = value` unpacks a tuple/list of names from a matching tuple/list literal with at
    /// least one attrs field specifier, the (name targets, values) element slices; otherwise `None`.
    /// A starred RHS element (`x, y = attr.ib(), *rest`) makes the elements non-positional, so it
    /// falls back to normal unpacking.
    pub(crate) fn attrs_unpacked_specifier_elements<'b>(
        &self,
        target: &'b Expr,
        value: &'b Expr,
    ) -> Option<(&'b [Expr], &'b [Expr])> {
        let (targets, values) = (unpack_elements(target)?, unpack_elements(value)?);
        (!targets.is_empty()
            && targets.len() == values.len()
            && targets.iter().all(|t| matches!(t, Expr::Name(_)))
            && !values.iter().any(|v| matches!(v, Expr::Starred(_)))
            && values.iter().any(|v| self.is_attrs_specifier_call(v)))
        .then_some((targets, values))
    }

    /// Classify a class-body assignment as an attrs `attr.ib()`/`field()` specifier and report its
    /// `@<field>.default`/`.validator` errors. Detected at binding so solving reads it by identity.
    pub(crate) fn attrs_field_specifier(
        &mut self,
        definition: &ClassFieldDefinition,
        field_name: &Name,
        range: TextRange,
        attrs_decorators: &AttrsDecoratorMethods,
    ) -> Option<AttrsFieldSpecifier> {
        let ClassFieldDefinition::AssignedInBody { value, .. } = definition else {
            return None;
        };
        let ExprOrBinding::Expr(Expr::Call(call)) = value.as_ref() else {
            return None;
        };
        let kind = self.attrs_field_specifier_kind(&call.func)?;
        // Only `attr.ib` accepts a positional default; `field`'s is keyword-only.
        let positional_default = (kind == AttrsFieldSpecifierKind::Attrib)
            .then(|| call.arguments.args.first())
            .flatten();
        let default_is_nothing = call
            .arguments
            .find_keyword("default")
            .map(|kw| &kw.value)
            .or(positional_default)
            .is_some_and(|e| self.as_special_export(e) == Some(SpecialExport::AttrsNothing));
        // attrs raises `DefaultAlreadySetError` for more than one `@<field>.default`.
        if attrs_decorators.duplicate_defaults.contains(field_name) {
            self.error(
                range,
                ErrorKind::BadClassDefinition,
                format!("`{field_name}` cannot have more than one `@{field_name}.default` method"),
            );
        }
        for BadAttrsMethod {
            range: method_range,
            reason,
            ..
        } in attrs_decorators
            .bad_default_signatures
            .iter()
            .filter(|m| &m.name == field_name)
        {
            self.error(
                *method_range,
                ErrorKind::BadClassDefinition,
                format!(
                    "The `@{field_name}.default` method must be callable with no argument other than `self`, but {}",
                    reason.describe()
                ),
            );
        }
        for BadAttrsMethod {
            range: method_range,
            reason,
            ..
        } in attrs_decorators
            .bad_validator_signatures
            .iter()
            .filter(|m| &m.name == field_name)
        {
            self.error(
                *method_range,
                ErrorKind::BadClassDefinition,
                format!(
                    "The `@{field_name}.validator` method must accept `(self, attribute, value)`, but {}",
                    reason.describe()
                ),
            );
        }
        for BadAttrsMethod {
            range: method_range,
            reason,
            ..
        } in attrs_decorators
            .bad_converter_signatures
            .iter()
            .filter(|m| &m.name == field_name)
        {
            self.error(
                *method_range,
                ErrorKind::BadClassDefinition,
                format!(
                    "The `@{field_name}.converter` method must accept `(self, field, value)`, but {}",
                    reason.describe()
                ),
            );
        }
        Some(AttrsFieldSpecifier {
            kind,
            default_is_nothing,
            // A duplicate is already an error; record no range so the return-type check skips it.
            default_decorator_method_range: if attrs_decorators
                .duplicate_defaults
                .contains(field_name)
            {
                None
            } else {
                attrs_decorators.defaults.get(field_name).copied()
            },
            converter_decorator_method_range: attrs_decorators.converters.get(field_name).copied(),
        })
    }
}

/// The element expressions of a tuple/list literal, for unpacking-assignment handling.
fn unpack_elements(expr: &Expr) -> Option<&[Expr]> {
    match expr {
        Expr::Tuple(t) => Some(&t.elts),
        Expr::List(l) => Some(&l.elts),
        _ => None,
    }
}
