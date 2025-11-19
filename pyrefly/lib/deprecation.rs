use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprStringLiteral;

/// Metadata extracted from a `@deprecated` decorator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeprecatedDecoration {
    pub message: Option<String>,
}

impl DeprecatedDecoration {
    pub fn new(message: Option<String>) -> Self {
        Self { message }
    }
}

fn attribute_to_name(expr: &ExprAttribute) -> Option<String> {
    let mut parts = Vec::new();
    let mut current = expr;
    loop {
        parts.push(current.attr.to_string());
        match &*current.value {
            Expr::Name(ExprName { id, .. }) => {
                parts.push(id.to_string());
                break;
            }
            Expr::Attribute(inner) => {
                current = inner;
            }
            _ => return None,
        }
    }
    parts.reverse();
    Some(parts.join("."))
}

fn decorator_name(expr: &Expr) -> Option<String> {
    if let Some(name) = expr.as_name_expr() {
        Some(name.id.to_string())
    } else if let Some(attr) = expr.as_attribute_expr() {
        attribute_to_name(attr)
    } else {
        None
    }
}

fn is_deprecated_target(name: &str) -> bool {
    matches!(
        name,
        "deprecated" | "warnings.deprecated" | "typing_extensions.deprecated"
    )
}

fn extract_message(call: &ExprCall) -> Option<String> {
    if let Some(arg) = call.arguments.args.first() {
        if let Expr::StringLiteral(ExprStringLiteral { value, .. }) = arg {
            return Some(value.to_string());
        }
    }
    for kw in &call.arguments.keywords {
        if let Some(keyword) = &kw.arg
            && keyword == "msg"
            && let Expr::StringLiteral(ExprStringLiteral { value, .. }) = &kw.value
        {
            return Some(value.to_string());
        }
    }
    None
}

/// Parse a decorator and return its deprecation metadata if it represents `@deprecated`.
pub fn parse_deprecated_decorator(decorator: &Decorator) -> Option<DeprecatedDecoration> {
    let call = decorator.expression.as_call_expr()?;
    let func_name = decorator_name(&call.func)?;
    if !is_deprecated_target(&func_name) {
        return None;
    }
    Some(DeprecatedDecoration::new(extract_message(call)))
}

/// Format a base description (`"`foo` is deprecated"`) with an optional detail message.
pub fn format_deprecated_message(base: impl Into<String>, message: Option<&str>) -> String {
    let base = base.into();
    match message.map(str::trim).filter(|msg| !msg.is_empty()) {
        Some(msg) => format!("{base}: {msg}"),
        None => base,
    }
}

/// Format a base description using metadata from a parsed decorator.
pub fn format_deprecated_with_decoration(
    base: impl Into<String>,
    decoration: Option<&DeprecatedDecoration>,
) -> String {
    format_deprecated_message(base, decoration.and_then(|d| d.message.as_deref()))
}
