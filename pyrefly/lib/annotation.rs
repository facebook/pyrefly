/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;

use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::Overload;
use pyrefly_types::types::OverloadType;
use pyrefly_types::types::Type;

/// Render an internal type as valid Python annotation syntax.
pub(crate) fn format_annotation(
    ty: &Type,
    typing_imports: &mut BTreeSet<&'static str>,
    uses_incomplete: &mut bool,
) -> String {
    if ty.is_any() {
        *uses_incomplete = true;
        return "Incomplete".to_owned();
    }
    // Internal function and overload displays use `(args) -> ret` / `Overload[...]`,
    // which are not valid Python annotations.
    if let Some(annotation) = format_callable_type(ty, typing_imports, uses_incomplete) {
        return annotation;
    }
    // Nested internal callable displays would also produce invalid Python, but
    // rewriting them would require reconstructing the enclosing type. Callers
    // can either emit `Incomplete` or decline to add the annotation.
    if ty.any(is_callable_type) {
        *uses_incomplete = true;
        return "Incomplete".to_owned();
    }
    if ty.any(|sub_type| matches!(sub_type, Type::SelfType(_))) {
        typing_imports.insert("Self");
    }
    if ty.any(|sub_type| matches!(sub_type, Type::Literal(_))) {
        typing_imports.insert("Literal");
    }
    let mut display = TypeDisplayContext::new(&[ty]);
    display.render_self_type_as_self();
    display.strip_library_schemas();
    let annotation = display.display(ty).to_string();
    if annotation.contains('@') || annotation.contains("Unknown") {
        *uses_incomplete = true;
        "Incomplete".to_owned()
    } else {
        annotation
    }
}

/// Whether the internal display for this type uses callable-only syntax.
fn is_callable_type(ty: &Type) -> bool {
    match ty {
        Type::Function(_) | Type::Callable(_) | Type::BoundMethod(_) | Type::Overload(_) => true,
        Type::Forall(forall) => matches!(
            &forall.body,
            Forallable::Function(_) | Forallable::Callable(_)
        ),
        _ => false,
    }
}

/// Render a callable-typed value as `typing.Callable[...]`.
fn format_callable_type(
    ty: &Type,
    typing_imports: &mut BTreeSet<&'static str>,
    uses_incomplete: &mut bool,
) -> Option<String> {
    match ty {
        Type::Function(func) => Some(callable_from_signature(
            &func.signature,
            typing_imports,
            uses_incomplete,
        )),
        Type::Callable(callable) => Some(callable_from_signature(
            callable,
            typing_imports,
            uses_incomplete,
        )),
        Type::BoundMethod(method) => match &method.func {
            BoundMethodType::Function(func) => {
                let signature = func.signature.strip_first_param().expect(
                    "BoundMethod::Function should always have at least a self/cls parameter",
                );
                Some(callable_from_signature(
                    &signature,
                    typing_imports,
                    uses_incomplete,
                ))
            }
            BoundMethodType::Forall(forall) => Some(callable_ellipsis(
                &forall.body.signature.ret,
                typing_imports,
                uses_incomplete,
            )),
            BoundMethodType::Overload(overload) => {
                Some(format_overload(overload, typing_imports, uses_incomplete))
            }
        },
        Type::Overload(overload) => {
            Some(format_overload(overload, typing_imports, uses_incomplete))
        }
        Type::Forall(forall) => match &forall.body {
            Forallable::Function(func) => Some(callable_ellipsis(
                &func.signature.ret,
                typing_imports,
                uses_incomplete,
            )),
            Forallable::Callable(callable) => Some(callable_ellipsis(
                &callable.ret,
                typing_imports,
                uses_incomplete,
            )),
            Forallable::TypeAlias(_) => None,
        },
        _ => None,
    }
}

/// Preserve required positional parameters and elide signatures that `Callable` cannot express.
fn callable_from_signature(
    signature: &Callable,
    typing_imports: &mut BTreeSet<&'static str>,
    uses_incomplete: &mut bool,
) -> String {
    typing_imports.insert("Callable");
    let ret = format_annotation(&signature.ret, typing_imports, uses_incomplete);
    match &signature.params {
        Params::List(params)
            if params.items().iter().all(|param| {
                matches!(
                    param,
                    Param::PosOnly(_, _, Required::Required) | Param::Pos(_, _, Required::Required)
                )
            }) =>
        {
            let rendered = params
                .items()
                .iter()
                .map(|param| format_annotation(param.as_type(), typing_imports, uses_incomplete))
                .collect::<Vec<_>>();
            format!("Callable[[{}], {}]", rendered.join(", "), ret)
        }
        _ => format!("Callable[..., {ret}]"),
    }
}

/// Render a callable whose parameter list cannot be faithfully expressed.
fn callable_ellipsis(
    ret: &Type,
    typing_imports: &mut BTreeSet<&'static str>,
    uses_incomplete: &mut bool,
) -> String {
    typing_imports.insert("Callable");
    let ret = format_annotation(ret, typing_imports, uses_incomplete);
    format!("Callable[..., {ret}]")
}

/// Render overloads with a shared return type, or an incomplete callable otherwise.
fn format_overload(
    overload: &Overload,
    typing_imports: &mut BTreeSet<&'static str>,
    uses_incomplete: &mut bool,
) -> String {
    let ret = |signature: &OverloadType| -> Type {
        match signature {
            OverloadType::Function(function) => function.signature.ret.clone(),
            OverloadType::Forall(forall) => forall.body.signature.ret.clone(),
        }
    };
    let first = ret(overload.signatures.first());
    if overload
        .signatures
        .iter()
        .all(|signature| ret(signature) == first)
    {
        callable_ellipsis(&first, typing_imports, uses_incomplete)
    } else {
        typing_imports.insert("Callable");
        *uses_incomplete = true;
        "Callable[..., Incomplete]".to_owned()
    }
}
