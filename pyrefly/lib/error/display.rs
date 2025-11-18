/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::FunctionKind;

use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckKind;
use crate::types::display::TypeDisplayContext;
use crate::types::types::Type;

impl ErrorContext {
    pub fn format(&self) -> String {
        match self {
            Self::BadContextManager(cm) => {
                format!("Cannot use `{cm}` as a context manager")
            }
            Self::UnaryOp(op, target) => {
                format!("Unary `{op}` is not supported on `{target}`")
            }
            Self::BinaryOp(op, left, right) => {
                let ctx = TypeDisplayContext::new(&[left, right]);
                format!(
                    "`{}` is not supported between `{}` and `{}`",
                    op,
                    ctx.display(left),
                    ctx.display(right)
                )
            }
            Self::InplaceBinaryOp(op, left, right) => {
                let ctx = TypeDisplayContext::new(&[left, right]);
                format!(
                    "`{}=` is not supported between `{}` and `{}`",
                    op,
                    ctx.display(left),
                    ctx.display(right)
                )
            }
            Self::Iteration(ty) => format!("Type `{ty}` is not iterable"),
            Self::AsyncIteration(ty) => format!("Type `{ty}` is not an async iterable"),
            Self::Await(ty) => format!("Type `{ty}` is not awaitable"),
            Self::Index(ty) => format!("Cannot index into `{ty}`"),
            Self::SetItem(ty) => format!("Cannot set item in `{ty}`"),
            Self::DelItem(ty) => format!("Cannot delete item in `{ty}`"),
            Self::MatchPositional(ty) => {
                format!("Cannot match positional sub-patterns in `{ty}`")
            }
            Self::ImportNotFound(import) => {
                format!("Could not find import of `{import}`")
            }
            Self::ImportNotTyped(import) => format!("Missing type stubs for `{import}`"),
        }
    }
}

impl TypeCheckKind {
    pub fn format_error(&self, got: &Type, want: &Type, current_module: ModuleName) -> String {
        let mut ctx = TypeDisplayContext::new(&[got, want]);
        match self {
            Self::MagicMethodReturn(cls, func) => {
                ctx.add(cls);
                format!(
                    "Return type `{}` of function `{}.{}` is not assignable to expected return type `{}`",
                    ctx.display(got),
                    ctx.display(cls),
                    func,
                    ctx.display(want),
                )
            }
            Self::AugmentedAssignment => {
                format!(
                    "Augmented assignment produces a value of type `{}`, which is not assignable to `{}`",
                    ctx.display(got),
                    ctx.display(want),
                )
            }
            Self::ImplicitFunctionReturn(has_explicit_return) => {
                if *has_explicit_return {
                    format!(
                        "Function declared to return `{}`, but one or more paths are missing an explicit `return`",
                        ctx.display(want),
                    )
                } else {
                    format!(
                        "Function declared to return `{}` but is missing an explicit `return`",
                        ctx.display(want)
                    )
                }
            }
            Self::ExplicitFunctionReturn => format!(
                "Returned type `{}` is not assignable to declared return type `{}`",
                ctx.display(got),
                ctx.display(want),
            ),
            Self::TypeGuardReturn => format!(
                "Returned type `{}` is not assignable to expected return type `bool` of type guard functions",
                ctx.display(got)
            ),
            Self::CallArgument(param, func_id) => {
                let param_desc = match param {
                    Some(name) => format!("parameter `{name}`"),
                    None => "parameter".to_owned(),
                };
                format!(
                    "Argument `{}` is not assignable to {} with type `{}`{}",
                    ctx.display(got),
                    param_desc,
                    ctx.display(want),
                    function_suffix(func_id.as_ref(), current_module),
                )
            }
            Self::CallVarArgs(arg_is_unpacked, param, func_id) => {
                let arg_desc = if *arg_is_unpacked {
                    "Unpacked argument"
                } else {
                    "Argument"
                };
                let param_desc = match param {
                    Some(name) => format!("parameter `*{name}` with type"),
                    None => "varargs type".to_owned(),
                };
                format!(
                    "{} `{}` is not assignable to {} `{}`{}",
                    arg_desc,
                    ctx.display(got),
                    param_desc,
                    ctx.display(want),
                    function_suffix(func_id.as_ref(), current_module),
                )
            }
            Self::CallKwArgs(arg, param, func_id) => {
                let arg_desc = match arg {
                    Some(arg) => format!("Keyword argument `{arg}` with type"),
                    None => "Unpacked keyword argument".to_owned(),
                };
                let param_desc = match param {
                    Some(param) => format!("parameter `**{param}` with type"),
                    None => "kwargs type".to_owned(),
                };
                format!(
                    "{} `{}` is not assignable to {} `{}`{}",
                    arg_desc,
                    ctx.display(got),
                    param_desc,
                    ctx.display(want),
                    function_suffix(func_id.as_ref(), current_module),
                )
            }
            Self::CallUnpackKwArg(param, func_id) => format!(
                "Unpacked keyword argument `{}` is not assignable to parameter `{}` with type `{}`{}",
                ctx.display(got),
                param,
                ctx.display(want),
                function_suffix(func_id.as_ref(), current_module),
            ),
            Self::FunctionParameterDefault(param) => format!(
                "Default `{}` is not assignable to parameter `{}` with type `{}`",
                ctx.display(got),
                param,
                ctx.display(want),
            ),
            Self::TypedDictKey(key) => format!(
                "`{}` is not assignable to TypedDict key{} with type `{}`",
                ctx.display(got),
                if let Some(key) = key {
                    format!(" `{key}`")
                } else {
                    "".to_owned()
                },
                ctx.display(want),
            ),
            Self::TypedDictUnpacking | Self::TypedDictOpenUnpacking => format!(
                "Unpacked `{}` is not assignable to `{}`",
                ctx.display(got),
                ctx.display(want)
            ),
            Self::Attribute(attr) => format!(
                "`{}` is not assignable to attribute `{}` with type `{}`",
                ctx.display(got),
                attr,
                ctx.display(want),
            ),
            Self::AnnotatedName(var) => format!(
                "`{}` is not assignable to variable `{}` with type `{}`",
                ctx.display(got),
                var,
                ctx.display(want),
            ),
            Self::IterationVariableMismatch(var, real_want) => format!(
                "Cannot use variable `{}` with type `{}` to iterate over elements of type `{}`",
                var,
                ctx.display(real_want),
                ctx.display(got),
            ),
            // In an annotated assignment, the variable, type, and assigned value are all in the
            // same statement, so we can make the error message more concise and assume the context
            // is clear from the surrounding code.
            //
            // TODO(stroxler): In an unpacked assignment to a name we would ideally provide the name in
            // the error message, but without a refactor of `bind_target` we don't have easy access to
            // that information when creating the binding, so we're stuck with just types for now.
            Self::AnnAssign | Self::UnpackedAssign => format!(
                "`{}` is not assignable to `{}`",
                ctx.display(got),
                ctx.display(want)
            ),
            Self::CycleBreaking => format!(
                "`{}` is not assignable to `{}` (caused by inconsistent types when breaking cycles)",
                ctx.display(got),
                ctx.display(want)
            ),
            Self::ExceptionClass => format!(
                "Invalid exception class: `{}` does not inherit from `{}`",
                ctx.display(got),
                ctx.display(want),
            ),
            Self::YieldValue => format!(
                "Type of yielded value `{}` is not assignable to declared return type `{}`",
                ctx.display(got),
                ctx.display(want),
            ),
            Self::YieldFrom => format!(
                "Cannot yield from a generator of type `{}` because it does not match the declared return type `{}`",
                ctx.display(got),
                ctx.display(want),
            ),
            Self::UnexpectedBareYield => format!(
                "Expected to yield a value of type `{}`, but a bare `yield` gives `None` instead",
                ctx.display(want),
            ),
            Self::PostInit => format!(
                "`__post_init__` type `{got}` is not assignable to expected type `{want}` generated from the dataclass's `InitVar` fields"
            ),
            Self::OverloadReturn => format!(
                "Overload return type `{got}` is not assignable to implementation return type `{want}`",
            ),
            Self::OverloadInput(overload_sig, impl_sig) => {
                format!(
                    "Implementation signature `{impl_sig}` does not accept all arguments that overload signature `{overload_sig}` accepts"
                )
            }
            Self::TypeVarSpecialization(name) => {
                format!(
                    "`{got}` is not assignable to upper bound `{want}` of type variable `{name}`"
                )
            }
            Self::Container => {
                format!("`{got}` is not assignable to contained type `{want}`")
            }
        }
    }
}

pub fn function_suffix(func_kind: Option<&FunctionKind>, current_module: ModuleName) -> String {
    match func_kind {
        Some(func) => format!(" in function `{}`", func.format(current_module)),
        None => "".to_owned(),
    }
}
