/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter::once;
use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_types::literal::Lit;
use pyrefly_types::literal::LitEnum;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ModModule;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::state::lsp::AllOffPartial;
use crate::state::lsp::InlayHintConfig;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::types::Type;

#[derive(Debug)]
pub struct ParameterAnnotation {
    pub text_size: TextSize,
    pub has_annotation: bool,
    pub ty: Option<Type>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamNameMatch<'param> {
    pub name: &'param Name,
    pub is_vararg_repeat: bool,
}

impl<'param> ParamNameMatch<'param> {
    fn new(name: &'param Name, is_vararg_repeat: bool) -> Self {
        Self {
            name,
            is_vararg_repeat,
        }
    }
}

// Re-export normalize_singleton_function_type_into_params which is shared with signature help
pub fn normalize_singleton_function_type_into_params(type_: Type) -> Option<Vec<Param>> {
    let callable = type_.to_callable()?;
    // We will drop the self parameter for signature help
    if let crate::types::callable::Params::List(params_list) = callable.params {
        if let Some(Param::PosOnly(Some(name), _, _) | Param::Pos(name, _, _)) =
            params_list.items().first()
            && (name.as_str() == "self" || name.as_str() == "cls")
        {
            let mut params = params_list.into_items();
            params.remove(0);
            return Some(params);
        }
        return Some(params_list.into_items());
    }
    None
}

impl<'a> Transaction<'a> {
    pub fn inlay_hints(
        &self,
        handle: &Handle,
        inlay_hint_config: InlayHintConfig,
    ) -> Option<Vec<(TextSize, Vec<(String, Option<TextRangeWithModule>)>)>> {
        let is_interesting = |e: &Expr, ty: &Type, class_name: Option<&Name>| {
            !ty.is_any()
                && match e {
                    Expr::Tuple(tuple) => {
                        !tuple.elts.is_empty() && tuple.elts.iter().all(|x| !Ast::is_literal(x))
                    }
                    Expr::Call(ExprCall { func, .. }) => {
                        // Exclude constructor calls
                        if let Expr::Name(name) = &**func
                            && let Some(class_name) = class_name
                        {
                            *name.id() != *class_name
                        } else if let Expr::Attribute(attr) = &**func
                            && let Some(class_name) = class_name
                        {
                            *attr.attr.id() != *class_name
                        } else {
                            true
                        }
                    }
                    Expr::Attribute(ExprAttribute {
                        box value, attr, ..
                    }) if let Type::Literal(Lit::Enum(box LitEnum { class, member, .. })) = ty => {
                        // Exclude enum literals
                        match value {
                            Expr::Name(object) => {
                                *object.id() != *class.name() || *attr.id() != *member
                            }
                            Expr::Attribute(ExprAttribute { attr: object, .. }) => {
                                *object.id() != *class.name() || *attr.id() != *member
                            }
                            _ => true,
                        }
                    }
                    _ => !Ast::is_literal(e),
                }
        };
        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                key @ Key::ReturnType(id) => {
                    if inlay_hint_config.function_return_types {
                        match bindings.get(bindings.key_to_idx(&Key::Definition(*id))) {
                            Binding::Function(x, _pred, _class_meta) => {
                                if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                    && let Some(mut ty) = self.get_type(handle, key)
                                    && !ty.is_any()
                                {
                                    let fun = bindings.get(bindings.get(*x).undecorated_idx);
                                    if fun.def.is_async
                                        && let Some(Some((_, _, return_ty))) = self
                                            .ad_hoc_solve(handle, |solver| {
                                                solver.unwrap_coroutine(&ty)
                                            })
                                    {
                                        ty = return_ty;
                                    }
                                    // Use get_types_with_locations to get type parts with location info
                                    let type_parts = ty.get_types_with_locations();
                                    let label_parts = once((" -> ".to_owned(), None))
                                        .chain(
                                            type_parts
                                                .iter()
                                                .map(|(text, loc)| (text.clone(), loc.clone())),
                                        )
                                        .collect();
                                    res.push((fun.def.parameters.range.end(), label_parts));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                key @ Key::Definition(_)
                    if inlay_hint_config.variable_types
                        && let Some(ty) = self.get_type(handle, key) =>
                {
                    let e = match bindings.get(idx) {
                        Binding::NameAssign {
                            annotation: None,
                            expr: e,
                            ..
                        } => Some(&**e),
                        Binding::Expr(None, e) => Some(e),
                        _ => None,
                    };
                    // If the inferred type is a class type w/ no type arguments and the
                    // RHS is a call to a function that's the same name as the inferred class,
                    // we assume it's a constructor and do not display an inlay hint
                    let class_name = if let Type::ClassType(cls) = &ty
                        && cls.targs().is_empty()
                    {
                        Some(cls.name())
                    } else {
                        None
                    };
                    if let Some(e) = e
                        && is_interesting(e, &ty, class_name)
                    {
                        // Use get_types_with_locations to get type parts with location info
                        let type_parts = ty.get_types_with_locations();
                        let label_parts = once((": ".to_owned(), None))
                            .chain(
                                type_parts
                                    .iter()
                                    .map(|(text, loc)| (text.clone(), loc.clone())),
                            )
                            .collect();
                        res.push((key.range().end(), label_parts));
                    }
                }
                _ => {}
            }
        }

        if inlay_hint_config.call_argument_names != AllOffPartial::Off {
            res.extend(
                self.add_inlay_hints_for_positional_function_args(handle)
                    .into_iter()
                    .map(|(pos, text)| (pos, vec![(text, None)])),
            );
        }

        Some(res)
    }

    fn collect_function_calls_from_ast(module: Arc<ModModule>) -> Vec<ExprCall> {
        fn collect_function_calls(x: &Expr, calls: &mut Vec<ExprCall>) {
            if let Expr::Call(call) = x {
                calls.push(call.clone());
            }
            x.recurse(&mut |x| collect_function_calls(x, calls));
        }

        let mut function_calls = Vec::new();
        module.visit(&mut |x| collect_function_calls(x, &mut function_calls));
        function_calls
    }

    fn add_inlay_hints_for_positional_function_args(
        &self,
        handle: &Handle,
    ) -> Vec<(TextSize, String)> {
        let mut param_hints: Vec<(TextSize, String)> = Vec::new();

        if let Some(mod_module) = self.get_ast(handle) {
            let function_calls = Self::collect_function_calls_from_ast(mod_module);

            for call in function_calls {
                if let Some(answers) = self.get_answers(handle) {
                    let callee_type = if let Some((overloads, chosen_idx)) =
                        answers.get_all_overload_trace(call.arguments.range)
                    {
                        // If we have overload information, use the chosen overload
                        overloads
                            .get(chosen_idx.unwrap_or_default())
                            .map(|c| Type::Callable(Box::new(c.clone())))
                    } else {
                        // Otherwise, try to get the type of the callee directly
                        answers.get_type_trace(call.func.range())
                    };

                    if let Some(params) =
                        callee_type.and_then(normalize_singleton_function_type_into_params)
                    {
                        for (arg_idx, arg) in call.arguments.args.iter().enumerate() {
                            // Skip keyword arguments - they already show their parameter name
                            let is_keyword_arg = call
                                .arguments
                                .keywords
                                .iter()
                                .any(|kw| kw.value.range() == arg.range());

                            if !is_keyword_arg
                                && let Some(param_match) =
                                    Self::param_name_for_positional_argument(&params, arg_idx)
                                && !param_match.is_vararg_repeat
                                && param_match.name.as_str() != "self"
                                && param_match.name.as_str() != "cls"
                            {
                                param_hints.push((
                                    arg.range().start(),
                                    format!("{}= ", param_match.name.as_str()),
                                ));
                            }
                        }
                    }
                }
            }
        }

        param_hints.sort_by_key(|(pos, _)| *pos);
        param_hints
    }

    pub(crate) fn param_name_for_positional_argument<'param>(
        params: &'param [Param],
        positional_arg_index: usize,
    ) -> Option<ParamNameMatch<'param>> {
        let mut positional_params_seen = 0;
        for param in params {
            match param {
                Param::PosOnly(name, ..) => {
                    if positional_params_seen == positional_arg_index {
                        return name.as_ref().map(|name| {
                            ParamNameMatch::new(name, /* is_vararg_repeat */ false)
                        });
                    }
                    positional_params_seen += 1;
                }
                Param::Pos(name, ..) => {
                    if positional_params_seen == positional_arg_index {
                        return Some(ParamNameMatch::new(name, false));
                    }
                    positional_params_seen += 1;
                }
                Param::VarArg(name, ..) => {
                    if positional_arg_index >= positional_params_seen {
                        return name.as_ref().map(|name| {
                            ParamNameMatch::new(name, positional_arg_index > positional_params_seen)
                        });
                    }
                    break;
                }
                Param::KwOnly(..) | Param::Kwargs(..) => {}
            }
        }
        None
    }

    fn filter_parameters(
        &self,
        param_with_default: ParameterWithDefault,
        handle: &Handle,
    ) -> Option<ParameterAnnotation> {
        if param_with_default.name() == "self" || param_with_default.name() == "cls" {
            return None;
        }
        let ty = match param_with_default.default() {
            Some(expr) => self.get_type_trace(handle, expr.range()),
            None => None,
        };
        Some(ParameterAnnotation {
            text_size: param_with_default.parameter.range().end(),
            ty,
            has_annotation: param_with_default.annotation().is_some(),
        })
    }

    fn collect_types_from_callees(&self, range: TextRange, handle: &Handle) -> Vec<Type> {
        fn callee_at(mod_module: Arc<ModModule>, position: TextSize) -> Option<ExprCall> {
            fn f(x: &Expr, find: TextSize, res: &mut Option<ExprCall>) {
                if let Expr::Call(call) = x
                    && call.func.range().contains_inclusive(find)
                {
                    f(call.func.as_ref(), find, res);
                    if res.is_some() {
                        return;
                    }
                    *res = Some(call.clone());
                } else {
                    x.recurse(&mut |x| f(x, find, res));
                }
            }
            let mut res = None;
            mod_module.visit(&mut |x| f(x, position, &mut res));
            res
        }
        match self.get_ast(handle) {
            Some(mod_module) => {
                let callee = callee_at(mod_module, range.start());
                match callee {
                    Some(ExprCall {
                        arguments: args, ..
                    }) => args
                        .args
                        .iter()
                        .filter_map(|arg| self.get_type_trace(handle, arg.range()))
                        .collect(),
                    None => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    fn collect_references(
        &self,
        handle: &Handle,
        idx: pyrefly_graph::index::Idx<Key>,
        bindings: crate::binding::bindings::Bindings,
        transaction: &mut CancellableTransaction,
    ) -> Vec<(pyrefly_python::module::Module, Vec<TextRange>)> {
        if let Key::Definition(id) = bindings.idx_to_key(idx)
            && let Some(module_info) = self.get_module_info(handle)
        {
            let definition_kind = crate::state::lsp::DefinitionMetadata::VariableOrAttribute(None);
            if let Ok(references) = transaction.find_global_references_from_definition(
                handle.sys_info(),
                definition_kind,
                TextRangeWithModule::new(module_info, id.range()),
            ) {
                return references;
            }
        }
        Vec::new()
    }

    pub fn infer_parameter_annotations(
        &self,
        handle: &Handle,
        cancellable_transaction: &mut CancellableTransaction,
    ) -> Vec<ParameterAnnotation> {
        if let Some(bindings) = self.get_bindings(handle) {
            let transaction = cancellable_transaction;
            fn transpose<T: Clone>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
                if v.is_empty() {
                    return Vec::new();
                }
                let max_len = v.iter().map(|row| row.len()).max().unwrap();
                let mut result = vec![Vec::new(); max_len];
                for row in v {
                    for (i, elem) in row.into_iter().enumerate() {
                        result[i].push(elem);
                    }
                }
                result
            }
            fn zip_types(
                inferred_types: Vec<Vec<Type>>,
                function_arguments: Vec<ParameterAnnotation>,
            ) -> Vec<ParameterAnnotation> {
                let zipped_inferred_types: Vec<Vec<Type>> = transpose(inferred_types);
                let types: Vec<(ParameterAnnotation, Vec<Type>)> =
                    match zipped_inferred_types.is_empty() {
                        true => function_arguments
                            .into_iter()
                            .map(
                                |arg: ParameterAnnotation| -> (ParameterAnnotation, Vec<Type>) {
                                    (arg, vec![])
                                },
                            )
                            .collect(),
                        false => function_arguments
                            .into_iter()
                            .zip(zipped_inferred_types)
                            .collect(),
                    };

                types
                    .into_iter()
                    .map(|(arg, mut ty)| {
                        let mut arg = arg;
                        if let Some(default_type) = arg.ty {
                            ty.push(default_type)
                        }
                        if ty.len() == 1 {
                            arg.ty = Some(ty[0].clone());
                        } else {
                            let ty = ty.into_iter().filter(|x| !x.is_any()).collect();
                            arg.ty = Some(Type::union(ty));
                        }
                        arg
                    })
                    .collect()
            }

            bindings
                .keys::<Key>()
                .flat_map(|idx| {
                    let binding = bindings.get(idx);
                    // Check if this binding is a function
                    if let Binding::Function(key_function, _, _) = binding {
                        let binding_func =
                            bindings.get(bindings.get(*key_function).undecorated_idx);
                        let args = binding_func.def.parameters.args.clone();
                        let func_args: Vec<ParameterAnnotation> = args
                            .into_iter()
                            .filter_map(|param_with_default| {
                                self.filter_parameters(param_with_default, handle)
                            })
                            .collect();
                        let references =
                            self.collect_references(handle, idx, bindings.clone(), transaction);
                        let ranges: Vec<&TextRange> =
                            references.iter().flat_map(|(_, range)| range).collect();
                        let inferred_types = ranges
                            .into_iter()
                            .map(|range| self.collect_types_from_callees(*range, handle));
                        zip_types(inferred_types.collect(), func_args)
                    } else {
                        vec![]
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    pub fn inferred_types(
        &self,
        handle: &Handle,
        return_types: bool,
        containers: bool,
    ) -> Option<Vec<(TextSize, Type, crate::state::lsp::AnnotationKind)>> {
        let is_interesting_type = |x: &Type| !x.is_any();
        let is_interesting_expr = |x: &Expr| !Ast::is_literal(x);
        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                // Return Annotation
                key @ Key::ReturnType(id) if return_types => {
                    match bindings.get(bindings.key_to_idx(&Key::Definition(*id))) {
                        Binding::Function(x, _pred, _class_meta) => {
                            if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                && let Some(ty) = self.get_type(handle, key)
                                && is_interesting_type(&ty)
                            {
                                let fun = bindings.get(bindings.get(*x).undecorated_idx);
                                res.push((
                                    fun.def.parameters.range.end(),
                                    ty,
                                    crate::state::lsp::AnnotationKind::Return,
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                // Only annotate empty containers for now
                key @ Key::Definition(_) if containers => {
                    if let Some(ty) = self.get_type(handle, key) {
                        let e = match bindings.get(idx) {
                            Binding::NameAssign {
                                annotation: None,
                                expr: e,
                                ..
                            } => match &**e {
                                Expr::List(ExprList { elts, .. }) => {
                                    if elts.is_empty() {
                                        Some(&**e)
                                    } else {
                                        None
                                    }
                                }
                                Expr::Dict(ExprDict { items, .. }) => {
                                    if items.is_empty() {
                                        Some(&**e)
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some(e) = e
                            && is_interesting_expr(e)
                            && is_interesting_type(&ty)
                        {
                            res.push((
                                key.range().end(),
                                ty,
                                crate::state::lsp::AnnotationKind::Variable,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use ruff_python_ast::name::Name;

    use super::Transaction;
    use crate::types::callable::Param;
    use crate::types::callable::Required;
    use crate::types::types::AnyStyle;
    use crate::types::types::Type;

    fn any_type() -> Type {
        Type::Any(AnyStyle::Explicit)
    }

    #[test]
    fn param_name_for_positional_argument_marks_vararg_repeats() {
        let params = vec![
            Param::Pos(Name::new_static("x"), any_type(), Required::Required),
            Param::VarArg(Some(Name::new_static("columns")), any_type()),
            Param::KwOnly(Name::new_static("kw"), any_type(), Required::Required),
        ];

        assert_eq!(match_summary(&params, 0), Some(("x", false)));
        assert_eq!(match_summary(&params, 1), Some(("columns", false)));
        assert_eq!(match_summary(&params, 3), Some(("columns", true)));
    }

    #[test]
    fn param_name_for_positional_argument_handles_missing_names() {
        let params = vec![
            Param::PosOnly(None, any_type(), Required::Required),
            Param::VarArg(None, any_type()),
        ];

        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 0).is_none());
        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 1).is_none());
        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 5).is_none());
    }

    #[test]
    fn duplicate_vararg_hints_are_not_emitted() {
        let params = vec![
            Param::Pos(Name::new_static("s"), any_type(), Required::Required),
            Param::VarArg(Some(Name::new_static("args")), any_type()),
            Param::KwOnly(Name::new_static("a"), any_type(), Required::Required),
        ];

        let labels: Vec<&str> = (0..4)
            .filter_map(|idx| {
                Transaction::<'static>::param_name_for_positional_argument(&params, idx)
            })
            .filter(|match_| !match_.is_vararg_repeat)
            .map(|match_| match_.name.as_str())
            .collect();

        assert_eq!(labels, vec!["s", "args"]);
    }

    fn match_summary(params: &[Param], idx: usize) -> Option<(&str, bool)> {
        Transaction::<'static>::param_name_for_positional_argument(params, idx)
            .map(|match_| (match_.name.as_str(), match_.is_vararg_repeat))
    }
}
