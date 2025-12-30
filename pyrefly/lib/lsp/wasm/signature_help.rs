/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use itertools::Itertools;
use lsp_types::Documentation;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::ParameterInformation;
use lsp_types::ParameterLabel;
use lsp_types::SignatureHelp;
use lsp_types::SignatureInformation;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::docstring::parse_parameter_documentation;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::state::lsp::FindPreference;
use crate::state::lsp::visit_keyword_arguments_until_match;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::callable::Params;
use crate::types::types::Type;

/// The currently active argument in a function call for signature help.
#[derive(Debug)]
pub(crate) enum ActiveArgument {
    /// The cursor is within an existing positional argument at the given index.
    Positional(usize),
    /// The cursor is within a keyword argument whose name is provided.
    Keyword(Name),
    /// The cursor is in the argument list but not inside any argument expression yet.
    Next(usize),
}

impl Transaction<'_> {
    fn visit_finding_signature_range(
        x: &Expr,
        find: TextSize,
        res: &mut Option<(TextRange, TextRange, ActiveArgument)>,
    ) {
        if let Expr::Call(call) = x
            && call.arguments.range.contains_inclusive(find)
        {
            if Self::visit_positional_signature_args(call, find, res) {
                return;
            }
            if Self::visit_keyword_signature_args(call, find, res) {
                return;
            }
            if res.is_none() {
                *res = Some((
                    call.func.range(),
                    call.arguments.range,
                    ActiveArgument::Next(call.arguments.len()),
                ));
            }
        } else {
            x.recurse(&mut |x| Self::visit_finding_signature_range(x, find, res));
        }
    }

    fn visit_positional_signature_args(
        call: &ExprCall,
        find: TextSize,
        res: &mut Option<(TextRange, TextRange, ActiveArgument)>,
    ) -> bool {
        for (i, arg) in call.arguments.args.as_ref().iter().enumerate() {
            if arg.range().contains_inclusive(find) {
                Self::visit_finding_signature_range(arg, find, res);
                if res.is_some() {
                    return true;
                }
                *res = Some((
                    call.func.range(),
                    call.arguments.range,
                    ActiveArgument::Positional(i),
                ));
                return true;
            }
        }
        false
    }

    fn visit_keyword_signature_args(
        call: &ExprCall,
        find: TextSize,
        res: &mut Option<(TextRange, TextRange, ActiveArgument)>,
    ) -> bool {
        let kwarg_start_idx = call.arguments.args.len();
        visit_keyword_arguments_until_match(call, |j, kw| {
            if kw.range.contains_inclusive(find) {
                Self::visit_finding_signature_range(&kw.value, find, res);
                if res.is_some() {
                    return true;
                }
                let active_argument = match kw.arg.as_ref() {
                    Some(identifier) => ActiveArgument::Keyword(identifier.id.clone()),
                    None => ActiveArgument::Positional(kwarg_start_idx + j),
                };
                *res = Some((call.func.range(), call.arguments.range, active_argument));
                true
            } else {
                false
            }
        })
    }

    /// Finds the callable(s) (multiple if overloads exist) at position in document, returning them, chosen overload index, and arg index
    pub(crate) fn get_callables_from_call(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<(Vec<Type>, usize, ActiveArgument, TextRange)> {
        let mod_module = self.get_ast(handle)?;
        let mut res = None;
        mod_module.visit(&mut |x| Self::visit_finding_signature_range(x, position, &mut res));
        let (callee_range, call_args_range, mut active_argument) = res?;
        // When the cursor is in the argument list but not inside any argument yet,
        // estimate the would-be positional index by counting commas up to the cursor.
        // This keeps signature help useful even before the user starts typing the next arg.
        if let ActiveArgument::Next(index) = &mut active_argument
            && let Some(next_index) =
                self.count_argument_separators_before(handle, call_args_range, position)
        {
            *index = next_index;
        }
        let answers = self.get_answers(handle)?;
        if let Some((overloads, chosen_overload_index)) =
            answers.get_all_overload_trace(call_args_range)
        {
            let callables = overloads.into_map(|callable| Type::Callable(Box::new(callable)));
            Some((
                callables,
                chosen_overload_index.unwrap_or_default(),
                active_argument,
                callee_range,
            ))
        } else {
            answers
                .get_type_trace(callee_range)
                .map(|t| self.coerce_type_to_callable(handle, t))
                .map(|t| (vec![t], 0, active_argument, callee_range))
        }
    }

    pub(crate) fn get_signature_help_at(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<SignatureHelp> {
        self.get_callables_from_call(handle, position).map(
            |(callables, chosen_overload_index, active_argument, callee_range)| {
                let parameter_docs = self.parameter_documentation_for_callee(handle, callee_range);
                let function_docstring = self.function_docstring_for_callee(handle, callee_range);
                let signatures = callables
                    .into_iter()
                    .map(|t| {
                        Self::create_signature_information(
                            t,
                            &active_argument,
                            parameter_docs.as_ref(),
                            function_docstring.as_ref(),
                        )
                    })
                    .collect_vec();
                let active_parameter = signatures
                    .get(chosen_overload_index)
                    .and_then(|info| info.active_parameter);
                SignatureHelp {
                    signatures,
                    active_signature: Some(chosen_overload_index as u32),
                    active_parameter,
                }
            },
        )
    }

    pub(crate) fn parameter_documentation_for_callee(
        &self,
        handle: &Handle,
        callee_range: TextRange,
    ) -> Option<HashMap<String, String>> {
        let position = callee_range.start();
        let docstring = self
            .find_definition(
                handle,
                position,
                FindPreference {
                    prefer_pyi: false,
                    ..Default::default()
                },
            )
            .into_iter()
            .find_map(|item| {
                item.docstring_range
                    .map(|range| (range, item.module.clone()))
            })
            .or_else(|| {
                self.find_definition(handle, position, FindPreference::default())
                    .into_iter()
                    .find_map(|item| {
                        item.docstring_range
                            .map(|range| (range, item.module.clone()))
                    })
            })?;
        let (range, module) = docstring;
        let docs = parse_parameter_documentation(module.code_at(range));
        if docs.is_empty() { None } else { Some(docs) }
    }

    /// Extract the full docstring for a callee to display in signature help.
    pub(crate) fn function_docstring_for_callee(
        &self,
        handle: &Handle,
        callee_range: TextRange,
    ) -> Option<Docstring> {
        let position = callee_range.start();
        let (docstring_range, module) = self
            .find_definition(
                handle,
                position,
                FindPreference {
                    prefer_pyi: false,
                    ..Default::default()
                },
            )
            .into_iter()
            .find_map(|item| {
                item.docstring_range
                    .map(|range| (range, item.module.clone()))
            })
            .or_else(|| {
                self.find_definition(handle, position, FindPreference::default())
                    .into_iter()
                    .find_map(|item| {
                        item.docstring_range
                            .map(|range| (range, item.module.clone()))
                    })
            })?;
        Some(Docstring(docstring_range, module))
    }

    pub(crate) fn create_signature_information(
        type_: Type,
        active_argument: &ActiveArgument,
        parameter_docs: Option<&HashMap<String, String>>,
        function_docstring: Option<&Docstring>,
    ) -> SignatureInformation {
        let type_ = type_.deterministic_printing();
        let label = type_.as_lsp_string(LspDisplayMode::SignatureHelp);
        let (parameters, active_parameter) = if let Some(params) =
            Self::normalize_singleton_function_type_into_params(type_)
        {
            // Create a type display context for consistent parameter formatting
            let param_types: Vec<&Type> = params.iter().map(|p| p.as_type()).collect();
            let mut type_ctx = TypeDisplayContext::new(&param_types);
            type_ctx.set_lsp_display_mode(LspDisplayMode::SignatureHelp);

            let active_parameter =
                Self::active_parameter_index(&params, active_argument).map(|idx| idx as u32);

            let parameter_info: Vec<ParameterInformation> = params
                .iter()
                .map(|param| ParameterInformation {
                    label: ParameterLabel::Simple(param.format_for_signature(&type_ctx)),
                    documentation: param
                        .name()
                        .and_then(|name| parameter_docs.and_then(|docs| docs.get(name.as_str())))
                        .map(|text| {
                            lsp_types::Documentation::MarkupContent(lsp_types::MarkupContent {
                                kind: lsp_types::MarkupKind::Markdown,
                                value: text.clone(),
                            })
                        }),
                })
                .collect();
            (Some(parameter_info), active_parameter)
        } else {
            (None, None)
        };
        SignatureInformation {
            label,
            documentation: function_docstring.map(|docstring| {
                Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: docstring.resolve(),
                })
            }),
            parameters,
            active_parameter,
        }
    }

    pub(crate) fn active_parameter_index(
        params: &[Param],
        active_argument: &ActiveArgument,
    ) -> Option<usize> {
        match active_argument {
            ActiveArgument::Positional(index) | ActiveArgument::Next(index) => {
                (*index < params.len()).then_some(*index)
            }
            ActiveArgument::Keyword(name) => params
                .iter()
                .position(|param| param.name().is_some_and(|param_name| param_name == name)),
        }
    }

    pub(crate) fn count_argument_separators_before(
        &self,
        handle: &Handle,
        arguments_range: TextRange,
        position: TextSize,
    ) -> Option<usize> {
        let module = self.get_module_info(handle)?;
        let contents = module.contents();
        let len = contents.len();
        let start = arguments_range.start().to_usize().min(len);
        let end = arguments_range.end().to_usize().min(len);
        let pos = position.to_usize().clamp(start, end);
        contents
            .get(start..pos)
            .map(|slice| slice.bytes().filter(|&b| b == b',').count())
            .or(Some(0))
    }

    pub(crate) fn normalize_singleton_function_type_into_params(type_: Type) -> Option<Vec<Param>> {
        let callable = type_.to_callable()?;
        if let Params::List(params_list) = callable.params {
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
}
