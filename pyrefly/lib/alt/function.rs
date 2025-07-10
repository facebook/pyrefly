/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use pyrefly_python::dunder;
use pyrefly_python::module_path::ModuleStyle;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Binding;
use crate::binding::binding::FunctionStubOrImpl;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyFunction;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::error::collector::ErrorCollector;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::kind::ErrorKind;
use crate::graph::index::Idx;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::callable::Callable;
use crate::types::callable::FuncFlags;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::keywords::DataclassTransformKeywords;
use crate::types::types::CalleeKind;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Overload;
use crate::types::types::OverloadType;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn solve_function_binding(
        &self,
        idx: Idx<KeyFunction>,
        predecessor: &mut Option<Idx<Key>>,
        class_metadata: Option<&Idx<KeyClassMetadata>>,
        errors: &ErrorCollector,
    ) -> Type {
        // Overloads in .pyi should not have an implementation.
        let skip_implementation = self.module_info().path().style() == ModuleStyle::Interface
            || class_metadata.is_some_and(|idx| self.get_idx(*idx).is_protocol());
        let def = self.get_idx(idx);
        if def.metadata.flags.is_overload {
            // This function is decorated with @overload. We should warn if this function is actually called anywhere.
            let successor = self.bindings().get(idx).successor;
            let ty = def.ty.clone();
            if successor.is_none() {
                // This is the last definition in the chain. We should produce an overload type.
                let last_range = def.id_range;
                let has_impl = def.stub_or_impl == FunctionStubOrImpl::Impl;
                let mut acc = Vec1::new((last_range, ty));
                let mut first = def;
                let mut impl_before_overload_range = None;
                while let Some(def) = self.step_pred(predecessor) {
                    if def.metadata.flags.is_overload {
                        acc.push((def.id_range, def.ty.clone()));
                        first = def;
                    } else {
                        impl_before_overload_range = Some(def.id_range);
                        break;
                    }
                }
                if !skip_implementation {
                    if let Some(range) = impl_before_overload_range {
                        self.error(
                            errors,
                            range,
                            ErrorKind::InvalidOverload,
                            None,
                            "@overload declarations must come before function implementation"
                                .to_owned(),
                        );
                    } else if has_impl {
                        self.error(
                            errors,
                            last_range,
                            ErrorKind::InvalidOverload,
                            None,
                            "@overload decorator should not be used on function implementation"
                                .to_owned(),
                        );
                    } else {
                        self.error(
                            errors,
                            first.id_range,
                            ErrorKind::InvalidOverload,
                            None,
                            "Overloaded function must have an implementation".to_owned(),
                        );
                    }
                }
                if acc.len() == 1 {
                    self.error(
                        errors,
                        first.id_range,
                        ErrorKind::InvalidOverload,
                        None,
                        "Overloaded function needs at least two @overload declarations".to_owned(),
                    );
                    acc.split_off_first().0.1
                } else {
                    acc.reverse();
                    Type::Overload(Overload {
                        signatures: self.extract_signatures(
                            first.metadata.kind.as_func_id().func,
                            acc,
                            errors,
                        ),
                        metadata: Box::new(first.metadata.clone()),
                    })
                }
            } else {
                ty
            }
        } else {
            let mut acc = Vec::new();
            let mut first = def;
            while let Some(def) = self.step_pred(predecessor)
                && def.metadata.flags.is_overload
            {
                acc.push((def.id_range, def.ty.clone()));
                first = def;
            }
            acc.reverse();
            if let Ok(defs) = Vec1::try_from_vec(acc) {
                if defs.len() == 1 {
                    self.error(
                        errors,
                        first.id_range,
                        ErrorKind::InvalidOverload,
                        None,
                        "Overloaded function needs at least two @overload declarations".to_owned(),
                    );
                    defs.split_off_first().0.1
                } else {
                    Type::Overload(Overload {
                        signatures: self.extract_signatures(
                            first.metadata.kind.as_func_id().func,
                            defs,
                            errors,
                        ),
                        metadata: Box::new(first.metadata.clone()),
                    })
                }
            } else {
                first.ty.clone()
            }
        }
    }

    pub fn function_definition(
        &self,
        def: &StmtFunctionDef,
        stub_or_impl: FunctionStubOrImpl,
        class_key: Option<&Idx<KeyClass>>,
        decorators: &[Idx<Key>],
        legacy_tparams: &[Idx<KeyLegacyTypeParam>],
        errors: &ErrorCollector,
    ) -> Arc<DecoratedFunction> {
        let defining_cls = class_key.and_then(|k| self.get_idx(*k).0.dupe());
        let mut self_type = if def.name.id == dunder::NEW || def.name.id == dunder::INIT_SUBCLASS {
            // __new__ and __init_subclass__ are staticmethods, and do not take a self parameter.
            None
        } else {
            defining_cls
                .as_ref()
                .map(|cls| Type::SelfType(self.as_class_type_unchecked(cls)))
        };

        let mut is_overload = false;
        let mut is_staticmethod = false;
        let mut is_classmethod = false;
        let mut is_deprecated = false;
        let mut is_property_getter = false;
        let mut is_property_setter_with_getter = None;
        let mut has_enum_member_decoration = false;
        let mut is_override = false;
        let mut has_final_decoration = false;
        let mut dataclass_transform_metadata = None;
        let decorators = decorators
            .iter()
            .filter(|k| {
                let decorator = self.get_idx(**k);
                let decorator_ty = decorator.ty();
                match decorator_ty.callee_kind() {
                    Some(CalleeKind::Function(FunctionKind::Overload)) => {
                        is_overload = true;
                        false
                    }
                    Some(CalleeKind::Class(ClassKind::StaticMethod)) => {
                        is_staticmethod = true;
                        false
                    }
                    Some(CalleeKind::Class(ClassKind::ClassMethod)) => {
                        is_classmethod = true;
                        false
                    }
                    Some(CalleeKind::Class(ClassKind::Property)) => {
                        is_property_getter = true;
                        false
                    }
                    Some(CalleeKind::Class(ClassKind::EnumMember)) => {
                        has_enum_member_decoration = true;
                        false
                    }
                    Some(CalleeKind::Function(FunctionKind::Override)) => {
                        is_override = true;
                        false
                    }
                    Some(CalleeKind::Function(FunctionKind::Final)) => {
                        has_final_decoration = true;
                        false
                    }
                    _ if matches!(decorator_ty, Type::ClassType(cls) if cls.has_qname("warnings", "deprecated")) => {
                        is_deprecated = true;
                        false
                    }
                    _ if decorator_ty.is_property_setter_decorator() => {
                        // When the `setter` attribute is accessed on a property, we return the
                        // getter with the is_property_setter_decorator flag set to true. See
                        // AnswersSolver::lookup_attr_from_attribute_base for details.
                        is_property_setter_with_getter = Some(decorator.arc_clone_ty());
                        false
                    }
                    _ if let Type::KwCall(call) = decorator_ty && call.has_function_kind(FunctionKind::DataclassTransform) => {
                        dataclass_transform_metadata = Some(DataclassTransformKeywords::from_type_map(&call.keywords));
                        false
                    }
                    _ => true,
                }
            })
            .collect::<Vec<_>>();

        // Look for a @classmethod or @staticmethod decorator and change the "self" type
        // accordingly. This is not totally correct, since it doesn't account for chaining
        // decorators, or weird cases like both decorators existing at the same time.
        if is_staticmethod {
            self_type = None;
        } else if is_classmethod {
            self_type = self_type.map(Type::type_form);
        }

        // Determine the type of the parameter based on its binding. Left is annotated parameter, right is unannotated
        let mut get_param_ty = |name: &Identifier, default: Option<&Expr>| {
            let ty = match self.bindings().get_function_param(name) {
                Either::Left(idx) => {
                    // If the parameter is annotated, we check the default value against the annotation
                    let param_ty = self.get_idx(idx).annotation.get_type().clone();
                    if let Some(default) = default
                        && (stub_or_impl != FunctionStubOrImpl::Stub
                            || !matches!(default, Expr::EllipsisLiteral(_)))
                    {
                        self.expr(
                            default,
                            Some((&param_ty, &|| {
                                TypeCheckContext::of_kind(TypeCheckKind::FunctionParameterDefault(
                                    name.id.clone(),
                                ))
                            })),
                            errors,
                        );
                    }
                    param_ty
                }
                Either::Right(var) => {
                    // If this is the first parameter and there is a self type, solve to `Self`.
                    // We only try to solve the first param for now. Other unannotated params
                    // are also Var. If a default value of type T is provided, it will resolve to Any | T.
                    // Otherwise, it will be forced to Any
                    if let Some(ty) = &self_type {
                        self.solver()
                            .is_subset_eq(&var.to_type(), ty, self.type_order());
                    } else if let Some(default) = default
                        && (stub_or_impl != FunctionStubOrImpl::Stub
                            || !matches!(default, Expr::EllipsisLiteral(_)))
                    {
                        let default_ty = self.expr(default, None, errors);
                        self.solver().is_subset_eq(
                            &self.union(Type::any_implicit(), default_ty),
                            &var.to_type(),
                            self.type_order(),
                        );
                    }
                    self.solver().force_var(var)
                }
            };
            self_type = None; // Stop using `self` type solve Var params after the first param.
            ty
        };
        let mut paramspec_args = None;
        let mut paramspec_kwargs = None;
        let mut params = Vec::with_capacity(def.parameters.len());
        params.extend(def.parameters.posonlyargs.iter().map(|x| {
            let ty = get_param_ty(&x.parameter.name, x.default.as_deref());
            let required = if x.default.is_some() {
                Required::Optional
            } else {
                Required::Required
            };
            Param::PosOnly(Some(x.parameter.name.id.clone()), ty, required)
        }));

        // See: https://typing.python.org/en/latest/spec/historical.html#positional-only-parameters
        let is_historical_args_usage =
            def.parameters.posonlyargs.is_empty() && def.parameters.kwonlyargs.is_empty();
        let mut seen_keyword_args = false;

        params.extend(def.parameters.args.iter().map(|x| {
            let ty = get_param_ty(&x.parameter.name, x.default.as_deref());
            let required = if x.default.is_some() {
                Required::Optional
            } else {
                Required::Required
            };

            // If the parameter begins but does not end with "__", it is a positional-only parameter.
            // See: https://typing.python.org/en/latest/spec/historical.html#positional-only-parameters
            if is_historical_args_usage
                && x.parameter.name.starts_with("__")
                && !x.parameter.name.ends_with("__")
            {
                if seen_keyword_args {
                    self.error(
                        errors,
                        x.parameter.name.range,
                        ErrorKind::BadFunctionDefinition,
                        None,
                        format!(
                            "Positional-only parameter `{}` cannot appear after keyword parameters",
                            x.parameter.name
                        ),
                    );
                }

                Param::PosOnly(Some(x.parameter.name.id.clone()), ty, required)
            } else {
                seen_keyword_args |=
                    x.parameter.name.as_str() != "self" && x.parameter.name.as_str() != "cls";
                Param::Pos(x.parameter.name.id.clone(), ty, required)
            }
        }));
        params.extend(def.parameters.vararg.iter().map(|x| {
            let ty = get_param_ty(&x.name, None);
            if let Type::Args(q) = &ty {
                paramspec_args = Some(q.clone());
            }
            Param::VarArg(Some(x.name.id.clone()), ty)
        }));
        if paramspec_args.is_some()
            && let Some(param) = def.parameters.kwonlyargs.first()
        {
            self.error(
                errors,
                param.range,
                ErrorKind::BadFunctionDefinition,
                None,
                format!(
                    "Keyword-only parameter `{}` may not appear after ParamSpec args parameter",
                    param.parameter.name
                ),
            );
        }
        params.extend(def.parameters.kwonlyargs.iter().map(|x| {
            let ty = get_param_ty(&x.parameter.name, x.default.as_deref());
            let required = if x.default.is_some() {
                Required::Optional
            } else {
                Required::Required
            };
            Param::KwOnly(x.parameter.name.id.clone(), ty, required)
        }));
        params.extend(def.parameters.kwarg.iter().map(|x| {
            let ty = match self.bindings().get_function_param(&x.name) {
                Either::Left(idx) => {
                    let annot = self.get_idx(idx);
                    annot.annotation.get_type().clone()
                }
                Either::Right(var) => self.solver().force_var(var),
            };
            if let Type::Kwargs(q) = &ty {
                paramspec_kwargs = Some(q.clone());
            }
            Param::Kwargs(Some(x.name.id().clone()), ty)
        }));
        let ret = self
            .get(&Key::ReturnType(ShortIdentifier::new(&def.name)))
            .arc_clone_ty();

        if matches!(&ret, Type::TypeGuard(_) | Type::TypeIs(_)) {
            // https://typing.python.org/en/latest/spec/narrowing.html#typeguard
            // https://typing.python.org/en/latest/spec/narrowing.html#typeis
            // TypeGuard and TypeIs must accept at least one positional argument.
            // If a type guard function is implemented as an instance method or
            // class method, the first positional argument maps to the second
            // parameter (after “self” or “cls”).
            let position_args_count = params
                .iter()
                .filter(|p| matches!(p, Param::Pos(..) | Param::PosOnly(..)))
                .count()
                - (if class_key.is_some() && !is_staticmethod {
                    1 // Subtract the "self" or "cls" parameter
                } else {
                    0
                });
            if position_args_count < 1 {
                self.error(
                    errors,
                    // The error should be raised on the line of the function
                    // definition, but using `def.range` would be too broad
                    // since it includes the decorators, which does not match
                    // the conformance testsuite.
                    def.name.range,
                    ErrorKind::BadFunctionDefinition,
                    None,
                    "Type guard functions must accept at least one positional argument".to_owned(),
                );
            }
        };

        if let Type::TypeIs(ty_narrow) = &ret {
            // https://typing.python.org/en/latest/spec/narrowing.html#typeis
            // The return type R must be assignable to I. The type checker
            // should emit an error if this condition is not met.
            let ty_arg = if class_key.is_some() && !is_staticmethod {
                // Skip the first argument (`self` or `cls`) if this is a method or class method.
                params.get(1)
            } else {
                params.first()
            };
            if let Some(ty_arg) = ty_arg {
                if !self.is_subset_eq(ty_narrow, ty_arg.param_to_type()) {
                    // If the narrowed type is not a subtype of the argument type, we report an error.
                    self.error(
                        errors,
                        def.name.range,
                        ErrorKind::BadFunctionDefinition,
                        None,
                        format!(
                            "Return type `{}` must be assignable to the first argument type `{}`",
                            self.for_display(*ty_narrow.clone()),
                            self.for_display(ty_arg.param_to_type().clone())
                        ),
                    );
                }
            }
        }

        let mut tparams = self.scoped_type_params(def.type_params.as_deref(), errors);
        let legacy_tparams = legacy_tparams
            .iter()
            .filter_map(|key| self.get_idx(*key).deref().parameter().cloned());
        tparams.extend(legacy_tparams);
        if paramspec_args != paramspec_kwargs {
            if paramspec_args.is_some() != paramspec_kwargs.is_some() {
                self.error(
                    errors,
                    def.range,
                    ErrorKind::InvalidParamSpec,
                    None,
                    "`ParamSpec` *args and **kwargs must be used together".to_owned(),
                );
            } else {
                self.error(
                    errors,
                    def.range,
                    ErrorKind::InvalidParamSpec,
                    None,
                    "*args and **kwargs must come from the same `ParamSpec`".to_owned(),
                );
            }
            // If ParamSpec args and kwargs are invalid, fall back to Any
            params = params
                .into_iter()
                .map(|p| match p {
                    Param::Kwargs(name, Type::Kwargs(_)) => Param::Kwargs(name, Type::any_error()),
                    Param::VarArg(name, Type::Args(_)) => Param::VarArg(name, Type::any_error()),
                    _ => p,
                })
                .collect();
        } else {
            params = params
                .into_iter()
                .filter_map(|p| match p {
                    Param::Kwargs(_, Type::Kwargs(_)) | Param::VarArg(_, Type::Args(_)) => None,
                    _ => Some(p),
                })
                .collect();
        }
        let callable = if let Some(q) = &paramspec_args
            && paramspec_args == paramspec_kwargs
        {
            Callable::concatenate(
                params
                    .into_iter()
                    .filter_map(|p| match p {
                        Param::PosOnly(_, ty, _) => Some(ty),
                        Param::Pos(_, ty, _) => Some(ty),
                        _ => None,
                    })
                    .collect(),
                Type::Quantified(q.clone()),
                ret,
            )
        } else {
            Callable::list(ParamList::new(params), ret)
        };
        let kind = FunctionKind::from_name(
            self.module_info().name(),
            defining_cls.as_ref().map(|cls| cls.name()),
            &def.name.id,
        );
        let metadata = FuncMetadata {
            kind,
            flags: FuncFlags {
                is_overload,
                is_staticmethod,
                is_classmethod,
                is_deprecated,
                is_property_getter,
                is_property_setter_decorator: false,
                is_property_setter_with_getter,
                has_enum_member_decoration,
                is_override,
                has_final_decoration,
                dataclass_transform_metadata,
            },
        };
        let mut ty = Forallable::Function(Function {
            signature: callable,
            metadata: metadata.clone(),
        })
        .forall(self.validated_tparams(def.range, tparams, errors));
        for x in decorators.into_iter().rev() {
            ty = match self.apply_decorator(*x, ty, errors) {
                // Preserve function metadata, so things like method binding still work.
                Type::Callable(c) => Type::Function(Box::new(Function {
                    signature: *c,
                    metadata: metadata.clone(),
                })),
                // Callback protocol. We convert it to a function so we can add function metadata.
                Type::ClassType(cls)
                    if self
                        .get_metadata_for_class(cls.class_object())
                        .is_protocol() =>
                {
                    let call_attr = self.instance_to_method(&cls).and_then(|call_attr| {
                        if let Type::BoundMethod(m) = call_attr {
                            let func = m.as_bound_function();
                            Some(func.to_unbound_callable().unwrap_or(func))
                        } else {
                            None
                        }
                    });
                    if let Some(mut call_attr) = call_attr {
                        call_attr.transform_toplevel_func_metadata(|m| {
                            *m = FuncMetadata {
                                kind: FunctionKind::CallbackProtocol(Box::new(cls.clone())),
                                flags: metadata.flags.clone(),
                            };
                        });
                        call_attr
                    } else {
                        cls.to_type()
                    }
                }
                t => t,
            }
        }
        Arc::new(DecoratedFunction {
            id_range: def.name.range,
            ty,
            metadata,
            stub_or_impl,
        })
    }

    /// If instances of this class are callable - that is, have a `__call__` method - return the method.
    pub fn instance_to_method(&self, cls: &ClassType) -> Option<Type> {
        self.get_instance_attribute(cls, &dunder::CALL)
            .and_then(|attr| self.resolve_as_instance_method(attr))
    }

    // Given the index to a function binding, return the previous function binding, if any.
    fn step_pred(&self, pred: &mut Option<Idx<Key>>) -> Option<Arc<DecoratedFunction>> {
        let pred_idx = (*pred)?;
        let mut b = self.bindings().get(pred_idx);
        while let Binding::Forward(k) = b {
            b = self.bindings().get(*k);
        }
        if let Binding::Function(idx, pred_, _) = b {
            let def = self.get_idx(*idx);
            *pred = *pred_;
            Some(def)
        } else {
            None
        }
    }

    fn extract_signatures(
        &self,
        func: Name,
        ts: Vec1<(TextRange, Type)>,
        errors: &ErrorCollector,
    ) -> Vec1<OverloadType> {
        ts.mapped(|(range, t)| match t {
            Type::Callable(callable) => OverloadType::Callable(*callable),
            Type::Function(function) => OverloadType::Callable(function.signature),
            Type::Forall(box Forall {
                tparams,
                body: Forallable::Function(func),
            }) => OverloadType::Forall(Forall {
                tparams,
                body: func,
            }),
            Type::Any(any_style) => {
                OverloadType::Callable(Callable::ellipsis(any_style.propagate()))
            }
            _ => {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidOverload,
                    None,
                    format!(
                        "`{}` has type `{}` after decorator application, which is not callable",
                        func,
                        self.for_display(t)
                    ),
                );
                OverloadType::Callable(Callable::ellipsis(Type::any_error()))
            }
        })
    }
}
