/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::types::BoundMethod;
use pyrefly_types::types::TParam;
use pyrefly_types::types::TParams;
use pyrefly_types::types::TParamsSource;
use pyrefly_types::types::Union;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallStyle;
use crate::alt::callable::CallArg;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::alt::types::decorated_function::Decorator;
use crate::alt::types::decorated_function::SpecialDecorator;
use crate::alt::types::decorated_function::UndecoratedFunction;
use crate::binding::binding::Binding;
use crate::binding::binding::FunctionParameter;
use crate::binding::binding::FunctionStubOrImpl;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyDecorator;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::callable::Callable;
use crate::types::callable::FuncFlags;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::PropertyMetadata;
use crate::types::callable::PropertyRole;
use crate::types::callable::Required;
use crate::types::class::ClassKind;
use crate::types::keywords::DataclassTransformMetadata;
use crate::types::types::CalleeKind;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Overload;
use crate::types::types::OverloadType;
use crate::types::types::Type;

fn is_class_property_decorator_class_object(cls: &Class) -> bool {
    let cls_name = cls.name();
    // Obviously this is just a very naive heuristic. But it's not a crazy convention
    // either and we have important customers that badly wants support of class properties.
    cls_name == "classproperty" || cls_name == "lazy_classproperty"
}

fn is_class_property_decorator_type(ty: &Type) -> bool {
    match ty {
        Type::ClassDef(cls) => is_class_property_decorator_class_object(cls),
        Type::ClassType(cls) => is_class_property_decorator_class_object(cls.class_object()),
        _ => false,
    }
}

struct ParentParamHints {
    posonly: VecDeque<Type>,
    positional: VecDeque<Type>,
    kwonly: SmallMap<Name, Type>,
    vararg: Option<Type>,
    kwargs: Option<Type>,
}

impl ParentParamHints {
    fn new(params: ParamList, drop_first: bool) -> Self {
        let mut items = params.into_items().into_iter();
        if drop_first {
            items.next();
        }
        let mut hints = Self {
            posonly: VecDeque::new(),
            positional: VecDeque::new(),
            kwonly: SmallMap::new(),
            vararg: None,
            kwargs: None,
        };
        for param in items {
            match param {
                Param::PosOnly(_, ty, _) => hints.posonly.push_back(ty),
                Param::Pos(_, ty, _) => hints.positional.push_back(ty),
                Param::VarArg(_, ty) => hints.vararg = Some(ty),
                Param::KwOnly(name, ty, _) => {
                    hints.kwonly.insert(name, ty);
                }
                Param::Kwargs(_, ty) => hints.kwargs = Some(ty),
            }
        }
        hints
    }

    fn take_posonly(&mut self) -> Option<Type> {
        self.posonly.pop_front()
    }

    fn take_positional(&mut self) -> Option<Type> {
        self.positional.pop_front()
    }

    fn take_vararg(&mut self) -> Option<Type> {
        self.vararg.take()
    }

    fn take_kwonly(&mut self, name: &Identifier) -> Option<Type> {
        self.kwonly.shift_remove(&name.id)
    }

    fn take_kwargs(&mut self) -> Option<Type> {
        self.kwargs.take()
    }
}

#[derive(Clone, Debug)]
struct DecoratorParamHints {
    positional: Vec<Type>,
    next_positional: usize,
}

impl DecoratorParamHints {
    fn from_callable(callable: Callable) -> Option<Self> {
        match callable.params {
            Params::List(params) => {
                let positional = params
                    .items()
                    .iter()
                    .filter_map(|param| match param {
                        Param::PosOnly(_, ty, _) | Param::Pos(_, ty, _) => Some(ty.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if positional.is_empty() {
                    None
                } else {
                    Some(Self {
                        positional,
                        next_positional: 0,
                    })
                }
            }
            _ => None,
        }
    }

    fn next_positional(&mut self) -> Option<Type> {
        if self.next_positional >= self.positional.len() {
            None
        } else {
            let ty = self.positional[self.next_positional].clone();
            self.next_positional += 1;
            Some(ty)
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn decorator_param_hints(
        &self,
        decorators: &[(Type, TextRange)],
    ) -> Option<DecoratorParamHints> {
        decorators.iter().rev().find_map(|(decorator_ty, _)| {
            decorator_ty
                .callable_first_param()
                .and_then(|param_ty| param_ty.callable_signatures().into_iter().next().cloned())
                .and_then(DecoratorParamHints::from_callable)
        })
    }

    pub fn solve_function_binding(
        &self,
        def: DecoratedFunction,
        predecessor: &mut Option<Idx<Key>>,
        class_metadata: Option<&Idx<KeyClassMetadata>>,
        errors: &ErrorCollector,
    ) -> Type {
        // Overloads in .pyi should not have an implementation.
        let skip_implementation = self.module().path().style() == ModuleStyle::Interface
            || class_metadata.is_some_and(|idx| self.get_idx(*idx).is_protocol())
            || def.metadata().flags.is_abstract_method;
        let mut ty = if def.metadata().flags.is_overload {
            // This function is decorated with @overload. We should warn if this function is actually called anywhere.
            let successor = self.get_function_successor(&def);
            if successor.is_none() {
                // This is the last definition in the chain. We should produce an overload type.
                let is_impl = def.is_impl();
                let mut acc =
                    Vec1::new((def.id_range(), (*def.ty).clone(), def.metadata().clone()));
                let mut impl_before_overload_range = None;
                while let Some(def) = self.step_pred(predecessor) {
                    if def.is_overload() {
                        acc.push((def.id_range(), (*def.ty).clone(), def.metadata().clone()));
                    } else {
                        impl_before_overload_range = Some(def.id_range());
                        break;
                    }
                }
                let first_range = acc.last().0;
                let last_range = acc.first().0;
                if !skip_implementation {
                    if let Some(range) = impl_before_overload_range {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidOverload),
                            "@overload declarations must come before function implementation"
                                .to_owned(),
                        );
                    } else if is_impl {
                        self.error(
                            errors,
                            last_range,
                            ErrorInfo::Kind(ErrorKind::InvalidOverload),
                            "@overload decorator should not be used on function implementation"
                                .to_owned(),
                        );
                    } else {
                        self.error(
                            errors,
                            first_range,
                            ErrorInfo::Kind(ErrorKind::InvalidOverload),
                            "Overloaded function must have an implementation".to_owned(),
                        );
                    }
                }
                if acc.len() == 1 {
                    self.error(
                        errors,
                        first_range,
                        ErrorInfo::Kind(ErrorKind::InvalidOverload),
                        "Overloaded function needs at least two @overload declarations".to_owned(),
                    );
                    acc.split_off_first().0.1
                } else {
                    acc.reverse();
                    self.check_decorator_consistency_no_implementation(&acc, errors);
                    let metadata = self.merge_overload_metadata_no_implementation(&acc);
                    let func_name = acc.first().2.kind.function_name().into_owned();
                    Type::Overload(Overload {
                        signatures: self
                            .extract_signatures(&func_name, acc, errors)
                            .mapped(|(_, sig)| sig),
                        metadata: Box::new(metadata.clone()),
                    })
                }
            } else {
                (*def.ty).clone()
            }
        } else {
            let mut acc = Vec::new();
            while let Some(def) = self.step_pred(predecessor)
                && def.is_overload()
            {
                acc.push((def.id_range(), (*def.ty).clone(), def.metadata().clone()));
            }
            acc.reverse();
            self.check_decorator_consistency_with_implementation(&acc, &def, errors);
            if let Ok(defs) = Vec1::try_from_vec(acc) {
                if defs.len() == 1 {
                    self.error(
                        errors,
                        defs.first().0,
                        ErrorInfo::Kind(ErrorKind::InvalidOverload),
                        "Overloaded function needs at least two @overload declarations".to_owned(),
                    );
                    defs.split_off_first().0.1
                } else {
                    let metadata = self
                        .merge_overload_metadata_with_implementation(&defs, def.metadata().clone());
                    let sigs = self.extract_signatures(
                        metadata.kind.function_name().as_ref(),
                        defs,
                        errors,
                    );
                    self.check_signature_consistency(&sigs, &def, errors);
                    Type::Overload(Overload {
                        signatures: sigs.mapped(|(_, sig)| sig),
                        metadata: Box::new(metadata),
                    })
                }
            } else {
                (*def.ty).clone()
            }
        };

        if let Some(metadata) = def
            .metadata()
            .flags
            .property_metadata
            .as_ref()
            .filter(|meta| matches!(meta.role, PropertyRole::DeleterDecorator))
        {
            ty = metadata
                .setter
                .clone()
                .unwrap_or_else(|| metadata.getter.clone());
            ty.transform_toplevel_func_metadata(|meta| {
                if let Some(property) = &mut meta.flags.property_metadata {
                    property.has_deleter = true;
                }
            });
        }

        if def.is_stub()
            && self.module().path().style() != ModuleStyle::Interface
            && let Some(cls) = def.defining_cls()
            && self.get_metadata_for_class(cls).is_protocol()
        {
            ty.transform_toplevel_func_metadata(|meta| {
                meta.flags.is_abstract_method = true;
            });
        }

        let sanitized = ty.without_property_metadata();
        ty.transform_toplevel_func_metadata(|meta| {
            if let Some(property) = &mut meta.flags.property_metadata {
                match property.role {
                    PropertyRole::Getter => {
                        property.getter = sanitized.clone();
                    }
                    PropertyRole::Setter => {
                        property.setter = Some(sanitized.clone());
                    }
                    _ => {}
                }
            }
        });

        ty
    }

    pub fn undecorated_function(
        &self,
        def: &StmtFunctionDef,
        stub_or_impl: FunctionStubOrImpl,
        class_key: Option<&Idx<KeyClass>>,
        decorators: &[Idx<KeyDecorator>],
        legacy_tparams: &[Idx<KeyLegacyTypeParam>],
        module_style: ModuleStyle,
        errors: &ErrorCollector,
    ) -> Arc<UndecoratedFunction> {
        let defining_cls = class_key.and_then(|k| self.get_idx(*k).0.dupe());
        let is_top_level_function = defining_cls.is_none();
        let mut self_type = defining_cls
            .as_ref()
            .map(|cls| Type::SelfType(self.as_class_type_unchecked(cls)));

        // __new__ is an implicit staticmethod, __init_subclass__ is an implicit classmethod
        // __new__, unlike decorated staticmethods, uses Self
        let is_dunder_new = defining_cls.is_some() && def.name.as_str() == dunder::NEW;
        let is_dunder_init_subclass =
            defining_cls.is_some() && def.name.as_str() == dunder::INIT_SUBCLASS;

        let mut flags = FuncFlags {
            is_staticmethod: is_dunder_new,
            is_classmethod: is_dunder_init_subclass,
            ..Default::default()
        };
        let mut found_class_property = false;
        let decorators = Box::from_iter(decorators.iter().filter_map(|k| {
            let decorator = self.get_idx(*k);
            let range = self.bindings().idx_to_key(*k).range();
            let keep = if let Some(special_decorator) = self.get_special_decorator(&decorator) {
                if is_top_level_function {
                    self.check_top_level_function_decorator(&special_decorator, range, errors);
                }
                !self.set_flag_from_special_decorator(&mut flags, &special_decorator)
            } else {
                if is_class_property_decorator_type(&decorator.ty) {
                    found_class_property = true;
                }
                true
            };
            if keep {
                Some((decorator.ty.clone(), range))
            } else {
                None
            }
        }));

        let mut decorator_param_hints = self.decorator_param_hints(&decorators);
        let mut parent_param_hints = if flags.is_override {
            defining_cls.as_ref().and_then(|cls| {
                self.inherited_method_signature(cls, &def.name.id).and_then(
                    |(params, inherited_flags)| {
                        if inherited_flags.is_staticmethod == flags.is_staticmethod
                            && inherited_flags.is_classmethod == flags.is_classmethod
                        {
                            Some(ParentParamHints::new(
                                params,
                                !inherited_flags.is_staticmethod,
                            ))
                        } else {
                            None
                        }
                    },
                )
            })
        } else {
            None
        };

        if stub_or_impl == FunctionStubOrImpl::Stub {
            flags.lacks_implementation = true;
        }
        if module_style == ModuleStyle::Interface {
            flags.defined_in_stub_file = true;
        }

        // Look for a @classmethod or @staticmethod decorator and change the "self" type
        // accordingly. This is not totally correct, since it doesn't account for chaining
        // decorators, or weird cases like both decorators existing at the same time.
        if flags.is_classmethod || found_class_property || is_dunder_new {
            self_type = self_type.map(Type::type_form);
        } else if flags.is_staticmethod {
            self_type = None;
        }
        let (params, paramspec) = self.get_params_and_paramspec(
            def,
            stub_or_impl,
            &mut self_type,
            &mut decorator_param_hints,
            &mut parent_param_hints,
            errors,
        );
        let mut tparams = self.scoped_type_params(def.type_params.as_deref(), errors);
        let legacy_tparams = legacy_tparams
            .iter()
            .filter_map(|key| self.get_idx(*key).deref().parameter().cloned());
        tparams.extend(legacy_tparams);
        let tparams = self.validated_tparams(def.range, tparams, TParamsSource::Function, errors);

        let kind =
            FunctionKind::from_name(self.module().dupe(), defining_cls.clone(), &def.name.id);
        let metadata = FuncMetadata { kind, flags };

        Arc::new(UndecoratedFunction {
            identifier: ShortIdentifier::new(&def.name),
            metadata,
            decorators,
            tparams,
            params,
            paramspec,
            stub_or_impl,
            defining_cls,
        })
    }

    pub fn decorated_function_type(
        &self,
        def: &UndecoratedFunction,
        stmt: &StmtFunctionDef,
        errors: &ErrorCollector,
    ) -> Arc<Type> {
        let mut ret = self
            .get(&Key::ReturnType(ShortIdentifier::new(&stmt.name)))
            .arc_clone_ty();
        // `stmt.returns` is always set to None because the binding step calls `mem::take` on it
        let has_return_annotation = self.bindings().function_has_return_annotation(&stmt.name);
        if stmt.is_async
            && def.metadata.flags.is_abstract_method
            && !self.behaves_like_any(&ret)
            && let Some((_, _, coroutine_ret)) = self.unwrap_coroutine(&ret)
            && !self.behaves_like_any(&coroutine_ret)
            && self.unwrap_async_iterator(&coroutine_ret).is_some()
        {
            self.error(
                errors,
                stmt.name.range(),
                ErrorInfo::Kind(ErrorKind::BadFunctionDefinition),
                "Abstract methods for async generators should use `def`, not `async def`"
                    .to_owned(),
            );
            ret = coroutine_ret;
        }
        if !has_return_annotation {
            self.error(
                errors,
                stmt.name.range(),
                ErrorInfo::Kind(ErrorKind::UnannotatedReturn),
                format!("`{}` is missing a return annotation", stmt.name),
            );
        }
        for p in stmt.parameters.iter() {
            let name = p.name().as_str();
            if p.annotation().is_none() && name != "cls" && name != "self" {
                self.error(
                    errors,
                    p.name().range(),
                    ErrorInfo::Kind(ErrorKind::UnannotatedParameter),
                    format!(
                        "`{}` is missing an annotation for parameter `{name}`",
                        stmt.name
                    ),
                );
            }
        }
        if matches!(&ret, Type::TypeGuard(_) | Type::TypeIs(_)) {
            self.validate_type_guard_positional_argument_count(
                &def.params,
                def.id_range(),
                &def.defining_cls,
                def.metadata.flags.is_staticmethod,
                errors,
            );
        };

        if let Type::TypeIs(ty_narrow) = &ret {
            self.validate_type_is_type_narrowing(
                &def.params,
                stmt,
                &def.defining_cls,
                def.metadata.flags.is_staticmethod,
                ty_narrow,
                errors,
            );
        }

        let callable = if let Some(q) = &def.paramspec {
            Callable::concatenate(
                def.params
                    .iter()
                    .filter_map(|p| match p {
                        Param::PosOnly(_, ty, req) => Some((ty.clone(), req.clone())),
                        Param::Pos(_, ty, req) => Some((ty.clone(), req.clone())),
                        _ => None,
                    })
                    .collect(),
                Type::Quantified(Box::new(q.clone())),
                ret,
            )
        } else {
            Callable::list(ParamList::new(def.params.clone()), ret)
        };
        let mut ty = Forallable::Function(Function {
            signature: callable,
            metadata: def.metadata.clone(),
        })
        .forall(def.tparams.dupe());
        ty = self.move_return_tparams_of_type(ty);
        for (x, range) in def.decorators.iter().rev() {
            ty = self.apply_function_decorator(x.clone(), ty, &def.metadata, *range, errors);
        }
        Arc::new(ty)
    }

    pub fn get_special_decorator(
        &'a self,
        decorator: &'a Decorator,
    ) -> Option<SpecialDecorator<'a>> {
        if decorator
            .ty
            .property_metadata()
            .is_some_and(|meta| matches!(meta.role, PropertyRole::DeleterDecorator))
        {
            return Some(SpecialDecorator::PropertyDeleter(&decorator.ty));
        }
        match decorator.ty.callee_kind() {
            Some(CalleeKind::Function(FunctionKind::Overload)) => Some(SpecialDecorator::Overload),
            Some(CalleeKind::Class(ClassKind::StaticMethod(name))) => {
                Some(SpecialDecorator::StaticMethod(name))
            }
            Some(CalleeKind::Class(ClassKind::ClassMethod(name))) => {
                Some(SpecialDecorator::ClassMethod(name))
            }
            Some(CalleeKind::Class(ClassKind::Property(name))) => {
                Some(SpecialDecorator::Property(name))
            }
            Some(CalleeKind::Class(ClassKind::CachedProperty(name))) => {
                Some(SpecialDecorator::CachedProperty(name))
            }
            Some(CalleeKind::Class(ClassKind::EnumMember)) => Some(SpecialDecorator::EnumMember),
            Some(CalleeKind::Function(FunctionKind::Override)) => Some(SpecialDecorator::Override),
            Some(CalleeKind::Function(FunctionKind::Final)) => Some(SpecialDecorator::Final),
            _ if let Some(deprecation) = &decorator.deprecation => {
                Some(SpecialDecorator::Deprecated(deprecation))
            }
            _ if decorator
                .ty
                .property_metadata()
                .is_some_and(|meta| matches!(meta.role, PropertyRole::SetterDecorator)) =>
            {
                Some(SpecialDecorator::PropertySetter(&decorator.ty))
            }
            _ if let Type::KwCall(call) = &decorator.ty
                && call.has_function_kind(FunctionKind::DataclassTransform) =>
            {
                Some(SpecialDecorator::DataclassTransformCall(&call.keywords))
            }
            Some(CalleeKind::Class(ClassKind::EnumNonmember)) => {
                Some(SpecialDecorator::EnumNonmember)
            }
            Some(CalleeKind::Function(FunctionKind::AbstractMethod)) => {
                Some(SpecialDecorator::AbstractMethod)
            }
            _ => None,
        }
    }

    /// If the decorator corresponds to a function flag, set the flag appropriately. Returns whether a flag was set.
    pub fn set_flag_from_special_decorator(
        &self,
        flags: &mut FuncFlags,
        decorator: &SpecialDecorator,
    ) -> bool {
        match decorator {
            SpecialDecorator::Overload => {
                flags.is_overload = true;
                true
            }
            SpecialDecorator::StaticMethod(_) => {
                flags.is_staticmethod = true;
                true
            }
            SpecialDecorator::ClassMethod(_) => {
                flags.is_classmethod = true;
                true
            }
            SpecialDecorator::Property(_) => {
                flags.property_metadata = Some(PropertyMetadata {
                    role: PropertyRole::Getter,
                    getter: Type::any_error(),
                    setter: None,
                    has_deleter: false,
                });
                true
            }
            SpecialDecorator::CachedProperty(_) => {
                flags.property_metadata = Some(PropertyMetadata {
                    role: PropertyRole::Getter,
                    getter: Type::any_error(),
                    setter: None,
                    has_deleter: false,
                });
                flags.is_cached_property = true;
                true
            }
            SpecialDecorator::EnumMember => {
                flags.has_enum_member_decoration = true;
                true
            }
            SpecialDecorator::Override => {
                flags.is_override = true;
                true
            }
            SpecialDecorator::Final => {
                flags.has_final_decoration = true;
                true
            }
            SpecialDecorator::Deprecated(deprecation) => {
                flags.deprecation = Some((**deprecation).clone());
                true
            }
            SpecialDecorator::PropertySetter(decorator) => {
                if let Some(metadata) = decorator.property_metadata() {
                    flags.property_metadata = Some(PropertyMetadata {
                        role: PropertyRole::Setter,
                        getter: metadata.getter.clone(),
                        setter: metadata.setter.clone(),
                        has_deleter: metadata.has_deleter,
                    });
                }
                true
            }
            SpecialDecorator::PropertyDeleter(decorator) => {
                flags.property_metadata = decorator.property_metadata();
                true
            }
            SpecialDecorator::DataclassTransformCall(kws) => {
                flags.dataclass_transform_metadata =
                    Some(DataclassTransformMetadata::from_type_map(kws));
                true
            }
            SpecialDecorator::AbstractMethod => {
                flags.is_abstract_method = true;
                true
            }
            _ => false,
        }
    }

    fn get_requiredness(
        &self,
        default: Option<&Expr>,
        check: Option<(&Type, &dyn Fn() -> TypeCheckContext)>,
        stub_or_impl: FunctionStubOrImpl,
        errors: &ErrorCollector,
    ) -> Required {
        match default {
            Some(default)
                if (stub_or_impl == FunctionStubOrImpl::Stub
                    || self.module().path().style() == ModuleStyle::Interface)
                    && matches!(default, Expr::EllipsisLiteral(_)) =>
            {
                Required::Optional(None)
            }
            Some(default) => Required::Optional(Some(self.expr(default, check, errors))),
            None => Required::Required,
        }
    }

    /// Determine the type and required-ness of a parameter.
    fn get_param_type_and_requiredness(
        &self,
        name: &Identifier,
        default: Option<&Expr>,
        stub_or_impl: FunctionStubOrImpl,
        self_type: &mut Option<Type>,
        hint: Option<Type>,
        errors: &ErrorCollector,
    ) -> (Type, Required) {
        // We only want to use self for the first param, so take & replace with None
        let self_type = std::mem::take(self_type);
        let (ty, required) = match self.bindings().get_function_param(name) {
            FunctionParameter::Annotated(idx) => {
                // If the parameter is annotated, we check the default value against the annotation
                let param_ty = self.get_idx(*idx).annotation.get_type().clone();
                let required = self.get_requiredness(
                    default,
                    Some((&param_ty, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::FunctionParameterDefault(
                            name.id.clone(),
                        ))
                    })),
                    stub_or_impl,
                    errors,
                );
                (param_ty, required)
            }
            FunctionParameter::Unannotated(var, _, _) => {
                let required = self.get_requiredness(default, None, stub_or_impl, errors);
                // If this is the first parameter and there is a self type, solve to `Self`.
                // We only try to solve the first param for now. Other unannotated params
                // are also Var. If a default value of type T is provided, it will resolve to Any | T.
                // Otherwise, it will be forced to Any
                if let Some(ty) = self_type {
                    self.solver().solve_parameter(*var, ty);
                } else if let Some(hint) = hint {
                    self.solver().solve_parameter(*var, hint);
                } else if let Required::Optional(Some(default_ty)) = &required {
                    self.solver().solve_parameter(
                        *var,
                        self.union(
                            Type::any_implicit(),
                            default_ty.clone().promote_literals(self.stdlib),
                        ),
                    );
                }
                (self.solver().force_var(*var), required)
            }
        };
        (ty, required)
    }

    fn get_params_and_paramspec(
        &self,
        def: &StmtFunctionDef,
        stub_or_impl: FunctionStubOrImpl,
        self_type: &mut Option<Type>,
        decorator_param_hints: &mut Option<DecoratorParamHints>,
        parent_param_hints: &mut Option<ParentParamHints>,
        errors: &ErrorCollector,
    ) -> (Vec<Param>, Option<Quantified>) {
        let mut paramspec_args = None;
        let mut paramspec_kwargs = None;
        let mut params = Vec::with_capacity(def.parameters.len());
        params.extend(def.parameters.posonlyargs.iter().map(|x| {
            let decorator_hint = decorator_param_hints
                .as_mut()
                .and_then(|hint| hint.next_positional());
            let parent_hint = if self_type.is_some() {
                None
            } else {
                parent_param_hints
                    .as_mut()
                    .and_then(|hint| hint.take_posonly())
            };
            let (ty, required) = self.get_param_type_and_requiredness(
                &x.parameter.name,
                x.default.as_deref(),
                stub_or_impl,
                self_type,
                decorator_hint.or(parent_hint),
                errors,
            );
            Param::PosOnly(Some(x.parameter.name.id.clone()), ty, required)
        }));

        // See: https://typing.python.org/en/latest/spec/historical.html#positional-only-parameters
        let is_historical_args_usage =
            def.parameters.posonlyargs.is_empty() && def.parameters.kwonlyargs.is_empty();
        let mut seen_keyword_args = false;

        params.extend(def.parameters.args.iter().map(|x| {
            let decorator_hint = decorator_param_hints
                .as_mut()
                .and_then(|hint| hint.next_positional());
            let parent_hint = if self_type.is_some() {
                None
            } else {
                parent_param_hints
                    .as_mut()
                    .and_then(|hint| hint.take_positional())
            };
            let (ty, required) = self.get_param_type_and_requiredness(
                &x.parameter.name,
                x.default.as_deref(),
                stub_or_impl,
                self_type,
                decorator_hint.or(parent_hint),
                errors,
            );

            // If the parameter begins but does not end with "__", it is a positional-only parameter.
            // See: https://typing.python.org/en/latest/spec/historical.html#positional-only-parameters
            if is_historical_args_usage && Ast::is_mangled_attr(&x.parameter.name.id) {
                if seen_keyword_args {
                    self.error(
                        errors,
                        x.parameter.name.range,
                        ErrorInfo::Kind(ErrorKind::BadFunctionDefinition),
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
            let parent_hint = parent_param_hints
                .as_mut()
                .and_then(|hint| hint.take_vararg());
            let (ty, _) = self.get_param_type_and_requiredness(
                &x.name,
                None,
                stub_or_impl,
                self_type,
                parent_hint,
                errors,
            );
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
                ErrorInfo::Kind(ErrorKind::BadFunctionDefinition),
                format!(
                    "Keyword-only parameter `{}` may not appear after ParamSpec args parameter",
                    param.parameter.name
                ),
            );
        }
        params.extend(def.parameters.kwonlyargs.iter().map(|x| {
            let parent_hint = parent_param_hints
                .as_mut()
                .and_then(|hint| hint.take_kwonly(&x.parameter.name));
            let (ty, required) = self.get_param_type_and_requiredness(
                &x.parameter.name,
                x.default.as_deref(),
                stub_or_impl,
                self_type,
                parent_hint,
                errors,
            );
            Param::KwOnly(x.parameter.name.id.clone(), ty, required)
        }));
        if let Some(x) = &def.parameters.kwarg {
            let parent_hint = parent_param_hints
                .as_mut()
                .and_then(|hint| hint.take_kwargs());
            let (ty, _) = self.get_param_type_and_requiredness(
                &x.name,
                None,
                stub_or_impl,
                self_type,
                parent_hint,
                errors,
            );
            if let Type::Kwargs(q) = &ty {
                paramspec_kwargs = Some(q.clone());
            }

            if let Type::Unpack(unpack) = &ty
                && let Type::TypedDict(typed_dict) = &**unpack
            {
                for (name, _) in self.typed_dict_fields(typed_dict) {
                    if params.iter().any(|param| {
                        matches!(
                            param,
                            Param::Pos(param_name, ..) | Param::KwOnly(param_name, ..)
                                if param_name == &name
                        )
                    }) {
                        self.error(
                            errors,
                            x.range,
                            ErrorInfo::Kind(ErrorKind::BadFunctionDefinition),
                            format!(
                                "TypedDict key '{name}' in **kwargs overlaps with parameter '{name}'"
                            ),
                        );
                    }
                }
            }

            params.push(Param::Kwargs(Some(x.name.id().clone()), ty));
        }

        let paramspec = if let Some(q) = &paramspec_args
            && paramspec_args == paramspec_kwargs
        {
            Some((**q).clone())
        } else if paramspec_args != paramspec_kwargs {
            if paramspec_args.is_some() != paramspec_kwargs.is_some() {
                self.error(
                    errors,
                    def.range,
                    ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                    "`ParamSpec` *args and **kwargs must be used together".to_owned(),
                );
            } else {
                self.error(
                    errors,
                    def.range,
                    ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
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
            None
        } else {
            params = params
                .into_iter()
                .filter_map(|p| match p {
                    Param::Kwargs(_, Type::Kwargs(_)) | Param::VarArg(_, Type::Args(_)) => None,
                    _ => Some(p),
                })
                .collect();
            None
        };
        (params, paramspec)
    }

    fn check_top_level_function_decorator(
        &self,
        decorator: &SpecialDecorator,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        let name = match decorator {
            SpecialDecorator::StaticMethod(name) => name.as_str(),
            SpecialDecorator::ClassMethod(name) => name.as_str(),
            SpecialDecorator::Property(name) => name.as_str(),
            SpecialDecorator::CachedProperty(name) => name.as_str(),
            SpecialDecorator::EnumMember => "member",
            SpecialDecorator::Override => "override",
            SpecialDecorator::Final => "final",
            SpecialDecorator::EnumNonmember => "nonmember",
            SpecialDecorator::AbstractMethod => "abstractmethod",
            _ => return,
        };
        self.error(
            errors,
            range,
            ErrorInfo::Kind(ErrorKind::InvalidDecorator),
            format!("Decorator `@{name}` can only be used on methods."),
        );
    }

    /// Check if `ty` is a generic function whose return type is a callable that contains type
    /// parameters that appear nowhere else in `ty`'s signature. If so, we make the return type
    /// generic in those type parameters and remove them from `ty`'s tparams. For example, we turn:
    ///   [T1, T2](x: T1, y: T1) -> ((T2) -> T2)
    /// into:
    ///   [T1](x: T1, y: T1) -> ([T2](T2) -> T2)
    fn move_return_tparams_of_type(&self, ty: Type) -> Type {
        match ty {
            Type::Forall(box Forall {
                tparams,
                body: Forallable::Function(func),
            }) => {
                let (tparams, signature) =
                    self.move_return_tparams_of_signature(tparams, func.signature);
                Forallable::Function(Function {
                    signature,
                    metadata: func.metadata,
                })
                .forall(tparams)
            }
            Type::Forall(box Forall {
                tparams,
                body: Forallable::Callable(signature),
            }) => {
                let (tparams, signature) =
                    self.move_return_tparams_of_signature(tparams, signature);
                Forallable::Callable(signature).forall(tparams)
            }
            _ => ty,
        }
    }

    fn move_return_tparams_of_signature(
        &self,
        tparams: Arc<TParams>,
        mut signature: Callable,
    ) -> (Arc<TParams>, Callable) {
        let returns_callable = match &signature.ret {
            Type::Callable(_) => true,
            Type::Union(box Union { members: ts, .. }) => {
                ts.iter().any(|t| matches!(t, Type::Callable(_)))
            }
            _ => false,
        };
        if !returns_callable {
            return (tparams, signature);
        }
        let (param_tparams, ret_tparams) = self.split_tparams(&tparams, &signature.params);
        if ret_tparams.is_empty() {
            (tparams, signature)
        } else {
            let make_tparams = |tparams: Vec<&TParam>| {
                Arc::new(TParams::new(tparams.into_iter().cloned().collect()))
            };
            // Recursively move type parameters in the return type so that
            // things like `[T]() -> (() -> (T) -> T)` get rewritten properly.
            let ret = self.move_return_tparams_of_type(
                self.make_generic_return(signature.ret, make_tparams(ret_tparams)),
            );
            signature.ret = ret;
            (make_tparams(param_tparams), signature)
        }
    }

    /// Split `tparams` by whether they appear in `params`.
    fn split_tparams<'b>(
        &self,
        tparams: &'b TParams,
        params: &Params,
    ) -> (Vec<&'b TParam>, Vec<&'b TParam>) {
        let mut param_qs = SmallSet::new();
        params.visit(&mut |ty| {
            ty.collect_quantifieds(&mut param_qs);
        });
        tparams
            .iter()
            .partition(|tparam| param_qs.contains(&tparam.quantified))
    }

    /// Turn any top-level Type::Callable(callable) in `ret` into Forall[tparams, callable].
    fn make_generic_return(&self, ret: Type, tparams: Arc<TParams>) -> Type {
        self.distribute_over_union(&ret, |ret| match ret {
            Type::Callable(callable) => {
                Forallable::Callable((**callable).clone()).forall(tparams.clone())
            }
            t => t.clone(),
        })
    }

    fn apply_function_decorator(
        &self,
        decorator: Type,
        decoratee: Type,
        metadata: &FuncMetadata,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        // Check if this is a decorator that's special-cased to preserve the decorated function's signature
        if let Type::KwCall(call) = &decorator
            && call.func_metadata.kind.is_signature_preserving_decorator()
        {
            return decoratee;
        }
        // Preserve function metadata, so things like method binding still work.
        let call_target =
            self.as_call_target_or_error(decorator, CallStyle::FreeForm, range, errors, None);
        // If the decoratee is generic, unwrap the `Forall` so that `call_infer` can treat the
        // type parameters as concrete in the raw inferred result; this avoids us replacing the
        // type vars with partial types.
        let (tparams_opt, decoratee_arg) = match &decoratee {
            Type::Forall(forall) => (Some(forall.tparams.clone()), forall.body.clone().as_type()),
            _ => (None, decoratee.clone()),
        };
        let arg = CallArg::ty(&decoratee_arg, range);
        // Compute the raw return type - this may need tweaks to handle Forall well.
        let inferred_ty =
            match self.call_infer(call_target, &[arg], &[], range, errors, None, None, None) {
                Type::Callable(c) => Type::Function(Box::new(Function {
                    signature: *c,
                    metadata: metadata.clone(),
                })),
                Type::Forall(box Forall {
                    tparams,
                    body: Forallable::Callable(c),
                }) => Forallable::Function(Function {
                    signature: c,
                    metadata: metadata.clone(),
                })
                .forall(tparams),
                // Callback protocol. We convert it to a function so we can add function metadata.
                Type::ClassType(cls)
                    if self
                        .get_metadata_for_class(cls.class_object())
                        .is_protocol() =>
                {
                    let call_attr = self.instance_as_dunder_call(&cls).and_then(|call_attr| {
                        if let Type::BoundMethod(m) = call_attr {
                            Some(
                                self.bind_boundmethod(&m, &mut |a, b| self.is_subset_eq(a, b))
                                    .unwrap_or(m.func.as_type()),
                            )
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
                Type::ClassType(cls) if cls.has_qname("functools", "_Wrapped") => decoratee.clone(),
                returned_ty => returned_ty,
            };

        // Given the raw `inferred_ty`, which may include `Type::Quantified` type variables coming from a
        // `Forall` in the original decoratee, we need to create the proper output type:
        // - If the return type is not a callable, we simply replace it with gradual types
        // - If the return type is a callable (possibly wrapped in a Forall with additional
        //   tparams introduced by the decorator), we merge the original tparams with the
        //   new ones (if any) and produce a Forall
        //
        // An invariant is that in no case should a `Type::Quantified` originating in a `Forall`
        // from either the decoratee *or* the raw `inferred_ty` make it into the final result
        // without either being wrapped in a `Forall` or converted to a gradual type.
        if let Some(tparams) = tparams_opt {
            // Identify which original tparams are actually used in the result
            // We scope this in a block to drop the borrow on inferred_ty immediately after scanning
            let relevant_tparams_vec: Vec<TParam> = {
                let mut used_quantifieds = SmallSet::new();
                inferred_ty.collect_quantifieds(&mut used_quantifieds);
                tparams
                    .iter()
                    .filter(|p| used_quantifieds.contains(&p.quantified))
                    .cloned()
                    .collect()
            };
            if !relevant_tparams_vec.is_empty() {
                let new_tparams = Arc::new(TParams::new(relevant_tparams_vec));
                return match inferred_ty {
                    // Merge tparams from decoratee and inferred_ty
                    Type::Forall(box forall) => {
                        let mut merged_tparams = (*new_tparams).clone();
                        merged_tparams.extend(&forall.tparams);
                        forall.body.forall(Arc::new(merged_tparams))
                    }
                    // Wrap callable inferred_ty in a Forall with the decoratee tparams that appear in it
                    Type::Function(f) => Forallable::Function(*f).forall(new_tparams),
                    Type::Callable(c) => Forallable::Callable(*c).forall(new_tparams),
                    // Convert any `Type::Qauntified` from the original Forall to gradual types if the type isn't callable
                    ty => {
                        let substitution_map: SmallMap<_, _> = new_tparams
                            .iter()
                            .map(|p| (&p.quantified, p.quantified.as_gradual_type()))
                            .collect();
                        ty.subst(&substitution_map.iter().map(|(k, v)| (*k, v)).collect())
                    }
                };
            }
        }

        inferred_ty
    }

    /// For a type guard function, validate whether it has at least one
    /// positional argument.
    fn validate_type_guard_positional_argument_count(
        &self,
        params: &[Param],
        id_range: TextRange,
        defining_cls: &Option<Class>,
        is_staticmethod: bool,
        errors: &ErrorCollector,
    ) {
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
            .saturating_sub(if defining_cls.is_some() && !is_staticmethod {
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
                id_range,
                ErrorInfo::Kind(ErrorKind::BadFunctionDefinition),
                "Type guard functions must accept at least one positional argument".to_owned(),
            );
        }
    }

    /// For a "TypeIs" type guard, validate whether the return type is a subtype
    /// of the first argument type.
    fn validate_type_is_type_narrowing(
        &self,
        params: &[Param],
        def: &StmtFunctionDef,
        defining_cls: &Option<Class>,
        is_staticmethod: bool,
        ty_narrow: &Type,
        errors: &ErrorCollector,
    ) {
        // https://typing.python.org/en/latest/spec/narrowing.html#typeis
        // The return type R must be assignable to I. The type checker
        // should emit an error if this condition is not met.
        let ty_arg = if defining_cls.is_some() && !is_staticmethod {
            // Skip the first argument (`self` or `cls`) if this is a method or class method.
            params.get(1)
        } else {
            params.first()
        };
        if let Some(ty_arg) = ty_arg
            && !self.is_subset_eq(ty_narrow, ty_arg.as_type())
        {
            // If the narrowed type is not a subtype of the argument type, we report an error.
            self.error(
                errors,
                def.name.range,
                ErrorInfo::Kind(ErrorKind::BadFunctionDefinition),
                format!(
                    "Return type `{}` must be assignable to the first argument type `{}`",
                    self.for_display(ty_narrow.clone()),
                    self.for_display(ty_arg.as_type().clone())
                ),
            );
        }
    }

    // Given the index to a function binding, return the previous function binding, if any.
    fn step_pred(&self, pred: &mut Option<Idx<Key>>) -> Option<DecoratedFunction> {
        let pred_idx = (*pred)?;
        let mut b = self.bindings().get(pred_idx);
        while let Binding::Forward(k) = b {
            b = self.bindings().get(*k);
        }
        if let Binding::Function(idx, pred_, _) = b {
            *pred = *pred_;
            Some(self.get_decorated_function(*idx))
        } else {
            None
        }
    }

    fn extract_signatures(
        &self,
        func: &Name,
        ts: Vec1<(TextRange, Type, FuncMetadata)>,
        errors: &ErrorCollector,
    ) -> Vec1<(TextRange, OverloadType)> {
        ts.mapped(|(range, t, metadata)| {
            (
                range,
                match t {
                    Type::Callable(callable) => OverloadType::Function(Function {
                        signature: *callable,
                        metadata,
                    }),
                    Type::Function(function) => OverloadType::Function(*function),
                    Type::Forall(box Forall {
                        tparams,
                        body: Forallable::Callable(callable),
                    }) => OverloadType::Forall(Forall {
                        tparams,
                        body: Function {
                            signature: callable,
                            metadata,
                        },
                    }),
                    Type::Forall(box Forall {
                        tparams,
                        body: Forallable::Function(func),
                    }) => OverloadType::Forall(Forall {
                        tparams,
                        body: func,
                    }),
                    Type::Any(any_style) => OverloadType::Function(Function {
                        signature: Callable::ellipsis(any_style.propagate()),
                        metadata,
                    }),
                    _ => {
                        self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidOverload),
                    format!(
                        "`{}` has type `{}` after decorator application, which is not callable",
                        func,
                        self.for_display(t)
                    ),
                );
                        OverloadType::Function(Function {
                            signature: Callable::ellipsis(Type::any_error()),
                            metadata,
                        })
                    }
                },
            )
        })
    }

    fn merge_overload_metadata_with_implementation(
        &self,
        overloads: &Vec1<(TextRange, Type, FuncMetadata)>,
        mut metadata: FuncMetadata,
    ) -> FuncMetadata {
        // `@dataclass_transform()` can be on any of the overloads or the implementation but not
        // more than one: https://typing.python.org/en/latest/spec/dataclasses.html#specification.
        let dataclass_transform_metadata = overloads
            .iter()
            .find_map(|(_, _, metadata)| metadata.flags.dataclass_transform_metadata.as_ref());
        // All other decorators must be present on the implementation:
        // https://typing.python.org/en/latest/spec/overload.html#invalid-overload-definitions.
        if dataclass_transform_metadata.is_some() {
            metadata.flags.dataclass_transform_metadata = dataclass_transform_metadata.cloned();
        }
        metadata
    }

    fn merge_overload_metadata_no_implementation(
        &self,
        overloads: &Vec1<(TextRange, Type, FuncMetadata)>,
    ) -> FuncMetadata {
        let (first, remaining) = overloads.split_first().unwrap();
        // When an overloaded function doesn't have a implementation, decorators like `@override` and `@final` should be applied
        // on the first overload: https://typing.python.org/en/latest/spec/overload.html#invalid-overload-definitions.
        let mut metadata = first.2.clone();
        // This does not apply to `@deprecated` - some overloads can be deprecated while others are fine.
        metadata.flags.deprecation = None;
        // `dataclass_transform()` can be on any of the overloads.
        if metadata.flags.dataclass_transform_metadata.is_none() {
            metadata.flags.dataclass_transform_metadata = remaining
                .iter()
                .find_map(|(_, _, metadata)| metadata.flags.dataclass_transform_metadata.clone());
        }
        metadata
    }

    fn subst_function(&self, tparams: &TParams, func: Function) -> Function {
        let mp = tparams
            .as_vec()
            .map(|p| (&p.quantified, p.restriction().as_type(self.stdlib)));
        match Type::Function(Box::new(func)).subst(&mp.iter().map(|(k, v)| (*k, v)).collect()) {
            Type::Function(func) => *func,
            // We passed a Function in, we must get a Function out
            _ => unreachable!(),
        }
    }

    fn check_signature_consistency(
        &self,
        overloads: &Vec1<(TextRange, OverloadType)>,
        def: &DecoratedFunction,
        errors: &ErrorCollector,
    ) {
        let impl_tparams = match &*def.ty {
            Type::Forall(forall) => Some(&forall.tparams),
            _ => None,
        };
        let impl_sig = {
            let sigs = def.ty.callable_signatures();
            if sigs.len() != 1 {
                // If this is somehow not a callable (len == 0), there's nothing to check.
                // An overload's implementation can't be overloaded (len > 1).
                return;
            }
            sigs[0]
        };
        let all_tparams = |tparams: Option<&Arc<TParams>>| match (tparams, def.defining_cls()) {
            (None, None) => None,
            (Some(_), None) => tparams.cloned(),
            (None, Some(cls)) => Some(self.get_class_tparams(cls)),
            (Some(tparams), Some(cls)) => {
                let mut all_tparams = (**tparams).clone();
                all_tparams.extend(&self.get_class_tparams(cls));
                Some(Arc::new(all_tparams))
            }
        };
        let sig_for_input_check = |sig: &Callable| {
            let mut sig = sig.clone();
            // Set the return type to `Any` so that we check just the input signature.
            sig.ret = Type::any_implicit();
            sig
        };
        for (range, overload) in overloads.iter() {
            let (overload_tparams, original_overload_func) = match overload {
                OverloadType::Function(func) => (None, func),
                OverloadType::Forall(forall) => (Some(&forall.tparams), &forall.body),
            };
            let overload_func = {
                if let Some(tparams) = all_tparams(overload_tparams) {
                    self.subst_function(&tparams, original_overload_func.clone())
                } else {
                    original_overload_func.clone()
                }
            };
            let impl_func = {
                let func = Function {
                    signature: impl_sig.clone(),
                    metadata: def.metadata().clone(),
                };
                if let Some(tparams) = all_tparams(impl_tparams) {
                    self.instantiate_fresh_function(&tparams, func).1
                } else {
                    func
                }
            };
            // See https://typing.python.org/en/latest/spec/overload.html#implementation-consistency.
            // We check that the input signature of the implementation is assignable to the input
            // signature of the overload and that the return type of the overload is assignable
            // to the return type of the implementation. (Note that the two assignability checks
            // are in opposite directions.)
            self.check_type(
                &Type::Callable(Box::new(sig_for_input_check(&impl_func.signature))),
                &Type::Callable(Box::new(sig_for_input_check(&overload_func.signature))),
                *range,
                errors,
                &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::OverloadInput(
                        original_overload_func.signature.clone(),
                        impl_sig.clone(),
                    ))
                },
            );
            self.check_type(
                &overload_func.signature.ret,
                &impl_func.signature.ret,
                *range,
                errors,
                &|| TypeCheckContext::of_kind(TypeCheckKind::OverloadReturn),
            );
        }
    }

    fn check_decorator_consistency_no_implementation(
        &self,
        overloads: &Vec1<(TextRange, Type, FuncMetadata)>,
        errors: &ErrorCollector,
    ) {
        let is_static_method = overloads.iter().any(|x| x.2.flags.is_staticmethod);
        let is_class_method = overloads.iter().any(|x| x.2.flags.is_classmethod);
        for (overload_range, _, overload_metadata) in overloads.iter().skip(1) {
            if overload_metadata.flags.has_final_decoration {
                self.error(
                                errors,
                                *overload_range,
                                ErrorInfo::Kind(ErrorKind::InvalidOverload),
                                "If an overloaded function has no implementation, `@final` should be applied to the first overload only.".to_owned(),
                            );
            }
            if overload_metadata.flags.is_override {
                self.error(
                                errors,
                                *overload_range,
                                ErrorInfo::Kind(ErrorKind::InvalidOverload),
                                "If an overloaded function has no implementation, `@override` should be applied to the first overload only.".to_owned(),
                            );
            }
        }
        for (overload_range, _, overload_metadata) in overloads.iter() {
            if overload_metadata.flags.is_staticmethod != is_static_method {
                self.error(
                                errors,
                                *overload_range,
                                ErrorInfo::Kind(ErrorKind::InvalidOverload),
                                "If `@staticmethod` is present on one overload, all overloads must have that decorator.".to_owned(),
                            );
            }
            if overload_metadata.flags.is_classmethod != is_class_method {
                self.error(
                                errors,
                                *overload_range,
                                ErrorInfo::Kind(ErrorKind::InvalidOverload),
                                "If `@classmethod` is present on one overload, all overloads must have that decorator.".to_owned(),
                            );
            }
        }
    }

    fn check_decorator_consistency_with_implementation(
        &self,
        overloads: &[(TextRange, Type, FuncMetadata)],
        def: &DecoratedFunction,
        errors: &ErrorCollector,
    ) {
        let is_static_method = def.metadata().flags.is_staticmethod
            || overloads.iter().any(|x| x.2.flags.is_staticmethod);
        let is_class_method = def.metadata().flags.is_classmethod
            || overloads.iter().any(|x| x.2.flags.is_classmethod);
        for (overload_range, _, overload_metadata) in overloads.iter() {
            if overload_metadata.flags.has_final_decoration {
                self.error(
                        errors,
                        *overload_range,
                        ErrorInfo::Kind(ErrorKind::InvalidOverload),
                        "`@final` should only be applied to the implementation of an overloaded function.".to_owned(),
                    );
            }
            if overload_metadata.flags.is_override {
                self.error(
                        errors,
                        *overload_range,
                        ErrorInfo::Kind(ErrorKind::InvalidOverload),
                        "`@override` should only be applied to the implementation of an overloaded function.".to_owned(),
                    );
            }
            if overload_metadata.flags.is_staticmethod != is_static_method {
                self.error(
                        errors,
                        *overload_range,
                        ErrorInfo::Kind(ErrorKind::InvalidOverload),
                        "If `@staticmethod` is present on any overload or the implementation, it should be on every overload and the implementation.".to_owned(),
                    );
            }
            if overload_metadata.flags.is_classmethod != is_class_method {
                self.error(
                        errors,
                        *overload_range,
                        ErrorInfo::Kind(ErrorKind::InvalidOverload),
                        "If `@classmethod` is present on any overload or the implementation, it should be on every overload and the implementation.".to_owned(),
                    );
            }
        }
        if def.metadata().flags.is_staticmethod != is_static_method {
            self.error(
                    errors,
                    def.id_range(),
                    ErrorInfo::Kind(ErrorKind::InvalidOverload),
                    "If `@staticmethod` is present on any overload or the implementation, it should be on every overload and the implementation.".to_owned(),
                );
        }
        if def.metadata().flags.is_classmethod != is_class_method {
            self.error(
                    errors,
                    def.id_range(),
                    ErrorInfo::Kind(ErrorKind::InvalidOverload),
                    "If `@classmethod` is present on any overload or the implementation, it should be on every overload and the implementation.".to_owned(),
                );
        }
    }

    pub fn bind_boundmethod(
        &self,
        m: &BoundMethod,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> Option<Type> {
        self.bind_function(&m.func.clone().as_type(), &m.obj, is_subset)
    }

    pub fn bind_dunder_new(&self, t: &Type, cls: ClassType) -> Option<Type> {
        self.bind_function(
            t,
            &Type::Type(Box::new(Type::SelfType(cls))),
            &mut |a, b| self.is_subset_eq(a, b),
        )
    }

    /// If this is an unbound callable (i.e., a callable that is not BoundMethod), strip the first parameter.
    /// If it is generic, we use the bound object to instantiate type variables in the first argument.
    fn bind_function(
        &self,
        t: &Type,
        obj: &Type,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> Option<Type> {
        let mut owner = Owner::new();
        match t {
            Type::Forall(forall) => match &forall.body {
                Forallable::Callable(c) => c.split_first_param(&mut owner).map(|(param, c)| {
                    let c =
                        self.instantiate_callable_self(&forall.tparams, obj, param, c, is_subset);
                    Type::Forall(Box::new(Forall {
                        tparams: forall.tparams.clone(),
                        body: Forallable::Callable(c),
                    }))
                }),
                Forallable::Function(f) => {
                    f.signature.split_first_param(&mut owner).map(|(param, c)| {
                        let c = self.instantiate_callable_self(
                            &forall.tparams,
                            obj,
                            param,
                            c,
                            is_subset,
                        );
                        Type::Forall(Box::new(Forall {
                            tparams: forall.tparams.clone(),
                            body: Forallable::Function(Function {
                                signature: c,
                                metadata: f.metadata.clone(),
                            }),
                        }))
                    })
                }
                Forallable::TypeAlias(_) => None,
            },
            Type::Callable(callable) => callable
                .split_first_param(&mut owner)
                .map(|(_, c)| Type::Callable(Box::new(c))),
            Type::Function(func) => func.signature.split_first_param(&mut owner).map(|(_, c)| {
                Type::Function(Box::new(Function {
                    signature: c,
                    metadata: func.metadata.clone(),
                }))
            }),
            Type::Overload(overload) => overload
                .signatures
                .try_mapped_ref(|x| match x {
                    OverloadType::Function(f) => f
                        .signature
                        .split_first_param(&mut owner)
                        .map(|(_, c)| {
                            OverloadType::Function(Function {
                                signature: c,
                                metadata: f.metadata.clone(),
                            })
                        })
                        .ok_or(()),
                    OverloadType::Forall(forall) => forall
                        .body
                        .signature
                        .split_first_param(&mut owner)
                        .map(|(param, c)| {
                            let c = self.instantiate_callable_self(
                                &forall.tparams,
                                obj,
                                param,
                                c,
                                is_subset,
                            );
                            OverloadType::Forall(Forall {
                                tparams: forall.tparams.clone(),
                                body: Function {
                                    signature: c,
                                    metadata: forall.body.metadata.clone(),
                                },
                            })
                        })
                        .ok_or(()),
                })
                .ok()
                .map(|signatures| {
                    Type::Overload(Overload {
                        signatures,
                        metadata: overload.metadata.clone(),
                    })
                }),
            _ => None,
        }
    }

    fn instantiate_callable_self(
        &self,
        tparams: &TParams,
        self_obj: &Type,
        self_param: &Type,
        callable: Callable,
        is_subset: &mut dyn FnMut(&Type, &Type) -> bool,
    ) -> Callable {
        self.solver().instantiate_callable_self(
            tparams,
            self_obj,
            self_param,
            callable,
            self.uniques,
            is_subset,
        )
    }
}
