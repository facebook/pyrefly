/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use itertools::EitherOrBoth;
use itertools::Itertools;
use ruff_python_ast::name::Name;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtClassDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallmap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::attr::Attribute;
use crate::alt::attr::NoAccessReason;
use crate::ast::Ast;
use crate::binding::binding::ClassFieldInitialization;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::dunder;
use crate::graph::index::Idx;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::annotation::Annotation;
use crate::types::annotation::Qualifier;
use crate::types::callable::Callable;
use crate::types::callable::CallableKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::class::Substitution;
use crate::types::class::TArgs;
use crate::types::class_metadata::ClassMetadata;
use crate::types::class_metadata::DataclassMetadata;
use crate::types::class_metadata::EnumMetadata;
use crate::types::literal::Lit;
use crate::types::special_form::SpecialForm;
use crate::types::type_var::Variance;
use crate::types::types::CalleeKind;
use crate::types::types::Decoration;
use crate::types::types::TParamInfo;
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::types::types::TypedDict;
use crate::types::types::TypedDictField;
use crate::util::display::count;
use crate::util::prelude::SliceExt;

/// Raw information about an attribute declared somewhere in a class. We need to
/// know whether it is initialized in the class body in order to determine
/// both visibility rules and whether method binding should be performed.
#[derive(Debug, Clone)]
pub struct ClassField(ClassFieldInner);

#[derive(Debug, Clone)]
pub enum ClassFieldInner {
    Simple {
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
    },
}

impl ClassField {
    fn new(
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
    ) -> Self {
        Self(ClassFieldInner::Simple {
            ty,
            annotation,
            initialization,
        })
    }

    pub fn recursive() -> Self {
        Self(ClassFieldInner::Simple {
            ty: Type::any_implicit(),
            annotation: None,
            initialization: ClassFieldInitialization::Class,
        })
    }

    pub fn visit_type_mut(&mut self, mut f: &mut dyn FnMut(&mut Type)) {
        match &mut self.0 {
            ClassFieldInner::Simple { ty, annotation, .. } => {
                f(ty);
                for a in annotation.iter_mut() {
                    a.visit_type_mut(&mut f);
                }
            }
        }
    }

    fn initialization(&self) -> ClassFieldInitialization {
        match &self.0 {
            ClassFieldInner::Simple { initialization, .. } => *initialization,
        }
    }

    fn instantiate_for(&self, cls: &ClassType) -> Self {
        match &self.0 {
            ClassFieldInner::Simple {
                ty,
                annotation,
                initialization,
            } => Self(ClassFieldInner::Simple {
                ty: cls.instantiate_member(ty.clone()),
                annotation: annotation.clone(),
                initialization: *initialization,
            }),
        }
    }

    fn depends_on_class_type_parameter(&self, cls: &Class) -> bool {
        let tparams = cls.tparams();
        let mut qs = SmallSet::new();
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => ty.collect_quantifieds(&mut qs),
        };
        tparams.quantified().any(|q| qs.contains(&q))
    }

    fn as_raw_special_method_type(self, cls: &ClassType) -> Option<Type> {
        match self.instantiate_for(cls).0 {
            ClassFieldInner::Simple { ty, .. } => match self.initialization() {
                ClassFieldInitialization::Class => Some(ty),
                ClassFieldInitialization::Instance => None,
            },
        }
    }

    fn as_special_method_type(self, cls: &ClassType) -> Option<Type> {
        self.as_raw_special_method_type(cls).and_then(|ty| {
            if is_unbound_function(&ty) {
                Some(make_bound_method(cls.self_type(), ty))
            } else {
                None
            }
        })
    }

    fn as_instance_attribute(self, cls: &ClassType) -> Attribute {
        match self.instantiate_for(cls).0 {
            ClassFieldInner::Simple { ty, .. } => match self.initialization() {
                ClassFieldInitialization::Class => bind_instance_attribute(cls, ty),
                ClassFieldInitialization::Instance => Attribute::read_write(ty),
            },
        }
    }

    fn as_class_attribute(self, cls: &Class) -> Attribute {
        match &self.0 {
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::Instance,
                ..
            } => Attribute::no_access(NoAccessReason::ClassUseOfInstanceAttribute(cls.clone())),
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::Class,
                ty,
                ..
            } => {
                if self.depends_on_class_type_parameter(cls) {
                    Attribute::no_access(NoAccessReason::ClassAttributeIsGeneric(cls.clone()))
                } else {
                    bind_class_attribute(cls, ty.clone())
                }
            }
        }
    }
}

fn is_unbound_function(ty: &Type) -> bool {
    match ty {
        Type::Forall(_, t) => is_unbound_function(t),
        Type::Callable(_, _) => true,
        _ => false,
    }
}

fn bind_class_attribute(cls: &Class, attr: Type) -> Attribute {
    match attr {
        Type::Decoration(Decoration::StaticMethod(box attr)) => Attribute::read_write(attr),
        Type::Decoration(Decoration::ClassMethod(box attr)) => {
            Attribute::read_write(make_bound_method(Type::ClassDef(cls.dupe()), attr))
        }
        attr => Attribute::read_write(attr),
    }
}

fn bind_instance_attribute(cls: &ClassType, attr: Type) -> Attribute {
    match attr {
        Type::Decoration(Decoration::StaticMethod(box attr)) => Attribute::read_write(attr),
        Type::Decoration(Decoration::ClassMethod(box attr)) => Attribute::read_write(
            make_bound_method(Type::ClassDef(cls.class_object().dupe()), attr),
        ),
        attr => Attribute::read_write(if is_unbound_function(&attr) {
            make_bound_method(cls.self_type(), attr)
        } else {
            attr
        }),
    }
}

fn make_bound_method(obj: Type, attr: Type) -> Type {
    // TODO(stroxler): Think about what happens if `attr` is not callable. This
    // can happen with the current logic if a decorator spits out a non-callable
    // type that gets wrapped in `@classmethod`.
    Type::BoundMethod(Box::new(obj), Box::new(attr))
}

impl Display for ClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ClassFieldInner::Simple {
                ty, initialization, ..
            } => {
                let initialized = match initialization {
                    ClassFieldInitialization::Class => "initialized in body",
                    ClassFieldInitialization::Instance => "not initialized in body",
                };
                write!(f, "@{} ({})", ty, initialized)
            }
        }
    }
}

/// Result of looking up a member of a class in the MRO, including a handle to the defining
/// class which may be some ancestor.
///
/// For example, given `class A: x: int; class B(A): pass`, the defining class
/// for attribute `x` is `A` even when `x` is looked up on `B`.
struct WithDefiningClass<T> {
    value: T,
    defining_class: Class,
}

impl<T> WithDefiningClass<T> {
    fn defined_on(&self, cls: &Class) -> bool {
        self.defining_class == *cls
    }
}

/// Private helper type used to share part of the logic needed for the
/// binding-level work of finding legacy type parameters versus the type-level
/// work of computing inherticance information and the MRO.
#[derive(Debug, Clone)]
enum BaseClass {
    TypedDict,
    Generic(Vec<Type>),
    Protocol(Vec<Type>),
    Expr(Expr),
}

impl BaseClass {
    pub fn can_apply(&self) -> bool {
        matches!(self, BaseClass::Generic(_) | BaseClass::Protocol(_))
    }

    pub fn apply(&mut self, args: Vec<Type>) {
        match self {
            BaseClass::Generic(xs) | BaseClass::Protocol(xs) => {
                xs.extend(args);
            }
            _ => panic!("cannot apply base class"),
        }
    }
}
impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn class_definition(
        &self,
        x: &StmtClassDef,
        fields: SmallSet<Name>,
        bases: &[Expr],
        legacy_tparams: &[Idx<KeyLegacyTypeParam>],
    ) -> Class {
        let scoped_tparams = self.scoped_type_params(x.type_params.as_deref());
        let bases = bases.map(|x| self.base_class_of(x));
        let tparams = self.class_tparams(&x.name, scoped_tparams, bases, legacy_tparams);
        Class::new(
            x.name.clone(),
            self.module_info().dupe(),
            tparams,
            fields.clone(),
        )
    }

    pub fn functional_class_definition(&self, name: &Identifier, fields: &SmallSet<Name>) -> Class {
        Class::new(
            name.clone(),
            self.module_info().dupe(),
            TParams::default(),
            fields.clone(),
        )
    }

    /// This helper deals with special cases where we want to intercept an `Expr`
    /// manually and create a special variant of `BaseClass` instead of calling
    /// `expr_untype` and creating a `BaseClass::Type`.
    ///
    /// TODO(stroxler): See if there's a way to express this more clearly in the types.
    fn special_base_class(&self, base_expr: &Expr) -> Option<BaseClass> {
        if let Expr::Name(name) = base_expr {
            match &*self.get(&Key::Usage(ShortIdentifier::expr_name(name))) {
                Type::Type(box Type::SpecialForm(special)) => match special {
                    SpecialForm::Protocol => Some(BaseClass::Protocol(Vec::new())),
                    SpecialForm::Generic => Some(BaseClass::Generic(Vec::new())),
                    SpecialForm::TypedDict => Some(BaseClass::TypedDict),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        }
    }

    fn base_class_of(&self, base_expr: &Expr) -> BaseClass {
        if let Some(special_base_class) = self.special_base_class(base_expr) {
            // This branch handles cases like `Protocol`
            special_base_class
        } else if let Expr::Subscript(subscript) = base_expr
            && let Some(mut special_base_class) = self.special_base_class(&subscript.value)
            && special_base_class.can_apply()
        {
            // This branch handles `Generic[...]` and `Protocol[...]`
            let args = Ast::unpack_slice(&subscript.slice).map(|x| self.expr_untype(x));
            special_base_class.apply(args);
            special_base_class
        } else {
            // This branch handles all other base classes.
            BaseClass::Expr(base_expr.clone())
        }
    }

    fn class_tparams(
        &self,
        name: &Identifier,
        scoped_tparams: Vec<TParamInfo>,
        bases: Vec<BaseClass>,
        legacy: &[Idx<KeyLegacyTypeParam>],
    ) -> TParams {
        let legacy_tparams = legacy
            .iter()
            .filter_map(|key| self.get_idx(*key).deref().parameter().cloned())
            .collect::<SmallSet<_>>();
        let legacy_map = legacy_tparams
            .iter()
            .map(|p| (p.quantified, p))
            .collect::<SmallMap<_, _>>();

        let lookup_tparam = |t: &Type| {
            let q = t.as_quantified()?;
            let p = legacy_map.get(&q);
            if p.is_none() {
                self.error(
                    name.range,
                    "Redundant type parameter declaration".to_owned(),
                );
            }
            p.map(|x| (*x).clone())
        };

        // TODO(stroxler): There are a lot of checks, such as that `Generic` only appears once
        // and no non-type-vars are used, that we can more easily detect in a dedictated class
        // validation step that validates all the bases. We are deferring these for now.
        let mut generic_tparams = SmallSet::new();
        let mut protocol_tparams = SmallSet::new();
        for base in bases.iter() {
            match base {
                BaseClass::Generic(ts) => {
                    for t in ts.iter() {
                        if let Some(p) = lookup_tparam(t) {
                            generic_tparams.insert(p);
                        }
                    }
                }
                BaseClass::Protocol(ts) if !ts.is_empty() => {
                    for t in ts.iter() {
                        if let Some(p) = lookup_tparam(t) {
                            protocol_tparams.insert(p);
                        }
                    }
                }
                _ => {}
            }
        }
        if !generic_tparams.is_empty() && !protocol_tparams.is_empty() {
            self.error(
                name.range,
                format!(
                    "Class `{}` specifies type parameters in both `Generic` and `Protocol` bases",
                    name.id,
                ),
            );
        }
        // Initialized the tparams: combine scoped and explicit type parameters
        let mut tparams = SmallSet::new();
        tparams.extend(scoped_tparams);
        tparams.extend(generic_tparams);
        tparams.extend(protocol_tparams);
        // Handle implicit tparams: if a Quantified was bound at this scope and is not yet
        // in tparams, we add it. These will be added in left-to-right order.
        let implicit_tparams_okay = tparams.is_empty();
        for p in legacy_tparams.iter() {
            if !tparams.contains(p) {
                if !implicit_tparams_okay {
                    self.error(
                        name.range,
                        format!(
                            "Class `{}` uses type variables not specified in `Generic` or `Protocol` base",
                            name.id,
                        ),
                    );
                }
                tparams.insert(p.clone());
            }
        }
        // TODO: This is a very bad variance inference algorithm.
        for tparam in tparams.iter_mut_unchecked() {
            if tparam.variance.is_none() {
                tparam.variance = Some(Variance::Invariant);
            }
        }
        self.type_params(name.range, tparams.into_iter().collect())
    }

    pub fn class_metadata_of(
        &self,
        cls: &Class,
        bases: &[Expr],
        keywords: &[(Name, Expr)],
        decorators: &[Decorator],
    ) -> ClassMetadata {
        let mut is_typed_dict = false;
        let mut is_named_tuple = false;
        let mut enum_metadata = None;
        let mut dataclass_metadata = None;
        let bases: Vec<BaseClass> = bases.iter().map(|x| self.base_class_of(x)).collect();
        let is_protocol = bases.iter().any(|x| matches!(x, BaseClass::Protocol(_)));
        let bases_with_metadata = bases
            .iter()
            .filter_map(|x| match x {
                BaseClass::Expr(x) => match self.expr_untype(x) {
                    Type::ClassType(c) => {
                        let cls = c.class_object();
                        let class_metadata = self.get_metadata_for_class(cls);
                        if class_metadata.is_typed_dict() {
                            is_typed_dict = true;
                        }
                        if class_metadata.is_named_tuple()
                        || cls.has_qname("typing", "NamedTuple")
                        {
                            is_named_tuple = true;
                        }
                        if is_protocol && !class_metadata.is_protocol() {
                            self.error(
                                x.range(),
                                "If `Protocol` is included as a base class, all other bases must be protocols.".to_owned(),
                            );
                        }
                        if dataclass_metadata.is_none() && let Some(base_dataclass) = class_metadata.dataclass_metadata() {
                            // If we inherit from a dataclass, copy its fields. Note that if this class is
                            // itself decorated with @dataclass, we'll recompute the fields and overwrite this.
                            dataclass_metadata = Some(DataclassMetadata {
                                fields: base_dataclass.fields.clone(),
                                synthesized_methods: SmallMap::new(),
                            });
                        }
                        Some((c, class_metadata))
                    }
                    Type::TypedDict(typed_dict) => {
                        is_typed_dict = true;
                        let class_object = typed_dict.class_object();
                        let class_metadata = self.get_metadata_for_class(class_object);
                        // In normal typechecking logic, TypedDicts should never be represented as ClassType.
                        // However, we convert it to a ClassType here so that MRO works properly and we can look up
                        // the types of the declared items.
                        Some((
                            ClassType::new(class_object.clone(), typed_dict.targs().clone()),
                            class_metadata,
                        ))
                    }
                    _ => None,
                },
                BaseClass::TypedDict => {
                    is_typed_dict = true;
                    None
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if is_named_tuple && bases_with_metadata.len() > 1 {
            self.error(
                cls.name().range,
                "Named tuples do not support multiple inheritance".to_owned(),
            );
        }
        let (metaclasses, keywords): (Vec<_>, Vec<(_, _)>) =
            keywords.iter().partition_map(|(n, x)| match n.as_str() {
                "metaclass" => Either::Left(x),
                _ => Either::Right((n.clone(), self.expr(x, None))),
            });

        let base_metaclasses = bases_with_metadata
            .iter()
            .filter_map(|(b, metadata)| metadata.metaclass().map(|m| (&b.name().id, m)))
            .collect::<Vec<_>>();
        let metaclass =
            self.calculate_metaclass(cls, metaclasses.into_iter().next(), &base_metaclasses);
        if let Some(metaclass) = &metaclass {
            self.check_base_class_metaclasses(cls, metaclass, &base_metaclasses);
            if self.solver().is_subset_eq(
                &Type::ClassType(metaclass.clone()),
                &Type::ClassType(self.stdlib.enum_meta()),
                self.type_order(),
            ) {
                if !cls.tparams().is_empty() {
                    self.error(cls.name().range, "Enums may not be generic.".to_owned());
                }
                enum_metadata = Some(EnumMetadata {
                    // A generic enum is an error, but we create Any type args anyway to handle it gracefully.
                    cls: ClassType::new(cls.clone(), self.create_default_targs(cls, None)),
                })
            }
            if is_typed_dict {
                self.error(
                    cls.name().range,
                    "Typed dictionary definitions may not specify a metaclass.".to_owned(),
                );
            }
        }
        for decorator in decorators {
            let ty_decorator = self.expr(&decorator.expression, None);
            if matches!(
                ty_decorator.callee_kind(),
                Some(CalleeKind::Callable(CallableKind::Dataclass))
            ) {
                let fields = self.get_dataclass_fields(cls, &bases_with_metadata);
                let init = self.get_dataclass_init(cls, &fields);
                dataclass_metadata = Some(DataclassMetadata {
                    fields: fields.into_keys().collect(),
                    synthesized_methods: smallmap! { dunder::INIT => init },
                });
            }
        }
        if is_typed_dict
            && let Some(bad) = bases_with_metadata.iter().find(|x| !x.1.is_typed_dict())
        {
            self.error(
                cls.name().range,
                format!("`{}` is not a typed dictionary. Typed dictionary definitions may only extend other typed dictionaries.", bad.0),
            );
        }
        ClassMetadata::new(
            cls,
            bases_with_metadata,
            metaclass,
            keywords,
            is_typed_dict,
            is_named_tuple,
            enum_metadata,
            is_protocol,
            dataclass_metadata,
            self.errors(),
        )
    }

    fn calculate_metaclass(
        &self,
        cls: &Class,
        raw_metaclass: Option<&Expr>,
        base_metaclasses: &[(&Name, &ClassType)],
    ) -> Option<ClassType> {
        let direct_meta = raw_metaclass.and_then(|x| self.direct_metaclass(cls, x));

        if let Some(metaclass) = direct_meta {
            Some(metaclass)
        } else {
            let mut inherited_meta: Option<ClassType> = None;
            for (_, m) in base_metaclasses.iter() {
                let m = (*m).clone();
                let accept_m = match &inherited_meta {
                    None => true,
                    Some(inherited) => self.solver().is_subset_eq(
                        &Type::ClassType(m.clone()),
                        &Type::ClassType(inherited.clone()),
                        self.type_order(),
                    ),
                };
                if accept_m {
                    inherited_meta = Some(m);
                }
            }
            inherited_meta
        }
    }

    fn check_base_class_metaclasses(
        &self,
        cls: &Class,
        metaclass: &ClassType,
        base_metaclasses: &[(&Name, &ClassType)],
    ) {
        // It is a runtime error to define a class whose metaclass (whether
        // specified directly or through inheritance) is not a subtype of all
        // base class metaclasses.
        let metaclass_type = Type::ClassType(metaclass.clone());
        for (base_name, m) in base_metaclasses.iter() {
            let base_metaclass_type = Type::ClassType((*m).clone());
            if !self
                .solver()
                .is_subset_eq(&metaclass_type, &base_metaclass_type, self.type_order())
            {
                self.error(
                    cls.name().range,
                    format!(
                        "Class `{}` has metaclass `{}` which is not a subclass of metaclass `{}` from base class `{}`",
                        cls.name().id,
                        metaclass_type,
                        base_metaclass_type,
                        base_name,
                    )
                );
            }
        }
    }

    fn direct_metaclass(&self, cls: &Class, raw_metaclass: &Expr) -> Option<ClassType> {
        match self.expr_untype(raw_metaclass) {
            Type::ClassType(meta) => {
                if self.solver().is_subset_eq(
                    &Type::ClassType(meta.clone()),
                    &Type::ClassType(self.stdlib.builtins_type()),
                    self.type_order(),
                ) {
                    Some(meta)
                } else {
                    self.error(
                        raw_metaclass.range(),
                        format!(
                            "Metaclass of `{}` has type `{}` which is not a subclass of `type`",
                            cls.name().id,
                            Type::ClassType(meta),
                        ),
                    );
                    None
                }
            }
            ty => {
                self.error(
                    cls.name().range,
                    format!(
                        "Metaclass of `{}` has type `{}` is not a simple class type.",
                        cls.name().id,
                        ty,
                    ),
                );
                None
            }
        }
    }

    pub fn get_metadata_for_class(&self, cls: &Class) -> Arc<ClassMetadata> {
        self.get_from_class(cls, &KeyClassMetadata(ShortIdentifier::new(cls.name())))
    }

    fn get_enum_from_class(&self, cls: &Class) -> Option<EnumMetadata> {
        self.get_metadata_for_class(cls).enum_metadata().cloned()
    }

    pub fn get_enum_from_class_type(&self, class_type: &ClassType) -> Option<EnumMetadata> {
        self.get_enum_from_class(class_type.class_object())
    }

    /// Given an identifier, see whether it is bound to an enum class. If so,
    /// return the enum, otherwise return `None`.
    pub fn get_enum_from_name(&self, name: Identifier) -> Option<EnumMetadata> {
        self.get_enum_from_key(
            self.bindings()
                .key_to_idx(&Key::Usage(ShortIdentifier::new(&name))),
        )
    }

    pub fn get_enum_from_key(&self, key: Idx<Key>) -> Option<EnumMetadata> {
        // TODO(stroxler): Eventually, we should raise type errors on generic Enum because
        // this doesn't make semantic sense. But in the meantime we need to be robust against
        // this possibility.
        match self.get_idx(key).deref() {
            Type::ClassDef(class) => self.get_enum_from_class(class),
            _ => None,
        }
    }

    fn check_and_create_targs(&self, cls: &Class, targs: Vec<Type>, range: TextRange) -> TArgs {
        let tparams = cls.tparams();
        let nargs = targs.len();
        let mut checked_targs = Vec::new();
        for pair in tparams.iter().zip_longest(targs) {
            match pair {
                EitherOrBoth::Both(_, arg) => {
                    checked_targs.push(arg);
                }
                EitherOrBoth::Left(param) if let Some(default) = &param.default => {
                    checked_targs.push(default.clone());
                }
                _ => {
                    self.error(
                        range,
                        format!(
                            "Expected {} for class `{}`, got {}.",
                            count(tparams.len(), "type argument"),
                            cls.name(),
                            nargs
                        ),
                    );
                    // We have either too few or too many targs. If too few, pad out with Any.
                    // If there are too many, the extra are ignored.
                    checked_targs
                        .extend(vec![Type::any_error(); tparams.len().saturating_sub(nargs)]);
                    break;
                }
            }
        }
        TArgs::new(checked_targs)
    }

    fn create_default_targs(
        &self,
        cls: &Class,
        // Placeholder for strict mode: we want to force callers to pass a range so
        // that we don't refactor in a way where none is available, but this is unused
        // because we do not have a strict mode yet.
        _range: Option<TextRange>,
    ) -> TArgs {
        let tparams = cls.tparams();
        if tparams.is_empty() {
            TArgs::default()
        } else {
            // TODO(stroxler): We should error here, but the error needs to be
            // configurable in the long run, and also suppressed in dependencies
            // no matter what the configuration is.
            //
            // Our plumbing isn't ready for that yet, so for now we are silently
            // using gradual type arguments.
            TArgs::new(vec![Type::any_error(); tparams.len()])
        }
    }

    fn type_of_instance(&self, cls: &Class, targs: TArgs) -> Type {
        let metadata = self.get_metadata_for_class(cls);
        if metadata.is_typed_dict() {
            let fields = self.get_typed_dict_fields(cls, &targs);
            Type::TypedDict(TypedDict::new(cls.dupe(), targs, fields))
        } else {
            Type::ClassType(ClassType::new(cls.dupe(), targs))
        }
    }

    /// Given a class or typed dictionary and some (explicit) type arguments, construct a `Type`
    /// that represents the type of an instance of the class or typed dictionary with those `targs`.
    pub fn specialize(&self, cls: &Class, targs: Vec<Type>, range: TextRange) -> Type {
        let targs = self.check_and_create_targs(cls, targs, range);
        self.type_of_instance(cls, targs)
    }

    /// Given a class or typed dictionary, create a `Type` that represents to an instance annotated
    /// with the class or typed dictionary's bare name. This will either have empty type arguments if the
    /// class or typed dictionary is not generic, or type arguments populated with gradual types if
    /// it is (e.g. applying an annotation of `list` to a variable means
    /// `list[Any]`).
    ///
    /// We require a range because depending on the configuration we may raise
    /// a type error when a generic class or typed dictionary is promoted using gradual types.
    pub fn promote(&self, cls: &Class, range: TextRange) -> Type {
        let targs = self.create_default_targs(cls, Some(range));
        self.type_of_instance(cls, targs)
    }

    /// Private version of `promote` that does not potentially
    /// raise strict mode errors. Should only be used for unusual scenarios.
    fn promote_silently(&self, cls: &Class) -> Type {
        let targs = self.create_default_targs(cls, None);
        self.type_of_instance(cls, targs)
    }

    pub fn unwrap_class_object_silently(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::ClassDef(c) => Some(self.promote_silently(c)),
            Type::TypeAlias(ta) => self.unwrap_class_object_silently(&ta.as_value(self.stdlib)),
            _ => None,
        }
    }

    /// Creates a type from the class with fresh variables for its type parameters.
    pub fn instantiate_fresh(&self, cls: &Class) -> Type {
        let qs = cls.tparams().quantified().collect::<Vec<_>>();
        let targs = TArgs::new(qs.map(|q| Type::Quantified(*q)));
        let promoted_cls = Type::type_form(self.type_of_instance(cls, targs));
        self.solver()
            .fresh_quantified(qs.as_slice(), promoted_cls, self.uniques)
            .1
    }

    /// Get an ancestor `ClassType`, in terms of the type parameters of `class`.
    fn get_ancestor(&self, class: &Class, want: &Class) -> Option<ClassType> {
        self.get_metadata_for_class(class)
            .ancestors(self.stdlib)
            .find(|ancestor| ancestor.class_object() == want)
            .cloned()
    }

    /// Is `want` a superclass of `class` in the class hierarchy? Will return `false` if
    /// `want` is a protocol, unless it is explicitly marked as a base class in the MRO.
    pub fn has_superclass(&self, class: &Class, want: &Class) -> bool {
        class == want || self.get_ancestor(class, want).is_some()
    }

    /// Return the type representing `class` upcast to `want`, if `want` is a
    /// supertype of `class` in the class hierarchy. Will return `None` if
    /// `want` is not a superclass, including if `want` is a protocol (unless it
    /// explicitly appears in the MRO).
    pub fn as_superclass(&self, class: &ClassType, want: &Class) -> Option<ClassType> {
        if class.class_object() == want {
            Some(class.clone())
        } else {
            self.get_ancestor(class.class_object(), want)
                .map(|ancestor| ancestor.substitute(&class.substitution()))
        }
    }

    pub fn calculate_class_field(
        &self,
        name: &Name,
        value_ty: &Type,
        annotation: Option<&Annotation>,
        initialization: ClassFieldInitialization,
        class_key: Idx<Key>,
        range: TextRange,
    ) -> ClassField {
        if let Some(enum_) = self.get_enum_from_key(class_key)
            && enum_.get_member(name).is_some()
            && matches!(initialization, ClassFieldInitialization::Class)
        {
            if annotation.is_some() {
                self.error(range, format!("Enum member `{}` may not be annotated directly. Instead, annotate the _value_ attribute.", name));
            }

            if let Some(enum_value_ty) = self.type_of_enum_value(enum_) {
                if !matches!(value_ty, Type::Tuple(_))
                    && !self
                        .solver()
                        .is_subset_eq(value_ty, &enum_value_ty, self.type_order())
                {
                    self.error(range, format!("The value for enum member `{}` must match the annotation of the _value_ attribute.", name));
                }
            }
        }
        if self.is_key_typed_dict(class_key)
            && matches!(initialization, ClassFieldInitialization::Class)
        {
            self.error(
                range,
                format!("TypedDict item `{}` may not be initialized.", name),
            );
        }
        let (ty, ann) = if let Some(ann) = annotation {
            match &ann.ty {
                Some(ty) => (ty, Some(ann)),
                None => (value_ty, Some(ann)),
            }
        } else {
            (value_ty, None)
        };
        ClassField::new(ty.clone(), ann.cloned(), initialization)
    }

    fn get_class_field(&self, cls: &Class, name: &Name) -> Option<ClassField> {
        let metadata = self.get_metadata_for_class(cls);
        if let Some(dataclass) = metadata.dataclass_metadata()
            && let Some(method) = dataclass.synthesized_methods.get(name)
        {
            Some(ClassField::new(
                method.clone(),
                None,
                ClassFieldInitialization::Class,
            ))
        } else if cls.contains(name) {
            let field = self.get_from_class(
                cls,
                &KeyClassField(ShortIdentifier::new(cls.name()), name.clone()),
            );
            Some((*field).clone())
        } else {
            None
        }
    }

    fn get_class_member(&self, cls: &Class, name: &Name) -> Option<WithDefiningClass<ClassField>> {
        if let Some(field) = self.get_class_field(cls, name) {
            Some(WithDefiningClass {
                value: field,
                defining_class: cls.dupe(),
            })
        } else {
            self.get_metadata_for_class(cls)
                .ancestors(self.stdlib)
                .filter_map(|ancestor| {
                    self.get_class_field(ancestor.class_object(), name)
                        .map(|field| WithDefiningClass {
                            value: field.instantiate_for(ancestor),
                            defining_class: ancestor.class_object().dupe(),
                        })
                })
                .next()
        }
    }

    // Get every member of a class, including those declared in parent classes.
    fn get_all_members(&self, cls: &Class) -> SmallMap<Name, (ClassField, Class)> {
        let mut members = SmallMap::new();
        for name in cls.fields() {
            if let Some(field) = self.get_class_field(cls, name) {
                members.insert(name.clone(), (field, cls.dupe()));
            }
        }
        for ancestor in self.get_metadata_for_class(cls).ancestors(self.stdlib) {
            for name in ancestor.class_object().fields() {
                if !members.contains_key(name) {
                    if let Some(field) = self.get_class_field(ancestor.class_object(), name) {
                        members.insert(
                            name.clone(),
                            (
                                field.instantiate_for(ancestor),
                                ancestor.class_object().dupe(),
                            ),
                        );
                    }
                }
            }
        }
        members
    }

    pub fn get_all_member_names(&self, cls: &Class) -> SmallSet<Name> {
        self.get_all_members(cls)
            .keys()
            .cloned()
            .collect::<SmallSet<_>>()
    }

    pub fn get_instance_attribute(&self, cls: &ClassType, name: &Name) -> Option<Attribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|member| member.value.as_instance_attribute(cls))
    }

    /// Gets an attribute from a class definition.
    ///
    /// Returns `None` if there is no such attribute, otherwise an `Attribute` object
    /// that describes whether access is allowed and the type if so.
    ///
    /// Access is disallowed for instance-only attributes and for attributes whose
    /// type contains a class-scoped type parameter - e.g., `class A[T]: x: T`.
    pub fn get_class_attribute(&self, cls: &Class, name: &Name) -> Option<Attribute> {
        if let Some(e) = self.get_enum_from_class(cls)
            && let Some(enum_member) = e.get_member(name)
        {
            Some(Attribute::read_write(Type::Literal(enum_member)))
        } else {
            let member = self.get_class_member(cls, name)?.value;
            Some(member.as_class_attribute(cls))
        }
    }

    /// Get the class's `__new__` method.
    ///
    /// This lookup skips normal method binding logic (it behaves like a cross
    /// between a classmethod and a constructor; downstream code handles this
    /// using the raw callable type).
    pub fn get_dunder_new(&self, cls: &ClassType) -> Option<Type> {
        let new_member = self.get_class_member(cls.class_object(), &dunder::NEW)?;
        if new_member.defined_on(self.stdlib.object_class_type().class_object()) {
            // The default behavior of `object.__new__` is already baked into our implementation of
            // class construction; we only care about `__new__` if it is overridden.
            None
        } else {
            new_member.value.as_raw_special_method_type(cls)
        }
    }

    /// Get the class's `__init__` method, if we should analyze it
    /// We skip analyzing the call to `__init__` if:
    /// (1) it isn't defined (possible if we've been passed a custom typeshed), or
    /// (2) the class overrides `object.__new__` but not `object.__init__`, in wich case the
    ///     `__init__` call always succeeds at runtime.
    pub fn get_dunder_init(&self, cls: &ClassType, overrides_new: bool) -> Option<Type> {
        let init_method = self.get_class_member(cls.class_object(), &dunder::INIT)?;
        if !(overrides_new
            && init_method.defined_on(self.stdlib.object_class_type().class_object()))
        {
            init_method.value.as_special_method_type(cls)
        } else {
            None
        }
    }

    /// Get the metaclass `__call__` method.
    pub fn get_metaclass_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        let metadata = self.get_metadata_for_class(cls.class_object());
        let metaclass = metadata.metaclass()?;
        let attr = self.get_class_member(metaclass.class_object(), &dunder::CALL)?;
        if attr.defined_on(self.stdlib.builtins_type().class_object()) {
            // The behavior of `type.__call__` is already baked into our implementation of constructors,
            // so we can skip analyzing it at the type level.
            None
        } else {
            attr.value.as_special_method_type(metaclass)
        }
    }

    fn is_key_typed_dict(&self, key: Idx<Key>) -> bool {
        match self.get_idx(key).deref() {
            Type::ClassDef(cls) => self.get_metadata_for_class(cls).is_typed_dict(),
            _ => false,
        }
    }

    fn get_typed_dict_fields(&self, cls: &Class, targs: &TArgs) -> SmallMap<Name, TypedDictField> {
        let tparams = cls.tparams();
        let substitution = Substitution::new(
            tparams
                .quantified()
                .zip(targs.as_slice().iter().cloned())
                .collect(),
        );
        self.get_all_members(cls)
            .iter()
            .filter_map(|(name, (field, cls))| {
                let metadata = self.get_metadata_for_class(cls);
                if !metadata.is_typed_dict() {
                    return None;
                }
                if let ClassField(ClassFieldInner::Simple {
                    annotation:
                        Some(Annotation {
                            ty: Some(ty),
                            qualifiers,
                        }),
                    ..
                }) = field
                {
                    let is_total = metadata
                        .get_keyword(&Name::new("total"))
                        .map_or(true, |ty| match ty {
                            Type::Literal(Lit::Bool(b)) => b,
                            _ => true,
                        });
                    Some((
                        name.clone(),
                        TypedDictField {
                            ty: substitution.substitute(ty.clone()),
                            required: if qualifiers.contains(&Qualifier::Required) {
                                true
                            } else if qualifiers.contains(&Qualifier::NotRequired) {
                                false
                            } else {
                                is_total
                            },
                            read_only: qualifiers.contains(&Qualifier::ReadOnly),
                        },
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Gets dataclass fields for an `@dataclass`-decorated class.
    fn get_dataclass_fields(
        &self,
        cls: &Class,
        bases_with_metadata: &[(ClassType, Arc<ClassMetadata>)],
    ) -> SmallMap<Name, ClassField> {
        let mut all_fields = SmallMap::new();
        for (base, metadata) in bases_with_metadata.iter().rev() {
            if let Some(dataclass) = metadata.dataclass_metadata() {
                for name in &dataclass.fields {
                    if let Some(field) = self.get_class_member(base.class_object(), name) {
                        all_fields.insert(name.clone(), field.value.instantiate_for(base));
                    }
                }
            }
        }
        for name in cls.fields() {
            if let Some(
                field @ ClassField(ClassFieldInner::Simple {
                    annotation: Some(Annotation { ty: Some(_), .. }),
                    ..
                }),
            ) = self.get_class_field(cls, name)
            {
                all_fields.insert(name.clone(), field);
            }
        }
        all_fields
    }

    /// Gets a dataclass field as a function param.
    fn get_dataclass_param(&self, name: &Name, field: &ClassField) -> Param {
        let ClassField(ClassFieldInner::Simple {
            ty,
            annotation: _,
            initialization,
        }) = field;
        let required = match initialization {
            ClassFieldInitialization::Class => Required::Required,
            ClassFieldInitialization::Instance => Required::Optional,
        };
        Param::Pos(name.clone(), ty.clone(), required)
    }

    /// Gets __init__ method for an `@dataclass`-decorated class.
    fn get_dataclass_init(&self, cls: &Class, fields: &SmallMap<Name, ClassField>) -> Type {
        let mut params = vec![Param::Pos(
            Name::new("self"),
            cls.self_type(),
            Required::Required,
        )];
        for (name, field) in fields {
            params.push(self.get_dataclass_param(name, field));
        }
        Type::Callable(
            Box::new(Callable::list(ParamList::new(params), Type::None)),
            CallableKind::Def,
        )
    }
}
