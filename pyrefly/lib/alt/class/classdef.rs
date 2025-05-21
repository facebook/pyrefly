/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::error::collector::ErrorCollector;
use crate::graph::index::Idx;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassFieldProperties;
use crate::types::class::ClassIndex;
use crate::types::class::ClassType;
use crate::types::class::TArgs;
use crate::types::typed_dict::TypedDict;
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::util::prelude::SliceExt;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // Given a constructor (__new__ or metaclass __call__) that returns `ty`, return true if the type is:
    // - SelfType or ClassType representing some subclass of `class`
    // - union only containing the aforementioned types
    // Docs:
    // https://typing.python.org/en/latest/spec/constructors.html#new-method
    // https://typing.python.org/en/latest/spec/constructors.html#converting-a-constructor-to-callable
    pub fn is_compatible_constructor_return(&self, ty: &Type, class: &Class) -> bool {
        match ty {
            Type::SelfType(ty_cls) | Type::ClassType(ty_cls) => {
                self.has_superclass(ty_cls.class_object(), class)
            }
            Type::Union(xs) => xs
                .iter()
                .all(|x| self.is_compatible_constructor_return(x, class)),
            _ => false,
        }
    }

    pub fn class_definition(
        &self,
        index: ClassIndex,
        x: &StmtClassDef,
        fields: SmallMap<Name, ClassFieldProperties>,
        bases: &[Expr],
        legacy_tparams: &[Idx<KeyLegacyTypeParam>],
        errors: &ErrorCollector,
    ) -> Class {
        let scoped_tparams = self.scoped_type_params(x.type_params.as_deref(), errors);
        let bases = bases.map(|x| self.base_class_of(x, errors));
        let tparams = self.class_tparams(&x.name, scoped_tparams, bases, legacy_tparams, errors);
        Class::new(
            index,
            x.name.clone(),
            self.module_info().dupe(),
            tparams,
            fields,
        )
    }

    pub fn functional_class_definition(
        &self,
        index: ClassIndex,
        name: &Identifier,
        fields: &SmallMap<Name, ClassFieldProperties>,
    ) -> Class {
        Class::new(
            index,
            name.clone(),
            self.module_info().dupe(),
            TParams::default(),
            fields.clone(),
        )
    }

    pub fn get_metadata_for_class(&self, cls: &Class) -> Arc<ClassMetadata> {
        self.get_from_class(cls, &KeyClassMetadata(cls.index()))
    }

    pub fn get_enum_from_class(&self, cls: &Class) -> Option<EnumMetadata> {
        self.get_metadata_for_class(cls).enum_metadata().cloned()
    }

    pub fn get_enum_from_class_type(&self, class_type: &ClassType) -> Option<EnumMetadata> {
        self.get_enum_from_class(class_type.class_object())
    }

    /// Creates default type arguments for a class, falling back to Any for type parameters without defaults.
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
            TArgs::new(
                tparams
                    .iter()
                    .map(|x| x.quantified.as_gradual_type())
                    .collect(),
            )
        }
    }

    /// Silently promotes a Class to a ClassType, using default type arguments. It is up to the
    /// caller to ensure they are not calling this method on a TypedDict class, which should be
    /// promoted to TypedDict instead of ClassType.
    pub fn promote_nontypeddict_silently_to_classtype(&self, cls: &Class) -> ClassType {
        ClassType::new(cls.dupe(), self.create_default_targs(cls, None))
    }

    fn type_of_instance(&self, cls: &Class, targs: TArgs) -> Type {
        let metadata = self.get_metadata_for_class(cls);
        if metadata.is_typed_dict() {
            Type::TypedDict(TypedDict::new(cls.dupe(), targs))
        } else {
            Type::ClassType(ClassType::new(cls.dupe(), targs))
        }
    }

    /// Given a class or typed dictionary and some (explicit) type arguments, construct a `Type`
    /// that represents the type of an instance of the class or typed dictionary with those `targs`.
    ///
    /// Note how this differs from `promote` and `instantiate`:
    /// specialize(list, [int]) == list[int]
    /// promote(list) == list[Any]
    /// instantiate(list) == list[T]
    pub fn specialize(
        &self,
        cls: &Class,
        targs: Vec<Type>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let targs = if cls.tparams().is_empty() && self.get_metadata_for_class(cls).has_base_any() {
            // If we didn't find type parameters for a class that inherits from Any, we don't know
            // how many parameters it has. Accept any number of arguments (by ignoring them).
            TArgs::default()
        } else {
            self.check_and_create_targs(cls.name(), cls.tparams(), targs, range, errors)
        };
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
    ///
    /// Note how this differs from `specialize` and `instantiate`:
    /// specialize(list, [int]) == list[int]
    /// promote(list) == list[Any]
    /// instantiate(list) == list[T]
    pub fn promote(&self, cls: &Class, range: TextRange) -> Type {
        let targs = self.create_default_targs(cls, Some(range));
        self.type_of_instance(cls, targs)
    }

    /// Version of `promote` that does not potentially raise errors.
    /// Should only be used for unusual scenarios.
    pub fn promote_silently(&self, cls: &Class) -> Type {
        let targs = self.create_default_targs(cls, None);
        self.type_of_instance(cls, targs)
    }

    /// Given a class or typed dictionary, create a `Type` that represents a generic instance of
    /// the class or typed dictionary.
    ///
    /// Note how this differs from `specialize` and `promote`:
    /// specialize(list, [int]) == list[int]
    /// promote(list) == list[Any]
    /// instantiate(list) == list[T]
    pub fn instantiate(&self, cls: &Class) -> Type {
        self.type_of_instance(cls, cls.tparams_as_targs())
    }

    /// Instantiates a class or typed dictionary with fresh variables for its type parameters.
    pub fn instantiate_fresh(&self, cls: &Class) -> Type {
        self.solver()
            .fresh_quantified(cls.tparams(), self.instantiate(cls), self.uniques)
            .1
    }

    pub fn unwrap_class_object_silently(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::ClassDef(c) if c.is_builtin("tuple") => Some(Type::any_tuple()),
            Type::ClassDef(c) => Some(self.instantiate_fresh(c)),
            Type::TypeAlias(ta) => self.unwrap_class_object_silently(&ta.as_value(self.stdlib)),
            // Note that for the purposes of type narrowing, we always unwrap Type::Type(Type::ClassType),
            // but it's not always a valid argument to isinstance/issubclass. expr_infer separately checks
            // whether the argument is valid.
            Type::Type(box ty @ (Type::ClassType(_) | Type::Quantified(_))) => Some(ty.clone()),
            Type::Type(box Type::Tuple(_)) => Some(Type::any_tuple()),
            Type::ClassType(cls) if cls.is_builtin("type") => Some(Type::any_implicit()),
            Type::Any(_) => Some(ty.clone()),
            _ => None,
        }
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

    pub fn extends_any(&self, cls: &Class) -> bool {
        self.get_metadata_for_class(cls).has_base_any()
    }

    pub fn class_self_param(&self, cls: &Class, named: bool) -> Param {
        let ty = self.instantiate(cls);
        let req = Required::Required;
        if named {
            Param::Pos(Name::new_static("self"), ty, req)
        } else {
            Param::PosOnly(ty, req)
        }
    }
}
