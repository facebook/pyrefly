/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_types::callable::Callable;
use pyrefly_types::special_form::SpecialForm;
use pyrefly_types::types::Union;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::abstract_class::AbstractClassMembers;
use crate::alt::types::class_bases::ClassBases;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::EnumMetadata;
use crate::binding::binding::KeyAbstractClassCheck;
use crate::binding::binding::KeyClassBaseType;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::error::collector::ErrorCollector;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassDefIndex;
use crate::types::class::ClassFieldProperties;
use crate::types::class::ClassType;
use crate::types::types::TParams;
use crate::types::types::Type;

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
            Type::Union(box Union { members: xs, .. }) => xs
                .iter()
                .all(|x| self.is_compatible_constructor_return(x, class)),
            _ => false,
        }
    }

    pub fn class_definition(
        &self,
        def_index: ClassDefIndex,
        x: &StmtClassDef,
        parent: &NestingContext,
        fields: SmallMap<Name, ClassFieldProperties>,
        tparams_require_binding: bool,
        errors: &ErrorCollector,
    ) -> Class {
        let name = &x.name;
        let precomputed_tparams = if tparams_require_binding {
            None
        } else {
            Some(self.calculate_class_tparams_no_legacy(name, x.type_params.as_deref(), errors))
        };
        Class::new(
            def_index,
            x.name.clone(),
            parent.dupe(),
            self.module().dupe(),
            precomputed_tparams,
            fields,
        )
    }

    pub fn functional_class_definition(
        &self,
        def_index: ClassDefIndex,
        name: &Identifier,
        parent: &NestingContext,
        fields: &SmallMap<Name, ClassFieldProperties>,
    ) -> Class {
        Class::new(
            def_index,
            name.clone(),
            parent.dupe(),
            self.module().dupe(),
            Some(Arc::new(TParams::default())),
            fields.clone(),
        )
    }

    pub fn get_metadata_for_class(&self, cls: &Class) -> Arc<ClassMetadata> {
        self.get_from_class(cls, &KeyClassMetadata(cls.index()))
            .unwrap_or_else(|| Arc::new(ClassMetadata::recursive()))
    }

    pub fn get_abstract_members_for_class(&self, cls: &Class) -> Arc<AbstractClassMembers> {
        self.get_from_class(cls, &KeyAbstractClassCheck(cls.index()))
            .unwrap_or_else(|| Arc::new(AbstractClassMembers::recursive()))
    }

    pub fn get_base_types_for_class(&self, cls: &Class) -> Arc<ClassBases> {
        self.get_from_class(cls, &KeyClassBaseType(cls.index()))
            .unwrap_or_else(|| Arc::new(ClassBases::recursive()))
    }

    pub fn get_mro_for_class(&self, cls: &Class) -> Arc<ClassMro> {
        self.get_from_class(cls, &KeyClassMro(cls.index()))
            .unwrap_or_else(|| Arc::new(ClassMro::recursive()))
    }

    pub fn get_class_field_map(&self, cls: &Class) -> SmallMap<Name, Arc<ClassField>> {
        let fields = cls.fields();
        let mut map = SmallMap::with_capacity(fields.len());

        for name in fields {
            let key = KeyClassField(cls.index(), name.clone());
            if let Some(field) = self.get_from_class(cls, &key) {
                map.insert(name.clone(), field);
            }
        }
        map
    }

    pub fn get_enum_from_class(&self, cls: &Class) -> Option<EnumMetadata> {
        self.get_metadata_for_class(cls).enum_metadata().cloned()
    }

    pub fn unwrap_class_object_silently(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::ClassDef(c) if c.is_builtin("tuple") => Some(self.instantiate_fresh_tuple()),
            Type::ClassDef(c) => Some(self.instantiate_fresh_class(c)),
            Type::TypeAlias(ta) => self.unwrap_class_object_silently(&ta.as_value(self.stdlib)),
            // Note that for the purposes of type narrowing, we always unwrap Type::Type(Type::ClassType),
            // but it's not always a valid argument to isinstance/issubclass. expr_infer separately checks
            // whether the argument is valid.
            Type::Type(box ty @ (Type::ClassType(_) | Type::Quantified(_) | Type::SelfType(_))) => {
                Some(ty.clone())
            }
            Type::Type(box Type::Tuple(_)) => Some(self.instantiate_fresh_tuple()),
            Type::Type(box Type::Any(a)) => Some(a.propagate()),
            Type::Type(box Type::SpecialForm(SpecialForm::Callable)) => Some(Type::Callable(
                Box::new(Callable::ellipsis(Type::any_implicit())),
            )),
            Type::None | Type::Type(box Type::None) => Some(Type::None),
            Type::ClassType(cls) if cls.is_builtin("type") => Some(Type::any_implicit()),
            Type::Any(_) => Some(ty.clone()),
            _ => None,
        }
    }

    /// Get an ancestor `ClassType`, in terms of the type parameters of `class`.
    fn get_ancestor(&self, class: &Class, want: &Class) -> Option<ClassType> {
        self.get_mro_for_class(class)
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
                .map(|ancestor| ancestor.substitute_with(&class.substitution()))
        }
    }

    pub fn extends_any(&self, cls: &Class) -> bool {
        self.get_metadata_for_class(cls).has_base_any()
    }

    pub fn class_self_param(&self, cls: &Class, posonly: bool) -> Param {
        let ty = self.instantiate(cls);
        let req = Required::Required;
        let name = Name::new_static("self");
        if posonly {
            Param::PosOnly(Some(name), ty, req)
        } else {
            Param::Pos(name, ty, req)
        }
    }
}
