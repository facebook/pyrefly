/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::name::Name;

use crate::class::Class;
use crate::class::ClassType;
use crate::types::TArgs;
use crate::types::TParams;
use crate::types::Type;

#[derive(Debug, Clone)]
struct StdlibError {
    bootstrapping: bool,
    name: &'static str,
}

type StdlibResult<T> = Result<T, StdlibError>;

#[derive(Debug, Clone)]
pub struct Stdlib {
    str: StdlibResult<ClassType>,
    bool: StdlibResult<ClassType>,
    int: StdlibResult<ClassType>,
    bytes: StdlibResult<ClassType>,
    bytearray: StdlibResult<ClassType>,
    float: StdlibResult<ClassType>,
    complex: StdlibResult<ClassType>,
    decimal: StdlibResult<ClassType>,
    date: StdlibResult<ClassType>,
    datetime: StdlibResult<ClassType>,
    time: StdlibResult<ClassType>,
    timedelta: StdlibResult<ClassType>,
    path: StdlibResult<ClassType>,
    uuid: StdlibResult<ClassType>,
    slice: StdlibResult<(Class, Arc<TParams>)>,
    base_exception: StdlibResult<ClassType>,
    /// Introduced in Python 3.11.
    base_exception_group: Option<StdlibResult<(Class, Arc<TParams>)>>,
    /// Introduced in Python 3.11.
    exception_group: Option<StdlibResult<(Class, Arc<TParams>)>>,
    list: StdlibResult<(Class, Arc<TParams>)>,
    dict: StdlibResult<(Class, Arc<TParams>)>,
    deque: StdlibResult<(Class, Arc<TParams>)>,
    frozenset: StdlibResult<(Class, Arc<TParams>)>,
    dict_items: StdlibResult<(Class, Arc<TParams>)>,
    dict_keys: StdlibResult<(Class, Arc<TParams>)>,
    dict_values: StdlibResult<(Class, Arc<TParams>)>,
    mapping: StdlibResult<(Class, Arc<TParams>)>,
    set: StdlibResult<(Class, Arc<TParams>)>,
    tuple: StdlibResult<(Class, Arc<TParams>)>,
    iterable: StdlibResult<(Class, Arc<TParams>)>,
    async_iterable: StdlibResult<(Class, Arc<TParams>)>,
    async_iterator: StdlibResult<(Class, Arc<TParams>)>,
    mutable_sequence: StdlibResult<(Class, Arc<TParams>)>,
    generator: StdlibResult<(Class, Arc<TParams>)>,
    async_generator: StdlibResult<(Class, Arc<TParams>)>,
    awaitable: StdlibResult<(Class, Arc<TParams>)>,
    coroutine: StdlibResult<(Class, Arc<TParams>)>,
    type_var: StdlibResult<ClassType>,
    /// Defined in `typing_extensions` util 3.13, defined in `typing` since 3.10.
    /// After 3.13, `typing_extensions` reexports from `typing`.
    /// For 3.10 to 3.12 defined separately in both locations.
    param_spec: StdlibResult<ClassType>,
    /// Moved from `typing` to `typing_extensions` in 3.10.
    param_spec_args: StdlibResult<ClassType>,
    /// Moved from `typing` to `typing_extensions` in 3.10.
    param_spec_kwargs: StdlibResult<ClassType>,
    /// Defined in `typing_extensions` until 3.13, defined in `typing` since 3.11.
    /// After 3.13, `typing_extensions` reexports from `typing`.
    /// For 3.11 and 3.12 defined separately in both locations.
    type_var_tuple: StdlibResult<ClassType>,
    /// Defined in `typing_extensions` until 3.14, defined in `typing` since 3.12.
    /// After 3.14, `typing_extensions` reexports from `typing`.
    /// For 3.12 and 3.13 defined separately in both locations.
    type_alias_type: StdlibResult<ClassType>,
    traceback_type: StdlibResult<ClassType>,
    builtins_type: StdlibResult<ClassType>,
    /// Introduced in Python 3.10.
    ellipsis_type: Option<StdlibResult<ClassType>>,
    /// Moved from `_typeshed` to `types` in 3.10.
    none_type: StdlibResult<ClassType>,
    function_type: StdlibResult<ClassType>,
    method_type: StdlibResult<ClassType>,
    enum_meta: StdlibResult<ClassType>,
    enum_flag: StdlibResult<ClassType>,
    enum_class: StdlibResult<ClassType>,
    /// A fallback class that contains attributes that all NamedTuple subclasses share. Note that
    /// this class has no direct runtime equivalent; typing.NamedTuple is a class in some Python
    /// versions and a function in others.
    named_tuple_fallback: StdlibResult<ClassType>,
    /// A fallback class that contains attributes that all TypedDict subclasses share. Note that
    /// this class does not exist at runtime.
    typed_dict_fallback: StdlibResult<ClassType>,
    property: StdlibResult<ClassType>,
    object: StdlibResult<ClassType>,
    /// Introduced in Python 3.10.
    union_type: Option<StdlibResult<ClassType>>,
}

impl Stdlib {
    pub fn new(
        version: PythonVersion,
        lookup_class: &dyn Fn(ModuleName, &Name) -> Option<(Class, Arc<TParams>)>,
    ) -> Self {
        Self::new_with_bootstrapping(false, version, lookup_class)
    }

    pub fn new_with_bootstrapping(
        bootstrapping: bool,
        version: PythonVersion,
        lookup_class: &dyn Fn(ModuleName, &Name) -> Option<(Class, Arc<TParams>)>,
    ) -> Self {
        let builtins = ModuleName::builtins();
        let types = ModuleName::types();
        let typing = ModuleName::typing();
        let typing_extensions = ModuleName::typing_extensions();
        let enum_ = ModuleName::enum_();
        let type_checker_internals = ModuleName::type_checker_internals();
        let collections_abc = ModuleName::collections_abc();

        let lookup_generic =
            |module: ModuleName, name: &'static str, args: usize| match lookup_class(
                module,
                &Name::new_static(name),
            ) {
                Some((cls, tparams)) if tparams.len() == args => Ok((cls, tparams)),
                _ => Err(StdlibError {
                    bootstrapping,
                    name,
                }),
            };
        let lookup_concrete = |module: ModuleName, name: &'static str| {
            lookup_generic(module, name, 0).map(|(obj, tparams)| {
                assert!(tparams.is_empty());
                ClassType::new(obj, TArgs::default())
            })
        };

        let none_location = if version.at_least(3, 10) {
            types
        } else {
            ModuleName::from_str("_typeshed")
        };

        let standardised = |major: u32, minor: u32| -> ModuleName {
            if version.at_least(major, minor) {
                typing
            } else {
                typing_extensions
            }
        };

        Self {
            str: lookup_concrete(builtins, "str"),
            bool: lookup_concrete(builtins, "bool"),
            int: lookup_concrete(builtins, "int"),
            bytes: lookup_concrete(builtins, "bytes"),
            bytearray: lookup_concrete(builtins, "bytearray"),
            float: lookup_concrete(builtins, "float"),
            complex: lookup_concrete(builtins, "complex"),
            decimal: lookup_concrete(ModuleName::from_str("decimal"), "Decimal"),
            date: lookup_concrete(ModuleName::from_str("datetime"), "date"),
            datetime: lookup_concrete(ModuleName::from_str("datetime"), "datetime"),
            time: lookup_concrete(ModuleName::from_str("datetime"), "time"),
            timedelta: lookup_concrete(ModuleName::from_str("datetime"), "timedelta"),
            path: lookup_concrete(ModuleName::from_str("pathlib"), "Path"),
            uuid: lookup_concrete(ModuleName::from_str("uuid"), "UUID"),
            slice: lookup_generic(builtins, "slice", 3),
            base_exception: lookup_concrete(builtins, "BaseException"),
            base_exception_group: version
                .at_least(3, 11)
                .then(|| lookup_generic(builtins, "BaseExceptionGroup", 1)),
            exception_group: version
                .at_least(3, 11)
                .then(|| lookup_generic(builtins, "ExceptionGroup", 1)),
            list: lookup_generic(builtins, "list", 1),
            dict: lookup_generic(builtins, "dict", 2),
            deque: lookup_generic(ModuleName::collections(), "deque", 1),
            frozenset: lookup_generic(builtins, "frozenset", 1),
            dict_items: lookup_generic(collections_abc, "dict_items", 2),
            dict_keys: lookup_generic(collections_abc, "dict_keys", 2),
            dict_values: lookup_generic(collections_abc, "dict_values", 2),
            set: lookup_generic(builtins, "set", 1),
            tuple: lookup_generic(builtins, "tuple", 1),
            builtins_type: lookup_concrete(builtins, "type"),
            ellipsis_type: version
                .at_least(3, 10)
                .then(|| lookup_concrete(types, "EllipsisType")),
            none_type: lookup_concrete(none_location, "NoneType"),
            iterable: lookup_generic(typing, "Iterable", 1),
            async_iterable: lookup_generic(typing, "AsyncIterable", 1),
            async_iterator: lookup_generic(typing, "AsyncIterator", 1),
            mutable_sequence: lookup_generic(typing, "MutableSequence", 1),
            generator: lookup_generic(typing, "Generator", 3),
            async_generator: lookup_generic(typing, "AsyncGenerator", 2),
            awaitable: lookup_generic(typing, "Awaitable", 1),
            coroutine: lookup_generic(typing, "Coroutine", 3),
            type_var: lookup_concrete(typing, "TypeVar"),
            param_spec: lookup_concrete(standardised(3, 10), "ParamSpec"),
            param_spec_args: lookup_concrete(standardised(3, 10), "ParamSpecArgs"),
            param_spec_kwargs: lookup_concrete(standardised(3, 10), "ParamSpecKwargs"),
            type_var_tuple: lookup_concrete(standardised(3, 11), "TypeVarTuple"),
            type_alias_type: lookup_concrete(standardised(3, 12), "TypeAliasType"),
            traceback_type: lookup_concrete(types, "TracebackType"),
            function_type: lookup_concrete(types, "FunctionType"),
            method_type: lookup_concrete(types, "MethodType"),
            mapping: lookup_generic(typing, "Mapping", 2),
            enum_meta: lookup_concrete(enum_, "EnumMeta"),
            enum_flag: lookup_concrete(enum_, "Flag"),
            enum_class: lookup_concrete(enum_, "Enum"),
            named_tuple_fallback: lookup_concrete(type_checker_internals, "NamedTupleFallback"),
            typed_dict_fallback: lookup_concrete(type_checker_internals, "TypedDictFallback"),
            property: lookup_concrete(builtins, "property"),
            object: lookup_concrete(builtins, "object"),
            union_type: version
                .at_least(3, 10)
                .then(|| lookup_concrete(types, "UnionType")),
        }
    }

    /// Create a new Stdlib with all types set to `Any``.
    ///
    /// This is needed because bootstrapping a `Stdlib` requires an `AnswersSolver` for the
    /// `lookup`, but `AnswersSolver` itself depends on `Stdlib`.
    ///
    /// It works because the lookups only need a tiny subset of all `AnswersSolver` functionality,
    /// none of which actually depends on `Stdlib`.
    pub fn for_bootstrapping() -> Stdlib {
        Self::new_with_bootstrapping(true, PythonVersion::default(), &|_, _| None)
    }

    fn unwrap<T>(x: &StdlibResult<T>) -> &T {
        match x {
            Ok(x) => x,
            Err(err) => {
                unreachable!(
                    "Stdlib missing class `{}`{}",
                    err.name,
                    if err.bootstrapping {
                        " (while bootstrapping)"
                    } else {
                        ""
                    },
                )
            }
        }
    }

    fn primitive(cls: &StdlibResult<ClassType>) -> &ClassType {
        Self::unwrap(cls)
    }

    pub fn object(&self) -> &ClassType {
        Self::primitive(&self.object)
    }

    pub fn bool(&self) -> &ClassType {
        Self::primitive(&self.bool)
    }

    pub fn builtins_type(&self) -> &ClassType {
        Self::primitive(&self.builtins_type)
    }

    pub fn enum_meta(&self) -> &ClassType {
        Self::primitive(&self.enum_meta)
    }

    pub fn enum_flag(&self) -> &ClassType {
        Self::primitive(&self.enum_flag)
    }

    pub fn enum_class(&self) -> &ClassType {
        Self::primitive(&self.enum_class)
    }

    pub fn named_tuple_fallback(&self) -> &ClassType {
        Self::primitive(&self.named_tuple_fallback)
    }

    pub fn typed_dict_fallback(&self) -> &ClassType {
        Self::primitive(&self.typed_dict_fallback)
    }

    pub fn ellipsis_type(&self) -> Option<&ClassType> {
        Some(Self::primitive(self.ellipsis_type.as_ref()?))
    }

    pub fn none_type(&self) -> &ClassType {
        Self::primitive(&self.none_type)
    }

    pub fn int(&self) -> &ClassType {
        Self::primitive(&self.int)
    }

    pub fn float(&self) -> &ClassType {
        Self::primitive(&self.float)
    }

    pub fn complex(&self) -> &ClassType {
        Self::primitive(&self.complex)
    }

    pub fn decimal(&self) -> &ClassType {
        Self::primitive(&self.decimal)
    }

    pub fn bytes(&self) -> &ClassType {
        Self::primitive(&self.bytes)
    }

    pub fn bytearray(&self) -> &ClassType {
        Self::primitive(&self.bytearray)
    }

    pub fn date(&self) -> &ClassType {
        Self::primitive(&self.date)
    }

    pub fn datetime(&self) -> &ClassType {
        Self::primitive(&self.datetime)
    }

    pub fn time(&self) -> &ClassType {
        Self::primitive(&self.time)
    }

    pub fn timedelta(&self) -> &ClassType {
        Self::primitive(&self.timedelta)
    }

    pub fn path(&self) -> &ClassType {
        Self::primitive(&self.path)
    }

    pub fn uuid(&self) -> &ClassType {
        Self::primitive(&self.uuid)
    }

    pub fn str(&self) -> &ClassType {
        Self::primitive(&self.str)
    }

    pub fn slice_class_object(&self) -> Class {
        Self::unwrap(&self.slice).0.dupe()
    }

    pub fn base_exception(&self) -> &ClassType {
        Self::primitive(&self.base_exception)
    }

    fn apply(
        class_and_tparams: &StdlibResult<(Class, Arc<TParams>)>,
        targs: Vec<Type>,
    ) -> ClassType {
        // Note: this construction will panic if we use `apply` with the wrong arity.
        let (class, tparams) = Self::unwrap(class_and_tparams);
        let targs = TArgs::new(tparams.dupe(), targs);
        ClassType::new(class.dupe(), targs)
    }

    pub fn base_exception_group(&self, x: Type) -> Option<ClassType> {
        Some(Self::apply(self.base_exception_group.as_ref()?, vec![x]))
    }

    pub fn exception_group(&self, x: Type) -> Option<ClassType> {
        Some(Self::apply(self.exception_group.as_ref()?, vec![x]))
    }

    pub fn union_type(&self) -> Option<&ClassType> {
        Some(Self::primitive(self.union_type.as_ref()?))
    }

    pub fn tuple_object(&self) -> &Class {
        &Self::unwrap(&self.tuple).0
    }

    pub fn tuple(&self, x: Type) -> ClassType {
        Self::apply(&self.tuple, vec![x])
    }

    pub fn list(&self, x: Type) -> ClassType {
        Self::apply(&self.list, vec![x])
    }

    pub fn list_object(&self) -> &Class {
        &Self::unwrap(&self.list).0
    }

    pub fn deque(&self, x: Type) -> ClassType {
        Self::apply(&self.deque, vec![x])
    }

    pub fn frozenset(&self, x: Type) -> ClassType {
        Self::apply(&self.frozenset, vec![x])
    }

    pub fn frozenset_object(&self) -> &Class {
        &Self::unwrap(&self.frozenset).0
    }

    pub fn dict(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.dict, vec![key, value])
    }

    pub fn dict_object(&self) -> &Class {
        &Self::unwrap(&self.dict).0
    }

    pub fn dict_items(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.dict_items, vec![key, value])
    }

    pub fn dict_keys(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.dict_keys, vec![key, value])
    }

    pub fn dict_values(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.dict_values, vec![key, value])
    }

    pub fn mapping(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.mapping, vec![key, value])
    }

    pub fn set(&self, x: Type) -> ClassType {
        Self::apply(&self.set, vec![x])
    }

    pub fn set_object(&self) -> &Class {
        &Self::unwrap(&self.set).0
    }

    pub fn iterable(&self, x: Type) -> ClassType {
        Self::apply(&self.iterable, vec![x])
    }

    pub fn async_iterable(&self, x: Type) -> ClassType {
        Self::apply(&self.async_iterable, vec![x])
    }

    pub fn async_iterator(&self, x: Type) -> ClassType {
        Self::apply(&self.async_iterator, vec![x])
    }

    pub fn mutable_sequence(&self, x: Type) -> ClassType {
        Self::apply(&self.mutable_sequence, vec![x])
    }

    pub fn generator(&self, yield_ty: Type, send_ty: Type, return_ty: Type) -> ClassType {
        Self::apply(&self.generator, vec![yield_ty, send_ty, return_ty])
    }

    pub fn async_generator(&self, yield_ty: Type, send_ty: Type) -> ClassType {
        Self::apply(&self.async_generator, vec![yield_ty, send_ty])
    }

    pub fn awaitable(&self, x: Type) -> ClassType {
        Self::apply(&self.awaitable, vec![x])
    }

    pub fn awaitable_object(&self) -> &Class {
        &Self::unwrap(&self.awaitable).0
    }

    pub fn coroutine(&self, yield_ty: Type, send_ty: Type, return_ty: Type) -> ClassType {
        Self::apply(&self.coroutine, vec![yield_ty, send_ty, return_ty])
    }

    pub fn coroutine_object(&self) -> &Class {
        &Self::unwrap(&self.coroutine).0
    }

    pub fn type_var(&self) -> &ClassType {
        Self::primitive(&self.type_var)
    }

    pub fn param_spec(&self) -> &ClassType {
        Self::primitive(&self.param_spec)
    }

    pub fn type_var_tuple(&self) -> &ClassType {
        Self::primitive(&self.type_var_tuple)
    }

    pub fn param_spec_args(&self) -> &ClassType {
        Self::primitive(&self.param_spec_args)
    }

    pub fn param_spec_kwargs(&self) -> &ClassType {
        Self::primitive(&self.param_spec_kwargs)
    }

    pub fn param_spec_args_as_tuple(&self) -> ClassType {
        self.tuple(self.object().clone().to_type())
    }

    pub fn param_spec_kwargs_as_dict(&self) -> ClassType {
        self.dict(
            self.str().clone().to_type(),
            self.object().clone().to_type(),
        )
    }

    pub fn type_alias_type(&self) -> &ClassType {
        Self::primitive(&self.type_alias_type)
    }

    pub fn traceback_type(&self) -> &ClassType {
        Self::primitive(&self.traceback_type)
    }

    pub fn function_type(&self) -> &ClassType {
        Self::primitive(&self.function_type)
    }

    pub fn method_type(&self) -> &ClassType {
        Self::primitive(&self.method_type)
    }

    pub fn property(&self) -> &ClassType {
        Self::primitive(&self.property)
    }
}
