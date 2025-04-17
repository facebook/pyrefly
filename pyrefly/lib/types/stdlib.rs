/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use ruff_python_ast::name::Name;

use crate::PythonVersion;
use crate::module::module_name::ModuleName;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::class::TArgs;
use crate::types::types::Type;

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
    float: StdlibResult<ClassType>,
    complex: StdlibResult<ClassType>,
    slice: StdlibResult<Class>,
    base_exception: StdlibResult<ClassType>,
    /// Introduced in Python 3.11.
    base_exception_group: Option<StdlibResult<Class>>,
    /// Introduced in Python 3.11.
    exception_group: Option<StdlibResult<Class>>,
    list: StdlibResult<Class>,
    dict: StdlibResult<Class>,
    mapping: StdlibResult<Class>,
    set: StdlibResult<Class>,
    tuple: StdlibResult<Class>,
    iterable: StdlibResult<Class>,
    async_iterable: StdlibResult<Class>,
    generator: StdlibResult<Class>,
    async_generator: StdlibResult<Class>,
    awaitable: StdlibResult<Class>,
    coroutine: StdlibResult<Class>,
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
    named_tuple: StdlibResult<ClassType>,
    property: StdlibResult<ClassType>,
    object: StdlibResult<ClassType>,
}

impl Stdlib {
    pub fn new(
        version: PythonVersion,
        lookup_class: &dyn Fn(ModuleName, &Name) -> Option<Class>,
    ) -> Self {
        Self::new_with_bootstrapping(false, version, lookup_class)
    }

    pub fn new_with_bootstrapping(
        bootstrapping: bool,
        version: PythonVersion,
        lookup_class: &dyn Fn(ModuleName, &Name) -> Option<Class>,
    ) -> Self {
        let builtins = ModuleName::builtins();
        let types = ModuleName::types();
        let typing = ModuleName::typing();
        let typing_extensions = ModuleName::typing_extensions();
        let enum_ = ModuleName::enum_();

        let lookup_generic =
            |module: ModuleName, name: &'static str, args: usize| match lookup_class(
                module,
                &Name::new_static(name),
            ) {
                Some(cls) if cls.tparams().len() == args => Ok(cls),
                _ => Err(StdlibError {
                    bootstrapping,
                    name,
                }),
            };
        let lookup_concrete = |module: ModuleName, name: &'static str| {
            lookup_generic(module, name, 0).map(|obj| ClassType::new(obj, TArgs::default()))
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
            float: lookup_concrete(builtins, "float"),
            complex: lookup_concrete(builtins, "complex"),
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
            set: lookup_generic(builtins, "set", 1),
            tuple: lookup_generic(builtins, "tuple", 1),
            builtins_type: lookup_concrete(builtins, "type"),
            ellipsis_type: version
                .at_least(3, 10)
                .then(|| lookup_concrete(types, "EllipsisType")),
            none_type: lookup_concrete(none_location, "NoneType"),
            iterable: lookup_generic(typing, "Iterable", 1),
            async_iterable: lookup_generic(typing, "AsyncIterable", 1),
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
            named_tuple: lookup_concrete(typing, "NamedTuple"),
            property: lookup_concrete(builtins, "property"),
            object: lookup_concrete(builtins, "object"),
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

    pub fn named_tuple(&self) -> &ClassType {
        Self::primitive(&self.named_tuple)
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

    pub fn bytes(&self) -> &ClassType {
        Self::primitive(&self.bytes)
    }

    pub fn str(&self) -> &ClassType {
        Self::primitive(&self.str)
    }

    pub fn slice(&self, targs: Vec<Type>) -> ClassType {
        Self::apply(&self.slice, targs)
    }

    pub fn slice_class_object(&self) -> Class {
        Self::unwrap(&self.slice).dupe()
    }

    pub fn base_exception(&self) -> &ClassType {
        Self::primitive(&self.base_exception)
    }

    fn apply(cls: &StdlibResult<Class>, targs: Vec<Type>) -> ClassType {
        // Note: this construction will panic if we use `apply` with the wrong arity.
        ClassType::new(Self::unwrap(cls).dupe(), TArgs::new(targs))
    }

    pub fn base_exception_group(&self, x: Type) -> Option<ClassType> {
        Some(Self::apply(self.base_exception_group.as_ref()?, vec![x]))
    }

    pub fn exception_group(&self, x: Type) -> Option<ClassType> {
        Some(Self::apply(self.exception_group.as_ref()?, vec![x]))
    }

    pub fn tuple(&self, x: Type) -> ClassType {
        Self::apply(&self.tuple, vec![x])
    }

    pub fn list(&self, x: Type) -> ClassType {
        Self::apply(&self.list, vec![x])
    }

    pub fn dict(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.dict, vec![key, value])
    }

    pub fn mapping(&self, key: Type, value: Type) -> ClassType {
        Self::apply(&self.mapping, vec![key, value])
    }

    pub fn set(&self, x: Type) -> ClassType {
        Self::apply(&self.set, vec![x])
    }

    pub fn iterable(&self, x: Type) -> ClassType {
        Self::apply(&self.iterable, vec![x])
    }

    pub fn async_iterable(&self, x: Type) -> ClassType {
        Self::apply(&self.async_iterable, vec![x])
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

    pub fn coroutine(&self, yield_ty: Type, send_ty: Type, return_ty: Type) -> ClassType {
        Self::apply(&self.coroutine, vec![yield_ty, send_ty, return_ty])
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
