/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Function;
use pyrefly_util::display::count;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::solver::solver::QuantifiedHandle;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::tuple::Tuple;
use crate::types::type_var::PreInferenceVariance;
use crate::types::typed_dict::TypedDict;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::TArgs;
use crate::types::types::TParam;
use crate::types::types::TParams;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Silently promotes a Class to a ClassType, using default type arguments. It is up to the
    /// caller to ensure they are not calling this method on a TypedDict class, which should be
    /// promoted to TypedDict instead of ClassType.
    pub fn promote_nontypeddict_silently_to_classtype(&self, cls: &Class) -> ClassType {
        ClassType::new(
            cls.dupe(),
            self.create_default_targs(self.get_class_tparams(cls), None),
        )
    }

    fn specialize_impl(
        &self,
        cls: &Class,
        targs: Vec<Type>,
        range: TextRange,
        validate_restriction: bool,
        errors: &ErrorCollector,
    ) -> Type {
        let metadata = self.get_metadata_for_class(cls);
        let tparams = self.get_class_tparams(cls);

        // We didn't find any type parameters for this class, but it may have ones we don't know about if:
        // - the class inherits from Any, or
        // - the class inherits from Generic[...] or Protocol [...]. We probably dropped the type
        //   arguments because we found an error in them.
        let has_unknown_tparams =
            tparams.is_empty() && (metadata.has_base_any() || metadata.has_generic_base_class());

        let targs = if !targs.is_empty() && has_unknown_tparams {
            // Accept any number of arguments (by ignoring them).
            TArgs::default()
        } else {
            self.create_targs(
                cls.name(),
                tparams,
                targs,
                range,
                validate_restriction,
                errors,
            )
        };
        self.type_of_instance(cls, targs)
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
        self.specialize_impl(cls, targs, range, true, errors)
    }

    pub fn specialize_in_base_class(
        &self,
        cls: &Class,
        targs: Vec<Type>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        self.specialize_impl(cls, targs, range, false, errors)
    }

    fn specialize_forall_impl(
        &self,
        forall: Forall<Forallable>,
        targs: Vec<Type>,
        range: TextRange,
        validate_restriction: bool,
        errors: &ErrorCollector,
    ) -> Type {
        let targs = self.create_targs(
            &forall.body.name(),
            forall.tparams.dupe(),
            targs,
            range,
            validate_restriction,
            errors,
        );
        forall.apply_targs(targs)
    }

    pub fn specialize_forall(
        &self,
        forall: Forall<Forallable>,
        targs: Vec<Type>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        self.specialize_forall_impl(forall, targs, range, true, errors)
    }

    pub fn specialize_forall_in_base_class(
        &self,
        forall: Forall<Forallable>,
        targs: Vec<Type>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        self.specialize_forall_impl(forall, targs, range, false, errors)
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
    pub fn promote(&self, cls: &Class, range: TextRange, errors: &ErrorCollector) -> Type {
        let targs = self.create_default_targs(
            self.get_class_tparams(cls),
            Some(&|tparam: &TParam| {
                errors.add(
                    range,
                    ErrorInfo::Kind(ErrorKind::ImplicitAny),
                    vec1![
                        format!(
                            "Cannot determine the type parameter `{}` for generic class `{}`",
                            tparam.name(),
                            cls.name(),
                        ),
                        "Either specify the type argument explicitly, or specify a default for the type variable.".to_owned(),
                    ],
                );
            }),
        );
        self.type_of_instance(cls, targs)
    }

    pub fn promote_forall(&self, forall: Forall<Forallable>, _range: TextRange) -> Type {
        // TODO(grievejia): We probably want to error here as well
        let targs = self.create_default_targs(forall.tparams.dupe(), None);
        forall.apply_targs(targs)
    }

    /// Version of `promote` that does not potentially raise errors.
    /// Should only be used for unusual scenarios.
    pub fn promote_silently(&self, cls: &Class) -> Type {
        let targs = self.create_default_targs(self.get_class_tparams(cls), None);
        self.type_of_instance(cls, targs)
    }

    fn targs_of_tparams(&self, class: &Class) -> TArgs {
        let tparams = self.get_class_tparams(class);
        TArgs::new(
            tparams.dupe(),
            tparams.quantifieds().map(|q| q.clone().to_type()).collect(),
        )
    }

    /// Given a class or typed dictionary, create a `Type` that represents a generic instance of
    /// the class or typed dictionary.
    ///
    /// Note how this differs from `specialize` and `promote`:
    /// specialize(list, [int]) == list[int]
    /// promote(list) == list[Any]
    /// instantiate(list) == list[T]
    pub fn instantiate(&self, cls: &Class) -> Type {
        self.type_of_instance(cls, self.targs_of_tparams(cls))
    }

    /// Gets this Class as a ClassType with its tparams as the arguments. For non-TypedDict
    /// classes, this is the type of an instance of this class. Unless you specifically need the
    /// ClassType inside the Type and know you don't have a TypedDict, you should instead use
    /// AnswersSolver::instantiate() to get an instance type.
    pub fn as_class_type_unchecked(&self, class: &Class) -> ClassType {
        ClassType::new(class.dupe(), self.targs_of_tparams(class))
    }

    /// Gets this Class as a TypedDict with its tparams as the arguments.
    pub fn as_typed_dict_unchecked(&self, class: &Class) -> TypedDict {
        let targs = self.targs_of_tparams(class);
        TypedDict::new(class.clone(), targs)
    }

    /// Instantiates a class or typed dictionary with fresh variables for its type parameters.
    pub fn instantiate_fresh_class(&self, cls: &Class) -> Type {
        self.solver()
            .fresh_quantified(
                &self.get_class_tparams(cls),
                self.instantiate(cls),
                self.uniques,
            )
            .1
    }

    pub fn instantiate_fresh_tuple(&self) -> Type {
        let quantified = Quantified::type_var_tuple(Name::new_static("Ts"), self.uniques, None);
        let tparams = TParams::new(vec![TParam {
            quantified: quantified.clone(),
            variance: PreInferenceVariance::Covariant,
        }]);
        let tuple_ty = Type::Tuple(Tuple::Unpacked(Box::new((
            Vec::new(),
            Type::Quantified(Box::new(quantified)),
            Vec::new(),
        ))));
        self.solver()
            .fresh_quantified(&tparams, tuple_ty, self.uniques)
            .1
    }

    pub fn instantiate_fresh_forall(&self, forall: Forall<Forallable>) -> (QuantifiedHandle, Type) {
        self.solver()
            .fresh_quantified(&forall.tparams, forall.body.as_type(), self.uniques)
    }

    pub fn instantiate_fresh_function(
        &self,
        tparams: &TParams,
        func: Function,
    ) -> (QuantifiedHandle, Function) {
        let (qs, t) =
            self.solver()
                .fresh_quantified(tparams, Type::Function(Box::new(func)), self.uniques);
        match t {
            Type::Function(func) => (qs, *func),
            // We passed a Function to fresh_quantified(), so we know we get a Function back out.
            _ => unreachable!(),
        }
    }

    pub fn instantiate_fresh_callable(
        &self,
        tparams: &TParams,
        c: Callable,
    ) -> (QuantifiedHandle, Callable) {
        let (qs, t) =
            self.solver()
                .fresh_quantified(tparams, Type::Callable(Box::new(c)), self.uniques);
        match t {
            Type::Callable(c) => (qs, *c),
            // We passed a Function to fresh_quantified(), so we know we get a Function back out.
            _ => unreachable!(),
        }
    }

    /// Creates default type arguments for a class, falling back to Any for type parameters without defaults.
    fn create_default_targs(
        &self,
        tparams: Arc<TParams>,
        on_fallback_to_gradual: Option<&dyn Fn(&TParam)>,
    ) -> TArgs {
        if tparams.is_empty() {
            TArgs::default()
        } else {
            let tys = tparams
                .iter()
                .map(|x| {
                    // TODO(grievejia): This is actually not a 100% accurate way of detecting graudal fallbacks:
                    // it will trigger when the tparam doesn't have a default, but won't trigger if
                    // - The tparam has a default, but that default is another type var without a default.
                    // - The tparam has a default, but part of that default type requires another fallback (e.g. the
                    //   default is `list[Foo]`, where `Foo` is a generic class whose tparam doesn't have a default).
                    //
                    // To make it 100% accurate, we actually need to hook the callback into `as_graudal_type()`. It's doable
                    // but could add a lot of complexities so let's keep it as an exercise in the future.
                    if let Some(f) = on_fallback_to_gradual
                        && x.quantified.default().is_none()
                    {
                        f(x);
                    }
                    x.quantified.as_gradual_type()
                })
                .collect();
            TArgs::new(tparams, tys)
        }
    }

    fn type_of_instance(&self, cls: &Class, targs: TArgs) -> Type {
        let metadata = self.get_metadata_for_class(cls);
        if metadata.is_typed_dict() {
            Type::TypedDict(TypedDict::new(cls.dupe(), targs))
        } else {
            Type::ClassType(ClassType::new(cls.dupe(), targs))
        }
    }

    fn create_targs(
        &self,
        name: &Name,
        tparams: Arc<TParams>,
        targs: Vec<Type>,
        range: TextRange,
        validate_restriction: bool,
        errors: &ErrorCollector,
    ) -> TArgs {
        let nparams = tparams.len();
        let nargs = targs.len();
        let mut checked_targs = Vec::new();
        let mut targ_idx = 0;
        let mut name_to_idx = SmallMap::new();
        for (param_idx, param) in tparams.iter().enumerate() {
            if let Some(arg) = targs.get(targ_idx) {
                // Get next type argument
                match param.quantified.kind() {
                    QuantifiedKind::TypeVarTuple => {
                        // We know that ParamSpec params must be matched by ParamSpec args, so chop off both params and args
                        // at the next ParamSpec when computing how many args the TypeVarTuple should consume.
                        let paramspec_param_idx =
                            self.peek_next_paramspec_param(param_idx + 1, &tparams);
                        let paramspec_arg_idx = self.peek_next_paramspec_arg(targ_idx, &targs);
                        let nparams_for_tvt = paramspec_param_idx.unwrap_or(nparams);
                        let nargs_for_tvt = paramspec_arg_idx.unwrap_or(nargs);
                        let args_to_consume = self.num_typevartuple_args_to_consume(
                            param_idx,
                            nparams_for_tvt,
                            targ_idx,
                            nargs_for_tvt,
                        );
                        let new_targ_idx = targ_idx + args_to_consume;
                        checked_targs.push(self.create_next_typevartuple_arg(
                            &targs[targ_idx..new_targ_idx],
                            range,
                            errors,
                        ));
                        targ_idx = new_targ_idx;
                    }
                    QuantifiedKind::ParamSpec if nparams == 1 && !arg.is_kind_param_spec() => {
                        // If the only type param is a ParamSpec and the type argument
                        // is not a parameter expression, then treat the entire type argument list
                        // as a parameter list
                        checked_targs.push(self.create_paramspec_value(&targs));
                        targ_idx = nargs;
                    }
                    QuantifiedKind::ParamSpec => {
                        checked_targs.push(self.create_next_paramspec_arg(arg, range, errors));
                        targ_idx += 1;
                    }
                    QuantifiedKind::TypeVar => {
                        checked_targs.push(self.create_next_typevar_arg(
                            param,
                            arg,
                            range,
                            validate_restriction,
                            errors,
                        ));
                        targ_idx += 1;
                    }
                }
            } else {
                // We've run out of arguments, and we have type parameters left to consume.
                checked_targs.extend(self.consume_remaining_tparams(
                    name,
                    &tparams,
                    param_idx,
                    &checked_targs,
                    nargs,
                    &name_to_idx,
                    range,
                    errors,
                ));
                break;
            }
            name_to_idx.insert(param.name(), param_idx);
        }
        if targ_idx < nargs {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "Expected {} for `{}`, got {}",
                    count(nparams, "type argument"),
                    name,
                    nargs
                ),
            );
        }
        drop(name_to_idx);
        TArgs::new(tparams, checked_targs)
    }

    fn peek_next_paramspec_param(&self, start_idx: usize, tparams: &TParams) -> Option<usize> {
        for (i, param) in tparams.iter().enumerate().skip(start_idx) {
            if param.quantified.is_param_spec() {
                return Some(i);
            }
        }
        None
    }

    fn peek_next_paramspec_arg(&self, start_idx: usize, args: &[Type]) -> Option<usize> {
        for (i, arg) in args.iter().enumerate().skip(start_idx) {
            if arg.is_kind_param_spec() {
                return Some(i);
            }
        }
        None
    }

    fn num_typevartuple_args_to_consume(
        &self,
        param_idx: usize,
        nparams: usize,
        targ_idx: usize,
        nargs: usize,
    ) -> usize {
        let n_remaining_params = nparams - param_idx - 1;
        let n_remaining_args = nargs - targ_idx;
        n_remaining_args.saturating_sub(n_remaining_params)
    }

    fn create_next_typevartuple_arg(
        &self,
        args: &[Type],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let mut prefix = Vec::new();
        let mut middle = Vec::new();
        let mut suffix = Vec::new();
        for arg in args {
            match arg {
                Type::Unpack(box Type::Tuple(Tuple::Concrete(elts))) => {
                    if middle.is_empty() {
                        prefix.extend_from_slice(elts);
                    } else {
                        suffix.extend_from_slice(elts);
                    }
                }
                Type::Unpack(t) => {
                    if !suffix.is_empty() {
                        middle.push(Type::unbounded_tuple(self.unions(suffix)));
                        suffix = Vec::new();
                    } else {
                        middle.push((**t).clone())
                    }
                }
                arg => {
                    let arg = if arg.is_kind_type_var_tuple() {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                            "`TypeVarTuple` must be unpacked".to_owned(),
                        )
                    } else {
                        arg.clone()
                    };
                    if middle.is_empty() {
                        prefix.push(arg);
                    } else {
                        suffix.push(arg);
                    }
                }
            }
        }
        match middle.as_slice() {
            [] => Type::concrete_tuple(prefix),
            [middle] => Type::unpacked_tuple(prefix, middle.clone(), suffix),
            // We can't precisely model unpacking two unbounded iterables, so we'll keep any
            // concrete prefix and suffix elements and merge everything in between into an unbounded tuple
            _ => {
                let middle_types: Vec<Type> = middle
                    .iter()
                    .map(|t| {
                        self.unwrap_iterable(t)
                            .unwrap_or(self.stdlib.object().clone().to_type())
                    })
                    .collect();
                Type::unpacked_tuple(
                    prefix,
                    Type::unbounded_tuple(self.unions(middle_types)),
                    suffix,
                )
            }
        }
    }

    fn create_paramspec_value(&self, targs: &[Type]) -> Type {
        let params: Vec<Param> = targs.map(|t| Param::PosOnly(None, t.clone(), Required::Required));
        Type::ParamSpecValue(ParamList::new(params))
    }

    fn create_next_paramspec_arg(
        &self,
        arg: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if arg.is_kind_param_spec() {
            arg.clone()
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                format!(
                    "Expected a valid ParamSpec expression, got `{}`",
                    self.for_display(arg.clone())
                ),
            );
            Type::Ellipsis
        }
    }

    fn create_next_typevar_arg(
        &self,
        param: &TParam,
        arg: &Type,
        range: TextRange,
        validate_restriction: bool,
        errors: &ErrorCollector,
    ) -> Type {
        match arg {
            Type::Unpack(_) => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadUnpacking),
                format!(
                    "Unpacked argument cannot be used for type parameter {}",
                    param.name()
                ),
            ),
            _ => {
                if arg.is_kind_type_var_tuple() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                        "`TypeVarTuple` must be unpacked".to_owned(),
                    )
                } else if arg.is_kind_param_spec() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                        "`ParamSpec` cannot be used for type parameter".to_owned(),
                    )
                } else {
                    let restriction = param.restriction();
                    if validate_restriction && restriction.is_restricted() {
                        let tcc = &|| {
                            TypeCheckContext::of_kind(TypeCheckKind::TypeVarSpecialization(
                                param.name().clone(),
                            ))
                        };
                        // In a legacy type alias, one old-style TypeVar can be specialized with
                        // another, which we handle by checking their upper bounds against each other.
                        let arg_for_check = {
                            let arg = arg.clone();
                            arg.transform(&mut |x| {
                                if let Type::TypeVar(tv) = x {
                                    *x = tv.restriction().as_type(self.stdlib);
                                }
                            })
                        };
                        self.check_type(
                            &arg_for_check,
                            &restriction.as_type(self.stdlib),
                            range,
                            errors,
                            tcc,
                        );
                    }
                    arg.clone()
                }
            }
        }
    }

    fn get_tparam_default(
        &self,
        param: &TParam,
        checked_targs: &[Type],
        name_to_idx: &SmallMap<&Name, usize>,
    ) -> Type {
        if let Some(default) = param.default() {
            default.clone().transform(&mut |default| {
                let typevar_name = match default {
                    Type::TypeVar(t) => Some(t.qname().id()),
                    Type::TypeVarTuple(t) => Some(t.qname().id()),
                    Type::ParamSpec(p) => Some(p.qname().id()),
                    Type::Quantified(q) => Some(q.name()),
                    _ => None,
                };
                if let Some(typevar_name) = typevar_name {
                    *default = if let Some(i) = name_to_idx.get(typevar_name) {
                        // The default of this TypeVar contains the value of a previous TypeVar.
                        checked_targs[*i].clone()
                    } else {
                        // The default refers to the value of a TypeVar that isn't in scope. We've
                        // already logged an error in TParams::new(); return a sensible default.
                        Type::any_implicit()
                    }
                }
            })
        } else {
            param.quantified.as_gradual_type()
        }
    }

    /// Consume all remaining type parameters after we've run out of arguments.
    fn consume_remaining_tparams(
        &self,
        name: &Name,
        tparams: &TParams,
        param_idx: usize,
        checked_targs: &[Type],
        nargs: usize,
        name_to_idx: &SmallMap<&Name, usize>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Vec<Type> {
        let all_remaining_params_can_be_empty = tparams
            .iter()
            .skip(param_idx)
            .all(|x| x.quantified.is_type_var_tuple() || x.default().is_some());
        if !all_remaining_params_can_be_empty {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!(
                    "Expected {} for `{}`, got {}",
                    count(tparams.len(), "type argument"),
                    name,
                    nargs,
                ),
            );
        }
        tparams
            .iter()
            .skip(param_idx)
            .map(|x| self.get_tparam_default(x, checked_targs, name_to_idx))
            .collect()
    }
}
