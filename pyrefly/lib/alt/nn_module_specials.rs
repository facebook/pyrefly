/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Special handling for PyTorch nn.Module types.
//!
//! This module contains special-case logic for nn.Module and related types:
//!
//! - **nn.Module call forwarding**: When calling an nn.Module instance, redirect to its `forward` method.
//!   This models PyTorch's `nn.Module.__call__` behavior.
//!
//! - **nn.ModuleDict[TypedDict]**: When ModuleDict is parameterized with a TypedDict,
//!   provide precise types for indexing and attribute access instead of generic `Module`.

use pyrefly_python::dunder;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::class::class_field::ClassAttribute;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::types::class::ClassType;

pub fn is_nn_module_dict(cls: &ClassType) -> bool {
    cls.class_object()
        .has_toplevel_qname("torch.nn", "ModuleDict")
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Check if the callee type is an nn.Module instance (ClassType or SelfType),
    /// and if so, return the type of its `forward` method for dispatch.
    ///
    /// This models PyTorch's `nn.Module.__call__` behavior, which delegates to `forward`.
    /// Returns `Some(forward_type)` if the callee is an nn.Module instance,
    /// `None` otherwise (to proceed with normal call handling).
    ///
    /// # Example
    /// ```python
    /// class MyModel(nn.Module):
    ///     def forward(self, x: Tensor) -> Tensor: ...
    ///
    /// model = MyModel()
    /// model(input)  # Calls model.forward(input)
    /// ```
    pub fn try_nn_module_forward_dispatch(
        &self,
        callee_ty: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        // Extract the class from ClassType or SelfType
        let class = match callee_ty {
            Type::ClassType(cls) => cls.class_object(),
            Type::SelfType(cls) => cls.class_object(),
            _ => return None,
        };

        // Check if this class inherits from torch.nn.Module
        let is_nn_module = class.has_toplevel_qname("torch.nn", "Module")
            || self
                .get_mro_for_class(class)
                .ancestors_no_object()
                .iter()
                .any(|ancestor| {
                    ancestor
                        .class_object()
                        .has_toplevel_qname("torch.nn", "Module")
                });

        if is_nn_module {
            let forward_name = Name::new("forward");
            Some(self.attr_infer_for_type(callee_ty, &forward_name, range, errors, None))
        } else {
            None
        }
    }

    /// Handle attribute access on `nn.ModuleDict[T]` where `T` is a TypedDict.
    ///
    /// PyTorch's `nn.ModuleDict` supports attribute-style access to its modules.
    /// When parameterized with a TypedDict type argument (e.g., `nn.ModuleDict[MyModules]`),
    /// we can provide precise types for attribute access.
    ///
    /// # Example
    /// ```python
    /// class MyModules(TypedDict):
    ///     encoder: nn.Linear[64, 128]
    ///     decoder: nn.Linear[128, 64]
    ///
    /// modules: nn.ModuleDict[MyModules] = ...
    /// modules.encoder  # Returns nn.Linear[64, 128], not just Module
    /// ```
    ///
    /// Returns `Some(ClassAttribute)` if the attribute is found in the TypedDict,
    /// `None` to fall back to normal attribute lookup.
    pub fn try_nn_module_dict_attr(
        &self,
        class: &ClassType,
        attr_name: &Name,
    ) -> Option<ClassAttribute> {
        if !is_nn_module_dict(class) {
            return None;
        }

        let first_targ = class.targs().as_slice().first()?;
        let Type::TypedDict(pyrefly_types::typed_dict::TypedDict::TypedDict(typed_dict_inner)) =
            first_targ
        else {
            return None;
        };

        // Check if the attr_name is a field in the TypedDict
        let has_field = self
            .get_metadata_for_class(typed_dict_inner.class_object())
            .typed_dict_metadata()
            .is_some_and(|metadata| metadata.fields.contains_key(attr_name));

        if !has_field {
            return None;
        }

        let field =
            self.get_field_from_current_class_only(typed_dict_inner.class_object(), attr_name)?;
        let field_ty = field.ty();
        let instantiated_ty = typed_dict_inner
            .targs()
            .substitution()
            .substitute_into(field_ty);

        Some(ClassAttribute::read_write(instantiated_ty))
    }

    /// Handle indexing on `nn.ModuleDict[T]` where `T` is a TypedDict.
    ///
    /// PyTorch's `nn.ModuleDict` is a dictionary-like container for modules. When parameterized
    /// with a TypedDict type argument (e.g., `nn.ModuleDict[MyModules]`), we can provide precise
    /// types for string literal key access.
    ///
    /// # Example
    /// ```python
    /// class MyModules(TypedDict):
    ///     encoder: nn.Linear[64, 128]
    ///     decoder: nn.Linear[128, 64]
    ///
    /// modules: nn.ModuleDict[MyModules] = ...
    /// modules["encoder"]  # Returns nn.Linear[64, 128], not just Module
    /// ```
    ///
    /// Falls back to `__getitem__` for non-literal keys or when no TypedDict type argument.
    pub fn try_nn_module_dict_index(
        &self,
        cls: &ClassType,
        base: &Type,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        // Check if the first type argument is a TypedDict
        if let Some(Type::TypedDict(pyrefly_types::typed_dict::TypedDict::TypedDict(
            typed_dict_inner,
        ))) = cls.targs().as_slice().first()
        {
            // Check if the slice is a string literal
            let key_ty = self.expr_infer(slice, errors);
            if let Type::Literal(box pyrefly_types::literal::Literal {
                value: Lit::Str(field_name),
                ..
            }) = &key_ty
            {
                // Look up the field in the TypedDict
                if let Some(metadata) = self
                    .get_metadata_for_class(typed_dict_inner.class_object())
                    .typed_dict_metadata()
                    && metadata.fields.contains_key(&Name::new(field_name))
                {
                    // Get the field type from the TypedDict class
                    if let Some(field) = self.get_field_from_current_class_only(
                        typed_dict_inner.class_object(),
                        &Name::new(field_name),
                    ) {
                        let field_ty = field.ty();
                        // Substitute type parameters if needed
                        return typed_dict_inner
                            .targs()
                            .substitution()
                            .substitute_into(field_ty);
                    }
                } else {
                    // Key not in TypedDict, report error
                    return self.error(
                        errors,
                        slice.range(),
                        ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                        format!(
                            "ModuleDict key `{}` not found in TypedDict `{}`",
                            field_name,
                            typed_dict_inner.name()
                        ),
                    );
                }
            }
        }
        // Fall back to calling __getitem__
        self.call_method_or_error(
            base,
            &dunder::GETITEM,
            range,
            &[CallArg::expr(slice)],
            &[],
            errors,
            Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
        )
    }
}
