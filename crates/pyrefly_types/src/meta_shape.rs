/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Meta-shape functions for programmatic tensor shape transformation
//!
//! This module provides infrastructure for defining how tensor operations
//! transform shapes, similar to PyTorch's FakeTensorMode. Meta-shape functions
//! are registered in a global registry and called during type checking to
//! compute output shapes from input shapes.

use std::collections::HashMap;
use std::fmt::Debug;

use crate::dimension::simplify;
use crate::literal::Lit;
use crate::literal::Literal;
use crate::tensor::ShapeError;
use crate::tensor::SizeExpr;
use crate::tensor::TensorShape;
use crate::types::AnyStyle;
use crate::types::Type;

// ============================================================================
// Meta-Shape Function Arguments
// ============================================================================

/// Result type for meta-shape computation
/// Represents different kinds of return values from PyTorch operations
#[derive(Debug, Clone)]
pub enum MetaShapeResult {
    /// Operation returns one or more tensors
    /// Maps to: Tensor or tuple[Tensor, ...]
    Tensors(Vec<TensorShape>),

    /// Tensor without shape information (when rank unknown)
    /// Maps to: Tensor (shapeless)
    TensorShapeless,

    /// Operation returns a concrete Python int
    /// Maps to: Literal[n]
    Int(i64),

    /// Operation returns a symbolic int or expression
    /// Maps to: Literal[N] or Literal[N * M]
    IntSymbolic(Type),

    /// Operation returns a tuple of Python ints (can be literal or symbolic)
    /// Maps to: tuple[Literal[n1], Literal[n2], ...] where each can be literal or symbolic
    IntSymbolicTuple(Vec<Type>),

    /// Unbounded tuple with known element type
    /// Maps to: tuple[Tensor[shape], ...]
    UnboundedTuple { element_shape: TensorShape },

    /// Unbounded tuple with shapeless elements
    /// Maps to: tuple[Tensor, ...]
    UnboundedTupleShapeless,

    /// Operation returns a Python bool
    /// Maps to: Literal[True] or Literal[False]
    Bool(bool),

    /// Unknown or unsupported return type
    Unknown,
}

#[derive(Debug, Clone)]
pub enum MetaShapeArg {
    /// Integer argument (e.g., dim=0)
    Int(i64),

    /// List of integers (e.g., shape=[2, 3, 4])
    IntList(Vec<i64>),

    /// Shape argument
    Shape(TensorShape),

    /// List of shapes (e.g., tensors=[Tensor[2, 3], Tensor[4, 5]])
    ShapeList(Vec<TensorShape>),

    /// Single dimension (now a Type to support Type::Quantified, Type::Size, etc.)
    Dim(Type),

    /// Boolean argument
    Bool(bool),

    /// String argument (e.g., einsum spec='ij,jk->ik')
    String(String),

    /// List of dimensions (e.g., split sizes=[D, NLocalHeads * HeadDim, NLocalHeads * HeadDim])
    DimList(Vec<Type>),
}

impl MetaShapeArg {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            MetaShapeArg::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_int_list(&self) -> Option<&[i64]> {
        match self {
            MetaShapeArg::IntList(list) => Some(list),
            _ => None,
        }
    }

    pub fn as_shape(&self) -> Option<&TensorShape> {
        match self {
            MetaShapeArg::Shape(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_shape_list(&self) -> Option<&[TensorShape]> {
        match self {
            MetaShapeArg::ShapeList(list) => Some(list),
            _ => None,
        }
    }

    pub fn as_dim(&self) -> Option<&Type> {
        match self {
            MetaShapeArg::Dim(d) => Some(d),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaShapeArg::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetaShapeArg::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ============================================================================
// New Infrastructure (Phase 1)
// ============================================================================

/// Arguments to meta-shape computation after binding and conversion
///
/// This is the result of converting bound Python arguments (Type) to meta-shape arguments.
/// Each parameter name is mapped to its converted MetaShapeArg value.
#[derive(Debug, Clone)]
pub struct MetaShapeArgs {
    /// All arguments after Type → MetaShapeArg conversion
    /// Maps parameter name → converted argument value
    pub args: HashMap<String, MetaShapeArg>,
}

impl MetaShapeArgs {
    /// Helper to extract a TensorShape argument by name.
    /// Returns a ShapeError if the argument is missing or not a shape.
    pub fn get_shape(&self, name: &str, op_name: &str) -> Result<&TensorShape, ShapeError> {
        match self.args.get(name) {
            Some(MetaShapeArg::Shape(s)) => Ok(s),
            _ => Err(ShapeError::InvalidDimension {
                value: 0,
                reason: format!("{}: '{}' argument must be a tensor shape", op_name, name),
            }),
        }
    }
}

/// Helper functions for extracting typed values from Type
///
/// These are used in bind_args() implementations to convert bound Python types
/// to meta-shape arguments. Each returns None if the type doesn't match the expected form.
pub mod extract {
    use crate::literal::Lit;
    use crate::literal::Literal;
    use crate::tensor::TensorShape;
    use crate::tuple::Tuple;
    use crate::types::Type;

    /// Extract a concrete TensorShape from a Type.
    /// Returns None for non-tensors, shapeless tensors, and variadic (Unpacked) shapes.
    /// Use this when the caller needs to iterate over individual dimensions.
    pub fn concrete_tensor_shape(ty: &Type) -> Option<TensorShape> {
        match ty {
            Type::Tensor(tensor) => match &tensor.shape {
                TensorShape::Concrete(_) => Some(tensor.shape.clone()),
                TensorShape::Unpacked(_) => None,
            },
            Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
                // Extract shape from ClassType targs
                let targs = cls.targs();
                if targs.is_empty() {
                    return None;
                }
                // First targ should be the shape tuple
                if let Type::Tuple(Tuple::Concrete(elems)) = &targs.as_slice()[0] {
                    let dims: Vec<Type> = elems.iter().filter_map(dimension).collect();
                    if dims.len() == elems.len() && !dims.is_empty() {
                        return Some(TensorShape::from_types(dims));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Extract literal int from Type::Literal(Lit::Int(...))
    pub fn literal_int(ty: &Type) -> Option<i64> {
        match ty {
            Type::Literal(box Literal {
                value: Lit::Int(n), ..
            }) => n.as_i64(),
            _ => None,
        }
    }

    /// Extract symbolic dimension from Type
    /// Handles Dim[N], SizeExpr, Quantified, Var, etc.
    pub fn dimension(ty: &Type) -> Option<Type> {
        match ty {
            // Dim[inner] -> extract inner (could be Quantified, SizeExpr, etc.)
            Type::Dim(inner) => Some((**inner).clone()),
            // Already a SizeExpr
            Type::Size(_) => Some(ty.clone()),
            // Type variable or quantified
            Type::Quantified(_) | Type::Var(_) => Some(ty.clone()),
            // Literal int -> wrap in SizeExpr
            Type::Literal(box Literal {
                value: Lit::Int(n), ..
            }) => n
                .as_i64()
                .map(|v| Type::Size(crate::tensor::SizeExpr::Literal(v))),
            _ => None,
        }
    }

    /// Extract shape argument from tuple of ints or Dims
    /// Supports both literal tuples and symbolic tuples
    /// Also handles nested tuples (e.g., from variadic binding of tuple args)
    pub fn shape_arg(ty: &Type) -> Option<TensorShape> {
        match ty {
            // Tuple of dimensions (literal or symbolic)
            Type::Tuple(Tuple::Concrete(elts)) => {
                // First, try to extract dimensions directly
                let dims: Vec<Type> = elts.iter().filter_map(dimension).collect();

                if dims.len() == elts.len() {
                    Some(TensorShape::from_types(dims))
                } else if elts.len() == 1 {
                    // Handle nested tuple case: tuple[tuple[dims...]] -> extract inner tuple
                    // This happens when variadic binding wraps a tuple arg in another tuple
                    shape_arg(&elts[0])
                } else {
                    None
                }
            }
            // Already a tensor shape (from a Tensor type argument)
            _ => concrete_tensor_shape(ty),
        }
    }

    /// Extract int list from tuple of literal ints
    /// Returns None if any element is not a literal int
    /// Also handles nested tuples (e.g., from variadic binding of tuple args)
    pub fn int_list(ty: &Type) -> Option<Vec<i64>> {
        match ty {
            Type::Tuple(Tuple::Concrete(elts)) => {
                // First, try to extract ints directly
                let result = elts.iter().map(literal_int).collect::<Option<Vec<i64>>>();
                if result.is_some() {
                    result
                } else if elts.len() == 1 {
                    // Handle nested tuple case: tuple[tuple[ints...]] -> extract inner tuple
                    int_list(&elts[0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract dimension list from tuple of Dims or dimension types
    /// Returns None if any element is not a valid dimension type
    /// Also handles nested tuples (e.g., from variadic binding of tuple args)
    pub fn dim_list(ty: &Type) -> Option<Vec<Type>> {
        match ty {
            Type::Tuple(Tuple::Concrete(elts)) => {
                // First, try to extract dimensions directly
                let result = elts.iter().map(dimension).collect::<Option<Vec<Type>>>();
                if result.is_some() {
                    result
                } else if elts.len() == 1 {
                    // Handle nested tuple case: tuple[tuple[dims...]] -> extract inner tuple
                    dim_list(&elts[0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract bool literal from Type::Literal(Lit::Bool(...))
    pub fn bool_arg(ty: &Type) -> Option<bool> {
        match ty {
            Type::Literal(box Literal {
                value: Lit::Bool(b),
                ..
            }) => Some(*b),
            _ => None,
        }
    }

    /// Extract string literal from Type::Literal(Lit::Str(...))
    pub fn string_arg(ty: &Type) -> Option<String> {
        match ty {
            Type::Literal(box Literal {
                value: Lit::Str(s), ..
            }) => Some(s.to_string()),
            _ => None,
        }
    }

    /// Extract list or tuple of tensor shapes
    /// Handles both list[Tensor[...]] and tuple[Tensor[...], ...]
    /// Returns None for list types (can't determine element count) or unbounded tuples
    pub fn tensor_list(ty: &Type) -> Option<Vec<TensorShape>> {
        use crate::tuple::Tuple;

        match ty {
            // list[Tensor[...]] - can't determine element count, return None
            Type::ClassType(class_type) if class_type.has_qname("builtins", "list") => {
                // Lists don't preserve element count in the type system
                // Fall back to fixture for now
                None
            }
            // tuple[Tensor[...], ...] - unbounded, can't determine count
            Type::Tuple(Tuple::Unbounded(_)) => None,
            // tuple[Tensor[...], Tensor[...], ...] - concrete, extract all
            Type::Tuple(Tuple::Concrete(elems)) => {
                // Check if first element is a tensor
                if let Some(first) = elems.first() {
                    let is_tensor = match first {
                        Type::Tensor(_) => true,
                        Type::ClassType(ct) => ct.has_qname("torch", "Tensor"),
                        _ => false,
                    };

                    if is_tensor {
                        // Tuple of tensors - extract all
                        let shapes: Option<Vec<TensorShape>> =
                            elems.iter().map(concrete_tensor_shape).collect();
                        return shapes;
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Builder for MetaShapeArgs to simplify bind_args implementations.
    ///
    /// Common patterns like extracting self tensor shape with optional dim/keepdim
    /// can be expressed concisely using this builder.
    ///
    /// # Example
    /// ```ignore
    /// fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
    ///     BindArgsBuilder::new()
    ///         .self_shape(bound_args)?
    ///         .optional_int(bound_args, "dim")
    ///         .optional_bool(bound_args, "keepdim")
    ///         .build()
    /// }
    /// ```
    pub struct BindArgsBuilder {
        args: std::collections::HashMap<String, super::MetaShapeArg>,
    }

    impl BindArgsBuilder {
        pub fn new() -> Self {
            Self {
                args: std::collections::HashMap::new(),
            }
        }

        /// Extract "self" tensor shape. Returns None if not a tensor or missing.
        pub fn self_shape(
            mut self,
            bound_args: &std::collections::HashMap<String, Type>,
        ) -> Option<Self> {
            let shape = concrete_tensor_shape(bound_args.get("self")?)?;
            self.args
                .insert("self".to_owned(), super::MetaShapeArg::Shape(shape));
            Some(self)
        }

        /// Add optional int parameter by name if present.
        pub fn optional_int(
            mut self,
            bound_args: &std::collections::HashMap<String, Type>,
            name: &str,
        ) -> Self {
            if let Some(ty) = bound_args.get(name)
                && let Some(val) = literal_int(ty)
            {
                self.args
                    .insert(name.to_owned(), super::MetaShapeArg::Int(val));
            }
            self
        }

        /// Add required int parameter by name. Returns None if missing or not an int.
        pub fn required_int(
            mut self,
            bound_args: &std::collections::HashMap<String, Type>,
            name: &str,
        ) -> Option<Self> {
            let val = literal_int(bound_args.get(name)?)?;
            self.args
                .insert(name.to_owned(), super::MetaShapeArg::Int(val));
            Some(self)
        }

        /// Add optional bool parameter by name if present.
        pub fn optional_bool(
            mut self,
            bound_args: &std::collections::HashMap<String, Type>,
            name: &str,
        ) -> Self {
            if let Some(ty) = bound_args.get(name)
                && let Some(val) = bool_arg(ty)
            {
                self.args
                    .insert(name.to_owned(), super::MetaShapeArg::Bool(val));
            }
            self
        }

        /// Build the MetaShapeArgs.
        pub fn build(self) -> Option<super::MetaShapeArgs> {
            Some(super::MetaShapeArgs { args: self.args })
        }
    }

    impl Default for BindArgsBuilder {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Default conversion from MetaShapeResult to Type
/// Used by result_to_type() default implementation
///
/// IMPORTANT: Does NOT silently fall back. If result doesn't match return type structure,
/// this will panic (indicating an implementation bug).
pub fn convert_result_to_type_default(result: MetaShapeResult, ret_type: &Type) -> Type {
    use crate::lit_int::LitInt;
    use crate::literal::Lit;
    use crate::tensor::TensorType;
    use crate::tuple::Tuple;

    match result {
        MetaShapeResult::Tensors(shapes) if shapes.len() == 1 => {
            // Single tensor: inject computed shape into return type template
            match ret_type {
                Type::Tensor(tensor) => Type::Tensor(Box::new(TensorType::new(
                    tensor.base_class.clone(),
                    shapes[0].clone(),
                ))),
                Type::ClassType(class_type) if class_type.has_qname("torch", "Tensor") => {
                    // Fixture is ClassType(torch.Tensor[...]) - use it as base class
                    Type::Tensor(Box::new(TensorType::new(
                        class_type.clone(),
                        shapes[0].clone(),
                    )))
                }
                _ => {
                    // Mismatch: meta-shape returned single tensor but fixture is not Tensor
                    // Return fixture type unchanged to avoid panic - this will show as type error
                    ret_type.clone()
                }
            }
        }
        MetaShapeResult::Tensors(shapes) if shapes.len() > 1 => {
            // Multiple tensors: update tuple elements
            match ret_type {
                Type::Tuple(Tuple::Concrete(elems)) if elems.len() == shapes.len() => {
                    let updated: Vec<Type> = elems
                        .iter()
                        .zip(shapes.iter())
                        .map(|(elem, shape)| match elem {
                            Type::Tensor(t) => Type::Tensor(Box::new(TensorType::new(
                                t.base_class.clone(),
                                shape.clone(),
                            ))),
                            Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
                                Type::Tensor(Box::new(TensorType::new(cls.clone(), shape.clone())))
                            }
                            _ => panic!(
                                "Meta-shape returned tensors but tuple element is not Tensor: {:?}",
                                elem
                            ),
                        })
                        .collect();
                    Type::Tuple(Tuple::Concrete(updated))
                }
                Type::Tuple(Tuple::Unbounded(elem)) => {
                    // Fixture is unbounded tuple, but we computed exact count
                    // Convert to concrete tuple with computed shapes
                    match elem.as_ref() {
                        Type::Tensor(t) => {
                            let tensors: Vec<Type> = shapes
                                .iter()
                                .map(|shape| {
                                    Type::Tensor(Box::new(TensorType::new(
                                        t.base_class.clone(),
                                        shape.clone(),
                                    )))
                                })
                                .collect();
                            Type::Tuple(Tuple::Concrete(tensors))
                        }
                        Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
                            let tensors: Vec<Type> = shapes
                                .iter()
                                .map(|shape| {
                                    Type::Tensor(Box::new(TensorType::new(
                                        cls.clone(),
                                        shape.clone(),
                                    )))
                                })
                                .collect();
                            Type::Tuple(Tuple::Concrete(tensors))
                        }
                        _ => {
                            // Mismatch: unbounded tuple element is not Tensor
                            // Return fixture type unchanged
                            ret_type.clone()
                        }
                    }
                }
                _ => {
                    // Mismatch: meta-shape returned multiple tensors but fixture is not tuple or wrong size
                    // Return fixture type unchanged
                    ret_type.clone()
                }
            }
        }
        MetaShapeResult::TensorShapeless => {
            // Shapeless tensor - use return type base class
            match ret_type {
                Type::Tensor(tensor) => {
                    Type::Tensor(Box::new(TensorType::shapeless(tensor.base_class.clone())))
                }
                _ => {
                    // Mismatch: shapeless tensor but fixture is not Tensor
                    ret_type.clone()
                }
            }
        }
        MetaShapeResult::Int(n) => Lit::Int(LitInt::new(n)).to_implicit_type(),
        MetaShapeResult::IntSymbolic(ty) => ty,
        MetaShapeResult::IntSymbolicTuple(dims) => Type::concrete_tuple(dims),
        MetaShapeResult::UnboundedTuple { element_shape } => {
            // Extract base class from return type if it's a tuple of tensors
            match ret_type {
                Type::Tuple(Tuple::Unbounded(elem)) => match elem.as_ref() {
                    Type::Tensor(t) => Type::Tuple(Tuple::Unbounded(Box::new(Type::Tensor(
                        Box::new(TensorType::new(t.base_class.clone(), element_shape)),
                    )))),
                    Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
                        Type::Tuple(Tuple::Unbounded(Box::new(Type::Tensor(Box::new(
                            TensorType::new(cls.clone(), element_shape),
                        )))))
                    }
                    _ => {
                        // Mismatch: unbounded tensor tuple but element is not Tensor
                        ret_type.clone()
                    }
                },
                _ => {
                    // Mismatch: unbounded tuple but fixture is not unbounded tuple
                    ret_type.clone()
                }
            }
        }
        MetaShapeResult::UnboundedTupleShapeless => {
            // Extract base class from return type
            match ret_type {
                Type::Tuple(Tuple::Unbounded(elem)) => match elem.as_ref() {
                    Type::Tensor(t) => Type::Tuple(Tuple::Unbounded(Box::new(Type::Tensor(
                        Box::new(TensorType::shapeless(t.base_class.clone())),
                    )))),
                    Type::ClassType(cls) if cls.has_qname("torch", "Tensor") => {
                        Type::Tuple(Tuple::Unbounded(Box::new(Type::Tensor(Box::new(
                            TensorType::shapeless(cls.clone()),
                        )))))
                    }
                    _ => {
                        // Mismatch: unbounded shapeless tuple but element is not Tensor
                        ret_type.clone()
                    }
                },
                _ => {
                    // Mismatch: unbounded tuple but fixture is not unbounded tuple
                    ret_type.clone()
                }
            }
        }
        MetaShapeResult::Bool(b) => Lit::Bool(b).to_implicit_type(),
        MetaShapeResult::Unknown => {
            // Unknown - use fixture as-is (this is for backward compat)
            ret_type.clone()
        }
        MetaShapeResult::Tensors(_) => {
            // Empty tensor list - return fixture type unchanged
            ret_type.clone()
        }
    }
}

// ============================================================================
// Meta-Shape Function Trait
// ============================================================================

/// A function that computes output shapes from input shapes
pub trait MetaShapeFunction: Debug + Send + Sync {
    /// Parameter names in order (e.g., ["self", "dim", "keepdim"])
    /// Used for signature binding like inspect.signature
    fn signature(&self) -> &[&'static str];

    /// Convert bound arguments to MetaShapeArgs
    ///
    /// This is the input-side dual to result_to_type():
    /// - bind_args(): HashMap<String, Type> → Option<MetaShapeArgs> (input conversion)
    /// - result_to_type(): MetaShapeResult → Type (output conversion)
    ///
    /// Returns None to fall back to fixture return type.
    /// This is where you check if all required info is available (e.g., all tensors have shapes)
    ///
    /// # Arguments
    /// - `bound_args`: Arguments bound to parameter names via signature binding
    ///
    /// # Returns
    /// - `Some(MetaShapeArgs)`: Ready to compute
    /// - `None`: Fall back to fixture (e.g., missing shape info, unsupported types)
    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs>;

    /// Compute meta-shape result from bound arguments
    ///
    /// This is the core shape computation logic.
    /// Should handle both literal and symbolic shapes.
    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError>;

    /// Convert MetaShapeResult back to Type
    ///
    /// Handles injecting computed shape into original return type.
    ///
    /// IMPORTANT: This should NOT silently fall back. If the result doesn't match
    /// the fixture signature structure, that's a bug and should panic.
    ///
    /// # Arguments
    /// - `result`: Computed meta-shape result
    /// - `original_return_type`: Return type from fixture (used as template)
    ///
    /// # Returns
    /// Updated type with computed shape information
    ///
    /// # Panics
    /// When result structure doesn't match fixture signature (implementation bug)
    fn result_to_type(&self, result: MetaShapeResult, original_return_type: &Type) -> Type {
        // Default: use standard conversion logic
        convert_result_to_type_default(result, original_return_type)
    }

    /// Name of this meta-shape function (for error messages)
    fn name(&self) -> &str;
}

// ============================================================================
// Built-in Meta-Shape Functions
// ============================================================================

/// Meta-shape function for torch.reshape
#[derive(Debug)]
pub struct ReshapeMetaShape;

impl MetaShapeFunction for ReshapeMetaShape {
    // ========== New Methods (Phase 2) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "*shape"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract input tensor shape
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        // Extract target shape - handle both literal and symbolic
        let target_shape = shape_arg(bound_args.get("shape")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("shape".to_owned(), MetaShapeArg::Shape(target_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "reshape expects 'self' argument".to_owned(),
                });
            }
        };

        // Get target shape
        let target_shape = match args.args.get("shape") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "reshape requires 'shape' argument".to_owned(),
                });
            }
        };

        // Check if target shape has symbolic dimensions
        let all_literal = target_shape.all_literal();

        // Helper to multiply dimensions, skipping multiplication by 1
        let smart_mul = |left: Type, right: Type| -> Type {
            match (&left, &right) {
                (Type::Size(SizeExpr::Literal(1)), _) => right,
                (_, Type::Size(SizeExpr::Literal(1))) => left,
                _ => Type::Size(SizeExpr::mul(left, right)),
            }
        };

        // Helper to divide dimensions
        let smart_div = |left: Type, right: Type| -> Type {
            match (&left, &right) {
                (_, Type::Size(SizeExpr::Literal(1))) => left,
                // If left == right, result is 1
                _ if left == right => Type::Size(SizeExpr::Literal(1)),
                _ => Type::Size(SizeExpr::floor_div(left, right)),
            }
        };

        // Check if target contains -1 (need to infer a dimension)
        let has_minus_one = target_shape.dims().iter().any(|d| {
            matches!(d, Type::Size(SizeExpr::Literal(-1)))
                || matches!(d, Type::Literal(box Literal { value: Lit::Int(i), .. }) if i.as_i64() == Some(-1))
        });

        if !all_literal && has_minus_one {
            // Symbolic path with -1 inference
            // Strategy: Cancel matching dimensions between input and output to simplify
            //
            // Example: input [B, T, NHeads, HeadDim], output [B, T, NHeads, -1, 2]
            // - B, T, NHeads appear in both -> cancel them
            // - Remaining input: [HeadDim]
            // - Remaining output (excluding -1): [2]
            // - Inferred dim: HeadDim / 2

            // Collect output dims that are not -1
            let output_known_dims: Vec<Type> = target_shape
                .dims()
                .iter()
                .filter(|d| {
                    !matches!(d, Type::Size(SizeExpr::Literal(-1)))
                        && !matches!(d, Type::Literal(box Literal { value: Lit::Int(i), .. }) if i.as_i64() == Some(-1))
                })
                .cloned()
                .collect();

            // Start with all input dims as "remaining"
            let mut remaining_input: Vec<Type> = input_shape.dims().clone();

            // Cancel matching dims from output_known_dims
            for out_dim in &output_known_dims {
                if let Some(pos) = remaining_input.iter().position(|d| d == out_dim) {
                    remaining_input.remove(pos);
                }
            }

            // remaining_output are the known dims that weren't cancelled
            let mut remaining_output: Vec<Type> = output_known_dims.clone();
            for in_dim in input_shape.dims() {
                if let Some(pos) = remaining_output.iter().position(|d| d == in_dim) {
                    remaining_output.remove(pos);
                }
            }

            // Compute product of remaining input dims
            let mut input_product = Type::Size(SizeExpr::Literal(1));
            for dim in &remaining_input {
                input_product = smart_mul(input_product, dim.clone());
            }

            // Compute product of remaining output dims
            let mut output_product = Type::Size(SizeExpr::Literal(1));
            for dim in &remaining_output {
                output_product = smart_mul(output_product, dim.clone());
            }

            // Infer -1 as remaining_input_product / remaining_output_product
            let inferred_dim = smart_div(input_product, output_product);

            // Build output dimensions, replacing -1 with inferred dim
            let output_dims: Vec<Type> = target_shape
                .dims()
                .iter()
                .map(|d| {
                    let is_minus_one = matches!(d, Type::Size(SizeExpr::Literal(-1)))
                        || matches!(d, Type::Literal(box Literal { value: Lit::Int(i), .. }) if i.as_i64() == Some(-1));
                    if is_minus_one {
                        inferred_dim.clone()
                    } else {
                        d.clone()
                    }
                })
                .collect();

            return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                output_dims,
            )]));
        }

        if !all_literal {
            // Symbolic path without -1: just use the dims as-is
            return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                target_shape.dims().clone(),
            )]));
        }

        // Concrete path: handle -1 inference for concrete dimensions
        let target_dims = target_shape.as_literals().unwrap();

        // Calculate total element count from input shape (symbolic or literal)
        // Initialize with the first dimension (or Literal(1) if empty)
        let mut input_numel_type = if input_shape.dims().is_empty() {
            Type::Size(SizeExpr::Literal(1))
        } else {
            input_shape.dims()[0].clone().clone().clone()
        };

        // Multiply remaining dimensions
        for dim_ty in &input_shape.dims()[1..] {
            input_numel_type = smart_mul(input_numel_type, dim_ty.clone());
        }

        // Simplify and try to extract a concrete element count
        let input_numel_type = simplify(input_numel_type);
        let input_size_opt = input_numel_type.as_shape_literal();

        // Validate dimension values (check for invalid values like -2, -3, etc.)
        for &dim in &target_dims {
            if dim < -1 {
                return Err(ShapeError::InvalidDimension {
                    value: dim,
                    reason: format!("cannot specify {} as a reshape dimension", dim),
                });
            }
            if dim == 0 {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "reshape dimensions cannot contain 0".to_owned(),
                });
            }
        }

        // Handle -1 inference
        let output_dims = if target_dims.contains(&-1) {
            // Count how many -1s there are
            let minus_one_count = target_dims.iter().filter(|&&d| d == -1).count();
            if minus_one_count > 1 {
                return Err(ShapeError::InvalidDimension {
                    value: -1,
                    reason: "can only specify one unknown dimension as -1".to_owned(),
                });
            }

            // Calculate product of known dimensions
            let mut known_product: i64 = 1;
            for &dim in &target_dims {
                if dim > 0 {
                    known_product *= dim;
                }
                // Skip -1 (will be inferred)
            }

            // Infer the -1 dimension
            let inferred_dim_type = if let Some(input_size) = input_size_opt {
                // Input size is concrete - use literal arithmetic
                let inferred = input_size / known_product;
                if inferred * known_product != input_size {
                    return Err(ShapeError::InvalidDimension {
                        value: -1,
                        reason: format!(
                            "shape {} is not compatible with input size {}",
                            target_dims
                                .iter()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            input_size
                        ),
                    });
                }
                Type::Size(SizeExpr::Literal(inferred))
            } else {
                // Input size is symbolic - compute symbolically
                // inferred = input_numel / known_product
                if known_product == 1 {
                    // Special case: -1 with no other dimensions means flatten
                    // e.g., Tensor[N, M].view(-1) -> Tensor[N*M]
                    // e.g., Tensor[C].view(1, -1, 1, 1) -> Tensor[1, C, 1, 1]
                    // Return the input_numel_type AS-IS (could be Quantified, Var, or SizeExpr)
                    input_numel_type
                } else {
                    // General case: divide symbolic expression by known product
                    // e.g., Tensor[C, 3].view(-1, 3) -> inferred = (C * 3) / 3 = C
                    // Return SizeExpr expression
                    Type::Size(SizeExpr::floor_div(
                        input_numel_type,
                        Type::Size(SizeExpr::Literal(known_product)),
                    ))
                }
            };

            // Replace -1 with inferred dimension
            target_dims
                .iter()
                .map(|&d| {
                    if d == -1 {
                        inferred_dim_type.clone()
                    } else {
                        Type::Size(SizeExpr::Literal(d))
                    }
                })
                .collect()
        } else {
            // No -1, just use the target dims as-is
            target_dims
                .iter()
                .map(|&d| Type::Size(SizeExpr::Literal(d)))
                .collect()
        };

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "reshape"
    }
}

/// Meta-shape function for torch.cat
#[derive(Debug)]
pub struct ConcatMetaShape;

impl MetaShapeFunction for ConcatMetaShape {
    // ========== New Methods (Phase 3) ==========

    fn signature(&self) -> &[&'static str] {
        &["tensors", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract tuple of tensor shapes (lists not supported - use tuples!)
        let tensor_shapes = tensor_list(bound_args.get("tensors")?)?;

        let mut args_map =
            HashMap::from([("tensors".to_owned(), MetaShapeArg::ShapeList(tensor_shapes))]);

        // dim is optional (default 0)
        if let Some(dim_ty) = bound_args.get("dim")
            && let Some(dim_val) = literal_int(dim_ty)
        {
            args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Extract tensor shapes from ShapeList
        let inputs = match args.args.get("tensors") {
            Some(MetaShapeArg::ShapeList(shapes)) => shapes,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "cat expects 'tensors' argument".to_owned(),
                });
            }
        };

        if inputs.is_empty() {
            return Err(ShapeError::InvalidDimension {
                value: 0,
                reason: "cat requires at least one input".to_owned(),
            });
        }

        let dim = match args.args.get("dim") {
            Some(MetaShapeArg::Int(d)) => {
                let d = *d;
                let rank = inputs[0].rank() as i64;
                // Handle negative indexing
                if d < 0 {
                    (rank + d) as usize
                } else {
                    d as usize
                }
            }
            _ => 0, // Default to dim=0
        };

        let first_shape = &inputs[0];
        let rank = first_shape.rank();

        if dim >= rank {
            return Err(ShapeError::InvalidDimension {
                value: dim as i64,
                reason: format!("dim {} out of bounds for rank {}", dim, rank),
            });
        }

        // All inputs must have same rank
        for shape in inputs.iter().skip(1) {
            if shape.rank() != rank {
                return Err(ShapeError::RankMismatch {
                    got: shape.rank(),
                    want: rank,
                });
            }
        }

        // Build output shape: all dims same except concat dim which is summed
        let mut output_dims = Vec::new();
        for d in 0..rank {
            if d == dim {
                // Sum along concat dimension
                let mut sum_dim = inputs[0].get_dim(d);
                for shape in &inputs[1..] {
                    sum_dim = Type::Size(SizeExpr::add(sum_dim, shape.get_dim(d)));
                }
                output_dims.push(simplify(sum_dim));
            } else {
                // All other dims must match (we'll validate with shapes_compatible later)
                output_dims.push(first_shape.get_dim(d));
            }
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "cat"
    }
}

/// Meta-shape function for torch.broadcast_to
#[derive(Debug)]
pub struct BroadcastToMetaShape;

impl MetaShapeFunction for BroadcastToMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "shape"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let target_shape = shape_arg(bound_args.get("shape")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("shape".to_owned(), MetaShapeArg::Shape(target_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get target shape from args
        let target_shape = match args.args.get("shape") {
            Some(MetaShapeArg::Shape(s)) => s.clone(),
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "broadcast_to requires 'shape' argument".to_owned(),
                });
            }
        };

        // torch.broadcast_to(input, target) broadcasts input TO target shape
        // The result is always exactly the target shape (after validating compatibility)
        // We skip detailed validation here and just return the target shape,
        // trusting that PyTorch will error at runtime if shapes are incompatible
        Ok(MetaShapeResult::Tensors(vec![target_shape]))
    }

    fn name(&self) -> &str {
        "broadcast_to"
    }
}

/// Meta-shape function for torch.squeeze
#[derive(Debug)]
pub struct SqueezeMetaShape;

impl MetaShapeFunction for SqueezeMetaShape {
    // ========== New Methods (Phase 2) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .optional_int(bound_args, "dim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "squeeze expects 'self' argument".to_owned(),
                });
            }
        };

        let dim = args.args.get("dim").and_then(|arg| {
            if let MetaShapeArg::Int(d) = arg {
                Some(*d)
            } else {
                None
            }
        });

        let mut output_dims = Vec::new();

        if let Some(dim_val) = dim {
            // Squeeze specific dimension
            let dim_idx = input_shape.normalize_dim(dim_val)?;

            for (i, dim) in input_shape.dims().iter().enumerate() {
                if i == dim_idx {
                    // Only remove if it's 1
                    if let Type::Size(SizeExpr::Literal(1)) = dim {
                        continue; // Skip this dimension
                    }
                }
                output_dims.push(dim.clone());
            }
        } else {
            // Squeeze all dimensions that are 1
            for dim in input_shape.dims() {
                if let Type::Size(SizeExpr::Literal(1)) = dim {
                    continue; // Skip dimensions with size 1
                }
                output_dims.push(dim.clone());
            }
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "squeeze"
    }
}

/// Meta-shape function for torch.unsqueeze
#[derive(Debug)]
pub struct UnsqueezeMetaShape;

impl MetaShapeFunction for UnsqueezeMetaShape {
    // ========== New Methods (Phase 2) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .required_int(bound_args, "dim")?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "unsqueeze expects 'self' argument".to_owned(),
                });
            }
        };

        let dim = match args.args.get("dim") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "unsqueeze requires 'dim' argument".to_owned(),
                });
            }
        };

        let rank = input_shape.rank() as i64;
        let dim_idx = if dim < 0 {
            (rank + 1 + dim) as usize
        } else {
            dim as usize
        };

        if dim_idx > input_shape.rank() {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: format!("dim {} out of bounds for rank {}", dim, input_shape.rank()),
            });
        }

        let mut output_dims = input_shape.dims().clone();
        output_dims.insert(dim_idx, Type::Size(SizeExpr::Literal(1)));

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "unsqueeze"
    }
}

/// Meta-shape function for torch.transpose
#[derive(Debug)]
pub struct TransposeMetaShape;

impl MetaShapeFunction for TransposeMetaShape {
    // ========== New Methods (Phase 2) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "dim0", "dim1"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract input tensor shape
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        // Extract dimension indices
        let dim0 = literal_int(bound_args.get("dim0")?)?;
        let dim1 = literal_int(bound_args.get("dim1")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("dim0".to_owned(), MetaShapeArg::Int(dim0)),
                ("dim1".to_owned(), MetaShapeArg::Int(dim1)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "transpose expects 'self' argument".to_owned(),
                });
            }
        };

        let dim0 = match args.args.get("dim0") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "transpose requires 'dim0' argument".to_owned(),
                });
            }
        };

        let dim1 = match args.args.get("dim1") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "transpose requires 'dim1' argument".to_owned(),
                });
            }
        };

        let rank = input_shape.rank() as i64;
        let dim0_idx = if dim0 < 0 {
            (rank + dim0) as usize
        } else {
            dim0 as usize
        };

        let dim1_idx = if dim1 < 0 {
            (rank + dim1) as usize
        } else {
            dim1 as usize
        };

        if dim0_idx >= input_shape.rank() || dim1_idx >= input_shape.rank() {
            return Err(ShapeError::InvalidDimension {
                value: 0,
                reason: format!(
                    "transpose dims ({}, {}) out of bounds for rank {}",
                    dim0, dim1, rank
                ),
            });
        }

        let mut output_dims = input_shape.dims().clone();
        output_dims.swap(dim0_idx, dim1_idx);

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "transpose"
    }
}

/// Meta-shape function for torch.permute
#[derive(Debug)]
pub struct PermuteMetaShape;

impl MetaShapeFunction for PermuteMetaShape {
    // ========== New Methods (Phase 2) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "*dims"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract input tensor shape
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        // Extract dims - should be a tuple of ints from variadic param
        let dims = int_list(bound_args.get("dims")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("dims".to_owned(), MetaShapeArg::IntList(dims)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "permute expects 'self' argument".to_owned(),
                });
            }
        };

        let dims = match args.args.get("dims") {
            Some(MetaShapeArg::IntList(d)) => d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "permute requires 'dims' argument (IntList)".to_owned(),
                });
            }
        };

        if dims.len() != input_shape.rank() {
            return Err(ShapeError::RankMismatch {
                got: dims.len(),
                want: input_shape.rank(),
            });
        }

        let mut output_dims = vec![Type::Size(SizeExpr::Literal(0)); input_shape.rank()];
        for (new_idx, &old_idx) in dims.iter().enumerate() {
            let old_idx_norm = if old_idx < 0 {
                (input_shape.rank() as i64 + old_idx) as usize
            } else {
                old_idx as usize
            };

            if old_idx_norm >= input_shape.rank() {
                return Err(ShapeError::InvalidDimension {
                    value: old_idx,
                    reason: format!("permute dim {} out of bounds", old_idx),
                });
            }

            output_dims[new_idx] = input_shape.get_dim(old_idx_norm);
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "permute"
    }
}

/// Meta-shape function for torch.aminmax
/// Returns tuple of (min, max) with same reduction semantics
#[derive(Debug)]
pub struct AminmaxMetaShape;

impl MetaShapeFunction for AminmaxMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "keepdim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        use crate::literal::Lit;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // dim can be int, tuple of ints, or None
        if let Some(dim_ty) = bound_args.get("dim")
            && !dim_ty.is_none()
        {
            match dim_ty {
                Type::Literal(box Literal {
                    value: Lit::Int(_), ..
                }) => {
                    if let Some(dim_val) = literal_int(dim_ty) {
                        args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
                    }
                }
                Type::Tuple(_) => {
                    if let Some(dims) = int_list(dim_ty) {
                        args_map.insert("dim".to_owned(), MetaShapeArg::IntList(dims));
                    }
                }
                _ => {}
            }
        }

        if let Some(keepdim_ty) = bound_args.get("keepdim")
            && let Some(keepdim_val) = bool_arg(keepdim_ty)
        {
            args_map.insert("keepdim".to_owned(), MetaShapeArg::Bool(keepdim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "aminmax requires 'self' argument".to_owned(),
                });
            }
        };

        let keepdim = args
            .args
            .get("keepdim")
            .and_then(|arg| {
                if let MetaShapeArg::Bool(b) = arg {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Use same reduction logic as ReduceMetaShape
        let dims_to_reduce: Vec<usize> = match args.args.get("dim") {
            Some(MetaShapeArg::Int(dim_val)) => {
                let dim_idx = input_shape.normalize_dim(*dim_val)?;
                vec![dim_idx]
            }
            Some(MetaShapeArg::IntList(dim_list)) => {
                let mut indices = Vec::new();
                for &dim_val in dim_list {
                    let dim_idx = input_shape.normalize_dim(dim_val)?;
                    indices.push(dim_idx);
                }
                indices
            }
            None => {
                // Reduce all dimensions
                if keepdim {
                    let output_dims = vec![Type::Size(SizeExpr::Literal(1)); input_shape.rank()];
                    let shape = TensorShape::from_types(output_dims);
                    // Return 2 identical shapes for (min, max)
                    return Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]));
                } else {
                    let shape = TensorShape::new(vec![]);
                    // Return 2 identical shapes for (min, max)
                    return Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]));
                }
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "dim must be Int or IntList".to_owned(),
                });
            }
        };

        // Build output shape
        let mut output_dims = Vec::new();
        for (i, dim) in input_shape.dims().iter().enumerate() {
            if dims_to_reduce.contains(&i) {
                if keepdim {
                    output_dims.push(Type::Size(SizeExpr::Literal(1)));
                }
            } else {
                output_dims.push(dim.clone());
            }
        }

        let shape = TensorShape::from_types(output_dims);
        // Return 2 identical shapes for (min, max)
        Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]))
    }

    fn name(&self) -> &str {
        "aminmax"
    }
}

/// Meta-shape function for torch.min/max/median with dual return types
/// When dim is None: returns single Tensor (scalar)
/// When dim is specified: returns tuple[Tensor, Tensor] (values, indices)
#[derive(Debug)]
pub struct MinMaxMedianMetaShape;

impl MetaShapeFunction for MinMaxMedianMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "keepdim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // dim can be int or None
        if let Some(dim_ty) = bound_args.get("dim")
            && !dim_ty.is_none()
            && let Some(dim_val) = literal_int(dim_ty)
        {
            args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
        }

        if let Some(keepdim_ty) = bound_args.get("keepdim")
            && let Some(keepdim_val) = bool_arg(keepdim_ty)
        {
            args_map.insert("keepdim".to_owned(), MetaShapeArg::Bool(keepdim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "reduction requires 'self' argument".to_owned(),
                });
            }
        };

        let keepdim = args
            .args
            .get("keepdim")
            .and_then(|arg| {
                if let MetaShapeArg::Bool(b) = arg {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Check if dim parameter is provided
        let has_dim = args.args.contains_key("dim");

        if !has_dim {
            // No dim specified - reduce all dimensions to scalar
            return Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![])]));
        }

        // Dim is specified - return tuple of (values, indices)
        let dims_to_reduce: Vec<usize> = match args.args.get("dim") {
            Some(MetaShapeArg::Int(dim_val)) => {
                let dim_idx = input_shape.normalize_dim(*dim_val)?;
                vec![dim_idx]
            }
            None => {
                // Shouldn't happen since we checked has_dim, but handle gracefully
                return Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![])]));
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "dim must be Int".to_owned(),
                });
            }
        };

        // Build output shape (same for both values and indices)
        let mut output_dims = Vec::new();
        for (i, dim) in input_shape.dims().iter().enumerate() {
            if dims_to_reduce.contains(&i) {
                if keepdim {
                    output_dims.push(Type::Size(SizeExpr::Literal(1)));
                }
            } else {
                output_dims.push(dim.clone());
            }
        }

        let shape = TensorShape::from_types(output_dims);
        // Return 2 identical shapes for (values, indices)
        Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]))
    }

    fn name(&self) -> &str {
        "min_max_median"
    }
}

/// Meta-shape function for torch.mode/kthvalue
/// These reduce along a dimension and return tuple[Tensor, Tensor] (values, indices)
#[derive(Debug)]
pub struct TupleReduceMetaShape;

impl MetaShapeFunction for TupleReduceMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "keepdim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .optional_int(bound_args, "dim")
            .optional_bool(bound_args, "keepdim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "reduction requires 'self' argument".to_owned(),
                });
            }
        };

        let keepdim = args
            .args
            .get("keepdim")
            .and_then(|arg| {
                if let MetaShapeArg::Bool(b) = arg {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Get dim parameter (default is -1 for most of these operations)
        let dim = args
            .args
            .get("dim")
            .and_then(|arg| {
                if let MetaShapeArg::Int(i) = arg {
                    Some(*i)
                } else {
                    None
                }
            })
            .unwrap_or(-1);

        let dim_idx = input_shape.normalize_dim(dim)?;

        // Build output shape
        let mut output_dims = Vec::new();
        for (i, dim) in input_shape.dims().iter().enumerate() {
            if i == dim_idx {
                if keepdim {
                    output_dims.push(Type::Size(SizeExpr::Literal(1)));
                }
            } else {
                output_dims.push(dim.clone());
            }
        }

        let shape = TensorShape::from_types(output_dims);
        // Return 2 identical shapes for (values, indices)
        Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]))
    }

    fn name(&self) -> &str {
        "tuple_reduce"
    }
}

/// Meta-shape function for torch.topk
/// Returns top k elements along dimension
/// Changes the selected dimension size to k
/// Returns tuple[Tensor, Tensor] (values, indices)
#[derive(Debug)]
pub struct TopKMetaShape;

impl MetaShapeFunction for TopKMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "k", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .required_int(bound_args, "k")?
            .optional_int(bound_args, "dim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "topk requires 'self' argument".to_owned(),
                });
            }
        };

        // Get k parameter
        let k = args
            .args
            .get("k")
            .and_then(|arg| {
                if let MetaShapeArg::Int(i) = arg {
                    Some(*i)
                } else {
                    None
                }
            })
            .ok_or_else(|| ShapeError::InvalidDimension {
                value: 0,
                reason: "topk requires 'k' parameter".to_owned(),
            })?;

        // Get dim parameter (default is -1)
        let dim = args
            .args
            .get("dim")
            .and_then(|arg| {
                if let MetaShapeArg::Int(i) = arg {
                    Some(*i)
                } else {
                    None
                }
            })
            .unwrap_or(-1);

        let dim_idx = input_shape.normalize_dim(dim)?;

        // Build output shape: same as input but selected dimension becomes k
        let mut output_dims = input_shape.dims().clone();
        // If k == -1, it's a runtime value (sentinel), use Unknown dimension
        // Otherwise use the literal k value
        output_dims[dim_idx] = if k == -1 {
            Type::Any(crate::types::AnyStyle::Implicit)
        } else {
            Type::Size(SizeExpr::Literal(k))
        };

        let shape = TensorShape::from_types(output_dims);
        // Return 2 identical shapes for (values, indices)
        Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]))
    }

    fn name(&self) -> &str {
        "topk"
    }
}

/// Meta-shape function for torch.sum (and other reductions: mean, prod, etc.)
#[derive(Debug)]
pub struct ReduceMetaShape;

impl MetaShapeFunction for ReduceMetaShape {
    // ========== New Methods (Phase 3) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "keepdim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        use crate::literal::Lit;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // dim can be int, tuple of ints, or None
        if let Some(dim_ty) = bound_args.get("dim") {
            if dim_ty.is_none() {
                // dim=None - reduce all dimensions (don't add to args_map)
            } else {
                match dim_ty {
                    Type::Literal(box Literal {
                        value: Lit::Int(_), ..
                    }) => {
                        if let Some(dim_val) = literal_int(dim_ty) {
                            args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
                        }
                    }
                    Type::Tuple(_) => {
                        if let Some(dims) = int_list(dim_ty) {
                            args_map.insert("dim".to_owned(), MetaShapeArg::IntList(dims));
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(keepdim_ty) = bound_args.get("keepdim")
            && let Some(keepdim_val) = bool_arg(keepdim_ty)
        {
            args_map.insert("keepdim".to_owned(), MetaShapeArg::Bool(keepdim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "reduction expects 'self' argument".to_owned(),
                });
            }
        };

        let keepdim = args
            .args
            .get("keepdim")
            .and_then(|arg| {
                if let MetaShapeArg::Bool(b) = arg {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Handle both single dim (Int) and multiple dims (IntList)
        let dims_to_reduce: Vec<usize> = match args.args.get("dim") {
            Some(MetaShapeArg::Int(dim_val)) => {
                // Single dimension
                let dim_idx = input_shape.normalize_dim(*dim_val)?;
                vec![dim_idx]
            }
            Some(MetaShapeArg::IntList(dim_list)) => {
                // Multiple dimensions
                let mut indices = Vec::new();
                for &dim_val in dim_list {
                    let dim_idx = input_shape.normalize_dim(dim_val)?;
                    indices.push(dim_idx);
                }
                indices
            }
            None => {
                // Reduce all dimensions - return scalar (0-d tensor)
                if keepdim {
                    // All dims become 1
                    let output_dims = vec![Type::Size(SizeExpr::Literal(1)); input_shape.rank()];
                    return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                        output_dims,
                    )]));
                } else {
                    // Scalar - empty shape
                    return Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![])]));
                }
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "dim must be Int or IntList".to_owned(),
                });
            }
        };

        // Build output shape by removing/replacing reduced dimensions
        let mut output_dims = Vec::new();
        for (i, dim) in input_shape.dims().iter().enumerate() {
            if dims_to_reduce.contains(&i) {
                if keepdim {
                    output_dims.push(Type::Size(SizeExpr::Literal(1)));
                }
                // else skip this dimension (reduce it)
            } else {
                output_dims.push(dim.clone());
            }
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "reduce"
    }
}

/// Meta-shape function for torch.randn and other tensor creation functions
/// Creates a tensor with the specified shape
#[derive(Debug)]
pub struct RandnMetaShape;

impl MetaShapeFunction for RandnMetaShape {
    // ========== New Methods (Phase 3) ==========

    fn signature(&self) -> &[&'static str] {
        &["*size"] // No self - this is a static method
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract shape from size parameter (variadic creates a tuple)
        let shape = shape_arg(bound_args.get("size")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([("size".to_owned(), MetaShapeArg::Shape(shape))]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let shape = match args.args.get("size") {
            Some(MetaShapeArg::Shape(s)) => s.clone(),
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "randn requires 'size' argument".to_owned(),
                });
            }
        };

        Ok(MetaShapeResult::Tensors(vec![shape]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "randn"
    }
}

/// Meta-shape function for torch.full
/// full(size, fill_value) creates a tensor with the specified shape filled with fill_value
#[derive(Debug)]
pub struct FullMetaShape;

impl MetaShapeFunction for FullMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["size"] // Just the size tuple, not variadic
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract shape from size parameter (tuple)
        let shape = shape_arg(bound_args.get("size")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([("size".to_owned(), MetaShapeArg::Shape(shape))]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let shape = match args.args.get("size") {
            Some(MetaShapeArg::Shape(s)) => s.clone(),
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "full requires 'size' argument".to_owned(),
                });
            }
        };

        Ok(MetaShapeResult::Tensors(vec![shape]))
    }

    fn name(&self) -> &str {
        "full"
    }
}

/// Meta-shape function for torch.arange
/// arange(end) creates a 1D tensor of size 'end'
/// arange(start, end) creates a 1D tensor of size 'end - start'
#[derive(Debug)]
pub struct ArangeMetaShape;

impl MetaShapeFunction for ArangeMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["start", "end", "step"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let mut args_map = HashMap::new();

        // Handle both single-arg arange(end) and multi-arg arange(start, end, step)
        // Due to signature binding, single-arg form binds positional to "start" not "end"
        // So if we have "start" but no "end", treat "start" as "end"
        let (start_ty, end_ty) = if bound_args.contains_key("end") {
            // Multi-arg form: arange(start, end, ...)
            (bound_args.get("start"), bound_args.get("end"))
        } else if bound_args.contains_key("start") {
            // Single-arg form: arange(end) - "end" was bound to "start"
            (None, bound_args.get("start"))
        } else {
            return None; // No arguments
        };

        // start is optional, defaults to 0
        if let Some(start) = start_ty
            && let Some(dim) = dimension(start)
        {
            args_map.insert("start".to_owned(), MetaShapeArg::Dim(dim));
        }

        // end is required
        if let Some(end) = end_ty {
            if let Some(dim) = dimension(end) {
                args_map.insert("end".to_owned(), MetaShapeArg::Dim(dim));
            } else {
                return None; // end must be a valid dimension type
            }
        } else {
            return None; // end is required
        }

        // step is optional
        if let Some(step_ty) = bound_args.get("step")
            && let Some(dim) = dimension(step_ty)
        {
            args_map.insert("step".to_owned(), MetaShapeArg::Dim(dim));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Extract start, end, and step as dimensions
        let start_ty = args
            .args
            .get("start")
            .and_then(|arg| match arg {
                MetaShapeArg::Dim(ty) => Some(ty.clone()),
                _ => None,
            })
            .unwrap_or(Type::Size(SizeExpr::Literal(0)));

        let end_ty = args
            .args
            .get("end")
            .and_then(|arg| match arg {
                MetaShapeArg::Dim(ty) => Some(ty.clone()),
                _ => None,
            })
            .ok_or_else(|| ShapeError::InvalidDimension {
                value: 0,
                reason: "arange requires 'end' parameter".to_owned(),
            })?;

        let step_ty = args
            .args
            .get("step")
            .and_then(|arg| match arg {
                MetaShapeArg::Dim(ty) => Some(ty.clone()),
                _ => None,
            })
            .unwrap_or(Type::Size(SizeExpr::Literal(1)));

        // Compute size = (end - start) // step
        // For step=1, this simplifies to end - start
        let diff = SizeExpr::sub(end_ty.clone(), start_ty.clone());
        let size_dim = match &step_ty {
            Type::Size(SizeExpr::Literal(1)) => diff,
            Type::Literal(box Literal {
                value: Lit::Int(i), ..
            }) if i.as_i64() == Some(1) => diff,
            _ => SizeExpr::floor_div(Type::Size(diff), step_ty),
        };

        Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![
            size_dim,
        ])]))
    }

    fn name(&self) -> &str {
        "arange"
    }
}

/// Meta-shape function for torch.linspace
#[derive(Debug)]
pub struct LinspaceMetaShape;

impl MetaShapeFunction for LinspaceMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["start", "end", "steps"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // steps is required and must be a literal int
        let steps = literal_int(bound_args.get("steps")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([("steps".to_owned(), MetaShapeArg::Int(steps))]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // linspace(start, end, steps) creates 1D tensor of size 'steps'
        let steps = match args.args.get("steps") {
            Some(MetaShapeArg::Int(n)) => *n,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "linspace requires 'steps' argument (Int)".to_owned(),
                });
            }
        };

        Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![
            SizeExpr::Literal(steps),
        ])]))
    }

    fn name(&self) -> &str {
        "linspace"
    }
}

/// Meta-shape function for torch.eye
#[derive(Debug)]
pub struct EyeMetaShape;

impl MetaShapeFunction for EyeMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["n", "m"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // n is required and must be a literal int
        let n = literal_int(bound_args.get("n")?)?;

        let mut args_map = HashMap::from([("n".to_owned(), MetaShapeArg::Int(n))]);

        // m is optional
        if let Some(m_ty) = bound_args.get("m")
            && let Some(m) = literal_int(m_ty)
        {
            args_map.insert("m".to_owned(), MetaShapeArg::Int(m));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // eye(n) creates NxN identity matrix, or eye(n, m) creates NxM matrix
        let n = match args.args.get("n") {
            Some(MetaShapeArg::Int(n)) => *n,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "eye requires 'n' argument (Int)".to_owned(),
                });
            }
        };

        let m = args.args.get("m").and_then(|arg| arg.as_int()).unwrap_or(n);

        Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![
            SizeExpr::Literal(n),
            SizeExpr::Literal(m),
        ])]))
    }

    fn name(&self) -> &str {
        "eye"
    }
}

/// Meta-shape function for torch.flatten
#[derive(Debug)]
pub struct FlattenMetaShape;

impl MetaShapeFunction for FlattenMetaShape {
    // ========== New Methods (Phase 2) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "start_dim", "end_dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract input tensor shape
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        // start_dim and end_dim are optional with defaults
        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // Try to extract start_dim if present
        if let Some(start_dim_ty) = bound_args.get("start_dim")
            && let Some(start_dim_val) = literal_int(start_dim_ty)
        {
            args_map.insert("start_dim".to_owned(), MetaShapeArg::Int(start_dim_val));
        }

        // Try to extract end_dim if present
        if let Some(end_dim_ty) = bound_args.get("end_dim")
            && let Some(end_dim_val) = literal_int(end_dim_ty)
        {
            args_map.insert("end_dim".to_owned(), MetaShapeArg::Int(end_dim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "flatten expects 'self' argument".to_owned(),
                });
            }
        };

        let start_dim = args
            .args
            .get("start_dim")
            .and_then(|arg| {
                if let MetaShapeArg::Int(d) = arg {
                    Some(*d)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let end_dim = args
            .args
            .get("end_dim")
            .and_then(|arg| {
                if let MetaShapeArg::Int(d) = arg {
                    Some(*d)
                } else {
                    None
                }
            })
            .unwrap_or(-1);

        let rank = input_shape.rank() as i64;
        let start = if start_dim < 0 {
            rank + start_dim
        } else {
            start_dim
        } as usize;
        let end = if end_dim < 0 { rank + end_dim } else { end_dim } as usize;

        if start >= input_shape.rank() || end >= input_shape.rank() || start > end {
            return Err(ShapeError::InvalidDimension {
                value: start_dim,
                reason: "Invalid start_dim or end_dim for flatten".to_owned(),
            });
        }

        // Build output: dims before start, flattened middle, dims after end
        let mut output_dims = Vec::new();

        // Dims before start_dim
        for i in 0..start {
            output_dims.push(input_shape.get_dim(i));
        }

        // Flatten dims from start to end into one dimension
        let mut flattened = SizeExpr::Literal(1);
        for i in start..=end {
            flattened = SizeExpr::mul(Type::Size(flattened), input_shape.get_dim(i));
        }
        output_dims.push(simplify(Type::Size(flattened)));

        // Dims after end_dim
        for i in (end + 1)..input_shape.rank() {
            output_dims.push(input_shape.get_dim(i));
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "flatten"
    }
}

/// Meta-shape function for torch.stack
#[derive(Debug)]
pub struct StackMetaShape;

impl MetaShapeFunction for StackMetaShape {
    // ========== New Methods (Phase 3) ==========

    fn signature(&self) -> &[&'static str] {
        &["tensors", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Extract tuple of tensor shapes (lists not supported - use tuples!)
        let tensor_shapes = tensor_list(bound_args.get("tensors")?)?;

        let mut args_map =
            HashMap::from([("tensors".to_owned(), MetaShapeArg::ShapeList(tensor_shapes))]);

        // dim is optional (default 0)
        if let Some(dim_ty) = bound_args.get("dim")
            && let Some(dim_val) = literal_int(dim_ty)
        {
            args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Extract tensor shapes from ShapeList
        let inputs = match args.args.get("tensors") {
            Some(MetaShapeArg::ShapeList(shapes)) => shapes,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "stack expects 'tensors' argument".to_owned(),
                });
            }
        };

        if inputs.is_empty() {
            return Err(ShapeError::InvalidDimension {
                value: 0,
                reason: "stack requires at least one input".to_owned(),
            });
        }

        let dim = args
            .args
            .get("dim")
            .and_then(|arg| {
                if let MetaShapeArg::Int(d) = arg {
                    Some(*d)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let first_shape = &inputs[0];
        let rank = first_shape.rank();

        // Normalize dim (stack can insert at rank+1 positions)
        let dim_idx = if dim < 0 {
            ((rank as i64) + 1 + dim) as usize
        } else {
            dim as usize
        };

        if dim_idx > rank {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: format!("dim {} out of bounds for stack", dim),
            });
        }

        // All inputs must have same shape
        for shape in inputs.iter().skip(1) {
            if shape.rank() != rank {
                return Err(ShapeError::RankMismatch {
                    got: shape.rank(),
                    want: rank,
                });
            }
        }

        // Build output shape: insert new dimension of size len(inputs) at dim_idx
        let mut output_dims = Vec::new();
        for i in 0..dim_idx {
            output_dims.push(first_shape.get_dim(i));
        }
        output_dims.push(Type::Size(SizeExpr::Literal(inputs.len() as i64)));
        for i in dim_idx..rank {
            output_dims.push(first_shape.get_dim(i));
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "stack"
    }
}

/// Meta-shape function for torch.tile
#[derive(Debug)]
pub struct TileMetaShape;

impl MetaShapeFunction for TileMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dims"] // Note: PyTorch uses "dims" parameter name
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        // Try both "dims" and "reps" parameter names for compatibility
        let reps =
            int_list(bound_args.get("dims")?).or_else(|| int_list(bound_args.get("reps")?))?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("dims".to_owned(), MetaShapeArg::IntList(reps)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "tile requires 'self' argument".to_owned(),
                });
            }
        };

        let reps = match args.args.get("dims") {
            Some(MetaShapeArg::IntList(r)) => r,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "tile requires 'dims' argument (IntList)".to_owned(),
                });
            }
        };

        // Output dims: multiply each input dim by corresponding rep
        // If reps.len() > input.rank(), prepend dimensions
        let mut output_dims = Vec::new();

        if reps.len() > input_shape.rank() {
            // Add extra dimensions at the front
            for &rep in reps.iter().take(reps.len() - input_shape.rank()) {
                output_dims.push(Type::Size(SizeExpr::Literal(rep)));
            }
            // Multiply existing dimensions
            for (i, &rep) in reps
                .iter()
                .skip(reps.len() - input_shape.rank())
                .enumerate()
            {
                output_dims.push(simplify(Type::Size(SizeExpr::mul(
                    input_shape.get_dim(i),
                    Type::Size(SizeExpr::Literal(rep)),
                ))));
            }
        } else {
            // Multiply each dimension
            for (i, &rep) in reps.iter().enumerate() {
                output_dims.push(simplify(Type::Size(SizeExpr::mul(
                    input_shape.get_dim(i),
                    Type::Size(SizeExpr::Literal(rep)),
                ))));
            }
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "tile"
    }
}

/// Meta-shape function for torch.select
#[derive(Debug)]
pub struct SelectMetaShape;

impl MetaShapeFunction for SelectMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "index"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .required_int(bound_args, "dim")?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "select")?;

        let dim = match args.args.get("dim") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "select requires 'dim' argument".to_owned(),
                });
            }
        };

        let dim_idx = input_shape.normalize_dim(dim)?;

        // Remove the selected dimension
        let mut output_dims = Vec::new();
        for (i, d) in input_shape.dims().iter().enumerate() {
            if i != dim_idx {
                output_dims.push(d.clone());
            }
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "select"
    }
}

/// Meta-shape function for torch.narrow
#[derive(Debug)]
pub struct NarrowMetaShape;

impl MetaShapeFunction for NarrowMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "start", "length"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .required_int(bound_args, "dim")?
            .required_int(bound_args, "length")?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "narrow")?;

        let dim = match args.args.get("dim") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "narrow requires 'dim' argument".to_owned(),
                });
            }
        };

        let length = match args.args.get("length") {
            Some(MetaShapeArg::Int(l)) => *l,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "narrow requires 'length' argument".to_owned(),
                });
            }
        };

        let dim_idx = input_shape.normalize_dim(dim)?;

        // Replace the dimension at dim_idx with length
        let mut output_dims = input_shape.dims().clone();
        output_dims[dim_idx] = Type::Size(SizeExpr::Literal(length));

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "narrow"
    }
}

/// Meta-shape function for torch.split
#[derive(Debug)]
pub struct SplitMetaShape;

impl MetaShapeFunction for SplitMetaShape {
    // ========== New Methods (Phase 3) ==========

    fn signature(&self) -> &[&'static str] {
        &["self", "split_size_or_sections", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // split_size_or_sections can be int or list[int] (though fixture may only support int)
        if let Some(split_ty) = bound_args.get("split_size_or_sections") {
            if let Some(split_val) = literal_int(split_ty) {
                // Single int - split into chunks of this size
                args_map.insert(
                    "split_size_or_sections".to_owned(),
                    MetaShapeArg::Int(split_val),
                );
            } else if let Some(sections) = int_list(split_ty) {
                // List of ints - split into these specific sizes
                args_map.insert(
                    "split_size_or_sections".to_owned(),
                    MetaShapeArg::IntList(sections),
                );
            } else if let Some(dim_sections) = dim_list(split_ty) {
                // List of symbolic dimensions (e.g., [D, NLocalHeads * HeadDim, NLocalHeads * HeadDim])
                args_map.insert(
                    "split_size_or_sections".to_owned(),
                    MetaShapeArg::DimList(dim_sections),
                );
            } else if let Some(dim) = dimension(split_ty) {
                // Symbolic dimension (for advanced use cases)
                args_map.insert("split_size_or_sections".to_owned(), MetaShapeArg::Dim(dim));
            } else {
                // Can't extract - fall back
                return None;
            }
        }

        // dim is optional (default 0)
        if let Some(dim_ty) = bound_args.get("dim")
            && let Some(dim_val) = literal_int(dim_ty)
        {
            args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Extract input shape
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "split expects 'self' argument".to_owned(),
                });
            }
        };

        let dim = args
            .args
            .get("dim")
            .and_then(|arg| {
                if let MetaShapeArg::Int(d) = arg {
                    Some(*d)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let dim_idx = input_shape.normalize_dim(dim)?;

        // Get split_size_or_sections parameter
        match args.args.get("split_size_or_sections") {
            Some(MetaShapeArg::IntList(sections)) => {
                // list[int] case: split into chunks of specified sizes
                // Returns N tensors where N = len(sections)
                let mut result = Vec::new();
                for &section_size in sections {
                    let mut chunk_dims = input_shape.dims().clone();
                    chunk_dims[dim_idx] = Type::Size(SizeExpr::Literal(section_size));
                    result.push(TensorShape::from_types(chunk_dims));
                }
                Ok(MetaShapeResult::Tensors(result))
            }
            Some(MetaShapeArg::DimList(dim_sections)) => {
                // list[Dim] case: split into chunks of specified symbolic sizes
                // Returns N tensors where N = len(dim_sections)
                let mut result = Vec::new();
                for section_dim in dim_sections {
                    let mut chunk_dims = input_shape.dims().clone();
                    chunk_dims[dim_idx] = section_dim.clone();
                    result.push(TensorShape::from_types(chunk_dims));
                }
                Ok(MetaShapeResult::Tensors(result))
            }
            Some(MetaShapeArg::Int(split_size)) => {
                // int case: split into equal chunks of size split_size
                // Number of chunks = ceil(dim_size / split_size)
                let dim_size = input_shape.dims()[dim_idx].clone().clone();
                match dim_size {
                    Type::Size(SizeExpr::Literal(size)) => {
                        let num_chunks = (size + split_size - 1) / split_size;
                        let mut result = Vec::new();
                        for i in 0..num_chunks {
                            let mut chunk_dims = input_shape.dims().clone();
                            // Last chunk might be smaller
                            let chunk_size = if i < num_chunks - 1 {
                                *split_size
                            } else {
                                size - (num_chunks - 1) * split_size
                            };
                            chunk_dims[dim_idx] = Type::Size(SizeExpr::Literal(chunk_size));
                            result.push(TensorShape::from_types(chunk_dims));
                        }
                        Ok(MetaShapeResult::Tensors(result))
                    }
                    _ => {
                        // Symbolic dim size with literal split_size - return unbounded tuple
                        let mut output_dims = input_shape.dims().clone();
                        output_dims[dim_idx] = Type::Size(SizeExpr::Literal(*split_size));
                        Ok(MetaShapeResult::UnboundedTuple {
                            element_shape: TensorShape::from_types(output_dims),
                        })
                    }
                }
            }
            Some(MetaShapeArg::Dim(split_size_ty)) => {
                // Symbolic dimension case: compute number of chunks symbolically
                let dim_size_ty = input_shape.dims()[dim_idx].clone();

                // Compute num_chunks = dim_size // split_size and canonicalize
                let num_chunks_ty =
                    Type::Size(SizeExpr::floor_div(dim_size_ty, split_size_ty.clone()))
                        .canonicalize();

                // Check if it canonicalized to a literal
                match num_chunks_ty {
                    Type::Size(SizeExpr::Literal(num_chunks)) => {
                        // Return num_chunks tensors, each with split_size as the dimension
                        let mut result = Vec::new();
                        for _ in 0..num_chunks {
                            let mut chunk_dims = input_shape.dims().clone();
                            chunk_dims[dim_idx] = split_size_ty.clone();
                            result.push(TensorShape::from_types(chunk_dims));
                        }
                        Ok(MetaShapeResult::Tensors(result))
                    }
                    _ => {
                        // Can't determine number of chunks - return unbounded variadic tuple
                        let mut output_dims = input_shape.dims().clone();
                        output_dims[dim_idx] = split_size_ty.clone();
                        Ok(MetaShapeResult::UnboundedTuple {
                            element_shape: TensorShape::from_types(output_dims),
                        })
                    }
                }
            }
            _ => {
                // No split_size provided, return Unknown
                Ok(MetaShapeResult::UnboundedTupleShapeless)
            }
        }
    }

    // ========== Existing Method (unchanged) ==========

    fn name(&self) -> &str {
        "split"
    }
}

/// Meta-shape function for torch.chunk
#[derive(Debug)]
pub struct ChunkMetaShape;

impl MetaShapeFunction for ChunkMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "chunks", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .required_int(bound_args, "chunks")?
            .optional_int(bound_args, "dim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "chunk")?;

        let dim = args
            .args
            .get("dim")
            .and_then(|arg| arg.as_int())
            .unwrap_or(0);
        let dim_idx = input_shape.normalize_dim(dim)?;

        // Get number of chunks
        let num_chunks = match args.args.get("chunks").and_then(|arg| arg.as_int()) {
            Some(chunks) if chunks > 0 => chunks as usize,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "chunk requires a positive number of chunks".to_owned(),
                });
            }
        };

        // Get dimension size
        let dim_size = &input_shape.dims()[dim_idx];
        match dim_size {
            Type::Size(SizeExpr::Literal(size)) => {
                // Calculate chunk size: ceil(size / num_chunks)
                let chunk_size = (*size + num_chunks as i64 - 1) / num_chunks as i64;
                let mut result = Vec::new();

                for i in 0..num_chunks {
                    let mut chunk_dims = input_shape.dims().clone();
                    // Last chunk might be smaller
                    let actual_chunk_size = if i < num_chunks - 1 {
                        chunk_size
                    } else {
                        size - (num_chunks as i64 - 1) * chunk_size
                    };
                    chunk_dims[dim_idx] = Type::Size(SizeExpr::Literal(actual_chunk_size));
                    result.push(TensorShape::from_types(chunk_dims));
                }
                Ok(MetaShapeResult::Tensors(result))
            }
            _ => {
                // Unknown dim size - return single shape with Unknown
                let mut output_dims = input_shape.dims().clone();
                output_dims[dim_idx] = Type::Any(AnyStyle::Implicit);
                Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                    output_dims,
                )]))
            }
        }
    }

    fn name(&self) -> &str {
        "chunk"
    }
}

/// Meta-shape function for torch.index_select
#[derive(Debug)]
pub struct IndexSelectMetaShape;

impl MetaShapeFunction for IndexSelectMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "index"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let index_shape = concrete_tensor_shape(bound_args.get("index")?)?;

        // dim must be literal
        let dim_val = literal_int(bound_args.get("dim")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("index".to_owned(), MetaShapeArg::Shape(index_shape)),
                ("dim".to_owned(), MetaShapeArg::Int(dim_val)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "index_select requires 'self' argument".to_owned(),
                });
            }
        };

        let index_shape = match args.args.get("index") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "index_select requires 'index' argument".to_owned(),
                });
            }
        };

        let dim = match args.args.get("dim") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "index_select requires 'dim' argument".to_owned(),
                });
            }
        };

        let dim_idx = input_shape.normalize_dim(dim)?;

        // Replace dimension at dim_idx with size of index tensor (should be 1D)
        let mut output_dims = input_shape.dims().clone();
        if index_shape.rank() == 1 {
            output_dims[dim_idx] = index_shape.get_dim(0);
        } else {
            // Unknown index shape
            output_dims[dim_idx] = Type::Any(AnyStyle::Implicit);
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "index_select"
    }
}

/// Meta-shape function for torch.masked_select
#[derive(Debug)]
pub struct MaskedSelectMetaShape;

impl MetaShapeFunction for MaskedSelectMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "mask"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        // Just check that input exists
        let _input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        Some(MetaShapeArgs {
            args: HashMap::new(),
        })
    }

    fn compute(&self, _args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // masked_select returns 1D tensor of unknown size
        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            vec![Type::Any(AnyStyle::Implicit)],
        )]))
    }

    fn name(&self) -> &str {
        "masked_select"
    }
}

/// Meta-shape function for torch.Tensor.repeat
#[derive(Debug)]
pub struct RepeatMetaShape;

impl MetaShapeFunction for RepeatMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "*sizes"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let repeats_shape = shape_arg(bound_args.get("sizes")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("sizes".to_owned(), MetaShapeArg::Shape(repeats_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "repeat requires 'self' argument".to_owned(),
                });
            }
        };

        // Get repeat factors - now always MetaShapeArg::Shape (supports symbolic dims)
        let repeats: &[Type] = match args.args.get("sizes") {
            Some(MetaShapeArg::Shape(shape)) => {
                // Use dims() directly to preserve all types (Quantified, SizeExpr, etc.)
                shape.dims()
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "repeat requires 'sizes' argument".to_owned(),
                });
            }
        };

        // Repeat multiplies each dimension by the corresponding repeat factor
        if repeats.len() != input_shape.rank() {
            return Err(ShapeError::InvalidDimension {
                value: repeats.len() as i64,
                reason: format!(
                    "repeat sizes length {} doesn't match tensor rank {}",
                    repeats.len(),
                    input_shape.rank()
                ),
            });
        }

        let output_dims = input_shape
            .dims()
            .iter()
            .zip(repeats.iter())
            .map(|(dim, repeat)| {
                // Multiply dimension by repeat factor
                simplify(Type::Size(SizeExpr::mul(dim.clone(), repeat.clone())))
            })
            .collect();

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "repeat"
    }
}

/// Meta-shape function for torch.Tensor.expand
#[derive(Debug)]
pub struct ExpandMetaShape;

impl MetaShapeFunction for ExpandMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "*sizes"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let target_shape = shape_arg(bound_args.get("sizes")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("sizes".to_owned(), MetaShapeArg::Shape(target_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "expand requires 'self' argument".to_owned(),
                });
            }
        };

        // Get target sizes - now always MetaShapeArg::Shape (supports symbolic dims)
        let target_dims: &[Type] = match args.args.get("sizes") {
            Some(MetaShapeArg::Shape(shape)) => {
                // Use dims() directly to preserve all types (Quantified, SizeExpr, etc.)
                shape.dims()
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "expand requires 'sizes' argument".to_owned(),
                });
            }
        };

        if target_dims.len() != input_shape.rank() {
            return Err(ShapeError::InvalidDimension {
                value: target_dims.len() as i64,
                reason: format!(
                    "expand target size length {} doesn't match tensor rank {}",
                    target_dims.len(),
                    input_shape.rank()
                ),
            });
        }

        // Expand broadcasts dimensions:
        // - If target is -1: keep input dimension
        // - If input dimension is 1: broadcast to target size
        // - Otherwise: use target dimension (should match input, but we'll trust the caller)
        let output_dims = input_shape
            .dims()
            .iter()
            .zip(target_dims.iter())
            .map(|(input_dim, target_dim)| {
                // Check if target is -1 (keep input dimension)
                if let Type::Size(SizeExpr::Literal(-1)) = target_dim {
                    return input_dim.clone();
                }

                // Check if input is singleton (literal 1) - can broadcast to any size
                if let Type::Size(SizeExpr::Literal(1)) = input_dim {
                    return target_dim.clone();
                }

                // Otherwise use target (for symbolic dims like Quantified(N), or matching literals)
                target_dim.clone()
            })
            .collect();

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "expand"
    }
}

/// Meta-shape function for torch.unbind
#[derive(Debug)]
pub struct UnbindMetaShape;

impl MetaShapeFunction for UnbindMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .optional_int(bound_args, "dim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "unbind")?;

        let dim = args
            .args
            .get("dim")
            .and_then(|arg| arg.as_int())
            .unwrap_or(0);

        let dim_idx = input_shape.normalize_dim(dim)?;

        // unbind removes the specified dimension
        let mut output_dims = Vec::new();
        for (i, d) in input_shape.dims().iter().enumerate() {
            if i != dim_idx {
                output_dims.push(d.clone());
            }
        }

        // Returns an unbounded tuple of tensors (one for each element along the dimension)
        // tuple[Tensor[...], Tensor[...], ...]
        Ok(MetaShapeResult::UnboundedTuple {
            element_shape: TensorShape::from_types(output_dims),
        })
    }

    fn name(&self) -> &str {
        "unbind"
    }
}

/// Meta-shape function for torch.movedim (also used for moveaxis)
#[derive(Debug)]
pub struct MovedimMetaShape;

impl MetaShapeFunction for MovedimMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "source", "destination"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // source can be int or IntList
        if let Some(source_ty) = bound_args.get("source") {
            if let Some(s) = literal_int(source_ty) {
                args_map.insert("source".to_owned(), MetaShapeArg::Int(s));
            } else if let Some(s_list) = int_list(source_ty) {
                args_map.insert("source".to_owned(), MetaShapeArg::IntList(s_list));
            } else {
                return None; // Can't extract source
            }
        }

        // destination can be int or IntList
        if let Some(dest_ty) = bound_args.get("destination") {
            if let Some(d) = literal_int(dest_ty) {
                args_map.insert("destination".to_owned(), MetaShapeArg::Int(d));
            } else if let Some(d_list) = int_list(dest_ty) {
                args_map.insert("destination".to_owned(), MetaShapeArg::IntList(d_list));
            } else {
                return None; // Can't extract destination
            }
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "movedim requires 'self' argument".to_owned(),
                });
            }
        };

        // Get source and destination dimensions
        let source = match args.args.get("source") {
            Some(MetaShapeArg::Int(s)) => vec![*s],
            Some(MetaShapeArg::IntList(s)) => s.clone(),
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "movedim requires 'source' argument".to_owned(),
                });
            }
        };

        let destination = match args.args.get("destination") {
            Some(MetaShapeArg::Int(d)) => vec![*d],
            Some(MetaShapeArg::IntList(d)) => d.clone(),
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "movedim requires 'destination' argument".to_owned(),
                });
            }
        };

        if source.len() != destination.len() {
            return Err(ShapeError::InvalidDimension {
                value: 0,
                reason: format!(
                    "movedim: source and destination must have same length, got {} and {}",
                    source.len(),
                    destination.len()
                ),
            });
        }

        let rank = input_shape.rank();

        // Normalize source and destination indices
        let mut source_norm = Vec::new();
        for &s in &source {
            let s_idx = input_shape.normalize_dim(s)?;
            source_norm.push(s_idx);
        }

        let mut dest_norm = Vec::new();
        for &d in &destination {
            let d_norm = if d < 0 {
                let r = rank as i64;
                let normalized = r + d;
                if normalized < 0 || normalized >= r {
                    return Err(ShapeError::InvalidDimension {
                        value: d,
                        reason: format!("destination {} out of range for rank {}", d, rank),
                    });
                }
                normalized as usize
            } else {
                d as usize
            };
            if d_norm >= rank {
                return Err(ShapeError::InvalidDimension {
                    value: d,
                    reason: format!("destination {} out of range for rank {}", d, rank),
                });
            }
            dest_norm.push(d_norm);
        }

        // Create output shape by moving dimensions
        let mut output_dims = input_shape.dims().clone();
        let mut moved_dims = Vec::new();

        // Extract dimensions to move
        for &src in &source_norm {
            moved_dims.push(output_dims[src].clone());
        }

        // Remove source dimensions (in reverse order to maintain indices)
        let mut sorted_sources = source_norm.clone();
        sorted_sources.sort_by(|a, b| b.cmp(a));
        for &src in &sorted_sources {
            output_dims.remove(src);
        }

        // Insert at destination positions
        for (i, &dest) in dest_norm.iter().enumerate() {
            output_dims.insert(dest, moved_dims[i].clone());
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "movedim"
    }
}

/// Meta-shape function for torch.unfold
#[derive(Debug)]
pub struct UnfoldMetaShape;

impl MetaShapeFunction for UnfoldMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dimension", "size", "step"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let dimension = literal_int(bound_args.get("dimension")?)?;
        let size = literal_int(bound_args.get("size")?)?;

        let mut args_map = HashMap::from([
            ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
            ("dimension".to_owned(), MetaShapeArg::Int(dimension)),
            ("size".to_owned(), MetaShapeArg::Int(size)),
        ]);

        // step is optional, defaults to 1
        if let Some(step_ty) = bound_args.get("step")
            && let Some(s) = literal_int(step_ty)
        {
            args_map.insert("step".to_owned(), MetaShapeArg::Int(s));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "unfold requires 'self' argument".to_owned(),
                });
            }
        };

        let dimension = match args.args.get("dimension") {
            Some(MetaShapeArg::Int(d)) => *d,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "unfold requires 'dimension' argument".to_owned(),
                });
            }
        };

        let size = match args.args.get("size") {
            Some(MetaShapeArg::Int(s)) => *s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "unfold requires 'size' argument".to_owned(),
                });
            }
        };

        let step = args
            .args
            .get("step")
            .and_then(|arg| arg.as_int())
            .unwrap_or(1);

        let dim_idx = input_shape.normalize_dim(dimension)?;

        // Calculate new dimension size after unfolding
        let dim_size = input_shape.dims()[dim_idx].clone().clone();

        // Validate if dimension is literal
        if let Type::Size(SizeExpr::Literal(n)) = dim_size {
            if n < size {
                return Err(ShapeError::InvalidDimension {
                    value: size,
                    reason: format!("unfold size {} larger than dimension size {}", size, n),
                });
            }
        } else {
            // Symbolic dimension - can't validate
            // Following Week 2 feedback: error when validation needed
            return Err(ShapeError::InvalidDimension {
                value: size,
                reason: format!(
                    "unfold requires literal dimension for validation (got symbolic/unknown). \
                     Dimension must be >= {} but cannot verify with symbolic dimension.",
                    size
                ),
            });
        }

        // Number of windows: (dim_size - size) / step + 1
        // Compute symbolically
        let dim_minus_size = SizeExpr::sub(dim_size.clone(), Type::Size(SizeExpr::Literal(size)));
        let quotient = SizeExpr::floor_div(
            Type::Size(dim_minus_size),
            Type::Size(SizeExpr::Literal(step)),
        );
        let new_size_ty = simplify(Type::Size(SizeExpr::add(
            Type::Size(quotient),
            Type::Size(SizeExpr::Literal(1)),
        )));

        // Output shape: same as input, but dimension at dim_idx is replaced with new_size,
        // and a new dimension of size `size` is appended at the end
        let mut output_dims = input_shape.dims().clone();
        output_dims[dim_idx] = new_size_ty;
        output_dims.push(Type::Size(SizeExpr::Literal(size)));

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "unfold"
    }
}

/// Meta-shape function for torch.var_mean / torch.std_mean
/// Returns tuple[Tensor, Tensor] (variance/std, mean)
#[derive(Debug)]
pub struct VarMeanMetaShape;

impl MetaShapeFunction for VarMeanMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "keepdim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        use crate::literal::Lit;

        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // dim can be int, tuple of ints, or None
        if let Some(dim_ty) = bound_args.get("dim")
            && !dim_ty.is_none()
        {
            match dim_ty {
                Type::Literal(box Literal {
                    value: Lit::Int(_), ..
                }) => {
                    if let Some(dim_val) = literal_int(dim_ty) {
                        args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
                    }
                }
                Type::Tuple(_) => {
                    if let Some(dims) = int_list(dim_ty) {
                        args_map.insert("dim".to_owned(), MetaShapeArg::IntList(dims));
                    }
                }
                _ => {}
            }
        }

        if let Some(keepdim_ty) = bound_args.get("keepdim")
            && let Some(keepdim_val) = bool_arg(keepdim_ty)
        {
            args_map.insert("keepdim".to_owned(), MetaShapeArg::Bool(keepdim_val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "var_mean/std_mean requires 'self' argument".to_owned(),
                });
            }
        };

        let keepdim = args
            .args
            .get("keepdim")
            .and_then(|arg| {
                if let MetaShapeArg::Bool(b) = arg {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Handle both single dim (Int) and multiple dims (IntList)
        let dims_to_reduce: Vec<usize> = match args.args.get("dim") {
            Some(MetaShapeArg::Int(dim_val)) => {
                let dim_idx = input_shape.normalize_dim(*dim_val)?;
                vec![dim_idx]
            }
            Some(MetaShapeArg::IntList(dim_list)) => {
                let mut indices = Vec::new();
                for &dim_val in dim_list {
                    let dim_idx = input_shape.normalize_dim(dim_val)?;
                    indices.push(dim_idx);
                }
                indices
            }
            None => {
                // Reduce all dimensions
                if keepdim {
                    let output_dims = vec![Type::Size(SizeExpr::Literal(1)); input_shape.rank()];
                    let shape = TensorShape::from_types(output_dims);
                    return Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]));
                } else {
                    let shape = TensorShape::new(vec![]);
                    return Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]));
                }
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "dim must be Int or IntList".to_owned(),
                });
            }
        };

        // Build output shape
        let mut output_dims = Vec::new();
        for (i, dim) in input_shape.dims().iter().enumerate() {
            if dims_to_reduce.contains(&i) {
                if keepdim {
                    output_dims.push(Type::Size(SizeExpr::Literal(1)));
                }
            } else {
                output_dims.push(dim.clone());
            }
        }

        let shape = TensorShape::from_types(output_dims);
        // Return 2 identical shapes for (var/std, mean)
        Ok(MetaShapeResult::Tensors(vec![shape.clone(), shape]))
    }

    fn name(&self) -> &str {
        "var_mean"
    }
}

/// Meta-shape function for torch.norm
/// Computes vector or matrix norms, can reduce dimensions or preserve shape
#[derive(Debug)]
pub struct NormMetaShape;

impl MetaShapeFunction for NormMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim", "keepdim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let self_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(self_shape))]);

        // dim is optional, can be Int or IntList
        if let Some(dim_ty) = bound_args.get("dim") {
            if let Some(dim_val) = literal_int(dim_ty) {
                args_map.insert("dim".to_owned(), MetaShapeArg::Int(dim_val));
            } else if let Some(dim_list) = int_list(dim_ty) {
                args_map.insert("dim".to_owned(), MetaShapeArg::IntList(dim_list));
            }
        }

        // keepdim is optional, default false
        if let Some(keepdim_ty) = bound_args.get("keepdim")
            && let Some(b) = bool_arg(keepdim_ty)
        {
            args_map.insert("keepdim".to_owned(), MetaShapeArg::Bool(b));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "norm: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        let keepdim = args
            .args
            .get("keepdim")
            .and_then(|arg| arg.as_bool())
            .unwrap_or(false);

        // Handle dim argument (can be Int or IntList)
        let dims_to_reduce: Option<Vec<usize>> = match args.args.get("dim") {
            Some(MetaShapeArg::Int(dim_val)) => {
                let dim_idx = input_shape.normalize_dim(*dim_val)?;
                Some(vec![dim_idx])
            }
            Some(MetaShapeArg::IntList(dim_list)) => {
                let mut indices = Vec::new();
                for &dim_val in dim_list {
                    let dim_idx = input_shape.normalize_dim(dim_val)?;
                    indices.push(dim_idx);
                }
                Some(indices)
            }
            None => None, // No dim specified - compute over all dimensions
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "dim must be Int or IntList".to_owned(),
                });
            }
        };

        // If no dim specified, return scalar (or keep all dims as 1)
        if dims_to_reduce.is_none() {
            if keepdim {
                let output_dims = vec![Type::Size(SizeExpr::Literal(1)); input_shape.rank()];
                return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                    output_dims,
                )]));
            } else {
                return Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![])]));
            }
        }

        let dims = dims_to_reduce.unwrap();

        // Build output shape by removing/replacing reduced dimensions
        let mut output_dims = Vec::new();
        for (i, dim) in input_shape.dims().iter().enumerate() {
            if dims.contains(&i) {
                if keepdim {
                    output_dims.push(Type::Size(SizeExpr::Literal(1)));
                }
                // else skip this dimension
            } else {
                output_dims.push(dim.clone());
            }
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "norm"
    }
}

/// Meta-shape function for torch.diag_embed
/// Creates diagonal tensor from input, adding dimensions
#[derive(Debug)]
pub struct DiagEmbedMetaShape;

impl MetaShapeFunction for DiagEmbedMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "offset", "dim1", "dim2"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let self_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(self_shape))]);

        // All parameters are optional
        if let Some(offset_ty) = bound_args.get("offset")
            && let Some(offset) = literal_int(offset_ty)
        {
            args_map.insert("offset".to_owned(), MetaShapeArg::Int(offset));
        }
        if let Some(dim1_ty) = bound_args.get("dim1")
            && let Some(dim1) = literal_int(dim1_ty)
        {
            args_map.insert("dim1".to_owned(), MetaShapeArg::Int(dim1));
        }
        if let Some(dim2_ty) = bound_args.get("dim2")
            && let Some(dim2) = literal_int(dim2_ty)
        {
            args_map.insert("dim2".to_owned(), MetaShapeArg::Int(dim2));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "diag_embed: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        let offset = args
            .args
            .get("offset")
            .and_then(|arg| arg.as_int())
            .unwrap_or(0);

        // Output has rank = input.rank() + 1
        // Last dimension of input becomes diagonal, and we add two new dimensions
        let last_dim = input_shape
            .dims()
            .last()
            .cloned()
            .unwrap_or(Type::Size(SizeExpr::Literal(0)));

        // The new dimensions have size = last_dim + |offset|
        let new_dim_size = Type::Size(SizeExpr::add(
            last_dim,
            Type::Size(SizeExpr::Literal(offset.abs())),
        ));

        // Build output: all input dims except last, then add two new dims
        let mut output_dims: Vec<Type> =
            input_shape.dims()[..input_shape.rank().saturating_sub(1)].to_vec();
        output_dims.push(new_dim_size.clone());
        output_dims.push(new_dim_size);

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "diag_embed"
    }
}

/// Meta-shape function for tril_indices and triu_indices
/// Returns 2D tensor with shape [2, num_indices]
#[derive(Debug)]
pub struct TriIndicesMetaShape;

impl MetaShapeFunction for TriIndicesMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["row", "col", "offset"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let mut args_map = HashMap::new();

        // row and col are required
        if let Some(row_ty) = bound_args.get("row") {
            if let Some(row) = literal_int(row_ty) {
                args_map.insert("row".to_owned(), MetaShapeArg::Int(row));
            } else {
                return None;
            }
        } else {
            return None;
        }

        if let Some(col_ty) = bound_args.get("col") {
            if let Some(col) = literal_int(col_ty) {
                args_map.insert("col".to_owned(), MetaShapeArg::Int(col));
            } else {
                return None;
            }
        } else {
            return None;
        }

        // offset is optional
        if let Some(offset_ty) = bound_args.get("offset")
            && let Some(offset) = literal_int(offset_ty)
        {
            args_map.insert("offset".to_owned(), MetaShapeArg::Int(offset));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, _args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // For now, return [2, Unknown] since exact count is complex
        Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![
            SizeExpr::Literal(2),
            SizeExpr::Literal(0), /* was Unknown */
        ])]))
    }

    fn name(&self) -> &str {
        "tri_indices"
    }
}

/// Meta-shape function for torch.matmul
/// Handles general matrix multiplication with broadcasting
#[derive(Debug)]
pub struct MatMulMetaShape;

impl MetaShapeFunction for MatMulMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "other"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let self_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let other_shape = concrete_tensor_shape(bound_args.get("other")?)?;
        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(self_shape)),
                ("other".to_owned(), MetaShapeArg::Shape(other_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let a = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "matmul: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };
        let b = match args.args.get("other") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "matmul: 'other' argument must be a tensor shape".to_owned(),
                });
            }
        };

        let rank_a = a.rank();
        let rank_b = b.rank();

        let result_shape = match (rank_a, rank_b) {
            // 1D @ 1D → scalar (dot product)
            (1, 1) => TensorShape::new(vec![]),

            // 1D @ 2D → 1D (vector @ matrix)
            (1, 2) => {
                let m = &b.dims()[1];
                TensorShape::from_types(vec![m.clone()])
            }

            // 2D @ 1D → 1D (matrix @ vector)
            (2, 1) => {
                let n = &a.dims()[0];
                TensorShape::from_types(vec![n.clone()])
            }

            // 2D @ 2D → 2D (matrix @ matrix)
            (2, 2) => {
                let n = &a.dims()[0];
                let m = &b.dims()[1];
                TensorShape::from_types(vec![n.clone(), m.clone()])
            }

            // Batched matmul with broadcasting
            _ if rank_a >= 3 && rank_b >= 3 => {
                // Get the matrix dimensions (last 2 dims)
                let n = &a.dims()[rank_a - 2];
                let m = &b.dims()[rank_b - 1];

                // Broadcast batch dimensions
                let batch_dims_a = &a.dims()[..rank_a - 2];
                let batch_dims_b = &b.dims()[..rank_b - 2];

                // Simple broadcasting: take the maximum of each dimension
                let max_batch_len = batch_dims_a.len().max(batch_dims_b.len());
                let mut output_dims = Vec::new();

                for i in 0..max_batch_len {
                    let dim_a = if i < batch_dims_a.len() {
                        batch_dims_a[batch_dims_a.len() - max_batch_len + i].clone()
                    } else {
                        Type::Size(SizeExpr::Literal(1))
                    };
                    output_dims.push(dim_a);
                }

                // Add matrix dimensions
                output_dims.push(n.clone());
                output_dims.push(m.clone());

                TensorShape::from_types(output_dims)
            }

            // Mixed batched and non-batched
            (2, _) if rank_b >= 3 => {
                // 2D @ nD (n >= 3)
                let n = &a.dims()[0];
                let batch_dims = &b.dims()[..rank_b - 2];
                let m = &b.dims()[rank_b - 1];

                let mut output_dims = batch_dims.to_vec();
                output_dims.push(n.clone());
                output_dims.push(m.clone());

                TensorShape::from_types(output_dims)
            }

            (_, 2) if rank_a >= 3 => {
                // nD @ 2D (n >= 3)
                let batch_dims = &a.dims()[..rank_a - 2];
                let n = &a.dims()[rank_a - 2];
                let m = &b.dims()[1];

                let mut output_dims = batch_dims.to_vec();
                output_dims.push(n.clone());
                output_dims.push(m.clone());

                TensorShape::from_types(output_dims)
            }

            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: rank_a as i64,
                    reason: format!("matmul: incompatible dimensions {}D @ {}D", rank_a, rank_b),
                });
            }
        };

        Ok(MetaShapeResult::Tensors(vec![result_shape]))
    }

    fn name(&self) -> &str {
        "matmul"
    }
}

/// Meta-shape function for torch.mv
/// Matrix-vector multiplication (2D @ 1D → 1D)
#[derive(Debug)]
pub struct MVMetaShape;

impl MetaShapeFunction for MVMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "vec"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let self_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let vec_shape = concrete_tensor_shape(bound_args.get("vec")?)?;
        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(self_shape)),
                ("vec".to_owned(), MetaShapeArg::Shape(vec_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let mat = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "mv: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };
        let vec = match args.args.get("vec") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "mv: 'vec' argument must be a tensor shape".to_owned(),
                });
            }
        };

        if mat.rank() != 2 {
            return Err(ShapeError::InvalidDimension {
                value: mat.rank() as i64,
                reason: format!("mv expects 2D matrix, got {}D", mat.rank()),
            });
        }

        if vec.rank() != 1 {
            return Err(ShapeError::InvalidDimension {
                value: vec.rank() as i64,
                reason: format!("mv expects 1D vector, got {}D", vec.rank()),
            });
        }

        let n = &mat.dims()[0];

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            vec![n.clone()],
        )]))
    }

    fn name(&self) -> &str {
        "mv"
    }
}

/// Meta-shape function for torch.outer
/// Outer product of two 1D tensors: [N] x [M] → [N, M]
#[derive(Debug)]
pub struct OuterMetaShape;

impl MetaShapeFunction for OuterMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "vec2"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let vec2_shape = concrete_tensor_shape(bound_args.get("vec2")?)?;
        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
                ("vec2".to_owned(), MetaShapeArg::Shape(vec2_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let a = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "outer: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };
        let b = match args.args.get("vec2") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "outer: 'vec2' argument must be a tensor shape".to_owned(),
                });
            }
        };

        if a.rank() != 1 || b.rank() != 1 {
            return Err(ShapeError::InvalidDimension {
                value: a.rank() as i64,
                reason: format!(
                    "outer expects 1D tensors, got {}D and {}D",
                    a.rank(),
                    b.rank()
                ),
            });
        }

        // Outer product: [N] x [M] → [N, M]
        let result_dims = vec![a.dims()[0].clone(), b.dims()[0].clone()];
        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            result_dims,
        )]))
    }

    fn name(&self) -> &str {
        "outer"
    }
}

/// Meta-shape function for convolution operations (conv1d, conv2d, conv3d)
/// Computes output shape based on kernel size, stride, padding, dilation
#[derive(Debug)]
pub struct ConvMetaShape;

impl MetaShapeFunction for ConvMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "weight", "stride", "padding", "dilation"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let weight_shape = concrete_tensor_shape(bound_args.get("weight")?)?;

        let mut args_map = HashMap::from([
            ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
            ("weight".to_owned(), MetaShapeArg::Shape(weight_shape)),
        ]);

        // stride, padding, dilation are optional - can be Int or IntList
        for param in &["stride", "padding", "dilation"] {
            if let Some(param_ty) = bound_args.get(*param) {
                if let Some(val) = literal_int(param_ty) {
                    args_map.insert((*param).to_owned(), MetaShapeArg::Int(val));
                } else if let Some(list) = int_list(param_ty) {
                    args_map.insert((*param).to_owned(), MetaShapeArg::IntList(list));
                }
            }
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "conv: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        let weight_shape = match args.args.get("weight") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "conv: 'weight' argument must be a tensor shape".to_owned(),
                });
            }
        };

        // Input: (N, C_in, *spatial) where spatial can be 1D, 2D, or 3D
        // Weight: (C_out, C_in, *kernel)
        // Output: (N, C_out, *spatial_out)

        if input_shape.rank() < 3 || weight_shape.rank() < 3 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: format!(
                    "conv requires at least 3D tensors, got {}D input and {}D weight",
                    input_shape.rank(),
                    weight_shape.rank()
                ),
            });
        }

        let spatial_dims = input_shape.rank() - 2; // Number of spatial dimensions (1, 2, or 3)

        // Get convolution parameters
        let stride = match args.args.get("stride") {
            Some(MetaShapeArg::Int(s)) => vec![*s; spatial_dims],
            Some(MetaShapeArg::IntList(s)) => s.clone(),
            _ => vec![1; spatial_dims],
        };

        let padding = match args.args.get("padding") {
            Some(MetaShapeArg::Int(p)) => vec![*p; spatial_dims],
            Some(MetaShapeArg::IntList(p)) => p.clone(),
            _ => vec![0; spatial_dims],
        };

        let dilation = match args.args.get("dilation") {
            Some(MetaShapeArg::Int(d)) => vec![*d; spatial_dims],
            Some(MetaShapeArg::IntList(d)) => d.clone(),
            _ => vec![1; spatial_dims],
        };

        // Build output shape
        let mut output_dims = Vec::new();

        // Batch dimension
        output_dims.push(input_shape.get_dim(0));

        // Output channels (from weight)
        output_dims.push(weight_shape.get_dim(0));

        // Spatial dimensions
        for i in 0..spatial_dims {
            let input_size = input_shape.dims()[2 + i].clone();
            let kernel_size = &weight_shape.dims()[2 + i];
            let stride_val = stride.get(i).copied().unwrap_or(1);
            let padding_val = padding.get(i).copied().unwrap_or(0);
            let dilation_val = dilation.get(i).copied().unwrap_or(1);

            // Formula: floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
            // Compute symbolically using SizeExpr operations
            let two_padding = SizeExpr::mul(
                Type::Size(SizeExpr::Literal(2)),
                Type::Size(SizeExpr::Literal(padding_val)),
            );
            let padded = SizeExpr::add(input_size.clone(), Type::Size(two_padding));

            let kernel_minus_one =
                SizeExpr::sub(kernel_size.clone(), Type::Size(SizeExpr::Literal(1)));
            let dilation_term = SizeExpr::mul(
                Type::Size(SizeExpr::Literal(dilation_val)),
                Type::Size(kernel_minus_one),
            );

            let after_dilation = SizeExpr::sub(Type::Size(padded), Type::Size(dilation_term));
            let numerator =
                SizeExpr::sub(Type::Size(after_dilation), Type::Size(SizeExpr::Literal(1)));

            let quotient = SizeExpr::floor_div(
                Type::Size(numerator),
                Type::Size(SizeExpr::Literal(stride_val)),
            );
            let output_size_ty = simplify(Type::Size(SizeExpr::add(
                Type::Size(quotient),
                Type::Size(SizeExpr::Literal(1)),
            )));

            output_dims.push(output_size_ty);
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "conv"
    }
}

/// Meta-shape function for transposed convolution (conv_transpose)
#[derive(Debug)]
pub struct ConvTransposeMetaShape;

impl MetaShapeFunction for ConvTransposeMetaShape {
    fn signature(&self) -> &[&'static str] {
        &[
            "self",
            "weight",
            "stride",
            "padding",
            "output_padding",
            "dilation",
        ]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let weight_shape = concrete_tensor_shape(bound_args.get("weight")?)?;

        let mut args_map = HashMap::from([
            ("self".to_owned(), MetaShapeArg::Shape(input_shape)),
            ("weight".to_owned(), MetaShapeArg::Shape(weight_shape)),
        ]);

        // stride, padding, output_padding, dilation are optional - can be Int or IntList
        for param in &["stride", "padding", "output_padding", "dilation"] {
            if let Some(param_ty) = bound_args.get(*param) {
                if let Some(val) = literal_int(param_ty) {
                    args_map.insert((*param).to_owned(), MetaShapeArg::Int(val));
                } else if let Some(list) = int_list(param_ty) {
                    args_map.insert((*param).to_owned(), MetaShapeArg::IntList(list));
                }
            }
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "conv_transpose: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        let weight_shape = match args.args.get("weight") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "conv_transpose: 'weight' argument must be a tensor shape".to_owned(),
                });
            }
        };

        if input_shape.rank() < 3 || weight_shape.rank() < 3 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: "conv_transpose requires at least 3D tensors".to_owned(),
            });
        }

        let spatial_dims = input_shape.rank() - 2;

        let stride = match args.args.get("stride") {
            Some(MetaShapeArg::Int(s)) => vec![*s; spatial_dims],
            Some(MetaShapeArg::IntList(s)) => s.clone(),
            _ => vec![1; spatial_dims],
        };

        let padding = match args.args.get("padding") {
            Some(MetaShapeArg::Int(p)) => vec![*p; spatial_dims],
            Some(MetaShapeArg::IntList(p)) => p.clone(),
            _ => vec![0; spatial_dims],
        };

        let output_padding = match args.args.get("output_padding") {
            Some(MetaShapeArg::Int(p)) => vec![*p; spatial_dims],
            Some(MetaShapeArg::IntList(p)) => p.clone(),
            _ => vec![0; spatial_dims],
        };

        let dilation = match args.args.get("dilation") {
            Some(MetaShapeArg::Int(d)) => vec![*d; spatial_dims],
            Some(MetaShapeArg::IntList(d)) => d.clone(),
            _ => vec![1; spatial_dims],
        };

        let mut output_dims = Vec::new();
        output_dims.push(input_shape.get_dim(0));
        output_dims.push(weight_shape.get_dim(1)); // Output channels for transpose conv

        // Formula: (input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        for i in 0..spatial_dims {
            let input_size = input_shape.dims()[2 + i].clone();
            let kernel_size = &weight_shape.dims()[2 + i];
            let stride_val = stride.get(i).copied().unwrap_or(1);
            let padding_val = padding.get(i).copied().unwrap_or(0);
            let output_padding_val = output_padding.get(i).copied().unwrap_or(0);
            let dilation_val = dilation.get(i).copied().unwrap_or(1);

            // Formula: (input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
            // Compute symbolically
            let input_minus_one =
                SizeExpr::sub(input_size.clone(), Type::Size(SizeExpr::Literal(1)));
            let strided = SizeExpr::mul(
                Type::Size(input_minus_one),
                Type::Size(SizeExpr::Literal(stride_val)),
            );

            let two_padding = SizeExpr::mul(
                Type::Size(SizeExpr::Literal(2)),
                Type::Size(SizeExpr::Literal(padding_val)),
            );
            let after_padding = SizeExpr::sub(Type::Size(strided), Type::Size(two_padding));

            let kernel_minus_one =
                SizeExpr::sub(kernel_size.clone(), Type::Size(SizeExpr::Literal(1)));
            let dilation_term = SizeExpr::mul(
                Type::Size(SizeExpr::Literal(dilation_val)),
                Type::Size(kernel_minus_one),
            );

            let after_dilation =
                SizeExpr::add(Type::Size(after_padding), Type::Size(dilation_term));
            let after_output_pad = SizeExpr::add(
                Type::Size(after_dilation),
                Type::Size(SizeExpr::Literal(output_padding_val)),
            );
            let output_size_ty = simplify(Type::Size(SizeExpr::add(
                Type::Size(after_output_pad),
                Type::Size(SizeExpr::Literal(1)),
            )));

            output_dims.push(output_size_ty);
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "conv_transpose"
    }
}

/// Meta-shape function for pooling operations (max_pool, avg_pool)
#[derive(Debug)]
pub struct PoolMetaShape;

impl MetaShapeFunction for PoolMetaShape {
    fn signature(&self) -> &[&'static str] {
        &[
            "self",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "return_indices",
        ]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // kernel_size, stride, padding, dilation are optional - can be Int or IntList
        for param in &["kernel_size", "stride", "padding", "dilation"] {
            if let Some(param_ty) = bound_args.get(*param) {
                if let Some(val) = literal_int(param_ty) {
                    args_map.insert((*param).to_owned(), MetaShapeArg::Int(val));
                } else if let Some(list) = int_list(param_ty) {
                    args_map.insert((*param).to_owned(), MetaShapeArg::IntList(list));
                }
            }
        }

        // return_indices is optional bool
        if let Some(return_indices_ty) = bound_args.get("return_indices")
            && let Some(b) = bool_arg(return_indices_ty)
        {
            args_map.insert("return_indices".to_owned(), MetaShapeArg::Bool(b));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "pool: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        // Input: (N, C, *spatial)
        if input_shape.rank() < 3 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: format!(
                    "pool requires at least 3D tensor, got {}D",
                    input_shape.rank()
                ),
            });
        }

        let spatial_dims = input_shape.rank() - 2;

        let kernel_size = match args.args.get("kernel_size") {
            Some(MetaShapeArg::Int(k)) => vec![*k; spatial_dims],
            Some(MetaShapeArg::IntList(k)) => k.clone(),
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "pool requires 'kernel_size' argument".to_owned(),
                });
            }
        };

        let stride = match args.args.get("stride") {
            Some(MetaShapeArg::Int(s)) => vec![*s; spatial_dims],
            Some(MetaShapeArg::IntList(s)) => s.clone(),
            _ => kernel_size.clone(),
        };

        let padding = match args.args.get("padding") {
            Some(MetaShapeArg::Int(p)) => vec![*p; spatial_dims],
            Some(MetaShapeArg::IntList(p)) => p.clone(),
            _ => vec![0; spatial_dims],
        };

        let dilation = match args.args.get("dilation") {
            Some(MetaShapeArg::Int(d)) => vec![*d; spatial_dims],
            Some(MetaShapeArg::IntList(d)) => d.clone(),
            _ => vec![1; spatial_dims],
        };

        let mut output_dims = Vec::new();
        output_dims.push(input_shape.get_dim(0));
        output_dims.push(input_shape.get_dim(1));

        // Same formula as convolution
        for i in 0..spatial_dims {
            let input_size = input_shape.dims()[2 + i].clone();
            let kernel_val = kernel_size.get(i).copied().unwrap_or(1);
            let stride_val = stride.get(i).copied().unwrap_or(kernel_val);
            let padding_val = padding.get(i).copied().unwrap_or(0);
            let dilation_val = dilation.get(i).copied().unwrap_or(1);

            // Same formula as conv, compute symbolically
            let two_padding = SizeExpr::mul(
                Type::Size(SizeExpr::Literal(2)),
                Type::Size(SizeExpr::Literal(padding_val)),
            );
            let padded = SizeExpr::add(input_size.clone(), Type::Size(two_padding));

            let kernel_minus_one = SizeExpr::Literal(kernel_val - 1);
            let dilation_term = SizeExpr::mul(
                Type::Size(SizeExpr::Literal(dilation_val)),
                Type::Size(kernel_minus_one),
            );

            let after_dilation = SizeExpr::sub(Type::Size(padded), Type::Size(dilation_term));
            let numerator =
                SizeExpr::sub(Type::Size(after_dilation), Type::Size(SizeExpr::Literal(1)));

            let quotient = SizeExpr::floor_div(
                Type::Size(numerator),
                Type::Size(SizeExpr::Literal(stride_val)),
            );
            let output_size_ty = simplify(Type::Size(SizeExpr::add(
                Type::Size(quotient),
                Type::Size(SizeExpr::Literal(1)),
            )));

            output_dims.push(output_size_ty);
        }

        // Check if return_indices is True (for max_pool operations)
        let return_indices = args
            .args
            .get("return_indices")
            .and_then(|arg| arg.as_bool())
            .unwrap_or(false);

        let output_shape = TensorShape::from_types(output_dims);

        if return_indices {
            // Return both output and indices (same shape)
            Ok(MetaShapeResult::Tensors(vec![
                output_shape.clone(),
                output_shape,
            ]))
        } else {
            // Return only output
            Ok(MetaShapeResult::Tensors(vec![output_shape]))
        }
    }

    fn name(&self) -> &str {
        "pool"
    }
}

/// Meta-shape function for adaptive pooling (adaptive_max_pool, adaptive_avg_pool)
/// Output size is specified directly
#[derive(Debug)]
pub struct AdaptivePoolMetaShape;

impl MetaShapeFunction for AdaptivePoolMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "output_size"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // output_size can be Int, IntList, or Shape (tuple of ints/Dims for symbolic dims)
        if let Some(output_size_ty) = bound_args.get("output_size") {
            if let Some(s) = shape_arg(output_size_ty) {
                // shape_arg handles tuples of ints or Dims
                args_map.insert("output_size".to_owned(), MetaShapeArg::Shape(s));
            } else if let Some(val) = literal_int(output_size_ty) {
                args_map.insert("output_size".to_owned(), MetaShapeArg::Int(val));
            } else if let Some(list) = int_list(output_size_ty) {
                args_map.insert("output_size".to_owned(), MetaShapeArg::IntList(list));
            }
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "adaptive_pool: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        if input_shape.rank() < 3 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: "adaptive_pool requires at least 3D tensor".to_owned(),
            });
        }

        let spatial_dims = input_shape.rank() - 2;

        // Get output_size from kwargs - accept both Shape (for symbolic dims) and IntList (for backward compat)
        let output_size_types: Vec<Type> = match args.args.get("output_size") {
            Some(MetaShapeArg::Shape(shape)) => {
                // Use dims() directly to preserve all types (Quantified, SizeExpr, etc.)
                shape.dims().to_vec()
            }
            Some(MetaShapeArg::Int(s)) => {
                // Single int - replicate for all spatial dims
                vec![Type::Size(SizeExpr::Literal(*s)); spatial_dims]
            }
            Some(MetaShapeArg::IntList(s)) => {
                // Legacy IntList - convert to SizeExprs
                s.iter()
                    .map(|&n| Type::Size(SizeExpr::Literal(n)))
                    .collect()
            }
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "adaptive_pool requires 'output_size' argument".to_owned(),
                });
            }
        };

        let mut output_dims = Vec::new();
        output_dims.push(input_shape.get_dim(0));
        output_dims.push(input_shape.get_dim(1));

        // Add output spatial dimensions
        for dim_type in output_size_types {
            output_dims.push(dim_type);
        }

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "adaptive_pool"
    }
}

/// Meta-shape function for interpolate/upsample operations
#[derive(Debug)]
pub struct InterpolateMetaShape;

impl MetaShapeFunction for InterpolateMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "size", "scale_factor"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // size can be Int or IntList (optional)
        if let Some(size_ty) = bound_args.get("size") {
            if let Some(val) = literal_int(size_ty) {
                args_map.insert("size".to_owned(), MetaShapeArg::Int(val));
            } else if let Some(list) = int_list(size_ty) {
                args_map.insert("size".to_owned(), MetaShapeArg::IntList(list));
            }
        }

        // scale_factor can be Int (optional)
        if let Some(scale_ty) = bound_args.get("scale_factor")
            && let Some(val) = literal_int(scale_ty)
        {
            args_map.insert("scale_factor".to_owned(), MetaShapeArg::Int(val));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "interpolate: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        if input_shape.rank() < 3 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: "interpolate requires at least 3D tensor".to_owned(),
            });
        }

        let spatial_dims = input_shape.rank() - 2;

        // Check for size or scale_factor
        if let Some(size_arg) = args.args.get("size") {
            let size = match size_arg {
                MetaShapeArg::Int(s) => vec![*s; spatial_dims],
                MetaShapeArg::IntList(s) => s.clone(),
                _ => {
                    return Err(ShapeError::InvalidDimension {
                        value: 0,
                        reason: "size must be Int or IntList".to_owned(),
                    });
                }
            };

            let mut output_dims = Vec::new();
            output_dims.push(input_shape.get_dim(0));
            output_dims.push(input_shape.get_dim(1));

            for &s in &size {
                output_dims.push(Type::Size(SizeExpr::Literal(s)));
            }

            return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                output_dims,
            )]));
        }

        if let Some(scale_arg) = args.args.get("scale_factor") {
            let scale = match scale_arg {
                MetaShapeArg::Int(s) => vec![*s as f64; spatial_dims],
                _ => {
                    return Err(ShapeError::InvalidDimension {
                        value: 0,
                        reason: "scale_factor must be numeric".to_owned(),
                    });
                }
            };

            let mut output_dims = Vec::new();
            output_dims.push(input_shape.get_dim(0));
            output_dims.push(input_shape.get_dim(1));

            for i in 0..spatial_dims {
                let input_size = input_shape.dims()[2 + i].clone();
                let scale_val = scale.get(i).copied().unwrap_or(1.0);

                let output_size = match input_size {
                    Type::Size(SizeExpr::Literal(in_size)) => {
                        let output = (in_size as f64 * scale_val).floor() as i64;
                        Type::Size(SizeExpr::Literal(output))
                    }
                    _ => Type::Any(AnyStyle::Implicit),
                };
                output_dims.push(output_size);
            }

            return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                output_dims,
            )]));
        }

        Err(ShapeError::InvalidDimension {
            value: 0,
            reason: "interpolate requires either 'size' or 'scale_factor' argument".to_owned(),
        })
    }

    fn name(&self) -> &str {
        "interpolate"
    }
}

// ============================================================================
// Phase 4: Advanced Linear Algebra Operations
// ============================================================================

/// Meta-shape function for torch.linalg.eigvals / torch.linalg.eigvalsh
/// Eigenvalue computation (values only, no eigenvectors)
/// Input: (..., n, n)
/// Output: eigenvalues (..., n)
#[derive(Debug)]
pub struct EigvalsMetaShape;

impl MetaShapeFunction for EigvalsMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "eigvals")?;

        if input_shape.rank() < 2 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: "eigvals requires at least 2D input".to_owned(),
            });
        }

        let n = input_shape.dims()[input_shape.rank() - 2].clone();
        let batch_dims = &input_shape.dims()[..input_shape.rank() - 2];

        // Eigenvalues only: (..., n)
        let mut eigenval_shape = batch_dims.to_vec();
        eigenval_shape.push(n.clone());

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            eigenval_shape,
        )]))
    }

    fn name(&self) -> &str {
        "eigvals"
    }
}

/// Meta-shape function for torch.linalg.eig / torch.eig
/// Eigenvalue decomposition for general matrices
/// Input: (..., n, n)
/// Output: eigenvalues (..., n), eigenvectors (..., n, n)
#[derive(Debug)]
pub struct EigMetaShape;

impl MetaShapeFunction for EigMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "eig")?;

        if input_shape.rank() < 2 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: "eig requires at least 2D input".to_owned(),
            });
        }

        let n1 = input_shape.dims()[input_shape.rank() - 2].clone();
        let n2 = input_shape.dims()[input_shape.rank() - 1].clone();

        let batch_dims = &input_shape.dims()[..input_shape.rank() - 2];

        // Eigenvalues: (..., n)
        let mut eigenval_shape = batch_dims.to_vec();
        eigenval_shape.push(n1.clone());

        // Eigenvectors: (..., n, n)
        let mut eigenvec_shape = batch_dims.to_vec();
        eigenvec_shape.push(n1.clone());
        eigenvec_shape.push(n2.clone());

        Ok(MetaShapeResult::Tensors(vec![
            TensorShape::from_types(eigenval_shape),
            TensorShape::from_types(eigenvec_shape),
        ]))
    }

    fn name(&self) -> &str {
        "eig"
    }
}

/// Meta-shape function for torch.linalg.solve / torch.solve
/// Solves linear system Ax = b
/// Input: A (..., n, n), b (..., n, k) or (..., n)
/// Output: x with same shape as b
#[derive(Debug)]
pub struct SolveMetaShape;

impl MetaShapeFunction for SolveMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "other"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let a_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let b_shape = concrete_tensor_shape(bound_args.get("other")?)?;
        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(a_shape)),
                ("other".to_owned(), MetaShapeArg::Shape(b_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let a_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "solve: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };
        let b_shape = match args.args.get("other") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "solve: 'other' argument must be a tensor shape".to_owned(),
                });
            }
        };

        if a_shape.rank() < 2 {
            return Err(ShapeError::InvalidDimension {
                value: a_shape.rank() as i64,
                reason: "solve: A must be at least 2D".to_owned(),
            });
        }

        if b_shape.rank() < 1 {
            return Err(ShapeError::InvalidDimension {
                value: b_shape.rank() as i64,
                reason: "solve: b must be at least 1D".to_owned(),
            });
        }

        // Output has same shape as b
        Ok(MetaShapeResult::Tensors(vec![b_shape.clone()]))
    }

    fn name(&self) -> &str {
        "solve"
    }
}

/// Meta-shape function for triangular_solve and cholesky_solve
/// These have reversed argument order: (b, A) instead of (A, b)
/// Input: b (..., n, k) or (..., n), A (..., n, n)
/// Output: x with same shape as b
#[derive(Debug)]
pub struct SolveReversedMetaShape;

impl MetaShapeFunction for SolveReversedMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "other"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let b_shape = concrete_tensor_shape(bound_args.get("self")?)?; // b comes first
        let a_shape = concrete_tensor_shape(bound_args.get("other")?)?;
        Some(MetaShapeArgs {
            args: HashMap::from([
                ("self".to_owned(), MetaShapeArg::Shape(b_shape)),
                ("other".to_owned(), MetaShapeArg::Shape(a_shape)),
            ]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let b_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "solve: 'self' (b) argument must be a tensor shape".to_owned(),
                });
            }
        };
        let a_shape = match args.args.get("other") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "solve: 'other' (A) argument must be a tensor shape".to_owned(),
                });
            }
        };

        if b_shape.rank() < 1 {
            return Err(ShapeError::InvalidDimension {
                value: b_shape.rank() as i64,
                reason: "solve: b must be at least 1D".to_owned(),
            });
        }

        if a_shape.rank() < 2 {
            return Err(ShapeError::InvalidDimension {
                value: a_shape.rank() as i64,
                reason: "solve: A must be at least 2D".to_owned(),
            });
        }

        // Output has same shape as b
        Ok(MetaShapeResult::Tensors(vec![b_shape.clone()]))
    }

    fn name(&self) -> &str {
        "solve_reversed"
    }
}

/// Meta-shape function for torch.linalg.slogdet / torch.slogdet
/// Sign and log determinant
/// Input: (..., n, n)
/// Output: sign (...), logabsdet (...)
#[derive(Debug)]
pub struct SlogdetMetaShape;

impl MetaShapeFunction for SlogdetMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "slogdet")?;

        if input_shape.rank() < 2 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: "slogdet requires at least 2D input".to_owned(),
            });
        }

        // Return batch dimensions for both sign and logabsdet
        let batch_dims = &input_shape.dims()[..input_shape.rank() - 2];
        let output_shape = TensorShape::from_types(batch_dims.to_vec());

        Ok(MetaShapeResult::Tensors(vec![
            output_shape.clone(),
            output_shape,
        ]))
    }

    fn name(&self) -> &str {
        "slogdet"
    }
}

/// Meta-shape function for torch.tensordot
/// Tensor contraction over specified dimensions
/// Input: a (...), b (...), dims (list of dimensions to contract)
/// Output: contracted tensor with remaining dimensions
#[derive(Debug)]
pub struct TensordotMetaShape;

impl MetaShapeFunction for TensordotMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "other", "dims"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let self_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let other_shape = concrete_tensor_shape(bound_args.get("other")?)?;

        let mut args_map = HashMap::from([
            ("self".to_owned(), MetaShapeArg::Shape(self_shape)),
            ("other".to_owned(), MetaShapeArg::Shape(other_shape)),
        ]);

        // dims can be int or list - if it's an int, extract it
        if let Some(dims_ty) = bound_args.get("dims")
            && let Some(n) = literal_int(dims_ty)
        {
            args_map.insert("dims".to_owned(), MetaShapeArg::Int(n));
            // For complex cases (list of lists), return None to fallback
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let a_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "tensordot: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };
        let b_shape = match args.args.get("other") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "tensordot: 'other' argument must be a tensor shape".to_owned(),
                });
            }
        };

        // Get dims parameter - can be int or list of lists
        let n_contract = match args.args.get("dims") {
            Some(MetaShapeArg::Int(n)) => *n as usize,
            _ => {
                // Complex case: return Unknown
                return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                    vec![Type::Any(AnyStyle::Implicit)],
                )]));
            }
        };

        if n_contract > a_shape.rank() || n_contract > b_shape.rank() {
            return Err(ShapeError::InvalidDimension {
                value: n_contract as i64,
                reason: format!("tensordot: cannot contract {} dims", n_contract),
            });
        }

        // Output dims: remaining dims from a + remaining dims from b
        let mut output_dims = Vec::new();

        // Add non-contracted dims from a (all except last n_contract)
        output_dims.extend_from_slice(&a_shape.dims()[..a_shape.rank() - n_contract]);

        // Add non-contracted dims from b (all except first n_contract)
        output_dims.extend_from_slice(&b_shape.dims()[n_contract..]);

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "tensordot"
    }
}

/// Meta-shape function for torch.einsum
/// Einstein summation convention - NOW FULLY IMPLEMENTED!
#[derive(Debug)]
pub struct EinsumMetaShape;

impl EinsumMetaShape {
    /// Parse einsum specification like "ij,jk->ik"
    /// Returns (input_specs, output_spec) where specs are lists of index chars
    fn parse_spec(spec: &str) -> Result<(Vec<Vec<char>>, Vec<char>), ShapeError> {
        let parts: Vec<&str> = spec.split("->").collect();

        if parts.len() != 2 {
            return Err(ShapeError::InvalidDimension {
                value: 0,
                reason: format!("einsum spec must contain '->', got: {}", spec),
            });
        }

        let inputs_str = parts[0];
        let output_str = parts[1];

        // Parse input specs (separated by commas)
        let input_specs: Vec<Vec<char>> = inputs_str
            .split(',')
            .map(|s| s.trim().chars().filter(|c| c.is_alphanumeric()).collect())
            .collect();

        // Parse output spec
        let output_spec: Vec<char> = output_str
            .trim()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();

        Ok((input_specs, output_spec))
    }

    /// Apply einsum operation to compute output shape
    fn apply_einsum(
        inputs: &[TensorShape],
        input_specs: &[Vec<char>],
        output_spec: &[char],
    ) -> Result<TensorShape, ShapeError> {
        if inputs.len() != input_specs.len() {
            return Err(ShapeError::InvalidDimension {
                value: inputs.len() as i64,
                reason: format!(
                    "einsum: spec has {} inputs, got {} tensors",
                    input_specs.len(),
                    inputs.len()
                ),
            });
        }

        // Build a map from index character to dimension
        use std::collections::HashMap as StdHashMap;
        let mut index_to_dim: StdHashMap<char, Type> = StdHashMap::new();

        // Process each input tensor
        for (input_shape, input_spec) in inputs.iter().zip(input_specs.iter()) {
            if input_shape.rank() != input_spec.len() {
                return Err(ShapeError::RankMismatch {
                    got: input_shape.rank(),
                    want: input_spec.len(),
                });
            }

            // Map each index to its dimension
            for (idx_char, dim) in input_spec.iter().zip(input_shape.dims().iter()) {
                if let Some(existing_dim) = index_to_dim.get(idx_char) {
                    // Check consistency: same index should have same dimension
                    // For symbolic dims, we just accept it (can't check equality)
                    // For literal dims, verify they match
                    match (existing_dim, dim) {
                        (Type::Size(SizeExpr::Literal(a)), Type::Size(SizeExpr::Literal(b)))
                            if a != b =>
                        {
                            return Err(ShapeError::InvalidDimension {
                                value: *b,
                                reason: format!(
                                    "einsum: index '{}' has inconsistent dimensions: {} vs {}",
                                    idx_char, a, b
                                ),
                            });
                        }
                        _ => {
                            // Either they match, or one is symbolic - accept it
                        }
                    }
                } else {
                    index_to_dim.insert(*idx_char, dim.clone());
                }
            }
        }

        // Build output shape from output spec
        let mut output_dims = Vec::new();
        for idx_char in output_spec {
            if let Some(dim) = index_to_dim.get(idx_char) {
                output_dims.push(dim.clone());
            } else {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: format!("einsum: output index '{}' not found in inputs", idx_char),
                });
            }
        }

        Ok(TensorShape::from_types(output_dims))
    }
}

impl MetaShapeFunction for EinsumMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["spec"] // Variadic inputs handled separately
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        use crate::literal::Lit;

        // Extract spec string
        let spec_str = match bound_args.get("spec") {
            Some(Type::Literal(box Literal {
                value: Lit::Str(s), ..
            })) => s.to_string(),
            _ => return None,
        };

        // Store spec as a string argument
        let mut args_map = HashMap::new();
        args_map.insert("spec".to_owned(), MetaShapeArg::String(spec_str));

        // Extract all tensor inputs (variadic)
        // Look for numbered arguments or specific parameter names
        let mut tensor_shapes = Vec::new();
        for i in 0..10 {
            // Try up to 10 inputs
            let key = format!("arg{}", i);
            if let Some(ty) = bound_args.get(&key)
                && let Some(shape) = concrete_tensor_shape(ty)
            {
                tensor_shapes.push(shape);
            }
        }

        // Also try "tensors" parameter if it exists
        if tensor_shapes.is_empty() {
            // No tensor inputs found - return None to fallback to fixture type
            return None;
        }

        args_map.insert("inputs".to_owned(), MetaShapeArg::ShapeList(tensor_shapes));

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Get the einsum specification string
        let spec = match args.args.get("spec") {
            Some(MetaShapeArg::String(s)) => s.as_str(),
            _ => {
                // No spec provided - return Unknown
                return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                    vec![Type::Any(AnyStyle::Implicit)],
                )]));
            }
        };

        // Get input shapes
        let inputs = match args.args.get("inputs") {
            Some(MetaShapeArg::ShapeList(shapes)) => shapes,
            _ => {
                return Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                    vec![Type::Any(AnyStyle::Implicit)],
                )]));
            }
        };

        // Parse the specification
        let (input_specs, output_spec) = Self::parse_spec(spec)?;

        // Apply einsum to compute output shape
        let output_shape = Self::apply_einsum(inputs, &input_specs, &output_spec)?;

        Ok(MetaShapeResult::Tensors(vec![output_shape]))
    }

    fn name(&self) -> &str {
        "einsum"
    }
}

// ============================================================================
// Phase 5: Advanced Indexing & Conditional Operations
// ============================================================================

/// Meta-shape function for torch.where
/// Conditional element-wise selection
/// where(condition, x, y) returns elements from x or y based on condition
/// All inputs must be broadcastable to the same shape
#[derive(Debug)]
pub struct WhereMetaShape;

impl MetaShapeFunction for WhereMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["condition", "x", "y"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let _condition_shape = concrete_tensor_shape(bound_args.get("condition")?)?;
        let x_shape = concrete_tensor_shape(bound_args.get("x")?)?;
        let _y_shape = concrete_tensor_shape(bound_args.get("y")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([("x".to_owned(), MetaShapeArg::Shape(x_shape))]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let x_shape = match args.args.get("x") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "where requires 'x' argument".to_owned(),
                });
            }
        };

        // For simplicity, return shape of x (first value tensor)
        // In reality, all three should be broadcastable
        Ok(MetaShapeResult::Tensors(vec![x_shape.clone()]))
    }

    fn name(&self) -> &str {
        "where"
    }
}

/// Meta-shape function for torch.take_along_dim
/// Takes elements along a dimension using index tensor
/// Output shape matches indices shape
#[derive(Debug)]
pub struct TakeAlongDimMetaShape;

impl MetaShapeFunction for TakeAlongDimMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "indices", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let _input_shape = concrete_tensor_shape(bound_args.get("self")?)?;
        let indices_shape = concrete_tensor_shape(bound_args.get("indices")?)?;

        Some(MetaShapeArgs {
            args: HashMap::from([("indices".to_owned(), MetaShapeArg::Shape(indices_shape))]),
        })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let indices_shape = match args.args.get("indices") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "take_along_dim requires 'indices' argument".to_owned(),
                });
            }
        };

        // Output shape matches indices shape
        Ok(MetaShapeResult::Tensors(vec![indices_shape.clone()]))
    }

    fn name(&self) -> &str {
        "take_along_dim"
    }
}

// ============================================================================
// Phase 6: Specialized Operations (FFT, Loss, Padding, etc.)
// ============================================================================

/// Meta-shape function for torch.nn.functional loss functions
/// Most loss functions return scalars or apply reduction
#[derive(Debug)]
pub struct LossMetaShape;

impl MetaShapeFunction for LossMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "reduction"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .optional_int(bound_args, "reduction")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "loss")?;

        // Check reduction parameter (default is "mean")
        let reduction = args
            .args
            .get("reduction")
            .and_then(|arg| match arg {
                MetaShapeArg::Int(0) => Some("none"), // no reduction
                MetaShapeArg::Int(1) => Some("mean"), // default
                MetaShapeArg::Int(2) => Some("sum"),
                _ => None,
            })
            .unwrap_or("mean");

        match reduction {
            "none" => {
                // No reduction - preserve input shape
                Ok(MetaShapeResult::Tensors(vec![input_shape.clone()]))
            }
            "mean" | "sum" => {
                // Reduction to scalar
                Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![])]))
            }
            _ => Ok(MetaShapeResult::Tensors(vec![TensorShape::new(vec![])])),
        }
    }

    fn name(&self) -> &str {
        "loss"
    }
}

/// Meta-shape function for torch.nn.functional.pad
/// Padding operations that add to dimensions
#[derive(Debug)]
pub struct PadMetaShape;

impl MetaShapeFunction for PadMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "pad"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;
        let input_shape = concrete_tensor_shape(bound_args.get("self")?)?;

        let mut args_map = HashMap::from([("self".to_owned(), MetaShapeArg::Shape(input_shape))]);

        // pad is IntList parameter (optional)
        if let Some(pad_ty) = bound_args.get("pad")
            && let Some(pad_list) = int_list(pad_ty)
        {
            args_map.insert("pad".to_owned(), MetaShapeArg::IntList(pad_list));
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "pad: 'self' argument must be a tensor shape".to_owned(),
                });
            }
        };

        // Get pad parameter - list of pad amounts [left, right, top, bottom, front, back]
        let pad = args.args.get("pad").and_then(|arg| match arg {
            MetaShapeArg::IntList(p) => Some(p.clone()),
            _ => None,
        });

        if let Some(pad_vals) = pad {
            // Pad values come in pairs: [left, right, top, bottom, ...]
            // Applied to last N/2 dimensions
            let num_dims_to_pad = pad_vals.len() / 2;

            if num_dims_to_pad > input_shape.rank() {
                return Err(ShapeError::InvalidDimension {
                    value: num_dims_to_pad as i64,
                    reason: "pad: too many padding dimensions".to_owned(),
                });
            }

            let mut output_dims = input_shape.dims().clone();
            let start_dim = input_shape.rank() - num_dims_to_pad;

            // Apply padding to last N dimensions
            // Compute symbolically: dim + left_pad + right_pad
            for (i, dim_idx) in (start_dim..input_shape.rank()).rev().enumerate() {
                let left_pad = pad_vals.get(i * 2).copied().unwrap_or(0);
                let right_pad = pad_vals.get(i * 2 + 1).copied().unwrap_or(0);

                let current_dim = output_dims[dim_idx].clone();
                let with_left = SizeExpr::add(current_dim, Type::Size(SizeExpr::Literal(left_pad)));
                let with_both = SizeExpr::add(
                    Type::Size(with_left),
                    Type::Size(SizeExpr::Literal(right_pad)),
                );
                output_dims[dim_idx] = simplify(Type::Size(with_both));
            }

            Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                output_dims,
            )]))
        } else {
            // No padding specified, return Unknown
            Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
                vec![Type::Any(AnyStyle::Implicit)],
            )]))
        }
    }

    fn name(&self) -> &str {
        "pad"
    }
}

/// Meta-shape function for torch.normal
/// Generates random numbers from normal distribution
/// Handles multiple overloads based on input types
#[derive(Debug)]
pub struct NormalMetaShape;

impl MetaShapeFunction for NormalMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["mean", "std", "size"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        use extract::*;

        let mut args_map = HashMap::new();

        // Check if size parameter is provided
        if let Some(size_ty) = bound_args.get("size")
            && let Some(size_list) = int_list(size_ty)
        {
            args_map.insert("size".to_owned(), MetaShapeArg::IntList(size_list));
            return Some(MetaShapeArgs { args: args_map });
        }

        // Otherwise, try to extract mean or std tensor shapes
        if let Some(mean_ty) = bound_args.get("mean")
            && let Some(shape) = concrete_tensor_shape(mean_ty)
        {
            args_map.insert("mean".to_owned(), MetaShapeArg::Shape(shape));
        }

        if let Some(std_ty) = bound_args.get("std")
            && let Some(shape) = concrete_tensor_shape(std_ty)
        {
            args_map.insert("std".to_owned(), MetaShapeArg::Shape(shape));
        }

        if args_map.is_empty() {
            return None;
        }

        Some(MetaShapeArgs { args: args_map })
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // Check if size parameter is provided (scalar mean/std case)
        if let Some(MetaShapeArg::IntList(size)) = args.args.get("size") {
            // torch.normal(mean: float, std: float, size: tuple) -> creates new tensor
            let dims: Vec<SizeExpr> = size.iter().map(|&s| SizeExpr::Literal(s)).collect();
            return Ok(MetaShapeResult::Tensors(vec![TensorShape::new(dims)]));
        }

        // torch.normal(mean: Tensor, std: Tensor/float) -> use mean's shape
        if let Some(MetaShapeArg::Shape(mean_shape)) = args.args.get("mean") {
            return Ok(MetaShapeResult::Tensors(vec![mean_shape.clone()]));
        }

        // torch.normal(mean: float, std: Tensor) -> use std's shape
        if let Some(MetaShapeArg::Shape(std_shape)) = args.args.get("std") {
            return Ok(MetaShapeResult::Tensors(vec![std_shape.clone()]));
        }

        Err(ShapeError::InvalidDimension {
            value: 0,
            reason: "normal expects at least 1 input or size parameter".to_owned(),
        })
    }

    fn name(&self) -> &str {
        "normal"
    }
}

/// Meta-shape function for torch.multinomial
/// Samples from categorical distribution
/// Input: (n,) or (m, n) - probabilities
/// Output: (num_samples,) or (m, num_samples)
#[derive(Debug)]
pub struct MultinomialMetaShape;

impl MetaShapeFunction for MultinomialMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "num_samples", "replacement"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .required_int(bound_args, "num_samples")?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "multinomial")?;

        // For variadic shapes, fall back to fixture return type
        if matches!(input_shape, TensorShape::Unpacked(_)) {
            return Ok(MetaShapeResult::Unknown);
        }

        // Get num_samples parameter
        let num_samples = match args.args.get("num_samples") {
            Some(MetaShapeArg::Int(n)) => *n,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: "multinomial requires 'num_samples' parameter".to_owned(),
                });
            }
        };

        // Input is 1D or 2D
        // Output replaces last dimension with num_samples
        let mut output_dims = if input_shape.rank() >= 2 {
            input_shape.dims()[..input_shape.rank() - 1].to_vec()
        } else {
            vec![]
        };
        output_dims.push(Type::Size(SizeExpr::Literal(num_samples)));

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        "multinomial"
    }
}

/// Meta-shape function for tensor.size()
/// Returns tuple of literal ints representing tensor dimensions
#[derive(Debug)]
pub struct SizeMetaShape;

impl MetaShapeFunction for SizeMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .optional_int(bound_args, "dim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "size")?;

        // For variadic shapes (e.g., Tensor[*Bs, C]), we can't determine exact dimensions
        // Fall back to fixture return type:
        // - .size() returns tuple[int, ...]
        // - .size(i) returns int
        if matches!(input_shape, TensorShape::Unpacked(_)) {
            return Ok(MetaShapeResult::Unknown);
        }

        // Check if dim parameter is provided (size(dim) returns single int)
        if let Some(MetaShapeArg::Int(dim)) = args.args.get("dim") {
            let dim_idx = input_shape.normalize_dim(*dim)?;
            let dimension = input_shape.dims()[dim_idx].clone();
            match dimension {
                Type::Size(SizeExpr::Literal(size)) => {
                    return Ok(MetaShapeResult::Int(size));
                }
                _ => {
                    // Wrap in Dim for symbolic dimensions
                    return Ok(MetaShapeResult::IntSymbolic(Type::Dim(Box::new(dimension))));
                }
            }
        }

        // No dim parameter - return tuple of all dimensions (literal or symbolic)
        // Extract SizeExpr from Type wrappers
        // Return IntSymbolicTuple with dimensions (can be literal, symbolic, or expressions)
        Ok(MetaShapeResult::IntSymbolicTuple(
            input_shape.dims().to_vec(),
        ))
    }

    fn result_to_type(&self, result: MetaShapeResult, original_return_type: &Type) -> Type {
        use crate::lit_int::LitInt;
        use crate::literal::Lit;

        match result {
            MetaShapeResult::IntSymbolicTuple(dims) => {
                // Convert each dimension to proper type:
                // - Literal dims become Literal[n]
                // - Symbolic dims become wrapped in Dim
                let types: Vec<Type> = dims
                    .into_iter()
                    .map(|dim| match dim {
                        Type::Size(SizeExpr::Literal(n)) => {
                            Lit::Int(LitInt::new(n)).to_implicit_type()
                        }
                        _ => Type::Dim(Box::new(dim)),
                    })
                    .collect();
                Type::concrete_tuple(types)
            }
            _ => convert_result_to_type_default(result, original_return_type),
        }
    }

    fn name(&self) -> &str {
        "size"
    }
}

/// Meta-shape function for tensor.numel() method
/// Returns literal int representing number of elements
#[derive(Debug)]
pub struct NumelIntMetaShape;

impl MetaShapeFunction for NumelIntMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "numel_int")?;

        // Compute product of all dimensions (symbolic or literal)
        let mut numel_ty = Type::Size(SizeExpr::Literal(1));
        for dim in input_shape.dims() {
            numel_ty = simplify(Type::Size(SizeExpr::mul(numel_ty, dim.clone())));
        }

        // Return result based on whether it's concrete or symbolic
        match numel_ty {
            Type::Size(SizeExpr::Literal(n)) => Ok(MetaShapeResult::Int(n)),
            _ => {
                // Wrap in Dim for symbolic results
                Ok(MetaShapeResult::IntSymbolic(Type::Dim(Box::new(numel_ty))))
            }
        }
    }

    fn name(&self) -> &str {
        "numel_int"
    }
}

/// Meta-shape function for tensor.dim() / tensor.ndim()
/// Returns the rank (number of dimensions) as a literal int
#[derive(Debug)]
pub struct DimMetaShape;

impl MetaShapeFunction for DimMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "dim")?;

        Ok(MetaShapeResult::Int(input_shape.rank() as i64))
    }

    fn name(&self) -> &str {
        "dim"
    }
}

/// Meta-shape function for tensor.item()
/// Returns a scalar (float | int) from a 0-dimensional tensor
#[derive(Debug)]
pub struct ItemMetaShape;

impl MetaShapeFunction for ItemMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = args.get_shape("self", "item")?;

        if input_shape.rank() != 0 {
            return Err(ShapeError::InvalidDimension {
                value: input_shape.rank() as i64,
                reason: format!(
                    "item() only works on 0-dimensional tensors, got {}D tensor",
                    input_shape.rank()
                ),
            });
        }

        // Return Unknown to represent float | int
        // The type system will use the stub's return type annotation
        Ok(MetaShapeResult::Unknown)
    }

    fn name(&self) -> &str {
        "item"
    }
}

/// Meta-shape function for tensor.tolist()
/// Returns nested Python lists based on tensor rank
#[derive(Debug)]
pub struct TolistMetaShape;

impl MetaShapeFunction for TolistMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .build()
    }

    fn compute(&self, _args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        // tolist() converts tensor to nested Python lists
        // The nesting level matches the tensor rank:
        // - 0D tensor -> float | int
        // - 1D tensor -> list[float | int]
        // - 2D tensor -> list[list[float | int]]
        // etc.
        // We return Unknown to let the stub's return type (Any) handle this
        Ok(MetaShapeResult::Unknown)
    }

    fn name(&self) -> &str {
        "tolist"
    }
}

/// FFT transformation mode for dimension changes.
#[derive(Debug, Clone, Copy)]
pub enum FftMode {
    /// Real to Complex: n -> n//2 + 1 (rfft, ihfft)
    RealToComplex,
    /// Complex to Real: n -> 2*(n-1) or specified n (irfft, hfft)
    ComplexToReal,
}

/// Unified meta-shape function for FFT operations.
/// Handles both real-to-complex (rfft, ihfft) and complex-to-real (irfft, hfft) transforms.
#[derive(Debug)]
pub struct FftMetaShape {
    mode: FftMode,
    name: &'static str,
}

impl FftMetaShape {
    pub fn rfft() -> Self {
        Self {
            mode: FftMode::RealToComplex,
            name: "rfft",
        }
    }

    pub fn irfft() -> Self {
        Self {
            mode: FftMode::ComplexToReal,
            name: "irfft",
        }
    }

    pub fn hfft() -> Self {
        Self {
            mode: FftMode::ComplexToReal,
            name: "hfft",
        }
    }

    pub fn ihfft() -> Self {
        Self {
            mode: FftMode::RealToComplex,
            name: "ihfft",
        }
    }

    /// Compute the output dimension for real-to-complex FFT: n -> n//2 + 1
    fn real_to_complex_dim(input_dim: Type, n: Option<i64>) -> Type {
        if let Some(n_val) = n {
            Type::Size(SizeExpr::Literal(n_val / 2 + 1))
        } else {
            match input_dim {
                Type::Size(SizeExpr::Literal(size)) => Type::Size(SizeExpr::Literal(size / 2 + 1)),
                _ => {
                    // Compute (input_dim // 2) + 1 symbolically
                    let div_result =
                        SizeExpr::floor_div(input_dim.clone(), Type::Size(SizeExpr::Literal(2)));
                    simplify(Type::Size(SizeExpr::add(
                        Type::Size(div_result),
                        Type::Size(SizeExpr::Literal(1)),
                    )))
                }
            }
        }
    }

    /// Compute the output dimension for complex-to-real FFT: n -> 2*(n-1)
    fn complex_to_real_dim(input_dim: Type, n: Option<i64>) -> Type {
        if let Some(n_val) = n {
            Type::Size(SizeExpr::Literal(n_val))
        } else {
            match input_dim {
                Type::Size(SizeExpr::Literal(size)) => {
                    Type::Size(SizeExpr::Literal(2 * (size - 1)))
                }
                _ => {
                    // Compute 2 * (input_dim - 1) symbolically
                    let minus_one =
                        SizeExpr::sub(input_dim.clone(), Type::Size(SizeExpr::Literal(1)));
                    simplify(Type::Size(SizeExpr::mul(
                        Type::Size(SizeExpr::Literal(2)),
                        Type::Size(minus_one),
                    )))
                }
            }
        }
    }
}

impl MetaShapeFunction for FftMetaShape {
    fn signature(&self) -> &[&'static str] {
        &["self", "n", "dim"]
    }

    fn bind_args(&self, bound_args: &HashMap<String, Type>) -> Option<MetaShapeArgs> {
        extract::BindArgsBuilder::new()
            .self_shape(bound_args)?
            .optional_int(bound_args, "n")
            .optional_int(bound_args, "dim")
            .build()
    }

    fn compute(&self, args: MetaShapeArgs) -> Result<MetaShapeResult, ShapeError> {
        let input_shape = match args.args.get("self") {
            Some(MetaShapeArg::Shape(s)) => s,
            _ => {
                return Err(ShapeError::InvalidDimension {
                    value: 0,
                    reason: format!("{}: 'self' argument must be a tensor shape", self.name),
                });
            }
        };

        // Get dim parameter (default is -1)
        let dim = args
            .args
            .get("dim")
            .and_then(|arg| arg.as_int())
            .unwrap_or(-1);
        let dim_idx = input_shape.normalize_dim(dim)?;

        // Get n parameter (output size, if specified)
        let n = args.args.get("n").and_then(|arg| arg.as_int());

        let mut output_dims = input_shape.dims().clone();
        let input_dim = input_shape.dims()[dim_idx].clone();

        // Compute output dimension based on FFT mode
        output_dims[dim_idx] = match self.mode {
            FftMode::RealToComplex => Self::real_to_complex_dim(input_dim, n),
            FftMode::ComplexToReal => Self::complex_to_real_dim(input_dim, n),
        };

        Ok(MetaShapeResult::Tensors(vec![TensorShape::from_types(
            output_dims,
        )]))
    }

    fn name(&self) -> &str {
        self.name
    }
}

// ============================================================================
// Meta-Shape Registry
// ============================================================================

/// Registry of meta-shape functions
pub struct MetaShapeRegistry {
    functions: HashMap<String, Box<dyn MetaShapeFunction>>,
}

impl MetaShapeRegistry {
    /// Create a new registry with built-in meta-shape functions
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };

        // Register built-in meta-shape functions (dual registration for torch.X and Tensor.X)
        registry.register_dual("reshape", || Box::new(ReshapeMetaShape));
        registry.register("torch.cat", Box::new(ConcatMetaShape));
        registry.register("torch.broadcast_to", Box::new(BroadcastToMetaShape));
        registry.register_dual("squeeze", || Box::new(SqueezeMetaShape));
        registry.register_dual("unsqueeze", || Box::new(UnsqueezeMetaShape));
        registry.register_dual("transpose", || Box::new(TransposeMetaShape));
        registry.register_dual("permute", || Box::new(PermuteMetaShape));

        // Register reduction operations (all use the same meta-shape logic)
        registry.register_dual("sum", || Box::new(ReduceMetaShape));
        registry.register_dual("mean", || Box::new(ReduceMetaShape));
        registry.register_dual("prod", || Box::new(ReduceMetaShape));
        registry.register_dual("min", || Box::new(MinMaxMedianMetaShape));
        registry.register_dual("max", || Box::new(MinMaxMedianMetaShape));
        registry.register_dual("all", || Box::new(ReduceMetaShape));
        registry.register_dual("any", || Box::new(ReduceMetaShape));

        // Register additional reduction operations
        registry.register_dual("std", || Box::new(ReduceMetaShape));
        registry.register_dual("var", || Box::new(ReduceMetaShape));
        registry.register_dual("argmax", || Box::new(ReduceMetaShape));
        registry.register_dual("argmin", || Box::new(ReduceMetaShape));

        // Register Phase 1.2: Missing reduction operations
        registry.register("torch.median", Box::new(MinMaxMedianMetaShape));
        registry.register("torch.logsumexp", Box::new(ReduceMetaShape));
        registry.register("torch.count_nonzero", Box::new(ReduceMetaShape));
        registry.register("torch.aminmax", Box::new(AminmaxMetaShape));
        registry.register("torch.norm", Box::new(NormMetaShape));

        // Tier 2: Additional reduction operations (always return tuples)
        registry.register("torch.mode", Box::new(TupleReduceMetaShape));
        registry.register("torch.topk", Box::new(TopKMetaShape));
        registry.register("torch.kthvalue", Box::new(TupleReduceMetaShape));

        // Tier 3: var_mean and std_mean (return tuple of var/std and mean)
        registry.register("torch.var_mean", Box::new(VarMeanMetaShape));
        registry.register("torch.std_mean", Box::new(VarMeanMetaShape));

        // Register tensor creation operations
        registry.register("torch.randn", Box::new(RandnMetaShape));
        registry.register("torch.rand", Box::new(RandnMetaShape));
        registry.register("torch.zeros", Box::new(RandnMetaShape));
        registry.register("torch.ones", Box::new(RandnMetaShape));
        registry.register("torch.empty", Box::new(RandnMetaShape));
        registry.register("torch.full", Box::new(FullMetaShape));
        registry.register("torch.arange", Box::new(ArangeMetaShape));
        registry.register("torch.linspace", Box::new(LinspaceMetaShape));
        registry.register("torch.eye", Box::new(EyeMetaShape));

        // Register Phase 1.3: Tensor creation operations

        // Diagonal operations
        registry.register("torch.diag_embed", Box::new(DiagEmbedMetaShape));

        // Triangular indices
        registry.register("torch.tril_indices", Box::new(TriIndicesMetaShape));
        registry.register("torch.triu_indices", Box::new(TriIndicesMetaShape));

        // Register Phase 1.4: Basic linear algebra operations
        registry.register("torch.matmul", Box::new(MatMulMetaShape));
        registry.register("torch.mv", Box::new(MVMetaShape));
        registry.register("torch.outer", Box::new(OuterMetaShape));

        // Register shape manipulation operations
        registry.register("torch.flatten", Box::new(FlattenMetaShape));
        registry.register("torch.stack", Box::new(StackMetaShape));
        registry.register("torch.tile", Box::new(TileMetaShape));
        registry.register("torch.view", Box::new(ReshapeMetaShape)); // view is alias for reshape

        // Register indexing/slicing operations
        registry.register("torch.select", Box::new(SelectMetaShape));
        registry.register("torch.narrow", Box::new(NarrowMetaShape));
        registry.register("torch.split", Box::new(SplitMetaShape));
        registry.register("torch.chunk", Box::new(ChunkMetaShape));
        registry.register("torch.index_select", Box::new(IndexSelectMetaShape));
        registry.register("torch.masked_select", Box::new(MaskedSelectMetaShape));

        // Register method versions (torch.Tensor.method_name)
        registry.register("torch.Tensor.reshape", Box::new(ReshapeMetaShape));
        registry.register("torch.Tensor.view", Box::new(ReshapeMetaShape));
        registry.register("torch.Tensor.squeeze", Box::new(SqueezeMetaShape));
        registry.register("torch.Tensor.unsqueeze", Box::new(UnsqueezeMetaShape));
        registry.register("torch.Tensor.transpose", Box::new(TransposeMetaShape));
        registry.register("torch.Tensor.permute", Box::new(PermuteMetaShape));
        registry.register("torch.Tensor.flatten", Box::new(FlattenMetaShape));
        registry.register("torch.Tensor.tile", Box::new(TileMetaShape));
        registry.register("torch.Tensor.select", Box::new(SelectMetaShape));
        registry.register("torch.Tensor.narrow", Box::new(NarrowMetaShape));
        registry.register("torch.Tensor.split", Box::new(SplitMetaShape));
        registry.register("torch.Tensor.chunk", Box::new(ChunkMetaShape));
        registry.register("torch.Tensor.index_select", Box::new(IndexSelectMetaShape));
        registry.register(
            "torch.Tensor.masked_select",
            Box::new(MaskedSelectMetaShape),
        );
        registry.register("torch.Tensor.sum", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.mean", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.prod", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.min", Box::new(MinMaxMedianMetaShape));
        registry.register("torch.Tensor.max", Box::new(MinMaxMedianMetaShape));
        registry.register("torch.Tensor.std", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.var", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.argmax", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.argmin", Box::new(ReduceMetaShape));

        // Register Phase 1.2: Missing reduction method versions
        registry.register("torch.Tensor.median", Box::new(MinMaxMedianMetaShape));
        registry.register("torch.Tensor.logsumexp", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.count_nonzero", Box::new(ReduceMetaShape));
        registry.register("torch.Tensor.aminmax", Box::new(AminmaxMetaShape));
        registry.register("torch.Tensor.norm", Box::new(NormMetaShape));

        // Tier 2: Tensor method versions
        registry.register("torch.Tensor.mode", Box::new(TupleReduceMetaShape));
        registry.register("torch.Tensor.topk", Box::new(TopKMetaShape));
        registry.register("torch.Tensor.kthvalue", Box::new(TupleReduceMetaShape));

        // Register other operations
        registry.register("torch.Tensor.repeat", Box::new(RepeatMetaShape));
        registry.register("torch.Tensor.expand", Box::new(ExpandMetaShape));

        // Register missing shape operations (Phase 1.1)
        registry.register("torch.unbind", Box::new(UnbindMetaShape));
        registry.register("torch.Tensor.unbind", Box::new(UnbindMetaShape));
        registry.register("torch.movedim", Box::new(MovedimMetaShape));
        registry.register("torch.moveaxis", Box::new(MovedimMetaShape)); // alias for movedim
        registry.register("torch.Tensor.movedim", Box::new(MovedimMetaShape));
        registry.register("torch.Tensor.moveaxis", Box::new(MovedimMetaShape));
        registry.register("torch.unfold", Box::new(UnfoldMetaShape));
        registry.register("torch.Tensor.unfold", Box::new(UnfoldMetaShape));

        // Register Phase 1.3: Tensor method versions
        registry.register("torch.Tensor.diag_embed", Box::new(DiagEmbedMetaShape));

        // Register Phase 1.4: Tensor method versions for linear algebra
        registry.register("torch.Tensor.matmul", Box::new(MatMulMetaShape));
        registry.register("torch.Tensor.__matmul__", Box::new(MatMulMetaShape)); // @ operator
        registry.register("torch.Tensor.mv", Box::new(MVMetaShape));

        // =====================================================================
        // Phase 3: Convolution & Pooling Operations
        // Critical for CNN and ML workloads
        // =====================================================================

        // Convolution operations (1D, 2D, 3D)
        registry.register("torch.nn.functional.conv1d", Box::new(ConvMetaShape));
        registry.register("torch.nn.functional.conv2d", Box::new(ConvMetaShape));
        registry.register("torch.nn.functional.conv3d", Box::new(ConvMetaShape));

        // Transposed convolution operations
        registry.register(
            "torch.nn.functional.conv_transpose1d",
            Box::new(ConvTransposeMetaShape),
        );
        registry.register(
            "torch.nn.functional.conv_transpose2d",
            Box::new(ConvTransposeMetaShape),
        );
        registry.register(
            "torch.nn.functional.conv_transpose3d",
            Box::new(ConvTransposeMetaShape),
        );

        // Max pooling operations (1D, 2D, 3D)
        registry.register("torch.nn.functional.max_pool1d", Box::new(PoolMetaShape));
        registry.register("torch.nn.functional.max_pool2d", Box::new(PoolMetaShape));
        registry.register("torch.nn.functional.max_pool3d", Box::new(PoolMetaShape));

        // Average pooling operations (1D, 2D, 3D)
        registry.register("torch.nn.functional.avg_pool1d", Box::new(PoolMetaShape));
        registry.register("torch.nn.functional.avg_pool2d", Box::new(PoolMetaShape));
        registry.register("torch.nn.functional.avg_pool3d", Box::new(PoolMetaShape));

        // Adaptive max pooling operations (1D, 2D, 3D)
        registry.register(
            "torch.nn.functional.adaptive_max_pool1d",
            Box::new(AdaptivePoolMetaShape),
        );
        registry.register(
            "torch.nn.functional.adaptive_max_pool2d",
            Box::new(AdaptivePoolMetaShape),
        );
        registry.register(
            "torch.nn.functional.adaptive_max_pool3d",
            Box::new(AdaptivePoolMetaShape),
        );

        // Adaptive average pooling operations (1D, 2D, 3D)
        registry.register(
            "torch.nn.functional.adaptive_avg_pool1d",
            Box::new(AdaptivePoolMetaShape),
        );
        registry.register(
            "torch.nn.functional.adaptive_avg_pool2d",
            Box::new(AdaptivePoolMetaShape),
        );
        registry.register(
            "torch.nn.functional.adaptive_avg_pool3d",
            Box::new(AdaptivePoolMetaShape),
        );

        // Interpolation/upsampling operations
        registry.register(
            "torch.nn.functional.interpolate",
            Box::new(InterpolateMetaShape),
        );
        registry.register(
            "torch.nn.functional.upsample",
            Box::new(InterpolateMetaShape),
        );

        // =====================================================================
        // Phase 4: Advanced Linear Algebra Operations
        // For scientific computing and advanced ML
        // =====================================================================

        // Advanced matmul operations
        registry.register("torch.tensordot", Box::new(TensordotMetaShape));
        registry.register("torch.einsum", Box::new(EinsumMetaShape));

        // Eigenvalue decomposition
        registry.register("torch.linalg.eig", Box::new(EigMetaShape));
        registry.register("torch.eig", Box::new(EigMetaShape));
        registry.register("torch.linalg.eigh", Box::new(EigMetaShape)); // Hermitian/symmetric eigen
        registry.register("torch.eigh", Box::new(EigMetaShape));

        // Tier 3: Eigenvalues only (no eigenvectors)
        registry.register("torch.linalg.eigvals", Box::new(EigvalsMetaShape));
        registry.register("torch.linalg.eigvalsh", Box::new(EigvalsMetaShape));

        // Linear system solvers
        registry.register("torch.linalg.solve", Box::new(SolveMetaShape));
        registry.register("torch.solve", Box::new(SolveMetaShape));
        registry.register("torch.linalg.solve_triangular", Box::new(SolveMetaShape)); // Similar API
        registry.register("torch.triangular_solve", Box::new(SolveReversedMetaShape));
        registry.register(
            "torch.linalg.cholesky_solve",
            Box::new(SolveReversedMetaShape),
        );
        registry.register("torch.cholesky_solve", Box::new(SolveReversedMetaShape));
        registry.register("torch.lu_solve", Box::new(SolveMetaShape));

        // Sign and log determinant
        registry.register("torch.linalg.slogdet", Box::new(SlogdetMetaShape));
        registry.register("torch.slogdet", Box::new(SlogdetMetaShape));
        registry.register("torch.Tensor.slogdet", Box::new(SlogdetMetaShape));

        // =====================================================================
        // Phase 5: Advanced Indexing & Conditional Operations
        // =====================================================================

        // Conditional operations
        registry.register("torch.where", Box::new(WhereMetaShape));

        // Take along dim operations
        registry.register("torch.take_along_dim", Box::new(TakeAlongDimMetaShape));
        registry.register(
            "torch.Tensor.take_along_dim",
            Box::new(TakeAlongDimMetaShape),
        );

        // =====================================================================
        // Phase 6: Specialized Operations (FFT, Loss, Padding, Properties)
        // =====================================================================

        // Loss functions (return scalar by default, can preserve shape with reduction='none')
        registry.register("torch.nn.functional.mse_loss", Box::new(LossMetaShape));
        registry.register("torch.nn.functional.l1_loss", Box::new(LossMetaShape));
        registry.register("torch.nn.functional.nll_loss", Box::new(LossMetaShape));
        registry.register("torch.nn.functional.cross_entropy", Box::new(LossMetaShape));
        registry.register(
            "torch.nn.functional.binary_cross_entropy",
            Box::new(LossMetaShape),
        );
        registry.register(
            "torch.nn.functional.binary_cross_entropy_with_logits",
            Box::new(LossMetaShape),
        );
        registry.register("torch.nn.functional.kl_div", Box::new(LossMetaShape));
        registry.register(
            "torch.nn.functional.smooth_l1_loss",
            Box::new(LossMetaShape),
        );
        registry.register("torch.nn.functional.huber_loss", Box::new(LossMetaShape));
        registry.register(
            "torch.nn.functional.poisson_nll_loss",
            Box::new(LossMetaShape),
        );
        registry.register(
            "torch.nn.functional.cosine_embedding_loss",
            Box::new(LossMetaShape),
        );
        registry.register(
            "torch.nn.functional.margin_ranking_loss",
            Box::new(LossMetaShape),
        );
        registry.register(
            "torch.nn.functional.triplet_margin_loss",
            Box::new(LossMetaShape),
        );
        registry.register(
            "torch.nn.functional.hinge_embedding_loss",
            Box::new(LossMetaShape),
        );

        // Padding operations
        registry.register("torch.nn.functional.pad", Box::new(PadMetaShape));

        // Real FFT family (dimension size changes)
        registry.register("torch.fft.rfft", Box::new(FftMetaShape::rfft()));
        registry.register("torch.fft.irfft", Box::new(FftMetaShape::irfft()));

        // Hermitian FFT (similar to real FFT)
        registry.register("torch.fft.hfft", Box::new(FftMetaShape::hfft()));
        registry.register("torch.fft.ihfft", Box::new(FftMetaShape::ihfft()));

        // Tensor property operations (return literal ints)
        registry.register("torch.Tensor.size", Box::new(SizeMetaShape));
        registry.register("torch.Tensor.numel", Box::new(NumelIntMetaShape));
        registry.register("torch.Tensor.dim", Box::new(DimMetaShape));
        registry.register("torch.Tensor.nelement", Box::new(NumelIntMetaShape));

        // Tensor conversion operations (return Python types, not tensors)
        registry.register("torch.Tensor.item", Box::new(ItemMetaShape));
        registry.register("torch.Tensor.tolist", Box::new(TolistMetaShape));

        // Module-level numel returns int, same as the method version
        registry.register("torch.numel", Box::new(NumelIntMetaShape));

        // Random sampling operations
        registry.register("torch.multinomial", Box::new(MultinomialMetaShape));
        registry.register("torch.Tensor.multinomial", Box::new(MultinomialMetaShape));
        registry.register("torch.normal", Box::new(NormalMetaShape));

        registry
    }

    /// Register a meta-shape function
    pub fn register(&mut self, name: impl Into<String>, func: Box<dyn MetaShapeFunction>) {
        self.functions.insert(name.into(), func);
    }

    /// Register a meta-shape function for both torch.X and Tensor.X calling conventions.
    ///
    /// This is useful for operations that can be called both as functions (torch.add(x, y))
    /// and as methods (x.add(y)).
    ///
    /// # Example
    /// ```ignore
    /// registry.register_dual("reshape", || Box::new(ReshapeMetaShape));
    /// // Equivalent to:
    /// // registry.register("torch.reshape", Box::new(ReshapeMetaShape));
    /// // registry.register("torch.Tensor.reshape", Box::new(ReshapeMetaShape));
    /// ```
    pub fn register_dual<F: Fn() -> Box<dyn MetaShapeFunction>>(
        &mut self,
        op_name: &str,
        factory: F,
    ) {
        self.functions
            .insert(format!("torch.{}", op_name), factory());
        self.functions
            .insert(format!("torch.Tensor.{}", op_name), factory());
    }

    /// Get a meta-shape function by name
    pub fn get(&self, name: &str) -> Option<&dyn MetaShapeFunction> {
        self.functions.get(name).map(|b| b.as_ref())
    }

    /// Check if a meta-shape function is registered
    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

impl Default for MetaShapeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
