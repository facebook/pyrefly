/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Meta-shape op registry.
//!
//! This module registers all PyTorch op shape functions in `TensorOpsRegistry`.
//! All op definitions are expressed in the DSL (parsed by `meta_shape_dsl.rs`) and
//! interpreted directly — no IR layer.

use std::collections::HashMap;
use std::sync::Arc;

use crate::meta_shape_dsl::DslFnDef;
use crate::meta_shape_dsl::DslMetaShapeFunction;
use crate::meta_shape_dsl::MetaShapeFunction;

// ============================================================================
// DSL-based MetaShapeFunction construction
// ============================================================================

/// Look up a DSL function by name and create a `DslMetaShapeFunction`.
fn dsl_fn(
    fn_lookup: &Arc<HashMap<String, Arc<DslFnDef>>>,
    name: &str,
) -> Box<dyn MetaShapeFunction> {
    let fn_def = Arc::clone(
        fn_lookup
            .get(name)
            .unwrap_or_else(|| panic!("DSL function `{name}` not found")),
    );
    Box::new(DslMetaShapeFunction {
        fn_def,
        fn_lookup: Arc::clone(fn_lookup),
    })
}

// ============================================================================
// Meta-Shape Registry
// ============================================================================

/// Registry mapping PyTorch op names to their shape functions.
///
/// All shape functions are backed by DSL definitions (see `meta_shape_dsl.rs`).
/// The DSL source is parsed once at registry construction time. Parsed definitions
/// are shared via `Arc` across all shape function instances.
pub struct TensorOpsRegistry {
    functions: HashMap<String, Box<dyn MetaShapeFunction>>,
}

impl TensorOpsRegistry {
    /// Create a new registry with built-in meta-shape functions.
    pub fn new() -> Self {
        // Parse DSL once; definitions are shared via Arc across all instances.
        let dsl_fns: Vec<Arc<DslFnDef>> = crate::meta_shape_dsl::parse_dsl(DSL_SOURCE)
            .expect("DSL source in tensor_ops_registry.rs has errors")
            .into_iter()
            .map(Arc::new)
            .collect();
        // Build function lookup table once, shared by all DslMetaShapeFunctions.
        let fn_lookup: Arc<HashMap<String, Arc<DslFnDef>>> = Arc::new(
            dsl_fns
                .iter()
                .map(|f| (f.name.clone(), Arc::clone(f)))
                .collect(),
        );
        let mut registry = Self {
            functions: HashMap::new(),
        };

        // Shape manipulation
        registry.register_dual("reshape", || dsl_fn(&fn_lookup, "reshape_ir"));
        registry.register("torch.cat", dsl_fn(&fn_lookup, "cat_ir"));
        registry.register("torch.broadcast_to", dsl_fn(&fn_lookup, "broadcast_to_ir"));
        registry.register_dual("squeeze", || dsl_fn(&fn_lookup, "squeeze_ir"));
        registry.register_dual("unsqueeze", || dsl_fn(&fn_lookup, "unsqueeze_ir"));
        registry.register_dual("transpose", || dsl_fn(&fn_lookup, "transpose_ir"));
        // torch.permute takes dims as a tuple; Tensor.permute takes *dims (variadic).
        // Both use the same DSL function; parameter binding matches by name.
        registry.register("torch.permute", dsl_fn(&fn_lookup, "permute_ir"));
        registry.register("torch.Tensor.permute", dsl_fn(&fn_lookup, "permute_ir"));
        registry.register("torch.flatten", dsl_fn(&fn_lookup, "flatten_ir"));
        registry.register("torch.stack", dsl_fn(&fn_lookup, "stack_ir"));
        registry.register("torch.tile", dsl_fn(&fn_lookup, "tile_ir"));
        registry.register("torch.view", dsl_fn(&fn_lookup, "reshape_ir"));
        registry.register("torch.unbind", dsl_fn(&fn_lookup, "unbind_ir"));
        registry.register("torch.Tensor.unbind", dsl_fn(&fn_lookup, "unbind_ir"));
        registry.register("torch.movedim", dsl_fn(&fn_lookup, "movedim_ir"));
        registry.register("torch.moveaxis", dsl_fn(&fn_lookup, "movedim_ir"));
        registry.register("torch.Tensor.movedim", dsl_fn(&fn_lookup, "movedim_ir"));
        registry.register("torch.Tensor.moveaxis", dsl_fn(&fn_lookup, "movedim_ir"));
        registry.register("torch.unfold", dsl_fn(&fn_lookup, "unfold_ir"));
        registry.register("torch.Tensor.unfold", dsl_fn(&fn_lookup, "unfold_ir"));

        // Method-only shape manipulation
        registry.register("torch.Tensor.reshape", dsl_fn(&fn_lookup, "reshape_ir"));
        registry.register("torch.Tensor.view", dsl_fn(&fn_lookup, "reshape_ir"));
        registry.register("torch.Tensor.squeeze", dsl_fn(&fn_lookup, "squeeze_ir"));
        registry.register("torch.Tensor.flatten", dsl_fn(&fn_lookup, "flatten_ir"));
        registry.register("torch.Tensor.tile", dsl_fn(&fn_lookup, "tile_ir"));
        registry.register(
            "torch.Tensor.diag_embed",
            dsl_fn(&fn_lookup, "diag_embed_ir"),
        );
        registry.register("torch.Tensor.repeat", dsl_fn(&fn_lookup, "repeat_ir"));
        registry.register("torch.Tensor.expand", dsl_fn(&fn_lookup, "expand_ir"));

        // Reduction operations
        registry.register_dual("sum", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("mean", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("prod", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("min", || dsl_fn(&fn_lookup, "min_max_median_ir"));
        registry.register_dual("max", || dsl_fn(&fn_lookup, "min_max_median_ir"));
        registry.register_dual("all", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("any", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("std", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("var", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("argmax", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register_dual("argmin", || dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register("torch.median", dsl_fn(&fn_lookup, "min_max_median_ir"));
        registry.register("torch.logsumexp", dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register("torch.count_nonzero", dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register("torch.aminmax", dsl_fn(&fn_lookup, "aminmax_ir"));
        registry.register("torch.norm", dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register("torch.mode", dsl_fn(&fn_lookup, "tuple_reduce_ir"));
        registry.register("torch.topk", dsl_fn(&fn_lookup, "topk_ir"));
        registry.register("torch.kthvalue", dsl_fn(&fn_lookup, "tuple_reduce_ir"));
        registry.register("torch.var_mean", dsl_fn(&fn_lookup, "aminmax_ir"));
        registry.register("torch.std_mean", dsl_fn(&fn_lookup, "aminmax_ir"));

        // Reduction method versions
        registry.register(
            "torch.Tensor.median",
            dsl_fn(&fn_lookup, "min_max_median_ir"),
        );
        registry.register("torch.Tensor.logsumexp", dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register(
            "torch.Tensor.count_nonzero",
            dsl_fn(&fn_lookup, "reduce_ir"),
        );
        registry.register("torch.Tensor.aminmax", dsl_fn(&fn_lookup, "aminmax_ir"));
        registry.register("torch.Tensor.norm", dsl_fn(&fn_lookup, "reduce_ir"));
        registry.register("torch.Tensor.mode", dsl_fn(&fn_lookup, "tuple_reduce_ir"));
        registry.register("torch.Tensor.topk", dsl_fn(&fn_lookup, "topk_ir"));
        registry.register(
            "torch.Tensor.kthvalue",
            dsl_fn(&fn_lookup, "tuple_reduce_ir"),
        );

        // Indexing/slicing
        registry.register("torch.select", dsl_fn(&fn_lookup, "select_ir"));
        registry.register("torch.narrow", dsl_fn(&fn_lookup, "narrow_ir"));
        registry.register("torch.split", dsl_fn(&fn_lookup, "split_ir"));
        registry.register("torch.chunk", dsl_fn(&fn_lookup, "chunk_ir"));
        registry.register("torch.index_select", dsl_fn(&fn_lookup, "index_select_ir"));
        registry.register("torch.Tensor.select", dsl_fn(&fn_lookup, "select_ir"));
        registry.register("torch.Tensor.narrow", dsl_fn(&fn_lookup, "narrow_ir"));
        registry.register("torch.Tensor.split", dsl_fn(&fn_lookup, "split_ir"));
        registry.register("torch.Tensor.chunk", dsl_fn(&fn_lookup, "chunk_ir"));
        registry.register(
            "torch.Tensor.index_select",
            dsl_fn(&fn_lookup, "index_select_ir"),
        );

        // Tensor creation
        registry.register("torch.randn", dsl_fn(&fn_lookup, "randn_ir"));
        registry.register("torch.rand", dsl_fn(&fn_lookup, "randn_ir"));
        registry.register("torch.zeros", dsl_fn(&fn_lookup, "randn_ir"));
        registry.register("torch.ones", dsl_fn(&fn_lookup, "randn_ir"));
        registry.register("torch.empty", dsl_fn(&fn_lookup, "randn_ir"));
        registry.register("torch.full", dsl_fn(&fn_lookup, "randn_ir"));
        registry.register("torch.arange", dsl_fn(&fn_lookup, "arange_ir"));
        registry.register("torch.linspace", dsl_fn(&fn_lookup, "linspace_ir"));
        registry.register("torch.eye", dsl_fn(&fn_lookup, "eye_ir"));
        registry.register("torch.diag_embed", dsl_fn(&fn_lookup, "diag_embed_ir"));
        registry.register("torch.tril_indices", dsl_fn(&fn_lookup, "tri_indices_ir"));
        registry.register("torch.triu_indices", dsl_fn(&fn_lookup, "tri_indices_ir"));

        // Linear algebra
        registry.register("torch.matmul", dsl_fn(&fn_lookup, "matmul_ir"));
        registry.register("torch.mv", dsl_fn(&fn_lookup, "mv_ir"));
        registry.register("torch.outer", dsl_fn(&fn_lookup, "outer_ir"));
        registry.register("torch.tensordot", dsl_fn(&fn_lookup, "tensordot_ir"));
        registry.register("torch.einsum", dsl_fn(&fn_lookup, "einsum_ir"));
        registry.register("torch.Tensor.matmul", dsl_fn(&fn_lookup, "matmul_ir"));
        registry.register("torch.Tensor.__matmul__", dsl_fn(&fn_lookup, "matmul_ir"));
        registry.register("torch.Tensor.mv", dsl_fn(&fn_lookup, "mv_ir"));

        // Eigenvalue decomposition
        registry.register("torch.linalg.eig", dsl_fn(&fn_lookup, "eig_ir"));
        registry.register("torch.eig", dsl_fn(&fn_lookup, "eig_ir"));
        registry.register("torch.linalg.eigh", dsl_fn(&fn_lookup, "eig_ir"));
        registry.register("torch.eigh", dsl_fn(&fn_lookup, "eig_ir"));
        registry.register("torch.linalg.eigvals", dsl_fn(&fn_lookup, "eigvals_ir"));
        registry.register("torch.linalg.eigvalsh", dsl_fn(&fn_lookup, "eigvals_ir"));

        // Linear solvers
        registry.register("torch.linalg.solve", dsl_fn(&fn_lookup, "solve_ir"));
        registry.register("torch.solve", dsl_fn(&fn_lookup, "solve_ir"));
        registry.register(
            "torch.linalg.solve_triangular",
            dsl_fn(&fn_lookup, "solve_ir"),
        );
        registry.register(
            "torch.triangular_solve",
            dsl_fn(&fn_lookup, "solve_reversed_ir"),
        );
        registry.register(
            "torch.linalg.cholesky_solve",
            dsl_fn(&fn_lookup, "solve_reversed_ir"),
        );
        registry.register(
            "torch.cholesky_solve",
            dsl_fn(&fn_lookup, "solve_reversed_ir"),
        );
        registry.register("torch.lu_solve", dsl_fn(&fn_lookup, "solve_ir"));

        // Determinant
        registry.register("torch.linalg.slogdet", dsl_fn(&fn_lookup, "slogdet_ir"));
        registry.register("torch.slogdet", dsl_fn(&fn_lookup, "slogdet_ir"));
        registry.register("torch.Tensor.slogdet", dsl_fn(&fn_lookup, "slogdet_ir"));

        // Convolution
        registry.register("torch.nn.functional.conv1d", dsl_fn(&fn_lookup, "conv_ir"));
        registry.register("torch.nn.functional.conv2d", dsl_fn(&fn_lookup, "conv_ir"));
        registry.register("torch.nn.functional.conv3d", dsl_fn(&fn_lookup, "conv_ir"));
        registry.register(
            "torch.nn.functional.conv_transpose1d",
            dsl_fn(&fn_lookup, "conv_transpose_ir"),
        );
        registry.register(
            "torch.nn.functional.conv_transpose2d",
            dsl_fn(&fn_lookup, "conv_transpose_ir"),
        );
        registry.register(
            "torch.nn.functional.conv_transpose3d",
            dsl_fn(&fn_lookup, "conv_transpose_ir"),
        );

        // Pooling
        registry.register(
            "torch.nn.functional.max_pool1d",
            dsl_fn(&fn_lookup, "pool_ir"),
        );
        registry.register(
            "torch.nn.functional.max_pool2d",
            dsl_fn(&fn_lookup, "pool_ir"),
        );
        registry.register(
            "torch.nn.functional.max_pool3d",
            dsl_fn(&fn_lookup, "pool_ir"),
        );
        registry.register(
            "torch.nn.functional.avg_pool1d",
            dsl_fn(&fn_lookup, "pool_ir"),
        );
        registry.register(
            "torch.nn.functional.avg_pool2d",
            dsl_fn(&fn_lookup, "pool_ir"),
        );
        registry.register(
            "torch.nn.functional.avg_pool3d",
            dsl_fn(&fn_lookup, "pool_ir"),
        );
        registry.register(
            "torch.nn.functional.adaptive_max_pool1d",
            dsl_fn(&fn_lookup, "adaptive_pool_ir"),
        );
        registry.register(
            "torch.nn.functional.adaptive_max_pool2d",
            dsl_fn(&fn_lookup, "adaptive_pool_ir"),
        );
        registry.register(
            "torch.nn.functional.adaptive_max_pool3d",
            dsl_fn(&fn_lookup, "adaptive_pool_ir"),
        );
        registry.register(
            "torch.nn.functional.adaptive_avg_pool1d",
            dsl_fn(&fn_lookup, "adaptive_pool_ir"),
        );
        registry.register(
            "torch.nn.functional.adaptive_avg_pool2d",
            dsl_fn(&fn_lookup, "adaptive_pool_ir"),
        );
        registry.register(
            "torch.nn.functional.adaptive_avg_pool3d",
            dsl_fn(&fn_lookup, "adaptive_pool_ir"),
        );

        // Interpolation
        registry.register(
            "torch.nn.functional.interpolate",
            dsl_fn(&fn_lookup, "interpolate_ir"),
        );
        registry.register(
            "torch.nn.functional.upsample",
            dsl_fn(&fn_lookup, "interpolate_ir"),
        );

        // Conditional operations
        registry.register("torch.where", dsl_fn(&fn_lookup, "where_ir"));
        registry.register(
            "torch.take_along_dim",
            dsl_fn(&fn_lookup, "take_along_dim_ir"),
        );
        registry.register(
            "torch.Tensor.take_along_dim",
            dsl_fn(&fn_lookup, "take_along_dim_ir"),
        );

        // Loss functions
        registry.register(
            "torch.nn.functional.mse_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register("torch.nn.functional.l1_loss", dsl_fn(&fn_lookup, "loss_ir"));
        registry.register(
            "torch.nn.functional.nll_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.cross_entropy",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.binary_cross_entropy",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.binary_cross_entropy_with_logits",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register("torch.nn.functional.kl_div", dsl_fn(&fn_lookup, "loss_ir"));
        registry.register(
            "torch.nn.functional.smooth_l1_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.huber_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.poisson_nll_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.cosine_embedding_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.margin_ranking_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.triplet_margin_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );
        registry.register(
            "torch.nn.functional.hinge_embedding_loss",
            dsl_fn(&fn_lookup, "loss_ir"),
        );

        // Padding
        registry.register("torch.nn.functional.pad", dsl_fn(&fn_lookup, "pad_ir"));

        // FFT
        registry.register("torch.fft.rfft", dsl_fn(&fn_lookup, "rfft_ir"));
        registry.register("torch.fft.irfft", dsl_fn(&fn_lookup, "irfft_ir"));
        registry.register("torch.fft.hfft", dsl_fn(&fn_lookup, "irfft_ir"));
        registry.register("torch.fft.ihfft", dsl_fn(&fn_lookup, "rfft_ir"));

        // Tensor properties
        registry.register("torch.Tensor.size", dsl_fn(&fn_lookup, "size_ir"));
        registry.register("torch.Tensor.numel", dsl_fn(&fn_lookup, "numel_ir"));
        registry.register("torch.Tensor.dim", dsl_fn(&fn_lookup, "dim_ir"));
        registry.register("torch.Tensor.nelement", dsl_fn(&fn_lookup, "numel_ir"));
        registry.register("torch.Tensor.item", dsl_fn(&fn_lookup, "item_ir"));
        registry.register("torch.Tensor.tolist", dsl_fn(&fn_lookup, "tolist_ir"));
        registry.register("torch.numel", dsl_fn(&fn_lookup, "numel_ir"));

        // Random sampling
        registry.register("torch.multinomial", dsl_fn(&fn_lookup, "multinomial_ir"));
        registry.register(
            "torch.Tensor.multinomial",
            dsl_fn(&fn_lookup, "multinomial_ir"),
        );
        registry.register("torch.normal", dsl_fn(&fn_lookup, "normal_ir"));

        registry
    }

    /// Register a meta-shape function.
    pub fn register(&mut self, name: impl Into<String>, func: Box<dyn MetaShapeFunction>) {
        self.functions.insert(name.into(), func);
    }

    /// Register a meta-shape function for both `torch.X` and `torch.Tensor.X`.
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

    /// Get a meta-shape function by name.
    pub fn get(&self, name: &str) -> Option<&dyn MetaShapeFunction> {
        self.functions.get(name).map(|b| b.as_ref())
    }
}

impl Default for TensorOpsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DSL source code
// ============================================================================

/// The full DSL source defining all tensor shape ops and utility functions.
/// This is valid Python syntax (a strict subset) that we parse with Pyrefly's parser.
const DSL_SOURCE: &str = r#"
def normalize_dim(rank: int, dim: int) -> int:
    if dim < 0:
        return dim + rank
    return dim

def int_max(a: int, b: int) -> int:
    if a > b:
        return a
    return b

def replace_dim(dims: list[int | symint], i: int, value: int | symint) -> list[int | symint]:
    return dims[:i] + [value] + dims[i + 1:]

def remove_dim(dims: list[int | symint], i: int) -> list[int | symint]:
    return dims[:i] + dims[i + 1:]

def insert_dim(dims: list[int | symint], i: int, value: int | symint) -> list[int | symint]:
    return dims[:i] + [value] + dims[i:]

def broadcast(a: list[int | symint], b: list[int | symint]) -> list[int | symint]:
    max_len = int_max(len(a), len(b))
    padded_a = [1 for _ in range(max_len - len(a))] + a
    padded_b = [1 for _ in range(max_len - len(b))] + b
    return [bd if ad == 1 else ad for ad, bd in zip(padded_a, padded_b)]

def broadcast_int(expr: int | symint | list[int | symint], n: int) -> list[int | symint]:
    if isinstance(expr, list):
        return expr
    return [expr for _ in range(n)]

def reduce_shape(dims: list[int | symint], dim: int | list[int] | None, keepdim: bool) -> list[int | symint]:
    if dim == None:
        if keepdim:
            return [1 for _ in range(len(dims))]
        return []
    dim_list = dim if isinstance(dim, list) else [dim]
    norm = [normalize_dim(len(dims), d) for d in dim_list]
    return [1 if i in norm else elem for i, elem in enumerate(dims) if not (i in norm) or keepdim]

def contains(lst: list[int], val: int) -> bool:
    return len([x for x in lst if x == val]) > 0

def scatter(size: int, indices: list[int], values: list[int], fill: int) -> list[int]:
    matches = [[k for k in range(len(indices)) if indices[k] == i] for i in range(size)]
    return [values[m[0]] if len(m) > 0 else fill for m in matches]

def move_dims(dims: list[int | symint], source: int | list[int], dest: int | list[int], rank: int) -> list[int | symint]:
    src = broadcast_int(source, 1)
    dst = broadcast_int(dest, 1)
    src_norm = [normalize_dim(rank, s) for s in src]
    dst_norm = [normalize_dim(rank, d) for d in dst]
    non_dst = [i for i in range(rank) if not contains(dst_norm, i)]
    remaining = [i for i in range(rank) if not contains(src_norm, i)]
    perm = scatter(rank, dst_norm + non_dst, src_norm + remaining, 0)
    return [dims[p] for p in perm]

def conv_spatial_out(input_dim: int | symint, kernel: int | symint, stride: int | symint, padding: int | symint, dilation: int | symint) -> int | symint:
    return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

def reshape_ir(self: Tensor, shape: list[int | symint]) -> Tensor:
    minus_one_count = len([d for d in shape if d == -1])
    if minus_one_count > 1:
        raise Error("can only specify one unknown dimension as -1")
    has_bad_neg = len([d for d in shape if isinstance(d, int) and d < -1]) > 0
    if has_bad_neg:
        raise Error("invalid negative dimension value (only -1 is allowed)")
    has_zero = len([d for d in shape if isinstance(d, int) and d == 0]) > 0
    if has_zero:
        raise Error("reshape dimensions cannot contain 0")
    if minus_one_count > 0:
        known = torch_shapes.prod([d for d in shape if d != -1])
        total = torch_shapes.prod(self.shape)
        if isinstance(total, int) and isinstance(known, int) and total % known != 0:
            raise Error("could not infer size for dimension -1: expected " + str(total) + " to be divisible by " + str(known))
        return Tensor(shape=[total // known if d == -1 else d for d in shape])
    return Tensor(shape=shape)

def squeeze_ir(self: Tensor, dim: int | None = None) -> Tensor:
    if dim == None:
        return Tensor(shape=[d for d in self.shape if d != 1])
    idx = normalize_dim(len(self.shape), dim)
    return Tensor(shape=[d for i, d in enumerate(self.shape) if not (i == idx and d == 1)])

def unsqueeze_ir(self: Tensor, dim: int) -> Tensor:
    d = normalize_dim(len(self.shape) + 1, dim)
    return Tensor(shape=insert_dim(self.shape, d, 1))

def transpose_ir(self: Tensor, dim0: int, dim1: int) -> Tensor:
    rank = len(self.shape)
    d0 = normalize_dim(rank, dim0)
    d1 = normalize_dim(rank, dim1)
    return Tensor(shape=[self.shape[d1] if i == d0 else self.shape[d0] if i == d1 else d for i, d in enumerate(self.shape)])

def permute_ir(self: Tensor, dims: list[int]) -> Tensor:
    rank = len(self.shape)
    if len(dims) != rank:
        raise Error("permute: expected " + str(rank) + " dims, got " + str(len(dims)))
    return Tensor(shape=[self.shape[normalize_dim(rank, d)] for d in dims])

def flatten_ir(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    rank = len(self.shape)
    s = normalize_dim(rank, start_dim)
    e = normalize_dim(rank, end_dim)
    return Tensor(shape=self.shape[:s] + [torch_shapes.prod(self.shape[s:e + 1])] + self.shape[e + 1:])

def expand_ir(self: Tensor, sizes: list[int | symint]) -> Tensor:
    return Tensor(shape=[d if t == -1 else t for d, t in zip(self.shape, sizes)])

def repeat_ir(self: Tensor, sizes: list[int | symint]) -> Tensor:
    return Tensor(shape=[d * r for d, r in zip(self.shape, sizes)])

def unbind_ir(self: Tensor, dim: int = 0) -> list[Tensor]:
    d = normalize_dim(len(self.shape), dim)
    return [Tensor(shape=remove_dim(self.shape, d)), ...]

def movedim_ir(self: Tensor, source: int | list[int], destination: int | list[int]) -> Tensor:
    return Tensor(shape=move_dims(self.shape, source, destination, len(self.shape)))

def unfold_ir(self: Tensor, dimension: int, size: int | symint, step: int = 1) -> Tensor:
    d = normalize_dim(len(self.shape), dimension)
    new_dim = (self.shape[d] - size) // step + 1
    return Tensor(shape=replace_dim(self.shape, d, new_dim) + [size])

def cat_ir(tensors: list[Tensor], dim: int = 0) -> Tensor:
    first = tensors[0]
    d = normalize_dim(len(first.shape), dim)
    return Tensor(shape=[torch_shapes.sum([t.shape[i] for t in tensors]) if i == d else dim_val for i, dim_val in enumerate(first.shape)])

def stack_ir(tensors: list[Tensor], dim: int = 0) -> Tensor:
    first = tensors[0]
    d = normalize_dim(len(first.shape) + 1, dim)
    return Tensor(shape=insert_dim(first.shape, d, len(tensors)))

def broadcast_to_ir(self: Tensor, shape: list[int | symint]) -> Tensor:
    return Tensor(shape=shape)

def tile_ir(self: Tensor, dims: list[int]) -> Tensor:
    rank = len(self.shape)
    if len(dims) > rank:
        extra = len(dims) - rank
        return Tensor(shape=[r for r in dims[:extra]] + [d * r for d, r in zip(self.shape, dims[extra:])])
    return Tensor(shape=[d * r for d, r in zip(self.shape, dims)])

def select_ir(self: Tensor, dim: int) -> Tensor:
    d = normalize_dim(len(self.shape), dim)
    return Tensor(shape=remove_dim(self.shape, d))

def narrow_ir(self: Tensor, dim: int, length: int | symint) -> Tensor:
    return Tensor(shape=replace_dim(self.shape, normalize_dim(len(self.shape), dim), length))

def split_ir(self: Tensor, split_size_or_sections: int | symint | list[int | symint] | None = None, dim: int = 0) -> list[Tensor]:
    d = normalize_dim(len(self.shape), dim)
    if isinstance(split_size_or_sections, list):
        return [Tensor(shape=replace_dim(self.shape, d, section)) for section in split_size_or_sections]
    if isinstance(split_size_or_sections, int):
        dim_val = self.shape[d]
        if isinstance(dim_val, int):
            count = (dim_val + split_size_or_sections - 1) // split_size_or_sections
            return [Tensor(shape=replace_dim(self.shape, d, split_size_or_sections if i < count - 1 else dim_val - (count - 1) * split_size_or_sections)) for i in range(count)]
        return [Tensor(shape=replace_dim(self.shape, d, split_size_or_sections)), ...]
    if split_size_or_sections != None:
        quotient = self.shape[d] // split_size_or_sections
        if isinstance(quotient, int):
            return [Tensor(shape=replace_dim(self.shape, d, split_size_or_sections)) for _ in range(quotient)]
        return [Tensor(shape=replace_dim(self.shape, d, split_size_or_sections)), ...]
    return Unknown

def chunk_ir(self: Tensor, chunks: int, dim: int = 0) -> list[Tensor]:
    d = normalize_dim(len(self.shape), dim)
    dim_val = self.shape[d]
    if isinstance(dim_val, int):
        chunk_size = (dim_val + chunks - 1) // chunks
        return [Tensor(shape=replace_dim(self.shape, d, chunk_size if i < chunks - 1 else dim_val - (chunks - 1) * chunk_size)) for i in range(chunks)]
    return [Tensor(shape=replace_dim(self.shape, d, dim_val // chunks)) for i in range(chunks)]

def index_select_ir(self: Tensor, dim: int, index: Tensor) -> Tensor:
    return Tensor(shape=replace_dim(self.shape, normalize_dim(len(self.shape), dim), index.shape[0]))

def reduce_ir(self: Tensor, dim: int | list[int] | None = None, keepdim: bool = False) -> Tensor:
    return Tensor(shape=reduce_shape(self.shape, dim, keepdim))

def min_max_median_ir(self: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    if dim == None:
        return Tensor(shape=[])
    s = reduce_shape(self.shape, dim, keepdim)
    return [Tensor(shape=s), Tensor(shape=s)]

def aminmax_ir(self: Tensor, dim: int | list[int] | None = None, keepdim: bool = False) -> [Tensor, Tensor]:
    s = reduce_shape(self.shape, dim, keepdim)
    return [Tensor(shape=s), Tensor(shape=s)]

def tuple_reduce_ir(self: Tensor, dim: int = -1, keepdim: bool = False) -> [Tensor, Tensor]:
    s = reduce_shape(self.shape, dim, keepdim)
    return [Tensor(shape=s), Tensor(shape=s)]

def topk_ir(self: Tensor, k: int | symint, dim: int = -1) -> [Tensor, Tensor]:
    s = replace_dim(self.shape, normalize_dim(len(self.shape), dim), k)
    return [Tensor(shape=s), Tensor(shape=s)]

def randn_ir(size: list[int | symint]) -> Tensor:
    return Tensor(shape=size)

def linspace_ir(steps: int | symint) -> Tensor:
    return Tensor(shape=[steps])

def eye_ir(n: int | symint, m: int | symint | None = None) -> Tensor:
    if m == None:
        return Tensor(shape=[n, n])
    return Tensor(shape=[n, m])

def arange_ir(start: int | symint | None = None, end: int | symint | None = None, step: int | symint | None = None) -> Tensor:
    if start != None and end != None and step != None:
        return Tensor(shape=[(end - start) // step])
    if start != None and end != None:
        return Tensor(shape=[end - start])
    if end != None:
        return Tensor(shape=[end])
    if start != None:
        return Tensor(shape=[start])
    return Unknown

def normal_ir(mean: Tensor | None = None, std: Tensor | None = None, size: list[int] | None = None) -> Tensor:
    if size != None:
        return Tensor(shape=[s for s in size])
    if mean != None:
        return Tensor(shape=mean.shape)
    if std != None:
        return Tensor(shape=std.shape)
    return Unknown

def diag_embed_ir(self: Tensor, offset: int = 0) -> Tensor:
    new_dim = self.shape[-1] + (offset if offset >= 0 else -offset)
    return Tensor(shape=self.shape[:-1] + [new_dim, new_dim])

def tri_indices_ir(row: int | symint, col: int | symint, offset: int = 0) -> Tensor:
    return Tensor(shape=[2, 0])

def matmul_ir(self: Tensor, other: Tensor) -> Tensor:
    r1 = len(self.shape)
    r2 = len(other.shape)
    if r1 == 1 and r2 == 1:
        return Tensor(shape=[])
    if r1 == 1 and r2 == 2:
        return Tensor(shape=[other.shape[1]])
    if r1 == 2 and r2 == 1:
        return Tensor(shape=[self.shape[0]])
    if r1 == 2 and r2 == 2:
        return Tensor(shape=[self.shape[0], other.shape[1]])
    if r1 == 2 and r2 >= 3:
        return Tensor(shape=other.shape[:-2] + [self.shape[0]] + [other.shape[-1]])
    if r1 >= 3 and r2 == 2:
        return Tensor(shape=self.shape[:-2] + [self.shape[-2]] + [other.shape[1]])
    if r1 >= 3 and r2 >= 3:
        return Tensor(shape=broadcast(self.shape[:-2], other.shape[:-2]) + [self.shape[-2]] + [other.shape[-1]])
    return Unknown

def mv_ir(self: Tensor, vec: Tensor) -> Tensor:
    if len(self.shape) != 2:
        raise Error("mv expects 2D matrix, got " + str(len(self.shape)) + "D tensor")
    if len(vec.shape) != 1:
        raise Error("mv expects 1D vector, got " + str(len(vec.shape)) + "D tensor")
    return Tensor(shape=[self.shape[0]])

def outer_ir(self: Tensor, vec2: Tensor) -> Tensor:
    if len(self.shape) != 1 or len(vec2.shape) != 1:
        raise Error("outer expects 1D tensors, got " + str(len(self.shape)) + "D and " + str(len(vec2.shape)) + "D")
    return Tensor(shape=[self.shape[0], vec2.shape[0]])

def tensordot_ir(self: Tensor, other: Tensor, dims: int) -> Tensor:
    return Tensor(shape=self.shape[:len(self.shape) - dims] + other.shape[dims:])

def apply_einsum(output_map: list[list[int]], check_pairs: list[list[int]], inputs: list[Tensor]) -> Tensor:
    bad_dims = [1 for i0, d0, i1, d1 in check_pairs if isinstance(inputs[i0].shape[d0], int) and isinstance(inputs[i1].shape[d1], int) and inputs[i0].shape[d0] != inputs[i1].shape[d1]]
    if len(bad_dims) > 0:
        raise Error("einsum: inconsistent dimensions for repeated index")
    return Tensor(shape=[inputs[inp].shape[dim] for inp, dim in output_map])

def einsum_ir(spec: str, inputs: list[Tensor] | None = None) -> Tensor:
    if inputs != None:
        output_map, check_pairs = torch_shapes.parse_einsum_equation(spec)
        return apply_einsum(output_map, check_pairs, inputs)
    return Unknown

def eigvals_ir(self: Tensor) -> Tensor:
    if len(self.shape) < 2:
        raise Error("eigvals requires at least 2D input, got " + str(len(self.shape)) + "D tensor")
    return Tensor(shape=self.shape[:-2] + [self.shape[-2]])

def eig_ir(self: Tensor) -> [Tensor, Tensor]:
    if len(self.shape) < 2:
        raise Error("eig requires at least 2D input, got " + str(len(self.shape)) + "D tensor")
    batch = self.shape[:-2]
    return [Tensor(shape=batch + [self.shape[-2]]), Tensor(shape=batch + self.shape[-2:])]

def slogdet_ir(self: Tensor) -> [Tensor, Tensor]:
    if len(self.shape) < 2:
        raise Error("slogdet requires at least 2D input, got " + str(len(self.shape)) + "D tensor")
    return [Tensor(shape=self.shape[:-2]), Tensor(shape=self.shape[:-2])]

def solve_ir(self: Tensor, other: Tensor) -> Tensor:
    return Tensor(shape=other.shape)

def solve_reversed_ir(self: Tensor, other: Tensor) -> Tensor:
    return Tensor(shape=self.shape)

def conv_ir(self: Tensor, weight: Tensor, stride: int | list[int] = 1, padding: int | list[int] = 0, dilation: int | list[int] = 1) -> Tensor:
    spatial_dims = len(self.shape) - 2
    stride_list = broadcast_int(stride, spatial_dims)
    padding_list = broadcast_int(padding, spatial_dims)
    dilation_list = broadcast_int(dilation, spatial_dims)
    return Tensor(shape=[self.shape[0], weight.shape[0]] + [conv_spatial_out(s, k, st, p, dil) for s, k, st, p, dil in zip(self.shape[2:], weight.shape[2:], stride_list, padding_list, dilation_list)])

def conv_transpose_ir(self: Tensor, weight: Tensor, stride: int | list[int] = 1, padding: int | list[int] = 0, output_padding: int | list[int] = 0, dilation: int | list[int] = 1) -> Tensor:
    spatial_dims = len(self.shape) - 2
    stride_list = broadcast_int(stride, spatial_dims)
    padding_list = broadcast_int(padding, spatial_dims)
    outpad_list = broadcast_int(output_padding, spatial_dims)
    dilation_list = broadcast_int(dilation, spatial_dims)
    return Tensor(shape=[self.shape[0], weight.shape[1]] + [(s - 1) * st - 2 * p + dil * (k - 1) + op + 1 for s, k, st, p, op, dil in zip(self.shape[2:], weight.shape[2:], stride_list, padding_list, outpad_list, dilation_list)])

def pool_ir(self: Tensor, kernel_size: int | list[int], stride: int | list[int] | None = None, padding: int | list[int] = 0, dilation: int | list[int] = 1, return_indices: bool = False) -> Tensor:
    spatial_dims = len(self.shape) - 2
    ks_list = broadcast_int(kernel_size, spatial_dims)
    stride_list = ks_list if stride == None else broadcast_int(stride, spatial_dims)
    padding_list = broadcast_int(padding, spatial_dims)
    dilation_list = broadcast_int(dilation, spatial_dims)
    out = [self.shape[0], self.shape[1]] + [conv_spatial_out(s, k, st, p, dil) for s, k, st, p, dil in zip(self.shape[2:], ks_list, stride_list, padding_list, dilation_list)]
    if return_indices:
        return [Tensor(shape=out), Tensor(shape=out)]
    return Tensor(shape=out)

def adaptive_pool_ir(self: Tensor, output_size: int | symint | list[int | symint]) -> Tensor:
    out_sizes = broadcast_int(output_size, len(self.shape) - 2)
    return Tensor(shape=[self.shape[0], self.shape[1]] + out_sizes)

def interpolate_ir(self: Tensor, size: int | list[int] | None = None, scale_factor: int | symint | None = None) -> Tensor:
    if size != None:
        return Tensor(shape=[self.shape[0], self.shape[1]] + broadcast_int(size, len(self.shape) - 2))
    if scale_factor != None:
        return Tensor(shape=[self.shape[0], self.shape[1]] + [d * scale_factor for d in self.shape[2:]])
    raise Error("interpolate requires either 'size' or 'scale_factor' argument")

def loss_ir(self: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "none":
        return Tensor(shape=self.shape)
    return Tensor(shape=[])

def pad_ir(self: Tensor, pad: list[int]) -> Tensor:
    rank = len(self.shape)
    num_pad_dims = len(pad) // 2
    offsets = [pad[(rank - 1 - i) * 2] + pad[(rank - 1 - i) * 2 + 1] if i >= rank - num_pad_dims else 0 for i in range(rank)]
    return Tensor(shape=[d + offsets[i] for i, d in enumerate(self.shape)])

def rfft_ir(self: Tensor, n: int | symint | None = None, dim: int = -1) -> Tensor:
    d = normalize_dim(len(self.shape), dim)
    if n != None:
        return Tensor(shape=replace_dim(self.shape, d, n // 2 + 1))
    return Tensor(shape=replace_dim(self.shape, d, self.shape[d] // 2 + 1))

def irfft_ir(self: Tensor, n: int | symint | None = None, dim: int = -1) -> Tensor:
    d = normalize_dim(len(self.shape), dim)
    if n != None:
        return Tensor(shape=replace_dim(self.shape, d, n))
    return Tensor(shape=replace_dim(self.shape, d, 2 * (self.shape[d] - 1)))

def size_ir(self: Tensor, dim: int | None = None) -> int | symint:
    if dim != None:
        return self.shape[normalize_dim(len(self.shape), dim)]
    return [d for d in self.shape]

def numel_ir(self: Tensor) -> int | symint:
    return torch_shapes.prod(self.shape)

def dim_ir(self: Tensor) -> int:
    return len(self.shape)

def item_ir(self: Tensor) -> Tensor:
    if len(self.shape) != 0:
        raise Error("item() only works on 0-dimensional tensors, got " + str(len(self.shape)) + "D tensor")
    return Unknown

def tolist_ir(self: Tensor) -> Tensor:
    return Unknown

def multinomial_ir(self: Tensor, num_samples: int | symint) -> Tensor:
    return Tensor(shape=self.shape[:-1] + [num_samples])

def where_ir(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return Tensor(shape=x.shape)

def take_along_dim_ir(self: Tensor, indices: Tensor) -> Tensor:
    return Tensor(shape=indices.shape)
"#;
