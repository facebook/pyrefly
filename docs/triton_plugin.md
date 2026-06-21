# Triton DSL Semantic Analyzer Plugin

Pyrefly provides native strict type checking for OpenAI's Triton GPU kernel DSL. By decorating functions with `@triton.jit(strict=True)`, Pyrefly will apply custom inference rules.

## Features
- **Dtype checking**: Enforces consistency in `tl.load` and `tl.store`.
- **Type promotion**: Promotes dtypes correctly for binary operations.
- **Rank consistency**: Verifies dimensions in `tl.arange` and `tl.broadcast`.
- **Constant folding**: Evaluates `tl.constexpr` arguments for basic dead-code elimination.

## Fallback Policy
Pyrefly adopts a permissive fallback strategy for Triton APIs to prevent false positives:
1. **Unknown API Calls**: Returns `Any` without emitting errors.
2. **Missing Shape Information**: Widens the shape to `AnyShape`.
3. **Inference Failures**: Any total failure during type deduction defaults to `Any` and emits a 'note' instead of an error.

## Limitations
- Complex shape inference beyond basic broadcasting is not yet implemented.
- Only functions strictly decorated with `@triton.jit(strict=True)` are analyzed by the Triton type engine.
