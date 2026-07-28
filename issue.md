## Problem
PyTorch relies heavily on Triton, OpenAI's GPU kernel DSL, which uses `@triton.jit` decorators on Python functions. Currently, Pyrefly treats the body of Triton kernels as regular Python. This leads to `Any` fallbacks or false positives because standard Python typing does not comprehend Triton's specific semantics, such as `tl.tensor`, block pointers, `tl.constexpr` parameters, and broadcasting rules.

## Proposed Solution
Introduce a Triton DSL Semantic Analyzer plugin. The plugin will intercept functions decorated with `@triton.jit` and apply Triton-specific type inference rules. This will make Pyrefly the first major type checker with native GPU kernel strict typing.

### High-level Design
- **AST Interception**: Add a hook in the AST visitor that identifies the `@triton.jit` decorator.
- **Triton Type Abstractions**: Introduce new hygienic types like `tl.tensor` and `AnyShape` into Pyrefly's type environment. These types will not leak into non-Triton contexts.
- **Inference Engine**: A custom AST visitor will evaluate `triton.language` calls based on Triton's strict semantics.
- **Fallback Strategy**: 
  - Unknown Triton APIs will return `Any` without emitting errors.
  - Missing shape information will widen to `AnyShape`.
  - Any failure in inference will default to `Any` and emit a note rather than an error.

### Scoped Tasks
- [ ] Create `TritonType` variants (`tl.tensor`, `AnyShape`).
- [ ] Implement detection of `@triton.jit` decorated functions and handle the `strict=True` opt-in mechanism.
- [ ] Implement type rules for `tl.load` and `tl.store`.
- [ ] Implement binary operation type promotion and rank consistency for `tl.arange`/`tl.broadcast`.
- [ ] Handle `tl.constexpr` parameters (treat as scalar values for basic constant folding).
- [ ] Add 5 positive and 5 negative test cases.
- [ ] Write documentation in `docs/triton_plugin.md`.

### Implementation Details
The implementation will be in Rust, integrated as a module that uses minimal hooks in the core Pyrefly AST analysis engine. The initial scope will be strictly limited to dtype checking for basic pointer operations, leaving complex shape inference for future updates.
