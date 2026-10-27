# SciPy syntax demo

These three independent module slices use plain NumPy annotations plus shape
comments. They do not import SciPy or its stub package. Their basenames are used
as local fixture modules (`fft_helper`, `cholesky`, and `distance`), so they cannot
accidentally masquerade as a complete `scipy-stubs` distribution.

## Provenance and adaptations

Reference: [scipy/scipy-stubs at
985553ed19eec8c1a03a6ddf1c59c3e6261f63f2](https://github.com/scipy/scipy-stubs/tree/985553ed19eec8c1a03a6ddf1c59c3e6261f63f2).

- `fft_helper.pyi` corresponds to
  [`scipy-stubs/fft/_helper.pyi`](https://github.com/scipy/scipy-stubs/blob/985553ed19eec8c1a03a6ddf1c59c3e6261f63f2/scipy-stubs/fft/_helper.pyi).
- `cholesky.pyi` corresponds to
  [`scipy-stubs/linalg/_decomp_cholesky.pyi`](https://github.com/scipy/scipy-stubs/blob/985553ed19eec8c1a03a6ddf1c59c3e6261f63f2/scipy-stubs/linalg/_decomp_cholesky.pyi).
- `distance.pyi` corresponds to
  [`scipy-stubs/spatial/distance.pyi`](https://github.com/scipy/scipy-stubs/blob/985553ed19eec8c1a03a6ddf1c59c3e6261f63f2/scipy-stubs/spatial/distance.pyi).

These are adapted API excerpts, **not unchanged upstream files**. The upstream
protocol and dtype-conversion overloads are reduced to NumPy-only cases, and
`optype` / `numpy-typing-compat` aliases are spelled with NumPy types instead. The
FFT excerpt also drops its `scipy._typing` dependency. Original function names,
parameter order, and defaults are retained within the selected cases.

The Cholesky return convention also follows
[SciPy at f5d04b01ed54ee27861e2aa2ed9c0e5484a79551](https://github.com/scipy/scipy/blob/f5d04b01ed54ee27861e2aa2ed9c0e5484a79551/scipy/linalg/_decomp_cholesky.py):
`cho_factor` returns an array-valued lower flag with the batch shape, including
an empty shape for an unbatched factor. This is not the Python-bool return
convention of some older SciPy releases.

The upstream BSD notice is retained in [LICENSE.scipy-stubs](LICENSE.scipy-stubs).

## Scope

- **FFT:** Python-integer lengths, real spacing, and the NumPy/CPU backend for
  `fftfreq` / `rfftfreq`. Shifts accept NumPy arrays and preserve their dtype and
  full shape. Backend dispatch and dtype-conversion overloads are omitted.
- **Cholesky:** float64 NumPy arrays. `cholesky` and `cho_factor` include batch
  dimensions. `cho_solve` shows unbatched vector and matrix right-hand sides;
  batched solve/broadcasting and other precision overloads are omitted.
- **Distances:** float64 NumPy arrays and Euclidean distances. Observation counts
  can differ, but feature dimensions must agree. `pdist` demonstrates the
  condensed length `M * (M - 1) // 2`. Other metrics and array-like inputs are
  omitted.

Shape contracts describe results of valid calls. They do not establish numerical
properties such as positive definiteness, finite values, or valid FFT axes.

## What to assess

Compare the small `int  # shaped: N` case with array `[0]` selectors,
comments directly beside tuple shape carriers, and the whole-type replacement
needed when an `NDArray` alias hides its shape argument. Try changing ordinary
annotations or adding API cases: the goal is to expose syntax friction, not to
maximize SciPy coverage.

## Syntax observations

- The scalar `n: int  # shaped: N` form is compact.
- Overload declarations place `# shapes:` between `@overload` and `def`, so Ruff
  preserves adjacent overloads without inserting a blank line between them.
- `[0]` preserves dtype annotations. Splitting the type lets a shape comment
  sit directly beside `tuple[int, ...]` instead of using a long selector.
- An `NDArray` alias hides the shape slot, requiring a full `= payload` replacement.
- `cho_factor` refines both the factor array and its array-valued lower flag.
  Per-component comments express both shapes without repeating the return type.
- Optional `out` arrays keep their shape comment on the array's carrier line,
  with `None` visibly outside the transformation.
- Ruff preserves comments on complete type components across formatting. Long
  alias-replacement payloads can still force additional line wrapping.

The [checked callers](examples) show inferred sizes and rejection of mismatched
shapes. Run this project from the corpus runner with `--project scipy`.
