---
title: Bringing NumPy's type-completeness score to nearly 90%
description: We tell the story of how we brought NumPy's type-completeness score from ~33% to nearly 90%
slug: numpy-type-completeness
authors: [marcogorelli]
tags: [typechecking, news]
hide_table_of_contents: false
---

Because [NumPy](https://numpy.org/) is one of the most downloaded packages in the Python ecosystem, any incremental improvement can have a large impact on the data science ecosystem. In particular, improvements related to static typing can improve developer experience and help downstream libraries write safer code. We'll tell the story about how we helped bring its type-completeness score to nearly 90% from an initial 33%.

**TL;DR**:

- NumPy's type-completeness score was ~33%.
- A one-line fix doubled coverage to over 80%.
- Fully typing MaskedArray pushed the score to nearly 90%.
- What's left? Top-level numpy.ma functions, more precise overloads, and adding a type-checker to NumPy's CI.

## Wait, what's type completeness?

Modern IDEs use type annotations to help developers by showing them helpful suggestions and highlighting syntax. [Pyright](https://github.com/microsoft/pyright) is a popular type-checker which, as well as checking for correctness and consistency, can also measure what percentage of a library's public API has type annotations. We call the percentage of fully-typed symbols exported by a library the _type-completeness score_.

For example, a module which exports functions `foo` and `bar`:

```python
def foo(a: int):
    return None

def bar() -> int:
    return 1
```

would have a 50% type-completeness score, because:

- `foo` is partially unknown, as it's missing a return annotation.
- `bar` is type-complete.

By changing the `foo` signature to be `def foo(a: int) -> None:`, the type-completeness score would jump to 100%. The more type-complete a library is, the more helpful the suggestions an IDE can show to the user.

Note that type-completeness only measures how much of the public API is covered by types. If you want to verify that those types are correct and self-consistent, you'll also need to run a type checker. The most used type checkers currently are [mypy](https://github.com/python/mypy) and Pyright, but [Pyrefly](https://github.com/facebook/pyrefly) and [ty](https://github.com/astral-sh/ty) are also attracting a lot of attention due to their impressive performance characteristics (note however that neither yet describes itself as production-ready, so temper your expectations accordingly if you try them out!).

## How type-complete is NumPy?

When we started this effort (in March 2025), we first tried measuring NumPy's type-completeness by running:

```console
pyright --verifytypes numpy
```

The output showed a completeness score of...18%. Wait, only 18%? This seemed very low, because by then typing efforts in NumPy had been ongoing for some time. Was something up with the metric? Upon closer inspection of the output, we noticed a few things:

- Some objects, such as `DTypeLike`, were reported to be "partially unknown", even though they had type annotations.
- The type-completeness report included test modules such as `numpy.tests.test_matlib`.

The first issue was caused by an import from the standard library `decimal` module, which itself is partially untyped. Given that this is outside of NumPy's direct control, we decided to exclude it from the coverage report by using `--ignoreexternal`, [as suggested by Eric Traut](https://github.com/microsoft/pyright/discussions/9911).

For the second issue, PyRight gives us an option to export the coverage report to json (`--outputjson`). We could then parse the json and exclude `numpy.tests`. Given that NumPy users wouldn't ordinarily interact with NumPy's internal test suite but that PyRight considers it public, we decided that it made sense for us to exclude tests in order to focus our efforts on what would make the biggest user-facing impact.

Once we'd addressed the two steps above, the baseline type-completeness score became 33%. This was our starting point. We could then focus our efforts on the remaining 67%!

## The one-line change which doubled type-completeness

Pyright's report includes classes, methods, functions, type aliases, and more. A lot of scientific Python code centres around some central classes such as `ndarray`. `ndarray` was reported as "partially unknown", but eye-balling the exported symbols related to that class revealed something interesting:

```python
In [13]: np.mean([x['isTypeKnown'] for x in exported if x['name'].startswith('numpy.ndarray.')])
Out[13]: np.float64(0.9811320754716981)
```

So, `ndarray` was reported as "partially unknown", but 98% of its methods had known types. It shouldn't be much effort to bring that number to 100%! In fact, all it took was a [one-line change to fix a typo in a type annotation](https://github.com/numpy/numpy/pull/28908): 

```diff
- def setfield(self, /, val: ArrayLike, dtype: DTypeLike, offset: CanIndex = 0) -> None: ...
+ def setfield(self, /, val: ArrayLike, dtype: DTypeLike, offset: SupportsIndex = 0) -> None: ...
```

That's it! `CanIndex` was an unknown symbol and was probably mistyped, and replacing it with the correct `SupportsIndex` one brought NumPy's overall type-completeness to over 80%! We then started examining other NumPy classes to see if there was anywhere else where we could make an impact, and hopefully a much larger one.

## Enter MaskedArray

When we looked at the percentage of typed symbols from the MaskedArray class, we noticed something interesting:

```python
In [14]: np.mean([x['isTypeKnown'] for x in exported if x['name'].startswith('numpy.ma.core.MaskedArray.')])
Out[14]: np.float64(0.2)
```

Only 20% of them were typed! That's quite a contrast with `ndarray`, which was already at 98% when we started. It's also a fairly widely used class, appearing in the codebases of pandas, scikit-learn, and xarray. Given how poorly typed it was, we decided it would be a good candidate to spend time on!

## Typing MaskedArray

The main difficulty in typing NumPy code isn't inferring the possible argument values (which is quite easy to do with automated tools), but rather dealing with the large number of overloads. This is because many of NumPy methods' return types depend on the exact combinations of the input types. For example, suppose we have a `MaskedArray` `ma` and want to count the number of non-null values:

- `ma.count()` returns an integer.
- `ma.count(axis=0)` returns an array.
- `ma.count(keepdims=True)` also returns an array, of the same shape as the input array.

We can type this by having a different overload of each of these different cases. This is a relatively simple example, but there are others where the number of necessary overloads was as high as 9! This isn't something which is easy to automate, and requires careful reading of the documentation and of the source code. It's a non-trivial amount of work.

But...we pulled through it, and thanks to some very timely and constructive reviews from the amazing [Joren Hammudoglu](https://github.com/jorenham), MaskedArray is now reported as 100% type-complete! As for the overall type-completeness, that's now at 88%. So...what's left?

## What's missing from NumPy?

In the MaskedArray module, there's still some untyped top-level functions, such as `numpy.ma.count`. Overloads could be made more precise and shape-preserving (e.g. if the input is 2D, then make sure to preserve this fact in the output where possible). There's some missing defaults in the stubs. There's no shortage of work here.

The biggest missing bit, however, is the elephant in the room: NumPy doesn't run a type-checker over its codebase in CI. It has some typing tests, sure, but that's different from running a type-checker. If any motivated reader is interested in making a significant open source contribution, then getting NumPy's typing into a state such that a type checker (and possibly even [stubtest](https://mypy.readthedocs.io/en/stable/stubtest.html)) can be run over it could be a great use of your time.

## Conclusion, acknowledgements

We've looked at how we contributed towards increasing NumPy's type-completeness. Given how widespread NumPy's adoption is, we expect this effort to have been an impactful one. We look forward to seeing what else we can achieve in this space - thank you to [Meta and Quansight Labs](https://discuss.python.org/t/call-for-suggestions-nominate-python-packages-for-typing-improvements/80186/1) for having funded and facilitated this effort!

