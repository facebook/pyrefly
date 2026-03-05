---
title: pandas' Public API Is Now Type-Complete!
description: We tell the story of how we helped make pandas' public API type-complete, and how to prevent it from regressing
slug: pandas-type-completeness
authors: [marcogorelli]
tags: [typechecking, news]
hide_table_of_contents: false
---

At time of writing, pandas is one of the most widely used Python libraries. It is [downloaded about half-a-billion times per month from PyPI](https://pypistats.org/packages/pandas), is supported by nearly all Python data science packages, and is generally required learning in data science curriculums. Despite modern alternatives existing, pandas' impact cannot be minimised or understated.

In order to improve the developer experience for pandas' users across the ecosystem, we at Quansight Labs (with support from the Pyrefly team at Meta) decided to focus on improving pandas' typing. Why? Because better type hints mean:

- More accurate and useful auto-completions from VSCode / PyCharm / NeoVIM / Positron / other IDEs.
- More robust pipelines, as some categories of bugs can be caught without even needing to execute your code.

By supporting the pandas community, pandas' public API is now type-complete (as measured by Pyright), up from 47% when we started the effort last year. We'll tell the story of how it happened - but first, we need to talk more about type completeness, and how we measure it.

<!-- truncate -->

## But first - how is type-completeness measured?

Pyright has a nifty little feature which helps us calculate the type-completeness of a library's public API. The general idea is:

- Find all public symbols exported by a package. For example, in pandas there's `pandas.DataFrame`, `pandas.read_csv`, `pandas.Series`, ...
- For each symbol, check where all its types are known. This includes function arguments, function return types, attributes, and base classes.
- If any type is unknown, then the whole symbol counts as unknown. For example, if a package exports a class `Foo` which has some missing type annotations, and a function `def bar() -> Foo: ...`, then the function `bar` also counts as type-unknown because it returns a type-unknown symbol (`Foo`).

Type-completeness is different from just calculating the percentage of missing type annotations, as it's biased towards heavily-used classes. In pandas, for example, `Series` appears as an argument and return type to at least some function in all of pandas' methods - therefore, no matter how type-complete the rest of pandas is, if `Series` isn't type-complete, then pandas' overall type-completess score will remain low.

By default, Pyright includes all public symbols. In practice, there are some pandas paths which are considered public according to Python's usual standards, but which pandas considers private, such as:

- `pandas.tests`.
- `pandas.conftest`.
- parts of `pandas.core` which aren't publictly re-exported in other places such as `pandas.api`.

We therefore amend Pyright's calculation to exclude these "technically public but not really" paths. This gave us a more useful measure of what part of the pandas API which users are expected to interact with is actually type-annotated.

## Moving the needle in pandas

Investigating sources of missing type-completeness in pandas was quite a circular exercise. For example, suppose that `DataFrame` and `Series` were type-complete, but `Index` had an untyped attribute. Here is what would would happen:

- `Index` would be reported as "partially unknown" because of its untyped attribute.
- `DataFrame` would be reported as "partially unknown" because its method `.index` returns `Index`, which is partially unknown.
- `Series` is reported as "partially unknown" because its method `to_frame` returns `DataFrame`, which is partially unknown.

It was clear, therefore, that incremental progress would be difficult. Because of how intertwined pandas' classes all are, we expected the type-completeness score to flatline for several months before suddenly spiking. And that's exactly what happened! Progress flat-lined at around 60-70%, before spiking up to 100%.

## How ruff helped

pandas-stubs uses the [ruff linter](https://docs.astral.sh/ruff/) to enforce code quality standards. Ruff is highly configurable and comes with many optional ones, one of which is [any-type (ANN401)](https://docs.astral.sh/ruff/rules/any-type/). A prerequisite to type-completeness is that types be present everywhere. In order to track progress, the `ANN401` rule was enabled across the codebase, with a few exclusions which were then addressed gradually. The general rule was: if you make a pull request which types a certain part of the codebase, then remove the `ANN401` exclusion for that part of the codebase so that it stays fully-typed in the future.

## Ensuring that type-completeness stays this high

Measuring type-completeness with Pyright is:

- Very easy if a package ships its own type annotations.
- Fairly tricky if a package relies on an external package for type annotations.

The situation for pandas is the latter, meaning that some extra work is needed to ensure type-completeness stays high in CI. In fact, it's even more complicated because there are parts of pandas which are technically public (according to Python's usual rules) but which pandas considers private! So, quite some work to get around Pyright's default score is needed.

The general idea is:

- Make a temporary virtual environment.
- Install pandas in it.
- Inline the pandas stubs into the same location, and add a `py.typed` file.
- Run `pyright` against the pandas package in that temporary virtual environment.
- Postprocess the output to remove technically-public-but-actually-private symbols.

The full script can be viewed [in the pandas-stubs repo](https://github.com/pandas-dev/pandas-stubs/blob/7ed11d172cd31c61b1adef265e196a1bb8b352da/scripts/type_completeness.py).

## Beyond Pyright - what about "Pyrefly report"?

Pyright's `--verifytypes` feature takes about 2 and a half minutes to run in pandas-stubs. There's room of improvement here - so much so, that the Pyrefly team is working on a [`pyrefly report`](https://pyrefly.org/en/docs/report/) which would work similarly. The `pyrefly report` API is not yet considered stable, so for now pandas-stubs uses Pyright's `--verifytypes` command, but hopefully a faster tool is on the horizon!

## Conclusion and next steps

I'm proud of how Quansight Labs (my employer), Meta, and open source communities were able to come together to make this happen. We plan to continue this work in other targeted open source projects, to keep supporting pandas and NumPy with the work we've already done, to improve typing support in IDEs such as [Marimo](https://marimo.io/), and to make `pyrefly report` production-ready.

Have any requests? Let us know [on Discourse](https://discuss.python.org/t/call-for-suggestions-nominate-python-packages-for-typing-improvements/80186/)
