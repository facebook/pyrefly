---
title: What's type-coverage, and why should I care?
description: Learn about an under-appreciated Pyrefly feature
slug: why-type-coverage
authors: [marcogorelli]
tags: [typechecking, news]
hide_table_of_contents: false
---

Pyrefly is one of the major new Python type-checkers taking the world by storm. It's often touted for its speed, usability, and typing-spec-conformance. But it also offers an extra nifty little feature you may never have heard of: type-coverage.

Let's learn about what it is, why you might want to care about it, and what it can do for you.

<!-- truncate -->

**TL;DR**:

- Type coverage measures what percentage of your project's typables are type-annotated.
- Optionally, you can choose to only include publicly-exported typables.
- You can use it similarly to how you would use test coverage.

## What's type coverage?

Similarly to how test coverage measures how much of a library's source code is hit by its tests, type coverage measures what percentage of a library's typables have type annotations. By typable, we mean something which can be meaningfully annotated with a type, such as function arguments and return types, class variables, and constants. Local variables in function bodies, on the other hand, are excluded, as users would never interact with them anyway.

If this sounds familiar to you, it may be because we've previously written about the topic in [typing pandas](https://pyrefly.org/blog/pandas-type-completeness/) and [typing numpy](https://pyrefly.org/blog/numpy-type-completeness/). In those posts, we were using Pyright's _verify-types_ feature. Now, however, Pyrefly ships its own type coverage tool (_pyrefly coverage_), which we're now recommending you use instead.

## Concrete example

Say we have a Python file which defines the following functions:

- `def one(a: int) -> int`
- `def two(b: int) -> None`
- `def three() -> None`

and a stub file (one which ends with `.pyi`) which contains:

- `def one(a: int) -> int`
- `def two(c) -> None`

Note how there are two issues with the stub file:

- The `c` parameter in `two` is misnamed (it should be `b`). It's also unannotated.
- The `three` function is missing entirely.

Note that if a stub file is present, then that's all the type-checker looks at. Type [typing spec](https://typing.python.org/en/latest/spec/distributing.html#import-resolution-ordering) enforces this: "If a stub file is found for a module, the type checker should not read the corresponding “real” module". So, if stub files are present, they could be both accurate and complete.

Here, however, the stub is incomplete. How could we have been alerted about this? Just running a type-checker isn't enough, `pyrefly check` would return zero errors in this code alone. Just running `ruff check` with `ANN` enabled isn't enough either, as that would only tell us that the `c` argument is unannotated (it wouldn't tell us anything about the `three` function missing from the stub). Fortunately, there is indeed a tool which can tell us that `three` is missing from the stubs: _pyrefly coverage_. In this case, running `pyrefly coverage check` tells us:

```console
$ pyrefly coverage check
 WARN `foo.three` is untyped [coverage-missing]
 --> src/foo/__init__.py:7:1
  |
7 | / def three() -> None:
8 | |     return None
  | |_______________-
  |
 WARN `foo.two` is not fully typed [coverage-partial]
 --> src/foo/__init__.pyi:2:1
  |
2 | def two(c) -> None: ...
  | -----------------------
  |
ERROR type coverage 60.00% (3 of 5 typable) is below the 100.00% threshold
```

Great! Now we can add `three` to the stubs and our users can safely use it with type-checking.

## Can't I just use Ruff's ANN rules?

The Python linter [ruff](https://docs.astral.sh/ruff) has a suite of typing-related rules grouped together under the [ANN](https://docs.astral.sh/ruff/rules/#flake8-annotations-ann) prefix. It's very useful for catching missing type annotations, and because Ruff analyses files statically, the implementation is blazingly fast. So what does Pyrefly's type-coverage offer on top of that? We present three benefits.

First, class variables. If you have a class which does any non-trivial logic in its `__init__` method, then those class variables might not get inferred uniformly across type-checkers (if at all!). `pyrefly coverage` enforces that you explicitly specify types for your class variables. [This PR which typed the `GroupBy` attributes](https://github.com/pola-rs/polars/pull/27903/changes) is a good example of this, where `offset`, `period`, and `closed` all go through non-trivial transformations in the `__init__` method.

Second, you might not currently have type annotations everywhere. By giving you a report, a coverage score, and the option to only include public typables (`--public-only`), `pyrefly coverage` allows you to focus your efforts on the highest-yield parts of your codebase and tell you how far along with your effort you are. We used this to [prioritise typing efforts in NumPy](https://pyrefly.org/blog/numpy-type-completeness/).

Finally, stub files. `pyrefly coverage` checks that you don't forget to include annotations for anything in your Python files, if you have stub files. [stubtest](https://mypy.readthedocs.io/en/stable/stubtest.html) can also do this, but that also checks for runtime behaviour and so is a more intensive check, whereas `pyrefly coverage` is fast enough that you could easily include it in a pre-commit configuration file without it impacting developer productivity.

## Who's using it?

Pyrefly coverage is a new-ish feature, yet it's already being used by a few major projects:

- [scipy-stubs](https://github.com/scipy/scipy-stubs).
- [NumPy](https://github.com/numpy/numpy).
- [Polars](https://github.com/pola-rs/polars).
- [Narwhals](https://github.com/narwhals-dev/narwhals).
- [sh](https://github.com/amoffat/sh).

If you would like to use it in your project and would like any assistance, [feel free to reach out](https://discord.gg/Cf7mFQtW7W), and if you encounter any issues, [please do report them](https://github.com/facebook/pyrefly/issues)!
