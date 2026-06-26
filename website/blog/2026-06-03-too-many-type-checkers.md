---
title: Are you really expected to run five type-checkers now?
description: Library maintainers may feel overwhelmed by the plurality of type checkers that exist. We offer some guidance on how to focus their efforts where they matter most.
slug: too-many-type-checkers
authors: [marcogorelli]
tags: [typechecking, news]
hide_table_of_contents: false
---

Mypy, Pyrefly, Pyright, ty, Zuban, and possibly more that will come in the future... how are library maintainers expected to cope?

**TL;DR**: Prioritise running as many type-checkers as possible on your test suite. Run at least one on your source code.

<!-- truncate -->

## The type checking that matters most (and why you've probably got it backwards)

If you only read one section of this blog post, please make it this one. Because this is where a lot of packages get it wrong. It's common to see packages run type checkers on their source code and to leave their tests untyped. **That approach has it backwards**.

Suppose you maintain a Python package. As a hypothetical user of your code, I don't particularly care about your internal development practices. Whether you use `ruff format` or `black`, how you sort your imports, whether you use `pytest` or `unittest`, none of this affects me. What I do care about is your public API and my experience interacting with it.

When you run a type-checker on your internal source code, you're mostly testing your internal logic. You can do that with whichever type checker you prefer, that's your choice. Which type-checker your users use, on the other hand, isn't.

By running as many type-checkers as possible over your test suite, you ensure that your package's public API works well for as many of your users as possible.

## The Polars story

Polars is a modern dataframe library which, since its launch in 2020, has been taking the data science world by storm. As a heavy user of the library, I was very interested in making its developer experience even better. If Polars' types are accurate, then as a user I get better auto-complete, documentation, and [protection from certain classes of bugs](https://pyrefly.org/blog/surprising-errors/). What would it take to add Pyrefly to Polars' continuous integration jobs?

I started investigating this, and quickly ran into some roadblocks. Pyrefly is generally stricter than mypy, so it required rewriting parts of the codebase or adding more explicit type annotations when instantiating variables. Furthermore, I encountered some bugs in Pyrefly, and encouragingly enough, fixes for the vast majority of them were shipped with the highly anticipated [v1 release](https://pyrefly.org/blog/v1.0/). I think it was worth it, especially as it [uncovered a medium-priority bug](https://github.com/pola-rs/polars/issues/27620), but I did have to ask myself whether going through this for another three type-checkers would be.

To illustrate this point, let's look at the function `DataType.__eq__`. In Python, any method `__eq__` is expected to return `bool`, and if it doesn't, then we need to explicitly tell type-checkers to ignore the type error. This function in Polars can also return different types depending on the inputs, thus requiring [overloads](https://typing.python.org/en/latest/spec/overload.html). To get this function to satisfy all of mypy, Pyrefly, and ty, we need to write:
```py
    @overload  # type: ignore[override]
    def __eq__(  # pyrefly: ignore[bad-override]
        self, other: pl.DataTypeExpr
    ) -> pl.Expr: ...

    @overload
    def __eq__(self, other: PolarsDataType) -> bool: ...

    def __eq__(self, other: pl.DataTypeExpr | PolarsDataType) -> pl.Expr | bool:  # ty: ignore[invalid-method-override]  # pyright: ignore[reportIncompatibleMethodOverride]
```

Wow, that's 4 different type-ignore comments for just 7 lines of code! You can see how a codebase quickly becomes polluted with such comments, or with workarounds to deal with different type-checkers' quirks. I don't think any library maintainer wants a codebase that looks like that. Surely there's a better way?

Instead of putting all your internals through multiple type-checkers, why not start by testing that all major type-checkers can be used with your library's public API? That's much more useful, so it's easier to justify spending time on it. But it's also easier, because you're just ensuring that, if your library gets used as-intended, then there are no type errors. In the case of `DataType.__eq__`, there's a test for it that looks like this:

```py
DTYPE_TEMPORAL_UNITS: Final[frozenset[TimeUnit]] = frozenset(["ns", "us", "ms"])

def test_dtype_time_units() -> None:
    # check (in)equality behaviour of temporal types that take units
    for time_unit in DTYPE_TEMPORAL_UNITS:
        assert pl.Datetime == pl.Datetime(time_unit)
        assert pl.Duration == pl.Duration(time_unit)

        assert pl.Datetime(time_unit) == pl.Datetime
        assert pl.Duration(time_unit) == pl.Duration
```
What's pleasing to see is that mypy, Pyrefly, Pyright, ty, Zuban all type-check this fine without reporting any errors! So even though the type-checkers disagree a bit on how the implementation should be written, they all agree about the effects on the public API. And that's what your users care about!

Getting Pyrefly to run on the whole Polars test suite was relatively painless, you can check out the [PR](https://github.com/pola-rs/polars/pull/27459) to verify this. To ease Polars' own internal development, we've also been exploring using Pyrefly on their source code, though that is a [larger effort](https://github.com/pola-rs/polars/issues/27049) and is being tackled incrementally.

## What about my source code? Why are there so many type checkers anyway?

The [typing spec](https://typing.python.org/en/latest/spec/) outlines a standard set of rules that type checkers are expected to adhere to. There are aspects of it that are a bit hazy, however, such as in cases where users under-specify typing information. In those cases, different type checkers make different design decisions:

- Some choose to be as strict as possible, emitting false-positives if necessary, but doing as much as possible to guard you from potential bugs.
- Others are more lenient and allow you to add type information to your codebase more gradually.

When it comes to type-checking your source code, it's good to ask yourself where on the strict vs lenient spectrum you want to be. Pyrefly is not only strict (though [this can be configured](https://pyrefly.org/en/docs/configuration/#configuration-options)), but also [fast](https://pyrefly.org/blog/speed-and-memory-comparison/) and [conformant](https://pyrefly.org/blog/typing-conformance-comparison/), making it an excellent choice. If you try it out on your projects and run into any issues, [please report them](https://github.com/facebook/pyrefly/issues) so that both you and all its other users can benefit from fixes!

## The bottom line

There are 5 Python type-checkers which get attention these days: mypy, Pyrefly, Pyright, ty, Zuban. Library maintainers may rightfully feel like running all 5 of them over their source code is too much maintenance effort and requires polluting their code with too many type-ignore comments. We have made the case that such effort would be better spent by running multiple type-checkers over their tests instead, as that will test how well the library can be type-checked when users interact with it.
