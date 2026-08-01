---
title: "Define less, check more: Pyrefly now speaks attrs"
description: attrs is the library that taught Python to write classes without boilerplate. Pyrefly now supports attrs out of the box, recognizing both its classic and modern APIs with no plugins or configuration.
slug: pyrefly-attrs
authors: [shobhitm]
tags: [attrs, typechecking, IDE]
hide_table_of_contents: false
---

Built-in support for [*attrs*](https://www.attrs.org/) is available in Pyrefly as
of version 1.2.0-dev.1, and ships in the upcoming 1.2.0 stable release. You get
accurate constructor signatures and better type safety for your *attrs* classes,
with no plugins or configuration to set up.

<!-- truncate -->

*attrs* consists of two separate APIs with quite different semantics: a classic
one under the `attr` namespace and a modern one under `attrs`. While
[PEP 681](https://peps.python.org/pep-0681/)'s `dataclass_transform` provides a
useful baseline for type checkers to understand parts of *attrs*, many features
cannot be expressed in terms of standard type annotations.

Historically, *attrs* users that want type checking have either had to use Mypy
(which implements dedicated *attrs* support via a plugin), limit themselves to a
subset of the API compatible with `dataclass_transform`, or just live with
limited type checking support.

At the time of writing, Pyrefly and Mypy are the only type checkers that provide
full support for *attrs*.

## **What *attrs* generates for you**

*attrs* lets you define a class by declaring its attributes, then generates the
boilerplate for you: `__init__`, `__repr__`, `__eq__`, ordering, hashing,
slots, and immutability. If you've used Python's built-in `dataclasses`, this
will feel familiar. *attrs* came first, and `dataclasses` was directly inspired by
it. *attrs* remains the more flexible of the two.

For example:

```python
from attrs import define, field

@define
class Tune:
    title: str
    key: str
    choruses: int = field(default=1)

bluebossa = Tune(title="Blue Bossa", key="Cm")
print(bluebossa)
```

With that single decorator, *attrs* writes a typed `__init__`, a readable
`__repr__`, and value-based equality for you. If you pass the wrong thing, say
`Tune(title="Blue Bossa", key=5)`, that's a bug you'd like to hear about before
you run the code.

## **Pyrefly & *attrs*: How it works**

So how does Pyrefly work with *attrs*? Here's what the support covers:

- **Understands the core decorators and field specifiers**: Pyrefly recognizes
  the modern API (`@define`, `@frozen`, `@mutable`, `attrs.field`) and the classic
  API (`@attr.s`, `@attr.ib`, `@attr.dataclass`), across both the `attr.` and
  `attrs.` namespaces. No import errors or stray red squiggles because your tools
  don't understand what *attrs* is.
- **No plugin required**: Pyrefly recognizes *attrs* classes out of the box. Unlike
  Mypy, it needs no *attrs* plugin, and there's nothing to configure.
- **Static analysis that reflects runtime logic**: much of *attrs*' behavior is
  decided when your class is created, including which assignments become
  fields, what the constructor looks like, and whether the class is ordered or
  frozen. Pyrefly's analysis mirrors those runtime rules as closely as possible,
  so what you see in your IDE matches what happens when the code runs.
- **Catches *attrs* misconfigurations**: many ways of misusing *attrs* don't fail
  until the class is built at runtime. Pyrefly understands *attrs*' own rules for
  how a valid class is put together, so it surfaces those mistakes statically, as
  you type, instead of leaving them to blow up later.

For the full list of supported features, check out the
[documentation](https://pyrefly.org/en/docs/attrs/).

### Automatic recognition: classic, modern, or both

This is where *attrs* differs from most libraries a type checker has to support.
*attrs* has spent ten years accumulating **two complete APIs**:

- The **classic** style, from *attrs* 15.0.0 (2015), *attrs*'s first release:
  `@attr.s` on the class, `attr.ib()` on each attribute.
- The **modern** style, added in *attrs* 20.1.0 (2020): `@attrs.define` and
  `attrs.field()`.

Both are fully supported by *attrs*, and since the classic style was never
deprecated, the same codebase can end up containing both, even though mixing
them is generally discouraged. The catch is that they don't behave identically.
The clearest example is how each one decides which lines in your class body are
actually fields. The modern
decorators read your annotations. The classic `@attr.s` ignores bare annotations
by default and only collects `attr.ib()` assignments, unless you opt in with
`auto_attribs=True`:

```python
import attr

@attr.s
class Horn:
    name: str            # NOT a field, just an annotation, under classic @attr.s
    serial = attr.ib()   # this is the only real field

Horn(serial="A-440")           # Pyrefly accepts this
Horn("Bach Strad", "A-440")    # Pyrefly reports an error, just like attrs would
```

Pyrefly reads how you wrote the class and adapts, the same way it reads a
Pydantic model's `strict` or `extra` settings. Because it resolves this per
class, a base in one style and a subclass in the other compose exactly as they
do at runtime.

The twist hiding in *attrs* is that the constructor doesn't always look like the
class. Add a `converter` to a field and the `__init__` parameter takes the
converter's input type, while the attribute keeps its output type, and
Pyrefly tracks both:

```python
from attrs import define, field

def to_bpm(s: str) -> int:
    return int(s.removesuffix(" bpm"))

@define
class Chart:
    tempo: int = field(converter=to_bpm)

medium_swing = Chart("120 bpm")   # takes a str, that's what to_bpm reads
reveal_type(medium_swing.tempo)   # int, what to_bpm hands back
```

`Chart("120 bpm")` type-checks because the converter accepts a `str`, while
`medium_swing.tempo` is an `int` everywhere you read it. That's the kind of
detail you'd otherwise only discover at runtime.

## **Getting Started**

There are no special configurations or plugins required to start using Pyrefly
with *attrs*:

1. [Install *attrs*](https://www.attrs.org/en/stable/overview.html).
2. [Install Pyrefly](https://pyrefly.org/en/docs/installation/) version 1.2.0-dev.1
   or later (*attrs* support ships in the 1.2.0 stable release).
3. Write your *attrs* classes as usual: classic, modern, or a mix.
4. Run Pyrefly, or use it in your IDE.

Pyrefly does not require *attrs*-specific configuration and recognizes both the
classic and modern APIs when available.

If you'd like to play with some examples, we put together
[a demo repo](https://github.com/shobhitmehro/pyrefly-attrs-demo) you can clone
and try out.

## **Conclusion**

Try *attrs* support in your own projects and let us know what works and what
doesn't. You can [open an issue on GitHub](https://github.com/facebook/pyrefly/issues)
or [connect with us on Discord](https://discord.gg/Cf7mFQtW7W).

*attrs* joins Pydantic and Django in the set of libraries Pyrefly supports without
a plugin. If there's a particular package you rely on, let us know.
