---
title: "Python Type Checker Comparison: Empty Container Inference"
description: Learn how different type checkers handle empty containers in Python, including Pyrefly, Ty, Pyright and Mypy.
slug: container-inference-comparison
authors: [yangdanny, jiachen]
tags: [typechecking]
image: https://pyrefly.org/assets/images/empty-container-blog-cef98862cb6b53726eb97df7ee6f2d14.png
hide_table_of_contents: false
---

Empty containers like `[]` and `{}` are everywhere in Python. It's super common to see functions start by creating an empty container, filling it up, and then returning the result.

Take this, for example:

```py
def my_func(ys: dict[str, int]):
  x = {}
  for k, v in ys.items():
    if some_condition(k):
      x.setdefault("group0", []).append((k, v))
    else:
      x.setdefault("group1", []).append((k, v))
  return x
```

This seemingly innocent coding pattern poses an interesting challenge for Python type checkers. Normally, when a type checker sees `x = y` without a type hint, it can just look at `y` to figure out `x`'s type. The problem is, when `y` is an empty container (like `x = {}` above), the checker knows it's a list or a dict, but has no clue what's going inside.

The big question is: How is the type checker supposed to analyze the rest of the function without knowing `x`'s type?

<!-- truncate -->

Different type checkers implement distinct strategies to answer this question. This post will examine these different approaches, weighing their pros and cons, and which type checkers implement each approach. The information presented should be useful as you evaluate and select a type checker.

## Strategy 1: Infer `Any` type for container elements

The simplest approach is just to use `Any` type for the items in the container. E.g. if the developer writes  `x = []` then the type checker would infer the type of `x` to be `list[Any]`. This is what Pyre, Ty, and Pyright[^1] mostly behave like at the time of writing.

Since the analysis does not require looking at any surrounding context at all, this approach is probably the easiest to understand and at the same time the most efficient for a type checker to implement.

Among the inference strategies we discuss today, inferring `list[Any]` produces the least amount of type errors: developers can insert anything into the list, and items read from the list will also be `Any`.

**On the other hand,** **by inferring `Any` we are effectively giving up type safety**.  The type checker won't have any false-positive errors, but it also won't catch any bugs. In our experience using Pyre in Instagram, this lets expensive runtime crashes slip into production.

To illustrate the pitfalls, let’s look at code snippet simplified from a real-world example:

```py
from dataclasses import dataclass

@dataclass
class MenuItem:
   title: str | None
   details: list[str]

def first_three_lines(menu_item: MenuItem) -> list[str]:
   lines = []  # `lines` is inferred as `list[Any]`
   if menu_item.title is not None:
       lines.append(menu_item.title)
   # bug: `list.append()` should be `list.extend()`
   lines.append(menu_item.details)
   return lines[:2]
```

This example has a bug where we accidentally call `append` instead of `extend`, creating a nested list instead of extending the list of strings. Using this strategy, the type checker infers the type of `lines` as `list[Any]` and does not warn the developer.

To improve type safety in these situations, type checkers that infer Any for empty containers can choose to generate extra type errors that warn the user about the insertion of an Any type. While this can reduce false negatives, it burdens developers by forcing them to explicitly annotate every empty container in order to silence the warnings.

## Strategy 2: Infer the container type from all usages

To infer a more precise type for an empty container, a type checker can look ahead and see how it is used.

```py
def my_func(some_condition):
  x = []
  if some_condition:
    x.append(1)
  return x # `x` is inferred as `list[int]`
```

In this example, we see that after the initial definition of `x`, there’s a subsequent usage of `x` that attempts to append `1` to it. Based on that usage, it’s highly likely that the intended type for `x` would be `list[int]`, and the type checker could infer that as the type for `x`.

But what happens if `x` has multiple uses later and each of them assumes different types? For example:

```py
def my_func2(some_condition):
  x = []
  if some_condition:
    x.append(1)
  else:
    x.append("foo")
  return x # `x` is inferred as `list[int | str]`
```

One solution is to look at *all* usages of `x`, assume all of them are valid, and infer the element type of the container to be a union of them all. In our example above, the type checker would infer `list[int | str]`. This is the approach taken by the Pytype type checker.

This approach is very permissive for code that tries to insert into the containers: no type errors will be reported at insertion time at all, similar to the infer-`Any` strategies mentioned before.  However, it provides better type safety over the infer-`Any` strategy when elements are read out of the container: any operations applied to the element must be valid for all inferred types in the union.

For instance:

```py
def my_func3(some_condition):
  x = []
  if some_condition:
    x.append(1)
  else:
    x.append("foo")
  return x[0] + 1  # ERROR! `+` is not applicable to `int | str` and `int`
```

If we call `my_func3(False)`, we will get a runtime crash that happens exactly where the type error is reported, so this inference strategy is also the one that models the Python runtime most closely.

However, **the location of a runtime crash is not necessarily the same as the location of the bug that caused the crash**.

While it’s helpful that the type checker gives an error when we try to use a value from a list that was modified incorrectly, what’s arguably more helpful is to know where we put the wrong thing into the list in the first place.

```py
from dataclasses import dataclass

@dataclass
class MenuItem:
   title: str | None
   details: list[str]

def first_three_lines(menu_item: MenuItem) -> list[str]:
   lines = []
   if menu_item.title is not None:
       lines.append(menu_item.title)
   # bug: `list.append()` should be `list.extend()`
   lines.append(menu_item.details)
   return lines[:2] # error: `list[str | list[str]]` is not assignable to `list[str]`
```

In the example above, the error would be raised on the return statement (saying that `list[str | list[str]]` can’t be assigned to `list[str]`), even though the issue we actually need to fix is on the previous line. While this example is only off by a single line, in the real world the location of the error and the location of the bug could be separated by hundreds of lines of code.

Implementing this empty container inference strategy in a type checker faces several practical engineering hurdles.

- Firstly, finding all usages of a variable (especially those in non-local scopes like global variables or class attributes) can be computationally expensive, so a type checker may need to compromise on exhaustiveness to maintain good performance.
- Secondly, if a container has many distinct usages, the resulting inferred element type can be a lengthy and complex union, and this complexity can negatively impact the readability of subsequent error messages, necessitating the use of additional heuristics to improve error message quality.

## Strategy 3: Infer the container type from only the first usage

This is similar to strategy 2, but we infer the type based only on the *first* time the list is used, where “first” is defined in terms of syntactical appearance.

```py
def my_func2(some_condition):
  x = []
  if some_condition:
    x.append(1) # `x` is inferred as `list[int]`
  else:
    x.append("foo") # error: cannot append `str` to `list[int]`
  return x
```

For example, in `my_func2()` above, the type checker infers the type for `x` based on its first usage `x.append(1)`, which results in `list[int]`.  When we do `x.append("foo")` on the following lines, the type checker emits an error saying we cannot insert `str` into `list[int]`.

If `x.append("foo")` is actually intentional, we can overrule the type checker’s guess by annotating the empty container with `list[int | str]`.

This is what Mypy currently does, and also the default behavior of Pyrefly.

The main benefit of this strategy is that the type errors it raises are closer to the location of the bug, so they are more actionable for finding and fixing bugs.

In our ongoing `first_three_lines()` example, Mypy and Pyrefly would both raise an error on the same line we need to change to fix the type error.

```py
from dataclasses import dataclass

@dataclass
class MenuItem:
   title: str | None
   details: list[str]

def first_three_lines(menu_item: MenuItem) -> list[str]:
   lines = []
   if menu_item.title is not None:
       lines.append(menu_item.title)
   # bug: `list.append()` should be `list.extend()`
   lines.append(menu_item.details) # error: cannot append `list[str]` to `list[str]`
   return lines[:2]
```

Of course, this guessing strategy is merely a heuristic that assumes the first usage of the container signifies the programmer's intent.

When the type checker guesses wrong it can lead to false positive type errors. If the developer actually intended to create & return a  `list[str | list[str]]`, they can overrule the type checker’s inferred type by annotating the variable like `lines: list[str | list[str]] = []`. Unlike the first strategy which requires annotations on every empty container for safety, this approach provides type safety while only requiring annotations when you disagree with the type checker.

## Summary

The challenge of inferring types for empty containers in Python is more than a technical curiosity. It has real implications for code safety, developer productivity, and the effectiveness of type checkers. As we've seen, each strategy for handling empty container inference comes with its own trade-offs:

- **Inferring Any** (Pyright, Ty, Pyre) is the easiest approach that will throw the least amount of type errors, but sacrifices  type safety.
- **Inferring from all usages** (Pytype) closely mirrors Python's runtime behavior, but it can complicate the process of identifying the root cause of bugs, particularly when the resulting errors manifest long after the initial mistake.
- **Inferring from the first usage** (Mypy, Pyrefly) provides more actionable error messages, helping developers fix issues at their origin, but may result in false positives if the initial usage doesn't reflect the programmer's true intent.

Ultimately, the choice of strategy depends on the priorities of your project and team, whether you value permissiveness, runtime fidelity, or actionable feedback.

The Pyrefly team believes that type safety is not the only goal of building a type checker – type errors also need to be actionable and easily-understood by users. As a result, we adopt the first-use inference by default which we believe gives the best balance of actionable error messages and type safety. We also recognize the need for greater flexibility from some users, so we've provided the ability to [disable first-use inference](https://pyrefly.org/en/docs/configuration/#infer-with-first-use), which would make Pyrefly behave more like Pyright.

What’s your take on empty container inference? Share your thoughts or questions with us on [Discord](https://discord.gg/Cf7mFQtW7W) or in a [GitHub Discussion](https://github.com/facebook/pyrefly/discussions)\!

---

[^1]:  Pyright’s behavior is actually [more intelligent](https://microsoft.github.io/pyright/#/type-inference?id=empty-list-and-dictionary-type-inference). But the fact remains that the inference algorithm it uses does not take any downstream context into consideration.
