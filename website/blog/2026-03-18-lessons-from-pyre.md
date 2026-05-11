---
title: Lessons from Pyre that Shaped Pyrefly
description: Lessons from developing Pyre that influenced how we designed Pyrefly.
slug: lessons-from-pyre
authors: [yangdanny, jiachen, stroxler]
tags: [typechecking]
hide_table_of_contents: false
---

# Lessons from Pyre that Shaped Pyrefly

[Pyrefly](https://github.com/facebook/pyrefly) is a next-generation Python type checker and language server, designed to be extremely fast and featuring advanced refactoring and type inference capabilities. This isn’t the Pyrefly team’s first time building a type checker for Python: Pyrefly is a successor to [Pyre](https://pyre-check.org/), the previous type checker our team developed.

A lot of Pyrefly’s design comes directly from our experience with Pyre. Some things worked well at scale, while other things were harder to live with day-to-day. After running a type checker on massive Python codebases for a long time, we got a clearer sense of which trade-offs actually mattered to users.

This post is a write-up of a few lessons from Pyre that influenced how we approached Pyrefly.

<!-- truncate -->

## The Origins of Pyre

Pyre started around 2017, with roots in [dataflow analysis tooling](https://pyre-check.org/docs/pysa-basics/) for Instagram. The Python tooling ecosystem looked pretty different back then:
- The [typing specification](https://typing.python.org/en/latest/spec/) didn't exist, and the type system was loosely "specified" through a series of PEPs describing individual features.
- Many modern typing features, such as literal types, dataclass transforms, param specs, type guards, etc. didn’t exist yet.
- The [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) was new and not yet the clear standard.
- [Mypy](https://github.com/python/mypy) was, for all practical purposes, the only viable Python type checker available at the time.
- Type checking massive monorepos written in Python was still an unsolved problem.

These factors shaped the development of Pyre, but resulted in some limitations later on as the Python tooling ecosystem matured.

## Language-server-first Architecture

Pyre didn’t start out as a language server; it was designed as a static analysis tool you run in CI (continuous integration) or from the CLI (command line interface).

Consequently, its design prioritized **throughput** to minimize the total wait time for users by maximizing parallel CPU usage. Adapting this CLI tool to function as an editor-integrated language server presented a significant challenge, as an editor environment demands high priority on **latency**, not throughput.

Applying Pyre's throughput-focused strategy to an IDE degraded the user experience, often consuming excessive machine resources and potentially freezing the editor interface while performing computations the user might not currently need.

This struggle with Pyre's latency in the IDE led to a temporary move to Pyright as the language server for Meta developers. While this provided some features, it introduced other user experience issues, such as mismatched type errors and inconsistent IDE hover results.

A core design goal for Pyrefly is to create a system flexible enough to handle both throughput-oriented workloads (like CLI type checking) and latency-sensitive workloads (required by the IDE/language server).

Additionally, error-recovery during parsing is much more important for a language server than for a CLI-only type checker. Language servers need to continue working when you're halfway through an edit, while type checkers are typically run only when you're done with an edit. Pyre had several options for parsing, including a Menhir-based parser and directly calling the CPython parser, but these were not robust to syntax errors. In Pyrefly, we use Astral's excellent [Ruff parser](https://github.com/astral-sh/ruff/tree/main/crates/ruff_python_parser), which is both speedy and battle-tested.

## OCaml vs. Rust: When Your Ecosystem Becomes a Ceiling

Pyre is implemented in OCaml. To bootstrap quickly, we borrowed a multi-processing and shared-memory library from mature OCaml type checkers at Facebook, like [Hack](https://hacklang.org/) and [Flow](https://flow.org/). This architecture was necessary because, prior to OCaml 5, the runtime relied on a global lock (much like Python's GIL) that prevented true multi-threading. Consequently, achieving parallelism meant relying entirely on OS-level process forking and inter-process communication (IPC).

While this borrowed architecture accelerated our initial development, it eventually forced us into a structural straitjacket:
- **Data structure rigidity:** Because of the IPC layer, shared data had to be crammed into string-to-string hashtables. Storing dynamic, complex objects (like self-referential types or closures) directly into shared memory incurred massive serialization and deserialization overhead.
- **Algorithmic inflexibility:** The IPC model is heavily biased toward throughput-oriented, strict "fork-join" workloads. This became increasingly awkward as we shifted toward latency-sensitive, demand-driven computational models that need to decide what to compute on the fly.

When OCaml 5 introduced native multi-threading, upgrading the compiler was tempting. However, we realized that language-level concurrency cannot erase historical platform baggage. OCaml's ecosystem has historically struggled with Windows, and as we deployed Pyre more and more broadly, we ran into platform related frictions more and more often.

Rust radically changes this equation by treating Windows, macOS, and Linux as true equals out of the box. Adopting an ecosystem where cross-platform reliability is a foundational tenet eliminated massive operational friction and completely unified our development experience when hacking on Pyrefly.

Switching to Rust also allowed us to grow our community of contributors much faster than if we had continued with OCaml.

## Irreversible AST Lowering

Early in the development of Pyre, a technique was used to accelerate iteration: leveraging the semantic similarity or equivalence between certain Python syntax patterns.

For instance, the following pairs are functionally similar:

| Construct 1 | Construct 2 |
| --- | --- |
| A match statement | A series of chained if/else statements |
| The legacy union type syntax  `Union[X, Y]` | The new syntax `X \| Y` |
| A NewType definition `X = NewType("X", Y)` | A standard class definition `class X(Y): pass` |
| The functional namedtuple definition `Point = namedtuple("Point", ["x", "y"])` | A class-based namedtuple definition `class Point(NamedTuple)` with `x` and `y` as members |

Once code was implemented to handle one syntax pattern, support for similar or identical patterns could be rapidly added by using AST transformation passes to "lower" the latter into the former.

While useful, this technique introduced risks: If the original range and position information were not carefully preserved, the quality of diagnostics could suffer to the point where they could not adequately support language server features.

A significant example of taking this AST lowering too far in Pyre involved how variable scopes were handled. Pyre included an AST transformation pass designed to perform alpha-conversion: renaming syntactical definitions to resolve name conflicts.

Consider this original code:

```python
from foo import x
def test():
  y = x
  x = "bar"
  z = x
```

The transformation process rewrote it as:

```python
from foo import x
def test():
  y = foo.x
  _local_test_x = "bar"
  z = _local_test_x
```

This transformation successfully resolved the name conflict between the imported `x` from `foo` and the local variable `x`. However, a crucial piece of information was lost: the transformed code obscures the fact that the `test()` function never directly references the module `foo`. Losing this kind of detail can significantly impacted certain language server operations, such as refactoring and find-references.

Learning from Pyre, we made a conscious effort to minimize AST fabrication or transformation in Pyrefly’s design.

## Attribute narrowing: Soundness vs. Usability

Pyre was originally designed with a strong focus on security, leading to a high emphasis on the soundness of its type checking. Security engineers generally favor a low tolerance for false negatives (missed errors), even if it means a higher tolerance for false positives (incorrect warnings). Consequently, Pyre is quite strict and opinionated about Python code style.

A prime example of this strictness is how Pyre handles type narrowing for attribute access. Consider this code:

```python
class C:
  x: int | str = ...

def narrowing_test(c: C) -> int:
  if isinstance(c.x, int):
    some_other_function(c)
    return c.x   # Pyre complains: `c.x` might be `str` here, incompatible with return type `int`
  ...
```

Technically, Pyre is correct. It cannot safely assume that `some_other_function()` won't mutate `c.x` to a different type, either directly or through an alias. To satisfy Pyre, users were forced to rewrite the code for safety:

```python
def narrowing_test(c: C) -> int:
  c_x = c.x
  if isinstance(c_x, int):
    some_other_function(c)
    return c_x
  ...
```

This strictness significantly impacted ergonomics, and complaints about this specific behavior were among the most frequent feedback we received.

In Pyrefly, we made a deliberate choice to allow this kind of type narrowing, prioritizing a better user experience for more general Python audiences. This represents a conscious trade-off: in a gradually typed language, enforcing extreme soundness can make the tool feel overly brittle.

## Caching Cyclic Data Dependencies

Pyre was designed with a strict, sequential phased structure for its internal computations to ensure fast, incremental type checking. To maintain simple and cost-effective cache invalidation, Pyre enforced a rigid rule: computations could only rely on the cached outputs of preceding phases. This method generated a perfectly acyclical dependency graph, enabling the system to update its cache via a single topological sweep.

However, while this strictly unidirectional approach kept the cache invalidation logic simple, it was fundamentally at odds with how Python code is written in the real world.

The problem is that effective type checking Python naturally relies on recursive or "same-phase" dependencies. Consider a simple class hierarchy:

```python
class A: ...
class B(A): ...
class C(B): ...
```

To compute the Method Resolution Order (MRO, the chain of class inheritance in Python) for `C`, it logically makes sense to just look up the already-computed MRO for `B`. However, Pyre couldn’t do this. Allowing `C` to depend on the cached result of B meant creating a dependency within the same phase. This violated the strict acyclic rule required by the caching system, meaning Pyre was unable to use its cache here. To get around this architectural wall, Pyre would have to recompute the MRO of `B` entirely from scratch, compute and cache `C`, and then throw `B`'s intermediate data away.

While simple structural checks like MRO incurred an acceptable cost for this redundant work, the absence of intermediate caching created significant performance bottlenecks for more complex computations. This was particularly true for heavyweight tasks, such as inferring attribute types in the presence of descriptors and decorators (which itself necessitates inferring the types of other attributes used as such). Consequently, the type checker ran extremely slowly when encountering decorator-heavy code patterns.

This limitation drove the foundational design of Pyrefly. Instead of forcing the dependency graph to remain acyclic, Pyrefly’s engine was built from the ground up to embrace cycles. By incorporating cycle detection and fixpoint resolution directly into the underlying system, Pyrefly can safely untangle and cache recursive dependencies. This algorithmic shift absorbs the complexity at the infrastructure level, allowing the tool to run fast while empowering the type checker to use smarter, more flexible inference heuristics without artificial restrictions.

## Conclusion

Building a type checker is ultimately a deeply pragmatic exercise. It is easy to get lost in the weeds of type theory, but at the end of the day, we are building a tool that developers have to live with every single day. Pyre taught us how to scale to massive monorepos, but more importantly, it showed us exactly where the friction lies.

Pyrefly is the direct result of those hard-earned lessons. We looked at the paper cuts, the IDE latency, and the rigid rules that frustrated our users in Pyre, and we made fixing them our foundational requirements. We chose usability over absolute soundness, and we built an architecture that embraces Python's dynamism instead of fighting it.

But the most important lesson we applied to Pyrefly isn't about code; it’s about how we build. Pyre was born as an internal Meta tool that was later released to the outside world. With Pyrefly, we flipped the script. We’ve been building this in the open, from day one, specifically for the broader Python ecosystem. We want to build a tool that actually makes your day-to-day coding experience better, and we can't do that in a vacuum.

**Come help us build Pyrefly!** We’ve gratefully accepted hundreds of PRs from dozens of open-source contributors on [Github](https://github.com/facebook/pyrefly) so far, and we wouldn’t be where we are without them. Jump into our [Discord](https://discord.gg/Cf7mFQtW7W), tell us what you are interested in, point out the rough edges, and help us figure out what lessons we need to learn next.
