---
title: "FwPython, Part 1: Setting the Stage"
description: FwPython is a compact, Python-like object-oriented language formalized in Lean 4, built to put controlled pressure on the parts of Python typing that type checkers actually have to reason about. This first post sets up the source language and runtime model.
slug: fwpython-part-1
authors: [jiachen]
tags: [fwpython, formalization, typing, lean]
hide_table_of_contents: false
---

*Companion post to my PyCon 2026 Typing Summit talk, "From Soundness to Blame: AI-Assisted Formalization of a Tiny Python." This is the first in a series.*

<!-- truncate -->

## What this series is about

Python's type system is practical, useful, but full of edge cases.

As a developer for Pyrefly, I frequently run into questions where the right behavior is not fully settled by the typing spec:

- How should narrowing behave in a union? What's the right interaction between `isinstance` and `Any`?
- When two callable intersections meet an `Any` argument, what comes back?
- The conformance tests cover some of this; the spec covers some of this; past that, type checker authors still have to make choices.

Those choices are usually local when you first encounter them. A rule looks reasonable in one example, questionable in another, and subtly different from what another checker does. I wanted a way to take some of those local rules and push them through the entire language: how do they interact with, e.g. method lookup, subtyping, narrowing, and calls? Do they still behave sensibly when other rules have to agree with them? Do they break something that previously seemed settled, or does it expose a real tradeoff?

FwPython is a small language for doing that experiment.

It is a compact object-oriented language formalized in [Lean 4](https://lean-lang.org/), a theorem prover and functional programming language. It is Python-like enough to talk about classes, inheritance, method lookup, first-class bound methods, callables, and branch-sensitive `isinstance` tests, but small enough that the runtime, type checker, and the claims connecting them can all be checked by one machine.

That makes FwPython useful as a model, but also defines its limits.

- It keeps nominal classes, multiple inheritance through [C3 linearization](https://www.python.org/download/releases/2.3/mro/), runtime method lookup, first-class bound methods, strict boolean conditionals, and branch-sensitive `isinstance` tests.
- It leaves out modules, top-level functions, fields, mutation, constructors, descriptors, metaclasses, decorators, async, Python's full argument-binding rules, generics, etc.
- A proof about FwPython is not automatically a proof about Python. The point is to put controlled pressure on the parts of Python typing that Pyrefly and other checkers actually have to reason about.

This first post sets up the source language and the runtime model. Later posts will use that setup to discuss the type system, soundness, gradual typing, and more.

## The components

The Lean development has five connected pieces:

1. **Source language**: the syntax of FwPython programs.
2. **Interpreter**: the small-step runtime semantics.
3. **Type checker**: the static rules for accepted programs.
4. **Theorems**: the formal claims relating typing to runtime behavior.
5. **Proofs**: the machine-checked arguments for those claims.

Lean matters here because definitions and proofs live in the same environment. The interpreter, type system, and theorem statements are ordinary Lean definitions; the proof terms are checked by Lean's kernel. For this post, the important consequence is that a claim about FwPython has to line up with the actual definitions. The runtime described below is the runtime the later proofs use.

## The language

A FwPython program has two components: a *class table* and a *main expression*. There are no top-level statements, no module system, and no separate function definitions. Methods live inside classes; classes live in the class table; the main expression is what gets evaluated when the program runs.

Here is the running example for this post. It uses Python-like notation for readability. I write `Callable[[Object], Object]` as blog notation for FwPython's unary callable type: an object that can be called with an `Object` and returns an `Object`.

```py
class Animal:
    def speak(self, x: Object) -> Object:
        return x

class Dog(Animal):
    def speak(self, x: Object) -> Object:
        return True

class SpeakerBox:
    def get_speaker(self, x: Dog) -> Callable[[Object], Object]:
        return x.speak

# main expression
((new SpeakerBox).get_speaker(new Dog))(None)
```

The main expression allocates a `SpeakerBox`, allocates a `Dog`, asks the box for that dog's `speak` method, and then calls the returned method value on `None`. The lookup for `speak` is ordinary dynamic dispatch through the dog's MRO; the extra point is that the selected method can leave the original expression as a first-class value. The whole program evaluates to `True`.

This example is small, but it touches several features FwPython is meant to isolate: nominal inheritance, runtime method lookup, first-class bound methods, and a call through a method value.

### Classes and methods

A class declaration carries three pieces of information:

- the class name
- a list of direct base classes
- a list of methods

A method declaration carries six pieces:

- the method name
- the receiver parameter name, like `self`
- one ordinary parameter name
- the parameter's type annotation
- the return type annotation
- the body expression

This is intentionally a small language. Classes have methods but no fields, so the core calculus can focus on dispatch, callability, subtyping, and narrowing without threading mutation through every invariant. Read-only field-like behavior can be encoded with a method that ignores its argument and returns a constant.

Methods are unary: one explicit argument in addition to the receiver. That is enough to exercise method lookup and callable variance while avoiding Python's positional, keyword, default, and variadic argument machinery. Those details matter for a production checker, but they would obscure the particular metatheory this project is trying to study.

If a class declares no explicit base classes, the runtime treats it as implicitly inheriting from `Object`, unless the class itself is `Object`. Multiple inheritance is supported through C3 linearization, the same MRO algorithm Python uses. In the running example, `Dog`'s MRO starts with `Dog`, then `Animal`, then `Object`, so `x.speak` resolves to `Dog.speak` before `Animal.speak`.

### Expressions

FwPython has nine expression forms.

| Form | Reading | What it does |
| :---- | :---- | :---- |
| `x` | variable | Look up `x` in the runtime environment. |
| `new C` | constructor | Allocate a fresh object of class `C`. No `__init__` runs. |
| `e.m` | attribute access | Resolve method `m` through the MRO of `e`'s class, allocate a fresh bound-method object, return its reference. |
| `e1(e2)` | call | Invoke `e1` on `e2`. Succeeds only when `e1` is a bound-method object. |
| `None` | constant | The distinguished `None` object. |
| `True` | constant | The distinguished `True` boolean object. |
| `False` | constant | The distinguished `False` boolean object. |
| `if e1 then e2 else e3` | conditional | Evaluate `e1`; on `True` continue with `e2`, on `False` continue with `e3`. |
| `isinstance(x, C)` | type test | Test whether the runtime class of the value bound to variable `x` lies under class `C` in its MRO. |

Four choices are worth calling out.

`new C` allocates an ordinary object and does not invoke a constructor. FwPython is studying object identity, dispatch, and method values, so initialization order and missing-field reasoning are outside this model. The evaluator also rejects `new Bool`, `new NoneType`, and `new BoundMethod`; those built-ins have controlled allocation paths in the runtime.

`isinstance(x, C)` takes a variable name as its first argument. That restriction exists because `isinstance` is also a narrowing construct. After the test, the type system can refine the environment entry for `x`; allowing an arbitrary expression would require a separate account of what value is being tracked across the branches.

Conditionals use exact booleans. A condition must evaluate to the fixed `True` object or the fixed `False` object. Python's truthiness protocol is rich and useful, but modeling it would add another dispatch mechanism before the series even reaches the typing questions it is trying to expose.

Attribute access is the heavyweight expression form. Evaluating `e.m` means evaluating `e`, looking up `m` through the runtime class's MRO, allocating a new heap object that stores the receiver and selected method declaration, and returning a reference to that heap object. This mirrors the Python behavior that makes `obj.method` a value you can pass around and call later.

## The runtime

The interpreter is deliberately boring at the value level. A runtime value has one shape:

```
objRef oid
```

Looking up `oid` in the heap recovers the information that matters operationally:

- a runtime class name
- a payload, either ordinary or bound-method

Ordinary payloads carry no extra data. Bound-method payloads carry the receiver object id and the method declaration chosen by lookup. The initial heap contains three ordinary objects at fixed ids: `None` at `0`, `True` at `1`, and `False` at `2`.

FwPython's type system needs to talk about several views of the same runtime value: "this object has class `C`", "this object is callable with parameter type `P` and return type `R`", and "this object's runtime class is not under `C`". All three are evidence about one heap object rather than separate value constructors.

### Why bound methods are objects

Python has many callable carriers: functions, lambdas, bound methods, descriptors, built-in functions, callable instances, and class objects. FwPython collapses that space to one producer of callable values: method binding. Attribute access resolves a method through the receiver's MRO, allocates a `BoundMethod` heap object, and returns its object reference.

In the running example, evaluating `x.speak` in the body of `get_speaker` performs three runtime steps:

1. look up the object currently bound to `x`;
2. compute that object's MRO and find `speak`;
3. allocate a `BoundMethod` object whose payload stores the receiver id and the selected `Dog.speak` declaration.

The final expression then calls the object returned by `get_speaker`. The call rule inspects the callee's heap entry. If the payload is bound-method data, the machine evaluates the argument, runs the stored method body with the receiver and parameter bound, and restores the caller environment afterward. If the callee is an ordinary object, the machine reports a runtime error.

The collapse to one callable carrier is a modeling choice to keep the formal story tied to method lookup plus first-class method values. A method value can be returned, stored in the environment, passed as an argument, and invoked later, and the runtime handles all of those cases with the heap object created by attribute access.

This also makes the later typing rules simpler to state. If the type checker says that a value has type `callable P R`, the runtime fact behind that claim is always the same: the value points to a `BoundMethod` object whose payload contains a receiver and a method declaration. The checker can then validate the callable type by checking the stored method's parameter and return types, and by checking if the method really came from a lookup on the receiver's class.

But there's a trade-off. FwPython skips over things like user-defined `__call__`, stand-alone functions, Class-object-as-callable, or full-blown descriptors. Sure, these are real parts of Python, but adding them would just add more "callable carriers" to track and significantly increase the proof workload. Instead, I've decided to spend that complexity budget focusing on method dispatch, bound methods, and the typing rules that govern them.

### Why the built-ins are ordinary heap objects

Four distinguished classes, `Object`, `Bool`, `NoneType`, and `BoundMethod`, live in the same class table as user-defined classes. Most machinery sees them as ordinary class entries, with a small number of runtime and typing rules giving them fixed meaning:

- `Object` is the nominal root of the hierarchy. Every class transitively inherits from `Object`.
- `Bool` is the runtime class of `True` and `False`.
- `NoneType` is the runtime class of `None`.
- `BoundMethod` is the runtime class of every bound-method object.

The constants are ordinary heap objects because the runtime has no special boolean or none value form. Conditionals distinguish truth from falsity by the two fixed object ids. `isinstance` treats `None`, `True`, and `False` by the same MRO test it uses for user-created objects.

The evaluator also refuses `new Bool`, `new NoneType`, and `new BoundMethod`. That restriction preserves the heap invariants the proofs rely on: the only boolean objects are the two fixed constants, the distinguished `None` object stays at its fixed id, and bound-method objects are introduced by method lookup with a receiver and method payload already attached. In the class table, `Bool`, `NoneType`, and `BoundMethod` are leaf built-ins, which is what lets the later negative-type rules make open-world-safe claims about them.

## Where this leaves us

So far we have the source language and the runtime model. Later posts will build on this setup to discuss the type system:

- if narrowing for `isinstance(x, C)`
- negation type and its restrictions
- subtyping rules
- how we could formally extend a fully static type system with gradual types like `Any`

Those topics are the reason for the small language above. The runtime carries just enough evidence for the type system to make precise claims, and no more than the later proofs are prepared to justify.

The project is open source at [github.com/grievejia/FwPython](https://github.com/grievejia/FwPython).
