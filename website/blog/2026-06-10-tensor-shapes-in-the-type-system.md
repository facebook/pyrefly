---
title: "Talk: Tensor Shapes in the Type System"
description: Bringing tensor shapes into Python's type system with Pyrefly.
slug: tensor-shapes-in-the-type-system
authors: [avikchaudhuri]
tags: [typechecking, pytorch]
hide_table_of_contents: false
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/HE5EyQW_7eY?si=lv8WcsyIELfDD2Bh" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<br/>

Why aren't tensor shapes part of the type system?

When you write a PyTorch model, the hardest part of composing tensor ops is keeping track of their shapes. The standard practice today is to write shapes down as comments as you go. This talk introduces an experimental Pyrefly feature that brings tensor shapes into Python's type system, so those shape annotations can become inferred type hints instead of comments.

The talk was originally presented at the PyCon US 2026 Typing Summit. The slides and edited transcript are provided below for your convenience.

📎 [Slides (PDF)](https://drive.google.com/file/d/13l3c0IKCtELt2kOa7mVU6ySolrJZTBFB/view)

Want to give it a spin? This feature is [available in Pyrefly today](https://pyrefly.org/en/docs/tensor-shapes/), and we'd love to hear your feedback and suggestions.

<!-- truncate -->

## The problem

Why are we working on this?

If you squint, a PyTorch model is just a function from a tensor to another tensor. In between is a whole bunch of transformations: you apply so-called tensor ops, and all they're doing is transforming these tensors from your input to your output. So you can think of it as a straight-line program, or as a graph.

The hardest part of model development—unless you're a researcher trying to change the architecture—is that once you know the architecture and you're just trying to compose these ops, you have to think about their shapes. Just like in regular programming, where you think about what function to call and how to compose their types, shapes are the unit of types in PyTorch programs, or in models in general.

The standard practice, which holds today, is that you write a bunch of comments in your code to track the shapes, because it's really difficult. These ops are non-trivial—they're doing complicated transformations—so you write down the shapes as comments as you go.

Here is a snippet of a thing called nanoGPT, written by Andrej Karpathy. Some of you might know him; he's a really famous AI researcher, but he's also a very good educator. Very early on, when GPT-2 came out, he came up with this tiny repository called nanoGPT, where he goes through the mechanics of how to put together a real model—but in a very educational way, so people can understand how the different parts of a classic LLM come together.

Even if you don't follow what's going on line by line, note the green comments. He's following a very standard convention of explaining shapes: when he creates a tensor and assigns it to `pos`, it has shape `T`. Then he puts it through a bunch of transformer calls, goes into a loop, and calls a bunch of blocks. These blocks are modules that themselves have other tensor ops inside them. Throughout, he's writing what look like tuples—`B, T`, embedding—and these are all shapes of tensors at that line.

So the question we're trying to answer is: wouldn't it be nice if, just like regular types, you had type hints provided by the type checker, so you don't have to write comments anymore?

This is the same snippet of code—and I'll show more snippets here and elsewhere—where the types just get inferred for you. This is not magic. You are annotating the boundaries of your functions, but that's common practice. We use generics—and I'll talk about what kind of generics we use—but once you do that, the types of all the locals get inferred, and the shapes all line up as inline type hints.

## How it works

That goes into the *why*. Let me talk at a high level about *how* we did this. There are three main ideas. I won't get into a lot of detail, but listen to these three ideas—you could have pieced this together in your heads as well.

### Idea 1: Symbolic arithmetic at the type level

The first idea has nothing to do with tensors per se. It introduces symbolic arithmetic at the type system level. We have a new kind of type argument that represents dimension sizes. Generic types take type arguments; this time, the way to think about this type argument is that it's an integer—but it could be an arbitrary integer, and therefore it could be a symbol, and the symbol is quantified as a type argument somewhere. So it could be a normal integer, it could be a symbol, or it could be an arithmetic expression on these symbols. These are all kinds of `int`, but they're symbolic expressions.

One key design decision I took pretty early is that I didn't want to throw a SAT solver into the type checker. There is no point where we ask a question like "is there an X such that 13 = X + 6?" Instead, we take the view that these could be expressions. We do need to decide equality between symbolic expressions, and the way we do that is by a normalization procedure: we simplify the two sides and then check for syntactic equality. The normalization procedure knows a bunch of equations that should hold, and it does its best. There's no claim that all of this is decidable, but it works pretty well in practice, and that's the limit to which we go. We do not introduce so-called existential types in this system.

### Idea 2: Two user-facing types

The second idea: now that you have symbolic arithmetic at the type level, we introduce just two user-facing types.

One is a tensor type, which can take dimensions as arguments. Any number of dimensions is fine, because you have multi-dimensional tensors. At the same time, if you have a situation where you don't know how many dimensions the tensor has, or you don't know some of the dimensions, you can always use `Any`. That's fine—it's just like regular optional typing.

But these shapes need to come from somewhere. At the base of all operations, you have some tensor creation operations, and they take integers as arguments—some configuration in your model. From those numbers, you decide how large these matrices of weights have to be. So everything comes down to integers at the end of the day, and this is where we wrap that symbolic arithmetic with something else called `Dim`.

Why `Dim`? The mental model is that this is just like `Literal`, except `Literal` takes concrete integers and `Dim` takes concrete *and* symbolic integers. If there were something in the Python type system that wrapped symbolic stuff, I would just use it. So `Dim` is a very light addition. It's not very essential to the system, but you do need such a type to propagate the integers to where they need to go.

### Idea 3: A tiny Python DSL for op behavior

The third idea is probably the most important in terms of practical implementation. At the end of the day, you have a large library of ops, and you need to define their typing behavior. One standard way to do that is type stubs. This works for some ops, and if somebody went out today and started doing this, it's probably the natural thing they'd land on. For example: matrix multiplication. The `mm` operator can be expressed in three sizes—you take a tensor of size `n×k` and a `k×m`, and it produces `n×m`. This is generic; it works for all instantiations of these type arguments. So that's fine.

But PyTorch has thousands of operators, and these operators are quite fancy—not in the sense that they're hard to understand, but in the sense that to specify the transformations they do on shapes, it's much easier to do it programmatically than to keep increasing the expressiveness of the type system. This is where I think some of the prior attempts wanted to encode these operators at the type level, and at some point you have to give up.

The approach we followed is heavily inspired by how symbolic shapes is implemented in the PyTorch compiler itself: for every op, you have a so-called fake op. The fake op mimics the behavior of the original op, but only at the level of shapes—no data. It just answers: given a tensor of this shape, what is the shape of the tensor I will output? I don't care about the data part.

For example, there are ops like reshape, flatten, and squeeze. A simple one is `repeat`, which takes a tensor and repeats it across a bunch of dimensions a certain number of ways. A one-line Python program using a comprehension can express what the shape should be. So what we did here is use a very tiny subset of Python—I hesitate to fully define it, but it's really simple. You can think of it as basically list comprehensions over ints. That turned out to be expressive enough to cover the vast majority of these thousands of ops. They're all declared in a certain way—and of course, a small subset don't even need this, like matrix multiplication, where normal type signatures are fine.

Using these declarations, we can plumb them through using symbolic arithmetic and derive types.

## Examples

This nice little model has tons of examples where everything works beautifully.

This is an MLP—a multi-layer perceptron—a little class that has a small number of tensor transformations. It's a building block for bigger layers. You can see there's some arithmetic going on: `4 * config.n_embedding`. And you can see those numbers show up in the types, so you have those shapes available as types. The class needed to be declared with one generic, `n_embedding`, which I didn't show here.

Another example, slightly more non-trivial: this is in the attention block. You call something called `self.c_attn`, which is a bunch of weights. Look at the second dimension, where you have three times the number of embeddings—that's the size of that dimension. Then you split on that dimension, expecting a three-way split, so you actually have a tuple of three tensors at that point. Then regular typing takes over. You have a bunch of transposes, you have floor div in there—all of these are supported, and they just work out fine.

This is lowered down in the same attention block—the standard implementation of attention, with a bunch of matrix multiplications, some masking, and so on. Again, at every line, Karpathy had to write down the shapes so that the next day when he comes back, they're understandable, and the people he's teaching can look at the code and easily understand it. The type hints just help you do that.

This wasn't just nanoGPT. nanoGPT was a great first example, but we went ahead and ported a whole bunch of other open-source models, covering a range of architectures—modern to a little less modern (who knows what modern means anymore)—and all sorts of coding patterns that are idiomatic in PyTorch. Most of them have no problem going through this, which also tests coverage of ops. The full list of these models is available in the GitHub repository; feel free to look at them.

## Coverage

Some general comments on coverage. The high-level point is that tensor programs might look scary to people who haven't seen them before, but at the end of the day they're pretty simple compositions of things. So it's not a big surprise that we tend to have high coverage, and on average we need very little suppression.

Some of the main sources of coverage loss are the same as in a typical Python program. If you have lists of things that aren't homogeneous, you lose some typing—and the standard workarounds also work. These have nothing to do with tensor checking.

There are a few interesting things I found over and over again in these models, if people want to talk to me later. Sometimes you have a list where the element at position `i` has a type that depends on `i`—the type is a function of `i`. It would be nice to have some way to express that. You can do fancy things like induction in your program, and that kind of works out, but I don't expect normal programmers to think in terms of induction. So the typical thing is to lose shapes on the way in and recover them on the way out through annotations.

Also, when we declare these symbols, we don't give them any information. But if you could declare them with some divisibility information, you could prove some more equations—and sometimes these equations keep coming up. We could choose to ignore them (we don't yet), but we could make this work if we added a bit more baggage at the creation sites.

## Using AI

In the theme of the times, we did use AI quite a bit in this project — mostly to port these models. nanoGPT I did by hand, but what you need to do is very mechanical, so it's natural to use an AI agent to try porting a model so that you don't have to rewrite it manually.

What surprised me was that the LLM was often surprised this sort of thing could even be done at the type-system level — which is not very surprising, since this is a new regime for type checkers. So it often makes pessimistic assumptions, and I have to say, "No, try it, this will work." Then it goes and tries and says, "Wow, okay, I'm so clever."

The next level of automation: when I was manually prompting and nagging it, there's a new wave of "can we automate ourselves out of using agents?" People are writing skills—sets of instructions or workflows that an agent can run in a loop. The port-model skill is available in the repository as well. We tried that, basically one-shotted a few models, and they ported fine. Built in is a loop where, after the LLM ports your model, it grades itself on type coverage and other metrics, reads other models, and decides whether to continue. Most of the time, two tries are enough; sometimes one try works.

So this is the big point I was trying to prove through all these experiments: this is usable in practice, and it is AI-friendly.

## Path to Production Usage

So what's the blocker here?

Some of you might have noticed that I was cheating a little bit, using TypeVars and arithmetic over TypeVars. As it turns out, if you try to evaluate these annotations at runtime, it doesn't work—for a very small reason: TypeVars error out at runtime when you do arithmetic on them.

There are solutions. You can use `from __future__ import annotations`, or we ship with a patched-up TypeVar you can use with old-style generics that works. But if you're using new-style generics, they're hard-coded into the internal TypeVar, which doesn't allow this. So that's the state today.

Other people have been using comparable systems. The most popular is probably jaxtyping, which does this kind of checking at runtime and has its own syntax. The syntax is more heavyweight than what we're proposing here, but it's something people use today. They're not very excited by it, but it's what they have. The jaxtyping syntax puts all of these dimensions inside a string, and we do accept that syntax. We don't have a story for runtime type checking yet, so the interoperation isn't bidirectional, but we can accept jaxtyping annotations. We might be meeting up with those folks in a month or so to discuss what can be done about the runtime parts.

This is something that, if we get it into the hands of model authors, they'll find very easy to use. So if, as a Python typing community, we can do more to enable this kind of thing, that would be great. I do think the syntax is pretty important for adoption, so it would be nice if we could relax that restriction I was talking about.

The other point, a little lower in priority: the symbolic arithmetic part has nothing to do with tensors per se and could be more generally useful—not just for NumPy, but pretty much any user of SymPy might find it useful. So it would be nice to not have this extra user-facing built-in type called `Dim`, and instead have something more general.

## Conclusion

Want to give it a spin? Check out [our docs](https://pyrefly.org/en/docs/tensor-shapes/) for how to enable it in Pyrefly. We'd love to hear your feedback and suggestions for how to make this more useful.

Right now this only works for Pytorch tensors, but rest assured that we plan to add support for other data types (like numpy arrays) in the near future.
