---
title: Why Today’s Python Developers Are Embracing Type Hints
description: What is Typed Python? Why is it important for Python developers today? How to can you get started?
slug: why-typed-python
authors: [abbym, pyreflyteam]
tags: [typechecking, python-typing-basics]
image: http://pyrefly.org/assets/images/why-typing-blog-6798a36e31db2a87069bd3636c0c97c2.png
hide_table_of_contents: false
---

![](./blog_imgs/why-typing-blog.png)

Python is one of the most successful programming languages out there, with it recently overtaking Javascript as the most popular language on GitHub, according to the latest [GitHub Octoverse report](https://github.blog/news-insights/octoverse/octoverse-2024/). The report emphasises the popularity of the language in the growing fields of AI, data science and scientific computing \- fields where speedy experimentation and iteration are critical, and where developers are coming from a broad range of STEM backgrounds, not necessarily computer science. But as the Python community expands and projects grow from experiments to production systems, that same flexibility can become a liability.

That’s why today we’re going to talk about typed Python \- what it is, why it’s become important for Python developers today, and how to get started using it to write higher quality, more reliable code.

<!-- truncate -->

## What is Typed Python?

Before we dive into why you should be using typed Python in your daily development lives, first we need to understand some core concepts and how we got here.

### Dynamic vs static typing

The classic Python programming language that you know and love is _dynamically typed_. What does that mean exactly? It means that types are determined at runtime, not when you write your code. Variables can hold any type of value, and you don't need to declare what type they are.

Here’s an example of dynamic typing in action:

```py
x = 5        # x is an integer
x = "hello"  # now x is a string
x = [1,2,3]  # now x is a list
```

This behaviour is one of the things that sets Python apart from languages that are _statically typed_, like Java or C++, which require you to declare types from the get go:

```c
int x = 5;
std::string x_str = "hello";
std::vector<int> x_vec = {1, 2, 3};
```

In the above example we can’t just reassign the variable `x` to a value of whatever type we want, it can only hold an integer because of the static typing nature of the C++ language.

The fact that Python is a dynamically typed language is one of the reasons it is so easy to use and popular amongst new and experienced programmers alike. It makes it easy to develop quick demos, experimental research and proof of concepts, without needing to spend precious development time declaring types. This flexibility has been instrumental in Python's adoption in AI, data science, and scientific computing, where researchers need to rapidly iterate and experiment with different approaches.

However… (surely you knew there was a “but” coming?)

We are quickly moving past the “proof-of-concept” phase for many of these industries. AI and machine learning efforts are actively being integrated into production applications, and with that comes production-level expectations of reliability and stability. Relying on dynamic typing opens these codebases up to a certain level of risk that may not be acceptable at the scale they are now expected to operate.

### Enter PEP 484: Static Typing Comes to Python

Cast your mind back to September 2014: Germany has just won the world cup, skinny jeans are still in fashion and Taylor Swift’s “Shake it Off” is number 1 on the charts. That same month [PEP 484](https://peps.python.org/pep-0484/) was first created, proposing the addition of type hints to Python, and fundamentally changing how future developers would be able to write and maintain Python code.

With PEP 484’s acceptance and introduction in Python 3.5, developers could now use static type annotations to declare the expected data types of function arguments and return values, and [subsequent PEPs have continually added more features](https://pyrefly.org/en/docs/python-features-and-peps/) to expand and refine Python's type system. Today you can write statically typed Python statements like this:

```py
def my_func(x: int, y: str) -> bool:
    z: str = str(x)
    return z == y
```

The key innovation of PEP 484 was introducing a _gradual_ type system that allows developers to slowly add type annotations over time without breaking existing code. The system works by:

- Only type-checking functions that have explicit return or parameter annotations
- Introducing the `Any` type as an escape hatch that has all possible attributes
- Assuming untyped functions implicitly return `Any`

This approach has meant developers can incrementally adopt typing, while still allowing them to take advantage of the default dynamic typing approach that makes Python so easy to work with and ideal for quick prototyping.

## Benefits of Python Type Hints: Write Better Code, Faster

So why specifically should you start using type hints in your Python code? Python type hints offer a range of advantages that can significantly improve the quality, maintainability, and scalability of your codebase, at the same time making it easier for other developers to understand your code and collaborate with you.

### Types help you catch bugs early

Type hints assist static analysis tools in identifying mismatches and potential errors before the code is executed, allowing for early bug detection. Take the following example:

```py
def add_numbers(a, b):
       return a + b
...
add_numbers(3, "4")  # Potential error
```

The above error might be easy to spot when you’re calling the function so close to where you’re defining it, but imagine you’re working across multiple files and/or with many lines of code separating them \- suddenly it’s not so easy\!

In comparison, if you’re using type hints in conjunction with a typechecking tool (such as Pyrefly or MyPy), you can catch this error much earlier \- when you’re actually writing the code, rather than when it fails at runtime:

```py
def add_numbers(a: int, b: int) -> int:
  return a + b
...
add_numbers(3, "4")  # a typechecker will catch this error at time of writing
```

Using a typechecker to highlight these types of errors also ensures you can catch an error like this even if you’ve missed this code path in your unit tests.

### Typed code is self-documenting

Another benefit of writing typed Python is that using function signatures and variable annotations provide clarity of intent for a given piece of code. In other words, it makes code easier to read and review. It makes refactoring safer and more predictable. It even helps new team members get up to speed quickly on what’s going on in your codebase without wasting their own time, or yours\!

Take the following example, without type hints you have to carefully read the internal function code to understand what type of parameters will work and what will be returned:

```py
def calculate_stats(data, weights):
    total = 0
    weighted_sum = 0
    for i, value in enumerate(data):
        if i < len(weights):
            weighted_sum += value * weights[i]
            total += weights[i]
    avg = weighted_sum / total if total > 0 else 0
    return avg, len(data)
```

With this version, you can tell instantly what type of arguments you should be passing and what you should expect to get back \- saving precious dev time and just generally making your life easier:

```py
def calculate_stats(data: list[float], weights: list[float]) -> tuple[float, int]:
    total = 0
    weighted_sum = 0
    for i, value in enumerate(data):
        if i < len(weights):
            weighted_sum += value * weights[i]
            total += weights[i]
    avg = weighted_sum / total if total > 0 else 0
    return avg, len(data)
```

I know I know \- ideally all developers should be adding clear docstrings with every function they write, but we know in reality it doesn’t always shape up that way\! Adding type hints is quicker than writing a typical docstring, won’t go stale (if enforced using a typechecker) and is better than no documentation at all. Modern Python typecheckers also have [IDE extensions](https://pyrefly.org/en/docs/IDE/) that include autocomplete functionality to make life easier.

### Typed Python helps you scale from proof-of-concept to production-ready

One of the most important benefits of using type annotations in your code is that it helps you scale your code faster and with less risk. For developers today, the pipeline from experimental code to production systems moves faster than ever, especially in AI and machine learning workflows where research prototypes must quickly evolve into robust, scalable applications.

For example, say there is a team of data scientists that has just published their findings and now needs to operationalize their models. If their published code already includes type hints it makes it much easier, quicker and safer for an engineering team to step in and integrate that research into production applications. In situations like these, type annotations act as a contract between different stages of development, making it clear how data flows through complex, multi-step processing pipelines. This is particularly valuable in AI workflows where a single type mismatch, like passing a NumPy array where a PyTorch tensor is expected, can cause silent failures or performance degradation that only surfaces under production load.

## Get Started with Typed Python today\!

So now you know what typed python is and why you should be doing it, how can you actually get started adding types to your code?

### Step 0 \- start early\!

As a general rule of thumb, the earlier in a project you start adding type annotations the better.
Type hints are much easier to add as you go than to retrofit across an entire codebase later.

As we’ve mentioned before, one of the great benefits of Python is that its dynamic typing default makes it very flexible and easy to get started with. So when you’re doing your initial experimentation and prototyping maybe you’re not thinking about making sure it’s type safe \- and that’s ok\! But as soon as you start to think your project might be going somewhere, if more than one person might be working on it, using it or just reading it, you should start adding type hints.

### Step 1 \- install a type checker

Choose and install a type checker that fits your needs. Typecheckers leverage the code annotations you write to provide important errors and warnings to ensure your codebase is type safe.

At Meta, we recommend [Pyrefly](https://pyrefly.org/), our new open-source type checker built in Rust. Pyrefly is designed to scale from small projects to massive codebases incredibly fast, while providing excellent developer experience. Read the [Pyrefly documentation](https://pyrefly.org/en/docs/) to understand configuration options and best practices, then start adding simple type hints to new functions before gradually working your way up to more complex scenarios.

You should also consider working with a typechecker that [supports IDE integration](https://pyrefly.org/en/docs/IDE/) to get real-time feedback as you write code. Pyrefly provides extensions for editors like VS Code, PyCharm, and Vim which will highlight errors and provide autocomplete suggestions based on your type annotations.

Adding your typechecker to your CI processes is also valuable for maintaining code quality at scale. You can [configure your CI/CD pipeline to run type checking](https://pyrefly.org/en/docs/installation/#add-pyrefly-to-ci) on every pull request, treating type errors as build failures.

### Step 2 \- make use of resources to get better at typing

Typing is one of those skills that gets better the more you practice it in your code, but there are also great resources out there for getting to grips with the functionality and diving deeper into the concepts:

- Official [Python typing](https://docs.python.org/3/library/typing.html) documentation \- The typing module docs provide comprehensive coverage of all available types
- [PEP 484 and related PEPs](https://pyrefly.org/en/docs/python-features-and-peps/) \- Understanding the foundational specifications helps you grasp the "why" behind typing decisions
- Documentation for your chosen typechecker, e.g. [Pyrefly Docs on learning typing](https://pyrefly.org/en/docs/typing-for-python-developers/)
- Join community forums and get support, e.g. [Pyrefly Discord](https://discord.gg/Cf7mFQtW7W), [Typing Discourse](https://discuss.python.org/c/typing)

## Conclusion

So there you have it \- a quick trip around the world of Python typing\! By now you’ve hopefully learnt that type hints aren't just another Python feature to add to the long list of things you’ll _definitely_ get round to implementing eventually \- they're a practical investment in your code's future. The upfront effort of adding type hints pays dividends in reduced debugging sessions, smoother code reviews, and fewer production issues. Most importantly, they give you the confidence to refactor and scale your codebase without fear of breaking things in unexpected ways. Start small by adding type annotations to your next function, add a type checker to your workflow, and before you know it writing typed Python will be second nature. Your future self (and your users and teammates\!) will thank you.
