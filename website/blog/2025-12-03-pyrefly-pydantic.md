---
title: Fly through data validation with Pyrefly’s new Pydantic integration
description: Pydantic is a data validation and data parsing library for Python. Pyrefly now supports Pydantic, allowing you to easily validate your data and get type errors before you run your code.
slug: pyrefly-pydantic
authors: [abbym, zeinam, rebeccachen]
tags: [pydantic, typechecking, IDE]
image: https://pyrefly.org/assets/images/pydantic-blog-6b683fc8f527ef950de38ee168925b51.png
hide_table_of_contents: false
---

![](./blog_imgs/pydantic-blog.png)

We’re excited to share that experimental support for Pydantic is now available in Pyrefly\! This integration aims to provide a seamless, out-of-the-box experience, allowing you to statically validate your Pydantic code as you type, rather than solely at runtime. No plugins or manual configuration required\!

<!-- truncate -->

Supporting third-party packages like Pydantic in a language server or type checker is a non-trivial challenge. Unlike the Python standard library, third-party packages may introduce their own conventions, dynamic behaviors, and runtime logic that can be difficult to analyze statically. Many type checkers either require plugins (like Mypy’s Pydantic plugin) or offer only limited support for these types of projects. At the time of writing, Mypy is currently the only other major typechecker that provides robust support for Pydantic.

Read on to learn more about how this works and how to get started, or watch the [video](https://www.youtube.com/watch?v=zXYpSQB57YI)!

##

## **Background \- Pydantic Basics**

Pydantic leverages type annotations for data validation and parsing. Similar to Python’s built-in `dataclasses`, it helps you create structured data containers in your code. But Pydantic takes it a step further by providing extensive runtime data validation, ensuring that data matches the expected types and formats as soon as it’s instantiated.

For example:

```py
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1)
    email: str

# Example usage
user = User(id=123, name="Alice", email="alice@example.com")
print(user)
```

Using the code above, if you were to create a User instance, Pydantic would automatically validate the types and constraints. So if you pass invalid data (e.g., `name=""` or `id="not an int"`), Pydantic will raise a `ValidationError` at runtime.

## **Pyrefly & Pydantic \- How it works**

So how does Pyrefly work with Pydantic? For this initial experimental support we focused on delivering the following experiences:

- **Understands models and constructs**: Pyrefly supports core Pydantic features, including `BaseModel`, `Field`, `ConfigDict`, as well as model-level configuration. No import errors or pesky red squiggles because your typechecker/language server doesn’t understand what Pydantic is.
- **Automatic Recognition**: Pyrefly detects Pydantic models and constructs out-of-the-box. There’s no need to install plugins, add configuration files, or tweak settings. You can just write your Pydantic code and Pyrefly will handle the rest.
- **Static Analysis That Reflects Runtime Logic:** One of the biggest challenges with Pydantic is that much of its validation and type coercion happens at runtime. Pyrefly’s static analysis engine is built to reflect Pydantic’s runtime logic as much as possible, ensuring that what you get in your IDE matches what will happen when your code runs.
- **Immediate Feedback:** As you write and edit your models, Pyrefly provides instant type errors and warnings. This helps you catch mistakes early, reducing debugging time and improving code quality.

For the full list of supported Pydantic features check out the documentation [here](https://pyrefly.org/en/docs/pydantic/#supported-features-with-examples).

### Automatic Recognition \- Write Pydantic-ally while Pyrefly gets out the way

Let's take a closer look at how Pyrefly works differently to other type checkers, particularly in its unobtrusive approach. A key illustration of this difference is Pyrefly's interpretation of strictness.

Pydantic offers flexible validation modes to suit different use cases:

- **Lax (Default) Mode:** In this mode, Pydantic will automatically coerce types where possible. For example, if you pass `”123”`(a string) to a field expecting an int, Pydantic will convert (i.e. coerce) it to `123` (an int). This is convenient for handling loosely-typed input, such as data from APIs or user forms.
- **Strict Mode:** here Pydantic enforces exact types, no coercion is performed. If you pass a string to an integer field, you’ll get a validation error. This is ideal for scenarios where data integrity is paramount.

The handy thing about Pyrefly is that it automatically respects your choices on this matter. It reads your model’s configuration (such as `ConfigDict` or model-level config) to determine which mode to apply. For example, in the following code we are explicitly setting the `age` field using strict mode. Pyrefly inspects this statement directly and adapts its analysis to match your intent (in this case by reporting an error if a string age is passed).

```py
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str
    age: int = Field(strict=True) # strict mode

# Pyrefly will report an error here.
y = User(name="Alice", age="30")
```

Another useful example of this can be seen with the use of extra parameters. Sometimes when writing Pydantic models you want to allow extra unspecified parameters when creating a model, and other times you want to explicitly forbid it (e.g. by using `extra=forbid`). Whether you’re using either approach or a mix of both in your code, Pyrefly can pick it up automatically from how you’ve written your Pydantic model:

```py
# Lax mode: We can pass extra fields to our model by default
class ModelAllow(BaseModel):
x: int

ModelAllow(x=1, y=2) # pyrefly won't report an error here

# Strict mode: we can forbid extra fields
class ModelForbid(BaseModel, extra="forbid"):
x: int

ModelForbid(x=1, y=2) # pyrefly will report an error here
```

Ultimately the value here is that you don’t have to manage external configuration files or plugins to get your type checker to work with Pydantic. This is a different approach compared to say Mypy \- which requires you to install a separate Pydantic plugin and configure it via your `mypy.ini` or `pyproject.toml` file.

## **Getting Started**

There are no special configurations or plugins required to start using Pyrefly with Pydantic\! The steps are as simple as:

1. [Install Pydantic](https://docs.pydantic.dev/latest/install/) (preferably v2).
2. [Install Pyrefly](https://pyrefly.org/en/docs/installation/) (v0.33.0 or later)
3. Write Pydantic models as usual.
4. Run Pyrefly / use it in your IDE

Simple as that\! If you’d like to have a go playing around with a few examples, we created [a demo repo](https://github.com/migeed-z/pyrefly-pydantic-demo) that you can clone and try out.

## **Conclusion**

Pyrefly’s Pydantic support is experimental and evolving, so you may encounter edge cases or features that aren’t fully covered yet. We’d like to strongly encourage you to try out Pydantic support in your projects and **let us know what works, what doesn’t, and what you’d like to see improved**\! Your insights and suggestions are invaluable in helping us refine this feature and prioritize enhancements. You can [open an issue in our GitHub repo](https://github.com/facebook/pyrefly/issues) or [connect with us on discord](https://discord.gg/Cf7mFQtW7W) if you have feedback to share.

Looking ahead, we’re also working on expanding Pyrefly’s capabilities to support additional third-party Python packages that are widely used in the ecosystem. We’re currently focused on adding support for Django and SQLAlchemy, but if there’s a particular library or framework you rely on, please let us know\! Your suggestions and feedback will directly influence our roadmap and help make Pyrefly even better for the Python community.
