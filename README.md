# Pyrefly: A fast type checker and language server for Python with powerful IDE features

[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![PyPI](https://img.shields.io/pypi/v/pyrefly?color=blue&label=pypi)](https://pypi.python.org/pypi/pyrefly)
[![VS Code](https://img.shields.io/badge/VS%20Code-Marketplace-blue)](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly)
[![Open VSX](https://img.shields.io/open-vsx/dt/meta/pyrefly?color=blue&label=Open%20VSX)](https://open-vsx.org/extension/meta/pyrefly)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/Cf7mFQtW7W)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pyrefly is a type checker and language server for Python, which provides
lightning-fast type checking along with IDE features such as code navigation,
semantic highlighting, and code completion. It is available as a
[command-line tool](https://pyrefly.org/en/docs/installation/) and an extension
for popular IDEs and editors such as
[VSCode](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly),
[Neovim](https://pyrefly.org/en/docs/IDE/#neovim),
[Zed](https://zed.dev/extensions/pyrefly), and
[more](https://pyrefly.org/en/docs/IDE/).

See the [Pyrefly website](https://pyrefly.org) for full documentation and how to
add Pyrefly to your editor of choice.

Pyrefly is currently in [beta](https://github.com/facebook/pyrefly/releases/tag/0.42.0).

### Key Features

- **Fast.** Pyrefly checks over 1.85 million lines of code per second, type checking projects like PyTorch 15x faster than Mypy and Pyright. In the IDE, rechecks typically complete in under 10 milliseconds after saving a file.
- **Production-proven at scale.** Pyrefly is the default type checker for Instagram's 20-million-line Python codebase at Meta, and has been adopted by large open source projects including PyTorch and JAX.
- **Full-featured language server.** Code navigation, autocomplete, hover information, inlay hints, semantic highlighting, and more, with consistent results across the CLI and your editor of choice.
- **Understands real-world Python.** Built-in support for frameworks like [Pydantic](https://pyrefly.org/en/docs/pydantic/) and [Django](https://pyrefly.org/en/docs/django/), with model validation, field types, and autocomplete that work out of the box.
- **Adoption-ready.** Migrate from Mypy or Pyright with `pyrefly init`, silence existing errors with `pyrefly suppress`, and generate type annotations with `pyrefly infer`. Start with one file and expand at your own pace.

### Getting Started

- Try out pyrefly in your browser: [Sandbox](https://pyrefly.org/sandbox/)
- Get the command-line tool: `pip install pyrefly`
- Get the IDE Extension: [IDE installation page](https://pyrefly.org/en/docs/IDE/)

### Version Policy

Pyrefly releases new minor versions (`0.x.0`) weekly, with patch fixes (`0.x.y`) in between as needed.
Pyrefly does *not* follow strict semantic versioning: any version may introduce new type errors and
other breaking changes. The [`pyrefly suppress`](https://pyrefly.org/en/docs/error-suppressions/)
command can be used to easily silence errors when upgrading to a new version.

## Getting Involved

If you have questions or would like to report a bug, please
[create an issue](https://github.com/facebook/pyrefly/issues).

See our
[contributing guide](https://github.com/facebook/pyrefly/blob/main/CONTRIBUTING.md)
and
[architecture overview](https://github.com/facebook/pyrefly/blob/main/ARCHITECTURE.md)
for information on how to contribute to Pyrefly.

Join our [Discord](https://discord.com/invite/Cf7mFQtW7W) to chat about Pyrefly
and types. This is also where we hold biweekly office hours.
