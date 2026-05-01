# Pyrefly: A fast type checker and language server for Python with powerful IDE features

[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![PyPI](https://img.shields.io/pypi/dm/pyrefly?color=blue&label=pypi)](https://pypi.python.org/pypi/pyrefly)
[![Visual Studio Marketplace](https://img.shields.io/visual-studio-marketplace/d/meta.pyrefly?color=blue&label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly)
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

Currently under active development with known issues. Please open an issue if
you find bugs.

### Getting Started

- Try out pyrefly in your browser: [Sandbox](https://pyrefly.org/sandbox/)
- Get the command-line tool: `pip install pyrefly`
- Get the IDE Extension: [IDE installation page](https://pyrefly.org/en/docs/IDE/)

### Key Features:

- Type Inference: Pyrefly infers types in most locations, apart from function
  parameters. It can infer types of variables and return types.
- Flow Types: Pyrefly can understand your program's control flow to refine
  static types.
- Incrementality: Pyrefly aims for large-scale incrementality at the module
  level, with optimized checking and parallelism.

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
