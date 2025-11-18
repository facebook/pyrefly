# Pyrefly VS Code Extension

The Pyrefly extension uses Pyrefly to provide language server features for
Python in VS Code. Please see [pyrefly.org](https://pyrefly.org/) for more
information.

## Features

The Pyrefly extension:

- Adds inline type errors matching the Pyrefly command-line to your editor
  (note: only shown when a pyrefly configuration is present or
  `python.pyrefly.displayTypeErrors` is `force-on`)
- Adds language features from Pyrefly's analysis like go-to definition, hover,
  etc. (full list [here](https://github.com/facebook/pyrefly/issues/344)) and
  disables Pylance completely (VSCode's built-in Python extension)

## Customization

By default, Pyrefly should work in the IDE with no configuration necessary. But
to ensure your project is set up properly, see
[configurations](https://pyrefly.org/en/docs/configuration/).

The following configuration options are IDE-specific and exposed as VSCode
settings:

- `python.pyrefly.displayTypeErrors` [enum: default, force-on, force-off]: If
  `'default'`, Pyrefly will only provide type check squiggles in the IDE if a
  `pyrefly.toml` is present. If `'force-off'`, Pyrefly will never provide type
  check squiggles in the IDE. If `'force-on'`, Pyrefly will always provide type
  check squiggles in the IDE.
- `python.pyrefly.disableLanguageServices` [boolean: false]: by default, Pyrefly
  will provide both type errors and other language features like go-to
  definition, intellisense, hover, etc. Enable this option to keep type errors
  from Pyrefly unchanged but use VSCode's Python extension for everything else.
- `python.pyrefly.disabledLanguageServices` [json: {}]: a config to disable
  certain lsp methods from pyrefly. For example, if you want go-to definition
  but not find-references.
- `pyrefly.lspPath` [string: '']: if your platform is not supported, you can
  build pyrefly from source and specify the binary here.
- `python.analysis.showHoverGoToLinks` [boolean: true]: Controls whether hover
  tooltips include "Go to definition" and "Go to type definition" navigation
  links. Set to `false` for cleaner tooltips with only type information.
