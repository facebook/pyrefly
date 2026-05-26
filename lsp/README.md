# Pyrefly VS Code Extension

The Pyrefly extension uses Pyrefly to provide language server features for
Python in VS Code. Please see [pyrefly.org](https://pyrefly.org/) for more
information.

## Features

The Pyrefly extension:

- Adds inline type errors matching the Pyrefly command-line to your editor.
  By default, a project without a Pyrefly configuration uses the
  [`basic`](https://pyrefly.org/en/docs/configuration/#preset-basic) preset
  or auto-migrated settings from a nearby `mypy.ini` / `pyrightconfig.json`.
  Set `python.pyrefly.typeCheckingMode` to choose a different preset.
- Adds language features from Pyrefly's analysis like go-to definition, hover,
  etc. (full list [here](https://github.com/facebook/pyrefly/issues/344)) and
  disables Pylance completely (VSCode's built-in Python extension)

## Customization

By default, Pyrefly should work in the IDE with no configuration necessary. But
to ensure your project is set up properly, see
[configurations](https://pyrefly.org/en/docs/configuration/).

The following configuration options are IDE-specific and exposed as VSCode
settings:

- `python.pyrefly.typeCheckingMode` [enum: auto, off, basic, legacy,
  default, strict; default: auto]: [Preset](https://pyrefly.org/en/docs/configuration/#preset)
  to use for files not covered by a `pyrefly.toml`. The default `auto`
  migrates a nearby mypy/pyright config when present, otherwise uses
  `basic`.
- `python.pyrefly.disableTypeErrors` [boolean: false]: If true, Pyrefly
  will not provide diagnostics for files in this workspace.
- `python.pyrefly.displayTypeErrors` (deprecated): replaced by
  `python.pyrefly.typeCheckingMode` and `python.pyrefly.disableTypeErrors`.
  Still accepted for backwards compatibility.
- `python.pyrefly.disableLanguageServices` [boolean: false]: by default, Pyrefly
  will provide both type errors and other language features like go-to
  definition, intellisense, hover, etc. Enable this option to keep type errors
  from Pyrefly unchanged but use VSCode's Python extension for everything else.
- `python.pyrefly.disabledLanguageServices` [json: {}]: a config to disable
  certain lsp methods from pyrefly. For example, if you want go-to definition
  but not find-references.
- `pyrefly.lspPath` [string: '']: if your platform is not supported, you can
  build pyrefly from source and specify the binary here.
- `python.pyrefly.configPath` [string: '']: path to a `pyrefly.toml` or
  `pyproject.toml` configuration file. When set, the LSP will use this config
  for all files in your workspace instead of the default Pyrefly config-finding logic
  wherever possible.
- `python.analysis.showHoverGoToLinks` [boolean: true]: Controls whether hover
  tooltips include "Go to definition" and "Go to type definition" navigation
  links. Set to `false` for cleaner tooltips with only type information.
