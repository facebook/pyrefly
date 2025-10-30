# Pyrefly VS Code Extension

The Pyrefly extension uses Pyrefly to provide language server features for
Python in VS Code. Please see [pyrefly.org](https://pyrefly.org/) for more
information.

## Features

The Pyrefly extension:

- Adds inline type errors matching the Pyrefly command-line to your editor
  (note: only shown when a pyrefly configuration is present or
  `python.pyrefly.disableTypeErrors` is `force-on`)
- Adds language features from Pyrefly's analysis like go-to definition, hover,
  etc. (full list [here](https://github.com/facebook/pyrefly/issues/344)) and
  disables Pylance completely (VSCode's built-in Python extension)

## Customization

By default, Pyrefly should work in the IDE with no configuration necessary. But
to ensure your project is set up properly, see
[configurations](https://pyrefly.org/en/docs/configuration/).

The following configuration options are IDE-specific and exposed as VSCode
settings:

- `python.pyrefly.disableTypeErrors` [enum: default, force-on, force-off]: by
  default, Pyrefly will only provide type errors in your project if a
  `pyrefly.toml` is present. Modify this setting to override the IDE
  diagnostics.
- `python.pyrefly.disableLanguageServices` [boolean: false]: by default, Pyrefly
  will provide both type errors and other language features like go-to
  definition, intellisense, hover, etc. Enable this option to keep type errors
  from Pyrefly unchanged but use VSCode's Python extension for everything else.
 - `python.pyrefly.analysis.disabledLanguageServices` [boolean: false]: an
   analysis-level option that disables language-service features produced by
   Pyrefly's analysis (for example: go-to definition, hover, and other
   navigation/intelligence features) while leaving Pyrefly diagnostics
   (type errors) intact. Use this when you want Pyrefly's type checking but
   prefer the editor's language features or another provider for those
   services.
- `pyrefly.lspPath` [string: '']: if your platform is not supported, you can
  build pyrefly from source and specify the binary here.
