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
- Adds a `Pyrefly: Infer Types for Current File` command that writes inferred
  type annotations to the active Python file.

## Customization

By default, Pyrefly should work in the IDE with no configuration necessary. But
to ensure your project is set up properly, see
[configurations](https://pyrefly.org/en/docs/configuration/).

The following configuration options are IDE-specific and exposed as VSCode
settings:

- `python.pyrefly.typeCheckingMode` [enum: auto, off, basic, legacy,
  default, strict, all; default: auto]: [Preset](https://pyrefly.org/en/docs/configuration/#preset)
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
- `python.analysis.autoImportCompletions` [boolean: true]: Controls whether
  completions include symbols that are not yet imported. When enabled, accepting
  such a completion also inserts the required import statement. Set to `false` to
  only complete symbols that are already in scope.

## Development

### Build the browser extension

Install Rust, Node.js, and wasm-pack 0.15.0. From the repository root, run:

```bash
bash pyrefly_wasm/build.sh --no-opt
cd lsp
npm ci
npm run compile
code .
```

Select the `Run Web Extension` launch configuration. The argument
`--extensionDevelopmentKind=web` selects the browser extension host.
Use **Developer: Show Running Extensions** to check the host.

The build consumes the generated JS and WASM pair in `pyrefly_wasm/target`.
It checks their hashes before bundling, including during watch builds.
Missing or mismatched inputs cause a build failure.
The optional optimized website bundle requires `wasm-opt`. The browser extension
uses the matching unoptimized binary.

### Browser support boundaries

Each workspace has a separate WASM state. The worker retains original URIs.
Untitled documents and files outside a workspace have separate, single-file states.
Notebook cells are excluded. Reload the window after adding or removing workspace folders.

The browser cannot use the native filesystem or start an interpreter.
Rust reports structured module requests before diagnostic filtering.
The extension reads candidate paths through `vscode.workspace.fs` in the requesting
workspace. It does not scan folders or search other workspace roots.
Python resolves relative module names in Rust. Package files precede module files,
and stub files precede source files. Loaded files, absent files, and reads in progress
are cached. File events invalidate the caches. Document edits share a 40 ms delay
and preserve the warm Rust state when the module set does not change.

Workspace imports use paths relative to the workspace root. Interpreter packages,
custom search paths, and nested project configurations are outside this initial
browser implementation. Each session supports at most 1000 files, 1 MiB per file,
and 10 MiB in total. The worker reports an error if a limit is exceeded.

The Python-version precedence is:

1. `python-version` in `pyrefly.toml`.
2. `python-version` in `[tool.pyrefly]` in `pyproject.toml`, if `pyrefly.toml` is absent.
3. The resource setting `python.pyrefly.pythonVersion`.
4. Python 3.12.

TOML updates take effect during rechecking. Reload the window after changing the
editor setting. Invalid versions produce configuration errors.
Browser support does not change the desktop analysis settings or its workspace-host requirement.

### Test and package

Run these commands from `lsp` after building WASM:

```bash
npm run test:web
npx @vscode/vsce package --no-dependencies --target web --out pyrefly-web.vsix
node tests/check-vsix.cjs pyrefly-web.vsix
unzip -q pyrefly-web.vsix -d web-test
npx playwright install chromium
npm run test:web:host
```

The worker tests use the actual WASM binary. The build tests check normal and watch
builds after regenerated bindings. The package check rejects absent or mismatched
assets. The browser-host test loads the extracted VSIX, checks activation and
diagnostics, and applies a converted auto-import completion.
The CI workflow builds WASM once and supplies that pair to native and web packaging jobs.

For the final check on vscode.dev, serve the extracted `web-test/extension`
directory over HTTPS with CORS enabled. Use **Developer: Install Extension From
Location...** to install from that URL, if the command is available in the deployed
VS Code version. Open a Python file with a type error. Check the Pyrefly diagnostic
and accept a `reveal_type` completion.
Record this result separately from local tests. A local browser-host test does
not establish that vscode.dev activation works.
