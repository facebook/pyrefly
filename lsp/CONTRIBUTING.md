# Contributing to Pyrefly LSP

Please see CONTRIBUTING.md in the root of the repository for more details on
contributing. This file shows how to develop the extension locally.

## Pre-requisites

- You need to have npm v7+ installed. Afterwards, run `npm install` in this
  folder and in `client`.

## Debugging (Recommended Development Workflow)

- Follow steps in Pre-requisites section.
- Ensure `cargo` is installed
- Open VS Code on the toplevel pyrefly folder.
- Switch to the Debug viewlet.
- Select `Run Installed Extension (pyrefly)` from the drop down.
- Run the launch config.
- By default, stderr of the language server will appear in the output pane of
  VSCode under "Pyrefly language server".
- Add `"pyrefly.trace.server": "verbose"` to the VSCode config. Then all the LSP
  JSON requests and responses will be logged together with stderr of language
  server in the output pane.

## Installing

- Follow steps in Pre-requisites section.
- Build the Pyrefly binary with
  `buck2 build pyrefly @fbcode//mode/opt --show-output` or `cargo build` and
  either:

1. Place binary at `lsp/bin/pyrefly(.exe)`
2. Add the `pyrefly.lspPath` configuration key to point at it after extension
   startup.

- Run `npm install vsce`
- Run `npm exec vsce package`
- In VS Code, go to Extensions, click on the "..." button in the Extensions bar,
  select "Install from VSIX" and then select the `pyrefly-1.0.0.vsix` file that
  was produced.

## Building for all Platforms

- Run
  [build_extension](https://github.com/facebook/pyrefly/actions/workflows/build_extension.yml)
  github workflow on your branch.

## Updating

Every few months security advisories will arrive about pinned versions of
packages.

- `npm audit` to see which packages have security updates.
- `npm audit fix` to fix those issues.
- Try `npm audit`, if it still has issues run `npm update`.
- `npm exec vsce package` to confirm everything still works.
