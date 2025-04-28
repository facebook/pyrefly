# Pyrefly WASM Build

Pyrefly's WASM build is used in browser-based applications like the
[Pyrefly sandbox](https://pyrefly.org/try/).

## Prerequisites

- Ensure you have clang installed. For mac, `brew install llvm`
  ([source](https://github.com/briansmith/ring/issues/1824)). For CentOS,
  `sudo dnf install clang`

## Building Pyrefly for WASM

Run `./build.sh`

## Common issues with the build

Certain features that are allowed in the other builds are unavailable in the
wasm build. You might errors like `could not find \`<>\` in the crate
root`. This can be because we've excluded that package from the wasm build with `#[cfg(not(target_arch
= "wasm32"))]`in the`lib.rs`/`mod.rs` file.
