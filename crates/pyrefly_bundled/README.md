# pyrefly_bundled

This crate bundles [typeshed](https://github.com/python/typeshed) type stubs
into the Pyrefly binary, providing type information for Python's standard
library and third-party packages without requiring external files at runtime.

## Overview

`pyrefly_bundled` embeds a compressed archive of typeshed `.pyi` (Python
Interface) files directly into the Pyrefly executable. This allows Pyrefly to
access type stubs for the Python standard library and third-party packages
without needing to download or locate them on the file system at runtime.

## How It Works

### Build Time

The build script (`build.rs`) runs during compilation and:

1. Locates the typeshed directory (from `TYPESHED_ROOT` env var or
   `third_party/typeshed`)
2. Creates a tar archive of the typeshed files
3. Compresses the archive using zstd
4. Saves it to the build output directory as `typeshed.tar.zst`
5. The archive is embedded into the binary using `include_bytes!` macro

## Updating Typeshed

The `update.py` script is used to fetch and update the bundled typeshed version:

```bash
uv run ./update.py
```

This script:

1. Downloads the latest typeshed tarball from GitHub (main branch)
2. Filters out unnecessary files (tests, Python 2 stubs, config files)
3. Keeps only `.py` and `.pyi` files from `stdlib/` and `stubs/` directories
4. Writes the trimmed typeshed to `third_party/typeshed/`
5. Creates metadata file with URL, SHA256, and timestamp

You can also specify a custom URL for the download using the `--url` flag. This
can be helpful, for example, when you changed the `update.py` script and want to
download & process the exact same typeshed revision as before. URL for the
current typeshed revision can be found in the metadata file
`third_party/typeshed/typeshed_metadata.json`.

## License

This code is licensed under the MIT license. See the LICENSE file in the root
directory.
