# Tests for pyrefly configuration files

## Error on a non-existent search-path/site-package-path

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/test && echo "" > $TMPDIR/test/empty.py && \
> echo -e "project_includes = [\"$TMPDIR/test/empty.py\"]\nsite_package_path = [\"$TMPDIR/test/abcd\"]\nsearch_path = [\"$TMPDIR/test/efgh\"]" > $TMPDIR/test/pyrefly.toml && \
> $PYREFLY check -c $TMPDIR/test/pyrefly.toml --python-version 3.13.0
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 WARN */pyrefly.toml: Invalid site-package-path: */abcd` does not exist (glob)
 WARN */pyrefly.toml: Invalid search-path: */efgh` does not exist (glob)
 INFO * errors* (glob)
[0]
```

## Dump config

```scrut
$ touch $TMPDIR/foo.py && mkdir $TMPDIR/bar && touch $TMPDIR/bar/baz.py && touch $TMPDIR/bar/qux.py && mkdir $TMPDIR/spp && touch $TMPDIR/spp/mylib.py \
> && $PYREFLY dump-config --site-package-path $TMPDIR/spp/ $TMPDIR/foo.py $TMPDIR/bar/*.py
Default configuration
  Using interpreter: * (glob)
  Covered files:
    */bar/baz.py (glob)
    */bar/qux.py (glob)
  Resolving imports from:
    Fallback search path (guessed from importing file with heuristics): * (glob)
    Site package path from user: * (glob)
    Site package path queried from interpreter: * (glob)
Default configuration
  Using interpreter: * (glob)
  Covered files:
    */foo.py (glob)
  Resolving imports from:
    Fallback search path (guessed from importing file with heuristics): * (glob)
    Site package path from user: * (glob)
    Site package path queried from interpreter: * (glob)
[0]
```

## Dump config with --full

<!-- The `Type checking:` settings are matched with a glob so that adding a new
`ConfigBase` setting does not break this test. -->

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/full/src && touch $TMPDIR/full/src/a.py && \
> printf 'project-includes = ["src/**/*.py"]\npython-version = "3.12.0"\nuse-ignore-files = false\nnot-a-real-key = 1\n\n[errors]\nmissing-import = "warn"\nbad-assignment = "error"\n\n[[sub-config]]\nmatches = "src/**"\ncheck-unannotated-defs = false\n' > $TMPDIR/full/pyrefly.toml && \
> $PYREFLY dump-config --full $TMPDIR/full/src/a.py
Configuration at `*/full/pyrefly.toml` (glob)
  Using interpreter: * (glob)
  Required version: (unset)
  Project files:
    Includes: [*/full/src/**/*.py] (glob)
    Excludes: * (glob)
    Excludes heuristics disabled: false
    Respecting ignore files: false
    Extra file extensions: []
  Python environment:
    Version: 3.12.0
    Platform: * (glob)
  Import settings:
    Search path heuristics disabled: false
    Fallback search path enabled: false
    Typeshed path: (unset)
  Reporting:
    Output format: (unset)
    Minimum severity: (unset)
    Baseline: (unset)
    Baseline error level: (unset)
    Skip LSP config indexing: false
  Type checking:
* (glob+)
  Error severity overrides:
    bad-assignment: error
    missing-import: warn
  Path-scoped overrides:
    Matching `*/full/src/**/*`: (glob)
      check-unannotated-defs: false
  Coverage overrides:
    Includes: (unset)
    Excludes: (unset)
  Build system:
    (none)
  Unrecognized config keys: [not-a-real-key]
  Covered files:
    */full/src/a.py (glob)
  Resolving imports from:
* (glob+)
[0]
```

## Specify both files and config

```scrut {output_stream: stderr}
$ echo "x: str = 0" > $TMPDIR/oops.py && echo "errors = { bad-assignment = false }" > $TMPDIR/pyrefly.toml && $PYREFLY check -c $TMPDIR/pyrefly.toml $TMPDIR/oops.py && rm $TMPDIR/pyrefly.toml
 INFO 0 errors
[0]
```

## Replaced imports resolve exported and missing names to Any

`replace-imports-with-any` discards the module's type information even when its
source exists. Both names the source exports and names it does not export are
therefore `Any`.

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/replace_with_any && \
> printf 'replace-imports-with-any = ["module"]\n' > $TMPDIR/replace_with_any/pyrefly.toml && \
> printf 'class Exported: ...\n' > $TMPDIR/replace_with_any/module.py && \
> printf 'from typing import Any, assert_type\nfrom module import Exported, Missing\n\nassert_type(Exported, Any)\nassert_type(Missing, Any)\n' > $TMPDIR/replace_with_any/main.py && \
> $PYREFLY check -c $TMPDIR/replace_with_any/pyrefly.toml --output-format=min-text $TMPDIR/replace_with_any/main.py
 INFO 0 errors
[0]
```

## Replaced imports remain dynamic when used as TypeVar bounds

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/replace_bound && \
> printf 'replace-imports-with-any = ["module.*"]\n' > $TMPDIR/replace_bound/pyrefly.toml && \
> printf 'class Foo: ...\n' > $TMPDIR/replace_bound/module.py && \
> printf 'from typing import TypeVar\nfrom module import Foo\n\nT = TypeVar("T", bound=Foo)\n\ndef f(arg: T) -> T:\n    arg.method()\n    return arg\n' > $TMPDIR/replace_bound/main.py && \
> $PYREFLY check -c $TMPDIR/replace_bound/pyrefly.toml --output-format=min-text $TMPDIR/replace_bound/main.py
 INFO 0 errors
[0]
```

## Untyped third-party imports are followed by default

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/untyped_import/site_packages/untyped_package && \
> printf '' > $TMPDIR/untyped_import/site_packages/untyped_package/__init__.py && \
> printf 'from untyped_package import missing\nmissing()\n' > $TMPDIR/untyped_import/main.py && \
> printf 'project-includes = ["main.py"]\nsite-package-path = ["site_packages"]\nskip-interpreter-query = true\n' > $TMPDIR/untyped_import/pyrefly.toml && \
> $PYREFLY check -c $TMPDIR/untyped_import/pyrefly.toml --output-format=min-text
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 1 error
[1]
```

## Replace untyped third-party imports with Any

Same project, with the option turned on: `untyped_package` becomes `typing.Any`,
so importing a name it does not define is no longer an error.

```scrut {output_stream: stderr}
$ $PYREFLY check -c $TMPDIR/untyped_import/pyrefly.toml --replace-untyped-imports-with-any untyped_package --output-format=min-text
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 0 errors
[0]
```

## `--replace-untyped-imports-with-any` ignores bundled stubs

Pyrefly's bundled `pandas` stubs should not prevent `pandas` from being detected as untyped.

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/untyped_import/site_packages/pandas && \
> printf '' > $TMPDIR/untyped_import/site_packages/pandas/__init__.py && \
> printf 'from pandas import missing\nmissing()\n' > $TMPDIR/untyped_import/main.py && \
> printf 'project-includes = ["main.py"]\nsite-package-path = ["site_packages"]\nskip-interpreter-query = true\n' > $TMPDIR/untyped_import/pyrefly.toml && \
> $PYREFLY check -c $TMPDIR/untyped_import/pyrefly.toml --output-format=min-text --replace-untyped-imports-with-any pandas
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 0 errors
[0]
```
