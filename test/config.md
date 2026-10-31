# Tests for pyrefly configuration files

## Error on a non-existent search-path/site-packages-path

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/test && echo "" > $TMPDIR/test/empty.py && \
> echo -e "project_includes = [\"$TMPDIR/test/empty.py\"]\nsite_package_path = [\"$TMPDIR/test/abcd\"]\nsearch_path = [\"$TMPDIR/test/efgh\"]" > $TMPDIR/test/pyrefly.toml && \
> $PYREFLY check -c $TMPDIR/test/pyrefly.toml --python-version 3.13.0
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 WARN */pyrefly.toml: Invalid site-packages-path: */abcd` does not exist (glob)
 WARN */pyrefly.toml: Invalid search-path: */efgh` does not exist (glob)
 INFO * errors* (glob)
[0]
```

## Dump config

```scrut
$ touch $TMPDIR/foo.py && mkdir $TMPDIR/bar && touch $TMPDIR/bar/baz.py && touch $TMPDIR/bar/qux.py && mkdir $TMPDIR/spp && touch $TMPDIR/spp/mylib.py \
> && $PYREFLY dump-config --site-packages-path $TMPDIR/spp/ $TMPDIR/foo.py $TMPDIR/bar/*.py
Default configuration
  Using interpreter: * (glob)
  Covered files:
    */bar/baz.py (glob)
    */bar/qux.py (glob)
  Resolving imports from:
    Fallback search path (guessed from importing file with heuristics): * (glob)
    Site packages path from user: * (glob)
    Site packages path queried from interpreter: * (glob)
Default configuration
  Using interpreter: * (glob)
  Covered files:
    */foo.py (glob)
  Resolving imports from:
    Fallback search path (guessed from importing file with heuristics): * (glob)
    Site packages path from user: * (glob)
    Site packages path queried from interpreter: * (glob)
[0]
```

## Dump config with --full

<!-- Default settings are matched with globs so that new config fields do not
require updating this test. -->

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/full/src && touch $TMPDIR/full/src/a.py && \
> printf 'project-includes = ["src/**/*.py"]\npython-version = "3.12.0"\nuse-ignore-files = false\nnot-a-real-key = 1\n\n[errors]\nmissing-import = "warn"\nbad-assignment = "error"\n\n[[sub-config]]\nmatches = "src/**"\ncheck-unannotated-defs = false\n\n[sub-config.errors]\nmissing-import = "ignore"\nbad-assignment = "warn"\n' > $TMPDIR/full/pyrefly.toml && \
> $PYREFLY dump-config --full --python-version 3.11.0 $TMPDIR/full/src/a.py
Configuration at `*/full/pyrefly.toml` (glob)
  Using interpreter: * (glob)
  Configuration settings (TOML):
* (glob+)
    not-a-real-key = 1
* (glob+)
    python-version = "3.11.0"
* (glob+)
    use-ignore-files = false
* (glob+)
    [errors]
    bad-assignment = "error"
    missing-import = "warn"
* (glob+)
    [[sub-config]]
    check-unannotated-defs = false
    matches = "*/full/src/**/*" (glob)
* (glob+)
    [sub-config.errors]
    bad-assignment = "warn"
    missing-import = "ignore"
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
> printf 'project-includes = ["main.py"]\nsite-packages-path = ["site_packages"]\nskip-interpreter-query = true\n' > $TMPDIR/untyped_import/pyrefly.toml && \
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
> printf 'project-includes = ["main.py"]\nsite-packages-path = ["site_packages"]\nskip-interpreter-query = true\n' > $TMPDIR/untyped_import/pyrefly.toml && \
> $PYREFLY check -c $TMPDIR/untyped_import/pyrefly.toml --output-format=min-text --replace-untyped-imports-with-any pandas
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 0 errors
[0]
```

## `required-version` that the running Pyrefly does not satisfy

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/required_version && \
> echo 'required-version = "<0"' > $TMPDIR/required_version/pyrefly.toml && \
> echo "x = 1" > $TMPDIR/required_version/a.py && \
> echo "x: str = 1" > $TMPDIR/required_version/b.py && \
> cd $TMPDIR/required_version && $PYREFLY check --output-format=min-text
 INFO Checking project configured at `*/pyrefly.toml` (glob)
ERROR */pyrefly.toml: Pyrefly * does not satisfy `required-version = "<0"` (glob)
Fatal configuration error
[4]
```

With several files to check, the check also stops before checking any file.

```scrut {output_stream: stderr}
$ cd $TMPDIR/required_version && $PYREFLY check a.py b.py --output-format=min-text
ERROR */pyrefly.toml: Pyrefly * does not satisfy `required-version = "<0"` (glob)
Fatal configuration error
[4]
```
