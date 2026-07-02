# Tests for specifying options as command line arguments

## We can manually override typing_extensions

<!-- See https://typing.python.org/en/latest/spec/distributing.html#import-resolution-ordering:
     typing_extensions.py on the search path takes precedence over typeshed
-->

```scrut {output_stream: stdout}
$ mkdir $TMPDIR/typing_extensions_project && \
> echo "x: int = 42" > $TMPDIR/typing_extensions_project/typing_extensions.py && \
> echo "from typing_extensions import x; y: int = x" > $TMPDIR/typing_extensions_project/foo.py && \
> $PYREFLY check $TMPDIR/typing_extensions_project/foo.py --search-path $TMPDIR/typing_extensions_project
[0]
```

## We don't accidentally override typing_extensions with an installed package

<!-- See https://typing.python.org/en/latest/spec/distributing.html#import-resolution-ordering:
     typeshed takes precedence over installed packages
-->

```scrut {output_stream.stdout}
$ mkdir $TMPDIR/site_package_path && \
> echo "x: int = 42" > $TMPDIR/site_package_path/typing_extensions.py && \
> echo "from typing import TypedDict; from typing_extensions import NotRequired; class C(TypedDict): x: NotRequired[int]" > $TMPDIR/site_package_path/lib.py && \
> echo "from lib import C; C()" > $TMPDIR/foo.py && \
> $PYREFLY check $TMPDIR/foo.py --site-package-path $TMPDIR/site_package_path
[0]
```

## Error on missing source

```scrut {output_stream.stdout}
$ mkdir $TMPDIR/site_package_missing_source && \
> mkdir $TMPDIR/site_package_missing_source/pkg-stubs && \
> echo "class X: ..." > $TMPDIR/site_package_missing_source/pkg-stubs/__init__.py && \
> echo "from pkg import X" > $TMPDIR/foo.py && \
> $PYREFLY check $TMPDIR/foo.py --error=missing-source --site-package-path $TMPDIR/site_package_missing_source --output-format=min-text
ERROR * Found stubs for `pkg`, but no source* (glob)
[1]
```

## We can disable `missing-source` on the command line

```scrut {output_stream.stdout}
$ mkdir $TMPDIR/error_missing_source && \
> echo -e '[errors]\nmissing-source="error"' > $TMPDIR/error_missing_source/pyrefly.toml && \
> echo "from pkg import X" > $TMPDIR/error_missing_source/foo.py && \
> $PYREFLY check $TMPDIR/error_missing_source/foo.py --ignore=missing-source --site-package-path $TMPDIR/site_package_missing_source --output-format=min-text
[0]
```

## We can enable an error that is default-off

```scrut {output_stream.stdout}
$ echo -e '[errors]\nmissing-source=true' > $TMPDIR/error_missing_source/pyrefly.toml && \
> $PYREFLY check $TMPDIR/error_missing_source/foo.py --site-package-path $TMPDIR/site_package_missing_source --output-format=min-text
ERROR * [missing-source] (glob)
[1]
```

## Regression test: we should still be able to find submodules when stubs are missing

<!-- The missing-module-attribute ignore is for compatibility with Python versions that do not have
     `typing.reveal_type` available and can be removed once we drop support for Python version < 3.11
-->

```scrut {output_stream.stdout}
$ touch $TMPDIR/pyrefly.toml && \
> mkdir $TMPDIR/site_package_missing_stubs && \
> mkdir $TMPDIR/site_package_missing_stubs/django && \
> touch $TMPDIR/site_package_missing_stubs/django/__init__.py && \
> mkdir $TMPDIR/site_package_missing_stubs/django/forms && \
> touch $TMPDIR/site_package_missing_stubs/django/forms/__init__.py && \
> echo "from django import forms; from typing import reveal_type; reveal_type(forms)" > $TMPDIR/foo.py && \
> $PYREFLY check $TMPDIR/foo.py --error untyped-import --ignore missing-module-attribute --site-package-path $TMPDIR/site_package_missing_stubs --output-format=min-text
ERROR * Cannot find type stubs for module `django` * (glob)
 INFO * revealed type: Module[django.forms] * (glob)
[1]
```

## The `--preset` flag sets the preset and suppresses the upsell

```scrut {output_stream: stderr}
$ PRESET_DIR=$(mktemp -d -p /tmp preset.XXXXXX) && \
> echo "x = unknown_name" > $PRESET_DIR/foo.py && \
> $PYREFLY check $PRESET_DIR/foo.py --preset off; rm -rf $PRESET_DIR
 INFO 0 errors
[0]
```

## Error mappings in the config file override `--preset`

```scrut {output_stream: stdout}
$ echo -e '[errors]\nimplicit-any = "ignore"' > $TMPDIR/pyrefly.toml && \
> echo "x = []" > $TMPDIR/foo.py && \
> $PYREFLY check $TMPDIR/foo.py --preset strict --output-format=min-text
[0]
```

## Scalar fields in the config file override `--preset`

```scrut {output_stream: stdout}
$ echo "check-unannotated-defs = true" > $TMPDIR/pyrefly.toml && \
> echo -e "def f():\n    x = unknown_name" > $TMPDIR/foo.py && \
> $PYREFLY check $TMPDIR/foo.py --preset basic --output-format=min-text
ERROR * [unknown-name] (glob)
[1]
```

## `--preset` in a directory with mypy.ini should not inherit migrated values

```scrut {output_stream: stdout}
$ MYPY_DIR=$(mktemp -d -p /tmp mypy.XXXXXX) && \
> echo -e "[mypy]\ncheck_untyped_defs = False" > $MYPY_DIR/mypy.ini && \
> echo -e "def f():\n    x: int = 'hello'" > $MYPY_DIR/foo.py && \
> $PYREFLY check $MYPY_DIR/foo.py --preset strict --output-format=min-text; rc=$?; rm -rf $MYPY_DIR; exit $rc
ERROR * [bad-assignment] (glob)
[1]
```

## CLI error flags override `--preset`

```scrut {output_stream: stdout}
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: int = 'hello'" > $TMPDIR/foo.py && \
> $PYREFLY check $TMPDIR/foo.py --preset off --error bad-assignment --output-format=min-text
ERROR * [bad-assignment] (glob)
[1]
```
