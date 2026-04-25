# Tests for `pyrefly init --non-interactive`

## Non-interactive mode creates config in empty directory

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/init_test && \
> $PYREFLY init --non-interactive $TMPDIR/init_test
 INFO New config written to `*/pyrefly.toml` (glob)
* (glob*)
[0]
```

## Non-interactive mode migrates mypy config

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/init_mypy && \
> echo -e "[mypy]\nignore_missing_imports = True" > $TMPDIR/init_mypy/mypy.ini && \
> $PYREFLY init --non-interactive $TMPDIR/init_mypy
 INFO Found an existing type checking configuration - setting up pyrefly ...
* (glob*)
[0]
```

## Non-interactive mode won't overwrite existing pyrefly config

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/init_existing && \
> echo "" > $TMPDIR/init_existing/pyrefly.toml && \
> $PYREFLY init --non-interactive $TMPDIR/init_existing
[1]
```
