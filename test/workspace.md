# Tests for checking the projects of a workspace

## Set up a workspace

The root config excludes every `vendor` directory. `packages/a` has no Pyrefly
config, so the root config governs it. `packages/b` has its own config, which
does not exclude `vendor`. Each file has one error, so the output shows which
files are checked.

```scrut
$ mkdir -p $TMPDIR/workspace/packages/a/vendor $TMPDIR/workspace/packages/b/vendor && \
> cd $TMPDIR/workspace && \
> printf '[tool.pyrefly]\nproject-excludes = ["**/vendor"]\n' > pyproject.toml && \
> printf '[project]\nname = "a"\n' > packages/a/pyproject.toml && \
> printf '[project]\nname = "b"\n\n[tool.pyrefly]\n' > packages/b/pyproject.toml && \
> for d in . packages/a packages/a/vendor packages/b packages/b/vendor; do echo "x: int = ''" > $d/m.py; done
[0]
```

## `--project-excludes` replaces the excludes of the config

The root config excludes `vendor`, but `packages/a/vendor/m.py` is checked.

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --project-excludes=packages/b --python-version 3.13.0 --output-format=min-text
ERROR m.py:1:* (glob)
*packages/a/m.py:1:* (glob)
*packages/a/vendor/m.py:1:* (glob)
[1]
```

## `--extend-project-excludes` adds to the excludes of the config

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --extend-project-excludes=packages/b --python-version 3.13.0 --output-format=min-text
ERROR m.py:1:* (glob)
*packages/a/m.py:1:* (glob)
[1]
```

## `--project-dir` checks the files of the project under a directory

The root config governs `packages/a`, so its `vendor` directory is excluded.

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --project-dir=packages/a --python-version 3.13.0 --output-format=min-text
*packages/a/m.py:1:* (glob)
[1]
```

## `--project-dir` uses the config of a directory that has one

The config of `packages/b` does not exclude `vendor`.

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --project-dir=packages/b --python-version 3.13.0 --output-format=min-text
*packages/b/m.py:1:* (glob)
*packages/b/vendor/m.py:1:* (glob)
[1]
```

## `--project-dir` checks the same files as `pyrefly check`

`packages/b` is below `.`, so the root config governs it, the same as in
`pyrefly check`.

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --project-dir=. --python-version 3.13.0 --output-format=min-text
ERROR m.py:1:* (glob)
*packages/a/m.py:1:* (glob)
*packages/b/m.py:1:* (glob)
[1]
```

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --python-version 3.13.0 --output-format=min-text
ERROR m.py:1:* (glob)
*packages/a/m.py:1:* (glob)
*packages/b/m.py:1:* (glob)
[1]
```

## A file belongs to the deepest `--project-dir` that contains it

`packages/b` is given, so its files use its own config. Errors are sorted by
module name, and `packages/b/m.py` is the module `m` of that config.

```scrut
$ cd $TMPDIR/workspace && $PYREFLY check --project-dir=. --project-dir=packages/b --python-version 3.13.0 --output-format=min-text
ERROR m.py:1:* (glob)
*packages/b/m.py:1:* (glob)
*packages/a/m.py:1:* (glob)
*packages/b/vendor/m.py:1:* (glob)
[1]
```
