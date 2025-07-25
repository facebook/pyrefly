# Tests about finding files

## We can find adjacent modules for unknown paths

```scrut
$ mkdir -p $TMPDIR/inner/foo && mkdir -p $TMPDIR/inner/baz && \
> echo "x: str = 12" > $TMPDIR/inner/foo/bar.py && \
> echo "import foo.bar; y: int = foo.bar.x" > $TMPDIR/inner/baz/quux.py && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/inner/baz/quux.py --output-format=min-text
*/inner/baz/quux.py:1:26-35: `str` is not assignable to `int` [bad-assignment] (glob)
[1]
```

## We can do our own globbing

```scrut
$ echo "x: str = 12" > $TMPDIR/glob1.py && \
> echo "x: str = 12" > $TMPDIR/glob2.py && \
> $PYREFLY check --python-version 3.13.0 "$TMPDIR/glob*.py" --output-format=min-text
*/glob*.py:1:* (glob)
*/glob*.py:1:* (glob)
[1]
```

## --search-path takes precedence over default typeshed

```scrut
$ PYREFLY_STDLIB_SEARCH_PATH=$TYPESHED_ROOT/typeshed/stdlib $PYREFLY check --python-version 3.13.0 $TYPESHED_ROOT/typeshed/stdlib/builtins.pyi --search-path $TYPESHED_ROOT/typeshed/stdlib --output-format=min-text 2>&1 | grep -v "overrides"
 INFO * errors* (glob)
[0]
```

## Src layout

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/src_project/src/foo && echo "x: int = 0" > $TMPDIR/src_project/src/foo/bar.py && echo "from foo.bar import x" > $TMPDIR/src_project/src/foo/baz.py && cd $TMPDIR/src_project && $PYREFLY check
 INFO Checking current directory with default configuration
 INFO 0 errors
[0]
```

## Relative --project-excludes

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/foo && touch $TMPDIR/foo/bar.py && echo "x: str = 0" > $TMPDIR/foo/problem.py && cd $TMPDIR/foo && $PYREFLY check "*.py" --project-excludes=problem.py
 INFO 0 errors
[0]
```

## Removing -stubs prefix (this import should work)

```scrut
$ mkdir $TMPDIR/test-stubs && touch $TMPDIR/test-stubs/utils.pyi && touch $TMPDIR/test-stubs/main.pyi && \
> echo "x: int = 1" > $TMPDIR/test-stubs/utils.pyi && \
> echo "from test.utils import x" > $TMPDIR/test-stubs/main.pyi && \
> $PYREFLY check --python-version 3.13.0 "$TMPDIR/test-stubs/main.pyi"
[0]
```
