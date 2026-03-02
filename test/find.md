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
$ PYREFLY_STDLIB_SEARCH_PATH=$TYPESHED_ROOT/typeshed/stdlib $PYREFLY check --python-version 3.13.0 $TYPESHED_ROOT/typeshed/stdlib/builtins.pyi --search-path $TYPESHED_ROOT/typeshed/stdlib --output-format=min-text --use-ignore-files=false --permissive-ignores=true 2>&1 | grep -v "overrides"
 INFO * errors* (glob)
[0]
```

## Ignore files are found correctly

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/ignores && echo "*" > $TMPDIR/ignores/.gitignore && \
> touch $TMPDIR/ignores/.ignore && mkdir -p $TMPDIR/ignores/.git/info && \
> touch $TMPDIR/ignores/.git/info/exclude && \
> echo "x: str = 12" > $TMPDIR/ignores/test.py && \
> touch $TMPDIR/ignores/pyrefly.toml && \
> $PYREFLY check --python-version 3.13.0 -c $TMPDIR/ignores/pyrefly.toml --output-format=min-text
 INFO Checking project configured at * (glob)
Pattern * is matched by `project-excludes` or ignore file. (glob)
`project-excludes`: [*], ignore files [*/.gitignore, */.ignore, */.git/info/exclude] (glob)
[1]
```

## Ignore files filter results

```scrut
$ mkdir $TMPDIR/gitignore && echo "error.py" > $TMPDIR/gitignore/.gitignore && \
> echo "from typing_extensions import reveal_type; reveal_type(1)" > $TMPDIR/gitignore/test.py && \
> echo "reveal_type(2)" > $TMPDIR/gitignore/error.py && \
> touch $TMPDIR/gitignore/pyrefly.toml && \
> $PYREFLY check --python-version 3.13.0 -c $TMPDIR/gitignore/pyrefly.toml --output-format=min-text
 INFO */gitignore/test.py* revealed type: Literal[1] * (glob)
[0]
```

## Ignore files work relative to their directory

```scrut
$ mkdir $TMPDIR/relative_ignore && \
> echo "my_project/gitignore_error.py" > $TMPDIR/relative_ignore/.gitignore && \
> mkdir -p $TMPDIR/relative_ignore/.git/info && \
> echo "my_project/gitexclude_error.py" > $TMPDIR/relative_ignore/.git/info/exclude && \
> mkdir $TMPDIR/relative_ignore/my_project && \
> echo "x: str = 1" > $TMPDIR/relative_ignore/my_project/gitignore_error.py && \
> echo "y: str = 2" > $TMPDIR/relative_ignore/my_project/gitexclude_error.py && \
> echo "x: int = 3" > $TMPDIR/relative_ignore/my_project/clean.py && \
> touch $TMPDIR/relative_ignore/my_project/pyrefly.toml && \
> $PYREFLY check --python-version 3.13.0 -c $TMPDIR/relative_ignore/my_project/pyrefly.toml --output-format=min-text
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

## Pyi files are preferred over py files, even if they occur later in the search path

```scrut
$ mkdir $TMPDIR/pyi-order && mkdir $TMPDIR/pyi-order/pyi && \
> echo "x: int = 1" > $TMPDIR/pyi-order/pyi/lib.pyi && \
> mkdir $TMPDIR/pyi-order/py && echo "x: str = '1'" > $TMPDIR/pyi-order/py/lib.py && \
> echo "from typing_extensions import reveal_type; from lib import x; reveal_type(x)" > $TMPDIR/pyi-order/test.py && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/pyi-order/test.py \
> --search-path $TMPDIR/pyi-order/py --search-path $TMPDIR/pyi-order/pyi --output-format=min-text
 INFO */pyi-order/test.py* revealed type: int * (glob)
[0]
```
