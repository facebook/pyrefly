# Simple CLI tests

## No errors on the empty file

```scrut {output_stream: stderr}
$ touch $TMPDIR/pyrefly.toml && \
> echo "" > $TMPDIR/empty.py && $PYREFLY check --python-version 3.13.0 $TMPDIR/empty.py -a
 INFO 0 errors* (glob)
[0]
```

## No errors on reveal_type

```scrut {output_stream: stderr}
$ touch $TMPDIR/pyrefly.toml && \
> echo -e "from typing import reveal_type\nreveal_type(1)" > $TMPDIR/empty.py && $PYREFLY check --python-version 3.13.0 $TMPDIR/empty.py -a
 INFO 0 errors* (glob)
[0]
```

## No errors on our test script

```scrut {output_stream: stderr}
$ touch $TMPDIR/pyrefly.toml && \
> cp $TEST_PY $TMPDIR/test.py && $PYREFLY check $TMPDIR/test.py
 INFO Loading new build system at * (glob?)
 INFO Querying Buck for source DB (glob?)
 INFO Source DB build ID: * (glob?)
 INFO Finished querying Buck for source DB (glob?)
 INFO 0 errors
[0]
```

## No errors on our Python code

```scrut {output_stream: stderr}
$ touch $(dirname $PYREFLY_PY)/pyrefly.toml && $PYREFLY check $PYREFLY_PY
 INFO 0 errors
[0]
```

## Text output on stdout

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 42" > $TMPDIR/test.py && $PYREFLY check $TMPDIR/test.py --output-format=min-text
ERROR */test.py:1:* (glob)
[1]
```

## JSON output on stdout

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 42" > $TMPDIR/test.py && $PYREFLY check $TMPDIR/test.py --output-format json | $JQ '.[] | length'
1
[0]
```

## We can typecheck two files with the same name

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 12" > $TMPDIR/same_name.py && \
> echo "x: str = True" > $TMPDIR/same_name.pyi && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/same_name.py $TMPDIR/same_name.pyi --output-format=min-text
ERROR */same_name.py*:1:10-* (glob)
ERROR */same_name.py*:1:10-* (glob)
[1]
```

## We don't report from nested files

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 12" > $TMPDIR/hidden1.py && \
> echo "import hidden1; y: int = hidden1.x" > $TMPDIR/hidden2.py && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/hidden2.py --output-format=min-text
ERROR */hidden2.py:1:26-35: `str` is not assignable to `int` [bad-assignment] (glob)
[1]
```

## We show how many warnings are hidden

```scrut {output_stream: stderr}
$ echo "x: str = 0" > $TMPDIR/test.py && \
> $PYREFLY check $TMPDIR/test.py --warn=bad-assignment
 INFO 0 errors (1 warning not shown, use `--min-severity=warn` to see it)* (glob)
[0]
```

## We show how many warnings are hidden, pluralized

```scrut {output_stream: stderr}
$ printf 'x: str = 0\ny: str = 0\n' > $TMPDIR/test.py && \
> $PYREFLY check $TMPDIR/test.py --warn=bad-assignment
 INFO 0 errors (2 warnings not shown, use `--min-severity=warn` to see them)* (glob)
[0]
```
