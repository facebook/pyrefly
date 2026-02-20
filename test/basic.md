# Simple CLI tests

## No errors on the empty file

```scrut {output_stream: stderr}
$ echo "" > $TMPDIR/empty.py && $PYREFLY check --python-version 3.13.0 $TMPDIR/empty.py -a
 INFO 0 errors* (glob)
[0]
```

## No errors on reveal_type

```scrut {output_stream: stderr}
$ echo -e "from typing import reveal_type\nreveal_type(1)" > $TMPDIR/empty.py && $PYREFLY check --python-version 3.13.0 $TMPDIR/empty.py -a
 INFO 0 errors* (glob)
[0]
```

## No errors on our test script

```scrut {output_stream: stderr}
$ $PYREFLY check $TEST_PY
 INFO Loading new build system at * (glob?)
 INFO Querying Buck for source DB (glob?)
 INFO Source DB build ID: * (glob?)
 INFO Finished querying Buck for source DB (glob?)
 INFO 0 errors
[0]
```

## No errors on our Python code

```scrut {output_stream: stderr}
$ $PYREFLY check $PYREFLY_PY
 INFO 0 errors
[0]
```

## Text output on stdout

```scrut
$ echo "x: str = 42" > $TMPDIR/test.py && $PYREFLY check $TMPDIR/test.py --output-format=min-text
ERROR */test.py:1:* (glob)
[1]
```

## JSON output on stdout

```scrut
$ echo "x: str = 42" > $TMPDIR/test.py && $PYREFLY check $TMPDIR/test.py --output-format json | $JQ '.[] | length'
1
[0]
```

## We can typecheck two files with the same name

```scrut
$ echo "x: str = 12" > $TMPDIR/same_name.py && \
> echo "x: str = True" > $TMPDIR/same_name.pyi && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/same_name.py $TMPDIR/same_name.pyi --output-format=min-text
ERROR */same_name.py*:1:10-* (glob)
ERROR */same_name.py*:1:10-* (glob)
[1]
```

## We don't report from nested files

```scrut
$ echo "x: str = 12" > $TMPDIR/hidden1.py && \
> echo "import hidden1; y: int = hidden1.x" > $TMPDIR/hidden2.py && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/hidden2.py --output-format=min-text
ERROR */hidden2.py:1:26-35: `str` is not assignable to `int` [bad-assignment] (glob)
[1]
```

## We can find a venv interpreter, even when not sourced

```scrut {output_stream: stderr}
$ python3 -m venv $TMPDIR/venv && \
> echo "import third_party.test2" > $TMPDIR/test.py && \
> export site_packages=$($TMPDIR/venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
> mkdir $site_packages/third_party && \
> echo "x = 1" > $site_packages/third_party/test2.py && \
> touch $TMPDIR/pyrefly.toml && \
> $PYREFLY check $TMPDIR/test.py
 INFO 0 errors* (glob)
[0]
```
