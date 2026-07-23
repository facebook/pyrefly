# Tests that return errors

## Error on a non-existent file

```scrut {output_stream: stderr}
$ $PYREFLY check $TMPDIR/does_not_exist --python-version 3.13.0
Path `*/does_not_exist` does not exist (glob)
[1]
```

## Error on a non-existent search path

```scrut {output_stream: stderr}
$ echo "" > $TMPDIR/empty.py && $PYREFLY check $TMPDIR/empty.py --search-path $TMPDIR/does_not_exist
Invalid --search-path: `*/does_not_exist` does not exist (glob)
[1]
```

## We do report from nested with --check-all

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 12" > $TMPDIR/shown1.py && \
> echo "import shown1; y: int = shown1.x" > $TMPDIR/shown2.py && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/shown2.py --check-all --output-format=min-text --min-severity=warn
 WARN importlib/abc.pyi:147:9-41: `ResourceReader` is deprecated [deprecated]
 WARN importlib/resources/__init__.pyi:49:9-29: `contents` is deprecated [deprecated]
 WARN importlib/resources/__init__.pyi:79:41-73: `ResourceReader` is deprecated [deprecated]
 WARN importlib/resources/_common.pyi:8:41-55: `ResourceReader` is deprecated [deprecated]
*/shown*.py:1:* (glob)
*/shown*.py:1:* (glob)
[1]
```

## We return an error when an entire project_includes pattern is matched by project_excludes

```scrut {output_stream: stderr}
$ $PYREFLY check --python-version 3.13.0 "$TMPDIR/*" --project-excludes="$TMPDIR/*"
 WARN Skipping include pattern `*` because it is matched by `project-excludes` or an ignore file. (glob)
`project-excludes`: * (glob)
No Python files matched pattern `*` (glob)
[1]
```

## --output-format controls error verbosity

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "1 + '2'" > $TMPDIR/bad.py && \
> $PYREFLY check $TMPDIR/bad.py --output-format=full-text
ERROR `+` is not supported * (glob)
 --> */bad.py:1:1 (glob)
  |
1 | 1 + '2'
  | -^^^---
  | |   |
  | |   has type `Literal['2']`
  | has type `Literal[1]`
  |
  Argument * is not assignable * (glob)
[1]
```

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "1 + '2'" > $TMPDIR/bad.py && \
> $PYREFLY check $TMPDIR/bad.py --output-format=min-text
ERROR */bad.py:1:1-8: `+` is not supported * (glob)
[1]
```

## `full-text-with-github` preserves diagnostics in GitHub Actions logs

```scrut
$ echo "x: str = 0" > $TMPDIR/bad_combined.py && \
> $PYREFLY check $TMPDIR/bad_combined.py --preset strict --output-format=full-text-with-github --summary=none
ERROR `Literal[0]` is not assignable to `str` [bad-assignment]
 --> */bad_combined.py:1:10 (glob)
  |
1 | x: str = 0
  |    ---   ^
  |    |
  |    declared type
  |
::error file=*/bad_combined.py,line=1,col=10,endLine=1,endColumn=11,title=Pyrefly bad-assignment::`Literal[0]` is not assignable to `str` (glob)
[1]
```

## Source code snippet

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo -e "def f(x: str): ...\nf(0.0)" > $TMPDIR/bad_call.py && \
> $PYREFLY check $TMPDIR/bad_call.py
ERROR Argument `float` is not assignable * (glob)
 --> */bad_call.py:2:3 (glob)
  |
2 | f(0.0)
  |   ^^^
  |
[1]
```

## Source code snippet with multi-byte character

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo -e "def f(x: str): ...\nλ = 0\nf(λ)" > $TMPDIR/bad_call.py && \
> $PYREFLY check $TMPDIR/bad_call.py
ERROR Argument `Literal[0]` is not assignable * (glob)
 --> */bad_call.py:3:3 (glob)
  |
3 | f(λ)
  |   ^
  |
[1]
```

## We replace compiled modules with Any

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> mkdir $TMPDIR/compiled && touch $TMPDIR/compiled/a.pyc && \
> touch $TMPDIR/compiled/b.pyc && touch $TMPDIR/c.pyc && touch $TMPDIR/d.pyc && \
> echo "from compiled import a; import compiled.b; import c; from . import d; reveal_type((a, compiled.b, c, d))" > $TMPDIR/compiled_import.py && \
> $PYREFLY check $TMPDIR/compiled_import.py
*ERROR `reveal_type` must be imported from `typing` for runtime usage* (glob)
* (glob+)
*INFO revealed type: tuple[Unknown, Module[compiled.b], Module[c], Unknown]* (glob)
* (glob+)
[1]
```

## `--min-severity warn` causes nonzero exit on warnings

```scrut
$ echo "x: str = 0" > $TMPDIR/test.py && \
> $PYREFLY check $TMPDIR/test.py --warn=bad-assignment --min-severity=warn --output-format=min-text
 WARN */test.py:1:10-11: `Literal[0]` is not assignable to `str` [bad-assignment] (glob)
[1]
```

## `--output-format junit-xml` emits well-formed XML

```scrut {output_stream: stdout}
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 0" > $TMPDIR/bad.py && \
> $PYREFLY check --output-format junit-xml $TMPDIR/bad.py 2>/dev/null
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="pyrefly" tests="1" failures="1" errors="0" time="0">
    <testcase classname="*/bad.py" name="bad-assignment:L1" file="*/bad.py" line="1" time="0"> (glob)
      <failure type="bad-assignment" message="`Literal[0]` is not assignable to `str`"><![CDATA[`Literal[0]` is not assignable to `str`]]></failure>
    </testcase>
  </testsuite>
</testsuites>
[1]
```

## `--output-format junit-xml` omits warnings unless `--min-severity=warn`

Severity filtering happens before formatting, so by default a warning-level
finding produces an empty suite:

```scrut {output_stream: stdout}
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 0" > $TMPDIR/warn.py && \
> $PYREFLY check --warn=bad-assignment --output-format junit-xml $TMPDIR/warn.py 2>/dev/null
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="pyrefly" tests="0" failures="0" errors="0" time="0">
  </testsuite>
</testsuites>
```

Lowering the threshold with `--min-severity=warn` includes it, rendered like any
other failure with `type` set to the error kind:

```scrut {output_stream: stdout}
$ touch $TMPDIR/pyrefly.toml && \
> echo "x: str = 0" > $TMPDIR/warn.py && \
> $PYREFLY check --warn=bad-assignment --min-severity=warn --output-format junit-xml $TMPDIR/warn.py 2>/dev/null
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="pyrefly" tests="1" failures="1" errors="0" time="0">
    <testcase classname="*/warn.py" name="bad-assignment:L1" file="*/warn.py" line="1" time="0"> (glob)
      <failure type="bad-assignment" message="`Literal[0]` is not assignable to `str`"><![CDATA[`Literal[0]` is not assignable to `str`]]></failure>
    </testcase>
  </testsuite>
</testsuites>
[1]
```
