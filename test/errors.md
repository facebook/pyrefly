# Tests that return errors

## Error on a non-existent file

```scrut {output_stream: stderr}
$ $PYREFLY check $TMPDIR/does_not_exist --python-version 3.13.0
No Python files matched pattern `*/does_not_exist` (glob)
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
$ echo "x: str = 12" > $TMPDIR/shown1.py && \
> echo "import shown1; y: int = shown1.x" > $TMPDIR/shown2.py && \
> $PYREFLY check --python-version 3.13.0 $TMPDIR/shown2.py --check-all --output-format=min-text
*/shown*.py:1:* (glob)
 WARN ast.pyi:1113:10-11: `Constant.n` is deprecated [deprecated]
 WARN ast.pyi:1113:10-18: `Constant.n` is deprecated [deprecated]
 WARN ast.pyi:1123:10-11: `Constant.s` is deprecated [deprecated]
 WARN ast.pyi:1123:10-18: `Constant.s` is deprecated [deprecated]
 WARN importlib/resources/__init__.pyi:51:9-29: `contents` is deprecated [deprecated]
*/shown*.py:1:* (glob)
 WARN typing_extensions.pyi:65:5-55: `no_type_check_decorator` is deprecated [deprecated]
[1]
```

## We return an error when an entire project_includes pattern is matched by project_excludes

```scrut {output_stream: stderr}
$ $PYREFLY check --python-version 3.13.0 "$TMPDIR/*" --project-excludes="$TMPDIR/*"
Pattern * is matched by `project-excludes` or ignore file. (glob)
`project-excludes`: * (glob)
[1]
```

## --output-format controls error verbosity

```scrut
$ echo "1 + '2'" > $TMPDIR/bad.py && \
> $PYREFLY check $TMPDIR/bad.py --output-format=full-text
ERROR `+` is not supported * (glob)
 --> */bad.py:1:1 (glob)
  |
1 | 1 + '2'
  | ^^^^^^^
  |
  Argument * is not assignable * (glob)
[1]
```

```scrut
$ echo "1 + '2'" > $TMPDIR/bad.py && \
> $PYREFLY check $TMPDIR/bad.py --output-format=min-text
ERROR */bad.py:1:1-8: `+` is not supported * (glob)
[1]
```

## Source code snippet

```scrut
$ echo -e "def f(x: str): ...\nf(0.0)" > $TMPDIR/bad_call.py && \
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
$ echo -e "def f(x: str): ...\nλ = 0\nf(λ)" > $TMPDIR/bad_call.py && \
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
$ mkdir $TMPDIR/compiled && touch $TMPDIR/compiled/a.pyc && \
> touch $TMPDIR/compiled/b.pyc && touch $TMPDIR/c.pyc && touch $TMPDIR/d.pyc && \
> echo "from compiled import a; import compiled.b; import c; from . import d; reveal_type((a, compiled.b, c, d))" > $TMPDIR/compiled_import.py && \
> $PYREFLY check $TMPDIR/compiled_import.py
*ERROR `reveal_type` must be imported from `typing` for runtime usage* (glob)
* (glob+)
*INFO revealed type: tuple[Any, Module[compiled.b], Module[c], Any]* (glob)
* (glob+)
[1]
```
