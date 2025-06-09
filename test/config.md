# Tests for pyrefly configuration files

## Error on a non-existent search-path/site-package-path

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/test && echo "" > $TMPDIR/test/empty.py && echo -e "project_includes = [\"$TMPDIR/test/empty.py\"]\nsite_package_path = [\"$TMPDIR/test/abcd\"]\nsearch_path = [\"$TMPDIR/test/abcd\"]" > $TMPDIR/test/pyrefly.toml && $PYREFLY check -c $TMPDIR/test/pyrefly.toml --python-version 3.13.0
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 WARN */pyrefly.toml: Invalid site_package_path: * does not exist (glob)
 WARN */pyrefly.toml: Invalid search_path: * does not exist (glob)
 INFO * errors* (glob)
[1]
```

## Dump config

```scrut
$ touch $TMPDIR/foo.py && mkdir $TMPDIR/bar && touch $TMPDIR/bar/baz.py && touch $TMPDIR/bar/qux.py && $PYREFLY dump-config $TMPDIR/foo.py $TMPDIR/bar/*.py
Default configuration
  Covered files:
    */bar/baz.py (glob)
    */bar/qux.py (glob)
  Fallback search path (guessed from project_includes): * (glob)
  Site package path (*): * (glob)
Default configuration
  Covered files:
    */foo.py (glob)
  Fallback search path (guessed from project_includes): * (glob)
  Site package path (*): * (glob)
[0]
```

## Specify both files and config

```scrut {output_stream: stderr}
$ echo "x: str = 0" > $TMPDIR/oops.py && echo "errors = { bad-assignment = false }" > $TMPDIR/pyrefly.toml && $PYREFLY check -c $TMPDIR/pyrefly.toml $TMPDIR/oops.py && rm $TMPDIR/pyrefly.toml
 INFO errors shown: 0* (glob)
[0]
```

## Error in implicit config (project mode)

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/implicit && touch $TMPDIR/implicit/empty.py && echo "oops oops" > $TMPDIR/implicit/pyrefly.toml && cd $TMPDIR/implicit && $PYREFLY check
 INFO Checking project configured at `*/pyrefly.toml` (glob)
ERROR */pyrefly.toml: TOML parse error* (glob)
* (glob*)
[1]
```

## Error in implicit config (file mode)

<!-- Reusing implicit dir with bad pyrefly.toml set up in "Error in implicit config (project mode)" -->

```scrut {output_stream: stderr}
$ $PYREFLY check $TMPDIR/implicit/empty.py
ERROR */pyrefly.toml: TOML parse error* (glob)
* (glob*)
[1]
```
