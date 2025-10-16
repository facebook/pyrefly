# Tests for pyrefly configuration files

## Error on a non-existent search-path/site-package-path

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/test && echo "" > $TMPDIR/test/empty.py && \
> echo -e "project_includes = [\"$TMPDIR/test/empty.py\"]\nsite_package_path = [\"$TMPDIR/test/abcd\"]\nsearch_path = [\"$TMPDIR/test/efgh\"]" > $TMPDIR/test/pyrefly.toml && \
> $PYREFLY check -c $TMPDIR/test/pyrefly.toml --python-version 3.13.0
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 WARN */pyrefly.toml: Invalid site-package-path: */abcd` does not exist (glob)
 WARN */pyrefly.toml: Invalid search-path: */efgh` does not exist (glob)
 INFO * errors* (glob)
[1]
```

## Dump config

```scrut
$ touch $TMPDIR/foo.py && mkdir $TMPDIR/bar && touch $TMPDIR/bar/baz.py && touch $TMPDIR/bar/qux.py && mkdir $TMPDIR/spp && touch $TMPDIR/spp/mylib.py \
> && $PYREFLY dump-config --site-package-path $TMPDIR/spp/ $TMPDIR/foo.py $TMPDIR/bar/*.py
Default configuration
  Using interpreter: * (glob)
  Covered files:
    */bar/baz.py (glob)
    */bar/qux.py (glob)
  Resolving imports from:
    Fallback search path (guessed from project_includes): * (glob)
    Site package path from user: * (glob)
    Site package path queried from interpreter: * (glob)
Default configuration
  Using interpreter: * (glob)
  Covered files:
    */foo.py (glob)
  Resolving imports from:
    Fallback search path (guessed from project_includes): * (glob)
    Site package path from user: * (glob)
    Site package path queried from interpreter: * (glob)
[0]
```

## Specify both files and config

```scrut {output_stream: stderr}
$ echo "x: str = 0" > $TMPDIR/oops.py && echo "errors = { bad-assignment = false }" > $TMPDIR/pyrefly.toml && $PYREFLY check -c $TMPDIR/pyrefly.toml $TMPDIR/oops.py && rm $TMPDIR/pyrefly.toml
 INFO 0 errors (1 ignored)
[0]
```

## Error in implicit config (project mode)

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/bad_config && touch $TMPDIR/bad_config/empty.py && echo "oops oops" > $TMPDIR/bad_config/pyrefly.toml && cd $TMPDIR/bad_config && $PYREFLY check
 INFO Checking project configured at `*/pyrefly.toml` (glob)
ERROR */pyrefly.toml: TOML parse error* (glob)
  |
1 | oops oops
  |      ^
key with no value* (glob)

Fatal configuration error
[1]
```

## Error in implicit config (file mode)

<!-- Reusing bad_config dir set up in "Error in implicit config (project mode)" -->

```scrut {output_stream: stderr}
$ $PYREFLY check $TMPDIR/bad_config/empty.py
ERROR */pyrefly.toml: TOML parse error* (glob)
  |
1 | oops oops
  |      ^
key with no value* (glob)

Fatal configuration error
[1]
```

## Error in explicit config (project mode)

<!-- Reusing bad_config dir set up in "Error in implicit config (project mode)" -->

```scrut {output_stream: stderr}
$ $PYREFLY check -c $TMPDIR/bad_config/pyrefly.toml
 INFO Checking project configured at `*/pyrefly.toml` (glob)
ERROR */pyrefly.toml: TOML parse error* (glob)
  |
1 | oops oops
  |      ^
key with no value* (glob)

Fatal configuration error
[1]
```

## Error in explicit config (file mode)

<!-- Reusing bad_config dir set up in "Error in implicit config (project mode)" -->

```scrut {output_stream: stderr}
$ $PYREFLY check -c $TMPDIR/bad_config/pyrefly.toml $TMPDIR/bad_config/empty.py
ERROR */pyrefly.toml: TOML parse error* (glob)
  |
1 | oops oops
  |      ^
key with no value* (glob)

Fatal configuration error
[1]
```

## Interpreter priority takes CLI interpreter

```scrut {output_stream: stdout}
$ mkdir $TMPDIR/interpreters && touch $TMPDIR/interpreters/test.py \
> touch $TMPDIR/test-interpreter && \
> echo 'python-interpreter = "$TMPDIR/test-interpreter"' > $TMPDIR/interpreters/pyrefly.toml && \
> mkdir $TMPDIR/interpreters/venv && touch $TMPDIR/interpreters/venv/python3 && \
> touch $TMPDIR/interpreters/venv/pyvenv.cfg && \
> mkdir $TMPDIR/alternative-venv && touch $TMPDIR/alternative-venv/python3 && \
> touch $TMPDIR/alternative-venv/pyvenv.cfg && \
> VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml \
> --python-interpreter-path "cli-interpreter"
Configuration at * (glob)
  Using interpreter: cli-interpreter
* (glob+)
[0]
```

## Interpreter priority takes activated interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */alternative-venv/python3 (glob)
* (glob+)
[0]
```

## Interpreter priority takes config-file interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */test-interpreter (glob)
* (glob+)
[0]
```

## Interpreter priority takes venv interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ echo "" > $TMPDIR/interpreters/pyrefly.toml && \
> $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */interpreters/venv/python3 (glob)
* (glob+)
[0]
```

## Interpreter priority takes system interpreter last

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ rm -rf $TMPDIR/interpreters/venv && \
> $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: /*/python3 (glob)
* (glob+)
[0]
```

## We'll use the first marker file we find as a project root

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/config_finder/project && \
> touch $TMPDIR/config_finder/pyproject.toml && \
> touch $TMPDIR/config_finder/project/pyproject.toml && \
> touch $TMPDIR/config_finder/project/main.py && \
> $PYREFLY dump-config $TMPDIR/config_finder/project/main.py
Default configuration for project root marked by `*/config_finder/project/pyproject.toml` (glob)
* (glob+)
[0]
```

## We'll prefer a Pyrefly config in a parent directory to a marker file

<!-- Reusing configs set up in tests between here and "We'll use the first marker file we find as a project root" -->

```scrut {output_stream: stdout}
$ touch $TMPDIR/config_finder/pyrefly.toml && \
> $PYREFLY dump-config $TMPDIR/config_finder/project/main.py
Configuration at `*/config_finder/pyrefly.toml` (glob)
* (glob+)
[0]
```

## We'll prefer a Pyproject with Pyrefly config in a parent directory to a marker file

<!-- Reusing configs set up in tests between here and "We'll use the first marker file we find as a project root" -->

```scrut {output_stream: stdout}
$ rm $TMPDIR/config_finder/pyrefly.toml && \
> echo "[tool.pyrefly]" > $TMPDIR/config_finder/pyproject.toml && \
> $PYREFLY dump-config $TMPDIR/config_finder/project/main.py
Configuration at `*/config_finder/pyproject.toml` (glob)
* (glob+)
[0]
```

## We'll use the first Pyrefly config we find

<!-- Reusing configs set up in tests between here and "We'll use the first marker file we find as a project root" -->

```scrut {output_stream: stdout}
$ echo "[tool.pyrefly]" > $TMPDIR/config_finder/project/pyproject.toml && \
> $PYREFLY dump-config $TMPDIR/config_finder/project/main.py
Configuration at `*/config_finder/project/pyproject.toml` (glob)
* (glob+)
[0]
```

## We'll prefer pyrefly.toml to pyproject.toml

<!-- Reusing configs set up in tests between here and "We'll use the first marker file we find as a project root" -->

```scrut {output_stream: stdout}
$ touch $TMPDIR/config_finder/project/pyrefly.toml && \
> $PYREFLY dump-config $TMPDIR/config_finder/project/main.py
Configuration at `*/config_finder/project/pyrefly.toml` (glob)
* (glob+)
[0]
```

## We can manually override typing_extensions

<!-- See https://typing.python.org/en/latest/spec/distributing.html#import-resolution-ordering:
     typing_extensions.py on the search path takes precedence over typeshed
-->

```scrut {output_stream: stdout}
$ mkdir $TMPDIR/typing_extensions_project && \
> echo "x: int = 42" > $TMPDIR/typing_extensions_project/typing_extensions.py && \
> echo "from typing import assert_type; from typing_extensions import x; assert_type(x, int)" > $TMPDIR/typing_extensions_project/foo.py && \
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

## Skip hidden directories

```scrut {output_stream.stdout}
$ mkdir $TMPDIR/contains_hidden && \
> mkdir $TMPDIR/contains_hidden/.hidden && \
> touch $TMPDIR/contains_hidden/ok.py && \
> echo "1 + 'oops'" > $TMPDIR/contains_hidden/.hidden/secret_error.py && \
> $PYREFLY check $TMPDIR/contains_hidden
[0]
```
