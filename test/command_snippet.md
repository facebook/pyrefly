# Tests for pyrefly check --command option

## Basic command snippet with type error

```scrut
$ $PYREFLY check --command "x: int = 'hello'"
ERROR `Literal['hello']` is not assignable to `int` [bad-assignment]
 --> <string>:1:10
  |
1 | x: int = 'hello'
  |          ^^^^^^^
  |
[1]
```

## Valid command snippet (no errors)

```scrut {output_stream: stderr}
$ $PYREFLY check --command "x: int = 42"
 INFO Checking current directory with default configuration
 INFO errors shown: 0* (glob)
[0]
```

## Command snippet with built-in module import

```scrut {output_stream: stderr}
$ $PYREFLY check --command "import sys; print(sys.version)"
 INFO Checking current directory with default configuration
 INFO errors shown: 0* (glob)
[0]
```

## Command snippet with local file import

```scrut
$ echo "x: int = 5" > $TEST_PY && $PYREFLY check --command "import test; from typing import reveal_type; reveal_type(test.x)"
 INFO revealed type: int [reveal-type]
 --> <string>:1:57
  |
1 | import test; from typing import reveal_type; reveal_type(test.x)
  |                                                         --------
  |
[0]
```

## Command snippet with typing imports and error

```scrut
$ $PYREFLY check --command "from typing import List; x: List[str] = [1, 2, 3]"
ERROR `list[int]` is not assignable to `list[str]` [bad-assignment]
 --> <string>:1:41
  |
1 | from typing import List; x: List[str] = [1, 2, 3]
  |                                         ^^^^^^^^^
  |
[1]
```

## Command snippet with multiple errors

```scrut
$ $PYREFLY check --command "def foo(x: str) -> int: return len(x); y: str = foo(42)"
ERROR Function declared to return `int`, but one or more paths are missing an explicit `return` [bad-return]
 --> <string>:1:20
  |
1 | def foo(x: str) -> int: return len(x); y: str = foo(42)
  |                    ^^^
  |
ERROR `int` is not assignable to `str` [bad-assignment]
 --> <string>:1:49
  |
1 | def foo(x: str) -> int: return len(x); y: str = foo(42)
  |                                                 ^^^^^^^
  |
ERROR Argument `Literal[42]` is not assignable to parameter `x` with type `str` in function `foo` [bad-argument-type]
 --> <string>:1:53
  |
1 | def foo(x: str) -> int: return len(x); y: str = foo(42)
  |                                                     ^^
  |
[1]
```

## Command option conflicts with files

```scrut {output_stream: stderr}
$ echo "x: int = 42" > $TMPDIR/test.py && $PYREFLY check --command "x: int = 42" $TMPDIR/test.py
error: the argument '--command <CODE>' cannot be used with '[FILES]...'

Usage: pyrefly check --command <CODE> [FILES]...

For more information, try '--help'.
[2]
```

## Command snippet with JSON output format

```scrut
$ $PYREFLY check --command "x: int = 'hello'" --output-format=json
{
  "errors": [
    {
      "line": 1,
      "column": 10,
      "stop_line": 1,
      "stop_column": 17,
      "path": "<string>",
      "code": -2,
      "name": "bad-assignment",
      "description": "`Literal['hello']` is not assignable to `int`",
      "concise_description": "`Literal['hello']` is not assignable to `int`"
    }
  ]
} (no-eol)
[1]
```

## Help text shows command option

```scrut
$ $PYREFLY check --help | grep -A 1 "command"
      --command <CODE>
          Type check a string of Python code directly
[0]
```