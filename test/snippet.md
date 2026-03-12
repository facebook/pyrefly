# Tests for pyrefly snippet command

## Basic snippet with type error

```scrut
$ $PYREFLY snippet "x: int = 'hello'"
ERROR `Literal['hello']` is not assignable to `int` [bad-assignment]
 --> snippet:1:10
  |
1 | x: int = 'hello'
  |          ^^^^^^^
  |
[1]
```

## Valid snippet (no errors)

```scrut {output_stream: stderr}
$ $PYREFLY snippet "x: int = 42"
 INFO Checking current directory with default configuration
 INFO 0 errors
[0]
```

## Snippet with built-in module import

```scrut {output_stream: stderr}
$ $PYREFLY snippet "import sys; print(sys.version)"
 INFO Checking current directory with default configuration
 INFO 0 errors
[0]
```

## Snippet with local file import

```scrut
$ echo "x: int = 5" > $TMPDIR/test.py && \
> touch $TMPDIR/pyrefly.toml && \
> $PYREFLY snippet "import test; reveal_type(test.x)" -c $TMPDIR/pyrefly.toml
ERROR `reveal_type` must be imported from `typing` for runtime usage [unimported-directive]
 --> snippet:1:14
  |
1 | import test; reveal_type(test.x)
  |              ^^^^^^^^^^^
  |
 INFO revealed type: int [reveal-type]
 --> snippet:1:25
  |
1 | import test; reveal_type(test.x)
  |                         --------
  |
[1]
```

## Snippet with typing imports and error

```scrut
$ $PYREFLY snippet "from typing import List; x: List[str] = [1, 2, 3]"
ERROR `list[int]` is not assignable to `list[str]` [bad-assignment]
 --> snippet:1:41
  |
1 | from typing import List; x: List[str] = [1, 2, 3]
  |                                         ^^^^^^^^^
  |
[1]
```

## Snippet with multiple errors

```scrut
$ $PYREFLY snippet "def foo(x: str) -> int: return len(x); y: str = foo(42)"
ERROR Function declared to return `int`, but one or more paths are missing an explicit `return` [bad-return]
 --> snippet:1:20
  |
1 | def foo(x: str) -> int: return len(x); y: str = foo(42)
  |                    ^^^
  |
ERROR `int` is not assignable to `str` [bad-assignment]
 --> snippet:1:49
  |
1 | def foo(x: str) -> int: return len(x); y: str = foo(42)
  |                                                 ^^^^^^^
  |
ERROR Argument `Literal[42]` is not assignable to parameter `x` with type `str` in function `foo` [bad-argument-type]
 --> snippet:1:53
  |
1 | def foo(x: str) -> int: return len(x); y: str = foo(42)
  |                                                     ^^
  |
[1]
```

## Snippet with JSON output format

```scrut
$ $PYREFLY snippet "x: int = 'hello'" --output-format=json
{
  "errors": [
    {
      "line": 1,
      "column": 10,
      "stop_line": 1,
      "stop_column": 17,
      "path": "snippet",
      "code": -2,
      "name": "bad-assignment",
      "description": "`Literal['hello']` is not assignable to `int`",
      "concise_description": "`Literal['hello']` is not assignable to `int`",
      "severity": "error"
    }
  ]
} (no-eol)
[1]
```

## Snippet with config file

```scrut {output_stream: stderr}
$ echo "python_version = \"3.11\"" > pyrefly.toml && $PYREFLY snippet "x: int = 42" --config pyrefly.toml
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 0 errors
[0]
```

## Help text shows snippet command

```scrut
$ $PYREFLY snippet --help | head -3
Check a Python code snippet

Usage: pyrefly snippet [OPTIONS] <CODE>
[0]
```

## Main help shows snippet command

```scrut
$ $PYREFLY --help | grep "snippet"
  snippet      Check a Python code snippet
[0]
```
