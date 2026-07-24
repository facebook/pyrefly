# Tests for `--suppress-errors`

## `suppress --remove-unused` preserves unused `# type: ignore` comments by default

```scrut
$ mkdir $TMPDIR/suppress_remove_unused_default && \
> printf 'a = 1  # pyrefly: ignore\nb = 2  # type: ignore\n' > $TMPDIR/suppress_remove_unused_default/main.py && \
> : > $TMPDIR/suppress_remove_unused_default/pyrefly.toml && \
> $PYREFLY suppress $TMPDIR/suppress_remove_unused_default/main.py \
>     --config $TMPDIR/suppress_remove_unused_default/pyrefly.toml \
>     --remove-unused \
>     >/dev/null 2>/dev/null && \
> cat $TMPDIR/suppress_remove_unused_default/main.py
a = 1
b = 2  # type: ignore
[0]
```

## `suppress --remove-unused=type` removes only unused `# type: ignore` comments

```scrut
$ mkdir $TMPDIR/suppress_remove_unused_type && \
> printf 'a = 1  # pyrefly: ignore\nb = 2  # type: ignore\n' > $TMPDIR/suppress_remove_unused_type/main.py && \
> : > $TMPDIR/suppress_remove_unused_type/pyrefly.toml && \
> $PYREFLY suppress $TMPDIR/suppress_remove_unused_type/main.py \
>     --config $TMPDIR/suppress_remove_unused_type/pyrefly.toml \
>     --remove-unused=type \
>     >/dev/null 2>/dev/null && \
> cat $TMPDIR/suppress_remove_unused_type/main.py
a = 1  # pyrefly: ignore
b = 2
[0]
```

## `suppress --remove-unused=all` removes both kinds of unused ignore

```scrut
$ mkdir $TMPDIR/suppress_remove_unused_all && \
> printf 'a = 1  # pyrefly: ignore\nb = 2  # type: ignore\n' > $TMPDIR/suppress_remove_unused_all/main.py && \
> : > $TMPDIR/suppress_remove_unused_all/pyrefly.toml && \
> $PYREFLY suppress $TMPDIR/suppress_remove_unused_all/main.py \
>     --config $TMPDIR/suppress_remove_unused_all/pyrefly.toml \
>     --remove-unused=all \
>     >/dev/null 2>/dev/null && \
> cat $TMPDIR/suppress_remove_unused_all/main.py
a = 1
b = 2
[0]
```

## `--suppress-errors` should not rewrite warnings hidden by `--min-severity`

Use an explicit empty config so the repro does not depend on any ancestor
`pyrefly.toml` discovered by upward config search.

```scrut
$ mkdir $TMPDIR/suppress_hidden_warning && \
> printf 'def f(x: str) -> None:\n    y = str(x)\n' > $TMPDIR/suppress_hidden_warning/main.py && \
> : > $TMPDIR/suppress_hidden_warning/pyrefly.toml && \
> $PYREFLY check $TMPDIR/suppress_hidden_warning/main.py \
>     --config $TMPDIR/suppress_hidden_warning/pyrefly.toml \
>     --min-severity error \
>     --suppress-errors \
>     --summary=none \
>     >/dev/null 2>/dev/null && \
> cat $TMPDIR/suppress_hidden_warning/main.py
def f(x: str) -> None:
    y = str(x)
[0]
```
