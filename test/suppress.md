# Tests for `--suppress-errors`

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
