# `permissive-ignores` and `enabled-ignores`

## By default `# type: ignore` and `# pyrefly: ignore` are enabled

```scrut
$ mkdir $TMPDIR/enabled_ignores && \
> touch $TMPDIR/enabled_ignores/pyrefly.toml && \
> echo -e "1 + '1' # type: ignore\n1 + '1' # pyrefly: ignore\n1 + '1' # pyright: ignore" > $TMPDIR/enabled_ignores/foo.py && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --output-format=min-text
ERROR */foo.py:3* (glob)
[1]
```

## We can enable just `# pyright: ignore`

```scrut
$ echo "enabled-ignores = ['pyright']" > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --output-format=min-text
ERROR */foo.py:1* (glob)
ERROR */foo.py:2* (glob)
[1]
```

## We can enable `permissive-ignores`

```scrut
$ echo "permissive-ignores = true" > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --output-format=min-text
[0]
```

## `enabled-ignores` takes precedence

```scrut {output_stream: combined}
$ echo -e "enabled-ignores = ['pyright']\npermissive-ignores = true" > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --output-format=min-text --summary=none
 WARN * `permissive-ignores` will be ignored. (glob)
ERROR */foo.py:1* (glob)
ERROR */foo.py:2* (glob)
[1]
```

## We can set `--enabled-ignores` on the command line

```scrut
$ : > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --enabled-ignores=pyright --output-format=min-text
ERROR */foo.py:1* (glob)
ERROR */foo.py:2* (glob)
[1]
```

## We cannot set both `--enabled-ignores` and `--permissive-ignores` on the command line

```scrut {output_stream: stderr}
$ rm -f $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --enabled-ignores=pyright --permissive-ignores
Cannot use both `--permissive-ignores` and `--enabled-ignores`
[1]
```

## Command line overrides config file

```scrut {output_stream: combined}
$ echo "enabled-ignores = ['pyright']" > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --permissive-ignores --output-format=min-text --summary=none
[0]
```

```scrut {output_stream: combined}
$ echo "permissive-ignores = true" > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py --enabled-ignores=pyright --output-format=min-text --summary=none
ERROR */foo.py:1* (glob)
ERROR */foo.py:2* (glob)
[1]
```

```scrut {output_stream: combined}
$ echo "enabled-ignores = ['pyright']" > $TMPDIR/enabled_ignores/pyrefly.toml && \
> $PYREFLY check $TMPDIR/enabled_ignores/foo.py \
> --permissive-ignores=false --output-format=min-text --summary=none
ERROR */foo.py:3* (glob)
[1]
```

## A malformed ignore comment is flagged (warn) without changing what it suppresses

```scrut
$ mkdir $TMPDIR/ig_lints && touch $TMPDIR/ig_lints/pyrefly.toml && \
> printf 'x: int = "oops"  # pyrefly: ignoree\n' > $TMPDIR/ig_lints/typo.py && \
> $PYREFLY check $TMPDIR/ig_lints/typo.py --min-severity=warn --output-format=min-text
 WARN */typo.py:1* (glob)
ERROR */typo.py:1* (glob)
[1]
```

## `invalid-ignore-comment` is a warning, so it is hidden at the default min-severity

```scrut
$ $PYREFLY check $TMPDIR/ig_lints/typo.py --output-format=min-text
ERROR */typo.py:1* (glob)
[1]
```

## `# type: list[int]` and valid ignores are never flagged

```scrut
$ printf 'y = list[int]  # type: list[int]\nz: int = "x"  # pyrefly: ignore[bad-assignment]\n' \
> > $TMPDIR/ig_lints/clean.py && \
> $PYREFLY check $TMPDIR/ig_lints/clean.py --min-severity=warn --output-format=min-text
[0]
```

## A bare `# pyrefly: ignore` is not flagged by default

```scrut
$ printf 'x: int = "x"  # pyrefly: ignore\n' > $TMPDIR/ig_lints/bare.py && \
> $PYREFLY check $TMPDIR/ig_lints/bare.py --min-severity=warn --output-format=min-text
[0]
```

## Ignore-comment diagnostics are unsuppressable: file-level `# pyrefly: ignore-errors` does not hide them

```scrut
$ printf '# pyrefly: ignore-errors\nx: int = "oops"  # pyrefly: ignoree\n' \
> > $TMPDIR/ig_lints/unsupp.py && \
> $PYREFLY check $TMPDIR/ig_lints/unsupp.py --min-severity=warn --output-format=min-text
 WARN */unsupp.py:2* (glob)
[1]
```

## ...nor can a `# pyrefly: ignore[invalid-ignore-comment]` on the same line silence it

```scrut
$ printf 'x = 1  # pyrefly: ignoree  # pyrefly: ignore[invalid-ignore-comment]\n' \
> > $TMPDIR/ig_lints/selfsupp.py && \
> $PYREFLY check $TMPDIR/ig_lints/selfsupp.py --min-severity=warn --output-format=min-text
 WARN */selfsupp.py:1* (glob)
[1]
```

## When `ignore-without-code` is enabled, only `# pyrefly: ignore` is flagged

```scrut
$ printf 'x: int = "x"  # pyrefly: ignore\nz: int = "x"  # type: ignore\n' \
> > $TMPDIR/ig_lints/scoped.py && \
> echo 'errors = {ignore-without-code = true}' > $TMPDIR/ig_lints/pyrefly.toml && \
> $PYREFLY check $TMPDIR/ig_lints/scoped.py --output-format=min-text
ERROR */scoped.py:1* (glob)
[1]
```
