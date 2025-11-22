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
$ rm -f $TMPDIR/enabled_ignores/pyrefly.toml && \
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
