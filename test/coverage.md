# Tests for coverage commands

## Coverage commands also warn on a non-existent search-path/site-packages-path

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/covwarn && echo "def f(): pass" > $TMPDIR/covwarn/f.py && \
> echo -e "project_includes = [\"$TMPDIR/covwarn/f.py\"]\nsite_package_path = [\"$TMPDIR/covwarn/abcd\"]\nsearch_path = [\"$TMPDIR/covwarn/efgh\"]" > $TMPDIR/covwarn/pyrefly.toml && \
> $PYREFLY coverage report -c $TMPDIR/covwarn/pyrefly.toml > /dev/null
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 WARN */pyrefly.toml: Invalid site-packages-path: */abcd` does not exist (glob)
 WARN */pyrefly.toml: Invalid search-path: */efgh` does not exist (glob)
[0]
```


## Main help shows `coverage` subcommand and not the hidden `report` alias

```scrut
$ $PYREFLY --help | grep -E "^ +(coverage|  report)"
  coverage     Type coverage commands
[0]
```

## `pyrefly coverage report --help` shows correct usage

```scrut
$ $PYREFLY coverage report --help | grep "^Usage:"
Usage: pyrefly coverage report [OPTIONS] [FILES]...
[0]
```

## Deprecated `pyrefly report` alias emits a warning on stderr

```scrut
$ touch $TMPDIR/pyrefly.toml && \
> echo "def f(x: int) -> int: return x" > $TMPDIR/test.py && \
> $PYREFLY report $TMPDIR/test.py 2>&1 | grep "warning:"
warning: `pyrefly report` is deprecated; use `pyrefly coverage report` instead
[0]
```

## `pyrefly coverage report` emits modules in a deterministic order across runs

```scrut
$ cd $TMPDIR && rm -rf detrepo && mkdir detrepo && cd detrepo && touch pyrefly.toml && \
> for i in 1 2 3 4 5 6; do echo "def f$i() -> int: return $i" > "m$i.py"; done && \
> $PYREFLY coverage report m1.py m2.py m3.py m4.py m5.py m6.py > a.json 2>/dev/null && \
> $PYREFLY coverage report m1.py m2.py m3.py m4.py m5.py m6.py > b.json 2>/dev/null && \
> diff a.json b.json && echo IDENTICAL
IDENTICAL
[0]
```
