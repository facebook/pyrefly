# Tests for automatic configuration

## Upsell fires once for multiple files sharing a config (stderr only)

Two assertions on the same scenario: (1) the upsell text appears on
stderr exactly once across all the user-arg files, and (2) stdout stays
clean — important so machine-readable output formats (json, omit-errors,
…) aren't polluted. Captured to files so the single block can pin
both streams.

`mktemp -d -p /tmp` ensures the upward config search cannot find a
`pyrefly.toml` belonging to the scrut test itself.

```scrut
$ UPSELL_SAME=$(mktemp -d -p /tmp upsell.XXXXXX) && \
> echo "x = 1" > $UPSELL_SAME/a.py && echo "y = 2" > $UPSELL_SAME/b.py && \
> $PYREFLY check $UPSELL_SAME/a.py $UPSELL_SAME/b.py \
>     >$UPSELL_SAME/out.txt 2>$UPSELL_SAME/err.txt; \
> echo "---STDOUT---"; cat $UPSELL_SAME/out.txt; \
> echo "---STDERR---"; cat $UPSELL_SAME/err.txt; \
> rm -rf $UPSELL_SAME
---STDOUT---
---STDERR---
 INFO 0 errors* (glob)
No `pyrefly.toml` found — using preset `basic`.
Run `pyrefly init` to continue setting up Pyrefly.
Docs: * (glob)
[0]
```

## Explicit `--config` suppresses the upsell

```scrut {output_stream: stderr}
$ mkdir $TMPDIR/upsell_explicit && touch $TMPDIR/upsell_explicit/cfg.toml && \
> echo "x = 1" > $TMPDIR/upsell_explicit/foo.py && \
> $PYREFLY check $TMPDIR/upsell_explicit/foo.py --config $TMPDIR/upsell_explicit/cfg.toml
 INFO 0 errors* (glob)
[0]
```

## Files spanning distinct config roots suppress the upsell

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/upsell_split/p1 && touch $TMPDIR/upsell_split/p1/pyrefly.toml && \
> echo "x = 1" > $TMPDIR/upsell_split/p1/a.py && \
> mkdir $TMPDIR/upsell_split/p2 && echo "y = 2" > $TMPDIR/upsell_split/p2/b.py && \
> $PYREFLY check $TMPDIR/upsell_split/p1/a.py $TMPDIR/upsell_split/p2/b.py
 INFO 0 errors* (glob)
[0]
```

## `--summary=none` suppresses the upsell

The upsell is part of the summary surface. Tools that pass
`--summary=none` (e.g. `pyrefly init`'s self-invocation, scripts that
already render their own summary) shouldn't have unsolicited copy
appended on stderr.

```scrut {output_stream: stderr}
$ UPSELL_QUIET=$(mktemp -d -p /tmp upsell.XXXXXX) && \
> echo "x = 1" > $UPSELL_QUIET/a.py && \
> $PYREFLY check $UPSELL_QUIET/a.py --summary=none; rm -rf $UPSELL_QUIET
[0]
```

## Project-mode upsell parity with file mode (no nearby config)

`pyrefly check` (project mode, no file args) and `pyrefly check <file>`
(file mode) should produce the same upsell output in an unconfigured
repo. Without resolver wiring on the project-mode path, `pyrefly
check` would silently skip the upsell that `pyrefly check <file>`
shows from the same directory.

`mktemp -d -p /tmp` keeps this project outside any test-local configuration.

```scrut {output_stream: stderr}
$ UPSELL_PROJ=$(mktemp -d -p /tmp upsell.XXXXXX) && \
> echo "x = 1" > $UPSELL_PROJ/a.py && \
> cd $UPSELL_PROJ && $PYREFLY check; cd / && rm -rf $UPSELL_PROJ
 INFO Checking current directory with auto configuration
 INFO 0 errors* (glob)
No `pyrefly.toml` found — using preset `basic`.
Run `pyrefly init` to continue setting up Pyrefly.
Docs: * (glob)
[0]
```

## Project-mode upsell with `.` as explicit cwd

`pyrefly check .` is the file-mode equivalent of `pyrefly check`: a
single user-supplied "file" argument that resolves to the cwd. It must
produce the same upsell as bare `pyrefly check`.

```scrut {output_stream: stderr}
$ UPSELL_DOT=$(mktemp -d -p /tmp upsell.XXXXXX) && \
> echo "x = 1" > $UPSELL_DOT/a.py && \
> cd $UPSELL_DOT && $PYREFLY check .; cd / && rm -rf $UPSELL_DOT
 INFO 0 errors* (glob)
No `pyrefly.toml` found — using preset `basic`.
Run `pyrefly init` to continue setting up Pyrefly.
Docs: * (glob)
[0]
```

## Migrated `mypy.ini` works when invoked from a subdirectory

Regression guard: when `pyrefly check` is invoked from a subdirectory
and the resolver migrates a `mypy.ini` that lives several levels
above, the migrated per-module overrides must still apply to the
right files. Pyrefly's own upward marker search currently lands the
configurer at `mypy.ini`'s parent — so root and config dir already
agree — but a future change that decouples those (e.g. routing
project-mode through a different entry point) could put them out of
sync, and a per-module override silently failing to match is the
exact kind of regression this test catches.

The setup: a `bad-assignment` in `app/models/foo.py` under
`[mypy-app.models]` with `disable_error_code = assignment`
(mypy's own code name for the same rule), and a separate
`bad-assignment` in `app/views/bar.py` with no override. Run
`pyrefly check` from `sub/inner`. The override must hide the
`models` error but leave the `views` error visible.

```scrut
$ MYPY_PARENT=$(mktemp -d -p /tmp upsell.XXXXXX) && \
> printf '[mypy]\nfiles = app\n\n[mypy-app.models]\ndisable_error_code = assignment\n' \
>     > $MYPY_PARENT/mypy.ini && \
> mkdir -p $MYPY_PARENT/app/models $MYPY_PARENT/app/views && \
> echo "x: str = 0" > $MYPY_PARENT/app/models/foo.py && \
> echo "y: str = 0" > $MYPY_PARENT/app/views/bar.py && \
> mkdir -p $MYPY_PARENT/sub/inner && \
> cd $MYPY_PARENT/sub/inner && \
> $PYREFLY check --output-format=min-text \
>     >$MYPY_PARENT/out.txt 2>$MYPY_PARENT/err.txt; \
> echo "---STDOUT---"; cat $MYPY_PARENT/out.txt; \
> echo "---STDERR---"; cat $MYPY_PARENT/err.txt; \
> cd / && rm -rf $MYPY_PARENT
---STDOUT---
ERROR */app/views/bar.py:1:* (glob)
---STDERR---
 INFO Found `*/mypy.ini` marking project root, checking root directory with auto configuration (glob)
 INFO 1 error* (glob)
No `pyrefly.toml` found — using settings imported from your `mypy.ini` (preset: legacy).
Run `pyrefly init` to continue setting up Pyrefly.
Docs: * (glob)
[0]
```

