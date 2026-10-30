# Tests for baseline matching

## Minimal baselines contain only the configured matching fields

```scrut
$ mkdir -p $TMPDIR/baseline_minimal && \
> echo "x: str = 1" > $TMPDIR/baseline_minimal/bad.py && \
> printf 'baseline = "baseline.json"\nbaseline-format = "minimal"\n' > $TMPDIR/baseline_minimal/pyrefly.toml && \
> cd $TMPDIR/baseline_minimal && \
> $PYREFLY check --update-baseline --summary=none --output-format=omit-errors >/dev/null 2>/dev/null
[1]
```

The default `column` mode needs only the path, error kind, and column.

```scrut {output_stream: stdout}
$ $JQ -c '.errors[0] | keys' $TMPDIR/baseline_minimal/baseline.json
["column","name","path"]
[0]
```

## A baseline missing the configured matching field explains how to update it

Changing the matching mode makes the column-only baseline invalid.

```scrut {output_stream: stderr}
$ printf 'baseline = "baseline.json"\nbaseline-format = "minimal"\nbaseline-matching-mode = "concise-description"\n' > $TMPDIR/baseline_minimal/pyrefly.toml && \
> cd $TMPDIR/baseline_minimal && \
> $PYREFLY check bad.py --summary=none
*failed to read baseline file*baseline file is invalid*rerun with `--update-baseline`*missing field `concise_description`* (glob)
[1]
```

Regenerating the baseline writes the matching fields for the new mode.

```scrut
$ cd $TMPDIR/baseline_minimal && \
> $PYREFLY check --update-baseline --summary=none --output-format=omit-errors >/dev/null 2>/dev/null
[1]
```

```scrut {output_stream: stdout}
$ $JQ -c '.errors[0] | keys' $TMPDIR/baseline_minimal/baseline.json
["concise_description","name","path"]
[0]
```

The concise-description mode keeps matching when the diagnostic moves to a
different column.

```scrut {output_stream: stderr}
$ printf 'if True:\n    x: str = 1\n' > $TMPDIR/baseline_minimal/bad.py && \
> cd $TMPDIR/baseline_minimal && \
> $PYREFLY check bad.py --summary=none
[0]
```

## `column-ordered` reports the diagnostic that was added, not a later one

`bad-assignment` is at column 10 and `bad-return` at column 12, so the baseline's keys
for this file are `[col 10, col 12, col 10]`. Inserting another column-10 diagnostic
between the first two aligns everything around it, leaving only the new line reported.

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_ordered && \
> printf 'x: str = 1\ndef f() -> int: return ""\nz: str = 1\n' > $TMPDIR/baseline_ordered/bad.py && \
> printf 'baseline = "baseline.json"\nbaseline-matching-mode = "column-ordered"\n' > $TMPDIR/baseline_ordered/pyrefly.toml && \
> cd $TMPDIR/baseline_ordered && \
> $PYREFLY check bad.py --update-baseline --summary=none >/dev/null 2>/dev/null; \
> printf 'x: str = 1\ny: str = 1\ndef f() -> int: return ""\nz: str = 1\n' > bad.py && \
> $PYREFLY check bad.py --output-format=min-text
ERROR *bad.py:2:*bad-assignment* (glob)
[1]
```

An unchanged file reports nothing.

```scrut {output_stream: stderr}
$ cd $TMPDIR/baseline_ordered && \
> $PYREFLY check bad.py --update-baseline --summary=none >/dev/null 2>/dev/null; \
> $PYREFLY check bad.py --summary=none
[0]
```

## `column-ordered` retires rows the alignment did not match

Removing two of the four baselined diagnostics leaves exactly two rows unmatched.

```scrut {output_stream: stderr}
$ printf 'x: str = 1\ndef f() -> int: return ""\n' > $TMPDIR/baseline_ordered/bad.py && \
> cd $TMPDIR/baseline_ordered && \
> $PYREFLY check bad.py --error-stale-baseline --summary=none
ERROR Baseline file has 2 unused suppressions; rerun with `--prune-baseline` to update it
[1]
```

## `column-ordered` records the severity threshold it was written with

A diagnostic below the recorded threshold takes no part in the alignment, so it cannot
displace one the baseline does record. The threshold is stored rather than taken from the
current run, so a later run with a different `--min-severity` still classifies rows the
same way.

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_threshold && \
> echo "x: str = 1" > $TMPDIR/baseline_threshold/bad.py && \
> printf 'baseline = "baseline.json"\nbaseline-matching-mode = "column-ordered"\n' > $TMPDIR/baseline_threshold/pyrefly.toml && \
> cd $TMPDIR/baseline_threshold && \
> $PYREFLY check bad.py --warn=bad-assignment --min-severity=warn --update-baseline --summary=none >/dev/null 2>/dev/null; \
> $JQ -r '.min_severity' baseline.json
warn
[0]
```

Checking the same project at the default threshold must not retire that row: the
diagnostic it describes is still there.

```scrut {output_stream: stderr}
$ cd $TMPDIR/baseline_threshold && \
> $PYREFLY check bad.py --warn=bad-assignment --error-stale-baseline --summary=none
[0]
```

