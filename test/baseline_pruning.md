# Tests for stale baseline detection and pruning

## `--error-stale-baseline` fails when a baseline entry's file is gone

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_error_stale && \
> echo "x: str = 1" > $TMPDIR/baseline_error_stale/bad.py && \
> echo '{"errors": [{"line": 1, "column": 10, "stop_line": 1, "stop_column": 11, "path": "bad.py", "code": -2, "name": "bad-assignment", "description": "test", "concise_description": "test"}, {"line": 1, "column": 1, "stop_line": 1, "stop_column": 2, "path": "gone.py", "code": -2, "name": "bad-return", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_error_stale/baseline.json && \
> touch $TMPDIR/baseline_error_stale/pyrefly.toml && \
> cd $TMPDIR/baseline_error_stale && \
> $PYREFLY check bad.py --baseline=baseline.json --error-stale-baseline --summary=none
ERROR Baseline file has 1 unused suppression; rerun with `--prune-baseline` to update it
[1]
```

## `--error-stale-baseline` succeeds when every baseline entry still matches

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_error_stale_clean && \
> echo "x: str = 1" > $TMPDIR/baseline_error_stale_clean/bad.py && \
> echo '{"errors": [{"line": 1, "column": 10, "stop_line": 1, "stop_column": 11, "path": "bad.py", "code": -2, "name": "bad-assignment", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_error_stale_clean/baseline.json && \
> touch $TMPDIR/baseline_error_stale_clean/pyrefly.toml && \
> cd $TMPDIR/baseline_error_stale_clean && \
> $PYREFLY check bad.py --baseline=baseline.json --error-stale-baseline --summary=none
[0]
```

## A stale entry for a checked file is reported

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_checked_fixed && \
> echo "x: str = 'fixed'" > $TMPDIR/baseline_checked_fixed/fixed.py && \
> echo '{"errors": [{"column": 10, "path": "fixed.py", "name": "bad-assignment", "concise_description": "test"}]}' > $TMPDIR/baseline_checked_fixed/baseline.json && \
> touch $TMPDIR/baseline_checked_fixed/pyrefly.toml && \
> cd $TMPDIR/baseline_checked_fixed && \
> $PYREFLY check fixed.py --baseline=baseline.json --error-stale-baseline --summary=none
ERROR Baseline file has 1 unused suppression; rerun with `--prune-baseline` to update it
[1]
```

## Existing files outside a narrowed check are retained

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_narrowed && \
> echo "x: str = 1" > $TMPDIR/baseline_narrowed/checked.py && \
> touch $TMPDIR/baseline_narrowed/unchecked.py $TMPDIR/baseline_narrowed/pyrefly.toml && \
> echo '{"errors": [{"column": 10, "path": "checked.py", "name": "bad-assignment", "concise_description": "checked"}, {"column": 1, "path": "unchecked.py", "name": "bad-return", "concise_description": "unchecked"}]}' > $TMPDIR/baseline_narrowed/baseline.json && \
> cd $TMPDIR/baseline_narrowed && \
> $PYREFLY check checked.py --baseline=baseline.json --prune-baseline --summary=none
 INFO Baseline file has no unused suppressions to remove
[0]
```

```scrut {output_stream: stdout}
$ $JQ '.errors | length' $TMPDIR/baseline_narrowed/baseline.json
2
[0]
```

## `--prune-baseline` drops stale entries without recording new errors

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_prune && \
> printf 'x: str = 1\nyyyy: int = ""\n' > $TMPDIR/baseline_prune/bad.py && \
> echo '{"errors": [{"line": 1, "column": 10, "stop_line": 1, "stop_column": 11, "path": "bad.py", "code": -2, "name": "bad-assignment", "description": "test", "concise_description": "test"}, {"line": 1, "column": 1, "stop_line": 1, "stop_column": 2, "path": "gone.py", "code": -2, "name": "bad-return", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_prune/baseline.json && \
> touch $TMPDIR/baseline_prune/pyrefly.toml && \
> cd $TMPDIR/baseline_prune && \
> $PYREFLY check bad.py --baseline=baseline.json --prune-baseline --summary=none --output-format=omit-errors
 INFO Removed 1 unused suppression from the baseline file
[1]
```

The still-matching entry is kept, and the new error on line 2 is not added.

```scrut {output_stream: stdout}
$ grep -c '"name":' $TMPDIR/baseline_prune/baseline.json
1
[0]
```

The surviving entry keeps its existing concise description rather than refreshing
it from the current error.

```scrut {output_stream: stdout}
$ grep -c '"concise_description": "test"' $TMPDIR/baseline_prune/baseline.json
1
[0]
```

## `--prune-baseline` does not apply `baseline-format`

Changing the configured format from full to minimal does not change the fields
of retained entries while pruning.

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_prune_full && \
> echo 'x: str = 1' > $TMPDIR/baseline_prune_full/bad.py && \
> echo '{"errors":[{"column":10,"path":"bad.py","name":"bad-assignment","concise_description":"test","severity":"error"},{"column":1,"path":"gone.py","name":"bad-return","concise_description":"stale","severity":"error"}]}' > $TMPDIR/baseline_prune_full/baseline.json && \
> printf 'baseline = "baseline.json"\nbaseline-format = "minimal"\n' > $TMPDIR/baseline_prune_full/pyrefly.toml && \
> cd $TMPDIR/baseline_prune_full && \
> $PYREFLY check bad.py --prune-baseline --summary=none --output-format=omit-errors >/dev/null 2>/dev/null && \
> $JQ -c '.errors[0] | keys' baseline.json
["column","concise_description","name","path","severity"]
[0]
```

Changing the configured format from minimal to full likewise leaves the fields
of retained entries minimal.

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_prune_minimal && \
> echo 'x: str = 1' > $TMPDIR/baseline_prune_minimal/bad.py && \
> echo '{"errors":[{"column":10,"path":"bad.py","name":"bad-assignment"},{"column":1,"path":"gone.py","name":"bad-return"}]}' > $TMPDIR/baseline_prune_minimal/baseline.json && \
> printf 'baseline = "baseline.json"\nbaseline-format = "full"\n' > $TMPDIR/baseline_prune_minimal/pyrefly.toml && \
> cd $TMPDIR/baseline_prune_minimal && \
> $PYREFLY check bad.py --prune-baseline --summary=none --output-format=omit-errors >/dev/null 2>/dev/null && \
> $JQ -c '.errors[0] | keys' baseline.json
["column","name","path"]
[0]
```

## Pruning keeps matching diagnostics below `--min-severity`

`--prune-baseline` only removes entries whose diagnostics no longer occur. In
contrast, `--update-baseline` regenerates the file using the severity threshold.

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_hidden_warning && \
> echo "x: str = 1" > $TMPDIR/baseline_hidden_warning/bad.py && \
> echo '{"errors": [{"column": 10, "path": "bad.py", "name": "bad-assignment", "concise_description": "test"}]}' > $TMPDIR/baseline_hidden_warning/baseline.json && \
> touch $TMPDIR/baseline_hidden_warning/pyrefly.toml && \
> cd $TMPDIR/baseline_hidden_warning && \
> $PYREFLY check bad.py --warn=bad-assignment --baseline=baseline.json --prune-baseline --summary=none
 INFO Baseline file has no unused suppressions to remove
[0]
```

## A baseline that cannot be parsed fails instead of silently passing `--error-stale-baseline`

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_broken && \
> echo "x: str = 1" > $TMPDIR/baseline_broken/bad.py && \
> echo 'not valid json' > $TMPDIR/baseline_broken/baseline.json && \
> touch $TMPDIR/baseline_broken/pyrefly.toml && \
> cd $TMPDIR/baseline_broken && \
> $PYREFLY check bad.py --baseline=baseline.json --error-stale-baseline --summary=none
*failed to read baseline file*baseline.json* (glob)
[1]
```

## A missing baseline file fails `--error-stale-baseline` instead of silently passing

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_missing_stale && \
> echo "x: str = 1" > $TMPDIR/baseline_missing_stale/bad.py && \
> touch $TMPDIR/baseline_missing_stale/pyrefly.toml && \
> cd $TMPDIR/baseline_missing_stale && \
> $PYREFLY check bad.py --baseline=missing.json --error-stale-baseline --summary=none
*requires an existing baseline file*missing.json*does not exist* (glob)
[1]
```

## A missing baseline file fails `--prune-baseline` instead of silently passing

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_missing_prune && \
> echo "x: str = 1" > $TMPDIR/baseline_missing_prune/bad.py && \
> touch $TMPDIR/baseline_missing_prune/pyrefly.toml && \
> cd $TMPDIR/baseline_missing_prune && \
> $PYREFLY check bad.py --baseline=missing.json --prune-baseline --summary=none
*requires an existing baseline file*missing.json*does not exist* (glob)
[1]
```

## The baseline actions are mutually exclusive

```scrut {output_stream: stderr}
$ cd $TMPDIR/baseline_prune && $PYREFLY check --prune-baseline --update-baseline
error: the argument '--prune-baseline' cannot be used with '--update-baseline'

Usage: pyrefly check --prune-baseline [FILES]...

For more information, try '--help'.
[2]
```

