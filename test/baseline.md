# Tests for baseline configuration

## Baseline in pyrefly.toml suppresses matching errors

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_test && \
> echo "x: str = 1" > $TMPDIR/baseline_test/bad.py && \
> echo '{"errors": [{"line": 1, "column": 10, "stop_line": 1, "stop_column": 11, "path": "bad.py", "code": -2, "name": "bad-assignment", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_test/baseline.json && \
> echo 'baseline = "baseline.json"' > $TMPDIR/baseline_test/pyrefly.toml && \
> cd $TMPDIR/baseline_test && $PYREFLY check
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 0 errors
[0]
```

## Without baseline config, errors are shown

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/no_baseline && \
> echo "z: str = 1" > $TMPDIR/no_baseline/bad.py && \
> touch $TMPDIR/no_baseline/pyrefly.toml && \
> cd $TMPDIR/no_baseline && $PYREFLY check --output-format=min-text
ERROR *bad.py* ?bad-assignment? (glob)
[1]
```

## CLI --baseline flag overrides config baseline

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_override && \
> echo "a: str = 1" > $TMPDIR/baseline_override/bad.py && \
> echo '{"errors": []}' > $TMPDIR/baseline_override/empty_baseline.json && \
> echo '{"errors": [{"line": 1, "column": 10, "stop_line": 1, "stop_column": 11, "path": "bad.py", "code": -2, "name": "bad-assignment", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_override/real_baseline.json && \
> echo 'baseline = "empty_baseline.json"' > $TMPDIR/baseline_override/pyrefly.toml && \
> cd $TMPDIR/baseline_override && $PYREFLY check --baseline=real_baseline.json
 INFO Checking project configured at `*/pyrefly.toml` (glob)
 INFO 0 errors
[0]
```

## Baseline path is resolved relative to config file

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_relative/subdir && \
> echo "b: str = 1" > $TMPDIR/baseline_relative/subdir/bad.py && \
> echo '{"errors": [{"line": 1, "column": 10, "stop_line": 1, "stop_column": 11, "path": "bad.py", "code": -2, "name": "bad-assignment", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_relative/my_baseline.json && \
> echo 'baseline = "my_baseline.json"' > $TMPDIR/baseline_relative/pyrefly.toml && \
> cd $TMPDIR/baseline_relative/subdir && $PYREFLY check bad.py
 INFO 0 errors
[0]
```

## Updating a baseline uses the path from pyproject.toml

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_update_from_pyproject && \
> echo "x: str = 1" > $TMPDIR/baseline_update_from_pyproject/bad.py && \
> printf '[tool.pyrefly]\nbaseline = "baseline.json"\n' > $TMPDIR/baseline_update_from_pyproject/pyproject.toml && \
> cd $TMPDIR/baseline_update_from_pyproject && \
> $PYREFLY check --update-baseline --output-format=min-text
ERROR *bad.py* ?bad-assignment? (glob)
[1]
```

```scrut {output_stream: stdout}
$ grep '"name": "bad-assignment"' $TMPDIR/baseline_update_from_pyproject/baseline.json
      "name": "bad-assignment",
[0]
```

The written baseline omits fields that are not used for matching.

```scrut {output_stream: stdout}
$ grep -cE '"(line|stop_line|stop_column|code|description)"' $TMPDIR/baseline_update_from_pyproject/baseline.json
0
[1]
```

## Updating a baseline requires a path from the CLI or configuration

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/baseline_update_without_path && \
> echo "x: str = 1" > $TMPDIR/baseline_update_without_path/bad.py && \
> touch $TMPDIR/baseline_update_without_path/pyrefly.toml && \
> cd $TMPDIR/baseline_update_without_path && \
> $PYREFLY check bad.py --update-baseline --summary=none
`--update-baseline` requires a baseline file set by `--baseline` or configuration
[1]
```

## `--update-baseline` populates an empty baseline file

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_update_empty && \
> echo "x: str = 1" > $TMPDIR/baseline_update_empty/bad.py && \
> printf '' > $TMPDIR/baseline_update_empty/baseline.json && \
> touch $TMPDIR/baseline_update_empty/pyrefly.toml && \
> cd $TMPDIR/baseline_update_empty && \
> $PYREFLY check bad.py --baseline=baseline.json --update-baseline --output-format=min-text
ERROR *bad.py* ?bad-assignment? (glob)
[1]
```

```scrut {output_stream: stdout}
$ grep '"name": "bad-assignment"' $TMPDIR/baseline_update_empty/baseline.json
      "name": "bad-assignment",
[0]
```

## Updating a baseline records unused `# type: ignore` errors

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_unused_type_ignore_update && \
> echo "# type: ignore" > $TMPDIR/baseline_unused_type_ignore_update/bad.py && \
> touch $TMPDIR/baseline_unused_type_ignore_update/pyrefly.toml && \
> cd $TMPDIR/baseline_unused_type_ignore_update && \
> $PYREFLY check --error=unused-type-ignore --baseline=baseline.json --update-baseline --output-format=min-text
ERROR *bad.py* ?unused-type-ignore? (glob)
[1]
```

```scrut {output_stream: stdout}
$ grep '"name": "unused-type-ignore"' $TMPDIR/baseline_unused_type_ignore_update/baseline.json
      "name": "unused-type-ignore",
[0]
```

## A baselined unused `# type: ignore` error is suppressed

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_unused_type_ignore_check && \
> echo "# type: ignore" > $TMPDIR/baseline_unused_type_ignore_check/bad.py && \
> echo '{"errors": [{"line": 1, "column": 1, "stop_line": 1, "stop_column": 2, "path": "bad.py", "code": -2, "name": "unused-type-ignore", "description": "test", "concise_description": "test"}]}' > $TMPDIR/baseline_unused_type_ignore_check/baseline.json && \
> touch $TMPDIR/baseline_unused_type_ignore_check/pyrefly.toml && \
> cd $TMPDIR/baseline_unused_type_ignore_check && \
> $PYREFLY check --error=unused-type-ignore --baseline=baseline.json --output-format=min-text
[0]
```

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
it from the current error. Re-serialization may normalize formatting and fields.

```scrut {output_stream: stdout}
$ grep -c '"concise_description": "test"' $TMPDIR/baseline_prune/baseline.json
1
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
