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
