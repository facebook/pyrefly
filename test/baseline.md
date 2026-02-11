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
