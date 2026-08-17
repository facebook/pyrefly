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

## Baselined findings can be emitted at a configured severity

```scrut
$ mkdir -p $TMPDIR/baseline_levels && \
> printf 'x: str = 1\n' > $TMPDIR/baseline_levels/matched.py && \
> printf 'def f() -> str:\n    return 1\n' > $TMPDIR/baseline_levels/new.py && \
> echo '{"errors":[{"column":10,"path":"matched.py","name":"bad-assignment","concise_description":"test","severity":"error"}]}' > $TMPDIR/baseline_levels/baseline.json && \
> printf 'baseline = "baseline.json"\nbaseline-error-level = "warn"\n' > $TMPDIR/baseline_levels/pyrefly.toml
[0]
```

The configured warning is hidden by the default error threshold.

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check matched.py --summary=none --output-format=min-text
[0]
```

At a matching threshold it is reported, marked as baselined, and affects the
exit status normally.

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check matched.py --min-severity=warn --summary=none --output-format=min-text
 WARN matched.py:1:10-11: * [bad-assignment] [baselined] (glob)
[1]
```

The CLI level overrides configuration. `info` is likewise hidden until the
minimum severity includes it.

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check matched.py --baseline-error-level=info --summary=none --output-format=min-text
[0]
```

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check matched.py --baseline-error-level=info --min-severity=info --summary=none --output-format=min-text
 INFO matched.py:1:10-11: * [bad-assignment] [baselined] (glob)
[1]
```

Explicit `ignore` hides matching findings while new errors are reported.

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --baseline-error-level=ignore --summary=none --output-format=min-text
ERROR new.py:2:12-13: * [bad-return] (glob)
[1]
```

## Baseline error levels do not increase a finding's severity

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_warning && \
> printf 'x: str = 1\n' > $TMPDIR/baseline_warning/warning.py && \
> printf 'baseline = "baseline.json"\n[errors]\nbad-assignment = "warn"\n' > $TMPDIR/baseline_warning/pyrefly.toml && \
> cd $TMPDIR/baseline_warning && \
> $PYREFLY check --update-baseline --min-severity=warn --summary=none --output-format=omit-errors >/dev/null 2>/dev/null; \
> $JQ -r '.errors[0].severity' baseline.json
warn
[0]
```

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_warning && \
> $PYREFLY check warning.py --baseline-error-level=error --min-severity=warn --summary=none --output-format=min-text
 WARN warning.py:1:10-11: * [bad-assignment] [baselined] (glob)
[1]
```

## `--only` filters baselined and new findings

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --baseline-error-level=error --only=bad-assignment --summary=none --output-format=min-text
ERROR matched.py:1:10-11: * [bad-assignment] [baselined] (glob)
[1]
```

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --baseline-error-level=error --only=bad-return --summary=none --output-format=min-text
ERROR new.py:2:12-13: * [bad-return] (glob)
[1]
```

## JSON records baseline provenance only when a baseline is configured

```scrut {output_stream: stdout}
$ cd $TMPDIR/no_baseline && \
> $PYREFLY check --summary=none --output=json:diagnostics.json --output=sarif:diagnostics.sarif >/dev/null 2>/dev/null; \
> $JQ -c '[.errors[] | has("baselined")]' diagnostics.json && \
> $JQ -c '[.runs[0].results[] | has("baselineState")]' diagnostics.sarif
[false]
[false]
[0]
```

With a loaded baseline, matched and new findings are both identified. Using
the `error` level reports both at full severity.

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --baseline-error-level=error --summary=none --output=json:diagnostics.json >/dev/null 2>/dev/null; \
> $JQ -c '[.errors[] | {path, severity, baselined}]' diagnostics.json
[{"path":"matched.py","severity":"error","baselined":true},{"path":"new.py","severity":"error","baselined":false}]
[0]
```

## SARIF records matching findings as unchanged and other findings as new

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --baseline-error-level=error --summary=none --output=sarif:diagnostics.sarif >/dev/null 2>/dev/null; \
> $JQ -c '[.runs[0].results[] | {path: .locations[0].physicalLocation.artifactLocation.uri, baselineState}]' diagnostics.sarif
[{"path":"matched.py","baselineState":"unchanged"},{"path":"new.py","baselineState":"new"}]
[0]
```

## Baseline regeneration reports configured but uncompared provenance

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_regeneration && \
> printf 'x: str = 1\n' > $TMPDIR/baseline_regeneration/bad.py && \
> touch $TMPDIR/baseline_regeneration/pyrefly.toml && \
> cd $TMPDIR/baseline_regeneration && \
> $PYREFLY check --baseline=baseline.json --update-baseline --summary=none --output=json:diagnostics.json --output=sarif:diagnostics.sarif >/dev/null 2>/dev/null; \
> $JQ -c '[.errors[] | .baselined]' diagnostics.json && \
> $JQ -c '[.runs[0].results[] | has("baselineState")]' diagnostics.sarif
[false]
[false]
[0]
```

## GitHub Actions annotations mark baselined findings in their title

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --baseline-error-level=warn --min-severity=warn --summary=none --output-format=github
::warning file=*/matched.py,line=1,col=10,endLine=1,endColumn=11,title=Pyrefly bad-assignment [baselined]::* (glob)
::error file=*/new.py,line=2,col=12,endLine=2,endColumn=13,title=Pyrefly bad-return::* (glob)
[1]
```

## Omitted-error output reports the number of baselined diagnostics

```scrut {output_stream: stderr}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check matched.py --baseline-error-level=warn --min-severity=warn --output-format=omit-errors --progress-bar=no
 INFO 1 diagnostic (1 baselined)
[1]
```

## Display metadata is not written into an updated baseline

```scrut {output_stream: stdout}
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check --update-baseline --min-severity=warn --summary=none --output-format=omit-errors >/dev/null 2>/dev/null; \
> $JQ -c '[.errors[] | {severity, baselined: has("baselined")}]' baseline.json
[{"severity":"error","baselined":false},{"severity":"error","baselined":false}]
[0]
```

## Pruning writes stored baseline severity and omits provenance

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/baseline_prune_provenance && \
> printf 'x: str = 1\n' > $TMPDIR/baseline_prune_provenance/matched.py && \
> echo '{"errors":[{"column":10,"path":"matched.py","name":"bad-assignment","concise_description":"test","severity":"info"},{"column":1,"path":"gone.py","name":"bad-return","concise_description":"stale","severity":"error"}]}' > $TMPDIR/baseline_prune_provenance/baseline.json && \
> touch $TMPDIR/baseline_prune_provenance/pyrefly.toml && \
> cd $TMPDIR/baseline_prune_provenance && \
> $PYREFLY check matched.py --baseline=baseline.json --baseline-error-level=warn --prune-baseline --summary=none --output-format=omit-errors >/dev/null 2>/dev/null; \
> $JQ -c '[.errors[] | {severity, baselined: has("baselined")}]' baseline.json
[{"severity":"info","baselined":false}]
[0]
```

## Baselined display findings are not converted into inline suppressions

```scrut
$ cd $TMPDIR/baseline_levels && \
> $PYREFLY check matched.py --baseline-error-level=error --only=bad-assignment --suppress-errors --summary=none >/dev/null 2>/dev/null; \
> ! grep -q pyrefly matched.py
[0]
```
