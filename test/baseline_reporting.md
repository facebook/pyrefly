# Tests for reporting baselined diagnostics

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
$ mkdir -p $TMPDIR/no_baseline && \
> echo "z: str = 1" > $TMPDIR/no_baseline/bad.py && \
> touch $TMPDIR/no_baseline/pyrefly.toml && \
> cd $TMPDIR/no_baseline && \
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
