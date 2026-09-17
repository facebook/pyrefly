# Bazel Check CLI Tests

## Main help lists bazel-check

```scrut
$ $PYREFLY --help | grep -F "bazel-check"
*bazel-check* (glob)
[0]
```

## Direct root error includes target in stderr and JSON

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_error/pkg && cd $TMPDIR/bazel_error && \
> echo "x: int = 'bad'" > pkg/app.py && \
> printf '%s' '{"target":{"label":"//pkg:app","workspace_name":"_main","package":"pkg","name":"app","rule_kind":"py_binary"},"check_roots":{"sources":["pkg/app.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"config":{"python_version":"3.12","system_platform":"linux"}}' > input.json && \
> $PYREFLY bazel-check --output out.json input.json > stdout.txt 2> stderr.txt; rc=$?; \
> test ! -s stdout.txt && \
> cat stderr.txt && \
> $JQ -r '[(.diagnostics | length), .diagnostics[0].target, .diagnostics[0].path, .diagnostics[0].name] | join(" ")' out.json; \
> exit $rc
Target //pkg:app

ERROR * [bad-assignment] (glob)
*pkg/app.py* (glob)
* (glob+)
1 //pkg:app pkg/app.py bad-assignment
[1]
```

## Root workspace canonical target label is normalized

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_canonical_label && cd $TMPDIR/bazel_canonical_label && \
> echo "x: int = 'bad'" > app.py && \
> printf '%s' '{"target":{"label":"@@//:app","workspace_name":"_main","package":"","name":"app","rule_kind":"py_binary"},"check_roots":{"sources":["app.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"config":{"python_version":"3.12","system_platform":"linux"}}' > input.json && \
> $PYREFLY bazel-check --output out.json input.json > stdout.txt 2> stderr.txt; rc=$?; \
> test ! -s stdout.txt && \
> cat stderr.txt && \
> $JQ -r '.diagnostics[0].target' out.json; \
> exit $rc
Target //:app

ERROR * [bad-assignment] (glob)
*app.py* (glob)
* (glob+)
//:app
[1]
```

## Dependency files are importable but not checked as roots

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_dep/pkg && cd $TMPDIR/bazel_dep && \
> echo "import pkg.dep" > pkg/app.py && \
> echo "x: int = 'bad'" > pkg/dep.py && \
> printf '%s' '{"target":{"label":"//pkg:app","workspace_name":"_main","package":"pkg","name":"app","rule_kind":"py_binary"},"check_roots":{"sources":["pkg/app.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"config":{"python_version":"3.12","system_platform":"linux"}}' > input.json && \
> $PYREFLY bazel-check --output out.json input.json > stdout.txt 2> stderr.txt && \
> test ! -s stdout.txt && test ! -s stderr.txt && \
> $JQ '.diagnostics | length' out.json
0
[0]
```

## Dependency type can produce a root diagnostic

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_dep_type/pkg && cd $TMPDIR/bazel_dep_type && \
> echo "import pkg.dep; y: int = pkg.dep.x" > pkg/app.py && \
> echo "x: str = 'typed'" > pkg/dep.py && \
> printf '%s' '{"target":{"label":"//pkg:app","workspace_name":"_main","package":"pkg","name":"app","rule_kind":"py_binary"},"check_roots":{"sources":["pkg/app.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"config":{"python_version":"3.12","system_platform":"linux"}}' > input.json && \
> $PYREFLY bazel-check --output out.json input.json > stdout.txt 2> stderr.txt; rc=$?; \
> test ! -s stdout.txt && \
> $JQ -r '[(.diagnostics | length), .diagnostics[0].path, .diagnostics[0].name] | join(" ")' out.json; \
> exit $rc
1 pkg/app.py bad-assignment
[1]
```

## Generated check root reports physical path

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_generated/bazel-out/darwin-fastbuild/bin && cd $TMPDIR/bazel_generated && \
> echo "x: int = 'bad'" > bazel-out/darwin-fastbuild/bin/generated.py && \
> printf '%s' '{"target":{"label":"//pkg:generated","workspace_name":"_main","package":"pkg","name":"generated","rule_kind":"py_library"},"check_roots":{"sources":["generated.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"path_overlays":[{"short_path":"generated.py","path":"bazel-out/darwin-fastbuild/bin/generated.py"}],"config":{"python_version":"3.12","system_platform":"linux"}}' > input.json && \
> $PYREFLY bazel-check --output out.json input.json > stdout.txt 2> stderr.txt; rc=$?; \
> test ! -s stdout.txt && \
> cat stderr.txt && \
> $JQ -r '[(.diagnostics | length), .diagnostics[0].path, .diagnostics[0].target] | join(" ")' out.json; \
> exit $rc
Target //pkg:generated

ERROR * [bad-assignment] (glob)
*bazel-out/darwin-fastbuild/bin/generated.py* (glob)
* (glob+)
1 bazel-out/darwin-fastbuild/bin/generated.py //pkg:generated
[1]
```

## Warning threshold controls output and exit status

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_warn/pkg && cd $TMPDIR/bazel_warn && \
> echo "x: int = 'bad'" > pkg/app.py && \
> printf '%s' '{"target":{"label":"//pkg:app","workspace_name":"_main","package":"pkg","name":"app","rule_kind":"py_binary"},"check_roots":{"sources":["pkg/app.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"config":{"python_version":"3.12","system_platform":"linux","error_severities":{"bad-assignment":"warn"}}}' > input.json && \
> $PYREFLY bazel-check --output default.json input.json > default.stdout 2> default.stderr && \
> test ! -s default.stdout && test ! -s default.stderr && \
> $JQ '.diagnostics | length' default.json && \
> $PYREFLY bazel-check --min-severity warn --output warn.json input.json > warn.stdout 2> warn.stderr; rc=$?; \
> test ! -s warn.stdout && \
> $JQ -r '[(.diagnostics | length), .diagnostics[0].name, .diagnostics[0].severity] | join(" ")' warn.json; \
> exit $rc
0
1 bad-assignment warn
[1]
```

## Directive output survives the default severity threshold

```scrut {output_stream: stdout}
$ mkdir -p $TMPDIR/bazel_reveal/pkg && cd $TMPDIR/bazel_reveal && \
> echo "from typing import reveal_type; reveal_type(1)" > pkg/app.py && \
> printf '%s' '{"target":{"label":"//pkg:app","workspace_name":"_main","package":"pkg","name":"app","rule_kind":"py_binary"},"check_roots":{"sources":["pkg/app.py"]},"search_path":{"workspace_name":"_main","python_import_all_repositories":false},"config":{"python_version":"3.12","system_platform":"linux"}}' > input.json && \
> $PYREFLY bazel-check --output out.json input.json > stdout.txt 2> stderr.txt && \
> test ! -s stdout.txt && \
> cat stderr.txt && \
> $JQ -r '[(.diagnostics | length), .diagnostics[0].path, .diagnostics[0].name] | join(" ")' out.json
Target //pkg:app

 INFO revealed type: Literal[1] [reveal-type]
*pkg/app.py* (glob)
* (glob+)
1 pkg/app.py reveal-type
[0]
```
