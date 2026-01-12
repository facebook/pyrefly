# Tests for interpreter selection

## Interpreter priority takes CLI interpreter

```scrut {output_stream: stdout}
$ mkdir $TMPDIR/interpreters && touch $TMPDIR/interpreters/test.py \
> touch $TMPDIR/test-interpreter && \
> echo 'python-interpreter = "$TMPDIR/test-interpreter"' > $TMPDIR/interpreters/pyrefly.toml && \
> mkdir $TMPDIR/interpreters/venv && touch $TMPDIR/interpreters/venv/python3 && \
> touch $TMPDIR/interpreters/venv/pyvenv.cfg && \
> mkdir $TMPDIR/alternative-venv && touch $TMPDIR/alternative-venv/python3 && \
> touch $TMPDIR/alternative-venv/pyvenv.cfg && \
> VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml \
> --python-interpreter-path "cli-interpreter"
Configuration at * (glob)
  Using interpreter: cli-interpreter
* (glob+)
[0]
```

## Interpreter priority takes config-file interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */test-interpreter (glob)
* (glob+)
[0]
```

## Interpreter priority takes activated interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ echo "" > $TMPDIR/interpreters/pyrefly.toml && \
> VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */alternative-venv/python3 (glob)
* (glob+)
[0]
```

## Interpreter priority takes venv interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ echo "" > $TMPDIR/interpreters/pyrefly.toml && \
> $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */interpreters/venv/python3 (glob)
* (glob+)
[0]
```

## Interpreter priority takes system interpreter last

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ rm -rf $TMPDIR/interpreters/venv && \
> $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: /*/python3 (glob)
* (glob+)
[0]
```
