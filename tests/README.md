# Synthetic Performance Tests

This directory contains synthetic Python workloads used to stress-test Pyrefly indexing and type-solving throughput.

## Structure

- `perf/scale_test/`: Generated "synthetic enterprise" package with dense type annotations, deep inheritance, and import meshes.

## Scale Test Goals

The scale test workload is designed to apply pressure to:

1. Deep recursive solving paths in inheritance hierarchies.
2. Generic and protocol-heavy type relationships.
3. Inter-module dependency graph construction.
4. File-volume and line-volume indexing throughput.

## Generator

The scale test modules are produced by:

- `perf/scale_test/generate_stress_test.py`

Run from repository root:

```bash
/Users/atharvranjan/pyrefly-1/.venv/bin/python tests/perf/scale_test/generate_stress_test.py --core-only
```

Generate full suite (default: 120 modules, min 520 lines each):

```bash
/Users/atharvranjan/pyrefly-1/.venv/bin/python tests/perf/scale_test/generate_stress_test.py
```

Customize generation:

```bash
/Users/atharvranjan/pyrefly-1/.venv/bin/python tests/perf/scale_test/generate_stress_test.py \
  --module-count 150 \
  --min-lines 700
```

## Expected Core Outputs

Running with `--core-only` emits the initial five baseline modules:

- `base_types.py`
- `data_layer.py`
- `service_mesh.py`
- `domain_models.py`
- `orchestration.py`

All generated files are valid Python and intentionally large to provide stable stress-testing inputs.
