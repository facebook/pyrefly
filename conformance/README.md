This folder contains the Python typing conformance test suite.

The `src/` directory contains several types of files:

- the conformance test sources, `.py` and `.pyi` files
- what pyrefly outputs for each conformance test: `conformance.exp`
- comparison of pyrefly's outputs with the expected errors for each conformance
  test: `conformance.result`
- high-level summary: `results.json`

Commands (from this directory):

GitHub developers:

- Update conformance test source files: `update_conformance_sources.sh`
- Update conformance test outputs:
  `cargo build && python3 conformance_output.py --executable ../target/debug/pyrefly ./third_party`
- Check conformance test outputs are up-to-date:
  `cargo build && python3 conformance_output.py --executable ../target/debug/pyrefly --mode check ./third_party`
- Check conformance test results against expected results:
  `cargo build && python3 conformance_output.py --executable ../target/debug/pyrefly --mode compare ./third_party`
  (emits JSON to stdout)

NOTE: `conformance_output.py` requires Python 3.9+.

Meta internal developers:

- Update conformance test source files: `update_conformance_sources.sh`
- Update conformance test outputs:
  `buck2 run :conformance_output_script -- ./third_party`
- Check conformance test outputs are up-to-date:
  `buck2 run :conformance_output_script -- --mode check ./third_party` (faster),
  or `buck2 test :conformance_output_test` (slower)
- Check conformance test results against expected results:
  `buck2 run :conformance_output_script -- --mode compare ./third_party` (emits
  JSON to stdout)
