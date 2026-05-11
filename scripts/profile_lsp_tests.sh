#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Strobelight profiling wrapper for Pyrefly LSP interaction tests
#
# This script builds and profiles Pyrefly's lsp_interaction tests using strobelight.
# It handles the complex setup required to run Buck-built Rust tests with profiling.

# Exit on error (-e), undefined variables (-u), and pipe failures (-o pipefail)
set -euo pipefail

# Default configuration
DURATION_MS=10000
EVENT="cycles"
VERBOSE=false
BUILD_ONLY=false
LIST_TESTS=false
TEST_FILTER=""

# Target for the test binary
TEST_TARGET="fbcode//pyrefly/pyrefly:pyrefly_library-unittest"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [TEST_FILTER]

Profile Pyrefly LSP interaction tests using strobelight.

Arguments:
  TEST_FILTER           Optional test name filter (e.g., 'test_initialize_basic', 'hover')
                        Matches tests containing this string. Default: all lsp_interaction tests.

Options:
  -d, --duration-ms MS  Profiling duration in milliseconds (default: 10000)
  -e, --event EVENT     Strobelight event to profile (default: cycles)
                        Common events: cycles, instructions, LLC-load-misses, cache-misses
  -v, --verbose         Enable verbose output for debugging
  --build-only          Build the test binary without running profiling
  --list                List available lsp_interaction tests
  -h, --help            Show this help message

Examples:
  # Profile all lsp_interaction tests
  $(basename "$0")

  # Profile specific test for 30 seconds
  $(basename "$0") -d 30000 test_initialize_basic

  # Profile with cache miss events
  $(basename "$0") -e LLC-load-misses hover

  # List available tests
  $(basename "$0") --list

  # Build only (for debugging)
  $(basename "$0") --build-only
EOF
}

log() {
    echo "[profile_lsp_tests] $*" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[profile_lsp_tests:verbose] $*" >&2
    fi
}

error() {
    echo "[profile_lsp_tests:ERROR] $*" >&2
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--duration-ms)
            DURATION_MS="$2"
            shift 2
            ;;
        -e|--event)
            EVENT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --list)
            LIST_TESTS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            TEST_FILTER="$1"
            shift
            ;;
    esac
done

# Change to fbcode directory for Buck commands
cd "$(dirname "$0")/../.."
FBCODE_DIR="$(pwd)"
REPO_ROOT="$(dirname "$FBCODE_DIR")"
log_verbose "Working directory: $FBCODE_DIR"
log_verbose "Repository root: $REPO_ROOT"

# Build the test binary
log "Building test binary..."
BUILD_OUTPUT=$(buck2 build @fbcode//mode/opt "$TEST_TARGET" --show-output 2>&1)
log_verbose "Build output: $BUILD_OUTPUT"

# Extract the binary path from build output
# Format: "fbcode//pyrefly/pyrefly:pyrefly_library-unittest buck-out/v2/gen/fbcode/.../pyrefly"
BINARY_PATH=$(echo "$BUILD_OUTPUT" | grep -E "^$TEST_TARGET" | awk '{print $2}')

if [[ -z "$BINARY_PATH" ]]; then
    error "Failed to extract binary path from build output"
fi

# Convert to absolute path (buck-out is at repo root, not fbcode)
if [[ ! "$BINARY_PATH" = /* ]]; then
    BINARY_PATH="$REPO_ROOT/$BINARY_PATH"
fi

log "Test binary: $BINARY_PATH"

if [[ ! -x "$BINARY_PATH" ]]; then
    error "Binary not found or not executable: $BINARY_PATH"
fi

# Find the test_env.json file
TEST_DIR=$(dirname "$BINARY_PATH")
TEST_ENV_FILE="$TEST_DIR/test_env.json"

log_verbose "Looking for test_env.json at: $TEST_ENV_FILE"

if [[ ! -f "$TEST_ENV_FILE" ]]; then
    # Try alternative location patterns
    TEST_ENV_FILE=$(find "$(dirname "$TEST_DIR")" -name "test_env.json" -type f 2>/dev/null | head -1)
    if [[ -z "$TEST_ENV_FILE" || ! -f "$TEST_ENV_FILE" ]]; then
        error "Could not find test_env.json"
    fi
fi

log_verbose "Found test_env.json: $TEST_ENV_FILE"

# Load environment variables from test_env.json
log "Loading test environment..."
while IFS="=" read -r key value; do
    export "$key"="$value"
    log_verbose "  $key=$value"
done < <(python3 -c "
import json
import sys

with open('$TEST_ENV_FILE') as f:
    env = json.load(f)

for key, value in env.items():
    print(f'{key}={value}')
")

# Set up LD_LIBRARY_PATH for shared library dependencies.
# Buck builds Rust tests with dynamically linked dependencies (e.g., Python libraries)
# that must be discoverable at runtime when running outside of buck test.
SHARED_LIBS_DIR="$TEST_DIR/__pyrefly__shared_libs_symlink_tree"
if [[ -d "$SHARED_LIBS_DIR" ]]; then
    export LD_LIBRARY_PATH="${SHARED_LIBS_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log_verbose "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
    log_verbose "Shared libs directory not found: $SHARED_LIBS_DIR (may not be needed)"
fi

if [[ "$BUILD_ONLY" == "true" ]]; then
    log "Build complete (--build-only specified)"
    log "Binary: $BINARY_PATH"
    log "Test env: $TEST_ENV_FILE"
    exit 0
fi

# Build the test filter for lsp_interaction tests
# Rust test filters use substring matching, not regex
# Full test paths look like: test::lsp::lsp_interaction::basic::test_initialize_basic
if [[ -n "$TEST_FILTER" ]]; then
    # Just use the filter directly - Rust will do substring matching
    # This allows matching by module (e.g., "hover") or test name (e.g., "test_initialize_basic")
    RUST_TEST_FILTER="$TEST_FILTER"
else
    # Run all lsp_interaction tests
    RUST_TEST_FILTER="lsp_interaction::"
fi

if [[ "$LIST_TESTS" == "true" ]]; then
    log "Available lsp_interaction tests:"
    "$BINARY_PATH" --list 2>/dev/null | grep "lsp_interaction::" | sed 's/: test$//' | sort
    exit 0
fi

# Run with strobelight profiling
log "Starting strobelight profiling..."
log "  Duration: ${DURATION_MS}ms"
log "  Event: $EVENT"
log "  Test filter: $RUST_TEST_FILTER"

strobe bpf \
    --duration-ms "$DURATION_MS" \
    --event "$EVENT" \
    --exec "$BINARY_PATH" \
    -- "$RUST_TEST_FILTER"

log "Profiling complete. Check the Scuba URL above for results."
