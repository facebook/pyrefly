#!/bin/bash
# Verification script for tensor shape model ports.
# Run on a model file to check for common issues.
# Usage: verify_port.sh path/to/model.py

set -euo pipefail

FILE="${1:?Usage: verify_port.sh <model_file.py>}"

if [[ ! -f "$FILE" ]]; then
    echo "File not found: $FILE"
    exit 1
fi

echo "=== Verifying shape port: $FILE ==="
echo ""

# Helper: count matching lines (avoids grep -c exit code issues)
count_matches() {
    grep -c "$@" 2>/dev/null || true
}

# 1. Count type: ignore
IGNORE_COUNT=$(count_matches 'type: ignore\[' "$FILE")
if [[ "$IGNORE_COUNT" -gt 0 ]]; then
    echo "⚠️  $IGNORE_COUNT type: ignore found. Audit each one:"
    grep -n 'type: ignore\[' "$FILE" | while read -r line; do
        echo "   $line"
    done
    echo "   → Is each one A1 (algebraic)? Or could you fix a stub instead?"
    echo ""
fi

# 2. Find bare Tensor in signatures (params and returns)
# Matches ": Tensor" not followed by "[" — captures bare annotations
BARE_SIG=$(grep -nE ':\s*Tensor\s*[=\),]|:\s*Tensor\s*$|->\s*Tensor\s*[:\|]|->\s*Tensor\s*$' "$FILE" 2>/dev/null || true)
BARE_COUNT=0
BARE_DEFS=""
BARE_LOCALS=""
if [[ -n "$BARE_SIG" ]]; then
    BARE_COUNT=$(echo "$BARE_SIG" | wc -l)
    # Separate into signature-level (def lines) and local variables
    BARE_DEFS=$(echo "$BARE_SIG" | grep -E 'def |self,' || true)
    BARE_LOCALS=$(echo "$BARE_SIG" | grep -vE 'def |self,' || true)
    BARE_DEF_COUNT=0
    BARE_LOCAL_COUNT=0
    if [[ -n "$BARE_DEFS" ]]; then
        BARE_DEF_COUNT=$(echo "$BARE_DEFS" | wc -l)
        echo "⚠️  $BARE_DEF_COUNT bare Tensor in signatures (params/returns):"
        echo "$BARE_DEFS" | while read -r line; do
            echo "   $line"
        done
        echo "   → Can you type these? Shapes should be known from the model."
        echo ""
    fi
    if [[ -n "$BARE_LOCALS" ]]; then
        BARE_LOCAL_COUNT=$(echo "$BARE_LOCALS" | wc -l)
        echo "⚠️  $BARE_LOCAL_COUNT bare Tensor in local variables:"
        echo "$BARE_LOCALS" | while read -r line; do
            echo "   $line"
        done
        echo "   → Is the shape genuinely unknowable? Or can you annotate?"
        echo ""
    fi
fi

# 3. Find int params that might should be Dim
INT_PARAMS=$(grep -n 'def __init__' "$FILE" 2>/dev/null | grep -E ':\s*int\s*[,=)]' || true)
if [[ -n "$INT_PARAMS" ]]; then
    echo "ℹ️  int params in constructors (check if any should be Dim):"
    echo "$INT_PARAMS" | while read -r line; do
        echo "   $line"
    done
    echo ""
fi

# 4. Check for assert_type usage — shaped vs bare
ASSERT_COUNT=$(count_matches 'assert_type' "$FILE")
if [[ "$ASSERT_COUNT" -eq 0 ]]; then
    echo "⚠️  No assert_type calls found. Add shape verification checkpoints."
    echo ""
else
    # Shaped: assert_type(x, Tensor[...])
    SHAPED_COUNT=$(count_matches 'assert_type(.*Tensor\[' "$FILE")
    # Bare: assert_type(x, Tensor) without [
    BARE_ASSERT=$(grep -n 'assert_type(.*Tensor)' "$FILE" 2>/dev/null | grep -v 'Tensor\[' || true)
    BARE_ASSERT_COUNT=0
    if [[ -n "$BARE_ASSERT" ]]; then
        BARE_ASSERT_COUNT=$(echo "$BARE_ASSERT" | wc -l)
    fi
    echo "✓ $ASSERT_COUNT assert_type checkpoints ($SHAPED_COUNT shaped, $BARE_ASSERT_COUNT bare)."
    if [[ "$BARE_ASSERT_COUNT" -gt 0 ]]; then
        echo ""
        echo "  Bare assert_type (tracking gaps):"
        echo "$BARE_ASSERT" | while read -r line; do
            LINE_NUM=$(echo "$line" | cut -d: -f1)
            # Check if the line has a comment explaining the root cause
            if echo "$line" | grep -q '#'; then
                echo "   ✓ $line"
            else
                echo "   ⚠️  $line  — MISSING root cause comment"
            fi
        done
    fi
    echo ""
fi

# 5. Check for smoke tests
TEST_COUNT=$(count_matches 'def test_' "$FILE")
if [[ "$TEST_COUNT" -eq 0 ]]; then
    echo "⚠️  No smoke tests (def test_*) found."
    echo ""
else
    echo "✓ $TEST_COUNT smoke tests found."
    echo ""
fi

# 6. Check for exclusion markers
EXCL_COUNT=$(count_matches -iE 'excl|excluded|not included|not ported|omitted' "$FILE")
if [[ "$EXCL_COUNT" -gt 0 ]]; then
    echo "⚠️  $EXCL_COUNT exclusion markers found:"
    grep -niE 'excl|excluded|not included|not ported|omitted' "$FILE" | while read -r line; do
        echo "   $line"
    done
    echo "   → Every class/method in the original must be in the port."
    echo ""
fi

# 7. Count classes and methods for manual comparison
CLASS_COUNT=$(count_matches '^class ' "$FILE")
METHOD_COUNT=$(count_matches '^\s*def ' "$FILE")
echo "ℹ️  Structure: $CLASS_COUNT classes, $METHOD_COUNT methods (compare with original)."
echo ""

# Summary
echo "=== Summary ==="
echo "  type: ignore:        $IGNORE_COUNT"
echo "  bare Tensor (sig):   $(echo "$BARE_DEFS" 2>/dev/null | grep -c . || echo 0)"
echo "  bare Tensor (var):   $(echo "$BARE_LOCALS" 2>/dev/null | grep -c . || echo 0)"
echo "  assert_type:         $ASSERT_COUNT (${SHAPED_COUNT:-0} shaped, ${BARE_ASSERT_COUNT:-0} bare)"
echo "  smoke tests:         $TEST_COUNT"
