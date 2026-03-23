#!/usr/bin/env bash
# monitor_run.sh — Watch a live Explorer run and detect stuck loops.
#
# Usage:
#   WATCH_DIR=$HOME/git/schrodinger-qwen bash scripts/monitor_run.sh
#   WATCH_DIR=$HOME/git/schrodinger-qwen INTERVAL=300 nohup bash scripts/monitor_run.sh &
#
# Environment:
#   WATCH_DIR   — directory containing chat_output/ (required)
#   INTERVAL    — seconds between checks (default: 900 = 15 min)
#   LOG_FILE    — path to log file (default: $WATCH_DIR/monitor.log)

set -uo pipefail

: "${WATCH_DIR:?Set WATCH_DIR to the run directory (e.g. \$HOME/git/schrodinger-qwen)}"
: "${INTERVAL:=900}"

CHAT_DIR="$WATCH_DIR/chat_output"
LOG_FILE="${LOG_FILE:-$WATCH_DIR/monitor.log}"

prev_error_sig=""
error_repeat=0
prev_iter=0

log() {
    local ts
    ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE"
}

# Return the latest file matching a glob, or empty string if none.
latest_file() {
    local pattern="$1"
    # shellcheck disable=SC2086
    ls -1t $pattern 2>/dev/null | head -1
}

# Count iterations by counting verifier files.
count_iterations() {
    # shellcheck disable=SC2086
    ls -1 "$CHAT_DIR"/*_verifier.md 2>/dev/null | wc -l | tr -d ' '
}

# Extract error summary lines from a verifier file.
extract_errors() {
    local vfile="$1"
    # Grab FAILED lines, collection errors, and structural issues
    grep -E 'FAILED|Import mismatch|only defines|ModuleNotFoundError|No code drafts|0 passed|collected 0 items|SyntaxError|IndentationError' "$vfile" 2>/dev/null || true
}

# Compute a simple signature of the errors (sorted, for comparison).
error_signature() {
    local vfile="$1"
    extract_errors "$vfile" | sort | md5sum | cut -d' ' -f1
}

check_once() {
    if [[ ! -d "$CHAT_DIR" ]]; then
        log "WAITING: $CHAT_DIR does not exist yet"
        return
    fi

    local iter
    iter=$(count_iterations)

    # If no verifier files yet, report and return.
    if [[ "$iter" -eq 0 ]]; then
        local file_count
        file_count=$(ls -1 "$CHAT_DIR"/*.md 2>/dev/null | wc -l | tr -d ' ')
        log "RUNNING: iteration 0 in progress ($file_count files so far)"
        return
    fi

    # Find the latest verifier file.
    local latest_v
    latest_v=$(latest_file "$CHAT_DIR/*_verifier.md")

    if [[ -z "$latest_v" ]]; then
        log "RUNNING: no verifier output yet"
        return
    fi

    local basename_v
    basename_v=$(basename "$latest_v")

    # Check if tests passed.
    if grep -q '✅ All tests passed' "$latest_v"; then
        # Check if there's a summary.json (run finished).
        if [[ -f "$CHAT_DIR/summary.json" ]]; then
            log "FINISHED: all tests passed at iteration $iter ✅  (summary.json present)"
        else
            log "PASSING: iteration $iter — tests pass, waiting for next phase or completion"
        fi
        prev_error_sig=""
        error_repeat=0
        prev_iter="$iter"
        return
    fi

    # Tests failed — extract details.
    local errors
    errors=$(extract_errors "$latest_v")
    local error_count
    error_count=$(echo "$errors" | grep -c 'FAILED' || true)
    local first_error
    first_error=$(echo "$errors" | head -1)

    # Check for specific anti-patterns.
    if echo "$errors" | grep -qi 'Import mismatch\|only defines'; then
        log "STRUCTURAL: import consistency error at iteration $iter — $first_error"
    elif echo "$errors" | grep -qi 'ModuleNotFoundError'; then
        log "STRUCTURAL: ModuleNotFoundError at iteration $iter — $first_error"
    elif echo "$errors" | grep -qi 'SyntaxError\|IndentationError'; then
        log "SYNTAX: code has syntax errors at iteration $iter — $first_error"
    elif echo "$errors" | grep -qi 'collected 0 items'; then
        log "BROKEN: no tests collected at iteration $iter — $first_error"
    elif echo "$errors" | grep -qi 'No code drafts\|0 passed, 0 failed\|no tests ran'; then
        log "BROKEN: no tests found at iteration $iter"
    fi

    # Stuck-loop detection: compare error signatures.
    local sig
    sig=$(error_signature "$latest_v")

    if [[ "$sig" == "$prev_error_sig" && "$iter" -ne "$prev_iter" ]]; then
        error_repeat=$((error_repeat + 1))
    elif [[ "$iter" -ne "$prev_iter" ]]; then
        error_repeat=1
    fi
    prev_error_sig="$sig"
    prev_iter="$iter"

    if [[ "$error_repeat" -ge 3 ]]; then
        log "⚠️  STUCK: identical error signature for $error_repeat consecutive iterations (iter $iter) — $first_error"
    else
        log "FAILING: iteration $iter — $error_count test(s) failed — $first_error"
    fi
}

# --- Main loop ---
log "=== Monitor started: watching $WATCH_DIR (interval: ${INTERVAL}s) ==="

while true; do
    check_once
    sleep "$INTERVAL"
done
