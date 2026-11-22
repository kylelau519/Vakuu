#!/bin/bash
#
# Runs a training loop (train_alternating.sh, train_claws.sh, ...) under a hard wall-clock limit:
# force-kills the whole process tree and restarts every WATCHDOG_TIMEOUT seconds, no matter what
# state it's in. This exists because the loop has a known, not-yet-root-caused hang somewhere inside
# a round; losing an in-flight round's data is fine (next round regenerates it), so this favors a
# periodic hard reset over trying to detect the specific hang.
#
# Usage: scripts/watchdog.sh [script-to-supervise] [timeout-seconds]
#   defaults: scripts/train_alternating.sh, 1800 (30 min). A bare script name is looked up in scripts/.
#
# Run it from an environment where `python` already has the dependencies (activate .venv first, or
# put it on PATH): the supervised script inherits PATH.

set -m  # job control on, so the backgrounded script becomes its own process group leader

vakuu="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -x "$vakuu/.venv/bin/python" ] && export PATH="$vakuu/.venv/bin:$PATH"

SCRIPT="${1:-$vakuu/scripts/train_alternating.sh}"
[ -e "$SCRIPT" ] || SCRIPT="$vakuu/scripts/$SCRIPT"
TIMEOUT="${2:-${WATCHDOG_TIMEOUT:-1800}}"
mkdir -p "$vakuu/data/spire"
LOG="$vakuu/data/spire/watchdog_$(basename "$SCRIPT" .sh)_$(date +%Y%m%d_%H%M%S).log"

cleanup_stragglers() {
    # The facilities put each worker in its own process group, so the plain group-kill of $pid below
    # does not reach them. Python 3.14 uses "forkserver" as the default start method on Linux, so
    # both spellings have to go, or idle workers pile up across restarts.
    pkill -9 -f "multiprocessing.spawn"
    pkill -9 -f "multiprocessing.forkserver"
    pkill -9 -f "Sts2Ai.SimWorker.dll run"
}

echo "=== [watchdog] Supervising $SCRIPT, hard-restarting every ${TIMEOUT}s. Log: $LOG ==="

while true; do
    echo "=== [watchdog] Launching $SCRIPT at $(date) ===" | tee -a "$LOG"
    "$SCRIPT" >> "$LOG" 2>&1 &
    pid=$!

    slept=0
    while kill -0 "$pid" 2>/dev/null && [ "$slept" -lt "$TIMEOUT" ]; do
        sleep 10
        slept=$((slept + 10))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "=== [watchdog] ${TIMEOUT}s elapsed -- force-killing process group $pid ===" | tee -a "$LOG"
    else
        echo "=== [watchdog] $SCRIPT exited on its own after ${slept}s -- restarting anyway ===" | tee -a "$LOG"
    fi

    kill -9 -- "-$pid" 2>/dev/null
    cleanup_stragglers
    sleep 5

    echo "=== [watchdog] Restarting in 3s... ===" | tee -a "$LOG"
    sleep 3
done
