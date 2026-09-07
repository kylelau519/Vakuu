#!/bin/bash
#
# Runs a training loop (training_alternating.sh, training_claws.sh, ...) under a hard
# wall-clock limit: force-kills the whole process tree and restarts every WATCHDOG_TIMEOUT
# seconds, no matter what state it's in. This exists because the loop has a known,
# not-yet-root-caused hang somewhere inside a round; losing an in-flight round's data is
# fine (next round regenerates it), so this favors a periodic hard reset over trying to
# detect the specific hang.
#
# Usage: ./watchdog_overnight.sh [script-to-supervise] [timeout-seconds]
#   defaults: training_alternating.sh, 1800 (30 min)

set -m  # job control on, so the backgrounded script becomes its own process group leader

# Detached/nohup launches don't inherit the interactive shell's `conda activate sts2`, so
# put the env's bin dir on PATH directly -- otherwise every `python` call in the supervised
# script fails immediately and the loop spins doing nothing.
export PATH="/opt/homebrew/Caskroom/miniconda/base/envs/sts2/bin:$PATH"

vakuu="/Users/kylelau519/Programming/Vakuu"
SCRIPT="${1:-$vakuu/training_alternating.sh}"
TIMEOUT="${2:-${WATCHDOG_TIMEOUT:-1800}}"
LOG="$vakuu/watchdog_$(basename "$SCRIPT" .sh)_$(date +%Y%m%d_%H%M%S).log"

cleanup_stragglers() {
    # facility_common.py gives each self-play worker (and its dotnet Sts2Headless child) its
    # own process group, so a plain group-kill of $pid below won't reach them -- same
    # belt-and-suspenders pkill training_claws.sh already does after every round.
    pkill -9 -f "multiprocessing.spawn"
    # Python 3.14 switched Linux's default multiprocessing start method from "spawn" to
    # "forkserver" -- see training_alternating.sh's matching line for the full explanation.
    # Without this, this is exactly the hang this watchdog exists to route around: the group
    # kill above misses these (confirmed by hand: they end up outside $pid's process group),
    # so the stale "spawn"-only pattern left them idling for 10+ hours across restarts.
    pkill -9 -f "multiprocessing.forkserver"
    pkill -9 -f "sts2-cli/src/Sts2Headless"
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
