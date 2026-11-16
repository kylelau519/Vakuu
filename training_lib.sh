# Shared helpers for training_*.sh (source this; it does nothing on its own).

kill_stragglers() {
    # Python 3.14 switched Linux's default multiprocessing start method to "forkserver"; a wedged
    # round's workers otherwise sit idle forever and pile up beside the next round's. (Matches every
    # such process on the machine, not only this script's -- don't run other multiprocessing jobs
    # alongside a training loop.)
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null
    pkill -9 -f "multiprocessing.forkserver" 2>/dev/null
    pkill -9 -f "sts2-cli/src/Sts2Headless" 2>/dev/null
    return 0
}

cleanup_procs() {
    kill_stragglers
    sleep 5
}

# Ctrl-C / kill on a training script: the facility (see facility_common.py) already kills its own
# workers within a few seconds and the workers kill themselves if it dies, so this is the backstop for
# anything else a half-finished stage left behind (materialize's pool, an orphaned CLR host). Bash runs
# the trap once the foreground python has exited, so the facility gets to clean up first.
install_traps() {
    trap 'echo; echo "Interrupted -- cleaning up leftover workers..."; kill_stragglers; exit 130' INT TERM HUP
    trap kill_stragglers EXIT
}

# Moves a finished round's raw run JSON into history_dir/round_<timestamp>/ and prunes down to the
# newest $window round directories, so training sees a small replay window instead of one round.
archive_round_data() {
    local raw_dir="$1" history_dir="$2" window="$3"
    if [ -n "$(find "$raw_dir" -maxdepth 1 -name '*.json' -print -quit)" ]; then
        local dest="$history_dir/round_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$dest"
        find "$raw_dir" -maxdepth 1 -name '*.json' -exec mv {} "$dest/" \;
    fi
    local dirs=("$history_dir"/round_*)
    local n=${#dirs[@]}
    if (( n > window )); then
        for ((i = 0; i < n - window; i++)); do rm -rf "${dirs[$i]}"; done
    fi
}

# Points policy.py at whichever trained checkpoints exist (unset = that component falls back to
# random / the tier list).
export_checkpoints() {
    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT
    export GOLDEN_COMPASS_CHECKPOINT="$vakuu/agents/golden_compass/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT" ] || unset GOLDEN_COMPASS_CHECKPOINT
}

backup_checkpoint() {
    local ckpt="$1" dir
    dir="$(dirname "$ckpt")/backups"
    mkdir -p "$dir"
    cp "$ckpt" "$dir/$(basename "$ckpt" .pt)_$(date +%Y%m%d_%H%M%S).pt"
}
