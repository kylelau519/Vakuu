#!/bin/bash
shopt -s nullglob

# materialize.py's ProcessPoolExecutor workers hand every tokenized example back to the parent
# as live torch.Tensor objects; torch's default cross-process tensor sharing opens one
# file-descriptor-backed /dev/shm segment per tensor in the RECEIVING process, and nothing ever
# closes them until the whole run's all_examples list is done and torch.save runs. The parent's
# default soft limit (1024 on this machine) gets eaten alive once a round's history window grows
# past a few hundred branch groups -- confirmed by hand: parent sat at 1008/1024 open fds with
# every worker idle, mid-handoff, unable to open one more segment. Raising the soft limit (well
# under the 1048576 hard limit already available) buys enough headroom for this dataset's size.
ulimit -n 65536

export vakuu="/home/kylelau519/Programming/Vakuu"

# Caps torch's per-process intra-op thread pool -- see training_claws.sh's own comment; same
# reasoning here (each branch/main worker process runs policy.py's torch inference).
export OMP_NUM_THREADS=1

# Each branch/main/materialize worker loads its own in-process CoreCLR (buffer_env.py's
# Sts2BufferSession), and sts2-cli's Sts2Headless.csproj builds it with Server GC, which spawns
# one GC thread PER LOGICAL CORE inside every single one of those processes. With this many
# workers running concurrently, that's dozens of processes each independently trying to use every
# core for GC -- confirmed by hand: one worker went from 50 OS threads down to 30 with this unset
# to 0 (Workstation GC, one GC thread total instead of one per core), and the difference scales
# with worker count, not a fixed cost. DOTNET_gcServer overrides the build's runtimeconfig.json at
# process start, same as any hosted CoreCLR -- no sts2-cli rebuild needed.
export DOTNET_gcServer=0

ROUND=1
HISTORY_WINDOW=5  # how many past rounds' raw data to keep + train on, alongside the fresh round

cleanup_procs() {
    pkill -9 -f "multiprocessing.spawn"
    # Python 3.14 switched Linux's default multiprocessing start method from "spawn" to
    # "forkserver" -- without this, a wedged round's worker processes (confirmed by hand:
    # idle forever, near-zero CPU, after the CLR-bridge's result never makes it back to the
    # parent) never get reaped between rounds, and each new round's materialize.py/
    # facility_claws.py piles up alongside the stuck one instead of replacing it.
    pkill -9 -f "multiprocessing.forkserver"
    pkill -9 -f "sts2-cli/src/Sts2Headless"
    sleep 5
}

# Moves a just-finished round's raw trial JSON out of raw_dir into history_dir/round_<timestamp>/
# instead of deleting it (previously: `find raw_dir -delete`), so materialize.py can later be
# pointed at the last $window rounds together for a proper accumulated retrain instead of only
# ever seeing one round's small on-policy batch -- see chat about survivorship/exploration-
# collapse risk from training on a single fresh round with no replay buffer. Prunes down to the
# newest $window round directories once archived.
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
        for ((i = 0; i < n - window; i++)); do
            rm -rf "${dirs[$i]}"
        done
    fi
}

while true; do
    echo "==========================================="
    echo " Starting Claws Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    # policy.py's card_reward decisions upgrade to this trained net (if one exists yet) instead of
    # the static tier-list heuristic -- see agents/whispering_earring/policy.py's
    # _choose_card_reward_decision.
    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""
    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"

    # map_select decisions upgrade to this trained net (if one exists yet) instead of
    # golden_compass/policy.py's DP heuristic -- see agents/whispering_earring/policy.py's
    # _choose_map_action. Trained at the end of this same round below (Training Golden Compass
    # section), off this round's own data/raw + data/raw_choices output.
    GOLDEN_COMPASS_CHECKPOINT_PATH="$vakuu/agents/golden_compass/model/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT_PATH" ] || GOLDEN_COMPASS_CHECKPOINT_PATH=""
    export GOLDEN_COMPASS_CHECKPOINT="$GOLDEN_COMPASS_CHECKPOINT_PATH"

    cleanup_procs
    
    echo "[claws 1/7] Archiving previous round's raw data (keeping last $HISTORY_WINDOW rounds)..."
    archive_round_data "$vakuu/agents/whispering_earring/data/raw" \
        "$vakuu/agents/whispering_earring/data/history/claws" "$HISTORY_WINDOW"
    find "$vakuu/agents/whispering_earring/data/raw/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[claws 2/7] Running facility_claws.py (Data Generation)..."
    # --map-branches-per-trial -1 (unlimited): every select_map_node decision spawns a
    # MapOptionJob per sibling for golden_compass -- see agents/golden_compass/model/README.md.
    # These land on the shared branch-worker pool's low-priority queue behind combat turn-
    # branches, so they can't starve/slow combat's own correctness, only opportunistically use
    # whatever worker time is left over. Dial down to a positive cap here if rounds start taking
    # noticeably longer than before.
    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 10 --branch-workers 15 --branches-per-turn 40 --main-workers 5 \
        --ascension 1 --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --map-branches-per-trial -1 --map-room-horizon 1

    echo "[claws 3/7] Floor/combat stats for this round..."
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --round "$ROUND" --no-card-stats

    echo "[claws 4/7] Materializing dataset (this round + up to $HISTORY_WINDOW archived rounds)..."
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        "$vakuu/agents/whispering_earring/data/history/claws/"*/ \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --jobs 5

    echo "[claws 5/7] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 40 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}" \
        --lr 6.0e-5 \
        --batch-size 512

    echo "[claws 6/7] Backing up checkpoint..."
    backup_dir="$vakuu/agents/claws/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/claws/model/claws.pt" \
        "$backup_dir/claws_$(date +%Y%m%d_%H%M%S).pt"

    echo "[claws 7/7] Single-rollout comparison (current vs previous round vs untrained init)..."
    # Frozen random-init baseline, generated once and reused every round -- train.py refuses to
    # overwrite an existing --out, so this only ever runs on the very first round.
    UNTRAINED_CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws_untrained.pt"
    if [ ! -f "$UNTRAINED_CLAWS_CHECKPOINT" ]; then
        python "$vakuu/agents/claws/model/train.py" \
            --epochs 0 --init "" --out "$UNTRAINED_CLAWS_CHECKPOINT"
    fi
    # The backup dir's newest file is the one just copied above (this round's post-training
    # checkpoint); the second-newest is last round's -- claws.pt itself already got overwritten
    # by --out above, so this is the only place last round's weights still exist.
    PREVIOUS_CLAWS_CHECKPOINT=$(ls -t "$backup_dir"/claws_*.pt 2>/dev/null | sed -n '2p')
    eval_args=(
        "current=$vakuu/agents/claws/model/claws.pt"
        "untrained=$UNTRAINED_CLAWS_CHECKPOINT"
    )
    [ -n "$PREVIOUS_CLAWS_CHECKPOINT" ] && eval_args+=("previous=$PREVIOUS_CLAWS_CHECKPOINT")
    python "$vakuu/agents/whispering_earring/eval_claws.py" \
        --checkpoints "${eval_args[@]}" \
        --ascension 1

    cleanup_procs

    echo "==========================================="
    echo " Starting Choices Paradox Self-Play Loop: Round $ROUND"
    echo "==========================================="

    # Refresh: the claws round above may have just written a new checkpoint.
    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""

    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"

    # Refresh: unchanged since the claws phase above (golden compass is only trained at the end
    # of this same round, below), kept here for the same reasoning as CLAWS_CHECKPOINT's refresh.
    GOLDEN_COMPASS_CHECKPOINT_PATH="$vakuu/agents/golden_compass/model/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT_PATH" ] || GOLDEN_COMPASS_CHECKPOINT_PATH=""
    export GOLDEN_COMPASS_CHECKPOINT="$GOLDEN_COMPASS_CHECKPOINT_PATH"

    raw_dir="$vakuu/agents/whispering_earring/data/raw_choices"

    echo "[choices 1/6] Archiving previous round's raw data (keeping last $HISTORY_WINDOW rounds)..."
    archive_round_data "$raw_dir" \
        "$vakuu/agents/whispering_earring/data/history/choices" "$HISTORY_WINDOW"
    find "$raw_dir/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[choices 2/6] Running facility_choices_paradox.py (Data Generation)..."
    # --map-branches-per-trial -1: same golden_compass MapOptionJob branching as the claws round
    # above, sharing this facility's own branch-worker pool with its meta-decision chains --
    # both land on the same low-priority queue behind combat turn-branches.
    python "$vakuu/agents/whispering_earring/facility_choices_paradox.py" \
        --trials 40 --branch-workers 8 --main-workers 12 \
        --branches-per-turn 4 --meta-branches-per-trial 20\
        --ascension 1 --raw-dir "$raw_dir" \
        --map-branches-per-trial -1 --map-room-horizon 1

    echo "[choices 3/6] Meta-decision stats for this round..."
    python "$vakuu/agents/whispering_earring/stats_choices.py" \
        --raw-dir "$raw_dir" \
        --round "$ROUND" \
        --jobs 6

    echo "[choices 4/6] Materializing dataset (this round + up to $HISTORY_WINDOW archived rounds)..."
    python "$vakuu/agents/choices_paradox/model/materialize.py" \
        "$raw_dir/" \
        "$vakuu/agents/whispering_earring/data/history/choices/"*/ \
        --out "$vakuu/agents/choices_paradox/model/dataset.pt" \
        --jobs 8

    echo "[choices 5/6] Training model..."
    python "$vakuu/agents/choices_paradox/model/train.py" \
        "$vakuu/agents/choices_paradox/model/dataset.pt" --epochs 80 \
        --out "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        --init "$CHOICES_CHECKPOINT" \
        --warm-start-from "${CLAWS_CHECKPOINT:-}" \
        --lr 7e-6 \
        --batch-size 64

    echo "[choices 6/6] Backing up checkpoint..."
    backup_dir="$vakuu/agents/choices_paradox/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        "$backup_dir/choices_paradox_$(date +%Y%m%d_%H%M%S).pt"

    cleanup_procs

    echo "==========================================="
    echo " Training Golden Compass: Round $ROUND"
    echo "==========================================="

    # No dedicated facility run -- both phases above already spawn MapOptionJob branches per
    # select_map_node decision (--map-branches-per-trial), landing in the same raw dirs as their
    # own main-line/meta-branch files, so this just materializes off the raw data both phases
    # already produced this round -- see agents/golden_compass/model/README.md's Design section.
    echo "[golden 1/2] Materializing dataset from claws + choices raw data..."
    python "$vakuu/agents/golden_compass/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        "$raw_dir/" \
        --out "$vakuu/agents/golden_compass/model/dataset.pt" \
        --jobs 8

    echo "[golden 2/2] Training model..."
    python "$vakuu/agents/golden_compass/model/train.py" \
        "$vakuu/agents/golden_compass/model/dataset.pt" --epochs 80 \
        --out "$vakuu/agents/golden_compass/model/golden_compass.pt" \
        --init "$GOLDEN_COMPASS_CHECKPOINT_PATH" \
        --batch-size 32

    echo "Backing up Golden Compass checkpoint..."
    backup_dir="$vakuu/agents/golden_compass/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/golden_compass/model/golden_compass.pt" \
        "$backup_dir/golden_compass_$(date +%Y%m%d_%H%M%S).pt"

    echo "==========================================="
    echo " Round $ROUND complete (claws + choices + golden compass)! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done
