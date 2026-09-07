#!/bin/bash

# See training_alternating.sh's matching line for the full explanation: materialize.py's parent
# process accumulates one open file-descriptor-backed shm segment per tensor handed back from a
# worker, never closed until the run finishes -- confirmed by hand to exhaust the default 1024
# soft limit and wedge every worker mid-handoff once a round's dataset grows large enough.
ulimit -n 65536

export vakuu="/home/kylelau519/Programming/Vakuu"

# Every branch/main worker is its own process (policy.py imports torch and runs inference --
# softmax/Dirichlet/multinomial -- in each one). Without this, each process defaults to one
# intra-op thread per core, so branch-workers x cores threads fight over the same cores --
# pure contention, no extra throughput (same thread-pool-sync-overhead effect
# claws/model/train.py's docstring found for a single process, just multiplied across every
# worker). macOS's Accelerate-backed torch build tolerates this better than Linux's
# OpenBLAS/OMP default, which is why it only bit on Linux.
export OMP_NUM_THREADS=1

# See training_alternating.sh's matching line for the full explanation: each branch/main/
# materialize worker also loads its own in-process CoreCLR (Sts2BufferSession), built with Server
# GC (one GC thread per logical core, per process) -- confirmed by hand to roughly balloon thread
# counts (50 -> 30 per worker with this set) once many workers run concurrently. Overrides
# sts2-cli's build-time runtimeconfig.json at process start; no rebuild needed.
export DOTNET_gcServer=0

ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    # policy.py's card_reward decisions upgrade to this trained net (if one exists yet) instead
    # of the static tier-list heuristic -- see agents/whispering_earring/policy.py's
    # _choose_card_reward_decision.
    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""
    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"

    # map_select decisions upgrade to this trained net (if one exists yet) instead of
    # golden_compass/policy.py's DP heuristic -- see agents/whispering_earring/policy.py's
    # _choose_map_action. Trained by the separate training_golden_compass.sh loop, off this
    # script's own data/raw output -- not produced here.
    GOLDEN_COMPASS_CHECKPOINT_PATH="$vakuu/agents/golden_compass/model/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT_PATH" ] || GOLDEN_COMPASS_CHECKPOINT_PATH=""
    export GOLDEN_COMPASS_CHECKPOINT="$GOLDEN_COMPASS_CHECKPOINT_PATH"

    echo "[1/6] Cleaning up old raw data..."
    find "$vakuu/agents/whispering_earring/data/raw/" -maxdepth 1 -name '*.json' -delete
    find "$vakuu/agents/whispering_earring/data/raw/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[2/6] Running facility_claws.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 5 --branch-workers 15 --branches-per-turn 40 --main-workers 3 \
        --ascension 1 --raw-dir "$vakuu/agents/whispering_earring/data/raw"

    echo "[3/6] Floor/combat stats for this round..."
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --round "$ROUND" --no-card-stats

    echo "[4/6] Materializing dataset..."
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --jobs 8

    echo "[5/6] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 40 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}"

    echo "[6/6] Backing up checkpoint..."
    backup_dir="$vakuu/agents/claws/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/claws/model/claws.pt" \
        "$backup_dir/claws_$(date +%Y%m%d_%H%M%S).pt"

    # Deliberately kill all the python multiprocessing to avoid zombie processes
    pkill -9 -f "multiprocessing.spawn"
    # Python 3.14 switched Linux's default multiprocessing start method from "spawn" to
    # "forkserver" -- see training_alternating.sh's matching line for the full explanation
    # (a wedged worker process from this pattern going stale was confirmed by hand to sit
    # idle for 10+ hours, never reaped, while the next round's own workers piled up next to it).
    pkill -9 -f "multiprocessing.forkserver"
    # Belt-and-suspenders for the same problem facility_common.py's Facility.stop() now handles
    # at the source (each worker is its own process group, killed via os.killpg so its `dotnet
    # run` Sts2Session children die with it) -- see training_choices.sh's matching line for the
    # full explanation (confirmed by hand: an orphaned one of these can sit around for DAYS,
    # invisible to the pkill above since that only ever matches the Python side).
    pkill -9 -f "sts2-cli/src/Sts2Headless"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done