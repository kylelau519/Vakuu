#!/bin/bash

# See training_alternating.sh's matching line for the full explanation: materialize.py's parent
# process accumulates one open file-descriptor-backed shm segment per tensor handed back from a
# worker, never closed until the run finishes -- confirmed by hand to exhaust the default 1024
# soft limit and wedge every worker mid-handoff once a round's dataset grows large enough.
ulimit -n 65536

export vakuu="/home/kylelau519/Programming/Vakuu"

# Caps torch's per-process intra-op thread pool -- see training_claws.sh's own comment; same
# reasoning here (each branch/main worker process runs policy.py's torch inference).
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
    echo " Starting Choices Paradox Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  

    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""  

    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT" 
    raw_dir="$vakuu/agents/whispering_earring/data/raw_choices"

    echo "[1/6] Cleaning up old raw data..."
    find "$raw_dir/" -maxdepth 1 -name '*.json' -delete
    find "$raw_dir/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[2/6] Running facility_choices_paradox.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_choices_paradox.py" \
        --trials 3 --branch-workers 10 --main-workers 3 \
        --branches-per-turn 4 --meta-branches-per-trial 6\
        --ascension 1 --raw-dir "$raw_dir"

    echo "[3/6] Meta-decision stats for this round..."
    python "$vakuu/agents/whispering_earring/stats_choices.py" \
        --raw-dir "$raw_dir" \
        --round "$ROUND"

    echo "[4/6] Materializing dataset..."
    python "$vakuu/agents/choices_paradox/model/materialize.py" \
        "$raw_dir/" \
        --out "$vakuu/agents/choices_paradox/model/dataset.pt" \
        --jobs 8

    echo "[5/6] Training model..."
    python "$vakuu/agents/choices_paradox/model/train.py" \
        "$vakuu/agents/choices_paradox/model/dataset.pt" --epochs 50 \
        --out "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        --init "$CHOICES_CHECKPOINT" \
        --warm-start-from "${CLAWS_CHECKPOINT:-}"

    echo "[6/6] Backing up checkpoint..."
    # Timestamped, never overwritten -- see training_claws.sh's own comment; same reasoning here.
    backup_dir="$vakuu/agents/choices_paradox/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        "$backup_dir/choices_paradox_$(date +%Y%m%d_%H%M%S).pt"

    # Deliberately kill all the python multiprocessing to avoid zombie processes
    pkill -9 -f "multiprocessing.spawn"
    # Python 3.14 switched Linux's default multiprocessing start method from "spawn" to
    # "forkserver" -- see training_alternating.sh's matching line for the full explanation
    # (a wedged worker process from this pattern going stale was confirmed by hand to sit
    # idle for 10+ hours, never reaped, while the next round's own workers piled up next to it).
    pkill -9 -f "multiprocessing.forkserver"
    # Belt-and-suspenders for the same problem facility_common.py's Facility.stop() now handles
    # at the source (each worker is its own process group, killed via os.killpg so its `dotnet
    # run` Sts2Session children die with it): catches anything from a round that predates that
    # fix, or any other way a worker's dotnet child could survive its parent. Confirmed by hand
    # that these can orphan (ppid 1) and sit there for DAYS otherwise -- the line above only ever
    # matches the Python side, never this. Scoped to this project's own dotnet invocation
    # specifically (not a bare "dotnet"), so it can't catch unrelated dotnet processes (e.g. an
    # IDE's own background tooling).
    pkill -9 -f "sts2-cli/src/Sts2Headless"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done
