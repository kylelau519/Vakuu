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

# SEEDS: the pool of seeds every run draws from, so progression shows on a few fixed maps. Default is
# five; SEEDS="a b c" picks your own, SEEDS=random goes back to a fresh seed per run. One character
# (TRAIN_CHARACTER, default Ironclad) because a seed plus a character is one map.
SEEDS="${SEEDS:-train-1 train-2 train-3 train-4 train-5}"
SEED_ARGS=()
[ "$SEEDS" != "random" ] && SEED_ARGS=(--seeds $SEEDS --characters "${TRAIN_CHARACTER:-Ironclad}")

source "$vakuu/training_lib.sh"
install_traps

ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    # Mainline map / card-reward picks come from golden_compass.pt once it exists (trained by
    # training_compass.sh or training_alternating.sh), the tier list / random before.
    export GOLDEN_COMPASS_CHECKPOINT="$vakuu/agents/golden_compass/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT" ] || unset GOLDEN_COMPASS_CHECKPOINT

    echo "[1/5] Cleaning up old raw data..."
    find "$vakuu/agents/whispering_earring/data/raw/" -maxdepth 1 -name '*.json' -delete

    # A trial = one mainline run plus --forks-per-turn forks of every turn of every combat it
    # played (see agents/claws/model/materialize.py). V is on-policy, so this round's data is only
    # valid for the checkpoint CLAWS_CHECKPOINT points at (unset = the untrained init).
    echo "[2/5] Running facility_claws.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 5 --branch-workers 15 --forks-per-turn 40 --main-workers 3 \
        --ascension 1 --raw-dir "$vakuu/agents/whispering_earring/data/raw" "${SEED_ARGS[@]}"

    # --target return while V is untrained or just reset; switch to q once V is calibrated.
    echo "[3/5] Materializing dataset..."
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --target "${CLAWS_TARGET:-return}" --jobs 8

    echo "[4/5] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 40 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}"

    echo "[5/5] Backing up checkpoint..."
    backup_dir="$vakuu/agents/claws/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/claws/model/claws.pt" \
        "$backup_dir/claws_$(date +%Y%m%d_%H%M%S).pt"

    # Is Claws actually better? Fixed seeds, fixed non-combat policy, only the combat checkpoint changes
    # (agents/whispering_earring/eval_claws.py). EVAL_RUNS=0 skips it; rows accumulate in data/claws_eval.jsonl.
    if [ "${EVAL_RUNS:-40}" -gt 0 ]; then
        [ "$ROUND" = 1 ] && python "$vakuu/agents/whispering_earring/eval_claws.py" --ckpt random \
            --label "base" --runs "${EVAL_RUNS:-40}" --workers "${WORKERS:-8}" | tail -2
        python "$vakuu/agents/whispering_earring/eval_claws.py" --ckpt "$vakuu/agents/claws/model/claws.pt" \
            --label "$(date +%d.%H%M)#$ROUND" --runs "${EVAL_RUNS:-40}" --workers "${WORKERS:-8}" | sed -n '/--- log/,$p'
    fi

    # Deliberately kill all the python multiprocessing to avoid zombie processes
    pkill -9 -f "multiprocessing.spawn"
    # Python 3.14 switched Linux's default multiprocessing start method from "spawn" to
    # "forkserver" -- see training_alternating.sh's matching line for the full explanation
    # (a wedged worker process from this pattern going stale was confirmed by hand to sit
    # idle for 10+ hours, never reaped, while the next round's own workers piled up next to it).
    pkill -9 -f "multiprocessing.forkserver"
    # Belt-and-suspenders for the same problem facility_common.py's Facility.stop() now handles
    # at the source (each worker is its own process group, killed via os.killpg so its `dotnet
    # run` Sts2Session children die with it) -- see training_alternating.sh / training_lib.sh's
    # cleanup_procs for the full explanation (confirmed by hand: an orphaned one of these can sit around for DAYS,
    # invisible to the pkill above since that only ever matches the Python side).
    pkill -9 -f "sts2-cli/src/Sts2Headless"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done