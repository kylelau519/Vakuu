#!/bin/bash
# The Golden Compass on spire-zero, meant to run NEXT TO train_claws.sh (both at once, each with its own share of the machine).
# Every round: COMPASS_TRIALS whole real-HP runs from floor 0 (RESUME_ACT=N: from act-N saves), Claws playing the combats and the
# compass the rest, from a SNAPSHOT of the Claws the other loop is training (copied and checked at the start of the round, so a file
# being written is never read) -> hazard-net training data -> train a candidate -> every GATE_EVERY rounds the 60% hazard-accuracy
# gate decides whether the candidate replaces the accepted compass. Progress rows: data/spire/progress_compass.jsonl.
#
# CPU split: this machine has the cores of both loops; give each its share, e.g. WORKERS=10 here and WORKERS=10 there.
# Warm start: runs/spire/compass/golden_compass.pt is made on the first start from runs/run2/compass/golden_compass.pt (else the main one),
# its embedding tables moved onto this game's vocabulary by vakuu/common/extend_vocab.py.
#
# Run via scripts/watchdog.sh:  ./scripts/watchdog.sh train_compass.sh 3600
shopt -s nullglob
ulimit -n 65536

export vakuu="${vakuu:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$vakuu"
export PYTHONPATH="$vakuu${PYTHONPATH:+:$PYTHONPATH}"   # the vakuu package; the python stages run as `python -m vakuu...` from the repo root
export OMP_NUM_THREADS=1
export SPIRE_ZERO="${SPIRE_ZERO:-$HOME/Programming/spire-zero}"
export DOTNET_ROOT="${DOTNET_ROOT:-/usr/lib64/dotnet}"
export DOTNET_gcServer=0
export COMPASS_CARD_AUX="${COMPASS_CARD_AUX:-0}"
export COMPASS_EPS="${COMPASS_EPS:-0.03}" COMPASS_TAU="${COMPASS_TAU:-0.35}"

RUN="${SPIRE_RUN:-$vakuu/runs/spire}"
DATA="${SPIRE_DATA:-$vakuu/data/spire}"
CKPT_DIR="${COMPASS_CKPT_DIR:-$vakuu/data/act_checkpoints}"
CLAWS_PT="${SPIRE_CLAWS_PT:-$RUN/claws/claws.pt}"          # trained by train_claws.sh; only read here
GC_MAIN="$vakuu/runs/base/compass"
ACC="$RUN/compass/golden_compass.pt"
CAND="$RUN/compass/golden_compass_candidate.pt"
SNAP="$RUN/compass/claws_snapshot.pt"
HISTORY_WINDOW="${HISTORY_WINDOW:-4}"
RESUME_ACT="${RESUME_ACT:-0}"
CHARACTERS="${CHARACTERS:-IRONCLAD}"
ASCENSION="${ASCENSION:-1}"
PY="${PY:-python}"

source "$vakuu/scripts/lib.sh"
# lib.sh's kill_stragglers is machine-wide (pkill multiprocessing.spawn / forkserver): with the Claws and compass loops running side
# by side each would kill the other's workers. Here it only ends this script's own children (the facilities end their workers themselves).
kill_stragglers() { pkill -9 -P $$ 2>/dev/null; return 0; }
install_traps

[ -e "$SPIRE_ZERO/engine/target/release/libsts2ai_engine.so" ] || { echo "the spire-zero engine is not built ($SPIRE_ZERO)"; exit 1; }
mkdir -p "$RUN/compass" "$RUN/claws" "$DATA/raw_runs" "$DATA/history/runs" "$DATA/raw_branch" "$DATA/history/branch"
if [ ! -f "$ACC" ]; then
    SRC="$GC_MAIN/golden_compass.pt"; [ -f "$vakuu/runs/run2/compass/golden_compass.pt" ] && SRC="$vakuu/runs/run2/compass/golden_compass.pt"
    $PY -m vakuu.common.extend_vocab compass "$SRC" "$ACC" || exit 1
fi
[ -f "$CAND" ] || cp "$ACC" "$CAND"
if [ ! -f "$CLAWS_PT" ]; then
    SRC="$CLAWS_MAIN/claws.pt"; [ -f "$vakuu/runs/run2/claws/claws.pt" ] && SRC="$vakuu/runs/run2/claws/claws.pt"
    $PY -m vakuu.common.extend_vocab claws "$SRC" "$CLAWS_PT" || exit 1
fi

snapshot_claws() {   # copy the other loop's Claws, retrying until the copy loads (it may be mid-write)
    local i
    for i in 1 2 3 4 5 6; do
        cp "$CLAWS_PT" "$SNAP.tmp" && $PY -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$SNAP.tmp" 2>/dev/null \
            && { mv "$SNAP.tmp" "$SNAP"; return 0; }
        sleep 10
    done
    echo "claws snapshot failed; keeping the previous one"; [ -f "$SNAP" ]
}

ROUND=1
while true; do
    echo "=========================================== Spire compass round $ROUND"
    snapshot_claws || exit 1
    archive_round_data "$DATA/raw_runs" "$DATA/history/runs" "$HISTORY_WINDOW"
    # Fork points for the branch stage: the compass runs are hosted in SL mode and keep BRANCH_FORKS card-reward and BRANCH_MAP_FORKS route
    # fork points each (random picks); both 0 (or BRANCH_SCOUTS>0: the old separate scout runs) = NoSL, nothing kept.
    FORK_ARGS=()
    rm -rf "$DATA/fork_points"
    if [ "${BRANCH_SCOUTS:-0}" -eq 0 ] && [ $(( ${BRANCH_FORKS:-1} + ${BRANCH_MAP_FORKS:-1} )) -gt 0 ]; then
        FORK_ARGS=(--fork-dir "$DATA/fork_points" --forks-per-run "${BRANCH_FORKS:-1}" --map-forks-per-run "${BRANCH_MAP_FORKS:-1}")
    fi
    CLAWS_CHECKPOINT="$SNAP" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0}" \
    $PY -m vakuu.golden_compass.facility --trials "${COMPASS_TRIALS:-250}" --branch-workers "${WORKERS:-8}" --ascension "${ASCENSION:-1}" \
        --raw-dir "$DATA/raw_runs" --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS "${FORK_ARGS[@]}"
    $PY -m vakuu.golden_compass.progress "$DATA/raw_runs" --label "$ROUND" --log "$DATA/progress_compass.jsonl"
    # Branch evaluation (card-reward and route forks with shuffled continuations, vakuu/branch/facility.py): what one choice costs,
    # as paired comparisons from the same state. Its files train the compass's aux head next to the run data (train.py --branch-dir).
    # Default: replay fork points kept by the compass runs; BRANCH_FORKS=0 BRANCH_MAP_FORKS=0 turns the stage off, BRANCH_SCOUTS=N the old scouts.
    BRANCH_ARGS=()
    BRANCH_K=(--k-per-unique 2 --k-min "${BRANCH_K_MIN:-2}" --k-max "${BRANCH_K_MAX:-4}" --map-k "${BRANCH_MAP_K:-4}" --horizon "${BRANCH_HORIZON:-8}")
    if [ "${#FORK_ARGS[@]}" -gt 0 ] || [ "${BRANCH_SCOUTS:-0}" -gt 0 ]; then
        archive_round_data "$DATA/raw_branch" "$DATA/history/branch" "${BRANCH_WINDOW:-5}"
        rm -rf "$DATA/raw_branch/clones"
        if [ "${#FORK_ARGS[@]}" -gt 0 ]; then   # replay the compass runs' own fork points
            CLAWS_CHECKPOINT="$SNAP" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0}" \
            $PY -m vakuu.branch.facility --points "$DATA/fork_points" --branch-workers "${WORKERS:-8}" --ascension "$ASCENSION" \
                --raw-dir "$DATA/raw_branch" --max-forks "${BRANCH_MAX_FORKS:-40}" "${BRANCH_K[@]}" --characters $CHARACTERS
        else   # old: separate scout runs
            CLAWS_CHECKPOINT="$SNAP" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0}" \
            $PY -m vakuu.branch.facility --trials "$BRANCH_SCOUTS" --branch-workers "${WORKERS:-8}" --ascension "$ASCENSION" \
                --raw-dir "$DATA/raw_branch" --forks-per-scout "${BRANCH_FORKS:-2}" --map-forks-per-scout "${BRANCH_MAP_FORKS:-1}" "${BRANCH_K[@]}" --characters $CHARACTERS
        fi
        rm -rf "$DATA/raw_branch/clones" "$DATA/fork_points/clones"
        BRANCH_ARGS=(--branch-dir "$DATA/raw_branch" "$DATA/history/branch"/*/)
    fi
    $PY -m vakuu.golden_compass.materialize "$DATA/raw_runs" "$DATA/history/runs"/*/ --out "$RUN/compass/dataset.pt"
    $PY -m vakuu.golden_compass.train "$RUN/compass/dataset.pt" --epochs "${COMPASS_EPOCHS:-5}" --patience "${COMPASS_PATIENCE:-2}" --lr "${COMPASS_LR:-2e-5}" --aux-weight "${COMPASS_AUX_WEIGHT:-0}" --branch-aux "${COMPASS_BRANCH_AUX:-0}" --out "$CAND" --init "$CAND" "${BRANCH_ARGS[@]}"
    backup_checkpoint "$CAND"
    cleanup_procs

    if [ "$ROUND" = 1 ] || [ $((ROUND % ${GATE_EVERY:-5})) = 0 ]; then
        echo "=========================================== Spire compass round $ROUND: gate (candidate vs accepted)"
        if ( set -o pipefail; $PY -m vakuu.golden_compass.compare "$CAND" "$ACC" --runs "${GATE_RUNS:-60}" --workers "${WORKERS:-8}" --claws "$SNAP" \
                --resume-act "$RESUME_ACT" --ckpt-dir "$CKPT_DIR" | tee -a "$DATA/gate_compass.log" ); then
            backup_checkpoint "$ACC"
            cp "$CAND" "$ACC"
            echo "gate: promoted the candidate" | tee -a "$DATA/gate_compass.log"
        else
            echo "gate: kept the accepted compass" | tee -a "$DATA/gate_compass.log"
        fi
        cleanup_procs
    fi
    [ -n "${MAX_ROUNDS:-}" ] && [ "$ROUND" -ge "$MAX_ROUNDS" ] && break
    ((ROUND++))
done
