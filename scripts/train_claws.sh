#!/bin/bash
# Claws only, on spire-zero (the real game v0.113.1 hosted by spire-zero's SimWorker + its Rust combat engine).
# Every round: CLAWS_TRIALS mainlines (RESUME_ACT=0: fresh runs from floor 0, the default; N: random act-N saves), each combat turn forked CLAWS_FORKS times IN THE ENGINE (no reload),
# the last CLAWS_WINDOW rounds of that data trained on, then (every EVAL_EVERY rounds and round 1) EVAL_RUNS greedy real-HP runs
# logged to data/spire/eval_claws.jsonl. Macro decisions are the stage-1 heuristics (spire/macro.py); the compass is not involved.
#
# Warm start: runs/spire/claws/claws.pt is made on the first start from runs/run2/claws/claws.pt (else the main claws.pt) with
# vakuu/common/extend_vocab.py: its embedding table is moved onto this game's vocabulary (new cards start as copies of UNKNOWN). Its history features are zero.
#
# Run via scripts/watchdog.sh:  ./scripts/watchdog.sh train_claws.sh 3600
shopt -s nullglob
ulimit -n 65536

export vakuu="${vakuu:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$vakuu"
export PYTHONPATH="$vakuu${PYTHONPATH:+:$PYTHONPATH}"   # the vakuu package; the python stages run as `python -m vakuu...` from the repo root
export OMP_NUM_THREADS=1
export SPIRE_ZERO="${SPIRE_ZERO:-$HOME/Programming/spire-zero}"
export DOTNET_ROOT="${DOTNET_ROOT:-/usr/lib64/dotnet}"
export DOTNET_gcServer=0

CKPT_DIR="${COMPASS_CKPT_DIR:-$vakuu/data/act_checkpoints}"
RUN="${SPIRE_RUN:-$vakuu/runs/spire}"
DATA="${SPIRE_DATA:-$vakuu/data/spire}"
CLAWS_PT="${SPIRE_CLAWS_PT:-$RUN/claws/claws.pt}"
CLAWS_WINDOW="${CLAWS_WINDOW:-3}"
CLAWS_MAIN="$vakuu/runs/base/claws"
PY="${PY:-python}"

source "$vakuu/scripts/lib.sh"
# lib.sh's kill_stragglers is machine-wide (pkill multiprocessing.spawn / forkserver): with the Claws and compass loops running side
# by side each would kill the other's workers. Here it only ends this script's own children (the facilities end their workers themselves).
kill_stragglers() { pkill -9 -P $$ 2>/dev/null; return 0; }
install_traps

RESUME_ACT="${RESUME_ACT:-0}"
CHARACTERS="${CHARACTERS:-IRONCLAD}"
if [ "$RESUME_ACT" != 0 ]; then compgen -G "$CKPT_DIR/*_act$RESUME_ACT.json" > /dev/null || { echo "no act-$RESUME_ACT saves in $CKPT_DIR"; exit 1; }; fi
[ -e "$SPIRE_ZERO/engine/target/release/libsts2ai_engine.so" ] || { echo "the spire-zero engine is not built ($SPIRE_ZERO)"; exit 1; }
mkdir -p "$RUN/claws" "$DATA/raw" "$DATA/history"
# The warm-start nets are moved onto this game version's vocabulary (vakuu/common/vocab_ids.json) by vakuu/common/extend_vocab.py.
if [ ! -f "$CLAWS_PT" ]; then
    SRC="$CLAWS_MAIN/claws.pt"; [ -f "$vakuu/runs/run2/claws/claws.pt" ] && SRC="$vakuu/runs/run2/claws/claws.pt"
    $PY -m vakuu.common.extend_vocab claws "$SRC" "$CLAWS_PT" || exit 1
fi
export CLAWS_CHECKPOINT="$CLAWS_PT"

# like archive_round_data, for the spire raw files (.pkl): last round's runs move to history, older ones are dropped
archive_pkl() {
    local raw="$1" hist="$2" window="$3"
    if [ -n "$(find "$raw" -maxdepth 1 -name '*.pkl' -print -quit)" ]; then
        local dest="$hist/round_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$dest"
        find "$raw" -maxdepth 1 -name '*.pkl' -exec mv {} "$dest/" \;
    fi
    local dirs=("$hist"/round_*)
    local n=${#dirs[@]}
    if (( n > window )); then
        for ((i = 0; i < n - window; i++)); do rm -rf "${dirs[$i]}"; done
    fi
}

# The compass the other loop trains (train_compass.sh) decides the mainlines' and the eval's macro choices when it exists: a
# validated copy per round, so a file being written is never read. Without one the stage-1 heuristics decide.
snapshot_compass() {
    local acc="$RUN/compass/golden_compass.pt" i
    [ -f "$acc" ] || { unset GOLDEN_COMPASS_CHECKPOINT; return 0; }
    for i in 1 2 3 4 5 6; do
        cp "$acc" "$RUN/claws/compass_snapshot.pt.tmp" && $PY -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$RUN/claws/compass_snapshot.pt.tmp" 2>/dev/null \
            && { mv "$RUN/claws/compass_snapshot.pt.tmp" "$RUN/claws/compass_snapshot.pt"; export GOLDEN_COMPASS_CHECKPOINT="$RUN/claws/compass_snapshot.pt"; return 0; }
        sleep 5
    done
    return 0
}

ROUND=1
while true; do
    echo "=========================================== Spire Claws round $ROUND (act-2 saves, engine forks)"
    snapshot_compass
    archive_pkl "$DATA/raw" "$DATA/history" "$((CLAWS_WINDOW - 1))"
    CLAWS_NOISE_EPS="${CLAWS_FORK_NOISE:-0.25}" \
    $PY -m vakuu.whispering_earring.facility_claws \
        --trials "${CLAWS_TRIALS:-30}" --branch-workers "${WORKERS:-16}" --forks-per-turn "${CLAWS_FORKS:-40}" \
        --ascension "${ASCENSION:-1}" --raw-dir "$DATA/raw" --dedup-forks --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS
    # Held-out set: HELDOUT_RUNS greedy lines  (default 20) (no exploration, no forks) played by the weights this round starts from, so the net has
    # never trained on them. Early stopping follows their value RMSE (only the draw order and the net's own errors are left) and every
    # fresh example trains. HELDOUT_RUNS=0: the old split of the fresh runs (--holdout).
    HELD_ARGS=(--holdout "${CLAWS_HOLDOUT:-0.25}")
    HELDOUT_N="${HELDOUT_RUNS:-20}"
    if [ "$HELDOUT_N" -gt 0 ]; then
        rm -rf "$DATA/raw_greedy"
        CLAWS_CHECKPOINT="$CLAWS_PT" CLAWS_GREEDY=1 \
        $PY -m vakuu.whispering_earring.facility_claws --trials "$HELDOUT_N" --branch-workers "${WORKERS:-16}" --forks-per-turn 0 \
            --ascension "${ASCENSION:-1}" --raw-dir "$DATA/raw_greedy" --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS
        if $PY -m vakuu.claws.materialize "$DATA/raw_greedy" --out "$RUN/claws/heldout.pt"; then
            HELD_ARGS=(--holdout 0 --heldout-file "$RUN/claws/heldout.pt")
        fi
    fi
    $PY -m vakuu.claws.materialize "$DATA/raw" "$DATA/history"/*/ --out "$RUN/claws/dataset.pt" --advantage "${CLAWS_ADVANTAGE:-turn}"
    $PY -m vakuu.claws.train "$RUN/claws/dataset.pt" --epochs "${CLAWS_EPOCHS:-70}" "${HELD_ARGS[@]}" --device cuda --batch-size "${CLAWS_BATCH:-512}" \
        --lr "${CLAWS_LR:-5e-5}" --out "$CLAWS_PT" --init "$CLAWS_PT"
    backup_checkpoint "$CLAWS_PT"
    cleanup_procs

    if [ "${EVAL_RUNS:-60}" -gt 0 ] && { [ "$ROUND" = 1 ] || [ $((ROUND % ${EVAL_EVERY:-1})) = 0 ]; }; then
        CLAWS_GREEDY=1 $PY -m vakuu.claws.evaluate --runs "${EVAL_RUNS:-60}" --workers "${WORKERS:-16}" --ascension "${ASCENSION:-1}" --resume-act "$RESUME_ACT" --characters $CHARACTERS \
            --ckpt-dir "$CKPT_DIR" --log "$DATA/eval_claws.jsonl" --detail-log "$DATA/eval_combats.jsonl" --label "$(date +%d.%H%M)#$ROUND" | tail -2
        cleanup_procs
    fi
    [ -n "${MAX_ROUNDS:-}" ] && [ "$ROUND" -ge "$MAX_ROUNDS" ] && break   # MAX_ROUNDS: a dry run
    ((ROUND++))
done
