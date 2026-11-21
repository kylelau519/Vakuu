#!/bin/bash
# The alternating loop on spire-zero (the real game v0.113.1 hosted by spire-zero's SimWorker + its Rust combat engine; no sts2-cli),
# from floor 0 by default. One round = Claws, then the Golden Compass, then (every GATE_EVERY rounds) the gate and (every EVAL_EVERY)
# a greedy eval:
#   1. Claws: CLAWS_TRIALS mainlines (macro decisions by the accepted compass), every combat turn forked CLAWS_FORKS times IN THE
#      ENGINE (no reload); train Claws on the last CLAWS_WINDOW rounds of that data for CLAWS_EPOCHS epochs.
#   2. Compass: COMPASS_TRIALS whole real-HP runs played by that Claws -> train a candidate on the last HISTORY_WINDOW rounds.
#   3. Gate (round 1, then every GATE_EVERY): the candidate replaces the accepted compass only if it beats it on >= 60% of the non-tied
#      fresh runs by hazard accuracy (golden_compass/compare.py); the outcome is appended to data/spire/gate_compass.log.
#   4. Eval (every EVAL_EVERY rounds, default 1 = every round): EVAL_RUNS greedy real-HP runs with the accepted compass -> data/spire/eval_claws.jsonl.
#
# Models live in agents/spire_run/ (warm start: made on the first start from agents/run2/, else the main nets, with their embedding
# tables moved onto this game's vocabulary by spire/extend_vocab.py); data in data/spire/. The main loop's files are never touched.
# RESUME_ACT=N starts runs from random act-N saves in data/act_checkpoints instead of floor 0. CHARACTERS (default IRONCLAD), ASCENSION (1).
#
# Do not run it together with training_claw_spire.sh / training_compass_spire.sh (they train the same files: those two are this loop
# split in halves, for running Claws and the compass side by side).
# Run via watchdog_overnight.sh:  ./watchdog_overnight.sh training_alternating_spire.sh 3600
shopt -s nullglob
ulimit -n 65536

export vakuu="${vakuu:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export OMP_NUM_THREADS=1
export SPIRE_ZERO="${SPIRE_ZERO:-$HOME/Programming/spire-zero}"
export DOTNET_ROOT="${DOTNET_ROOT:-/usr/lib64/dotnet}"
export DOTNET_gcServer=0
export COMPASS_CARD_AUX="${COMPASS_CARD_AUX:-0.3}"
export COMPASS_EPS="${COMPASS_EPS:-0.03}" COMPASS_TAU="${COMPASS_TAU:-0.35}"

RUN="${SPIRE_RUN:-$vakuu/agents/spire_run}"
DATA="${SPIRE_DATA:-$vakuu/data/spire}"
CKPT_DIR="${COMPASS_CKPT_DIR:-$vakuu/data/act_checkpoints}"
CLAWS_MAIN="$vakuu/agents/claws/model"
GC_MAIN="$vakuu/agents/golden_compass"
SPIRE="$vakuu/agents/spire"
CLAWS_PT="${SPIRE_CLAWS_PT:-$RUN/claws/claws.pt}"
ACC="$RUN/compass/golden_compass.pt"          # the accepted compass: plays, decides the macro choices
CAND="$RUN/compass/golden_compass_candidate.pt"  # the one the trainer continues from; the gate decides if it replaces ACC
CLAWS_WINDOW="${CLAWS_WINDOW:-3}"
HISTORY_WINDOW="${HISTORY_WINDOW:-10}"
RESUME_ACT="${RESUME_ACT:-0}"
CHARACTERS="${CHARACTERS:-IRONCLAD}"
ASCENSION="${ASCENSION:-1}"
PY="${PY:-python}"
export COMPASS_FACILITY="$SPIRE/facility_compass.py"   # compare.py plays its gate runs with this facility

source "$vakuu/training_lib.sh"
# training_lib's kill_stragglers is machine-wide (pkill multiprocessing.spawn / forkserver); here it only ends this script's own children
# (the facilities end their workers themselves), so another loop on the machine is not hurt.
kill_stragglers() { pkill -9 -P $$ 2>/dev/null; return 0; }
install_traps

[ -e "$SPIRE_ZERO/engine/target/release/libsts2ai_engine.so" ] || { echo "the spire-zero engine is not built ($SPIRE_ZERO)"; exit 1; }
if [ "$RESUME_ACT" != 0 ]; then compgen -G "$CKPT_DIR/*_act$RESUME_ACT.json" > /dev/null || { echo "no act-$RESUME_ACT saves in $CKPT_DIR"; exit 1; }; fi
mkdir -p "$RUN/claws" "$RUN/compass" "$DATA/raw" "$DATA/raw_runs" "$DATA/raw_branch" "$DATA/history/claws" "$DATA/history/runs" "$DATA/history/branch"

if [ ! -f "$CLAWS_PT" ]; then
    SRC="$CLAWS_MAIN/claws.pt"; [ -f "$vakuu/agents/run2/claws/claws.pt" ] && SRC="$vakuu/agents/run2/claws/claws.pt"
    $PY "$SPIRE/extend_vocab.py" claws "$SRC" "$CLAWS_PT" || exit 1
fi
if [ ! -f "$ACC" ]; then
    SRC="$GC_MAIN/golden_compass.pt"; [ -f "$vakuu/agents/run2/compass/golden_compass.pt" ] && SRC="$vakuu/agents/run2/compass/golden_compass.pt"
    $PY "$SPIRE/extend_vocab.py" compass "$SRC" "$ACC" || exit 1
fi
[ -f "$CAND" ] || cp "$ACC" "$CAND"

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

ROUND=1
while true; do
    echo "=========================================== Spire round $ROUND: claws (engine forks)"
    archive_pkl "$DATA/raw" "$DATA/history/claws" "$((CLAWS_WINDOW - 1))"
    CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${CLAWS_FORK_NOISE:-0.4}" \
    $PY "$SPIRE/facility_claws.py" \
        --trials "${CLAWS_TRIALS:-20}" --branch-workers "${WORKERS:-17}" --forks-per-turn "${CLAWS_FORKS:-20}" \
        --ascension "$ASCENSION" --raw-dir "$DATA/raw" --dedup-forks --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS
    $PY "$SPIRE/materialize.py" "$DATA/raw" "$DATA/history/claws"/*/ --out "$RUN/claws/dataset.pt" --advantage "${CLAWS_ADVANTAGE:-turn}"
    $PY "$CLAWS_MAIN/train.py" "$RUN/claws/dataset.pt" --epochs "${CLAWS_EPOCHS:-70}" --holdout "${CLAWS_HOLDOUT:-0.25}" --device cuda --batch-size "${CLAWS_BATCH:-512}" \
        --lr "${CLAWS_LR:-1e-5}" --out "$CLAWS_PT" --init "$CLAWS_PT"
    backup_checkpoint "$CLAWS_PT"
    cleanup_procs

    echo "=========================================== Spire round $ROUND: golden compass"
    archive_round_data "$DATA/raw_runs" "$DATA/history/runs" "$HISTORY_WINDOW"
    CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0.05}" \
    $PY "$SPIRE/facility_compass.py" --trials "${COMPASS_TRIALS:-100}" --branch-workers "${WORKERS:-17}" --ascension "$ASCENSION" \
        --raw-dir "$DATA/raw_runs" --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS
    $PY "$GC_MAIN/progress.py" "$DATA/raw_runs" --label "$ROUND" --log "$DATA/progress_compass.jsonl"
    # Branch evaluation (card-reward and route forks with shuffled continuations, agents/spire/facility_branch.py): what one choice costs,
    # as paired comparisons from the same state. Its files train the compass's aux head next to the run data (train.py --branch-dir).
    # BRANCH_SCOUTS=0 turns the stage off.
    BRANCH_ARGS=()
    if [ "${BRANCH_SCOUTS:-6}" -gt 0 ]; then
        archive_round_data "$DATA/raw_branch" "$DATA/history/branch" "${BRANCH_WINDOW:-5}"
        rm -rf "$DATA/raw_branch/clones"
        CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0.05}" \
        $PY "$SPIRE/facility_branch.py" --trials "${BRANCH_SCOUTS:-6}" --branch-workers "${WORKERS:-17}" --ascension "$ASCENSION" \
            --raw-dir "$DATA/raw_branch" --forks-per-scout "${BRANCH_FORKS:-2}" --map-forks-per-scout "${BRANCH_MAP_FORKS:-1}" --horizon "${BRANCH_HORIZON:-6}" --characters $CHARACTERS
        rm -rf "$DATA/raw_branch/clones"
        BRANCH_ARGS=(--branch-dir "$DATA/raw_branch" "$DATA/history/branch"/*/)
    fi
    $PY "$GC_MAIN/materialize.py" "$DATA/raw_runs" "$DATA/history/runs"/*/ --out "$RUN/compass/dataset.pt"
    # lr 2e-5: on a net that already trained for many rounds it kept the held-out loss nearly flat, where 1e-4 rose every epoch
    $PY "$GC_MAIN/train.py" "$RUN/compass/dataset.pt" --epochs "${COMPASS_EPOCHS:-100}" --patience "${COMPASS_PATIENCE:-10}" --lr "${COMPASS_LR:-2e-5}" --out "$CAND" --init "$CAND" "${BRANCH_ARGS[@]}"
    backup_checkpoint "$CAND"
    cleanup_procs

    if [ "$ROUND" = 1 ] || [ $((ROUND % ${GATE_EVERY:-5})) = 0 ]; then
        echo "=========================================== Spire round $ROUND: compass gate (candidate vs accepted)"
        if ( set -o pipefail; $PY "$GC_MAIN/compare.py" "$CAND" "$ACC" --runs "${GATE_RUNS:-100}" --workers "${WORKERS:-17}" --claws "$CLAWS_PT" \
                --resume-act "$RESUME_ACT" --ckpt-dir "$CKPT_DIR" | tee -a "$DATA/gate_compass.log" ); then
            backup_checkpoint "$ACC"
            cp "$CAND" "$ACC"
            echo "gate: promoted the candidate" | tee -a "$DATA/gate_compass.log"
        else
            echo "gate: kept the accepted compass" | tee -a "$DATA/gate_compass.log"
        fi
        cleanup_procs
    fi

    if [ "${EVAL_RUNS:-100}" -gt 0 ] && { [ "$ROUND" = 1 ] || [ $((ROUND % ${EVAL_EVERY:-1})) = 0 ]; }; then
        CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_GREEDY=1 COMPASS_EPS=0 COMPASS_TAU=0 \
        $PY "$SPIRE/eval_claws.py" --runs "${EVAL_RUNS:-100}" --workers "${WORKERS:-17}" --ascension "$ASCENSION" --resume-act "$RESUME_ACT" \
            --characters $CHARACTERS --ckpt-dir "$CKPT_DIR" --log "$DATA/eval_claws.jsonl" --label "$(date +%d.%H%M)#$ROUND" | tail -2
        cleanup_procs
    fi

    [ -n "${MAX_ROUNDS:-}" ] && [ "$ROUND" -ge "$MAX_ROUNDS" ] && break   # MAX_ROUNDS: a dry run
    ((ROUND++))
done
