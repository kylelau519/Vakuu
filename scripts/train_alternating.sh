#!/bin/bash
# The alternating loop on spire-zero (the real game v0.113.1 hosted by spire-zero's SimWorker + its Rust combat engine),
# from floor 0 by default. One round = Claws, then the Golden Compass, then (every GATE_EVERY rounds) the gate and (every EVAL_EVERY)
# a greedy eval:
#   1. Claws: CLAWS_TRIALS mainlines (macro decisions by the accepted compass), every combat turn forked CLAWS_FORKS times IN THE
#      ENGINE (no reload); train Claws on the last CLAWS_WINDOW rounds of that data for CLAWS_EPOCHS epochs.
#   2. Compass: COMPASS_TRIALS whole real-HP runs played by that Claws -> train a candidate on the last HISTORY_WINDOW rounds.
#   3. Gate (round 1, then every GATE_EVERY): the candidate replaces the accepted compass only if it beats it on >= 60% of the non-tied
#      fresh runs by hazard accuracy (golden_compass/compare.py); the outcome is appended to data/spire/gate_compass.log.
#   4. Eval (every EVAL_EVERY rounds, default 1 = every round): EVAL_RUNS greedy real-HP runs with the accepted compass -> data/spire/eval_claws.jsonl.
#
# Models live in runs/spire/ (warm start: made on the first start from runs/run2/, else the main nets, with their embedding
# tables moved onto this game's vocabulary by vakuu/common/extend_vocab.py); data in data/spire/. The main loop's files are never touched.
# RESUME_ACT=N starts runs from random act-N saves in data/act_checkpoints instead of floor 0. CHARACTERS (default IRONCLAD), ASCENSION (1).
#
# Do not run it together with train_claws.sh / train_compass.sh (they train the same files: those two are this loop
# split in halves, for running Claws and the compass side by side).
# Run via scripts/watchdog.sh:  ./scripts/watchdog.sh train_alternating.sh 3600
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
CLAWS_MAIN="$vakuu/runs/base/claws"
GC_MAIN="$vakuu/runs/base/compass"
CLAWS_PT="${SPIRE_CLAWS_PT:-$RUN/claws/claws.pt}"
ACC="$RUN/compass/golden_compass.pt"          # the accepted compass: plays, decides the macro choices
CAND="$RUN/compass/golden_compass_candidate.pt"  # the one the trainer continues from; the gate decides if it replaces ACC
CLAWS_WINDOW="${CLAWS_WINDOW:-3}"
HISTORY_WINDOW="${HISTORY_WINDOW:-4}"
RESUME_ACT="${RESUME_ACT:-0}"
CHARACTERS="${CHARACTERS:-IRONCLAD}"
ASCENSION="${ASCENSION:-1}"
PY="${PY:-python}"

source "$vakuu/scripts/lib.sh"
# lib.sh's kill_stragglers is machine-wide (pkill multiprocessing.spawn / forkserver); here it only ends this script's own children
# (the facilities end their workers themselves), so another loop on the machine is not hurt.
# The background Claws training (below) is a child too: the per-stage cleanups spare it, the exit/interrupt traps do not.
kill_stragglers() { local p; for p in $(pgrep -P $$ 2>/dev/null); do [ "$p" = "${CLAWS_TRAIN_PID:-}" ] || kill -9 "$p" 2>/dev/null; done; return 0; }
kill_all() { [ -n "${CLAWS_TRAIN_PID:-}" ] && kill -9 "$CLAWS_TRAIN_PID" 2>/dev/null; pkill -9 -P $$ 2>/dev/null; return 0; }
trap 'echo; echo "Interrupted -- cleaning up leftover workers..."; kill_all; exit 130' INT TERM HUP
trap kill_all EXIT

# Claws training (GPU) overlaps the compass stage (CPU): starts it in the background writing $CLAWS_PT.new, and
# finish_claws_train swaps it in before anything needs the new weights (the gate, the eval, the next round's Claws lines). The compass lines
# of that round are played by the previous round's Claws weights. Its log: data/spire/claws_train.log.
CLAWS_TRAIN_PID=""
finish_claws_train() {
    [ -n "$CLAWS_TRAIN_PID" ] || return 0
    if wait "$CLAWS_TRAIN_PID" && [ -s "$CLAWS_PT.new" ]; then
        mv "$CLAWS_PT.new" "$CLAWS_PT"; backup_checkpoint "$CLAWS_PT"
        echo "claws training finished:"; tail -n 3 "$DATA/claws_train.log"
    else
        echo "claws training FAILED (kept the previous weights); see $DATA/claws_train.log"; rm -f "$CLAWS_PT.new"
    fi
    CLAWS_TRAIN_PID=""
}

[ -e "$SPIRE_ZERO/engine/target/release/libsts2ai_engine.so" ] || { echo "the spire-zero engine is not built ($SPIRE_ZERO)"; exit 1; }
if [ "$RESUME_ACT" != 0 ]; then compgen -G "$CKPT_DIR/*_act$RESUME_ACT.json" > /dev/null || { echo "no act-$RESUME_ACT saves in $CKPT_DIR"; exit 1; }; fi
mkdir -p "$RUN/claws" "$RUN/compass" "$DATA/raw" "$DATA/raw_runs" "$DATA/raw_branch" "$DATA/history/claws" "$DATA/history/runs" "$DATA/history/branch"

if [ ! -f "$CLAWS_PT" ]; then
    SRC="$CLAWS_MAIN/claws.pt"; [ -f "$vakuu/runs/run2/claws/claws.pt" ] && SRC="$vakuu/runs/run2/claws/claws.pt"
    $PY -m vakuu.common.extend_vocab claws "$SRC" "$CLAWS_PT" || exit 1
fi
if [ ! -f "$ACC" ]; then
    SRC="$GC_MAIN/golden_compass.pt"; [ -f "$vakuu/runs/run2/compass/golden_compass.pt" ] && SRC="$vakuu/runs/run2/compass/golden_compass.pt"
    $PY -m vakuu.common.extend_vocab compass "$SRC" "$ACC" || exit 1
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
    # The compass that builds the decks Claws trains on is epsilon-greedy (tau 0, 10% uniform picks): its card scores are nearly flat, so sampling
    # them (tau 0.35) is close to a random pick and weakens the decks (act 1: +2.5 HP per combat, half the boss clears), while eps 0.1 costs about
    # +0.4 HP and keeps some card diversity. CLAWS_COMPASS_EPS / _TAU override. The evals below are fully greedy.
    CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${CLAWS_FORK_NOISE:-0.25}" \
    COMPASS_EPS="${CLAWS_COMPASS_EPS:-0.1}" COMPASS_TAU="${CLAWS_COMPASS_TAU:-0}" \
    $PY -m vakuu.whispering_earring.facility_claws \
        --trials "${CLAWS_TRIALS:-20}" --branch-workers "${WORKERS:-17}" --forks-per-turn "${CLAWS_FORKS:-20}" \
        --ascension "$ASCENSION" --raw-dir "$DATA/raw" --dedup-forks --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS
    # Held-out set: HELDOUT_RUNS greedy lines  (default 20) (no exploration, no forks) played by the weights this round starts from, so the net has
    # never trained on them. Early stopping follows their value RMSE (only the draw order and the net's own errors are left) and every
    # fresh example trains. HELDOUT_RUNS=0: the old split of the fresh runs (--holdout).
    HELD_ARGS=(--holdout "${CLAWS_HOLDOUT:-0.25}")
    HELDOUT_N="${HELDOUT_RUNS:-20}"
    if [ "$HELDOUT_N" -gt 0 ]; then
        rm -rf "$DATA/raw_greedy"
        GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_CHECKPOINT="$CLAWS_PT" CLAWS_GREEDY=1 COMPASS_EPS="${CLAWS_COMPASS_EPS:-0.1}" COMPASS_TAU="${CLAWS_COMPASS_TAU:-0}" \
        $PY -m vakuu.whispering_earring.facility_claws --trials "$HELDOUT_N" --branch-workers "${WORKERS:-17}" --forks-per-turn 0 \
            --ascension "${ASCENSION:-1}" --raw-dir "$DATA/raw_greedy" --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS
        if $PY -m vakuu.claws.materialize "$DATA/raw_greedy" --out "$RUN/claws/heldout.pt"; then
            HELD_ARGS=(--holdout 0 --heldout-file "$RUN/claws/heldout.pt")
        fi
    fi
    $PY -m vakuu.claws.materialize "$DATA/raw" "$DATA/history/claws"/*/ --out "$RUN/claws/dataset.pt" --advantage "${CLAWS_ADVANTAGE:-turn}"
    echo "--- claws training, round $ROUND ($(date +%H:%M:%S))" >> "$DATA/claws_train.log"
    $PY -m vakuu.claws.train "$RUN/claws/dataset.pt" --epochs "${CLAWS_EPOCHS:-70}" "${HELD_ARGS[@]}" --device cuda --batch-size "${CLAWS_BATCH:-1024}" \
        --lr "${CLAWS_LR:-1e-5}" --out "$CLAWS_PT.new" --init "$CLAWS_PT" >> "$DATA/claws_train.log" 2>&1 &
    CLAWS_TRAIN_PID=$!
    cleanup_procs

    echo "=========================================== Spire round $ROUND: golden compass"
    archive_round_data "$DATA/raw_runs" "$DATA/history/runs" "$HISTORY_WINDOW"
    # Fork points for the branch stage: the compass runs are hosted in SL mode and keep BRANCH_FORKS card-reward and BRANCH_MAP_FORKS route
    # fork points each (random picks); both 0 (or BRANCH_SCOUTS>0: the old separate scout runs) = NoSL, nothing kept.
    FORK_ARGS=()
    rm -rf "$DATA/fork_points"
    if [ "${BRANCH_SCOUTS:-0}" -eq 0 ] && [ $(( ${BRANCH_FORKS:-1} + ${BRANCH_MAP_FORKS:-1} )) -gt 0 ]; then
        FORK_ARGS=(--fork-dir "$DATA/fork_points" --forks-per-run "${BRANCH_FORKS:-1}" --map-forks-per-run "${BRANCH_MAP_FORKS:-1}")
    fi
    CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0}" \
    $PY -m vakuu.whispering_earring.facility_compass --trials "${COMPASS_TRIALS:-250}" --branch-workers "${WORKERS:-17}" --ascension "$ASCENSION" \
        --raw-dir "$DATA/raw_runs" --ckpt-dir "$CKPT_DIR" --resume-act "$RESUME_ACT" --characters $CHARACTERS "${FORK_ARGS[@]}"
    $PY -m vakuu.golden_compass.progress "$DATA/raw_runs" --label "$ROUND" --log "$DATA/progress_compass.jsonl"
    # Branch evaluation (card-reward and route forks with shuffled continuations, vakuu/whispering_earring/facility_branch.py): what one choice costs,
    # as paired comparisons from the same state. Its files train the compass's aux head next to the run data (train.py --branch-dir).
    # Default: replay fork points kept by the compass runs; BRANCH_FORKS=0 BRANCH_MAP_FORKS=0 turns the stage off, BRANCH_SCOUTS=N the old scouts.
    BRANCH_ARGS=()
    BRANCH_K=(--k-per-unique 2 --k-min "${BRANCH_K_MIN:-2}" --k-max "${BRANCH_K_MAX:-4}" --map-k "${BRANCH_MAP_K:-4}" --horizon "${BRANCH_HORIZON:-8}")
    if [ "${#FORK_ARGS[@]}" -gt 0 ] || [ "${BRANCH_SCOUTS:-0}" -gt 0 ]; then
        archive_round_data "$DATA/raw_branch" "$DATA/history/branch" "${BRANCH_WINDOW:-5}"
        rm -rf "$DATA/raw_branch/clones"
        if [ "${#FORK_ARGS[@]}" -gt 0 ]; then   # replay the compass runs' own fork points
            CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0}" \
            $PY -m vakuu.whispering_earring.facility_branch --points "$DATA/fork_points" --branch-workers "${WORKERS:-17}" --ascension "$ASCENSION" \
                --raw-dir "$DATA/raw_branch" --max-forks "${BRANCH_MAX_FORKS:-40}" "${BRANCH_K[@]}" --characters $CHARACTERS
        else   # old: separate scout runs
            CLAWS_CHECKPOINT="$CLAWS_PT" GOLDEN_COMPASS_CHECKPOINT="$ACC" CLAWS_NOISE_EPS="${COMPASS_CLAWS_NOISE:-0}" \
            $PY -m vakuu.whispering_earring.facility_branch --trials "$BRANCH_SCOUTS" --branch-workers "${WORKERS:-17}" --ascension "$ASCENSION" \
                --raw-dir "$DATA/raw_branch" --forks-per-scout "${BRANCH_FORKS:-2}" --map-forks-per-scout "${BRANCH_MAP_FORKS:-1}" "${BRANCH_K[@]}" --characters $CHARACTERS
        fi
        rm -rf "$DATA/raw_branch/clones" "$DATA/fork_points/clones"
        BRANCH_ARGS=(--branch-dir "$DATA/raw_branch" "$DATA/history/branch"/*/)
    fi
    $PY -m vakuu.golden_compass.materialize "$DATA/raw_runs" "$DATA/history/runs"/*/ --out "$RUN/compass/dataset.pt"
    # lr 2e-5: on a net that already trained for many rounds it kept the held-out loss nearly flat, where 1e-4 rose every epoch
    $PY -m vakuu.golden_compass.train "$RUN/compass/dataset.pt" --epochs "${COMPASS_EPOCHS:-20}" --patience "${COMPASS_PATIENCE:-10}" --lr "${COMPASS_LR:-2e-5}" --aux-weight "${COMPASS_AUX_WEIGHT:-0}" --branch-aux "${COMPASS_BRANCH_AUX:-0}" --out "$CAND" --init "$CAND" "${BRANCH_ARGS[@]}"
    backup_checkpoint "$CAND"
    cleanup_procs

    finish_claws_train
    if [ "$ROUND" = 1 ] || [ $((ROUND % ${GATE_EVERY:-5})) = 0 ]; then
        echo "=========================================== Spire round $ROUND: compass gate (candidate vs accepted)"
        if ( set -o pipefail; $PY -m vakuu.golden_compass.compare "$CAND" "$ACC" --runs "${GATE_RUNS:-100}" --workers "${WORKERS:-17}" --claws "$CLAWS_PT" \
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
        $PY -m vakuu.claws.evaluate --runs "${EVAL_RUNS:-100}" --workers "${WORKERS:-17}" --ascension "$ASCENSION" --resume-act "$RESUME_ACT" \
            --characters $CHARACTERS --ckpt-dir "$CKPT_DIR" --log "$DATA/eval_claws.jsonl" --detail-log "$DATA/eval_combats.jsonl" --label "$(date +%d.%H%M)#$ROUND" | tail -2
        cleanup_procs
    fi

    [ -n "${MAX_ROUNDS:-}" ] && [ "$ROUND" -ge "$MAX_ROUNDS" ] && break   # MAX_ROUNDS: a dry run
    ((ROUND++))
done
