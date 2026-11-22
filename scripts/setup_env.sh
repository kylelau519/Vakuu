#!/bin/bash
# One-time Python environment for Vakuu, plus a check of the spire-zero side. Run from the repo root.
#
# Prerequisites this script does NOT install:
#   - .NET 9+ SDK, and a Rust toolchain (spire-zero's combat engine)
#   - Slay the Spire 2 v0.113.1 (Steam public-beta branch): spire-zero needs the game's assemblies, see its README
#   - Python 3.11+ with venv, and (for GPU boxes) an NVIDIA driver
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# torch's default Linux wheel already bundles CUDA -- verify the driver actually picks it up.
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# spire-zero: the game host (src/Sts2Ai.SimWorker) and the Rust combat engine. Vakuu looks for it at $SPIRE_ZERO
# (default ~/Programming/spire-zero). Building it is spire-zero's own business; its README has the steps:
#     git clone https://github.com/shinnku-nikaidou/spire-zero.git ~/Programming/spire-zero
#     (copy the game's assemblies, then)  dotnet build Sts2Ai.sln -c Release   and   cargo build --release  (in engine/)
SZ="${SPIRE_ZERO:-$HOME/Programming/spire-zero}"
echo
if [ -e "$SZ/engine/target/release/libsts2ai_engine.so" ] && [ -e "$SZ/src/Sts2Ai.SimWorker/bin/Release/net9.0/Sts2Ai.SimWorker.dll" ]; then
    echo "spire-zero found and built at $SZ"
else
    echo "spire-zero is not built at $SZ (engine/target/release/libsts2ai_engine.so, src/Sts2Ai.SimWorker/bin/Release/net9.0/Sts2Ai.SimWorker.dll)."
    echo "Build it first; set SPIRE_ZERO if it lives elsewhere."
fi

echo
echo "Activate with: source .venv/bin/activate     Tests: python -m pytest tests"
echo "Needs data/act_checkpoints (act-boundary saves) for --resume-act runs; scripts/train_alternating.sh plays from floor 0 without them."
