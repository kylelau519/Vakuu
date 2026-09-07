#!/bin/bash
# One-time environment setup on a fresh machine. Run from the directory that should hold
# both repos as siblings (sts2_path.py's default STS2_CLI_ROOT assumes this layout).
#
# Prerequisites this script does NOT install:
#   - .NET 9+ SDK              https://dotnet.microsoft.com/download
#   - Slay the Spire 2 on Steam, already installed and logged in
#   - Python 3.11+ with venv, and (for GPU boxes) an NVIDIA driver
set -e


python3 -m venv Vakuu/.venv
source Vakuu/.venv/bin/activate
pip install --upgrade pip
pip install -r Vakuu/requirements.txt

# torch's default Linux wheel already bundles CUDA -- verify the driver actually picks it up.
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Copies game DLLs from the Steam install, IL-patches sts2.dll for headless mode, builds
# Sts2Headless. Pass a game dir explicitly if auto-detection misses it.
(cd sts2-cli && ./setup.sh)

echo
echo "Setup complete. Activate with: source Vakuu/.venv/bin/activate"
echo "Smoke test: cd Vakuu && python agents/whispering_earring/facility_choices_paradox.py --trials 1 --branch-workers 1 --main-workers 1 --branches-per-turn 1 --meta-branches-per-trial 1 --ascension 1 --raw-dir /tmp/smoke_test"
