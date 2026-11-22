#!/bin/bash
# Rewrites data/game_ids.json and data/card_table.json from the game spire-zero hosts (run after a game update; nets then
# need retraining: the token numbers shift).
set -e
SZ="${SPIRE_ZERO:-$HOME/Programming/spire-zero}"
export DOTNET_ROOT="${DOTNET_ROOT:-/usr/lib64/dotnet}"
B="$SZ/src/Sts2Ai.SimWorker/bin/Release/net9.0"
here="$(cd "$(dirname "$0")" && pwd)"
(cd "$SZ" && dotnet "$B/Sts2Ai.SimWorker.dll" tables --out "$here/data/game_ids.json" && dotnet "$B/Sts2Ai.SimWorker.dll" cardtable --out "$here/data/card_table.json")
