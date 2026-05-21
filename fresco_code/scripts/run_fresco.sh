#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
ROOT_DIR="$(pwd)"

FRESCO="${ROOT_DIR}/source/fresco.exe"
TEMPLATE="${ROOT_DIR}/templates/reaction.in.tpl"
KD_SCRIPT="${ROOT_DIR}/scripts/kd_potential.py"
RUN_DIR="${ROOT_DIR}/runs"

if [[ $# -ne 7 ]]; then
    echo "Usage: $0 REACTION_LABEL PROJECTILE Z A START_ENERGY END_ENERGY STEP_ENERGY"
    echo "Example: $0 p_4He_pn_3He a 1 1 5.0 400.0 5.0"
    exit 1
fi

REACTION_LABEL="$1"
PROJECTILE="$2"
Z="$3"
A="$4"
START_ENERGY="$5"
END_ENERGY="$6"
STEP_ENERGY="$7"

# One folder per reaction, independent of energy.
# Example: runs/p_56_26/
REACTION_DIR="${RUN_DIR}/${REACTION_LABEL}"

# Clear this reaction's previous generated files before starting.
# This removes old .in, .out, and fort.* files for this reaction only.
rm -rf "$REACTION_DIR"

GEN_DIR="${REACTION_DIR}/generated"
OUT_DIR="${REACTION_DIR}/outputs"
WORK_DIR="${REACTION_DIR}/work"

mkdir -p "$GEN_DIR" "$OUT_DIR" "$WORK_DIR"

run_case() {
    local projectile="$1"
    local z="$2"
    local a="$3"
    local energy="$4"

    local label="${REACTION_LABEL}_${energy}MeV"
    local infile="${GEN_DIR}/${label}.in"
    local outfile="${OUT_DIR}/${label}.out"
    local case_workdir="${WORK_DIR}/${energy}MeV"

    mkdir -p "$case_workdir"

    "$PYTHON" ./scripts/kd_potential.py \
        --projectile "$projectile" \
        --z "$z" \
        --a "$a" \
        --energy "$energy" \
        --template "$TEMPLATE" \
        --output "$infile"

    (
        cd "$case_workdir"
        "$FRESCO" < "$infile" > "$outfile"
    )

    echo "Finished ${label}"
}

for energy in $(seq "$START_ENERGY" "$STEP_ENERGY" "$END_ENERGY"); do
    run_case "$PROJECTILE" "$Z" "$A" "$energy"
done