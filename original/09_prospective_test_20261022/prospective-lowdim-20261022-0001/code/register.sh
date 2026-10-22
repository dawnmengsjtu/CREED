#!/bin/bash
# PROSPECTIVE_LOWDIM_V1 registration (PLAN.md section 12). Run ONCE, after INPUTS.env is filled and PLAN.md is final,
# and before code/run.sh. Writes SHA256SUMS (PLAN.md, INPUTS.env, every file under code/) and PLAN_REGISTERED_AT.txt,
# then makes them read-only. Refuses to run twice. Reads no numeric value of any data file (only file hashes).
set -euo pipefail
RUN=/public/home/mengxl/creed-research-runs/prospective-lowdim-20261022-0001
cd $RUN
for f in SHA256SUMS PLAN_REGISTERED_AT.txt data_gdsc1 dryrun output; do test ! -e $f || { echo "STOP: $f exists"; exit 1; }; done
test -f PLAN.md && ! grep -q '待填' PLAN.md || { echo "STOP: PLAN.md missing or still contains 待填"; exit 1; }
sha256sum -c code/FROZEN_SHA256SUMS
grep -Eq '^GDSC1_SHA256=[0-9a-f]{64}$' INPUTS.env || { echo "STOP: GDSC1_SHA256 not filled in INPUTS.env"; exit 1; }
source ./INPUTS.env
test -f "$GDSC1_FILE" || { echo "STOP: GDSC1_FILE not found: $GDSC1_FILE"; exit 1; }
test -f "$GDSC2_FILE" || { echo "STOP: GDSC2_FILE not found: $GDSC2_FILE"; exit 1; }
test "$(sha256sum "$GDSC1_FILE" | cut -d' ' -f1)" = "$GDSC1_SHA256" || { echo "STOP: GDSC1 sha differs from INPUTS.env"; exit 1; }
test "$(sha256sum "$GDSC2_FILE" | cut -d' ' -f1)" = 9d6dc82b9a072bb00b6139dcd6c4c11de1e047f210382fb726e92ab8dd0eae2a || { echo "STOP: GDSC2 24Jul22 sha"; exit 1; }
if find code -name '__pycache__' | grep -q .; then echo "STOP: remove code/**/__pycache__ first"; exit 1; fi
{ sha256sum PLAN.md INPUTS.env; find code -type f | sort | xargs sha256sum; } > SHA256SUMS
date -Is > PLAN_REGISTERED_AT.txt
chmod a-w PLAN.md INPUTS.env SHA256SUMS PLAN_REGISTERED_AT.txt
find code -type f -exec chmod a-w {} +
chmod a-w code code/frozen
cat SHA256SUMS; echo "registered at $(cat PLAN_REGISTERED_AT.txt)"
