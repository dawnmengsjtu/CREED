#!/bin/bash
set -euo pipefail
task_root=/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001
previous_root=/public/home/mengxl/creed-research-runs/privileged-geometry-20261008-0001
fm_root=/public/home/mengxl/creed-research-runs/gdsc-metabolite-query-20260908-0001/output/result
source_root=/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result
docker run --rm --network none --user 1011:1011 --cpus 2 --memory 8g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  --entrypoint /home/node/.local/bin/python \
  -v "$task_root:/run" -v "$previous_root:/previous:ro" \
  -v "$fm_root:/fm:ro" -v "$source_root:/source:ro" \
  sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a \
  /run/verification/verify_saved_extra.py --run /run --previous /previous \
  --fm-parent /fm --source /source/source-inputs.npz \
  --report /run/verification/saved_results_verification.json
