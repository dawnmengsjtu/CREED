#!/bin/bash
set -eu
scenario="$1"
case "$scenario" in tissue|new_cell) ;; *) exit 2 ;; esac
task_root=/public/home/mengxl/creed-research-runs/privileged-geometry-20261008-0001
docker run -d --name "creed-geometry-${scenario}-v2-20261008" \
  --network none --user 1011:1011 --cpus 4 --memory 16g \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e PYTHONPATH=/oldcode/src \
  --entrypoint /home/node/.local/bin/python \
  -v /public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result:/parent:ro \
  -v /public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/code:/oldcode:ro \
  -v "$task_root/code:/code:ro" -v "$task_root/output:/out" \
  sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a \
  /code/extend_geometry.py --source /parent/source-inputs.npz --previous "/out/$scenario" --output "/out/${scenario}_v2"
