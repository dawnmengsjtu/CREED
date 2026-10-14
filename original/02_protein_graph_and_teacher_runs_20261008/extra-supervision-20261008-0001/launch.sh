#!/bin/bash
set -eu
scenario="$1"
case "$scenario" in tissue|new_cell) ;; *) exit 2 ;; esac
task_root=/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001
docker run -d --name "creed-extra-${scenario}-20261008" \
  --network none --user 1011:1011 --cpus 4 --memory 16g \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e PYTHONPATH=/oldcode/src \
  --entrypoint /home/node/.local/bin/python \
  -v /public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result:/parent:ro \
  -v /public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/code:/oldcode:ro \
  -v /public/home/mengxl/creed-research-runs/privileged-geometry-20261008-0001/output:/previous:ro \
  -v /public/home/mengxl/creed-research-runs/gdsc-metabolite-query-20260908-0001/output/result:/fmparent:ro \
  -v "$task_root/resources:/resources:ro" -v "$task_root/code:/code:ro" -v "$task_root/output:/out" \
  sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a \
  /code/run_extra.py --source /parent/source-inputs.npz --previous "/previous/${scenario}_v2" \
  --extra /resources/m_gdsc_extra19.npz --extra-sha d4e7dc603f491217805a1d06b673f8c2662a8c0f2ad77ec875f7d7868526384c \
  --fm-parent /fmparent --output "/out/$scenario"
