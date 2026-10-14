set -euo pipefail
base=/public/home/mengxl/creed-research-runs
run="$base/cptac-domain-experts-20260907-0001"
docker run --name creed-cptac-domain-experts-20260907-0001 --network none --user 0 \
 -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONPATH=/code/src:/code/scripts \
 --mount type=bind,source="$run/code",target=/code,readonly \
 --mount type=bind,source="$base/cptac-three-inputs-20260907-0001/result",target=/prepared,readonly \
 --entrypoint /home/node/.local/bin/python creed-vcp-runtime:py3112-torch220-esm321-v9 \
 -u /code/scripts/run_cptac_domain_experts.py --prepared /prepared --output /tmp/result
docker cp creed-cptac-domain-experts-20260907-0001:/tmp/result "$run/result"
echo COMPLETE_CPTAC_DOMAIN_EXPERTS
