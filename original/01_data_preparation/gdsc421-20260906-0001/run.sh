set -euo pipefail
base=/public/home/mengxl/creed-research-runs
legacy=/public/home/mengxl/openclaw/dawn/.openclaw/workspace
run="$base/gdsc421-20260906-0001"
docker run --name creed-gdsc421-20260906-0001 --network none --user 0 \
 -e OPENBLAS_NUM_THREADS=4 -e PYTHONPATH=/code/src:/code/scripts \
 --mount type=bind,source="$run/code",target=/code,readonly \
 --mount type=bind,source="$base/data421-0002",target=/data,readonly \
 --mount type=bind,source="$legacy/bioagent/data/raw/gdsc/GDSC2_fitted_dose_response_24Jul22.csv",target=/source.csv,readonly \
 --entrypoint /home/node/.local/bin/python creed-vcp-runtime:py3112-torch220-esm321-v9 \
 -u /code/scripts/prepare_gdsc421.py --data /data --source /source.csv --output /tmp/result
docker cp creed-gdsc421-20260906-0001:/tmp/result "$run/result"
echo COMPLETE_GDSC421
