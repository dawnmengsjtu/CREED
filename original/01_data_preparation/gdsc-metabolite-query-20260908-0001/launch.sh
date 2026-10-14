set -euo pipefail
mode="$1"
base=/public/home/mengxl/creed-research-runs
producer="$base/gdsc-metabolite-query-20260908-0001"
mounts=()
if test "$mode" = producer; then
  run="$producer"
  name=creed-gdsc-metabolite-query-20260908-0001
  script=/code/scripts/run_gdsc_metabolite_query.py
  args=(--output /output/result)
elif test "$mode" = verifier; then
  run="$base/gdsc-metabolite-query-verify-20260908-0001"
  name=creed-gdsc-metabolite-query-verify-20260908-0001
  test "$(cat "$producer/exit-status.txt")" = 0
  script=/code/scripts/verify_gdsc_metabolite_query.py
  args=(--result /result --code-root /producer-code --output /output/verified.json)
  mounts+=(--mount "type=bind,source=$producer/output/result,target=/result,readonly")
  mounts+=(--mount "type=bind,source=$producer/code,target=/producer-code,readonly")
else
  exit 64
fi
for file in manifest.json metabolites.npy metabolite_ids.npy sample_ids.npy scgpt.npy; do
  mounts+=(--mount "type=bind,source=$base/data421-0002/$file,target=/data/$file,readonly")
done
for file in manifest.json rna_log_cpm.npy gene_ids.npy sample_ids.npy; do
  mounts+=(--mount "type=bind,source=$base/rna421-20260906-0002/$file,target=/rna/$file,readonly")
done
for file in manifest.json auc.npy drug_ids.npy sample_ids.npy; do
  mounts+=(--mount "type=bind,source=$base/gdsc421-20260906-0001/result/$file,target=/gdsc/$file,readonly")
done
for alias in reference measured-reference; do
  if test "$alias" = reference; then
    source="$base/gdsc-virtual-utility-20260906-0001/result"
  else
    source="$base/gdsc-measured-full-20260907-0002/result"
  fi
  files=(splits.json result.json)
  for fold in 0 1 2 3 4; do
    files+=("selection-$fold.json" "predictions-$fold.npz")
  done
  for file in "${files[@]}"; do
    mounts+=(--mount "type=bind,source=$source/$file,target=/$alias/$file,readonly")
  done
done
set +e
docker run --name "$name" --network none --user 0 --cpus 4 --memory 8g \
  -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONPATH=/code/src:/code/scripts \
  --mount "type=bind,source=$run/code,target=/code,readonly" \
  --mount "type=bind,source=$run/output,target=/output" "${mounts[@]}" \
  --entrypoint /home/node/.local/bin/python sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a \
  -u "$script" --data /data --rna /rna --gdsc /gdsc --reference /reference \
  --measured-reference /measured-reference --source-bindings /code/GDSC_METABOLITE_QUERY_BINDINGS.json "${args[@]}"
status=$?
printf '%s\n' "$status" > "$run/exit-status.txt"
printf 'GDSC_METABOLITE_QUERY_%s_EXIT=%s\n' "$mode" "$status"
exit "$status"
