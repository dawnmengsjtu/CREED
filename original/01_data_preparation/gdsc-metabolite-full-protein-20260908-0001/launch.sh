set -euo pipefail
mode="$1"
base=/public/home/mengxl/creed-research-runs
parent="$base/gdsc-metabolite-query-20260908-0001/output/result"
receipt="$base/gdsc-metabolite-query-verify-20260908-0001/output/verified.json"
protein="$base/protein421-full-20260907-0001/result"
producer="$base/gdsc-metabolite-full-protein-20260908-0001"
mounts=()
if test "$mode" = producer; then
  run="$producer"
  name=creed-gdsc-metabolite-full-protein-20260908-0001
  script=/code/scripts/run_gdsc_metabolite_full_protein.py
  args=(--output /output/result)
elif test "$mode" = verifier; then
  run="$base/gdsc-metabolite-full-protein-verify-20260908-0001"
  name=creed-gdsc-metabolite-full-protein-verify-20260908-0001
  test "$(cat "$producer/exit-status.txt")" = 0
  script=/code/scripts/verify_gdsc_metabolite_full_protein.py
  args=(--result /result --code-root /producer-code --output /output/verified.json)
  mounts+=(--mount "type=bind,source=$producer/output/result,target=/result,readonly")
  mounts+=(--mount "type=bind,source=$producer/code,target=/producer-code,readonly")
else
  exit 64
fi
files=(summary.json source-inputs.npz splits.json)
for fold in 0 1 2 3 4; do
  files+=("fold-$fold/selection.json" "fold-$fold/predictions.npz")
done
for file in "${files[@]}"; do
  mounts+=(--mount "type=bind,source=$parent/$file,target=/parent/$file,readonly")
done
for file in manifest.json proteins.npy protein_ids.npy sample_ids.npy; do
  mounts+=(--mount "type=bind,source=$protein/$file,target=/protein/$file,readonly")
done
set +e
docker run --name "$name" --network none --user 0 --cpus 4 --memory 8g \
  -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONPATH=/code/src:/code/scripts \
  --mount "type=bind,source=$run/code,target=/code,readonly" \
  --mount "type=bind,source=$run/output,target=/output" \
  --mount "type=bind,source=$receipt,target=/parent-verification.json,readonly" "${mounts[@]}" \
  --entrypoint /home/node/.local/bin/python sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a \
  -u "$script" --parent /parent --verification /parent-verification.json --protein /protein \
  --source-bindings /code/GDSC_METABOLITE_FULL_PROTEIN_BINDINGS.json "${args[@]}"
status=$?
printf '%s\n' "$status" > "$run/exit-status.txt"
printf 'GDSC_METABOLITE_FULL_PROTEIN_%s_EXIT=%s\n' "$mode" "$status"
exit "$status"
