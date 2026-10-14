#!/usr/bin/env bash
set -euo pipefail
CREED_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CREED_PYTHON="${CREED_PYTHON:-/public/home/mengxl/.conda/envs/CellFM/bin/python}"
CREED_RSCRIPT="${CREED_RSCRIPT:-/public/home/mengxl/.conda/envs/creed-r/bin/Rscript}"
[[ -x "$CREED_PYTHON" ]] || CREED_PYTHON="$(command -v python3)"
[[ -x "$CREED_RSCRIPT" ]] || CREED_RSCRIPT="$(command -v Rscript)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
check_delivery() {
  if "$CREED_PYTHON" -c "import fitz, PIL" 2>/dev/null; then "$CREED_PYTHON" "$CREED_SCRIPT_DIR/check_delivery.py"
  else echo "WARN: PyMuPDF (fitz) not importable by $CREED_PYTHON; skipping check_delivery.py (structural PDF QA)" >&2; fi
}
case "${1:-render}" in
  data)
    "$CREED_PYTHON" "$CREED_SCRIPT_DIR/prepare_gdsc.py"
    "$CREED_PYTHON" "$CREED_SCRIPT_DIR/prepare_cptac.py"
    "$CREED_PYTHON" "$CREED_SCRIPT_DIR/prepare_protocol.py"
    ;;
  panels)
    for f in 2 3 4 5 6 7 8 9; do "$CREED_RSCRIPT" "$CREED_SCRIPT_DIR/render_panels.R" "$f"; done
    ;;
  assemble)
    for f in 2 3 4 5 6 7 8 9; do "$CREED_RSCRIPT" "$CREED_SCRIPT_DIR/assemble.R" "$f"; done
    check_delivery
    ;;
  render)
    # For an already reviewed design. Initial work must inspect panels before assembly.
    for f in 2 3 4 5 6 7 8 9; do
      "$CREED_RSCRIPT" "$CREED_SCRIPT_DIR/render_panels.R" "$f"
      "$CREED_RSCRIPT" "$CREED_SCRIPT_DIR/assemble.R" "$f"
    done
    check_delivery
    ;;
  *) echo "Usage: run.sh data|panels|assemble|render" >&2; exit 2 ;;
esac
