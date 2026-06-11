#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/clip_diffusion_fewshot_ccds}"
PYTHON="${PYTHON:-/root/clip_diffusion_fewshot_ccds/.venv/bin/python}"
CONFIG="${CONFIG:-configs/sweeps/pets20_160_sp80_core_realft.yaml}"
METHOD="${METHOD:-margin_topk}"
RUN_HEAVY="${RUN_HEAVY:-0}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

printf '\n[1/2] Verifying existing best-result artifacts...\n'
"$PYTHON" scripts/verify_best_pets20_artifacts.py --config "$CONFIG" --write-report

printf '\n[2/2] Regenerating best-result summary tables...\n'
"$PYTHON" scripts/summarize_best_pets20_results.py --config "$CONFIG"

cat <<'MSG'

Default reproduction is intentionally lightweight:
- It verifies existing generated images, CLIP scores, selected CSVs, merged train CSVs, summaries, metrics, and checkpoints.
- It regenerates research_process summary tables.
- It does NOT rerun Stable Diffusion, CLIP scoring, or 30+5 epoch classifier training.

To rerun classifier training for the three best-result seeds, explicitly set RUN_HEAVY=1:
  RUN_HEAVY=1 bash scripts/reproduce_best_pets20_margin_topk.sh

Full regeneration of Stable Diffusion candidates and CLIP scores is heavier still and should be run manually only when needed.
MSG

if [[ "$RUN_HEAVY" == "1" ]]; then
  printf '\nRUN_HEAVY=1 detected. Rerunning classifier training for seeds 0, 1, 2.\n'
  for seed in 0 1 2; do
    "$PYTHON" scripts/train_classifier.py --config "$CONFIG" --method "$METHOD" --seed "$seed"
  done
else
  printf '\nRUN_HEAVY is not 1; skipping heavy classifier retraining.\n'
fi
