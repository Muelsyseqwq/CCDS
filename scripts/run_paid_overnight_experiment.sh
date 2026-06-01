#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/clip_diffusion_fewshot_ccds"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/flowers20_5shot.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/results/overnight_runs/$RUN_ID"
LOG_FILE="$LOG_DIR/overnight_${RUN_ID}.log"
RUN_STAGE_40="${RUN_STAGE_40:-1}"
RUN_STAGE_80="${RUN_STAGE_80:-1}"
SNAPSHOT_EXISTING_STAGE_40="${SNAPSHOT_EXISTING_STAGE_40:-0}"
METHODS=(real_only traditional_aug diffusion_random clip_topk margin_topk ccds)
SEEDS=(0 1 2)

mkdir -p "$LOG_DIR" "$RUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONPATH="$ROOT/src"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

log() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
  log "RUN: $*"
  "$@"
}

snapshot_stage() {
  local stage_name="$1"
  local candidates="$2"
  local epochs="$3"
  local stage_dir="$RUN_DIR/$stage_name"
  mkdir -p "$stage_dir"

  log "Snapshotting $stage_name outputs to $stage_dir"
  cp "$CONFIG" "$stage_dir/config_used.yaml"
  git rev-parse HEAD > "$stage_dir/git_commit.txt" || true
  git status --short --branch > "$stage_dir/git_status.txt" || true

  "$PYTHON" - "$stage_dir" "$stage_name" "$candidates" "$epochs" <<'PY'
import json
import shutil
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
stage_name = sys.argv[2]
candidates = int(sys.argv[3])
epochs = int(sys.argv[4])
root = Path('/root/clip_diffusion_fewshot_ccds')
project = 'ccds_flowers20_5shot'
methods = ['real_only', 'traditional_aug', 'diffusion_random', 'clip_topk', 'margin_topk', 'ccds']
seeds = [0, 1, 2]

manifest = {
    'stage_name': stage_name,
    'project_name': project,
    'candidates_per_class': candidates,
    'expected_generated_rows': 20 * candidates,
    'selected_per_class': 10,
    'train_epochs': epochs,
    'methods': methods,
    'seeds': seeds,
}
(stage_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

copy_files = [
    root / 'results' / project / 'generation_metadata.csv',
    root / 'results' / project / 'clip_scores.csv',
    root / 'results' / project / 'clip_image_embeddings.npz',
    root / 'results' / 'classifier' / 'all_results.csv',
]
for src in copy_files:
    if src.exists():
        dst = stage_dir / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

selected_dir = root / 'results' / project / 'selected'
if selected_dir.exists():
    dst = stage_dir / 'results' / project / 'selected'
    shutil.copytree(selected_dir, dst, dirs_exist_ok=True)

# Save classifier metrics/summaries but skip large model.pt checkpoints.
clf_root = root / 'results' / 'classifier' / project
for method in methods:
    for seed in seeds:
        src_dir = clf_root / method / f'seed{seed}'
        if not src_dir.exists():
            continue
        dst_dir = stage_dir / 'results' / 'classifier' / project / method / f'seed{seed}'
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ['metrics.json', 'summary.csv']:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)

# Save generated/scoring figures, but not raw generated images.
fig_root = root / 'figures'
if fig_root.exists():
    shutil.copytree(fig_root, stage_dir / 'figures', dirs_exist_ok=True)
PY
}

validate_stage() {
  local stage_name="$1"
  local candidates="$2"
  local epochs="$3"
  local stage_dir="$RUN_DIR/$stage_name"
  mkdir -p "$stage_dir"
  log "Validating $stage_name outputs"
  "$PYTHON" - "$stage_dir" "$stage_name" "$candidates" "$epochs" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

stage_dir = Path(sys.argv[1])
stage_name = sys.argv[2]
candidates = int(sys.argv[3])
epochs = int(sys.argv[4])
root = Path('/root/clip_diffusion_fewshot_ccds')
project = 'ccds_flowers20_5shot'
expected_generated = 20 * candidates
expected_selected = 20 * 10
methods = ['real_only', 'traditional_aug', 'diffusion_random', 'clip_topk', 'margin_topk', 'ccds']
seeds = [0, 1, 2]

checks = {}
meta_path = root / 'results' / project / 'generation_metadata.csv'
score_path = root / 'results' / project / 'clip_scores.csv'
emb_path = root / 'results' / project / 'clip_image_embeddings.npz'
summary_path = root / 'results' / 'classifier' / 'all_results.csv'

meta = pd.read_csv(meta_path)
checks['metadata_rows'] = len(meta)
checks['metadata_expected'] = expected_generated
checks['metadata_missing_paths'] = int((~meta['image_path'].map(lambda p: Path(p).exists())).sum())
checks['metadata_labels'] = int(meta['label'].nunique())
if len(meta) != expected_generated or checks['metadata_missing_paths'] != 0 or checks['metadata_labels'] != 20:
    raise SystemExit(f"Metadata validation failed: {checks}")

scores = pd.read_csv(score_path)
checks['score_rows'] = len(scores)
checks['score_expected'] = expected_generated
checks['score_labels'] = int(scores['target_label'].nunique())
if len(scores) != expected_generated or checks['score_labels'] != 20:
    raise SystemExit(f"Score validation failed: {checks}")

with np.load(emb_path, allow_pickle=True) as emb:
    if 'image_paths' in emb.files:
        embedding_count = len(emb['image_paths'])
        embedding_schema = 'image_paths_array'
    else:
        embedding_count = len(emb.files)
        embedding_schema = 'one_key_per_image'
checks['embedding_schema'] = embedding_schema
checks['embedding_count'] = int(embedding_count)
checks['embedding_expected'] = expected_generated
if checks['embedding_count'] != expected_generated:
    raise SystemExit(f"Embedding validation failed: {checks}")

selected = {}
for strategy in ['random', 'clip_topk', 'margin_topk', 'ccds']:
    p = root / 'results' / project / 'selected' / f'selected_{strategy}.csv'
    df = pd.read_csv(p)
    selected[strategy] = {
        'rows': int(len(df)),
        'labels': int(df['target_label'].nunique()),
        'missing_paths': int((~df['image_path'].map(lambda x: Path(x).exists())).sum()),
    }
    if selected[strategy]['rows'] != expected_selected or selected[strategy]['labels'] != 20 or selected[strategy]['missing_paths'] != 0:
        raise SystemExit(f"Selection validation failed for {strategy}: {selected[strategy]}")
checks['selected'] = selected

summary = pd.read_csv(summary_path)
sub = summary[(summary['project_name'] == project) & (summary['epochs'] == epochs)]
found = set(zip(sub['method'].astype(str), sub['seed'].astype(int)))
expected = {(m, s) for m in methods for s in seeds}
missing = sorted(expected - found)
checks['classifier_rows_for_epoch'] = int(len(sub))
checks['classifier_missing_method_seed'] = missing
if missing:
    raise SystemExit(f"Classifier validation failed; missing {missing}")

validation_path = stage_dir / 'validation.json'
validation_path.write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
PY
}

train_grid() {
  local epochs="$1"
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_cmd "$PYTHON" scripts/train_classifier.py \
        --config "$CONFIG" \
        --method "$method" \
        --seed "$seed" \
        --epochs "$epochs"
    done
  done
}

run_stage() {
  local stage_name="$1"
  local candidates="$2"
  local epochs="$3"
  log "===== START $stage_name: candidates_per_class=$candidates train_epochs=$epochs ====="

  run_cmd "$PYTHON" scripts/generate_candidates.py --config "$CONFIG" --limit-per-class "$candidates"
  run_cmd "$PYTHON" scripts/score_candidates.py --config "$CONFIG"
  run_cmd "$PYTHON" scripts/select_candidates.py --config "$CONFIG" --strategy all
  train_grid "$epochs"
  validate_stage "$stage_name" "$candidates" "$epochs"
  snapshot_stage "$stage_name" "$candidates" "$epochs"

  log "===== DONE $stage_name ====="
}

snapshot_existing_stage() {
  local stage_name="$1"
  local candidates="$2"
  local epochs="$3"
  log "===== SNAPSHOT EXISTING $stage_name: candidates_per_class=$candidates train_epochs=$epochs ====="
  validate_stage "$stage_name" "$candidates" "$epochs"
  snapshot_stage "$stage_name" "$candidates" "$epochs"
  log "===== DONE SNAPSHOT EXISTING $stage_name ====="
}

log "Overnight CCDS experiment started"
log "RUN_ID=$RUN_ID"
log "ROOT=$ROOT"
log "CONFIG=$CONFIG"
log "RUN_DIR=$RUN_DIR"
log "LOG_FILE=$LOG_FILE"
log "RUN_STAGE_40=$RUN_STAGE_40"
log "RUN_STAGE_80=$RUN_STAGE_80"
log "SNAPSHOT_EXISTING_STAGE_40=$SNAPSHOT_EXISTING_STAGE_40"
log "Git commit: $(git rev-parse HEAD || true)"
log "Git status:"
git status --short --branch || true
log "Disk usage:"
df -h /root /root/gpufree-data || true
log "GPU info:"
nvidia-smi || true
log "Python environment:"
"$PYTHON" - <<'PY'
import sys
import torch
print('python=', sys.executable)
print('torch=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
PY

if [[ "$SNAPSHOT_EXISTING_STAGE_40" == "1" ]]; then
  snapshot_existing_stage "stage_40cand_10epoch" 40 10
elif [[ "$RUN_STAGE_40" == "1" ]]; then
  run_stage "stage_40cand_10epoch" 40 10
else
  log "Skipping stage_40cand_10epoch"
fi

if [[ "$RUN_STAGE_80" == "1" ]]; then
  run_stage "stage_80cand_20epoch" 80 20
else
  log "Skipping stage_80cand_20epoch"
fi

log "Final result table for ccds_flowers20_5shot:"
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
p = Path('/root/clip_diffusion_fewshot_ccds/results/classifier/all_results.csv')
df = pd.read_csv(p)
sub = df[df['project_name'] == 'ccds_flowers20_5shot'].copy()
sub = sub.sort_values(['epochs', 'seed', 'method'])
print(sub[['method','seed','epochs','accuracy','macro_f1','best_val_accuracy','train_size','test_size']].to_csv(index=False))
PY

log "Overnight CCDS experiment finished successfully"
log "Snapshots saved under $RUN_DIR"
