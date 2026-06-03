# CFRD-MMR strategy

CFRD-MMR means **CLIP-Filtered Real-feature Diverse Selection**. It uses CLIP only as a semantic filter, then selects generated candidates by real few-shot feature similarity and MMR diversity in a ResNet-50 feature space.

## Strategy

For each class:

1. Sort generated candidates by `target_score` and keep `selection.cfrd_clip_top_m` candidates.
2. Extract ResNet-50 penultimate features for the real few-shot training images and generated candidates.
3. Compute `real_sim(x) = max_i cosine(feature(x), feature(real_i))`.
4. Greedily select `selection.selected_per_class` candidates with MMR:

```text
score(x) = cfrd_real_weight * real_sim(x)
         - cfrd_diversity_weight * max_{s in selected} cosine(feature(x), feature(s))
```

The first selected image uses `real_sim` only.

## Config

Example config:

```text
configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml
```

Important fields:

```yaml
selection:
  selected_per_class: 10
  cfrd_clip_top_m: 60
  cfrd_real_weight: 0.70
  cfrd_diversity_weight: 0.30
  cfrd_feature_batch_size: 32
  cfrd_feature_num_workers: 4
  cfrd_features_npz: results/ccds_pets20_cfrd_mmr_top60_resnet/features/cfrd_resnet50_seed{seed}.npz
  strategies:
  - cfrd_mmr
```

The example reuses the existing Pets20 candidates and CLIP scores from `ccds_pets20_5shot`.

## Run selection

Run selection separately for each few-shot seed because real-feature similarity depends on the seed-specific real training split:

```bash
python scripts/select_candidates.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml \
  --strategy cfrd_mmr \
  --prototype-seed 0

python scripts/select_candidates.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml \
  --strategy cfrd_mmr \
  --prototype-seed 1

python scripts/select_candidates.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml \
  --strategy cfrd_mmr \
  --prototype-seed 2
```

Each run writes both:

```text
results/<project_name>/selected/selected_cfrd_mmr_seed{seed}.csv
results/<project_name>/selected/selected_cfrd_mmr.csv
```

The seed-specific file is the one used by `train_classifier.py` when available.

## Train classifier

```bash
python scripts/train_classifier.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml \
  --method cfrd_mmr \
  --seed 0

python scripts/train_classifier.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml \
  --method cfrd_mmr
```

## Diagnostics

Selection writes diagnostics to:

```text
results/research_process/<project_name>_cfrd_diagnostics_seed{seed}.csv
```

The diagnostics include:

- `overlap_with_clip_topk_pct`
- `selected_real_sim_mean`
- `clip_topk_real_sim_mean`
- `selected_redundancy_mean`
- selected vs CLIP Top-K target-score and margin-score means
- per-selected-sample `real_sim`, `redundancy`, and `is_clip_topk`

Use this file to check whether CFRD is actually moving away from CLIP Top-K while increasing real-feature similarity and/or reducing redundancy.

## Diagnostics summary

After selection, summarize diagnostics with:

```bash
python scripts/summarize_cfrd_diagnostics.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml \
  --seed 0
```

This writes:

```text
results/research_process/<project_name>_cfrd_diagnostics_seed{seed}_summary.csv
```

## First experiment matrix

Start small:

| Config | Purpose |
|---|---|
| `configs/sweeps/pets20_cfrd_mmr_top40_resnet.yaml` | More conservative CLIP filter |
| `configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml` | Wider CLIP filter, default CFRD-MMR |
| `configs/sweeps/pets20_cfrd_mmr_top60_resnet_sw075.yaml` | Reuse top60 selected CSV, lower synthetic loss weight to 0.75 |
| `configs/sweeps/pets20_cfrd_mmr_top60_resnet_sw050.yaml` | Reuse top60 selected CSV, lower synthetic loss weight to 0.50 |
| `configs/sweeps/pets20_cfrd_mmr_top60_resnet_realft.yaml` | Reuse top60 selected CSV, then run 5-epoch real-only fine-tuning |

## Real-only fine-tuning stage

`train_classifier.py` supports an optional real-only fine-tuning stage after the main real+synthetic training stage:

```yaml
classifier:
  real_finetune_epochs: 5
  real_finetune_lr: 0.0001
```

The script loads the best validation checkpoint from the main stage, fine-tunes only on the seed-specific real few-shot train split, and keeps updating the best checkpoint by validation accuracy. This is intended to pull the decision boundary back toward the real data distribution after synthetic augmentation.

Run the CFRD top60 real-finetune variant with:

```bash
python scripts/train_classifier.py \
  --config configs/sweeps/pets20_cfrd_mmr_top60_resnet_realft.yaml \
  --method cfrd_mmr \
  --seed 0
```

Compare against:

- `clip_topk`
- `anchored_ccds`
- `reliability_diverse_substitution_ccds` if needed
