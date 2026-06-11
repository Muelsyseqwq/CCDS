# CLIP-Guided Diffusion Sample Selection for Few-Shot Image Classification

This repository contains a research-oriented implementation of diffusion-based data augmentation for few-shot image classification. The project originated from CCDS (CLIP Class-Consistency and Diversity Selection), and has since grown into a broader codebase for studying CLIP-guided synthetic sample selection strategies, including margin-based selection, CCDS, anchored CCDS, confusion-adaptive CCDS, and real-feature diversity variants.

The current strongest verified result is obtained on a 20-class subset of Oxford-IIIT Pets under a 5-shot setting. Importantly, this best result is achieved by the `margin_topk` large-candidate configuration with real-only fine-tuning, not by the original CCDS variant. The best 3-seed mean accuracy is **90.93%**.

## Highlights

- Implements an end-to-end few-shot augmentation workflow: split creation, diffusion candidate generation, CLIP scoring, candidate selection, classifier training, and result summarization.
- Treats CCDS as one member of a broader selection-method family rather than as the sole final method.
- Supports multiple selection strategies, including `random`, `clip_topk`, `margin_topk`, `ccds`, `anchored_ccds`, `confusion_adaptive_ccds`, and `cfrd_mmr`.
- Provides reproducible configuration files for Flowers20 and Pets20 experiments.
- Archives the current best Pets20 5-shot result from `margin_topk` large-candidate selection with lightweight summary tables committed to the repository.
- Keeps large generated images, model checkpoints, embeddings, and full result artifacts out of Git by default.

## Method Overview

The project is built around the observation that not every diffusion-generated image is useful for few-shot classifier training. It therefore separates generation from selection and evaluates several selection rules under the same training pipeline:

1. **Candidate generation.** Stable Diffusion v1.5 produces multiple class-conditioned images for each target class.
2. **Semantic scoring.** CLIP text-image similarity is used to compute:
   - target-class score,
   - strongest non-target confuser score,
   - class-consistency margin.
3. **Candidate selection.** Different strategies select synthetic samples from the candidate pool. CCDS combines class-consistency and diversity, while the current best configuration uses `margin_topk`, which selects candidates with the strongest target-vs-confuser margin from a larger candidate pool.
4. **Classifier evaluation.** A ResNet-50 classifier with an ImageNet-pretrained frozen backbone is trained on the real few-shot set plus selected synthetic samples. The best setting uses an additional real-only fine-tuning stage.

## Current Best Result

The best verified setting is:

| Item | Value |
|---|---|
| Dataset | Oxford-IIIT Pets20 |
| Task | 20-way 5-shot classification |
| Project name | `ccds_pets20_160_sp80_core_realft` |
| Selection method | `margin_topk` |
| Generated candidates | 160 per class |
| Selected synthetic samples | 80 per class |
| Classifier | ResNet-50, frozen backbone |
| Training | 30 epochs real+synthetic + 5 epochs real-only fine-tuning |
| Config | `configs/sweeps/pets20_160_sp80_core_realft.yaml` |

### Aggregate Result

| Setting | Method | Seeds | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std |
|---|---|---:|---:|---:|---:|---:|
| Pets20 160/sp80 + RealFT | `margin_topk` | 3 | **0.9093** | 0.0023 | **0.9087** | 0.0019 |

### Per-Seed Result

| Seed | Accuracy | Macro F1 | Best val accuracy | Train size | Test size |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9107 | 0.9098 | 0.9150 | 1700 | 1981 |
| 1 | 0.9066 | 0.9065 | 0.9250 | 1700 | 1981 |
| 2 | 0.9107 | 0.9098 | 0.9000 | 1700 | 1981 |

### Comparison with Representative Baselines

| Setting | Method | Selected/class | Seeds | Accuracy mean | Macro F1 mean | Delta acc. vs best |
|---|---|---:|---:|---:|---:|---:|
| Pets20 fair RealFT | `real_only` | 0 | 3 | 0.8127 | 0.8064 | -0.0966 |
| Pets20 fair RealFT | `traditional_aug` | 0 | 3 | 0.8048 | 0.7974 | -0.1045 |
| Pets20 fair RealFT | `diffusion_random` | 10 | 3 | 0.8713 | 0.8694 | -0.0380 |
| Pets20 fair RealFT | `clip_topk` | 10 | 3 | 0.8992 | 0.8982 | -0.0101 |
| Pets20 fair RealFT | `anchored_ccds` | 10 | 3 | 0.8957 | 0.8946 | -0.0136 |
| Pets20 D7 anchored + RealFT | `anchored_ccds` | 10 | 3 | 0.8999 | 0.8993 | -0.0094 |
| Pets20 160/sp60 + RealFT | `ccds` | 60 | 3 | 0.9073 | 0.9066 | -0.0020 |
| Pets20 160/sp80 + RealFT | `margin_topk` | 80 | 3 | **0.9093** | **0.9087** | 0.0000 |
| Pets20 160/sp100 + RealFT | `margin_topk` | 100 | 3 | 0.8984 | 0.8978 | -0.0109 |
| Pets20 160/sp100 + RealFT | `anchored_ccds` | 100 | 3 | 0.9039 | 0.9032 | -0.0054 |
| Pets20 160/sp80 + RealFT | `cfrd_mmr` | 80 | 1 | 0.9051 | 0.9045 | -0.0042 |

The committed lightweight result files are located under:

```text
results/research_process/
├── pets20_160_sp80_core_realft_best_summary.csv
├── pets20_160_sp80_core_realft_per_seed.csv
├── pets20_160_sp80_core_realft_vs_baselines.csv
├── pets20_160_sp80_core_realft_tables.md
└── pets20_160_sp80_core_realft_verification.txt
```

Large artifacts such as full generated images, CLIP embeddings, classifier checkpoints, and complete training outputs are intentionally not committed.

## Reproducing the Best Result Summary

The default reproduction command is intentionally lightweight. It checks existing artifacts and regenerates summary tables; it does **not** rerun Stable Diffusion generation, CLIP scoring, or the full 30+5 epoch classifier training.

```bash
bash scripts/reproduce_best_pets20_margin_topk.sh
```

Expected verification summary:

```text
Verification: PASS
accuracy_mean=0.9093050648
macro_f1_mean=0.9086824557
```

To run the individual steps:

```bash
# Verify that the required artifacts exist and match the expected result.
PYTHONPATH=src /root/clip_diffusion_fewshot_ccds/.venv/bin/python \
  scripts/verify_best_pets20_artifacts.py --write-report

# Regenerate compact summary tables.
PYTHONPATH=src /root/clip_diffusion_fewshot_ccds/.venv/bin/python \
  scripts/summarize_best_pets20_results.py
```

To rerun the three classifier training seeds for the best setting, explicitly opt in to the heavier run:

```bash
RUN_HEAVY=1 bash scripts/reproduce_best_pets20_margin_topk.sh
```

Full regeneration of diffusion candidates and CLIP scores is heavier still and should be launched manually only when needed.

## Installation

```bash
pip install -r requirements.txt
```

The project is script-based. Most commands should be run from the repository root with `PYTHONPATH=src` when importing the local `ccds` package directly.

A frozen environment snapshot from the known working virtual environment is provided in:

```text
requirements-lock.txt
```

GPU/CUDA binary compatibility still depends on the host driver and platform.

## Repository Structure

```text
clip_diffusion_fewshot_ccds/
├── configs/                  # Experiment configurations and sweeps
├── data/splits/              # Lightweight few-shot split CSVs committed for reproducibility
├── docs/                     # Research notes, handoff summaries, and method documentation
├── figures/                  # Selected lightweight figures for reporting
├── results/research_process/ # Small committed summary tables for the best result
├── scripts/                  # Command-line entry points
├── src/ccds/                 # Core implementation
└── tests/                    # Unit tests for data, metrics, and selection utilities
```

Important implementation files:

- `src/ccds/selection.py`: selection strategies, including margin-based, CCDS, anchored, and adaptive variants.
- `src/ccds/clip_scoring.py`: CLIP feature extraction and target/confuser/margin scoring.
- `src/ccds/diffusion.py`: Stable Diffusion candidate generation wrapper.
- `src/ccds/data.py`: CSV-backed dataset construction and real/synthetic train-set merging.
- `src/ccds/pets.py`: Oxford-IIIT Pets split construction.
- `scripts/train_classifier.py`: frozen-backbone ResNet-50 training and evaluation.

## Common Workflows

### Quick Engineering Smoke Test

Use a no-diffusion smoke configuration to verify the pipeline logic without downloading or running Stable Diffusion:

```bash
python scripts/run_quick_pipeline.py \
  --config configs/flowers2_1shot_smoke.yaml \
  --real-as-generated \
  --epochs 1 \
  --seed 0
```

### Create Few-Shot Splits

```bash
python scripts/make_splits.py --config configs/pets20_5shot.yaml
```

### Generate Candidates

```bash
python scripts/generate_candidates.py --config configs/pets20_5shot.yaml
```

### Score Candidates with CLIP

```bash
python scripts/score_candidates.py --config configs/pets20_5shot.yaml
```

### Select Synthetic Samples

```bash
python scripts/select_candidates.py \
  --config configs/sweeps/pets20_160_sp80_core_realft.yaml \
  --strategy margin_topk
```

### Train a Classifier

```bash
python scripts/train_classifier.py \
  --config configs/sweeps/pets20_160_sp80_core_realft.yaml \
  --method margin_topk \
  --seed 0
```

## Tests

```bash
PYTHONPATH=/root/clip_diffusion_fewshot_ccds/src \
  /root/clip_diffusion_fewshot_ccds/.venv/bin/python -m pytest tests
```

Current checked test status:

```text
15 passed
```

## Artifact and Version-Control Policy

The repository is intended to stay lightweight and reproducible. Therefore:

- Committed: source code, experiment configs, few-shot split CSVs, documentation, selected figures, and compact result summary tables.
- Not committed by default: `generated/`, full `results/`, CLIP embeddings, model checkpoints, logs, and raw datasets.
- Best-result classifier metrics and checkpoints used for verification are stored locally under `/root/gpufree-data/clip_diffusion_fewshot_ccds_results/`.
- The lightweight verification script checks these local artifacts when available.

This policy avoids pushing large binary artifacts while still preserving the code, configuration, and numerical summary needed to understand and reproduce the reported result.

## Notes on Interpretation

The repository name and historical documentation still use CCDS because the project started from CLIP Class-Consistency and Diversity Selection. In the current codebase, however, CCDS should be read as part of a broader family of CLIP-guided diffusion sample selection methods. The best verified Pets20 result reported here is specifically the `margin_topk` large-candidate setting with real-only fine-tuning, not the original CCDS method.

The current best result is a 3-seed empirical result on a Pets20 5-shot subset. It should be treated as an experimental result for a specific configuration, not as a general claim across all datasets or backbones. The comparison table includes both complete 3-seed settings and one single-seed diagnostic result; the latter is marked accordingly.

## License and Citation

No formal paper citation is currently provided. If this repository is used in a report or derivative project, cite the repository and the exact configuration file used for the reported experiment.
