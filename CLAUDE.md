# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This repository implements CCDS (CLIP Class-Consistency and Diversity Selection), a few-shot image classification augmentation pipeline for an Oxford Flowers102 subset. The pipeline generates class-conditioned images with Stable Diffusion v1.5, scores candidates with CLIP target/confuser similarities and class-consistency margin, selects generated samples with several strategies, then evaluates them with a frozen-backbone ResNet-50 classifier.

## Current best result and delivery focus

The latest best-result delivery files have been merged from the worktree into this main folder. Current best result:

- Config: `configs/sweeps/pets20_160_sp80_core_realft.yaml`
- Dataset: Oxford-IIIT Pets20, 20-way 5-shot
- Project: `ccds_pets20_160_sp80_core_realft`
- Method: `margin_topk`
- Scale: 160 generated candidates/class, 80 selected/class
- Training: ResNet-50 frozen backbone, 30 epochs real+synthetic + 5 epochs real-only fine-tuning
- 3-seed result: 0.9093050648 accuracy / 0.9086824557 macro F1

Important artifact split:

- CLIP scores, selected CSVs, merged train CSVs, and final research summaries are under this main folder's `results/`.
- Classifier summaries, metrics, and checkpoints for the best result remain in `/root/gpufree-data/clip_diffusion_fewshot_ccds_results/classifier/ccds_pets20_160_sp80_core_realft/margin_topk/seed*/`.

Use lightweight verification by default:

```bash
bash scripts/reproduce_best_pets20_margin_topk.sh
PYTHONPATH=src /root/clip_diffusion_fewshot_ccds/.venv/bin/python scripts/verify_best_pets20_artifacts.py --write-report
PYTHONPATH=src /root/clip_diffusion_fewshot_ccds/.venv/bin/python scripts/summarize_best_pets20_results.py
```

Do not rerun Stable Diffusion generation, CLIP scoring, or 30+5 epoch classifier training unless the user explicitly asks. The reproduction script only retrains when `RUN_HEAVY=1` is set.

Environment notes:

- `requirements.txt` is the human-maintained dependency list.
- `requirements-lock.txt` is a frozen snapshot from the known working virtualenv. GPU/CUDA binary compatibility still depends on the host driver/platform.

## Environment and dependencies

```bash
pip install -r requirements.txt
```

The code is script-based and imports the local package by adding `src` to `sys.path`; there is no package install step, Makefile, pyproject, or lint config in the current repository. Lightweight pytest tests cover dataframe/selection/metric utilities.

First runs may download Flowers102, ResNet-50 weights, CLIP weights, and Stable Diffusion weights. Full diffusion generation is GPU-heavy; local 8GB GPUs are intended for small pipeline checks, while the docs recommend a 24GB GPU for full 20-class x 80-candidate generation.

## Common commands

Use `configs/flowers20_5shot.yaml` as the default experiment config unless intentionally changing the experiment.

```bash
# Create Flowers102 few-shot splits and class map CSVs
python scripts/make_splits.py --config configs/flowers20_5shot.yaml

# Skip automatic dataset download if data is already present
python scripts/make_splits.py --config configs/flowers20_5shot.yaml --no-download

# Train baselines
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method real_only --seed 0
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method traditional_aug --seed 0

# Quick classifier smoke run (closest equivalent to a single test for training code)
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method real_only --seed 0 --epochs 1

# Small local generation smoke run
python scripts/generate_candidates.py --config configs/flowers20_5shot.yaml --limit-per-class 2

# Full/default candidate generation from config
python scripts/generate_candidates.py --config configs/flowers20_5shot.yaml

# Score generated candidates with CLIP
python scripts/score_candidates.py --config configs/flowers20_5shot.yaml

# Select generated samples; `all` writes every configured strategy
python scripts/select_candidates.py --config configs/flowers20_5shot.yaml --strategy all
python scripts/select_candidates.py --config configs/flowers20_5shot.yaml --strategy ccds

# Train classifiers using selected generated samples
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method diffusion_random --seed 0
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method clip_topk --seed 0
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method margin_topk --seed 0
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method ccds --seed 0

# Plot result and score-distribution figures
python scripts/plot_results.py

# Unit tests for utility logic
PYTHONPATH=/root/clip_diffusion_fewshot_ccds/src python -m pytest tests

# Preferred no-diffusion end-to-end smoke pipeline
python scripts/run_quick_pipeline.py --config configs/flowers2_1shot_smoke.yaml --real-as-generated --epochs 1 --seed 0

# Split + real-only baseline smoke
python scripts/run_quick_pipeline.py --skip-generation

# Smoke pipeline including tiny diffusion generation
python scripts/run_quick_pipeline.py
```

Use `pytest tests` for fast utility checks. Use the no-diffusion quick pipeline as the main end-to-end smoke test after pipeline changes.

## Script dependencies and expected outputs

The scripts are ordered; later steps assume artifacts from earlier steps exist:

1. `scripts/make_splits.py` creates `data/splits/flowers{N}_{shot}shot_seed{seed}_{train,val,test}.csv` and `data/splits/flowers{N}_class_map.csv`.
2. `scripts/train_classifier.py --method real_only|traditional_aug` trains on real split CSVs and updates `results/classifier/all_results.csv`.
3. `scripts/generate_candidates.py` reads the class map and writes generated images under `generation.output_dir` plus experiment-specific `results/<project_name>/generation_metadata.csv` unless overridden in config.
4. `scripts/score_candidates.py` reads generation metadata and class map, then writes experiment-specific CLIP scores, image embeddings, and score-distribution figures.
5. `scripts/select_candidates.py` reads CLIP scores; CCDS also requires the experiment-specific embeddings NPZ. It writes `results/<project_name>/selected/selected_{strategy}.csv` and selection grids.
6. `scripts/train_classifier.py --method diffusion_random|clip_topk|margin_topk|ccds` merges the real split with the corresponding selected CSV and writes per-method metrics/checkpoints under `results/classifier/<project_name>/`.
7. `scripts/plot_results.py` reads `results/classifier/all_results.csv` and `results/clip_scores.csv` if present, then writes summary tables and figures.

## Architecture overview

- `configs/flowers20_5shot.yaml` is the central experiment definition: seeds, dataset class count/shot, generation hyperparameters and prompts, CLIP model settings, selection hyperparameters, and classifier settings.
- `src/ccds/config.py` resolves project-relative paths from the repository root and centralizes `project_name`-based artifact paths. Most scripts accept relative paths and pass them through `resolve_path()`.
- `src/ccds/flowers.py` builds few-shot Flowers102 CSV splits. It remaps selected Flowers102 class indices to contiguous labels `0..num_classes-1` and writes a class map used by generation and scoring.
- `src/ccds/diffusion.py` wraps `diffusers.StableDiffusionPipeline` for per-class candidate generation. It enables attention/vae slicing when available, saves images in per-class directories, and records absolute image paths in `results/generation_metadata.csv`.
- `src/ccds/prompts.py` expands class names into the prompt templates from the config; both generation and CLIP text scoring use these templates.
- `src/ccds/clip_scoring.py` loads OpenCLIP, averages prompt-template text features per class, encodes generated images, computes target score, highest non-target confuser score, and margin score, and stores image embeddings for diversity selection.
- `src/ccds/selection.py` implements Random, CLIP Top-K, Margin Top-K, and CCDS. CCDS first takes the top `top_m_for_diversity` margin samples per class, then uses KMeans over CLIP image embeddings and keeps the highest-margin sample per cluster.
- `src/ccds/data.py` provides the CSV-backed image dataset, ImageNet-style transforms, traditional augmentation, and real/generated training CSV merging.
- `scripts/train_classifier.py` builds a torchvision ResNet-50 with ImageNet weights, optionally freezes the backbone according to config, trains/evaluates, and writes checkpoints and JSON/CSV metrics.
- `src/ccds/metrics.py` and `src/ccds/visualize.py` contain classification metrics and plotting helpers used by training, scoring, selection, and plotting scripts.

## Config knobs that commonly change

In `configs/flowers20_5shot.yaml`:

- `dataset.num_classes`, `dataset.shot`, and `dataset.class_indices` define the Flowers102 subset and few-shot size.
- `generation.num_candidates_per_class`, `generation.num_inference_steps`, `generation.guidance_scale`, `generation.dtype`, and prompt templates control candidate generation.
- `selection.selected_per_class` and `selection.top_m_for_diversity` control final augmentation size and CCDS diversity filtering.
- `classifier.epochs`, `classifier.batch_size`, `classifier.lr`, and `classifier.freeze_backbone` control classifier training.

Default artifacts are isolated by `project_name`: metadata/scores/embeddings/selected CSVs live under `results/<project_name>/`, classifier run directories live under `results/classifier/<project_name>/`, and the global classifier summary remains `results/classifier/all_results.csv`.
