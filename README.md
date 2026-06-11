# CCDS: CLIP Class-Consistency and Diversity Selection for Few-shot Classification

## 当前最佳结果（2026-06-11）

当前正式整理版本以本 worktree 为准：

```text
/root/clip_diffusion_fewshot_ccds/.claude/worktrees/anchored-ccds-results
```

全项目最可信主结果已经更新为 Pets20 large-synthetic 设置：

| item | value |
|---|---|
| dataset | Oxford-IIIT Pets20, 20-way 5-shot |
| project | `ccds_pets20_160_sp80_core_realft` |
| method | `margin_topk` |
| candidates/class | 160 |
| selected/class | 80 |
| classifier | frozen-backbone ResNet-50 |
| training | 30 epochs real+synthetic + 5 epochs RealFT |
| accuracy mean | **0.9093050648** |
| macro F1 mean | **0.9086824557** |

关键证据与轻量复查入口：

```bash
# 验证已有 artifacts，并重新生成最终汇总表；默认不重训、不重生成
bash scripts/reproduce_best_pets20_margin_topk.sh

# 单独验证 artifacts
PYTHONPATH=src /root/clip_diffusion_fewshot_ccds/.venv/bin/python scripts/verify_best_pets20_artifacts.py --write-report

# 单独生成 summary/per-seed/vs-baselines 表
PYTHONPATH=src /root/clip_diffusion_fewshot_ccds/.venv/bin/python scripts/summarize_best_pets20_results.py
```

详细交付摘要见 `docs/最终结果摘要.md`。历史 Flowers20 / Anchored CCDS / CFRD-MMR 结果仍保留为研究过程和消融证据，但不再是当前最终最强主结果。


本项目实现一个面向小样本图像分类的扩散生成增强流程：使用 Stable Diffusion 生成类别候选图像，使用 CLIP 计算类别一致性 margin，并结合多样性筛选选择更适合分类训练的生成样本。

## 项目目标

- 在 Oxford Flowers102 子集上构建 few-shot 分类任务；
- 使用 Stable Diffusion v1.5 生成每类候选样本；
- 实现 CLIP Target Score、Confuser Score、Class-Consistency Margin；
- 实现 Random、CLIP Top-K、Margin Top-K、CCDS、Anchored CCDS、Confusion-Adaptive CCDS 等生成样本筛选策略；
- 使用 ResNet-50 frozen backbone 进行分类评估；
- 输出主实验、消融实验和可视化结果。

## 当前新增策略

- `cfrd_mmr`：CLIP-Filtered Real-feature Diverse Selection。CLIP 只做 top-M 语义过滤，最终用真实 few-shot ResNet50 特征接近度和 MMR 类内多样性选样。详见 `docs/cfrd_mmr_strategy.md`。

## 推荐实验设置

```text
数据集：Oxford Flowers102
类别数：20 类主实验，10 类保底
shot：5-shot 主实验，可补 1-shot
候选生成：每类 80 张
最终筛选：每类 10 张
扩散模型：runwayml/stable-diffusion-v1-5
CLIP：open_clip ViT-B/32
分类器：ImageNet pretrained ResNet-50 frozen backbone
```

## 目录结构

```text
clip_diffusion_fewshot_ccds/
├── configs/              # 实验配置
├── data/
│   ├── raw/              # 原始数据或下载缓存说明
│   └── splits/           # few-shot 划分文件
├── generated/            # 扩散生成图像
├── scripts/              # 命令行入口脚本
├── src/ccds/             # 核心 Python 包
├── results/              # CSV、JSON、训练结果
├── figures/              # 可视化图片
├── docs/                 # 项目报告和说明文档
└── notebooks/            # 可选分析 notebook
```

## 推荐执行顺序

1. 准备环境；
2. 创建 few-shot 划分；
3. 训练 Real Only / Traditional Augmentation baseline；
4. 批量生成扩散候选图像；
5. 计算 CLIP 分数和 margin；
6. 生成筛选列表；
7. 训练各增强策略分类器；
8. 汇总结果和可视化。

## 初始命令草案

```bash
# 1. 创建环境后安装依赖
pip install -r requirements.txt

# 2. 创建 few-shot split
python scripts/make_splits.py --config configs/flowers20_5shot.yaml

# 3. 训练 baseline
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method real_only
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method traditional_aug

# 4. 生成扩散候选图像
python scripts/generate_candidates.py --config configs/flowers20_5shot.yaml

# 5. CLIP 打分
python scripts/score_candidates.py --config configs/flowers20_5shot.yaml

# 6. 筛选生成样本
python scripts/select_candidates.py --config configs/flowers20_5shot.yaml --strategy ccds
python scripts/select_candidates.py --config configs/sweeps/pets20_ca2_adaptive_m6_M8_top50_w702010.yaml --strategy confusion_adaptive_ccds
python scripts/select_candidates.py --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml --strategy cfrd_mmr --prototype-seed 0

# 7. 训练增强方法
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method ccds
python scripts/train_classifier.py --config configs/sweeps/pets20_ca2_adaptive_m6_M8_top50_w702010.yaml --method confusion_adaptive_ccds --seed 0
python scripts/train_classifier.py --config configs/sweeps/pets20_cfrd_mmr_top60_resnet.yaml --method cfrd_mmr --seed 0
```

## 当前阶段

当前仓库已经包含可运行的主流程脚本：数据集划分、baseline 训练、扩散候选生成、CLIP 打分、候选筛选、增强训练和结果绘图。

建议先用无扩散 smoke 配置验证工程链路：

```bash
python scripts/run_quick_pipeline.py \
  --config configs/flowers2_1shot_smoke.yaml \
  --real-as-generated \
  --epochs 1 \
  --seed 0
```

该命令使用 Flowers102 held-out 图片模拟生成候选，避免第一次验证时下载和运行 Stable Diffusion。通过后再按 `docs/运行指南.md` 跑正式 20 类实验。

结果现在按 `project_name` 隔离在 `results/<project_name>/` 下，分类器汇总保存在 `results/classifier/all_results.csv`，聚合表保存在 `results/classifier/summary_by_method.csv`。

`confusion_adaptive_ccds`（CA-CCDS）是 `anchored_ccds` 的类别自适应版本：先按每类 CLIP Top-K 的平均 margin 估计类别难度，低 margin 的混淆/困难类别保留更少 CLIP anchor、替换更多样本；高 margin 的容易类别保留更多 CLIP Top-K anchor。因此它实际是在按类别自适应控制与 `clip_topk` 的 overlap。

## 已完成实验摘要

已完成 `configs/flowers20_5shot.yaml` 下的 Flowers20 5-shot 实验：

```text
20 类 × 每类 80 张扩散候选图 = 1600 张生成图
每类选择 10 张生成图用于增强
7 种方法 × 3 个随机种子 × 20 epoch
```

3 个随机种子的聚合结果如下：

| method | accuracy mean | accuracy std | macro F1 mean | macro F1 std |
|---|---:|---:|---:|---:|
| real_only | 0.8344 | 0.0069 | 0.8031 | 0.0142 |
| traditional_aug | 0.8269 | 0.0176 | 0.7993 | 0.0290 |
| diffusion_random | 0.8561 | 0.0073 | 0.8206 | 0.0077 |
| clip_topk | 0.8614 | 0.0125 | 0.8284 | 0.0106 |
| margin_topk | 0.8433 | 0.0054 | 0.8103 | 0.0068 |
| ccds | 0.8508 | 0.0038 | 0.8207 | 0.0047 |
| anchored_ccds | 0.8667 | 0.0114 | 0.8313 | 0.0186 |

结论：扩散增强整体优于 `real_only` 和 `traditional_aug`；新增 `anchored_ccds` 在保留每类 7 张 CLIP Top-K 高置信 anchor 的基础上，用质量-多样性重排序补齐样本，当前 3-seed 平均 Accuracy 达到 0.8667，超过原强 baseline `clip_topk` 的 0.8614。完整 overnight 实验报告保存在本地结果快照：

```text
results/overnight_runs/paid_overnight_20260601_stage80_resume/stage_80cand_20epoch/experiment_summary.md
```

注意：`generated/`、`results/`、`logs/`、checkpoint 和 embedding 文件默认不提交到 GitHub。
