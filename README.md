# CCDS: CLIP Class-Consistency and Diversity Selection for Few-shot Classification

本项目实现一个面向小样本图像分类的扩散生成增强流程：使用 Stable Diffusion 生成类别候选图像，使用 CLIP 计算类别一致性 margin，并结合多样性筛选选择更适合分类训练的生成样本。

## 项目目标

- 在 Oxford Flowers102 子集上构建 few-shot 分类任务；
- 使用 Stable Diffusion v1.5 生成每类候选样本；
- 实现 CLIP Target Score、Confuser Score、Class-Consistency Margin；
- 实现 Random、CLIP Top-K、Margin Top-K、CCDS 四种生成样本筛选策略；
- 使用 ResNet-50 frozen backbone 进行分类评估；
- 输出主实验、消融实验和可视化结果。

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

# 7. 训练增强方法
python scripts/train_classifier.py --config configs/flowers20_5shot.yaml --method ccds
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

## 已完成实验摘要

已完成 `configs/flowers20_5shot.yaml` 下的 Flowers20 5-shot 实验：

```text
20 类 × 每类 80 张扩散候选图 = 1600 张生成图
每类选择 10 张生成图用于增强
6 种方法 × 3 个随机种子 × 20 epoch
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

结论：扩散增强整体优于 `real_only` 和 `traditional_aug`；当前配置下 `clip_topk` 取得最高平均 Accuracy，CCDS 在多随机种子下取得稳定提升。完整实验报告保存在本地结果快照：

```text
results/overnight_runs/paid_overnight_20260601_stage80_resume/stage_80cand_20epoch/experiment_summary.md
```

注意：`generated/`、`results/`、`logs/`、checkpoint 和 embedding 文件默认不提交到 GitHub。
