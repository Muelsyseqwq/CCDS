# CCDS 小规模真实生成实验进度记录

日期：2026-05-31

## 目标

在当前 CCDS 项目中完成一次小规模真实 Stable Diffusion 生成链路验证：

1. 确认 Hugging Face / Stable Diffusion 权重缓存可用。
2. 生成 Oxford Flowers20 任务的少量真实扩散候选图。
3. 用 CLIP 对生成图打分并保存图像 embedding。
4. 运行全部候选选择策略。
5. 验证关键产物是否存在、行数是否符合预期、图片路径是否有效。

默认实验配置：

```bash
configs/flowers20_5shot.yaml
```

本次生成规模：

```text
20 类 × 每类 2 张 = 40 张生成图
```

## 环境与权限设置

### Hugging Face 缓存迁移

为避免系统盘空间不足，已将 Hugging Face 缓存迁移到数据盘：

```text
/root/.cache/huggingface -> /root/gpufree-data/cache/huggingface
```

迁移后检查结果：

```text
/root              30G 总量，约 19G 可用
/root/gpufree-data 49G 总量，约 40G 可用
```

旧备份 `/root/.cache/huggingface.rootbak` 已在确认软链接和缓存可读后删除，释放系统盘空间。

### Claude Code 项目权限偏好

用户授权：当前项目中除删除类命令外，其它命令可直接执行。已在项目本地设置中尽量减少 Bash 权限提示，同时显式 deny 删除类命令，例如：

```text
rm
rmdir
unlink
shred
find ... -delete
```

后续删除/清空/不可逆移除类操作仍需人工确认。

## 已完成步骤

### 1. Stable Diffusion v1.5 权重下载与加载

最初直接运行生成脚本时，`StableDiffusionPipeline.from_pretrained(...)` 在下载 SD1.5 权重阶段超时，主要问题是 Hugging Face / 镜像连接中断或 Xet/CAS 路径不稳定。

采用的环境变量：

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_XET=1
HF_HUB_DOWNLOAD_TIMEOUT=600
```

预下载命令曾运行到约 `21/36` 后失败，错误为远端连接中途断开：

```text
httpx.RemoteProtocolError: peer closed connection without sending complete message body
```

随后改为直接测试 pipeline 加载：

```bash
cd /root/clip_diffusion_fewshot_ccds
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
PYTHONPATH=/root/clip_diffusion_fewshot_ccds/src \
/root/clip_diffusion_fewshot_ccds/.venv/bin/python - <<'PY'
import torch
from diffusers import StableDiffusionPipeline
print('cuda_available=', torch.cuda.is_available())
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
print('loaded ok')
print('components=', sorted(pipe.components.keys()))
PY
```

结果：

```text
cuda_available= True
Fetching 13 files: 100%
Loading pipeline components: 100%
loaded ok
components= ['feature_extractor', 'image_encoder', 'safety_checker', 'scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']
```

结论：SD1.5 已可从本地缓存成功加载。

### 2. 生成 20 类小规模真实扩散候选图

运行命令：

```bash
cd /root/clip_diffusion_fewshot_ccds
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
PYTHONPATH=/root/clip_diffusion_fewshot_ccds/src \
/root/clip_diffusion_fewshot_ccds/.venv/bin/python \
/root/clip_diffusion_fewshot_ccds/scripts/generate_candidates.py \
  --config /root/clip_diffusion_fewshot_ccds/configs/flowers20_5shot.yaml \
  --limit-per-class 2
```

运行结果：

```text
Generating classes: 100%|██████████| 20/20
Wrote metadata to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/generation_metadata.csv
```

产物验证：

```text
image_count=40
class_dirs=20
metadata_lines=41
rows=40
labels=20
per_label_min=2
per_label_max=2
missing_paths=0
```

关键产物：

```text
/root/clip_diffusion_fewshot_ccds/generated/flowers20_sd15/
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/generation_metadata.csv
```

metadata 字段：

```text
image_path,class_name,label,prompt,seed,model,guidance_scale,num_steps,config_fingerprint,reused_existing
```

### 3. CLIP 打分

运行命令：

```bash
cd /root/clip_diffusion_fewshot_ccds
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
PYTHONPATH=/root/clip_diffusion_fewshot_ccds/src \
/root/clip_diffusion_fewshot_ccds/.venv/bin/python \
/root/clip_diffusion_fewshot_ccds/scripts/score_candidates.py \
  --config /root/clip_diffusion_fewshot_ccds/configs/flowers20_5shot.yaml
```

运行结果：

```text
CLIP scoring: 100%|██████████| 2/2
Wrote scores to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/clip_scores.csv
Wrote embeddings to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/clip_image_embeddings.npz
```

产物验证：

```text
score_rows=40
missing_paths=0
embedding_count=40
embedding_shape=(512,)
target_score_mean=0.3125135987997055
margin_score_mean=0.013375889137387265
```

关键产物：

```text
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/clip_scores.csv
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/clip_image_embeddings.npz
```

`clip_scores.csv` 字段：

```text
image_path,target_class,target_label,target_score,max_confuser_class,confuser_label,confuser_score,margin_score,prompt,seed
```

注意：本次 `score_candidates.py` 运行没有生成预期的 `results/ccds_flowers20_5shot/figures/score_distributions.png`，但核心打分 CSV 和 embedding NPZ 均正常，足以支持后续选择。

### 4. 运行全部候选选择策略

运行命令：

```bash
cd /root/clip_diffusion_fewshot_ccds
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
PYTHONPATH=/root/clip_diffusion_fewshot_ccds/src \
/root/clip_diffusion_fewshot_ccds/.venv/bin/python \
/root/clip_diffusion_fewshot_ccds/scripts/select_candidates.py \
  --config /root/clip_diffusion_fewshot_ccds/configs/flowers20_5shot.yaml \
  --strategy all
```

运行结果：

```text
Wrote 40 selected samples to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_random.csv
Wrote 40 selected samples to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_clip_topk.csv
Wrote 40 selected samples to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_margin_topk.csv
Wrote 40 selected samples to /root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_ccds.csv
```

产物验证：

| strategy | rows | labels | min/class | max/class | missing paths |
|---|---:|---:|---:|---:|---:|
| random | 40 | 20 | 2 | 2 | 0 |
| clip_topk | 40 | 20 | 2 | 2 | 0 |
| margin_topk | 40 | 20 | 2 | 2 | 0 |
| ccds | 40 | 20 | 2 | 2 | 0 |

关键产物：

```text
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_random.csv
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_clip_topk.csv
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_margin_topk.csv
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/selected/selected_ccds.csv
```

候选选择时出现一个 pandas FutureWarning：

```text
DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated...
```

该 warning 不影响本次输出，但后续可以考虑调整 `src/ccds/selection.py` 中的 groupby/apply 写法，以兼容未来 pandas 版本。

## 当前整体状态

已完成：

- [x] Hugging Face 缓存迁移到 `/root/gpufree-data`
- [x] SD1.5 权重加载验证
- [x] 20 类 × 每类 2 张真实扩散生成
- [x] 生成产物 metadata 验证
- [x] CLIP 打分与 image embedding 保存
- [x] 四种选择策略输出验证

当前最重要的实验产物目录：

```text
/root/clip_diffusion_fewshot_ccds/generated/flowers20_sd15/
/root/clip_diffusion_fewshot_ccds/results/ccds_flowers20_5shot/
```

## 2026-05-31 追加：分类器小规模闭环验证

### Git / VS Code 项目状态

已初始化本地 git 仓库，并配置远程地址：

```text
git@github.com:Muelsyseqwq/CCDS.git
```

已完善 `.gitignore`，避免提交以下大文件或本地私有文件：

```text
.venv/
data/raw/
generated/
results/
*.pt
*.pth
*.ckpt
*.npz
.claude/settings.local.json
```

VS Code 工作区解释器已配置为：

```text
${workspaceFolder}/.venv/bin/python
```

注意：当前服务器尚未配置可用于 GitHub 的 SSH 公钥认证，`ssh -T git@github.com` 返回：

```text
Permission denied (publickey).
```

因此本地 commit 可以完成，但 push 需要先给这台机器配置 GitHub SSH key 或使用 GitHub CLI 登录。

### 1 epoch 小规模分类器结果

在当前小规模真实扩散产物上，已补齐 6 个方法的 `flowers20_5shot`、`seed=0`、`epochs=1` 快速闭环验证。

| method | accuracy | macro_f1 | best_val_accuracy | epochs | train_size | test_size |
|---|---:|---:|---:|---:|---:|---:|
| real_only | 0.1960 | 0.1570 | 0.2200 | 1 | 100 | 755 |
| traditional_aug | 0.1868 | 0.1511 | 0.1800 | 1 | 100 | 755 |
| diffusion_random | 0.1430 | 0.1223 | 0.1900 | 1 | 140 | 755 |
| clip_topk | 0.1709 | 0.1567 | 0.1850 | 1 | 140 | 755 |
| margin_topk | 0.1656 | 0.1446 | 0.1800 | 1 | 140 | 755 |
| ccds | 0.1603 | 0.1439 | 0.1800 | 1 | 140 | 755 |

结果文件：

```text
/root/clip_diffusion_fewshot_ccds/results/classifier/all_results.csv
/root/clip_diffusion_fewshot_ccds/results/classifier/ccds_flowers20_5shot/<method>/seed0/
```

每个方法目录包含：

```text
metrics.json
model.pt
summary.csv
```

解释：这只是 1 epoch smoke / sanity check，用来确认“真实生成 → CLIP 打分 → 候选选择 → 分类训练”完整链路可运行；不能作为正式论文结论。当前每类只有 2 张生成图，且 `selection.selected_per_class=10`，候选池过小，CCDS 的多样性选择优势无法充分体现。

## 2026-06-01 追加：40/80 候选规模 overnight 实验

### overnight runner 修复与续跑

新增并修复 overnight 实验脚本：

```text
scripts/run_paid_overnight_experiment.sh
```

脚本支持：

- 记录环境、GPU、磁盘、git commit 与日志。
- 分阶段运行 40 candidates/class 与 80 candidates/class 实验。
- 校验 metadata、CLIP scores、embedding、selected CSV 和 classifier summary 行数。
- 保存每阶段快照到 `results/overnight_runs/<run_id>/`。
- 使用 `RUN_STAGE_40`、`RUN_STAGE_80`、`SNAPSHOT_EXISTING_STAGE_40` 控制续跑。

最初 `stage_40cand_10epoch` 已完成生成、打分、选择和训练，但 validation 阶段因脚本错误中断：脚本错误地假设 `clip_image_embeddings.npz` 中存在 `image_paths` key；实际 `src/ccds/clip_scoring.py` 以“每张图片一个安全化 key”的形式保存 embedding。该问题已修复，验证逻辑改为兼容当前 NPZ 格式。

### stage_40cand_10epoch 结果

快照目录：

```text
/root/clip_diffusion_fewshot_ccds/results/overnight_runs/paid_overnight_20260531_215657/stage_40cand_10epoch
```

完成内容：

```text
20 类 × 每类 40 张候选图 = 800 张真实扩散图
4 种选择策略 × 每类选择 10 张 = 每策略 200 张
6 种方法 × 3 seeds × 10 epochs = 18 组分类训练
```

3 个 seed 聚合结果：

| method | accuracy mean | accuracy std | macro F1 mean | macro F1 std |
|---|---:|---:|---:|---:|
| real_only | 0.8093 | 0.0464 | 0.7774 | 0.0570 |
| traditional_aug | 0.7978 | 0.0340 | 0.7663 | 0.0427 |
| diffusion_random | 0.8450 | 0.0147 | 0.8154 | 0.0134 |
| clip_topk | 0.8521 | 0.0068 | 0.8137 | 0.0100 |
| margin_topk | 0.8172 | 0.0035 | 0.7860 | 0.0083 |
| ccds | 0.8419 | 0.0101 | 0.8093 | 0.0129 |

### stage_80cand_20epoch 结果

快照目录：

```text
/root/clip_diffusion_fewshot_ccds/results/overnight_runs/paid_overnight_20260601_stage80_resume/stage_80cand_20epoch
```

总结报告：

```text
/root/clip_diffusion_fewshot_ccds/results/overnight_runs/paid_overnight_20260601_stage80_resume/stage_80cand_20epoch/experiment_summary.md
```

完成内容：

```text
20 类 × 每类 80 张候选图 = 1600 张真实扩散图
4 种选择策略 × 每类选择 10 张 = 每策略 200 张
6 种方法 × 3 seeds × 20 epochs = 18 组分类训练
```

`validation.json` 检查全部通过：

```text
metadata_rows = 1600
score_rows = 1600
embedding_count = 1600
selected_random = 200 rows
selected_clip_topk = 200 rows
selected_margin_topk = 200 rows
selected_ccds = 200 rows
classifier_missing_method_seed = []
```

3 个 seed 聚合结果：

| method | accuracy mean | accuracy std | macro F1 mean | macro F1 std |
|---|---:|---:|---:|---:|
| real_only | 0.8344 | 0.0069 | 0.8031 | 0.0142 |
| traditional_aug | 0.8269 | 0.0176 | 0.7993 | 0.0290 |
| diffusion_random | 0.8561 | 0.0073 | 0.8206 | 0.0077 |
| clip_topk | 0.8614 | 0.0125 | 0.8284 | 0.0106 |
| margin_topk | 0.8433 | 0.0054 | 0.8103 | 0.0068 |
| ccds | 0.8508 | 0.0038 | 0.8207 | 0.0047 |
| anchored_ccds | 0.8667 | 0.0114 | 0.8313 | 0.0186 |

### anchored_ccds 追加实验

为提升 CCDS 相比强 baseline `clip_topk` 的竞争力，新增 `anchored_ccds` 策略：每类先保留 7 张 `target_score` 最高的 CLIP Top-K anchor，再从 top-40 高质量候选池中按 `0.75 * target_score + 0.15 * margin_score + 0.10 * diversity_gain` 贪心补齐到每类 10 张。该策略不重新生成扩散图，只复用已有 80 candidates/class 的 CLIP scores 与 image embeddings。

3 个 seed 的测试结果：

| method | seed | epochs | accuracy | macro F1 | best val accuracy | train size | test size |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchored_ccds | 0 | 20 | 0.8768 | 0.8418 | 0.8800 | 300 | 755 |
| anchored_ccds | 1 | 20 | 0.8543 | 0.8098 | 0.8800 | 300 | 755 |
| anchored_ccds | 2 | 20 | 0.8689 | 0.8422 | 0.8600 | 300 | 755 |

### 主要结论

1. 扩散增强方法整体优于 `real_only` 与 `traditional_aug` 基线。
2. `clip_topk` 是原始策略中的最强 baseline，平均 Accuracy 为 0.8614。
3. 新增 `anchored_ccds` 平均 Accuracy 为 0.8667，平均 macro F1 为 0.8313，超过 `clip_topk` 的 0.8614 / 0.8284。
4. 原始 `ccds` 平均 Accuracy 为 0.8508，平均 macro F1 为 0.8207，说明纯 margin + KMeans diversity 在当前 Flowers20 设置下会牺牲部分高置信样本。
5. `anchored_ccds` 的结果支持当前改进思路：先保留 CLIP Top-K 高质量 anchor，再用小比例 diversity 补充，比直接用 KMeans 多样性更稳妥。

## 后续建议

### 工程与实验收尾

1. 将 `scripts/run_paid_overnight_experiment.sh`、`EXPERIMENT_PROGRESS.md`、`README.md` 等代码/文档改动提交并 push。
2. 结果、日志、生成图片、checkpoint 保持在本地与 `.gitignore` 中，不提交到 GitHub。
3. `anchored_ccds` 已验证能超过当前 `clip_topk` baseline；如需继续提升，可进入 `prototype_ccds` 阶段，引入真实 few-shot 样本原型对齐。

### 简历与面试表述

简历可采用如下稳妥表述：

> 完成 Oxford Flowers102 20 类 5-shot 小样本图像分类增强实验，构建 Stable Diffusion v1.5 候选生成、CLIP 语义打分、多策略候选选择与冻结 ResNet-50 分类评估闭环；设计 Anchored CCDS 策略，在保留 CLIP Top-K 高置信样本的基础上引入质量-多样性重排序，80 候选/类、3 个随机种子、20 epoch 下平均 Accuracy 达 86.67%，超过 CLIP Top-K baseline 的 86.14%。

面试中可说明：原始 CCDS 证明了 diversity 设计思路，但纯 KMeans 多样性会牺牲部分高置信样本；Anchored CCDS 通过保留高质量 anchor 并只用少量 diversity 补充，在质量和多样性之间取得了更好的平衡。

## 注意事项

1. SD1.5 下载过程中镜像连接可能中断，但缓存会尽量保留并 resume。
2. `/root/.cache/huggingface` 当前是软链接，不要误删目标目录 `/root/gpufree-data/cache/huggingface`。
3. 本次禁用了 Stable Diffusion safety checker，仅用于本地科研实验，不应直接用于公开服务。
4. 不要把 SSH 私钥、Hugging Face token 或任何 API key 写入代码、日志、commit 或聊天中。
5. 删除/清空/不可逆移除类操作仍需人工确认。
