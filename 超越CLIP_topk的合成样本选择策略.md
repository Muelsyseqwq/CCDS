# 超越 CLIP Top-K 的合成样本选择策略

> 适用项目：Stable Diffusion 生成候选图像 → CLIP 打分选择 top-k → 使用选中样本训练 / 微调 ResNet50 few-shot 图像分类器。

## 0. 研究问题

当前 pipeline 中，CLIP top-k 选择生成样本的效果很强。问题是：**有没有 synthetic sample selection/filtering 策略，可以超过或更稳定地优于单纯 CLIP top-k？**

结论先说：**想“超过 CLIP top-k”，不要直接抛弃 CLIP，而是把 CLIP 从“唯一排序器”改成“语义/质量门槛”，再叠加真实样本分布、特征多样性、目标模型反馈和 curriculum。**

单纯 CLIP top-k 很强，是因为它很好地过滤了语义错误和低质量图；但它也容易选出“最像文本提示词的标准图”，导致样本同质化、分布偏、缺少决策边界附近的有效样本，最终不一定最适合 ResNet50。

---

## 1. 为什么 CLIP top-k 很难打败？

相关代表是 **CaFo**：[*Prompt, Generate, Then Cache: Cascade of Foundation Models Makes Strong Few-Shot Learners*](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.html)，CVPR 2023。它使用生成模型产生图片，再用 **CLIP 过滤 top-K′ generated images**，然后结合 CLIP/DINO cache 做 few-shot learning。这个思路和当前 pipeline 非常接近。

CLIP top-k 强的原因：

1. **语义一致性强**：能去掉和类别文本不匹配的生成图。
2. **噪声过滤强**：生成图里有大量伪影、错类、奇怪构图，CLIP 能快速筛掉。
3. **few-shot 场景下 label noise 代价很大**：错一个合成样本可能比少一个样本更伤。
4. **实现简单稳定**：无须训练额外选择器。

但它的核心缺点是：

> CLIP top-k 选的是“最符合文本的图”，不一定是“最能提升 ResNet50 分类边界的图”。

所以超过它的方向不是“找另一个单分数模型替代 CLIP”，而是做 **多目标选择**。

---

## 2. 最推荐的策略：CLIP 门槛 + 真实样本特征距离 + 多样性选择

### 2.1 核心思想

先用 CLIP 过滤掉明显错类样本，然后在剩下的候选图中，优先选择：

- 和 few-shot 真实样本在视觉特征空间中接近；
- 但彼此之间不要太像；
- 能覆盖该类别内部不同模式。

这比直接选 CLIP top-k 更可能有效，因为它同时控制：

| 目标 | CLIP top-k | 改进策略 |
|---|---:|---:|
| 语义正确 | 强 | 强 |
| 接近真实训练分布 | 不一定 | 更强 |
| 类内多样性 | 弱 | 强 |
| 对 ResNet50 有用 | 间接 | 更直接 |

### 2.2 相关论文

1. **CaFo** 使用 CLIP top-K′ 过滤生成图，是当前 baseline 的强相关论文：
   [Prompt, Generate, Then Cache, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.html)

2. **Rare plant classification** 论文提出了 **Real Feature Filtering, RFF**：用少量真实样本的特征去过滤离真实分布太远的合成图。
   [Can Synthetic Plant Images From Generative Models Facilitate Rare Species Identification and Classification?, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/GCV/html/Dasgupta_Can_Synthetic_Plant_Images_From_Generative_Models_Facilitate_Rare_Species_CVPRW_2024_paper.html)

3. **DINOv2** 适合做视觉特征空间 backbone，因为它提供了强通用视觉表征：
   [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

4. **Chamfer Guidance** 这类工作也说明，用真实样本和合成样本在 DINOv2 等特征空间里的距离来引导生成/选择，可能提升 synthetic data utility：
   [Increasing the Utility of Synthetic Images through Chamfer Guidance](https://openreview.net/forum?id=X3RVQNOIXZ)

### 2.3 可落地做法

假设每个类别生成 `N=200~1000` 张候选图，最终选 `k=16/32/64` 张。

对每个候选图计算：

- `s_clip`：CLIP image-text 相似度；
- `s_real`：候选图到该类 few-shot 真实图的平均/最大相似度，特征可用 DINOv2、ResNet50 penultimate layer 或 CLIP image encoder；
- `s_div`：与已选样本的去冗余项；
- `s_quality`：aesthetic score、生成质量分、低分辨率/伪影过滤等，可选。

一个简单有效的选择分数：

```text
score(x) = α * CLIP_text_sim(x, class)
         + β * max_sim_to_real_features(x, real_class_images)
         - γ * max_sim_to_selected(x, selected_images)
```

推荐初始权重：

```text
α = 0.4
β = 0.4
γ = 0.2
```

更简单一点：**CLIP 只做阈值，不做最终排序**。

```text
Step 1: 保留 CLIP 分数 top 30% 或超过阈值的图
Step 2: 在保留图中，用 DINOv2/ResNet 特征做 k-center / k-medoids / MMR 选样
```

这通常比“直接 CLIP top-k”更容易超过 baseline，因为它解决了 top-k 同质化问题。

---

## 3. 第二优先级：CLIP top-k 改成“每个 cluster 选 top-m”

### 3.1 核心思想

CLIP top-k 经常会选到同一类视觉模式。例如“狗”全是正面头像，“鸟”全是居中清晰侧身，“花”全是单朵标准图。这对 few-shot 分类不一定最有用。

改成：

1. 先用 CLIP 筛掉差图；
2. 用 DINOv2/CLIP image feature 聚类；
3. 每个 cluster 里选 CLIP 分数最高的若干张；
4. 保证每个类别内部有多样性。

### 3.2 相关论文

- [Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion](https://arxiv.org/abs/2302.03298) 直接强调 diffusion synthetic dataset 中 **diversity** 对分类性能的重要性。
- [Explore the Power of Synthetic Data on Few-Shot Object Detection, CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Lin_Explore_the_Power_of_Synthetic_Data_on_Few-Shot_Object_Detection_CVPRW_2023_paper.pdf) 虽然是 few-shot detection，但也涉及 Stable Diffusion 生成、数据选择、聚类/代表样本、CLIP 相似度过滤，思路可迁移到分类。

### 3.3 实现建议

对每个类别：

```text
生成 500 张图
↓
CLIP 过滤保留 top 150
↓
DINOv2 特征聚类成 C=8/16 个 cluster
↓
每个 cluster 选 CLIP 分最高的 top-m
↓
合并得到最终 k 张
```

这个方法很适合作为第一个改进实验，因为不需要训练额外模型，代码也简单。

---

## 4. 第三优先级：目标模型感知选择，让 ResNet50 参与选图

### 4.1 核心思想

最终分类器是 ResNet50，而不是 CLIP。CLIP 分高的图未必对 ResNet50 最有用。所以可以加入 **target-model-aware selection**。

### 4.2 ResNet 置信度过滤

先用真实 few-shot 样本训练一个初始 ResNet50，然后用它预测合成图。

保留满足：

```text
CLIP 认为是该类
且 ResNet50 也认为是该类
```

的样本。

优点：降低错类合成图。
缺点：初始 ResNet few-shot 很弱，可能过度保守。

### 4.3 不确定性选择

选 ResNet50 对目标类别“有点不确定但不是明显错”的样本。

例如：

```text
0.4 < p_resnet(class | x) < 0.8
且 CLIP_score 高
```

这类样本可能更靠近决策边界，比“ResNet 已经非常确信”的简单样本更有训练价值。

### 4.4 Influence / gradient matching

更高级的做法是估计某个合成样本加入训练后，对验证集 loss 是否有正面影响。相关方向可参考：

- [Influence Selection for Active Learning, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Influence_Selection_for_Active_Learning_ICCV_2021_paper.html)

这类方法理论上很适合当前问题，因为它回答的是：

> 哪些 synthetic samples 对当前模型最有帮助？

但实现复杂度明显更高，建议作为第三阶段实验。

---

## 5. 第四优先级：synthetic-to-real curriculum，而不是一次性混合

### 5.1 核心思想

不要一开始就把所有合成图混进训练。可以按“从容易到真实”或“从 synthetic 到 real”的顺序训练。

相关论文：

- [Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html)
- 对应项目页：[DisCL project](https://joliang17.github.io/DisCL/)
- 代码仓库：[tianyi-lab/DisCL](https://github.com/tianyi-lab/DisCL)

这篇的思想是生成一系列从 synthetic 到 real 的图像，并设计 curriculum training。它也使用 CLIPScore 过滤低质量图。

### 5.2 简化训练策略

```text
Epoch 1-20: 真实图 + 高 CLIP、高 real-feature 相似的 synthetic
Epoch 21-40: 加入更多 diverse synthetic
Epoch 41-60: 降低 synthetic 权重，更多依赖真实 few-shot 图
```

或者：

```text
先训练 synthetic-heavy
再用 real-only / real-dominant 微调
```

这比直接把 synthetic 和 real 混一起训练更稳。

---

## 6. 第五优先级：改生成方式，而不是只改筛选方式

如果 CLIP top-k 已经很强，说明筛选空间可能到瓶颈了。下一步应该提高候选图质量和分布覆盖。

### 6.1 Few-shot guided generation

代表论文：

- [DataDream: Few-shot Guided Dataset Generation, ECCV 2024](https://arxiv.org/abs/2407.10910)
- 官方代码：[ExplainableML/DataDream](https://github.com/ExplainableML/DataDream)

DataDream 用 few-shot 真实样本指导 diffusion 生成分类数据集，比单纯 text prompt 更接近目标域。

### 6.2 LoRA / in-domain synthesis

代表论文：

- [DISEF: Diversified In-domain Synthesis with Efficient Fine-tuning for Few-shot Classification](https://arxiv.org/abs/2312.03046)
- 代码：[vturrisi/disef](https://github.com/vturrisi/disef)

DISEF 的重点是 **in-domain** 和 **diversified**。如果类别和 Stable Diffusion 预训练分布有差距，LoRA 类方法可能比后处理筛选更有效。

### 6.3 Caption-guided augmentation

代表论文：

- [Cap2Aug: Caption Guided Image Data Augmentation, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Roy_Cap2Aug_Caption_Guided_Image_Data_Augmentation_WACV_2025_paper.html)
- 代码：[aniket004/Cap_2_Aug](https://github.com/aniket004/Cap_2_Aug)

思路是从真实图生成 caption，再用 caption 做 image-to-image augmentation。这通常比纯 text-to-image 更稳，因为生成过程受真实图约束。

---

## 7. 不建议只换成 BLIP / SigLIP / ALIGN / DINO 排序

不建议把 DINO/DINOv2/BLIP/ALIGN/SigLIP 只是当成“CLIP 替代品”，而应该当成“互补评分器”。

原因：

- CLIP image-text 相似度天然适合做文本类别匹配；
- DINOv2 更适合做视觉相似度、聚类、多样性、real-feature filtering；
- BLIP 更适合检查 caption 是否真的描述了图像；
- SigLIP/ALIGN 可能在部分语义匹配上更强，但如果只是换一个 image-text scorer，提升未必稳定。

更推荐 ensemble：

```text
语义一致性：CLIP / SigLIP
真实分布接近：DINOv2 / ResNet50 feature
图像描述一致性：BLIP caption → text-class consistency
多样性：DINOv2 clustering / k-center
目标模型收益：ResNet uncertainty / influence
```

---

## 8. 最值得做的 5 个实验

### 实验 1：CLIP top-k baseline

这是当前方法，必须保留。

```text
生成 N 张
按 CLIP image-text score 排序
选 top-k
训练 ResNet50
```

记录：

- k = 8, 16, 32, 64, 128；
- 每类生成数量 N；
- synthetic:real 比例；
- 最终 test accuracy / macro-F1；
- 每类 accuracy。

### 实验 2：CLIP threshold + DINOv2 k-center

这是最推荐先做的。

```text
生成 N 张
CLIP 保留 top 30% 或 top 3k
用 DINOv2 提特征
用 k-center greedy 选 k 张
训练 ResNet50
```

预期：如果 CLIP top-k 太同质，这个方法很可能超过它。

### 实验 3：CLIP threshold + real-feature filtering + diversity

```text
生成 N 张
CLIP 过滤
计算候选图到真实 few-shot 图的 DINOv2/ResNet 特征相似度
去掉离真实样本太远的图
剩余图做 diversity selection
```

推荐分数：

```text
score = 0.4 * CLIP_score
      + 0.4 * similarity_to_real_class
      - 0.2 * similarity_to_selected
```

这比实验 2 更接近 RFF 思路，参考 rare plant paper 的 [Real Feature Filtering](https://openaccess.thecvf.com/content/CVPR2024W/GCV/html/Dasgupta_Can_Synthetic_Plant_Images_From_Generative_Models_Facilitate_Rare_Species_CVPRW_2024_paper.html)。

### 实验 4：ResNet-aware selection

先训练一个初始 ResNet50，然后让它参与选图。

```text
candidate 必须满足：
CLIP_score 高
ResNet 预测目标类概率不是太低
DINOv2 距离真实样本不太远
```

可以试三种版本：

```text
A: 选 ResNet 也高置信的样本
B: 选 ResNet 中等不确定的样本
C: 选加入训练后 validation loss 下降最多的样本
```

A 简单但可能保守；B 可能更能提升边界；C 最强但实现复杂。

### 实验 5：synthetic curriculum

比较两种训练策略：

```text
普通混合：
real + selected synthetic 一起训练

curriculum：
前期 synthetic-heavy
后期 real-heavy 或 real-only fine-tune
```

参考 [Diffusion Curriculum](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html)。

---

## 9. 推荐的最终 selection pipeline

如果只选一个最可能超过 CLIP top-k 的版本，建议：

```text
For each class c:

1. Stable Diffusion 生成 N=500~1000 张候选图

2. CLIP image-text score:
   保留 top 30% 或 top 3k
   只作为语义正确性过滤，不直接最终 top-k

3. DINOv2 / ResNet50 feature extraction:
   提取候选图特征
   提取该类 few-shot 真实图特征

4. Real feature filtering:
   去掉距离真实样本中心太远的 synthetic images

5. Diversity selection:
   在剩余图中做 k-center greedy / k-medoids / MMR

6. 可选 ResNet-aware rerank:
   去掉 ResNet 明显判成其他类的图
   或加入中等不确定样本

7. Training:
   synthetic + real 训练
   最后 real-only fine-tune 5~10 epochs
```

简化公式：

```text
score(x) =
    0.35 * normalized(CLIP_text_score)
  + 0.35 * normalized(max DINO_sim_to_real_class)
  + 0.15 * normalized(ResNet_target_prob)
  - 0.15 * normalized(max DINO_sim_to_selected)
```

如果 few-shot 真实图很少，比如 1-shot 或 2-shot，`similarity_to_real_class` 不要权重太高，否则会过拟合那几张真实图。可以改成：

```text
1-shot/2-shot:
CLIP 0.5, real feature 0.25, diversity 0.25

5-shot/10-shot:
CLIP 0.35, real feature 0.4, diversity 0.25
```

---

## 10. 需要注意的反例和风险

### 10.1 synthetic data 不一定真的比检索真实图强

NeurIPS 2024 的论文 [The Unmet Promise of Synthetic Training Images: Using Retrieved Real Images Performs Better](https://papers.nips.cc/paper_files/paper/2024/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html) 提醒了一个重要问题：对于分类任务，**从大规模数据源检索相关真实图，有时会匹配或超过 Stable Diffusion 生成图**。

如果任务允许外部数据，建议加一个 baseline：

```text
CLIP retrieve real images from web/LAION/local dataset
vs
Stable Diffusion synthetic images
```

这个 baseline 很重要。

### 10.2 real-feature filtering 可能过拟合 few-shot 真实图

如果每类只有 1 张真实图，按照真实图特征筛选可能导致 synthetic 图太像那一张，损失多样性。

解决：

```text
1-shot: real-feature filtering 只做弱约束
5-shot+: 可以加强 real-feature filtering
```

### 10.3 CLIP text prompt 会影响筛选结果

如果 prompt 太简单，例如只有 `"a photo of a {class}"`，CLIP top-k 会偏向最典型样本。可以试：

```text
a photo of a {class}
a close-up photo of a {class}
a {class} in natural environment
a cropped image of a {class}
a low-resolution photo of a {class}
a side view of a {class}
```

然后做 prompt ensemble。

### 10.4 synthetic:real 比例不能无限加

很多论文都观察到合成数据有边际收益递减。建议扫：

```text
每类 synthetic k = 4, 8, 16, 32, 64, 128
```

不要默认越多越好。

---

## 11. 论文清单按用途整理

### 11.1 CLIP top-k / few-shot synthetic baseline

- [Prompt, Generate, Then Cache: Cascade of Foundation Models Makes Strong Few-Shot Learners, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.html)
  和当前 pipeline 最接近：生成图片，用 CLIP 过滤 top-K′，再用于 few-shot。

- [Is synthetic data from generative models ready for image recognition?](https://arxiv.org/abs/2210.07574)
  早期系统性研究生成图像用于 image recognition 的效果。

- [Synthetic Data from Diffusion Models Improves ImageNet Classification](https://arxiv.org/abs/2304.08466)
  说明 diffusion synthetic data 在大规模分类上可以有效提升性能。

### 11.2 Few-shot guided generation / in-domain generation

- [DataDream: Few-shot Guided Dataset Generation, ECCV 2024](https://arxiv.org/abs/2407.10910)
  用 few-shot examples 指导 diffusion 生成分类数据集。

- [DISEF: Diversified In-domain Synthesis with Efficient Fine-tuning for Few-shot Classification](https://arxiv.org/abs/2312.03046)
  强调 diversified in-domain synthesis 和 LoRA efficient fine-tuning。

- [Cap2Aug: Caption Guided Image Data Augmentation, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Roy_Cap2Aug_Caption_Guided_Image_Data_Augmentation_WACV_2025_paper.html)
  caption-guided image-to-image augmentation，更贴近真实图分布。

### 11.3 多样性 / 分布覆盖

- [Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion](https://arxiv.org/abs/2302.03298)
  重点说明 synthetic data diversity 对分类性能的重要性。

- [Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html)
  用 synthetic-to-real curriculum 提升训练效果，也包含 CLIPScore filtering。

- [Enhance Image Classification via Inter-Class Image Mixup with Diffusion Model, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Enhance_Image_Classification_via_Inter-Class_Image_Mixup_with_Diffusion_Model_CVPR_2024_paper.html)
  关注 diffusion augmentation 中 faithfulness 和 diversity 的平衡。

### 11.4 真实特征过滤 / DINOv2 特征空间

- [Can Synthetic Plant Images From Generative Models Facilitate Rare Species Identification and Classification?, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/GCV/html/Dasgupta_Can_Synthetic_Plant_Images_From_Generative_Models_Facilitate_Rare_Species_CVPRW_2024_paper.html)
  提出 Real Feature Filtering，和 few-shot 分类场景非常接近。

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
  可作为 synthetic image selection 的视觉特征 backbone。

- [Representation-Conditioned Diffusion Models for Guided Training Data Generation](https://arxiv.org/abs/2605.27495)
  2026 预印本，涉及 DINOv2/DINOv3/CLIP 表征条件扩散和样本过滤，方向高度相关。

- [Increasing the Utility of Synthetic Images through Chamfer Guidance](https://openreview.net/forum?id=X3RVQNOIXZ)
  用 DINOv2 特征空间中的 Chamfer 距离增强 synthetic image utility。

### 11.5 目标模型感知 / active learning / influence

- [Influence Selection for Active Learning, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Influence_Selection_for_Active_Learning_ICCV_2021_paper.html)
  可迁移到 synthetic image selection：估计样本对模型性能的正向影响。

### 11.6 重要反例 baseline

- [The Unmet Promise of Synthetic Training Images: Using Retrieved Real Images Performs Better, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html)
  提醒：检索真实图是必须比较的强 baseline。

---

## 12. 最终建议

如果目标是发论文或做出明显提升，建议优先尝试这个题目：

> **CLIP-Filtered Real-Feature Diverse Selection for Diffusion-Augmented Few-Shot Classification**

核心卖点：

1. 保留 CLIP top-k 的语义过滤优势；
2. 引入 DINOv2/ResNet real-feature filtering，减少 synthetic-real domain gap；
3. 引入 diversity/coreset selection，避免 CLIP top-k 同质化；
4. 引入 ResNet-aware reranking，让选择目标和最终分类器一致；
5. 在多个 few-shot 设置下比较：1-shot、2-shot、5-shot、10-shot。

最小实验矩阵：

| 方法 | 说明 |
|---|---|
| Real only | 不用合成图 |
| Random synthetic | 随机选合成图 |
| CLIP top-k | 当前强 baseline |
| CLIP threshold + DINO k-center | 第一推荐改进 |
| CLIP + real-feature filtering | 第二推荐改进 |
| CLIP + real-feature + diversity | 最可能超过 top-k |
| CLIP + real-feature + diversity + ResNet-aware | 高级版本 |

如果只做一个改进，建议做：

```text
CLIP top-30% filtering
+
DINOv2 feature k-center / MMR selection
+
最后 real-only fine-tune
```

这条路线实现成本低，而且最有希望稳定超过单纯 CLIP top-k。
