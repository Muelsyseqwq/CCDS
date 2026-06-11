# Pets20 160/sp80 margin_topk + RealFT tables

## Best summary

- Accuracy mean: 0.9093050648
- Macro F1 mean: 0.9086824557
- Seeds: 3

## Per-seed results

| project_name | method | seed | accuracy | macro_f1 | best_val_accuracy | epochs | real_finetune_epochs | real_finetune_lr | train_size | real_finetune_size | test_size | selected_per_class | config_path | summary_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ccds_pets20_160_sp80_core_realft | margin_topk | 0 | 0.9106511863 | 0.9097863998 | 0.9150000000 | 30 | 5.0000000000 | 0.0001000000 | 1700 | 100.0000000000 | 1981 | 80 | configs/sweeps/pets20_160_sp80_core_realft.yaml | /root/gpufree-data/clip_diffusion_fewshot_ccds_results/classifier/ccds_pets20_160_sp80_core_realft/margin_topk/seed0/summary.csv |
| ccds_pets20_160_sp80_core_realft | margin_topk | 1 | 0.9066128218 | 0.9064586355 | 0.9250000000 | 30 | 5.0000000000 | 0.0001000000 | 1700 | 100.0000000000 | 1981 | 80 | configs/sweeps/pets20_160_sp80_core_realft.yaml | /root/gpufree-data/clip_diffusion_fewshot_ccds_results/classifier/ccds_pets20_160_sp80_core_realft/margin_topk/seed1/summary.csv |
| ccds_pets20_160_sp80_core_realft | margin_topk | 2 | 0.9106511863 | 0.9098023320 | 0.9000000000 | 30 | 5.0000000000 | 0.0001000000 | 1700 | 100.0000000000 | 1981 | 80 | configs/sweeps/pets20_160_sp80_core_realft.yaml | /root/gpufree-data/clip_diffusion_fewshot_ccds_results/classifier/ccds_pets20_160_sp80_core_realft/margin_topk/seed2/summary.csv |

## Baseline comparison

| setting | project_name | method | num_seeds | candidates_per_class | selected_per_class | epochs | real_finetune_epochs | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | best_val_accuracy_mean | delta_accuracy_vs_best | delta_macro_f1_vs_best | train_size | test_size | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pets20 fair RealFT real_only | ccds_pets20_5shot_30ep_realft | real_only | 3 | 80 | 10 | 30 | 5 | 0.8127208481 | 0.0411214777 | 0.8064039896 | 0.0505746612 | 0.8550000000 | -0.0965842167 | -0.1022784662 | 100 | 1981 | 30+5 RealFT fair baseline; real images only. |
| Pets20 fair RealFT traditional_aug | ccds_pets20_5shot_30ep_realft | traditional_aug | 3 | 80 | 10 | 30 | 5 | 0.8048123843 | 0.0412153543 | 0.7974133882 | 0.0508858132 | 0.8516666667 | -0.1044926805 | -0.1112690675 | 100 | 1981 | 30+5 RealFT fair baseline with traditional augmentation. |
| Pets20 fair RealFT diffusion_random | ccds_pets20_5shot_30ep_realft | diffusion_random | 3 | 80 | 10 | 30 | 5 | 0.8712771328 | 0.0102088584 | 0.8693697110 | 0.0114000503 | 0.9033333333 | -0.0380279320 | -0.0393127447 | 300 | 1981 | 30+5 RealFT, 10 selected/class random synthetic baseline. |
| Pets20 fair RealFT clip_topk | ccds_pets20_5shot_30ep_realft | clip_topk | 3 | 80 | 10 | 30 | 5 | 0.8992091536 | 0.0101629944 | 0.8982194315 | 0.0110704765 | 0.9166666667 | -0.0100959112 | -0.0104630243 | 300 | 1981 | Strong 30+5 RealFT baseline, 10 selected/class. |
| Pets20 fair RealFT margin_topk | ccds_pets20_5shot_30ep_realft | margin_topk | 3 | 80 | 10 | 30 | 5 | 0.8872623254 | 0.0065103625 | 0.8855060045 | 0.0069581490 | 0.9000000000 | -0.0220427394 | -0.0231764513 | 300 | 1981 | 30+5 RealFT baseline, 10 selected/class. |
| Pets20 fair RealFT ccds | ccds_pets20_5shot_30ep_realft | ccds | 3 | 80 | 10 | 30 | 5 | 0.8892815077 | 0.0084669329 | 0.8882336383 | 0.0084304289 | 0.9100000000 | -0.0200235571 | -0.0204488175 | 300 | 1981 | 30+5 RealFT baseline, 10 selected/class. |
| Pets20 fair RealFT anchored_ccds | ccds_pets20_5shot_30ep_realft | anchored_ccds | 3 | 80 | 10 | 30 | 5 | 0.8956755847 | 0.0075775401 | 0.8945891966 | 0.0080373672 | 0.9183333333 | -0.0136294801 | -0.0140932591 | 300 | 1981 | 30+5 RealFT baseline, 10 selected/class. |
| Pets20 D7 anchored + RealFT | ccds_pets20_d7_anchor6_top60_w701515_realft | anchored_ccds | 3 | 80 | 10 | 30 | 5 | 0.8998822144 | 0.0088639187 | 0.8992820277 | 0.0090266802 | 0.9133333333 | -0.0094228504 | -0.0094004280 | 300 | 1981 | Tuned anchored CCDS result before large-synthetic sweep. |
| Pets20 160/sp60 CCDS + RealFT | ccds_pets20_160_sp60_core_realft | ccds | 3 | 160 | 60 | 30 | 5 | 0.9072858826 | 0.0048505924 | 0.9066117036 | 0.0047011024 | 0.9050000000 | -0.0020191822 | -0.0020707521 | 1300 | 1981 | Large synthetic ablation; 60 selected/class. |
| Pets20 160/sp80 margin_topk + RealFT (best) | ccds_pets20_160_sp80_core_realft | margin_topk | 3 | 160 | 80 | 30 | 5 | 0.9093050648 | 0.0023315508 | 0.9086824557 | 0.0019259013 | 0.9133333333 | 0.0000000000 | 0.0000000000 | 1700 | 1981 | Current best 3-seed mean; 80 selected/class. |
| Pets20 160/sp100 margin_topk + RealFT | ccds_pets20_160_sp100_core_realft | margin_topk | 3 | 160 | 100 | 30 | 5 | 0.8983678277 | 0.0134159174 | 0.8977857397 | 0.0128946858 | 0.8983333333 | -0.0109372371 | -0.0108967161 | 2100 | 1981 | Large synthetic ablation; 100 selected/class. |
| Pets20 160/sp100 anchored_ccds + RealFT | ccds_pets20_160_sp100_core_realft | anchored_ccds | 3 | 160 | 100 | 30 | 5 | 0.9039205788 | 0.0027801971 | 0.9031948655 | 0.0030330060 | 0.8900000000 | -0.0053844859 | -0.0054875902 | 2100 | 1981 | Large synthetic ablation; 100 selected/class. |
| Pets20 160/sp80 DINOv2 CFRD-MMR + RealFT | ccds_pets20_160_sp80_cfrd_mmr_dinov2_realft | cfrd_mmr | 1 | 160 | 80 | 30 | 5 | 0.9050984351 |  | 0.9045135395 |  | 0.9100000000 | -0.0042066296 | -0.0041689163 | 1700 | 1981 | Single-seed diagnostic/ablation currently available. |
