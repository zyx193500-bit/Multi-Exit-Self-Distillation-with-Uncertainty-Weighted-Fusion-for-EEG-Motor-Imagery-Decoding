# TCFormer 多出口可视化总汇总

- 汇总目录：`C:\Users\zyx19\Desktop\zhengliu\蒸馏\TCFormer\analysis\single_eeg_feature_viz_collected\TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331`
- 原始结果目录：`C:\Users\zyx19\Desktop\zhengliu\蒸馏\TCFormer\results\TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331`
- 平均主输出准确率：87.04 +/- 9.19
- Shallow 出口准确率：83.02 +/- 8.34
- Mid 出口准确率：86.38 +/- 9.06
- Deep 出口准确率：86.57 +/- 8.88
- Final 出口准确率：86.00 +/- 9.64
- 平均 Kappa：0.827 +/- 0.123
- 平均 Loss：0.609 +/- 0.281
- 总训练时间：330.89 min
- 平均响应时间：16.34 ms

## Subject 汇总表

| Subject | Sample | True | Pred | Conf | Test Acc | Kappa | Top-5 Saliency | Top-5 Grad x Input |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | hand(L) | hand(L) | 0.6958 | 0.9306 | 0.9074 | C2, C4, CP4, CPz, FC4 | C2, C4, CP4, CPz, FC4 |
| 2 | 1 | hand(L) | hand(L) | 0.6465 | 0.7014 | 0.6019 | C3, CPz, C4, Cz, FC3 | C3, C4, CPz, Cz, FC3 |
| 3 | 0 | feet | feet | 0.9675 | 0.9757 | 0.9676 | CPz, C4, CP3, Cz, FCz | CPz, Cz, C4, CP3, FCz |
| 4 | 12 | feet | feet | 0.9687 | 0.8924 | 0.8565 | POz, CP3, C6, Cz, CP2 | POz, CP3, C6, CP2, Cz |
| 5 | 2 | hand(L) | hand(L) | 0.9565 | 0.8125 | 0.7500 | C3, C5, CPz, CP3, C1 | C3, C5, CPz, CP3, C1 |
| 6 | 1 | hand(L) | hand(L) | 0.8389 | 0.7361 | 0.6481 | Cz, CP2, CP4, P2, P1 | Cz, CP2, CP4, FCz, FC3 |
| 7 | 9 | tongue | tongue | 0.9678 | 0.9514 | 0.9352 | C3, Cz, CPz, CP3, C1 | Cz, C3, CPz, CP3, C1 |
| 8 | 0 | feet | feet | 0.9779 | 0.9201 | 0.8935 | POz, CPz, CP3, P2, CP2 | POz, CPz, CP3, P2, CP2 |
| 9 | 0 | feet | feet | 0.9629 | 0.9132 | 0.8843 | CPz, Pz, FC3, CP4, C5 | CPz, FC3, Pz, C5, C1 |
