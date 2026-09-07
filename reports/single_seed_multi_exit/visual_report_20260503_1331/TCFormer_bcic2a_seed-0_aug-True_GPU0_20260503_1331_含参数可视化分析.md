# Transformer 多出口可视化与分析

## 放置建议

- 正文插入位置：放在中文结果讨论结束、英文部分开始之前，作为“单样本可解释性结果与可视化分析”小节。
- 图注放置位置：放在原文“图注模板”部分后，补充多出口对应的 Figure 5 与 Figure 6 图注。
- 附录放置位置：放在文末，作为“9个subject单样本可解释性案例总览”。

## 正文可直接插入内容

### 本次运行参数说明

为保证本次可视化与结果分析可复现，下面补充本轮正式训练所使用的关键参数。参数按训练设置、优化策略、预处理配置、骨干网络设置以及蒸馏与多出口机制分组展示，可直接附在方法或实验设置部分之后。

#### 训练设置

- model=TCFormer
- dataset_name=bcic2a
- subject_ids=all
- seed=0
- gpu_id=0
- max_epochs=1000
- precision=16-mixed
- accumulate_grad_batches=2
- selection_metric=val_acc
- selection_mode=max
- save_best_checkpoint=True

#### 早停与监控

- early_stopping.enabled=False
- early_stopping.monitor=val_acc
- early_stopping.mode=max
- early_stopping.patience=40
- early_stopping.min_delta=0.0

#### 优化与调度

- optimizer=adam
- lr=0.0009
- weight_decay=0.001
- scheduler=True
- warmup_epochs=20
- warmup_epochs_loso=3
- beta_1=0.5

#### 预处理与数据增强

- batch_size=16
- sfreq=250
- start=0.0
- stop=0.0
- z_scale=True
- interaug=True
- low_cut=None
- high_cut=None

#### 骨干网络参数

- F1=32
- D=2
- d_group=16
- q_heads=4
- kv_heads=2
- trans_depth=5
- tcn_depth=2
- kernel_length_tcn=4
- pool_length_1=8
- pool_length_2=7
- temp_kernel_lengths=[20, 32, 64]
- use_group_attn=True
- dropout_conv=0.4
- trans_dropout=0.4
- dropout_tcn=0.3

#### 蒸馏与多出口参数

- primary_output=ensemble
- distill_temperature=3.0
- byot_proj_dim=128
- byot_normalize_hints=True
- byot_distill_start_epoch=30
- byot_distill_ramp_epochs=20
- byot_hint_start_epoch=60
- byot_hint_ramp_epochs=20
- byot_ce_weight_shallow=0.8
- byot_ce_weight_mid=0.9
- byot_ce_weight_deep=1.0
- byot_kd_weight_shallow=0.5
- byot_kd_weight_mid=0.6
- byot_kd_weight_deep=0.7
- byot_hint_weight_shallow=0.15
- byot_hint_weight_mid=0.15
- byot_hint_weight_deep=0.1
- teacher_loss_weight=0.8
- teacher_ensemble_weight=3.0
- ensemble_loss_weight=1.2
- ensemble_prior_shallow=0.45
- ensemble_prior_mid=0.9
- ensemble_prior_deep=1.15

### 单样本可解释性结果与可视化分析

为进一步支撑本文关于“EEG 浅层、中层与深层特征均包含有效判别信息，而多出口动态集成能够提升整体稳定性”的核心观点，这里补充 Transformer 多出口模型的单样本可解释性分析结果。

本次可解释性分析基于完整训练结果目录 `TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331` 进行，代表性样本选自 subject 3 的测试集第 0 个 trial。该样本真实标签为 feet，模型最终预测也为 feet，集成输出概率为 0.9675。出口权重分别为 shallow=0.0931, mid=0.3152, deep=0.3634, final=0.2283，说明当前判断主要依赖 deep 和 mid 两个较深层出口。

从阶段特征图可以观察到，卷积前端主要提取局部节律和短时波形模式；shallow 分支保留较早期的局部判别线索；mid 与 deep 分支逐步强化跨时间上下文关系；TCN 模块进一步整合时序演化；final 分支则保留最终分类前最紧凑、最具类别相关性的高层表示。

在输入层面，电极贡献分析显示，代表性样本的关键支持证据主要集中在 CPz、C4、CP3、Cz、FCz 等通道，Gradient x Input 给出的高响应电极主要包括 CPz、Cz、C4、CP3、FCz，显著时间窗口主要位于 335，345，344，326，327，325，297，336，346，288，334，354，287，343，355 附近。

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject3_sample0/paper_overview_figure.png)

图5. 代表性单样本（subject 3, sample 0）的可解释性总览图。该图同时展示了原始 EEG、各阶段特征演化过程、电极贡献排序以及近似头皮重要性分布，可用于支撑多出口分层特征与动态集成行为的结果讨论。

此外，为避免代表性样本分析的偶然性，这里进一步对 9 个 subject 的最佳 checkpoint 分别自动选择一个预测正确的测试样本并进行同样的可解释性分析。结果表明，9 个被试均成功找到可解释案例，集成置信度大致分布在 0.6465 到 0.9779 之间，Final 出口权重大致分布在 0.2054 到 0.2681 之间。

表A. 9个被试单样本可解释案例摘要

## 图注模板补充

- Figure 5. Representative single-trial interpretability overview including raw EEG, stage-wise feature maps, channel-importance bars, and scalp saliency map.
- Figure 6. Subject-wise representative single-trial interpretability summaries across all nine subjects.

## 代表性样本中文解释

### 样本结论

- 预测类别：feet
- 真实类别：feet
- 集成预测置信度：0.9675
- 出口权重：shallow=0.0931, mid=0.3152, deep=0.3634, final=0.2283

### 论文式图注

图X展示了一个被正确分类样本从原始信号到多出口分层特征表示的逐层演化过程。Conv Features 展示了卷积前端提取的局部多尺度节律模式。Shallow Routed Features 反映了浅层出口所利用的早期判别线索。Mid Routed Features 和 Deep Routed Features 分别对应中层与深层语义强化结果。TCN Features 展示了时序卷积网络对多阶段信息进行动态整合后的高强度判别特征。Final Routed Features 则保留了最终分类前最紧凑、最具类别相关性的证据。

### 中文论文段落

从出口权重可以看出，当前样本的分类决策主要依赖 deep 和 mid 两个较深层出口，而 shallow 出口贡献相对较小。这说明模型更多依赖中深层时序语义与最终高层表示完成判断，多出口结构不仅提供了层次化监督，也能够反映浅层、中层与深层特征在判别任务中的不同作用。

输入梯度分析进一步表明，模型关注的关键证据主要集中在 CPz、C4、CP3、Cz、FCz 等少数中央区及邻近电极，以及一段相对集中的时间窗口内。这说明模型执行的是任务相关的稀疏证据选择，而不是简单依赖全通道整体振幅变化进行分类。脚部运动意象通常更关注中央区和顶中央区附近共同节律变化。

### 定量摘要

- raw_signal: shape=[22, 1000]，abs_mean=0.7093，abs_max=3.7410
- conv_features: shape=[48, 17]，abs_mean=0.3836，abs_max=1.9394
- shallow_routed: shape=[48, 17]，abs_mean=0.2736，abs_max=2.0973
- mid_routed: shape=[48, 17]，abs_mean=0.2360，abs_max=1.2408
- deep_routed: shape=[48, 17]，abs_mean=0.2551，abs_max=1.2729
- tcn_features: shape=[64, 17]，abs_mean=0.4383，abs_max=2.8690
- final_routed: shape=[64, 17]，abs_mean=0.1376，abs_max=0.8830

### 电极与时间证据

- Saliency 前10电极：CPz、C4、CP3、Cz、FCz
- Gradient x Input 前10电极：CPz、Cz、C4、CP3、FCz
- 最显著时间点：335，345，344，326，327，325，297，336，346，288，334，354，287，343，355

## 总的可视化

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/overall_dashboard.png)

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/overall_subject_gallery.png)

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/overall_metrics.png)

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/overall_electrodes.png)

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/results/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/confmats/avg_confusion_matrix.png)

## 附录：9个subject单样本可解释性案例总览

### Subject 1

- 样本索引：8
- 真实类别：hand(L)
- 预测类别：hand(L)
- 集成置信度：0.6958
- 代表样本出口权重：shallow=0.0904, mid=0.3146, deep=0.3597, final=0.2353

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject1_sample8/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 hand(L)，测试准确率为 0.9306，Kappa 为 0.9074。该被试表现非常稳定，说明多出口结构在该个体上形成了清晰而稳定的分层判别表征。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 C2、C4、CP4、CPz、FC4，Grad x Input 主要集中在 C2、C4、CP4、CPz、FC4，说明模型更依赖少数关键通道形成最终判别。

### Subject 2

- 样本索引：1
- 真实类别：hand(L)
- 预测类别：hand(L)
- 集成置信度：0.6465
- 代表样本出口权重：shallow=0.0871, mid=0.2981, deep=0.3467, final=0.2681

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject2_sample1/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 hand(L)，测试准确率为 0.7014，Kappa 为 0.6019。该被试达到可用水平，说明模型能够抓住主要判别证据，但个体差异仍带来一定混淆。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 C3、CPz、C4、Cz、FC3，Grad x Input 主要集中在 C3、C4、CPz、Cz、FC3，说明模型更依赖少数关键通道形成最终判别。

### Subject 3

- 样本索引：0
- 真实类别：feet
- 预测类别：feet
- 集成置信度：0.9675
- 代表样本出口权重：shallow=0.0931, mid=0.3152, deep=0.3634, final=0.2283

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject3_sample0/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 feet，测试准确率为 0.9757，Kappa 为 0.9676。该被试表现非常稳定，说明多出口结构在该个体上形成了清晰而稳定的分层判别表征。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 CPz、C4、CP3、Cz、FCz，Grad x Input 主要集中在 CPz、Cz、C4、CP3、FCz，说明模型更依赖少数关键通道形成最终判别。

### Subject 4

- 样本索引：12
- 真实类别：feet
- 预测类别：feet
- 集成置信度：0.9687
- 代表样本出口权重：shallow=0.0855, mid=0.3114, deep=0.3603, final=0.2428

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject4_sample12/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 feet，测试准确率为 0.8924，Kappa 为 0.8565。该被试整体表现较稳定，说明模型已经学到可靠的类别相关模式，但仍存在少量边界样本。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 POz、CP3、C6、Cz、CP2，Grad x Input 主要集中在 POz、CP3、C6、CP2、Cz，说明模型更依赖少数关键通道形成最终判别。

### Subject 5

- 样本索引：2
- 真实类别：hand(L)
- 预测类别：hand(L)
- 集成置信度：0.9565
- 代表样本出口权重：shallow=0.0921, mid=0.3101, deep=0.3590, final=0.2388

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject5_sample2/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 hand(L)，测试准确率为 0.8125，Kappa 为 0.7500。该被试整体表现较稳定，说明模型已经学到可靠的类别相关模式，但仍存在少量边界样本。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 C3、C5、CPz、CP3、C1，Grad x Input 主要集中在 C3、C5、CPz、CP3、C1，说明模型更依赖少数关键通道形成最终判别。

### Subject 6

- 样本索引：1
- 真实类别：hand(L)
- 预测类别：hand(L)
- 集成置信度：0.8389
- 代表样本出口权重：shallow=0.0911, mid=0.3170, deep=0.3638, final=0.2281

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject6_sample1/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 hand(L)，测试准确率为 0.7361，Kappa 为 0.6481。该被试达到可用水平，说明模型能够抓住主要判别证据，但个体差异仍带来一定混淆。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 Cz、CP2、CP4、P2、P1，Grad x Input 主要集中在 Cz、CP2、CP4、FCz、FC3，说明模型更依赖少数关键通道形成最终判别。

### Subject 7

- 样本索引：9
- 真实类别：tongue
- 预测类别：tongue
- 集成置信度：0.9678
- 代表样本出口权重：shallow=0.1070, mid=0.3202, deep=0.3674, final=0.2054

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject7_sample9/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 tongue，测试准确率为 0.9514，Kappa 为 0.9352。该被试表现非常稳定，说明多出口结构在该个体上形成了清晰而稳定的分层判别表征。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 C3、Cz、CPz、CP3、C1，Grad x Input 主要集中在 Cz、C3、CPz、CP3、C1，说明模型更依赖少数关键通道形成最终判别。

### Subject 8

- 样本索引：0
- 真实类别：feet
- 预测类别：feet
- 集成置信度：0.9779
- 代表样本出口权重：shallow=0.0961, mid=0.3009, deep=0.3483, final=0.2547

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject8_sample0/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 feet，测试准确率为 0.9201，Kappa 为 0.8935。该被试表现非常稳定，说明多出口结构在该个体上形成了清晰而稳定的分层判别表征。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 POz、CPz、CP3、P2、CP2，Grad x Input 主要集中在 POz、CPz、CP3、P2、CP2，说明模型更依赖少数关键通道形成最终判别。

### Subject 9

- 样本索引：0
- 真实类别：feet
- 预测类别：feet
- 集成置信度：0.9629
- 代表样本出口权重：shallow=0.0926, mid=0.3013, deep=0.3424, final=0.2637

![](C:/Users/zyx19/Desktop/zhengliu/蒸馏/TCFormer/analysis/single_eeg_feature_viz_collected/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331_subject9_sample0/paper_overview_figure.png)

简要分析：该被试代表性样本被正确识别为 feet，测试准确率为 0.9132，Kappa 为 0.8843。该被试表现非常稳定，说明多出口结构在该个体上形成了清晰而稳定的分层判别表征。当前样本主要依赖 deep 和 mid 出口完成判断。Saliency 主要集中在 CPz、Pz、FC3、CP4、C5，Grad x Input 主要集中在 CPz、FC3、Pz、C5、C1，说明模型更依赖少数关键通道形成最终判别。

