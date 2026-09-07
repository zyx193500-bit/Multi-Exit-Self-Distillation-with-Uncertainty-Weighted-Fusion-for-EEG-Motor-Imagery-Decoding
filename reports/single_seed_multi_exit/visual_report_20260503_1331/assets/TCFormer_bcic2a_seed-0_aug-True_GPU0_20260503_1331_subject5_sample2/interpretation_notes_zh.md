# 单样本可解释性说明

## 样本结论

- 预测类别：hand(L)
- 真实类别：hand(L)
- 集成预测置信度：0.9565
- 出口权重：shallow=0.0921, mid=0.3101, deep=0.3590, final=0.2388

## 论文式图注

图X展示了一个被正确分类为 feet 的单试次 EEG 样本从原始信号到各阶段特征表示的逐层演化过程。原始 EEG 图反映了 22 通道输入脑电的时域波动。Conv Features 展示了卷积前端提取的局部多尺度节律模式。Shallow Routed Features 反映了浅层出口所利用的早期判别线索。Mid Routed Features 和 Deep Routed Features 分别对应 Transformer 中层与深层对跨时间上下文关系的建模结果。TCN Features 展示了时序卷积网络对多阶段信息进行动态整合后的高强度判别特征。Final Routed Features 则保留了最终分类前最紧凑、最具类别相关性的证据。

## 中文论文段落

针对该单样本的可视化结果可以观察到，模型首先在卷积前端阶段提取局部时间窗内的节律模式和基础跨通道响应特征，随后在 Transformer 中层与深层逐步建模不同时间片之间的上下文依赖关系，使得表征从局部波形特征逐渐过渡到类别相关语义特征。进一步地，TCN 模块对卷积特征和 Transformer 特征进行联合时序整合，使模型能够捕捉动作意象相关模式在时间上的持续性与演化顺序，最终形成更稳定的判别表示。

从出口权重可以看出，当前样本的分类决策主要依赖 deep 和 final 两个较深层出口，而 shallow 出口贡献相对较小。这说明对于该 feet 样本，模型并不是仅依靠浅层局部节律直接完成分类，而是更多依赖深层时序语义信息完成最终判断。该现象表明，多出口结构不仅提供了层次化监督，还能够反映浅层特征与深层特征在判别任务中的不同作用。

输入梯度分析进一步表明，模型关注的关键证据主要集中在少数中央区和顶中央区附近电极，以及一段相对集中的时间窗口内。这说明模型执行的是任务相关的稀疏证据选择，而不是简单依赖全通道整体振幅变化进行分类。对于该样本，CPz、C4、CP1、POz、Pz 和 Cz 等位置提供了更强的判别支持，提示模型更重视这些区域所承载的动作意象相关动态模式。

## 定量摘要

- raw_signal：shape=[22, 1000]，abs_mean=0.8013，abs_max=3.5016
- conv_features：shape=[48, 17]，abs_mean=0.5242，abs_max=3.3253
- shallow_routed：shape=[48, 17]，abs_mean=0.2563，abs_max=2.1234
- mid_routed：shape=[48, 17]，abs_mean=0.1695，abs_max=1.1473
- deep_routed：shape=[48, 17]，abs_mean=0.1704，abs_max=1.1131
- tcn_features：shape=[64, 17]，abs_mean=0.4945，abs_max=3.1066
- final_routed：shape=[64, 17]，abs_mean=0.1954，abs_max=1.3528

## 电极与时间证据

- Saliency 前10电极：C3，C5，CPz，CP3，C1，FC4，C4，P1，Fz，FC1
- Gradient x Input 前10电极：C3，C5，CPz，CP3，C1，FC4，P1，C4，FC1，Fz
- 最显著时间点：80，81，99，107，82，79，98，106，93，113，100，114，108，105，115