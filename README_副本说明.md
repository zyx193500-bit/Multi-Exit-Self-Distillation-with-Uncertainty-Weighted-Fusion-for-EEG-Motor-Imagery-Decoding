# TCFormer 精简副本

生成时间：2026-09-07（Asia/Shanghai）

这是从原仓库提取出的只读式工作副本。生成过程中没有删除、移动或修改原仓库内容。

## 目录

- `code/`：训练基础代码、模型与数据模块、TCFormer 多出口训练配置、消融配置。
- `reports/single_seed_multi_exit/`：seed 0 多出口训练结果、后续推理审计、可视化报告及其资产。
- `reports/tcformer_multi_exit_ablation/`：TCFormer 多出口消融运行报告；包含完整运行，也保留少量中断运行的配置和进度记录。
- `visualization/`：训练曲线、混淆矩阵、特征阶段、总体面板、方法图和 seed 0 审计图的生成代码。
- `ABLATION_STATUS.csv`：各消融方法完整/中断运行的状态索引。
- `MANIFEST_SHA256.csv`：副本文件的相对路径、大小和 SHA-256；清单不包含清单文件自身。

## 核心代码范围

`code/` 保留当前仓库中的通用训练骨架：

- `configs/`
- `datamodules/`
- `models/`
- `utils/`
- `train_pipeline.py`
- `train_pipeline_original_protocol.py`
- 结果汇总、基础测试、依赖说明和许可证
- `tcformer_author_protocol/`
- `tcformer_ablation_methods/`

源 Git 基线提交：`04ef5751db9d3cca2ece7d10e438244938ce44bb`。提取时原仓库存在未提交/已暂存改动，因此本副本代表 2026-09-07 当时工作区的实际文件快照，而不是该提交的纯净导出。

## 报告范围

单种子多出口部分保留：

- BCIC-IV-2a seed 0 原始训练报告、曲线和混淆矩阵。
- BCIC-IV-2b seed 0 原始训练报告、曲线和混淆矩阵。
- 2026-09-06 基于归档 checkpoint 完成的 seed 0 推理审计，包括 CSV/JSON、图、HTML/PDF 预览和验证记录。
- 2026-05-03 生成的 TCFormer 多出口可视化 DOCX、Markdown 和配套资产。

消融部分保留所有非权重报告。共有 9 个方法存在完整 `results.txt`；`shallow_mid_deep_final` 只有中断运行记录。其它方法中也有少量早期中断尝试，详情见 `ABLATION_STATUS.csv`。

## 明确排除

- 原始 EEG 数据与下载文件
- 数据预处理缓存
- checkpoint 和其它模型权重（`.ckpt`、`.pt`、`.pth`）
- 大体积逐试次预测数组 `predictions.npz`
- `.git`、IDE 工程配置、`__pycache__` 和 `.pyc`
- ATCNet、CTNet、EEGNet、MSCFormer 等模型的实验结果目录
- 论文修订临时目录和无关输出

注意：需要重新执行 checkpoint 推理或样本级特征可视化时，仍需自行提供原始数据和 checkpoint；本副本保存的是基础代码与已有报告，不是完整训练环境镜像。
