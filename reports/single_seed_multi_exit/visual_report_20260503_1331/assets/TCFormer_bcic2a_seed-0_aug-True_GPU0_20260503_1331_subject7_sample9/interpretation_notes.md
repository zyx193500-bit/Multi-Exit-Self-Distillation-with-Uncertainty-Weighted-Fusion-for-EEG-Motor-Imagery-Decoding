# Single-Sample Interpretation Notes

- Predicted class: tongue
- True class: tongue
- Ensemble confidence: 0.9678
- Exit weights: shallow=0.1070, mid=0.3202, deep=0.3674, final=0.2054

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.7310, abs_max=4.0838
- conv_features: shape=[48, 17], abs_mean=0.5614, abs_max=2.9813
- shallow_routed: shape=[48, 17], abs_mean=0.2174, abs_max=1.5651
- mid_routed: shape=[48, 17], abs_mean=0.1527, abs_max=0.7878
- deep_routed: shape=[48, 17], abs_mean=0.1498, abs_max=0.6217
- tcn_features: shape=[64, 17], abs_mean=0.4137, abs_max=3.0978
- final_routed: shape=[64, 17], abs_mean=0.0914, abs_max=0.6882

## Electrode-Level Evidence

- Top channels by saliency: C3, Cz, CPz, CP3, C1, FCz, C4, C5, CP1, FC3
- Top channels by grad x input: Cz, C3, CPz, CP3, C1, C4, FCz, CP1, CP2, CP4
- Top salient time points: 292, 237, 238, 270, 242, 247, 291, 248, 265, 287, 271, 243, 334, 229, 246