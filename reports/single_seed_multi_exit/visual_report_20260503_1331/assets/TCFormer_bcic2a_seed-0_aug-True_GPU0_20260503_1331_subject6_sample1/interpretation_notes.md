# Single-Sample Interpretation Notes

- Predicted class: hand(L)
- True class: hand(L)
- Ensemble confidence: 0.8389
- Exit weights: shallow=0.0911, mid=0.3170, deep=0.3638, final=0.2281

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.8260, abs_max=3.6723
- conv_features: shape=[48, 17], abs_mean=0.3843, abs_max=3.1624
- shallow_routed: shape=[48, 17], abs_mean=0.2297, abs_max=1.9853
- mid_routed: shape=[48, 17], abs_mean=0.1883, abs_max=2.2561
- deep_routed: shape=[48, 17], abs_mean=0.1850, abs_max=1.6540
- tcn_features: shape=[64, 17], abs_mean=0.3172, abs_max=3.0674
- final_routed: shape=[64, 17], abs_mean=0.1185, abs_max=0.7797

## Electrode-Level Evidence

- Top channels by saliency: Cz, CP2, CP4, P2, P1, C4, POz, FC3, FCz, C6
- Top channels by grad x input: Cz, CP2, CP4, FCz, FC3, FC2, FC4, C4, P2, C3
- Top salient time points: 175, 194, 195, 174, 176, 67, 68, 75, 73, 193, 74, 76, 72, 79, 78