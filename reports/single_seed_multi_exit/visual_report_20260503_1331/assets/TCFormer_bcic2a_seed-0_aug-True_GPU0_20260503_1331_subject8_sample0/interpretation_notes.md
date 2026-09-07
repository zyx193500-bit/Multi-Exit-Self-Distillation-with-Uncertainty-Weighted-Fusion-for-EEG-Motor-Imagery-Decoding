# Single-Sample Interpretation Notes

- Predicted class: feet
- True class: feet
- Ensemble confidence: 0.9779
- Exit weights: shallow=0.0961, mid=0.3009, deep=0.3483, final=0.2547

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.6642, abs_max=3.7304
- conv_features: shape=[48, 17], abs_mean=0.2777, abs_max=2.8102
- shallow_routed: shape=[48, 17], abs_mean=0.2675, abs_max=2.1169
- mid_routed: shape=[48, 17], abs_mean=0.4429, abs_max=2.1109
- deep_routed: shape=[48, 17], abs_mean=0.4811, abs_max=2.3898
- tcn_features: shape=[64, 17], abs_mean=0.5953, abs_max=7.7773
- final_routed: shape=[64, 17], abs_mean=0.4944, abs_max=3.3322

## Electrode-Level Evidence

- Top channels by saliency: POz, CPz, CP3, P2, CP2, CP4, Pz, P1, C5, CP1
- Top channels by grad x input: POz, CPz, CP3, P2, CP2, CP4, P1, CP1, C5, Pz
- Top salient time points: 196, 197, 275, 269, 270, 195, 276, 221, 252, 198, 220, 251, 246, 253, 245