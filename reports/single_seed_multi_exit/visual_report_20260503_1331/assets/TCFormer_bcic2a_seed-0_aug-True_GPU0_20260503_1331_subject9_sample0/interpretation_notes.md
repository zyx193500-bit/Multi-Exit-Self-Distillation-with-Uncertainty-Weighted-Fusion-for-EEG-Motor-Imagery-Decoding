# Single-Sample Interpretation Notes

- Predicted class: feet
- True class: feet
- Ensemble confidence: 0.9629
- Exit weights: shallow=0.0926, mid=0.3013, deep=0.3424, final=0.2637

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.5986, abs_max=3.6002
- conv_features: shape=[48, 17], abs_mean=0.3867, abs_max=2.1396
- shallow_routed: shape=[48, 17], abs_mean=0.3053, abs_max=1.7495
- mid_routed: shape=[48, 17], abs_mean=0.1975, abs_max=1.2255
- deep_routed: shape=[48, 17], abs_mean=0.2398, abs_max=1.9259
- tcn_features: shape=[64, 17], abs_mean=0.5299, abs_max=3.5453
- final_routed: shape=[64, 17], abs_mean=0.2161, abs_max=1.3168

## Electrode-Level Evidence

- Top channels by saliency: CPz, Pz, FC3, CP4, C5, CP3, C1, FCz, POz, C3
- Top channels by grad x input: CPz, FC3, Pz, C5, C1, CP4, Cz, CP3, FCz, C3
- Top salient time points: 282, 283, 281, 289, 290, 267, 266, 291, 288, 268, 379, 265, 292, 299, 280