# Single-Sample Interpretation Notes

- Predicted class: feet
- True class: feet
- Ensemble confidence: 0.9675
- Exit weights: shallow=0.0931, mid=0.3152, deep=0.3634, final=0.2283

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.7093, abs_max=3.7410
- conv_features: shape=[48, 17], abs_mean=0.3836, abs_max=1.9394
- shallow_routed: shape=[48, 17], abs_mean=0.2736, abs_max=2.0973
- mid_routed: shape=[48, 17], abs_mean=0.2360, abs_max=1.2408
- deep_routed: shape=[48, 17], abs_mean=0.2551, abs_max=1.2729
- tcn_features: shape=[64, 17], abs_mean=0.4383, abs_max=2.8690
- final_routed: shape=[64, 17], abs_mean=0.1376, abs_max=0.8830

## Electrode-Level Evidence

- Top channels by saliency: CPz, C4, CP3, Cz, FCz, CP4, C3, CP1, POz, Pz
- Top channels by grad x input: CPz, Cz, C4, CP3, FCz, C3, CP1, CP4, POz, C1
- Top salient time points: 335, 345, 344, 326, 327, 325, 297, 336, 346, 288, 334, 354, 287, 343, 355