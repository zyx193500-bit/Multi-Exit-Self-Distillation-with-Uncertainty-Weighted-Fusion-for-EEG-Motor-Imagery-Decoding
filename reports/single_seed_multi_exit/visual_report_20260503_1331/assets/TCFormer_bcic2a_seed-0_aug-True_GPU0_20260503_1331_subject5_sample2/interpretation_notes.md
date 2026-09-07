# Single-Sample Interpretation Notes

- Predicted class: hand(L)
- True class: hand(L)
- Ensemble confidence: 0.9565
- Exit weights: shallow=0.0921, mid=0.3101, deep=0.3590, final=0.2388

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.8013, abs_max=3.5016
- conv_features: shape=[48, 17], abs_mean=0.5242, abs_max=3.3253
- shallow_routed: shape=[48, 17], abs_mean=0.2563, abs_max=2.1234
- mid_routed: shape=[48, 17], abs_mean=0.1695, abs_max=1.1473
- deep_routed: shape=[48, 17], abs_mean=0.1704, abs_max=1.1131
- tcn_features: shape=[64, 17], abs_mean=0.4945, abs_max=3.1066
- final_routed: shape=[64, 17], abs_mean=0.1954, abs_max=1.3528

## Electrode-Level Evidence

- Top channels by saliency: C3, C5, CPz, CP3, C1, FC4, C4, P1, Fz, FC1
- Top channels by grad x input: C3, C5, CPz, CP3, C1, FC4, P1, C4, FC1, Fz
- Top salient time points: 80, 81, 99, 107, 82, 79, 98, 106, 93, 113, 100, 114, 108, 105, 115