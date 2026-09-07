# Single-Sample Interpretation Notes

- Predicted class: hand(L)
- True class: hand(L)
- Ensemble confidence: 0.6465
- Exit weights: shallow=0.0871, mid=0.2981, deep=0.3467, final=0.2681

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.5777, abs_max=4.6327
- conv_features: shape=[48, 17], abs_mean=0.4163, abs_max=1.6257
- shallow_routed: shape=[48, 17], abs_mean=0.2139, abs_max=1.0468
- mid_routed: shape=[48, 17], abs_mean=0.2766, abs_max=1.2065
- deep_routed: shape=[48, 17], abs_mean=0.2837, abs_max=1.2287
- tcn_features: shape=[64, 17], abs_mean=0.4444, abs_max=3.1854
- final_routed: shape=[64, 17], abs_mean=0.1917, abs_max=1.3015

## Electrode-Level Evidence

- Top channels by saliency: C3, CPz, C4, Cz, FC3, C2, POz, C6, P1, P2
- Top channels by grad x input: C3, C4, CPz, Cz, FC3, C2, POz, C6, P1, P2
- Top salient time points: 153, 155, 154, 165, 88, 179, 167, 77, 76, 166, 89, 98, 832, 823, 178