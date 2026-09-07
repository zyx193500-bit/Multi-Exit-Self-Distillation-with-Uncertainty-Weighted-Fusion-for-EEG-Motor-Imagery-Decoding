# Single-Sample Interpretation Notes

- Predicted class: hand(L)
- True class: hand(L)
- Ensemble confidence: 0.6958
- Exit weights: shallow=0.0904, mid=0.3146, deep=0.3597, final=0.2353

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.6538, abs_max=3.2813
- conv_features: shape=[48, 17], abs_mean=0.3454, abs_max=2.3373
- shallow_routed: shape=[48, 17], abs_mean=0.2177, abs_max=1.5801
- mid_routed: shape=[48, 17], abs_mean=0.1219, abs_max=0.8350
- deep_routed: shape=[48, 17], abs_mean=0.1202, abs_max=0.6793
- tcn_features: shape=[64, 17], abs_mean=0.3321, abs_max=2.5194
- final_routed: shape=[64, 17], abs_mean=0.0991, abs_max=0.4985

## Electrode-Level Evidence

- Top channels by saliency: C2, C4, CP4, CPz, FC4, Cz, C6, CP3, P2, FCz
- Top channels by grad x input: C2, C4, CP4, CPz, FC4, Cz, C6, CP3, P2, Pz
- Top salient time points: 279, 280, 181, 278, 352, 281, 68, 342, 73, 273, 69, 269, 327, 219, 182