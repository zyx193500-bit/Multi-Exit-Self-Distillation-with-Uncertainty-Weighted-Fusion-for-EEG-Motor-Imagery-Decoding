# Single-Sample Interpretation Notes

- Predicted class: feet
- True class: feet
- Ensemble confidence: 0.9687
- Exit weights: shallow=0.0855, mid=0.3114, deep=0.3603, final=0.2428

## Figure Captions

- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.
- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.
- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.
- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.
- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.
- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.
- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.

## Quantitative Stage Summary

- raw_signal: shape=[22, 1000], abs_mean=0.7002, abs_max=4.8644
- conv_features: shape=[48, 17], abs_mean=0.4086, abs_max=2.2466
- shallow_routed: shape=[48, 17], abs_mean=0.2648, abs_max=2.1636
- mid_routed: shape=[48, 17], abs_mean=0.1832, abs_max=1.6614
- deep_routed: shape=[48, 17], abs_mean=0.2146, abs_max=1.8772
- tcn_features: shape=[64, 17], abs_mean=0.4924, abs_max=3.6215
- final_routed: shape=[64, 17], abs_mean=0.2182, abs_max=1.7294

## Electrode-Level Evidence

- Top channels by saliency: POz, CP3, C6, Cz, CP2, CPz, CP4, FC4, CP1, C5
- Top channels by grad x input: POz, CP3, C6, CP2, Cz, CPz, CP4, CP1, FC3, FC4
- Top salient time points: 147, 146, 640, 704, 716, 142, 705, 148, 658, 717, 145, 141, 657, 143, 634