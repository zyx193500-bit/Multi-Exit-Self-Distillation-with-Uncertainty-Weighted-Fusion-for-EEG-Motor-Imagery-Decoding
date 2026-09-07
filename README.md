# Multi-Exit-Self-Distillation-with-Uncertainty-Weighted-Fusion-for-EEG-Motor-Imagery-Decoding

Motor-imagery decoding from electroencephalography (EEG) is a core task in non-invasive
brain-computer interfaces (BCIs). Most existing deep EEG decoders rely only on the final exit
for supervision and inference, so local time-frequency patterns, short-term rhythmic variations
and sequential contextual information preserved in shallow and middle representations are not
explicitly converted into supervised discriminative information. To address this problem, we
propose a multi-exit self-distillation framework with learnable uncertainty-weighted fusion for
EEG motor-imagery decoding. The framework is general and can be used as an add-on module
for different EEG classification backbones. Without changing the main backbone architecture,
it attaches exits to multiple representation stages. During training, the prediction distribution
of the final exit is used as an internal reference within the same network. Together with label
supervision, feature constraints and inter-exit self-distillation, this improves the discriminative
capacity of shallow, middle and deep exits. During inference, learnable branch-wise uncertainty
parameters and preset exit-prior coefficients define global fusion weights for the logits from
all exits, producing the final prediction. We validate the framework on BCI Competition IV-2a
and IV-2b using five backbones: TCFormer, ATCNet, CTNet, MSCFormer and EEGNet. The
results show overall gains across the evaluated backbones. With TCFormer, fused accuracies
reach 87.04% on IV-2a and 89.99% on IV-2b, compared with 84.80% and 87.63% for the
single-exit baselines, giving improvements of 2.24 and 2.36 percentage points, respectively.
Ablation experiments further characterize the roles of multi-level exits, inter-exit distillation,
and uncertainty-weighted fusion and show that their effects are configuration dependent.
<img width="1418" height="827" alt="Main figure" src="https://github.com/user-attachments/assets/e4b92f99-571e-448a-848a-2b5540d8f65c" />
