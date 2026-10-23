# Paper-ready framing

## New title
**An Explainable and Confidence-Calibrated Deep Learning Framework for Reliable Brain Tumor Classification from MRI**

## Dataset split protocol

The current implementation pools images from the Kaggle dataset's original `Training/` and `Testing/` folders and creates a stratified 70% training / 15% validation / 15% test split (seed 42). Exact total and per-class counts are saved in `outputs/split_summary.json`.

This is a custom re-split, not the original Kaggle train/test protocol. When comparing with papers that retain the official test set, clearly disclose this difference and avoid treating the reported scores as a directly controlled comparison. For architecture selection, use validation macro F1; fit temperature scaling and the confidence threshold on validation data only; evaluate the test set after those choices are frozen.

## Core research question
Can a brain-tumor MRI classifier be made more trustworthy by combining strong transfer-learning baselines with calibrated confidence, selective prediction, and post-hoc visual explanations?

## Main contributions
1. Systematic comparison of a custom CNN and three ImageNet-pretrained architectures.
2. Post-hoc temperature scaling to reduce probability miscalibration.
3. A validation-derived selective-prediction threshold that can abstain on low-confidence cases.
4. Grad-CAM explanations for qualitative interpretability.
5. Evaluation using discrimination **and** reliability metrics: Accuracy, Precision, Recall, Macro F1, AUC, ECE, Brier score, coverage, and selective accuracy.
6. Optional independent external-folder evaluation pathway.

## Suggested objectives
- Preprocess and augment four-class brain MRI data.
- Compare CNN, ResNet50, EfficientNet-B0 and DenseNet121.
- Select the strongest model using validation macro F1.
- Calibrate its predictive probabilities using temperature scaling.
- Quantify calibration using ECE and Brier score.
- Introduce selective prediction so low-confidence scans can be flagged as uncertain.
- Explain model decisions with Grad-CAM.
- Evaluate generalization on an independent dataset if a truly non-overlapping dataset is available.

## Suggested ablation table
| Variant | Accuracy | Macro F1 | Macro AUC | ECE | Brier | Coverage | Selective Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best model, raw confidence | | | | | | 100% | |
| + Temperature scaling | | | | | | 100% | |
| + Selective prediction | | | | | | | |

## Suggested conclusion wording
The main contribution should not be framed only as obtaining the highest accuracy. Emphasize that the framework adds probability calibration, a data-derived abstention mechanism, and visual explanations to a standard MRI classifier, making evaluation closer to reliability-oriented medical AI.
