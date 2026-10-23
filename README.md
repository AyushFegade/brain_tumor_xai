# An Explainable and Confidence-Calibrated Deep Learning Framework for Reliable Brain Tumor Classification from MRI

This project extends the original **Brain Tumor Detection using CNN** plan into a research-oriented MRI classification system.

## Task

Four-class brain MRI classification:

- glioma
- meningioma
- no tumor
- pituitary tumor

Primary dataset: Kaggle `masoudnickparvar/brain-tumor-mri-dataset`.

## What makes this version stronger than a normal model-comparison project

1. Benchmarks **Custom CNN, ResNet50, EfficientNet-B0 and DenseNet121**.
2. Uses **validation macro F1** as the architecture/model-selection metric, not test performance.
3. Performs **temperature scaling** on validation logits.
4. Reports **Expected Calibration Error (ECE)** and **multiclass Brier score**.
5. Learns an **uncertainty/abstention threshold from validation data** to target 90% coverage.
6. Produces **Grad-CAM** heatmaps for interpretability.
7. Exports a deployable model bundle and a **Streamlit demo**.
8. Includes an optional script for **external-folder validation**.

## Architecture

MRI image -> resize/normalize/augmentation -> four candidate CNN models -> validation-based best model
-> logits -> temperature scaling -> calibrated probabilities -> selective prediction
-> class + confidence + uncertainty flag -> Grad-CAM explanation

See `architecture.mmd` for a Mermaid diagram.

## Colab / GPU training

```bash
pip install -r requirements.txt
python train.py --head-epochs 3 --fine-epochs 5
```

For a quicker pipeline test:

```bash
python train.py --head-epochs 1 --fine-epochs 1
```

For the final paper run, increase the epochs and report the exact configuration you use.

The training script downloads the Kaggle dataset automatically using `kagglehub` when `--dataset-root` is not supplied.

### Split protocol (default: 70:15:15)

By default, the code pools images from the dataset's original `Training/` and `Testing/` folders and then creates a **stratified 70% training / 15% validation / 15% test split** (seed 42). On a 7,200-image copy, this is approximately 5,040 / 1,080 / 1,080 images; exact counts are printed and saved in `outputs/split_summary.json`.

**Research caveat:** because the original `Testing/` folder is pooled before re-splitting, this protocol does not preserve Kaggle's official test partition. Results should not be presented as directly comparable to papers that evaluate on the untouched official test set. Architecture selection, temperature fitting, and confidence-threshold selection use validation data only; the new test set is held out for final evaluation. This ZIP contains code and notebook files, not a newly trained checkpoint. Retrain after updating the split before reporting new results or treating the Streamlit app's model as the new experiment.

You can change the proportions as long as they sum to 1.0, for example:

```bash
python train.py --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

If you already downloaded the dataset:

```bash
python train.py --dataset-root /path/to/brain-tumor-mri-dataset
```

## Outputs

After training, `outputs/` contains:

- `split_summary.json` (overall and per-class split counts)
- `model_comparison.csv`
- `model_comparison.png`
- raw model checkpoints
- `classification_report.json`
- `confusion_matrix.png`
- `roc_curves.png`
- `reliability_diagram.png`
- `calibration_metrics.json`
- `gradcam_examples/`
- `best_model_bundle.pth`
- `run_summary.json`

## Run the web demo

```bash
streamlit run app.py
```

Upload a JPG/PNG MRI. The app shows:

- predicted class
- temperature-calibrated confidence
- uncertain/accepted status
- all class probabilities
- Grad-CAM heatmap

## Optional external validation

Prepare a directory with exactly the same four class folder names as the training dataset, then run:

```bash
python evaluate_external.py --external-dir /path/to/external_test
```

Do **not** claim external generalization unless the external dataset is genuinely independent and does not overlap with the source data used in the Kaggle composite dataset.

## Recommended paper experiments

- Baseline comparison: CNN vs ResNet50 vs EfficientNet-B0 vs DenseNet121
- Calibration: ECE and Brier score before vs after temperature scaling
- Selective classification: coverage vs selective accuracy
- Explainability: representative Grad-CAM examples for correct and incorrect cases
- Error analysis by class
- Optional independent external validation
- Ablation: best model without calibration/abstention vs full pipeline

## Important research limitation

This is a research/educational system, **not a clinical diagnostic device**. A high test-set score on a public benchmark does not establish safety or clinical validity.

## Reproducibility

Random seed: 42 by default. Train/validation/test splitting is stratified. The original Kaggle Training and Testing folders are pooled before splitting, so the original official test partition is not preserved. Model selection and calibration are validation-only; the test set is reserved for final reporting.
