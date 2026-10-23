
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.core import (
    ProjectConfig, build_loaders, collect_logits, compute_metrics,
    detailed_report, expected_calibration_error, export_bundle,
    fit_model, fit_temperature, get_gradcam_target_layer, GradCAM,
    multiclass_brier, seed_everything, selective_metrics,
    softmax_with_temperature, choose_confidence_threshold
)


ARCHS = ["custom_cnn", "resnet50", "efficientnet_b0", "densenet121"]


def download_dataset():
    import kagglehub
    return kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")


def plot_confusion(y, probs, class_names, out_path):
    cm = confusion_matrix(y, np.argmax(probs, axis=1))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_reliability(probs, labels, out_path, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_roc(labels, probs, class_names, out_path):
    y_bin = label_binarize(labels, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(df, out_path):
    """Plot final test metrics; model selection itself is validation-only."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["test_accuracy"], width, label="Accuracy")
    ax.bar(x, df["test_f1_macro"], width, label="Macro F1")
    ax.bar(x + width, df["test_auc_ovr_macro"], width, label="Macro AUC")
    ax.set_xticks(x, df["architecture"], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-Out Test Comparison (Selected on Validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_gradcam_examples(model, architecture, test_loader, class_names, device, out_dir, n=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_gradcam_target_layer(model, architecture)
    gradcam = GradCAM(model, target_layer)
    saved = 0

    for x, y in test_loader:
        for i in range(x.size(0)):
            xi = x[i:i+1].to(device)
            cam, logits = gradcam(xi)
            pred = int(logits.argmax(dim=1).item())

            # convert normalized tensor back approximately for visualization
            img = x[i].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(img * std + mean, 0, 1)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img)
            ax.imshow(cam, alpha=0.4)
            ax.axis("off")
            ax.set_title(f"True: {class_names[int(y[i])]} | Pred: {class_names[pred]}")
            fig.tight_layout()
            fig.savefig(out_dir / f"gradcam_{saved:02d}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

            saved += 1
            if saved >= n:
                gradcam.close()
                return
    gradcam.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--fine-epochs", type=int, default=5)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        parser.error("--train-ratio + --val-ratio + --test-ratio must equal 1.0")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(42)

    cfg = ProjectConfig(
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_epochs,
        target_coverage=args.target_coverage,
        train_size=args.train_ratio,
        val_size=args.val_ratio,
        test_size=args.test_ratio,
    )

    dataset_root = args.dataset_root or download_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", dataset_root)

    train_loader, val_loader, test_loader, class_names, split_info = build_loaders(dataset_root, cfg)
    print("Classes:", class_names)
    with open(out / "split_summary.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # First select the architecture using validation metrics only.
    rows = []
    trained = {}
    for architecture in ARCHS:
        print(f"\n===== Training {architecture} =====")
        model, history, best_f1, best_epoch = fit_model(
            architecture, len(class_names),
            (train_loader, val_loader, test_loader), cfg, device
        )
        trained[architecture] = model

        with open(out / f"{architecture}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        val_logits_arch, val_y_arch = collect_logits(model, val_loader, device)
        val_probs_arch = torch.softmax(val_logits_arch, dim=1).numpy()
        val_metrics = compute_metrics(val_y_arch.numpy(), val_probs_arch)
        row = {
            "architecture": architecture,
            "best_epoch": int(best_epoch),
            "best_val_f1_macro": float(best_f1),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        torch.save(model.state_dict(), out / f"{architecture}_raw.pth")
        print(architecture, "validation metrics:", val_metrics)

    best_row = max(rows, key=lambda row: row["val_f1_macro"])
    best_arch = str(best_row["architecture"])
    best_model = trained[best_arch]
    print("\nBest architecture selected using validation macro-F1:", best_arch)

    # After selection is frozen, evaluate every candidate once on the held-out test set.
    test_labels = None
    for row in rows:
        architecture = row["architecture"]
        test_logits_arch, test_y_arch = collect_logits(trained[architecture], test_loader, device)
        test_probs_arch = torch.softmax(test_logits_arch, dim=1).numpy()
        test_metrics = compute_metrics(test_y_arch.numpy(), test_probs_arch)
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        if test_labels is None:
            test_labels = test_y_arch
        elif not torch.equal(test_labels, test_y_arch):
            raise RuntimeError("Test labels changed unexpectedly between model evaluations.")

    df = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    df.to_csv(out / "model_comparison.csv", index=False)
    plot_model_comparison(df, out / "model_comparison.png")

    val_logits, val_y = collect_logits(best_model, val_loader, device)
    test_logits, test_y = collect_logits(best_model, test_loader, device)

    uncal_val_probs = torch.softmax(val_logits, dim=1).numpy()
    uncal_test_probs = torch.softmax(test_logits, dim=1).numpy()

    temperature = fit_temperature(val_logits, val_y)
    cal_val_probs = softmax_with_temperature(val_logits, temperature).numpy()
    cal_test_probs = softmax_with_temperature(test_logits, temperature).numpy()

    threshold = choose_confidence_threshold(cal_val_probs, cfg.target_coverage)

    calibration = {
        "temperature": temperature,
        "confidence_threshold": threshold,
        "target_validation_coverage": cfg.target_coverage,
        "uncalibrated_test_ece": expected_calibration_error(uncal_test_probs, test_y.numpy()),
        "calibrated_test_ece": expected_calibration_error(cal_test_probs, test_y.numpy()),
        "uncalibrated_test_brier": multiclass_brier(uncal_test_probs, test_y.numpy()),
        "calibrated_test_brier": multiclass_brier(cal_test_probs, test_y.numpy()),
        "selective_test": selective_metrics(cal_test_probs, test_y.numpy(), threshold),
    }
    with open(out / "calibration_metrics.json", "w") as f:
        json.dump(calibration, f, indent=2)

    report = detailed_report(test_y.numpy(), cal_test_probs, class_names)
    with open(out / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plot_confusion(test_y.numpy(), cal_test_probs, class_names, out / "confusion_matrix.png")
    plot_roc(test_y.numpy(), cal_test_probs, class_names, out / "roc_curves.png")
    plot_reliability(cal_test_probs, test_y.numpy(), out / "reliability_diagram.png")
    save_gradcam_examples(
        best_model, best_arch, test_loader, class_names, device,
        out / "gradcam_examples", n=8
    )

    export_bundle(
        out / "best_model_bundle.pth",
        best_arch, best_model, class_names, cfg, temperature, threshold
    )

    summary = {
        "project_title": "An Explainable and Confidence-Calibrated Deep Learning Framework for Reliable Brain Tumor Classification from MRI",
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "best_architecture": best_arch,
        "temperature": temperature,
        "confidence_threshold": threshold,
        "device": str(device),
        "split_protocol": "stratified 70:15:15 over pooled original Training and Testing folders",
        "split_counts": split_info["counts"],
        "total_images": split_info["total_images"],
        "architecture_selection_metric": "validation macro F1",
    }
    with open(out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCompleted. Results saved to", out.resolve())


if __name__ == "__main__":
    main()
